"""Natural language → read-only SQL over the CADMIA database.

The LLM only writes the SELECT statement (text); every number shown to the user
comes from executing that statement locally against SQLite, never from the model.
The generated SQL is validated to be a single read-only SELECT
and executed on a read-only connection, so a wrong or malicious statement cannot
modify the database.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from cadmia.llm.config import LLMConfig
from cadmia.llm.provider import LLMError, Provider
from cadmia.llm.registry import get_provider

_SYSTEM = (
    "You translate natural-language questions about a SQLite database of "
    "computational-chemistry results (structures, calculations, descriptors, "
    "adsorption, stability, provenance) into SQL.\n"
    "Rules:\n"
    "- Output exactly ONE SQLite SELECT statement (WITH ... SELECT is allowed). "
    "No markdown fences, no commentary, no trailing semicolon.\n"
    "- Use only the tables and columns in the schema provided with the question.\n"
    "- Never write statements that modify the database.\n"
    "- Prefer explicit column lists over SELECT *; add ORDER BY when ranking is implied.\n"
    "- Energies: energy_au/e_*_au are Hartree, e_*_kcal are kcal/mol, *_eV are eV."
)

# Statement-level write verbs. `replace` is intentionally absent: it is also a
# read-only scalar function, and the REPLACE *statement* is already caught by the
# leading-keyword check (head must be SELECT/WITH).
# `pragma\w*` (not `pragma`) so the table-valued form `pragma_table_info(...)` cannot slip
# through: the trailing `_` is a word char, so `\bpragma\b` would not match it. `load_extension`
# is blocked too. This is defense in depth — the connection is already opened `mode=ro` with a
# single-statement guard and extensions disabled.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma\w*|"
    r"vacuum|trigger|reindex|load_extension)\b",
    re.IGNORECASE,
)


# Wall-clock budget for the generated SELECT. `mode=ro` and the single-statement
# guard bound what the query can *do*, not what it can cost: a recursive CTE or a
# cartesian join materializes before fetchmany(limit) ever sees a row. The progress
# handler is the only way to bound that from Python.
STATEMENT_TIMEOUT_S = 30.0

# VM instructions between progress-handler calls: small enough to notice the
# deadline promptly, large enough that the callback is not the bottleneck.
_PROGRESS_INTERVAL = 10_000


def _deadline_handler(timeout_s: float) -> Callable[[], int]:
    """Return a progress handler that aborts the statement after *timeout_s*.

    SQLite aborts the running statement when the handler returns non-zero, which
    surfaces as sqlite3.OperationalError("interrupted").
    """
    deadline = time.monotonic() + timeout_s
    return lambda: 1 if time.monotonic() > deadline else 0


@dataclass
class SQLQueryResult:
    question: str
    sql:      str
    columns:  List[str] = field(default_factory=list)
    rows:     List[tuple] = field(default_factory=list)
    model:    str = ""
    provider: str = ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _strip_sql_noise(sql: str) -> str:
    """Replace string/identifier literals and comments with spaces.

    The structural checks (statement count, leading keyword, write verbs) run on
    this skeleton so a ``;`` or a keyword inside quoted text or a comment cannot
    fool them (e.g. ``WHERE name = 'a;b'`` or ``replace()`` in a column name).
    """
    out: List[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "-" and i + 1 < n and sql[i + 1] == "-":          # -- line comment
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":          # /* block comment */
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2
            out.append(" ")
            continue
        if c in ("'", '"'):                                        # quoted literal/identifier
            quote = c
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:          # '' or "" escape
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(c)
        i += 1
    return "".join(out)


def validate_readonly_sql(sql: str) -> str:
    """Validate that *sql* is a single read-only SELECT; return it normalized.

    Raises LLMError otherwise. This is the safety gate between the model's text
    and the database — nothing the model writes can bypass it. Structural checks
    ignore quoted text and comments so legitimate SELECTs (a ``;`` inside a string,
    the ``replace()`` function) are not rejected.
    """
    cleaned = _strip_fences(sql).strip()
    if not cleaned:
        raise LLMError("The model returned an empty SQL statement.")
    skeleton = _strip_sql_noise(cleaned)
    if len([s for s in skeleton.split(";") if s.strip()]) > 1:
        raise LLMError("Multiple SQL statements are not allowed.")
    tokens = skeleton.split()
    head = tokens[0].lower() if tokens else ""
    if head not in ("select", "with"):
        raise LLMError(f"Only SELECT queries are allowed (got '{(head or '∅').upper()}').")
    match = _FORBIDDEN.search(skeleton)
    if match:
        raise LLMError(f"Forbidden keyword in generated SQL: {match.group(0).upper()}.")
    return cleaned.rstrip(";").strip()


def database_schema(db_path: str | Path) -> str:
    """Return the CREATE TABLE statements of the database, for the LLM prompt."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0] for r in rows if r[0])


def generate_sql(
    question: str,
    schema: str,
    *,
    provider: Optional[Provider] = None,
    config: Optional[LLMConfig] = None,
) -> tuple[str, Any]:
    """Ask the configured LLM for a SELECT answering *question*; return (sql, response).

    The returned SQL has already passed validate_readonly_sql.
    """
    provider = provider or get_provider(config)
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    response = provider.complete(prompt, system=_SYSTEM)
    return validate_readonly_sql(response.text), response


def ask_database(
    question: str,
    db_path: str | Path,
    *,
    limit: int = 50,
    timeout_s: float = STATEMENT_TIMEOUT_S,
    provider: Optional[Provider] = None,
    config: Optional[LLMConfig] = None,
) -> SQLQueryResult:
    """Answer *question* by generating and executing a read-only SELECT.

    Execution uses a read-only SQLite connection (URI mode=ro), aborts after
    *timeout_s* seconds and fetches at most *limit* rows.
    """
    path = Path(db_path)
    if not path.exists():
        raise LLMError(f"No database found at {path}.")
    schema = database_schema(path)
    sql, response = generate_sql(question, schema, provider=provider, config=config)

    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    started = time.monotonic()
    try:
        conn.set_progress_handler(_deadline_handler(timeout_s), _PROGRESS_INTERVAL)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(limit)
    except sqlite3.Error as exc:
        if time.monotonic() - started >= timeout_s:
            raise LLMError(
                f"Generated SQL exceeded the {timeout_s:g} s budget and was aborted. "
                f"Ask a narrower question.\nSQL: {sql}"
            )
        raise LLMError(f"Generated SQL failed to execute: {exc}\nSQL: {sql}")
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()
    return SQLQueryResult(
        question=question, sql=sql, columns=columns, rows=rows,
        model=response.model, provider=response.provider,
    )
