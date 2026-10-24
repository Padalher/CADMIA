"""Literature Agent — reference discovery from free bibliographic APIs (no API key).

Queries Crossref and OpenAlex over plain HTTP (stdlib only, no SDK), normalizes the
results to a common ``Reference`` record, dedupes by DOI, ranks, and renders the list as
Markdown, BibTeX or JSON. Identifiers and numbers (DOI, year, citation count) always come
from the API, never from a language model. An optional LLM pass can add a
prose relevance note; it degrades cleanly when no API key is configured.

Requires internet at call time, but no API key, workstation or HPC connection.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
# A contact mailto puts Crossref/OpenAlex requests in their faster "polite" pool.
CONTACT = "cadmia@users.noreply.github.com"

SOURCES = ("crossref", "openalex", "semanticscholar")


class LiteratureError(RuntimeError):
    """Network or response error from a bibliographic API."""


@dataclass
class Reference:
    title:     str
    authors:   List[str] = field(default_factory=list)
    year:      Optional[int] = None
    container: str = ""           # journal / venue
    doi:       Optional[str] = None
    url:       Optional[str] = None
    citations: Optional[int] = None
    type:      str = ""
    source:    str = ""           # crossref / openalex / semanticscholar
    abstract:  str = ""           # plain-text abstract when the API provides one

    def to_dict(self) -> dict:
        return {
            "title": self.title, "authors": self.authors, "year": self.year,
            "container": self.container, "doi": self.doi, "url": self.url,
            "citations": self.citations, "type": self.type, "source": self.source,
            "abstract": self.abstract,
        }


@dataclass
class LiteratureResult:
    query:      str
    references: List[Reference] = field(default_factory=list)
    warnings:   List[str] = field(default_factory=list)
    narrative:  Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "query": self.query,
                "n_references": len(self.references),
                "references": [r.to_dict() for r in self.references],
                "narrative": self.narrative,
                "warnings": self.warnings,
            },
            indent=2, ensure_ascii=False,
        )


# ─────────────────────────────────────────────
# HTTP (single network choke point — monkeypatched in tests)
# ─────────────────────────────────────────────

def _http_get_json(url: str, params: Dict[str, str], *, timeout_s: int) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    request = urllib.request.Request(
        full, headers={"accept": "application/json", "user-agent": f"CADMIA (mailto:{CONTACT})"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LiteratureError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise LiteratureError(f"Could not reach {url}: {exc.reason}") from exc
    except (ValueError, TimeoutError) as exc:
        raise LiteratureError(f"Bad response from {url}: {exc}") from exc


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def _norm_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi or None


def _strip_jats(text: Optional[str]) -> str:
    """Crossref abstracts come as JATS XML; strip tags to plain text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _reconstruct_inverted_abstract(inv: Optional[dict]) -> str:
    """Rebuild an abstract from OpenAlex's abstract_inverted_index (word -> positions)."""
    if not isinstance(inv, dict) or not inv:
        return ""
    positions: List[tuple] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _ref_from_crossref(item: dict) -> Reference:
    title = (item.get("title") or [""])
    authors = [
        " ".join(p for p in (a.get("given"), a.get("family")) if p)
        for a in item.get("author", [])
    ]
    year = None
    parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    if parts and parts[0] and parts[0][0]:
        year = int(parts[0][0])
    container = (item.get("container-title") or [""])
    return Reference(
        title=title[0] if title else "",
        authors=[a for a in authors if a],
        year=year,
        container=container[0] if container else "",
        doi=_norm_doi(item.get("DOI")),
        url=item.get("URL"),
        citations=item.get("is-referenced-by-count"),
        type=item.get("type", ""),
        source="crossref",
        abstract=_strip_jats(item.get("abstract")),
    )


def _ref_from_openalex(item: dict) -> Reference:
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in item.get("authorships", [])
    ]
    source_obj = (item.get("primary_location") or {}).get("source") or {}
    return Reference(
        title=item.get("title") or item.get("display_name") or "",
        authors=[a for a in authors if a],
        year=item.get("publication_year"),
        container=source_obj.get("display_name", "") or "",
        doi=_norm_doi(item.get("doi")),
        url=item.get("doi") or (item.get("primary_location") or {}).get("landing_page_url"),
        citations=item.get("cited_by_count"),
        type=item.get("type", ""),
        source="openalex",
        abstract=_reconstruct_inverted_abstract(item.get("abstract_inverted_index")),
    )


def _ref_from_semanticscholar(item: dict) -> Reference:
    authors = [a.get("name", "") for a in item.get("authors", [])]
    ext = item.get("externalIds") or {}
    venue = item.get("venue") or ""
    return Reference(
        title=item.get("title") or "",
        authors=[a for a in authors if a],
        year=item.get("year"),
        container=venue,
        doi=_norm_doi(ext.get("DOI")),
        url=item.get("url"),
        citations=item.get("citationCount"),
        type=(item.get("publicationTypes") or [""])[0] or "",
        source="semanticscholar",
        abstract=(item.get("abstract") or "").strip(),
    )


# ─────────────────────────────────────────────
# Searches
# ─────────────────────────────────────────────

def search_crossref(query: str, *, rows: int = 20, from_year: Optional[int] = None,
                    timeout_s: int = 30) -> List[Reference]:
    params = {
        "query": query,
        "rows": str(rows),
        "select": "title,author,issued,container-title,DOI,URL,is-referenced-by-count,type",
        "mailto": CONTACT,
    }
    if from_year:
        params["filter"] = f"from-pub-date:{from_year}-01-01"
    data = _http_get_json(CROSSREF_API, params, timeout_s=timeout_s)
    items = (data.get("message") or {}).get("items", [])
    return [_ref_from_crossref(it) for it in items]


def search_openalex(query: str, *, rows: int = 20, from_year: Optional[int] = None,
                    timeout_s: int = 30) -> List[Reference]:
    params = {"search": query, "per-page": str(min(rows, 200)), "mailto": CONTACT}
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    data = _http_get_json(OPENALEX_API, params, timeout_s=timeout_s)
    return [_ref_from_openalex(it) for it in data.get("results", [])]


def search_semantic_scholar(query: str, *, rows: int = 20, from_year: Optional[int] = None,
                            timeout_s: int = 30) -> List[Reference]:
    params = {
        "query": query,
        "limit": str(min(rows, 100)),
        "fields": "title,authors,year,venue,externalIds,url,citationCount,publicationTypes,abstract",
    }
    if from_year:
        params["year"] = f"{from_year}-"
    data = _http_get_json(SEMANTIC_SCHOLAR_API, params, timeout_s=timeout_s)
    return [_ref_from_semanticscholar(it) for it in data.get("data", [])]


def fetch_by_doi(doi: str, *, timeout_s: int = 30) -> Reference:
    clean = _norm_doi(doi)
    data = _http_get_json(f"{CROSSREF_API}/{urllib.parse.quote(clean)}", {}, timeout_s=timeout_s)
    message = data.get("message")
    if not isinstance(message, dict):
        raise LiteratureError(f"No Crossref record for DOI {doi}.")
    return _ref_from_crossref(message)


# ─────────────────────────────────────────────
# Dedupe / rank
# ─────────────────────────────────────────────

def dedupe(refs: List[Reference]) -> List[Reference]:
    """Merge duplicates by DOI (preferred) or normalized title; keep the richest record."""
    seen: Dict[str, Reference] = {}
    order: List[str] = []
    for r in refs:
        key = r.doi or ("title:" + re.sub(r"\s+", " ", r.title.strip().lower()))
        if not key.strip() or key == "title:":
            key = f"_id{len(order)}"
        if key not in seen:
            seen[key] = r
            order.append(key)
        else:
            cur = seen[key]
            # Carry over an abstract from whichever record has one.
            if not cur.abstract and r.abstract:
                cur.abstract = r.abstract
            # Prefer the record that carries a citation count / more authors.
            if (r.citations or -1) > (cur.citations or -1) or (
                len(r.authors) > len(cur.authors) and cur.citations is None
            ):
                if cur.citations is not None and r.citations is None:
                    r.citations = cur.citations
                if not r.abstract and cur.abstract:
                    r.abstract = cur.abstract
                seen[key] = r
    return [seen[k] for k in order]


def rank(refs: List[Reference], *, by: str = "relevance") -> List[Reference]:
    if by == "citations":
        return sorted(refs, key=lambda r: (r.citations or 0), reverse=True)
    if by == "year":
        return sorted(refs, key=lambda r: (r.year or 0), reverse=True)
    return refs  # "relevance": keep API order


# ─────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────

def _bibtex_key(ref: Reference, used: set) -> str:
    family = ""
    if ref.authors:
        family = ref.authors[0].split()[-1]
    family = re.sub(r"[^A-Za-z]", "", family) or "ref"
    word = ""
    for tok in re.findall(r"[A-Za-z]+", ref.title):
        if len(tok) > 3:
            word = tok.lower()
            break
    base = f"{family.lower()}{ref.year or ''}{word}"
    key, i = base, 1
    while key in used:
        i += 1
        key = f"{base}{i}"
    used.add(key)
    return key


def to_bibtex(refs: List[Reference]) -> str:
    used: set = set()
    blocks = []
    for ref in refs:
        key = _bibtex_key(ref, used)
        entry = "article" if "journal" in ref.type or ref.type == "journal-article" else "misc"
        fields = [f"  title = {{{ref.title}}}"]
        if ref.authors:
            fields.append("  author = {" + " and ".join(ref.authors) + "}")
        if ref.container:
            fields.append(f"  journal = {{{ref.container}}}")
        if ref.year:
            fields.append(f"  year = {{{ref.year}}}")
        if ref.doi:
            fields.append(f"  doi = {{{ref.doi}}}")
        blocks.append(f"@{entry}{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_markdown(result: LiteratureResult) -> str:
    lines = [f"# References — {result.query}", ""]
    if result.narrative:
        lines += ["> " + result.narrative.replace("\n", "\n> "), ""]
    for i, ref in enumerate(result.references, 1):
        authors = ", ".join(ref.authors[:3]) + (" et al." if len(ref.authors) > 3 else "")
        cite = f" (cited by {ref.citations})" if ref.citations is not None else ""
        doi = f" DOI: {ref.doi}" if ref.doi else ""
        venue = f" *{ref.container}*" if ref.container else ""
        year = f" {ref.year}" if ref.year else ""
        lines.append(f"{i}. {authors}. \"{ref.title}.\"{venue}{year}.{doi}{cite}")
        if ref.abstract:
            snippet = ref.abstract if len(ref.abstract) <= 300 else ref.abstract[:297].rstrip() + "…"
            lines.append(f"   > {snippet}")
    if result.warnings:
        lines += ["", "---", "_Warnings:_"] + [f"- {w}" for w in result.warnings]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run(
    query: str = "",
    *,
    sources: "tuple[str, ...] | list[str]" = ("crossref",),
    limit: int = 10,
    from_year: Optional[int] = None,
    sort: str = "relevance",
    dois: Optional[List[str]] = None,
    narrate: bool = False,
    language: str = "en",
    timeout_s: int = 30,
) -> LiteratureResult:
    """Discover references for *query* (and/or fetch explicit *dois*) from free APIs.

    Parameters
    ----------
    query     : free-text topic (e.g. "conceptual DFT heterogeneous catalysis").
    sources   : any of "crossref", "openalex".
    limit     : maximum number of references returned after dedupe/rank.
    from_year : keep only works published on/after this year.
    sort      : "relevance" (API order), "citations", or "year".
    dois      : explicit DOIs to fetch and prepend (always via Crossref).
    narrate   : add an optional LLM relevance note (needs an API key; degrades cleanly).
    language  : narration language ("en"/"es").
    """
    result = LiteratureResult(query=query)
    collected: List[Reference] = []

    for doi in dois or []:
        try:
            collected.append(fetch_by_doi(doi, timeout_s=timeout_s))
        except LiteratureError as exc:
            result.warnings.append(f"DOI {doi}: {exc}")

    rows = max(limit * 2, limit)  # over-fetch so dedupe still leaves enough
    for src in sources:
        if not query:
            break
        try:
            if src == "crossref":
                collected += search_crossref(query, rows=rows, from_year=from_year, timeout_s=timeout_s)
            elif src == "openalex":
                collected += search_openalex(query, rows=rows, from_year=from_year, timeout_s=timeout_s)
            elif src == "semanticscholar":
                collected += search_semantic_scholar(query, rows=rows, from_year=from_year, timeout_s=timeout_s)
            else:
                result.warnings.append(f"Unknown source '{src}' (available: {', '.join(SOURCES)}).")
        except LiteratureError as exc:
            result.warnings.append(f"{src}: {exc}")

    refs = rank(dedupe(collected), by=sort)[:limit]
    result.references = refs

    if narrate and refs:
        try:
            result.narrative = _narrate(refs, query, language)
        except Exception as exc:  # never let optional narration break the result
            result.warnings.append(f"Narration skipped: {exc}")

    return result


def _narrate(refs: List[Reference], topic: str, language: str) -> str:
    """Optional one-paragraph relevance note via the configured LLM (text only, no numbers)."""
    from cadmia.llm.config import load_config
    from cadmia.llm.registry import get_provider

    provider = get_provider(load_config())
    listing = "\n".join(
        f"- {r.title} ({r.year or 'n.d.'})" for r in refs[:10]
    )
    lang = "Spanish" if language.startswith("es") else "English"
    system = (
        "You summarize the relevance of a reference list for a research topic. "
        f"Write ONE short paragraph in {lang}. Do not invent or cite DOIs, years, or "
        "numbers; refer to works only by their titles. Be factual and concise."
    )
    user = f"Topic: {topic}\n\nReferences (titles only):\n{listing}"
    return provider.complete(user, system=system).text.strip()
