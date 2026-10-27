"""Module results inside a CADMIA project: where each module saves itself, and how.

A project is the folder `cadmia init` creates, marked by its manifest. Inside one, every analysis
stage writes ``results/<stage>.json`` (``results/<stage>/<item>.json`` for the stages run on
several inputs) with its data and its full text report, and `cadmia report` assembles them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "cadmia_project.yaml"
RESULTS_DIR = "results"
MULTI_ITEM_STAGES = ("structure", "analysis")


CONFIG_PATH = Path.home() / ".cadmia" / "config.yaml"


def report_paths_mode() -> str:
    """How text reports print file paths: ``relative`` (the default) or ``absolute``, from
    ``report_paths:`` in ~/.cadmia/config.yaml. JSON results and the database always keep the
    absolute path and the file hash, whatever this says."""
    try:
        import yaml
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:                       # no file, unreadable, or no PyYAML: the default
        return "relative"
    return "absolute" if str(data.get("report_paths", "")).strip().lower() == "absolute" else "relative"


def set_report_paths_mode(mode: str) -> Path:
    """Write ``report_paths: <mode>`` to ~/.cadmia/config.yaml, keeping any other keys there."""
    import yaml
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}
    data["report_paths"] = mode
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return CONFIG_PATH


def shown_path(path) -> str:
    """A path as a text report prints it. Relative mode: relative to the project, else to the
    current folder, else only the file name, so a shared report does not carry the user's home
    folder. Absolute mode: the full path."""
    if not path:
        return ""
    p = Path(path)
    full = p.resolve() if p.exists() else p
    if report_paths_mode() == "absolute":
        return str(full)
    if not full.is_absolute():
        return str(p)
    for base in (find_project_root(), Path.cwd().resolve()):
        if base is not None:
            try:
                return str(full.relative_to(base))
            except ValueError:
                pass
    return full.name


TEXT_WIDTH = 80


def fit_width(text: str, width: int = TEXT_WIDTH) -> str:
    """Wrap the prose lines of a text report at *width* columns, as Gaussian and ORCA keep
    theirs. A line whose content has a run of two spaces is a table row and is left alone:
    tables may run wider, and their own layout keeps them within 100 columns."""
    import textwrap
    out = []
    for line in text.split("\n"):
        body = line.lstrip(" ")
        if len(line) <= width or "  " in body:
            out.append(line)
            continue
        indent = " " * (len(line) - len(body))
        out += textwrap.wrap(body, width, initial_indent=indent, subsequent_indent=indent,
                             break_long_words=False, break_on_hyphens=False)
    return "\n".join(out)


PATHS_NOTE = {
    "relative": "relative to the project; absolute paths stay in results/*.json "
                "('cadmia paths absolute' prints them)",
    "absolute": "absolute ('cadmia paths relative' hides them); they show user "
                "and folder names, check before sharing this file",
}


def find_project_root(start: "Path | None" = None) -> Optional[Path]:
    """The nearest folder, from *start* up, that holds a project manifest; None outside one."""
    here = (start or Path.cwd()).resolve()
    return next((d for d in (here, *here.parents) if (d / MANIFEST_NAME).is_file()), None)


def stage_result_path(stage: str, item: str = "", root: "Path | None" = None) -> Optional[Path]:
    """Where *stage* keeps its result inside the project, or None outside a project."""
    root = root or find_project_root()
    if root is None:
        return None
    if stage in MULTI_ITEM_STAGES:
        return root / RESULTS_DIR / stage / f"{item or stage}.json"
    return root / RESULTS_DIR / f"{stage}.json"


def _source(payload: dict) -> str:
    """What the result was computed from: the adduct, the structure, or the stage's inputs."""
    sources = payload.get("sources")
    return (payload.get("adduct_path") or (payload.get("structure") or {}).get("source_file")
            or (json.dumps(sources, sort_keys=True) if sources else ""))


def save_stage_result(stage: str, payload: dict, report_text: str, item: str = "") -> Optional[Path]:
    """Save a module's result, data and full text report in one file, if inside a project.

    The text is rendered from the same result at the same moment as the data, so the two can
    never disagree, and `cadmia report` can assemble the output without recomputing anything.
    A result is never overwritten by one computed from other files: running the same inputs
    again updates its file, and different inputs go to the next free copy, ``<name>-2.json``,
    ``<name>-3.json``… That name is the result's persistent id (``config_id`` for an adsorption
    configuration).
    """
    path = stage_result_path(stage, item)
    if path is None:
        return None
    source = _source(payload)
    if source:
        original = path
        copies = (path.with_name(f"{path.stem}-{n}.json") for n in range(2, 1000))
        path = next(c for c in (path, *copies) if not c.is_file() or _saved_source(c) in ("", source))
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"config_id": path.stem} if stage == "analysis" else {}
    if source and path != original:
        extra["copy_of"] = original.name
    data = json_safe({**payload, **extra, "module": stage, "report_text": report_text})
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    return path


def _saved_source(path: Path) -> str:
    try:
        return _source(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, AttributeError):
        return ""


def json_safe(value):
    """NaN and infinities as null: JSON has no spelling for them, and a reader would choke."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


STAGE_ORDER = ("structure", "cdft", "analysis", "stability")


def load_stage_results(root: "Path | None" = None, copies: bool = True) -> dict:
    """Every saved module result of the project, as {module: [dict, ...]} in module order.

    A multi-item module contributes one dict per file, sorted by name; a module that was not run
    is absent. With ``copies=False`` the copies (``cdft-2.json``, made from other inputs) are left
    out and listed under ``"_copies"``. Unreadable files are skipped with their error under
    ``"_errors"``.
    """
    root = root or find_project_root()
    found: dict = {}
    if root is None:
        return found
    base = root / RESULTS_DIR
    errors = []
    for stage in STAGE_ORDER:
        files = (sorted((base / stage).glob("*.json"), key=_copy_order) if stage in MULTI_ITEM_STAGES
                 else sorted([base / f"{stage}.json", *base.glob(f"{stage}-*.json")], key=_copy_order))
        items = []
        for f in files:
            if not f.is_file():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("not a CADMIA module result (no JSON object)")
                if not copies and data.get("copy_of"):
                    found.setdefault("_copies", []).append(str(f.relative_to(root)))
                    continue
                items.append({**data, "_file": str(f.relative_to(root))})
            except (OSError, ValueError) as exc:
                errors.append(f"{f.relative_to(root)}: {exc}")
        if items:
            found[stage] = items
    if errors:
        found["_errors"] = errors
    return found


def _copy_order(path: Path):
    """``name``, ``name-2``, ``name-3``… in that order, after any other name."""
    base, _, n = path.stem.rpartition("-")
    return (base, int(n)) if n.isdigit() and base else (path.stem, 1)


def saved_report(stage: str, root: "Path | None" = None) -> "tuple[str, Path]":
    """The full text report of *stage* from what it saved in the project, without recomputing
    anything: ``cadmia <stage> --report FILE`` with no inputs. A multi-item stage gives every
    item, each under its own heading. Returns ``(text, results_dir)``; raises LookupError
    outside a project or before the stage has run."""
    root = root or find_project_root()
    if root is None:
        raise LookupError("--report without inputs reads the project's results/; run it inside "
                          "a `cadmia init` project, or give the module its inputs.")
    items = load_stage_results(root).get(stage)
    if not items:
        raise LookupError(f"No saved {stage} result in {root / RESULTS_DIR}: run the module first.")
    texts = [d.get("report_text", "").rstrip() for d in items]
    if len(items) > 1:
        names = [Path(d.get("_file") or f"item {i}").stem for i, d in enumerate(items, 1)]
        texts = [f" ── {name} ".ljust(72, "─") + "\n" + t for name, t in zip(names, texts)]
    return "\n\n".join(texts) + "\n", root / RESULTS_DIR
