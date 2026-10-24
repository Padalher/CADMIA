"""JSON sidecars and durable text writes.

Every agent that writes artifacts to disk also drops its machine-readable record next to
them, with a canonical name and no flag to remember: the geometries a step produces are for
the next engine run, the sidecar is for the next *step* (and for the record of what was
decided). `cadmia submit` already worked this way — it writes cadmia_jobs.json because
`cadmia monitor` needs it — and this generalizes that.

The state files these produce (``pipeline.json``, ``cadmia_jobs.json``, the per-stage
JSON) drive `--resume`: a run interrupted by a power cut must find them intact, not
half-written. `atomic_write_text` is the single primitive for that — write to a sibling
temp file, fsync, then `os.replace` (atomic on POSIX) — so a reader never sees a truncated
file. Every persistent writer in CADMIA routes through it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from cadmia.core.provenance import Provenance


@contextlib.contextmanager
def file_lock(path: "Path | str", *, blocking: bool = True) -> Iterator[None]:
    """Advisory exclusive lock on ``<path>.lock``, released on exit or process death.

    Serializes concurrent writers of a shared state file — two ``cadmia monitor`` on the same
    ``cadmia_jobs.json`` — so a load→modify→save cycle cannot silently lose another process's
    update (a ``done`` job reverting to ``running``). Uses a POSIX advisory lock
    (`fcntl.flock`), which the OS releases automatically if the holder crashes, so there is no
    stale-lock problem. With ``blocking=False`` a lock already held raises `BlockingIOError`.
    On a platform without `fcntl` the lock degrades to a no-op.
    """
    try:
        import fcntl
    except ImportError:                       # non-POSIX: no advisory locks
        yield
        return
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(fd, flags)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def atomic_write_text(path: "Path | str", text: str, *, encoding: str = "utf-8") -> Path:
    """Write *text* to *path* atomically: no reader ever sees a partial file.

    Writes to a temporary sibling on the same filesystem, flushes it to disk, then renames
    it over the destination with `os.replace` (atomic on POSIX). Creates the parent
    directory. Raises `OSError` on failure, leaving any previous *path* untouched.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def read_json(path: "Path | str") -> Any:
    """Load JSON from *path*, turning a truncated/corrupt file into an actionable error.

    A state file left half-written by an interrupted run would otherwise surface as a raw
    `JSONDecodeError` deep inside `--resume`; this raises `ValueError` naming the file and
    what to do about it.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{p} is not valid JSON (truncated or corrupted, likely an interrupted "
            f"write): {exc}. Delete it to start fresh, or restore it from a backup."
        ) from exc


def _stamp_provenance(text: str, artifacts: "Optional[List[Path | str]]" = None) -> str:
    """Record which code wrote this sidecar, so its artifacts can be regenerated.

    The parameters of a step are already in the sidecar; the code that read them is not,
    and `__version__` alone does not separate two commits of the same alpha. Stamped here
    rather than in each agent because this is the single point every sidecar goes through.

    A caller that stamped its own provenance keeps it, and anything that is not a JSON
    object is written untouched — a sidecar is a convenience and must never cost a result.
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or "provenance" in data:
            return text
        stamp = Provenance.for_run().to_dict()
        provenance = {k: v for k, v in stamp.items() if v not in (None, [])}
        if artifacts:
            fingerprint = artifact_fingerprint(artifacts)
            if fingerprint:
                provenance["artifacts"] = fingerprint
                data["artifact_paths"] = [str(a) for a in artifacts]
        data["provenance"] = provenance
        return json.dumps(data, indent=2)
    except (ValueError, TypeError):
        return text


def artifact_fingerprint(paths: "Iterable[Path | str]") -> Optional[dict]:
    """A single hash over the files a sidecar describes, plus how many there were.

    The provenance stamp records the command, the commit and the hour, which says how an
    artifact was made but not WHICH artifact. Regenerate the files and the sidecar still reads
    as if it described them; the two drift apart with nothing to say so.

    The fingerprint is the sha256 of one sorted line per file, ``<name> <sha256 of contents>``,
    so it changes if any listed file changes, disappears, or is replaced by one of the same
    size. Names are relative to the common parent so the value does not depend on where the
    directory sits.

    Returns None when there is nothing to fingerprint. Files that cannot be read are listed in
    ``missing`` rather than silently skipped: a fingerprint over an unknown subset would be
    worse than none.
    """
    items = [Path(p) for p in paths]
    if not items:
        return None

    base = Path(os.path.commonpath([str(p.resolve().parent) for p in items])) \
        if len(items) > 1 else items[0].resolve().parent

    lines, missing = [], []
    for item in sorted(items, key=str):
        try:
            name = str(item.resolve().relative_to(base))
        except ValueError:
            name = item.name
        digest = _sha256_file(item)
        if digest is None:
            missing.append(name)
            continue
        lines.append(f"{name} {digest}")

    if not lines:
        return {"n": 0, "sha256": None, "missing": missing}
    combined = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    out = {"n": len(lines), "sha256": combined}
    if missing:
        out["missing"] = missing
    return out


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def verify_sidecar(path: "Path | str", pattern: str = "*.xyz") -> List[str]:
    """Compare a sidecar's artifact fingerprint against what is on disk now.

    Returns one message per discrepancy, empty when everything matches. Three things can be
    wrong and they are not the same thing:

    * ``modified`` — the listed files are not the ones that were listed. Something regenerated
      them, and the sidecar's command and commit no longer describe what is there.
    * ``missing`` — a listed file is gone.
    * ``unlisted`` — the directory holds files matching *pattern* that the sidecar does not
      list. This is the one that bites: a second run with fewer sites overwrites the names it
      produces and leaves the rest of the previous run in place, so the next stage reads more
      files than the sidecar accounts for.

    A sidecar written before fingerprints existed has nothing to compare and reports nothing.
    """
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"could not read the sidecar {target}: {exc}"]
    if not isinstance(data, dict):
        return []

    recorded = (data.get("provenance") or {}).get("artifacts")
    listed = data.get("artifact_paths") or []
    if not recorded or not listed:
        return []

    problems: List[str] = []
    now = artifact_fingerprint(listed)
    if now and now.get("missing"):
        problems.append(f"{len(now['missing'])} file(s) listed in {target.name} are gone: "
                        + ", ".join(sorted(now["missing"])[:5]))
    if now and now.get("sha256") and now["sha256"] != recorded.get("sha256"):
        problems.append(f"the files listed in {target.name} are not the ones it describes; "
                        "they were regenerated after it was written")

    known = {Path(p).name for p in listed}
    on_disk = {f.name for f in target.parent.glob(pattern)}
    unlisted = sorted(on_disk - known)
    if unlisted:
        problems.append(f"{len(unlisted)} file(s) in {target.parent.name}/ are not listed in "
                        f"{target.name}, probably left by an earlier run: "
                        + ", ".join(unlisted[:5]) + (" …" if len(unlisted) > 5 else ""))
    return problems


def check_input_directory(directory: "Path | str") -> List[str]:
    """Warn about a directory whose sidecar no longer describes what it holds.

    Called by the stages that CONSUME a directory another stage produced, because that is where
    the mismatch does damage: reading files the previous stage's record does not account for.
    Returns messages, never raises, and says nothing when there is no sidecar to check.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    problems: List[str] = []
    for name in ("geometry.json", "prescreen.json"):
        sidecar = d / name
        if sidecar.is_file():
            problems += verify_sidecar(sidecar)
    return problems


def write_json_sidecar(path: "Path | str", text: str,
                       artifacts: "Iterable[Path | str]" = ()) -> Tuple[Optional[str], Optional[str]]:
    """Write *text* to *path*, creating the parent directory.

    ``artifacts`` are the files this sidecar describes. Their fingerprint is recorded so that a
    later run can tell whether the sidecar still matches what is on disk.

    Returns ``(written_path, error)``: a sidecar is a convenience, so a failure to write it
    is reported as a warning by the caller and never aborts the step that produced the data.
    """
    target = Path(path)
    text = _stamp_provenance(text, list(artifacts))
    try:
        atomic_write_text(target, text if text.endswith("\n") else text + "\n")
    except OSError as exc:
        return None, f"could not write the JSON sidecar {target}: {exc}"
    return str(target), None
