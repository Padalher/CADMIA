"""Provenance — traceability metadata attached to core objects (calculations, descriptors…)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

_HASH_CHUNK_SIZE = 1 << 20   # 1 MiB — avoids reading a whole large output into memory at once


def _sha256_file(path: Path) -> str:
    """SHA256 of *path*, streamed in chunks (no file-size cap yet — a transversal follow-up)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


_GIT_TIMEOUT_S = 2.0   # a hung git must never block writing a result
_MAX_COMMAND_CHARS = 400   # a shell glob of thousands of files must not fill the sidecar


def command_line() -> str:
    """The current invocation, with the program by name: its install path would put the
    user's home directory into every shared report and database row."""
    return " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])


@lru_cache(maxsize=1)
def _git_commit() -> Optional[str]:
    """Short commit of the checkout this package lives in, or None.

    Only a source checkout (development, editable install) has one; a tarball install does
    not, and that is not an error. The `-dirty` suffix matters as much as the hash: data
    produced from an edited tree cannot be regenerated from the commit alone.
    """
    repo = Path(__file__).resolve().parent.parent.parent
    # `git -C` walks up until it finds a .git, so an installation that lives inside somebody
    # else's repository — a venv under a versioned $HOME, say — would stamp THAT repo's commit
    # as the software version. Only a checkout of CADMIA itself has a .git right here.
    if not (repo / ".git").exists():
        return None

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                 text=True, timeout=_GIT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    head = _git("rev-parse", "--short", "HEAD")
    if not head:
        return None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{head}-dirty" if dirty else head


def software_version() -> str:
    """`cadmia <version> (<commit>)` — the commit is what `<version>` alone cannot say."""
    from cadmia import __version__
    commit = _git_commit()
    return f"cadmia {__version__} ({commit})" if commit else f"cadmia {__version__}"


@dataclass
class Provenance:
    """Where a piece of data came from and how it was produced."""

    source_file:      Optional[str] = None
    parser:           Optional[str] = None
    parser_version:   Optional[str] = None
    command:          Optional[str] = None
    timestamp:        Optional[str] = None
    file_hash:        Optional[str] = None
    warnings:         List[str] = field(default_factory=list)
    software_version: Optional[str] = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        parser: Optional[str] = None,
        parser_version: Optional[str] = None,
        command: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        software_version: Optional[str] = None,
    ) -> "Provenance":
        """Build provenance from a file: records its SHA256 and the current timestamp."""
        p = Path(path)
        file_hash = _sha256_file(p) if p.exists() else None
        notes = list(warnings or [])
        if file_hash is None:
            # A provenance with no hash cannot prove which file it describes, which is the whole
            # point of recording one. Silence would let it pass for a complete record.
            notes.append(f"source file not found when hashing: {p}")
        return cls(
            source_file=str(p),
            parser=parser,
            parser_version=parser_version,
            command=command,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=file_hash,
            warnings=notes,
            software_version=software_version,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def for_run(cls, *, command: Optional[str] = None) -> "Provenance":
        """Provenance of a *run* rather than of a file read: what code ran, when, and how.

        The command is truncated: a shell expands `configs/*.xyz` before the process sees it,
        so argv can carry a thousand paths, and repeating them in every sidecar buries the
        record it was meant to be. What identifies the run is the head of the command plus
        how many arguments followed.
        """
        text = command if command is not None else command_line()
        argv = text.split()
        if len(text) > _MAX_COMMAND_CHARS:
            head = text[:_MAX_COMMAND_CHARS].rsplit(" ", 1)[0]
            elided = len(argv) - len(head.split())
            text = f"{head} … (+{elided} arguments)"
        return cls(
            command=text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            software_version=software_version(),
        )
