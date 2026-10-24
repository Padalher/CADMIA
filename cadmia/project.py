"""Project scaffolding for CADMIA (`cadmia init`).

Creates a tidy working directory for a study: the pipeline subfolders, a project manifest
with the level-of-theory defaults, and (optionally) a local SQLite database.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List

_SUBDIRS = ("scf", "configs", "adsorption", "reports")

_MANIFEST = """# CADMIA project manifest
name: {name}
created: {created}

# Local database for this project (used by --db / --db-path).
db: ./.cadmia/cadmia.db
"""


def init_project(
    directory: "str | Path" = ".",
    *,
    name: str = "",
    with_db: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Scaffold a project under *directory*.

    Creates the pipeline subfolders, ``cadmia_project.yaml`` and (when *with_db*) an
    initialized ``.cadmia/cadmia.db``, the database every command finds from inside the tree. Existing files are preserved unless *overwrite* is set.
    Returns a summary dict with ``root``, ``created`` (new paths) and ``skipped``.
    """
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    project_name = name or root.name

    created: List[str] = []
    skipped: List[str] = []

    for sub in _SUBDIRS:
        d = root / sub
        if d.exists():
            skipped.append(f"{sub}/")
        else:
            d.mkdir(parents=True)
            created.append(f"{sub}/")

    manifest = root / "cadmia_project.yaml"
    if manifest.exists() and not overwrite:
        skipped.append(manifest.name)
    else:
        manifest.write_text(
            _MANIFEST.format(name=project_name, created=date.today().isoformat()),
            encoding="utf-8",
        )
        created.append(manifest.name)

    from cadmia.db.session import PROJECT_DB_NAME
    db_path = root / ".cadmia" / PROJECT_DB_NAME
    if with_db:
        if db_path.exists() and not overwrite:
            skipped.append(f".cadmia/{PROJECT_DB_NAME}")
        else:
            from cadmia.db.session import init_db
            db_path.parent.mkdir(exist_ok=True)
            init_db(db_path).close()
            created.append(f".cadmia/{PROJECT_DB_NAME}")

    return {
        "root": str(root),
        "name": project_name,
        "created": created,
        "skipped": skipped,
        "db": str(db_path) if with_db else None,
    }
