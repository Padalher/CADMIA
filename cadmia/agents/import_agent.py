"""Import Agent — ingest existing calculation outputs into the CADMIA database.

Bridges "I already have a pile of .out/.log files" to the traceable DB v2: for each file it
detects the engine, parses the geometry and SCF energy, and writes a structure + calculation
+ provenance row (idempotent on the geometry's SHA256). Reuses the existing parsers — it does
not re-implement any parsing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Engine normal-termination markers (stable across versions).
_TERMINATION = {
    "gaussian": ("Normal termination",),
    "orca": ("****ORCA TERMINATED NORMALLY****", "ORCA TERMINATED NORMALLY"),
}


@dataclass
class ImportedItem:
    path:           str
    engine:         str = ""
    formula:        str = ""
    energy_au:      Optional[float] = None
    status:         str = ""
    structure_id:   Optional[str] = None
    calculation_id: Optional[str] = None
    n_bonds:        int = 0
    error:          Optional[str] = None


@dataclass
class ImportResult:
    items:    List[ImportedItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for it in self.items if it.error is None)

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_imported": self.n_ok,
                "n_total": len(self.items),
                "items": [it.__dict__ for it in self.items],
                "warnings": self.warnings,
            },
            indent=2,
        )


def _termination_status(path: Path, engine: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    markers = _TERMINATION.get(engine, ())
    return "done" if any(m in text for m in markers) else "incomplete"


def _parse_one(path: Path, engine: Optional[str]):
    from cadmia.agents.cdft_agent import _detect_engine
    from cadmia.io.gaussian import parse_gaussian_geometry
    from cadmia.io.orca import parse_orca_geometry
    from cadmia.io.electronic import parse_gaussian_electronic, parse_orca_electronic

    eng = engine or _detect_engine(path)
    if eng == "orca":
        structure = parse_orca_geometry(path)
        state = parse_orca_electronic(path, structure)
    else:
        structure = parse_gaussian_geometry(path)
        state = parse_gaussian_electronic(path, structure)
    return eng, structure, state


def import_outputs(
    paths: "List[str | Path]",
    *,
    db_path: "str | Path | None" = None,
    engine: Optional[str] = None,
    functional: Optional[str] = None,
    basis: Optional[str] = None,
    dispersion: Optional[str] = None,
    task: str = "scf",
    command: str = "",
) -> ImportResult:
    """Parse each output in *paths* and persist it to the database.

    engine/functional/basis/dispersion/task are recorded as given (engine is auto-detected
    per file when not supplied). Files that fail to parse are reported per item and never
    abort the batch.
    """
    from cadmia.db.session import init_db, DEFAULT_DB
    from cadmia.db import repository
    from cadmia.core import Provenance
    from cadmia import __version__

    result = ImportResult()
    target = Path(db_path) if db_path else DEFAULT_DB
    conn = init_db(target)
    try:
        for raw in paths:
            path = Path(raw)
            item = ImportedItem(path=str(path))
            try:
                eng, structure, state = _parse_one(path, engine)
                item.engine = eng
                item.energy_au = state.energy_hartree
                item.status = _termination_status(path, eng)

                provenance = Provenance.from_file(
                    path.resolve(), parser="import",
                    command=command or f"cadmia import {path}",
                    software_version=f"cadmia {__version__}",
                )
                ids = repository.persist_imported_calculation(
                    conn,
                    atoms=structure.atoms, positions=structure.positions,
                    name=structure.structure_id or path.stem,
                    engine=eng, energy_au=state.energy_hartree, task=task,
                    functional=functional, basis=basis, dispersion=dispersion,
                    status=item.status, output_path=path, provenance=provenance,
                )
                item.formula = ids["formula"]
                item.structure_id = ids["molecule_id"]
                item.calculation_id = ids["calculation_id"]

                # Mayer bond orders (ORCA Print[P_Mayer] 1) when present.
                from cadmia.io.mayer import parse_mayer
                mayer = parse_mayer(path)
                if mayer and mayer.bonds:
                    repository.persist_bond_descriptors(
                        conn, ids["calculation_id"], mayer.bonds,
                        order_type="mayer", positions=structure.positions,
                    )
                    item.n_bonds = len(mayer.bonds)

                repository.insert_workflow_run(
                    conn, command=command or f"cadmia import {path}", agent="import",
                    status="ok", ids=ids,
                )
            except Exception as exc:                 # one bad file must not stop the batch
                item.error = f"{type(exc).__name__}: {exc}"
                result.warnings.append(f"{path}: {item.error}")
            result.items.append(item)
        conn.commit()
    finally:
        conn.close()
    return result
