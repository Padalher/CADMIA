"""Data access layer — insert/query operations for the CADMIA database."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ─────────────────────────────────────────────
# v2: provenance & structures
# ─────────────────────────────────────────────

def insert_provenance(conn: sqlite3.Connection, provenance: Any) -> str:
    """Insert a provenance row from a cadmia.core.Provenance (or compatible dict); return its id."""
    d = provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance)
    prov_id = _uuid()
    conn.execute(
        """INSERT INTO provenance
           (id, source_file, file_hash, parser, parser_version,
            command, software_version, timestamp, warnings_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (prov_id, d.get("source_file"), d.get("file_hash"), d.get("parser"),
         d.get("parser_version"), d.get("command"), d.get("software_version"),
         d.get("timestamp") or _now(), json.dumps(d.get("warnings") or [])),
    )
    return prov_id


def _default_provenance(output_path: str | Path, *, parser: str) -> Any:
    from cadmia import __version__
    from cadmia.core import Provenance
    version = f"cadmia {__version__}"
    path = Path(output_path) if output_path else None
    if path and path.is_file():
        return Provenance.from_file(path, parser=parser, software_version=version)
    # No real file to hash (e.g. in-memory geometry): record what we have.
    return Provenance(source_file=str(output_path or ""), parser=parser,
                      software_version=version)


def upsert_structure(
    conn: sqlite3.Connection,
    *,
    structure_id: str,
    xyz_text: str,
    name: str = "",
    formula: str = "",
    n_atoms: Optional[int] = None,
    charge: int = 0,
    multiplicity: int = 1,
    system_id: Optional[str] = None,
) -> str:
    """Insert a structures row if absent (idempotent on the SHA256 id); return the id.

    Mirror of the molecules table during the v1→v2 transition: callers that write a
    molecule also write the structure with the *same id*, so existing FKs stay valid.
    """
    if conn.execute("SELECT id FROM structures WHERE id = ?", (structure_id,)).fetchone():
        return structure_id
    conn.execute(
        """INSERT INTO structures
           (id, system_id, name, formula, n_atoms, charge, multiplicity, xyz, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (structure_id, system_id, name, formula, n_atoms, charge, multiplicity,
         xyz_text, _now()),
    )
    return structure_id


def insert_workflow_run(
    conn: sqlite3.Connection,
    *,
    command: str,
    agent: str,
    status: str = "ok",
    ids: Optional[Dict[str, Any]] = None,
) -> str:
    """Record one CLI/pipeline invocation in workflow_runs; return its id."""
    run_id = _uuid()
    conn.execute(
        """INSERT INTO workflow_runs (id, command, agent, status, ids_json, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, command, agent, status, json.dumps(ids or {}, default=str), _now()),
    )
    return run_id


# ─────────────────────────────────────────────
# molecules
# ─────────────────────────────────────────────

def upsert_molecule(
    conn: sqlite3.Connection,
    xyz_path: str | Path,
    name: str = "",
) -> str:
    """Insert a molecule if it does not exist; return its id (SHA256 of .xyz content).

    Idempotent: calling twice with the same file returns the same id.
    """
    path = Path(xyz_path)
    mol_id = _sha256_file(path)

    if conn.execute("SELECT id FROM molecules WHERE id = ?", (mol_id,)).fetchone():
        return mol_id

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        n_atoms = int(lines[0].strip())
    except (ValueError, IndexError):
        n_atoms = None

    formula = _formula_from_xyz(lines[2:2 + n_atoms] if n_atoms else lines[2:])

    conn.execute(
        """INSERT INTO molecules (id, name, formula, n_atoms, xyz_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (mol_id, name or path.stem, formula, n_atoms, str(path.resolve()), _now()),
    )
    upsert_structure(conn, structure_id=mol_id, xyz_text=text,
                     name=name or path.stem, formula=formula, n_atoms=n_atoms)
    return mol_id


def get_molecule(conn: sqlite3.Connection, mol_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM molecules WHERE id = ?", (mol_id,)).fetchone()
    return dict(row) if row else None


def get_structure(conn: sqlite3.Connection, structure_id: str) -> Optional[Dict[str, Any]]:
    """v2 reader: fetch a structures row (falls back to molecules for pre-v2 data)."""
    row = conn.execute("SELECT * FROM structures WHERE id = ?", (structure_id,)).fetchone()
    if row:
        return dict(row)
    return get_molecule(conn, structure_id)


def _formula_from_xyz(atom_lines: List[str]) -> str:
    from cadmia.core.elements import hill_formula
    symbols: List[str] = []
    for line in atom_lines:
        parts = line.split()
        if parts:
            symbols.append(parts[0].capitalize())
    if not symbols:
        return ""
    return hill_formula(symbols)


# ─────────────────────────────────────────────
# calculations
# ─────────────────────────────────────────────

def insert_calculation(
    conn: sqlite3.Connection,
    molecule_id: str,
    engine: str,
    task: str,
    output_path: str | Path,
    energy_au: Optional[float] = None,
    functional: Optional[str] = None,
    basis: Optional[str] = None,
    dispersion: Optional[str] = None,
    wall_time_s: Optional[int] = None,
    status: str = "done",
    input_path: Optional[str] = None,
    provenance_id: Optional[str] = None,
) -> str:
    """Insert a calculation record and return its id (UUID)."""
    calc_id = _uuid()
    conn.execute(
        """INSERT INTO calculations
           (id, molecule_id, engine, task, functional, basis, dispersion,
            energy_au, wall_time_s, status, input_path, output_path, created_at,
            provenance_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            calc_id, molecule_id, engine, task, functional, basis, dispersion,
            energy_au, wall_time_s, status,
            str(Path(input_path).resolve()) if input_path else None,
            str(Path(output_path).resolve()) if output_path else None,
            _now(),
            provenance_id,
        ),
    )
    return calc_id


def get_calculations(
    conn: sqlite3.Connection,
    molecule_id: str,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM calculations WHERE molecule_id = ? ORDER BY created_at",
        (molecule_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# descriptors  (from CDFTResult)
# ─────────────────────────────────────────────

def insert_descriptors_from_cdft(
    conn: sqlite3.Connection,
    cdft_result: Any,   # cadmia.agents.cdft_agent.CDFTResult
    calculation_id: str,
    provenance_id: Optional[str] = None,
) -> List[str]:
    """Insert per-atom descriptors from a CDFTResult; return list of inserted ids."""
    gd = cdft_result.global_descriptors
    ids = []
    for site in cdft_result.sites:
        desc_id = _uuid()
        conn.execute(
            """INSERT INTO descriptors
               (id, calculation_id, atom_index, atom_symbol,
                fukui_minus, fukui_plus, delta_f,
                homo_eV, lumo_eV, gap_eV, hardness, mu, omega, provenance_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                desc_id, calculation_id,
                site.atom_index, site.atom_symbol,
                site.fukui_minus, site.fukui_plus, site.delta_f,
                gd.get("homo_eV"),    gd.get("lumo_eV"),  gd.get("gap_eV"),
                gd.get("hardness_eV"), gd.get("mu_eV"),   gd.get("omega_eV"),
                provenance_id,
            ),
        )
        ids.append(desc_id)
    return ids


def get_descriptors(
    conn: sqlite3.Connection,
    calculation_id: str,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM descriptors WHERE calculation_id = ? ORDER BY atom_index",
        (calculation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# run_log
# ─────────────────────────────────────────────

def log_run(
    conn: sqlite3.Connection,
    agent: str,
    action: str,
    status: str,
    message: str = "",
    input_hash: Optional[str] = None,
    output_hash: Optional[str] = None,
) -> str:
    """Log an agent action to run_log; return the inserted id."""
    log_id = _uuid()
    conn.execute(
        """INSERT INTO run_log
           (id, agent, action, input_hash, output_hash, status, message, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (log_id, agent, action, input_hash, output_hash, status, message, _now()),
    )
    return log_id


def get_run_log(
    conn: sqlite3.Connection,
    agent: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return the most recent run_log entries, optionally filtered by agent."""
    if agent:
        rows = conn.execute(
            "SELECT * FROM run_log WHERE agent = ? ORDER BY timestamp DESC LIMIT ?",
            (agent, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM run_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# High-level orchestration
# ─────────────────────────────────────────────

def persist_cdft_result(
    conn: sqlite3.Connection,
    cdft_result: Any,   # cadmia.agents.cdft_agent.CDFTResult
    *,
    engine: str,
    output_path: str | Path,
    task: str = "cdft",
    provenance: Optional[Any] = None,   # cadmia.core.Provenance
) -> Dict[str, Any]:
    """Persist a full CDFTResult: molecule + calculation + per-atom descriptors + run_log.

    The molecule geometry is reconstructed from the ranked sites (each carries its
    symbol and coordinates), so no separate .xyz file is required. The molecule id
    is the SHA256 of the canonical XYZ text, making this idempotent at the molecule
    level: the same geometry always maps to the same molecule id.

    A provenance row is always written (built from ``output_path`` when not supplied)
    and linked to the calculation and descriptor rows.

    Returns a dict with ``molecule_id``, ``calculation_id``, ``descriptor_ids``
    and ``provenance_id``.
    """
    sites = sorted(cdft_result.sites, key=lambda s: s.atom_index)
    atom_lines = [
        f"{s.atom_symbol} {s.coordinates[0]:.8f} {s.coordinates[1]:.8f} {s.coordinates[2]:.8f}"
        for s in sites
    ]
    name = cdft_result.cluster or "cluster"
    xyz_text = f"{len(sites)}\n{name}\n" + "\n".join(atom_lines) + "\n"
    mol_id = hashlib.sha256(xyz_text.encode("utf-8")).hexdigest()
    formula = _formula_from_xyz(atom_lines)

    if not conn.execute("SELECT id FROM molecules WHERE id = ?", (mol_id,)).fetchone():
        conn.execute(
            """INSERT INTO molecules (id, name, formula, n_atoms, xyz_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mol_id, name, formula, len(sites), "", _now()),
        )
    upsert_structure(conn, structure_id=mol_id, xyz_text=xyz_text,
                     name=name, formula=formula, n_atoms=len(sites))

    prov_id = insert_provenance(
        conn, provenance or _default_provenance(output_path, parser="cdft"),
    )
    calc_id = insert_calculation(
        conn, molecule_id=mol_id, engine=engine, task=task,
        output_path=output_path, status="done", provenance_id=prov_id,
    )
    descriptor_ids = insert_descriptors_from_cdft(conn, cdft_result, calc_id,
                                                  provenance_id=prov_id)
    log_run(
        conn, agent="cdft", action="persist", status="ok",
        message=f"{name}: {len(descriptor_ids)} descriptors persisted",
        input_hash=cdft_result.calculation_id,
    )
    return {
        "molecule_id": mol_id,
        "calculation_id": calc_id,
        "descriptor_ids": descriptor_ids,
        "provenance_id": prov_id,
    }


def persist_imported_calculation(
    conn: sqlite3.Connection,
    *,
    atoms: List[str],
    positions: Any,                 # iterable of (x, y, z)
    name: str,
    engine: str,
    energy_au: Optional[float] = None,
    task: str = "scf",
    functional: Optional[str] = None,
    basis: Optional[str] = None,
    dispersion: Optional[str] = None,
    status: str = "done",
    output_path: str | Path = "",
    provenance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Persist a calculation imported from an existing output file.

    Reconstructs the molecule/structure from the parsed geometry (same SHA256-of-XYZ id
    scheme as persist_cdft_result, so re-importing the same geometry is idempotent), writes
    a provenance row (from *output_path* when not supplied) and one calculation row.
    Returns ``molecule_id``, ``calculation_id``, ``provenance_id``, ``formula`` and ``n_atoms``.
    """
    atom_lines = [
        f"{a} {float(p[0]):.8f} {float(p[1]):.8f} {float(p[2]):.8f}"
        for a, p in zip(atoms, positions)
    ]
    label = name or "structure"
    xyz_text = f"{len(atom_lines)}\n{label}\n" + "\n".join(atom_lines) + "\n"
    mol_id = hashlib.sha256(xyz_text.encode("utf-8")).hexdigest()
    formula = _formula_from_xyz(atom_lines)

    if not conn.execute("SELECT id FROM molecules WHERE id = ?", (mol_id,)).fetchone():
        conn.execute(
            """INSERT INTO molecules (id, name, formula, n_atoms, xyz_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mol_id, label, formula, len(atom_lines), "", _now()),
        )
    upsert_structure(conn, structure_id=mol_id, xyz_text=xyz_text,
                     name=label, formula=formula, n_atoms=len(atom_lines))

    prov_id = insert_provenance(
        conn, provenance or _default_provenance(output_path, parser="import"),
    )
    calc_id = insert_calculation(
        conn, molecule_id=mol_id, engine=engine, task=task, output_path=output_path,
        energy_au=energy_au, functional=functional, basis=basis, dispersion=dispersion,
        status=status, provenance_id=prov_id,
    )
    log_run(conn, agent="import", action="persist", status="ok",
            message=f"{label} ({formula}): {engine} {task}", input_hash=mol_id)
    return {
        "molecule_id": mol_id,
        "calculation_id": calc_id,
        "provenance_id": prov_id,
        "formula": formula,
        "n_atoms": len(atom_lines),
    }


def persist_bond_descriptors(
    conn: sqlite3.Connection,
    calculation_id: str,
    bonds: Any,                       # iterable of cadmia.io.mayer.MayerBond
    *,
    order_type: str = "mayer",
    positions: Any = None,            # optional Nx3 for distance_A
) -> List[str]:
    """Insert one bond_descriptors row per bond; return the inserted ids.

    When *positions* (atom coordinates, 0-based to match the Mayer indices) is given, the
    interatomic distance is stored in distance_A.
    """
    import math

    ids: List[str] = []
    for b in bonds:
        distance = None
        if positions is not None:
            try:
                pi, pj = positions[b.atom_i], positions[b.atom_j]
                distance = math.dist((float(pi[0]), float(pi[1]), float(pi[2])),
                                     (float(pj[0]), float(pj[1]), float(pj[2])))
            except (IndexError, TypeError):
                distance = None
        bid = _uuid()
        conn.execute(
            """INSERT INTO bond_descriptors
               (id, calculation_id, atom_a_idx, atom_b_idx, atom_a_symbol, atom_b_symbol,
                bond_order, bond_order_type, distance_A, bond_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bid, calculation_id, b.atom_i, b.atom_j, b.symbol_i, b.symbol_j,
             float(b.order), order_type, distance, None),
        )
        ids.append(bid)
    return ids


# ─────────────────────────────────────────────
# adsorption
# ─────────────────────────────────────────────

def insert_adsorption(
    conn: sqlite3.Connection,
    cluster_id: str,
    probe: str,
    *,
    site_atom_idx: Optional[int] = None,
    config_index: Optional[int] = None,
    e_ads_kcal: Optional[float] = None,
    e_bsse_kcal: Optional[float] = None,
    delta_bond_A: Optional[float] = None,
    calc_id: Optional[str] = None,
    provenance_id: Optional[str] = None,
) -> str:
    """Insert an adsorption configuration row; return its id."""
    ads_id = _uuid()
    conn.execute(
        """INSERT INTO adsorption
           (id, cluster_id, probe, site_atom_idx, config_index,
            e_ads_kcal, e_bsse_kcal, delta_bond_A, calc_id, provenance_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ads_id, cluster_id, probe, site_atom_idx, config_index,
         e_ads_kcal, e_bsse_kcal, delta_bond_A, calc_id, provenance_id),
    )
    return ads_id


def insert_adsorption_config(
    conn: sqlite3.Connection,
    *,
    structure_id: str,
    probe: str,
    site_atom_idx: Optional[int] = None,
    site_atom_symbol: Optional[str] = None,
    config_index: Optional[int] = None,
    orientation: Optional[str] = None,
    approach_angle: Optional[float] = None,
    azimuthal_angle: Optional[float] = None,
    placement_dist_A: Optional[float] = None,
    xyz_path: Optional[str] = None,
) -> str:
    """Insert an adsorption_configs row (v2); return its id.

    Idempotent on (structure_id, probe, site_atom_idx, orientation). Re-persisting
    the same starting configuration returns the existing id, so analysis results attach to the
    config the geometry agent created.

    The key is deliberately NOT `config_index`: that is the ordinal a sweep happened to assign,
    it moves whenever the clash filter rejects a different number of placements, and two sweeps
    over the same cluster file collide on it — the older row wins and the newer geometry is
    dropped without a word. `orientation` carries mode, polar and azimuth, which is intrinsic to
    the placement and reproducible. `config_index` is kept in the row as presentation order.
    """
    row = conn.execute(
        """SELECT id FROM adsorption_configs
           WHERE structure_id = ? AND probe = ?
             AND site_atom_idx IS ? AND orientation IS ?""",
        (structure_id, probe, site_atom_idx, orientation),
    ).fetchone()
    if row:
        return row[0]
    cfg_id = _uuid()
    conn.execute(
        """INSERT INTO adsorption_configs
           (id, structure_id, probe, site_atom_idx, site_atom_symbol, config_index,
            orientation, approach_angle, azimuthal_angle, placement_dist_A, xyz_path,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cfg_id, structure_id, probe, site_atom_idx, site_atom_symbol, config_index,
         orientation, approach_angle, azimuthal_angle, placement_dist_A, xyz_path,
         _now()),
    )
    return cfg_id


def find_adsorption_config_by_stem(conn: sqlite3.Connection, stem: str) -> Optional[str]:
    """The config id whose starting geometry has this filename stem, or None.

    `geometry` writes ``<stem>.xyz`` and every later stage keeps that stem: `prepare` names the
    input after it and the engine names the output after the input. So the stem is what survives
    the round trip from the configuration to the adduct output that `analyze` finally reads,
    which is the link the two ends need.

    It is a LOOKUP, not an identity: the row it finds already carries the intrinsic identity of
    Orientation included. When there is no such row, the caller must not invent one.
    """
    if not stem:
        return None
    row = conn.execute(
        "SELECT id FROM adsorption_configs "
        "WHERE xyz_path IS NOT NULL AND (xyz_path LIKE ? OR xyz_path LIKE ? OR xyz_path = ?)",
        (f"%/{stem}.xyz", f"%\\{stem}.xyz", f"{stem}.xyz"),
    ).fetchone()
    return row[0] if row else None


def find_adsorption_config_by_xyz(conn: sqlite3.Connection, xyz_path: str) -> Optional[str]:
    """The config id whose starting geometry is this file, or None.

    Matched on the resolved path: `geometry` wrote the file and the screen reads that same
    file, so the path is the direct link between the two stages and needs no filename parsing
    — which is exactly the fragile convention this chain exists to replace.
    """
    resolved = str(Path(xyz_path).resolve())
    row = conn.execute(
        "SELECT id FROM adsorption_configs WHERE xyz_path = ? OR xyz_path = ?",
        (resolved, str(xyz_path)),
    ).fetchone()
    return row[0] if row else None


def upsert_screening_result(
    conn: sqlite3.Connection,
    *,
    stage: str,
    source: str,
    config_id: Optional[str] = None,
    optimized_xyz: Optional[str] = None,
    energy_hartree: Optional[float] = None,
    rel_kcal: Optional[float] = None,
    rank: Optional[int] = None,
    kept: Optional[bool] = None,
    converged: Optional[bool] = None,
    duplicate_of: Optional[str] = None,
    cluster_skeleton: Optional[str] = None,
    provenance_id: Optional[str] = None,
) -> str:
    """Record one file's fate at one screening stage; return the row id.

    Keyed on (stage, source): re-running a screen over the same files updates the row instead
    of piling up a second history of the same decision.
    """
    row = conn.execute(
        "SELECT id FROM screening_results WHERE stage = ? AND source = ?", (stage, source)
    ).fetchone()
    values = (config_id, optimized_xyz, energy_hartree, rel_kcal, rank,
              None if kept is None else int(kept),
              None if converged is None else int(converged),
              duplicate_of, cluster_skeleton, provenance_id)
    if row:
        conn.execute(
            """UPDATE screening_results SET config_id = ?, optimized_xyz = ?, energy_hartree = ?,
                   rel_kcal = ?, rank = ?, kept = ?, converged = ?, duplicate_of = ?,
                   cluster_skeleton = ?, provenance_id = ? WHERE id = ?""",
            (*values, row[0]),
        )
        return row[0]
    sid = _uuid()
    conn.execute(
        """INSERT INTO screening_results
           (id, stage, source, config_id, optimized_xyz, energy_hartree, rel_kcal, rank,
            kept, converged, duplicate_of, cluster_skeleton, provenance_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (sid, stage, source, *values, _now()),
    )
    return sid


def insert_adsorption_result(
    conn: sqlite3.Connection,
    config_id: str,
    *,
    e_ads_kcal: Optional[float] = None,
    e_bsse_kcal: Optional[float] = None,
    e_def_cluster_kcal: Optional[float] = None,
    e_def_probe_kcal: Optional[float] = None,
    e_int_kcal: Optional[float] = None,
    delta_bond_A: Optional[float] = None,
    activated: Optional[bool] = None,
    calc_id: Optional[str] = None,
    provenance_id: Optional[str] = None,
) -> str:
    """Upsert an adsorption_results row (v2, FK to adsorption_configs); return its id.

    Idempotent on ``config_id``: re-analyzing the same configuration (e.g. a resumed
    pipeline re-run with ``--db``) refreshes the existing result instead of inserting a
    duplicate, so a config never ends up with two competing energies. Since
    `insert_adsorption_config` is itself idempotent on (structure, probe, site, orientation) --
    the intrinsic identity (cluster, probe, site, orientation), not the positional `config_index` it replaced -- the same
    physical configuration always resolves to the same ``config_id`` and thus the same row.
    """
    activated_int = None if activated is None else int(activated)
    row = conn.execute(
        "SELECT id FROM adsorption_results WHERE config_id = ?", (config_id,)
    ).fetchone()
    if row:
        res_id = row[0]
        conn.execute(
            """UPDATE adsorption_results
               SET e_ads_kcal = ?, e_bsse_kcal = ?, e_def_cluster_kcal = ?,
                   e_def_probe_kcal = ?, e_int_kcal = ?, delta_bond_A = ?, activated = ?,
                   calc_id = ?, provenance_id = ?, created_at = ?
               WHERE id = ?""",
            (e_ads_kcal, e_bsse_kcal, e_def_cluster_kcal, e_def_probe_kcal, e_int_kcal,
             delta_bond_A, activated_int, calc_id, provenance_id, _now(), res_id),
        )
        return res_id
    res_id = _uuid()
    conn.execute(
        """INSERT INTO adsorption_results
           (id, config_id, e_ads_kcal, e_bsse_kcal, e_def_cluster_kcal,
            e_def_probe_kcal, e_int_kcal, delta_bond_A, activated, calc_id,
            provenance_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (res_id, config_id, e_ads_kcal, e_bsse_kcal, e_def_cluster_kcal,
         e_def_probe_kcal, e_int_kcal, delta_bond_A, activated_int, calc_id, provenance_id,
         _now()),
    )
    return res_id


def persist_geometry_result(
    conn: sqlite3.Connection,
    geometry_result: Any,   # cadmia.agents.geometry_agent.GeometryResult
    *,
    cluster_xyz: str | Path,
) -> Dict[str, Any]:
    """Persist a GeometryResult: cluster molecule + starting configs + run_log.

    Writes one adsorption_configs row per config (v2, full placement geometry) and
    mirrors each into the v1 adsorption table while it still has readers. Energies
    are left NULL — they are only known once the adsorption calculations are run.
    """
    cluster_id = upsert_molecule(conn, cluster_xyz, name=geometry_result.cluster or "")
    ads_ids, config_ids = [], []
    for cfg in geometry_result.configs:
        config_ids.append(insert_adsorption_config(
            conn, structure_id=cluster_id, probe=cfg.probe,
            site_atom_idx=cfg.site_atom_idx, site_atom_symbol=cfg.site_atom_symbol,
            config_index=cfg.config_index, orientation=cfg.orientation,
            approach_angle=cfg.approach_angle, azimuthal_angle=cfg.azimuthal_angle,
            placement_dist_A=cfg.placement_dist,
            xyz_path=str(cfg.xyz_path) if cfg.xyz_path else None,
        ))
        ads_ids.append(insert_adsorption(
            conn, cluster_id, cfg.probe,
            site_atom_idx=cfg.site_atom_idx, config_index=cfg.config_index,
        ))
    log_run(
        conn, agent="geometry", action="persist", status="ok",
        message=f"{geometry_result.cluster}: {len(config_ids)} {geometry_result.probe} configs persisted",
    )
    return {"cluster_id": cluster_id, "adsorption_ids": ads_ids, "config_ids": config_ids}


def persist_calculation_result(
    conn: sqlite3.Connection,
    calculation_result: Any,   # cadmia.agents.calculation_agent.CalculationResult
    *,
    task: str = "sp",
) -> Dict[str, Any]:
    """Persist a CalculationResult: one calculation + hpc_jobs row per submitted job + run_log.

    Idempotent on ``(profile, label)``: re-persisting a batch (a resumed `cadmia run --db`)
    refreshes the existing calculation/hpc_jobs rows instead of inserting duplicates, so a job
    is never recorded twice. The job label (the input stem) is unique within a profile.
    """
    profile = calculation_result.profile
    calc_ids, hpc_job_ids = [], []
    for job in calculation_result.jobs:
        output_path = job.output_path or job.remote_output
        input_path  = str(job.input_path) if job.input_path else None
        prior = conn.execute(
            "SELECT id, calc_id FROM hpc_jobs WHERE profile = ? AND label = ?",
            (profile, job.label),
        ).fetchone()
        if prior:
            hpc_id, calc_id = prior[0], prior[1]
            conn.execute(
                """UPDATE calculations SET engine = ?, task = ?, status = ?,
                       input_path = ?, output_path = ? WHERE id = ?""",
                (job.engine, task, job.status,
                 str(Path(input_path).resolve()) if input_path else None,
                 str(Path(output_path).resolve()) if output_path else None, calc_id),
            )
            conn.execute(
                """UPDATE hpc_jobs SET job_id = ?, engine = ?, status = ?,
                       remote_work_dir = ?, remote_input = ?, remote_output = ?
                   WHERE id = ?""",
                (job.job_id or None, job.engine, job.status,
                 job.remote_work_dir or None, job.remote_input or None,
                 job.remote_output or None, hpc_id),
            )
        else:
            calc_id = insert_calculation(
                conn, molecule_id=None, engine=job.engine, task=task,
                output_path=output_path, status=job.status, input_path=input_path,
            )
            hpc_id = _uuid()
            conn.execute(
                """INSERT INTO hpc_jobs
                   (id, calc_id, profile, job_id, label, engine, status,
                    remote_work_dir, remote_input, remote_output, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (hpc_id, calc_id, profile, job.job_id or None,
                 job.label, job.engine, job.status,
                 job.remote_work_dir or None, job.remote_input or None,
                 job.remote_output or None, _now()),
            )
        calc_ids.append(calc_id)
        hpc_job_ids.append(hpc_id)
    log_run(
        conn, agent="calculation", action="submit",
        status="ok" if calculation_result.n_failed == 0 else "warning",
        message=f"profile {calculation_result.profile}: {len(calc_ids)} job(s) submitted",
    )
    return {"calculation_ids": calc_ids, "hpc_job_ids": hpc_job_ids}


def persist_benchmark_result(
    conn: sqlite3.Connection,
    benchmark_result: Any,   # cadmia.agents.benchmark_agent.BenchmarkResult
) -> Dict[str, Any]:
    """Persist a BenchmarkResult: a calculation row per engine + a run_log comparison entry."""
    params = benchmark_result.params
    functional = getattr(params, "functional", None) if params else None
    basis = getattr(params, "basis", None) if params else None
    dispersion = getattr(params, "dispersion", None) if params else None
    task = getattr(params, "task", None) if params else None

    calc_ids = []
    for eng_res in (benchmark_result.gaussian_result, benchmark_result.orca_result):
        if eng_res is None:
            continue
        calc_ids.append(insert_calculation(
            conn, molecule_id=None, engine=eng_res.engine, task=task or "opt",
            output_path=eng_res.output_path, energy_au=eng_res.energy_hartree,
            functional=functional, basis=basis, dispersion=dispersion,
            wall_time_s=int(eng_res.wall_time_s) if eng_res.wall_time_s else None,
            status="done" if eng_res.validation_ok else "failed",
        ))

    comp = benchmark_result.comparison
    if comp:
        status = "ok" if comp.engine_agreement else "warning"
        message = (f"{benchmark_result.cluster}: {comp.recommendation}; "
                   f"ΔE={comp.delta_e_kcal} kcal/mol, RMSD={comp.rmsd_ang} Å")
    else:
        status, message = "warning", f"{benchmark_result.cluster}: no comparison produced"
    log_run(conn, agent="benchmark", action="compare", status=status, message=message)
    return {"calculation_ids": calc_ids}


def persist_analysis_result(
    conn: sqlite3.Connection,
    analysis_result: Any,   # cadmia.agents.analysis_agent.AnalysisResult
    *,
    calc_id: Optional[str] = None,
    provenance: Optional[Any] = None,   # cadmia.core.Provenance
) -> Dict[str, Any]:
    """Persist an AnalysisResult as one adsorption row (+ run_log).

    The cluster molecule is reconstructed from the embedded isolated-cluster geometry
    (SHA256 of the canonical XYZ → idempotent molecule id, same scheme as the CDFT path).
    The BSSE-corrected adsorption energy is stored in ``e_bsse_kcal`` when available;
    ``delta_bond_A`` holds the largest probe-bond elongation. When a ``provenance``
    is supplied it is stored and linked to the adsorption row.
    """
    atom_lines = [
        f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
        for sym, pos in zip(analysis_result.cluster_atoms, analysis_result.cluster_positions)
    ]
    name = analysis_result.cluster or "cluster"
    xyz_text = f"{len(atom_lines)}\n{name}\n" + "\n".join(atom_lines) + "\n"
    cluster_id = hashlib.sha256(xyz_text.encode("utf-8")).hexdigest()
    formula = _formula_from_xyz(atom_lines)
    if not conn.execute("SELECT id FROM molecules WHERE id = ?", (cluster_id,)).fetchone():
        conn.execute(
            """INSERT INTO molecules (id, name, formula, n_atoms, xyz_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cluster_id, name, formula, len(atom_lines), "", _now()),
        )
    upsert_structure(conn, structure_id=cluster_id, xyz_text=xyz_text,
                     name=name, formula=formula, n_atoms=len(atom_lines))

    prov_id = insert_provenance(conn, provenance) if provenance is not None else None
    delta_bond = max((b.delta_A for b in analysis_result.probe_bonds), default=None)
    ads_id = insert_adsorption(
        conn, cluster_id, analysis_result.probe,
        site_atom_idx=analysis_result.site_atom_index,
        config_index=analysis_result.config_index,
        e_ads_kcal=analysis_result.e_ads_kcal,
        e_bsse_kcal=analysis_result.e_ads_bsse_kcal,
        delta_bond_A=delta_bond,
        calc_id=calc_id,
        provenance_id=prov_id,
    )
    # Attach the result to the configuration row `geometry` created, found by the filename stem
    # the two stages share. Creating one here instead would have to invent an identity: calling
    # insert_adsorption_config without `orientation` makes the idempotent lookup match ANY adduct
    # on the same (cluster, probe, site), so the second configuration of a site overwrote the
    # energy of the first. Better to persist no configuration than a wrong one.
    stem = Path(analysis_result.adduct_path).stem if analysis_result.adduct_path else ""
    config_id = find_adsorption_config_by_stem(conn, stem)

    result_id = None
    note = ""
    if config_id is None:
        # adsorption_results.config_id is NOT NULL, and there is no honest value to put there:
        # the row belongs to a configuration nobody recorded. The energetics still land in the
        # `adsorption` row above, so nothing is lost; what is skipped is the v2 result row.
        note = (f" — no configuration row matches '{stem}', so the v2 result row was skipped; "
                "run `geometry --db` first to link the two")
    else:
        result_id = insert_adsorption_result(
            conn, config_id,
            e_ads_kcal=analysis_result.e_ads_kcal,
            e_bsse_kcal=analysis_result.e_ads_bsse_kcal,
            e_def_cluster_kcal=analysis_result.e_def_cluster_kcal,
            e_def_probe_kcal=analysis_result.e_def_probe_kcal,
            e_int_kcal=analysis_result.e_int_kcal,
            delta_bond_A=delta_bond,
            activated=analysis_result.activated,
            calc_id=calc_id,
            provenance_id=prov_id,
        )
    log_run(
        conn, agent="analysis", action="persist",
        status="ok" if config_id else "warning",
        message=(f"{name}/{analysis_result.probe} cfg {analysis_result.config_index}: "
                 f"E_ads={analysis_result.e_ads_kcal} kcal/mol"
                 + (" (activated)" if analysis_result.activated else "") + note),
    )
    return {"cluster_id": cluster_id, "adsorption_id": ads_id,
            "config_id": config_id, "result_id": result_id, "provenance_id": prov_id}


def persist_stability_result(
    conn: sqlite3.Connection,
    stability_result: Any,   # cadmia.agents.stability_agent.StabilityResult
    *,
    provenance: Optional[Any] = None,   # cadmia.core.Provenance
) -> Dict[str, Any]:
    """Persist a StabilityResult: mixed molecule + one stability row + run_log.

    The mixed molecule is reconstructed from its embedded geometry (SHA256 of the canonical
    XYZ → idempotent molecule id, same scheme as the CDFT/analysis paths). When a
    ``provenance`` is supplied it is stored and linked to the stability row.
    """
    atom_lines = [
        f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
        for sym, pos in zip(stability_result.mixed_atoms, stability_result.mixed_positions)
    ]
    name = stability_result.mixed or stability_result.formula or "cluster"
    xyz_text = f"{len(atom_lines)}\n{name}\n" + "\n".join(atom_lines) + "\n"
    mol_id = hashlib.sha256(xyz_text.encode("utf-8")).hexdigest()
    if not conn.execute("SELECT id FROM molecules WHERE id = ?", (mol_id,)).fetchone():
        conn.execute(
            """INSERT INTO molecules (id, name, formula, n_atoms, xyz_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mol_id, name, stability_result.formula, len(atom_lines), "", _now()),
        )
    upsert_structure(conn, structure_id=mol_id, xyz_text=xyz_text,
                     name=name, formula=stability_result.formula, n_atoms=len(atom_lines))

    prov_id = insert_provenance(conn, provenance) if provenance is not None else None
    refs_json = json.dumps([
        {"label": r.label, "formula": r.formula, "coefficient": r.coefficient, "e_au": r.e_hartree}
        for r in stability_result.references
    ])
    stab_id = _uuid()
    conn.execute(
        """INSERT INTO stability
           (id, molecule_id, engine, e_mixed_au, e_rel_au, e_rel_kcal,
            e_rel_eV_per_atom, n_atoms, references_json, balanced, energy_basis, created_at,
            provenance_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (stab_id, mol_id, stability_result.engine, stability_result.e_mixed_hartree,
         stability_result.e_rel_hartree, stability_result.e_rel_kcal,
         stability_result.e_rel_eV_per_atom, stability_result.n_atoms,
         refs_json, 1 if stability_result.balanced else 0,
         stability_result.energy_basis, _now(), prov_id),
    )
    log_run(
        conn, agent="stability", action="persist",
        status="ok" if stability_result.balanced else "warning",
        message=(f"{name}: E_rel({stability_result.energy_basis})="
                 f"{stability_result.e_rel_kcal} kcal/mol"
                 + ("" if stability_result.balanced else " (stoichiometry unbalanced)")),
    )
    return {"molecule_id": mol_id, "stability_id": stab_id, "provenance_id": prov_id}
