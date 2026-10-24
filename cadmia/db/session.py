"""Database connection management and schema initialization."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DEFAULT_DB: Path = Path.home() / ".cadmia" / "cadmia.db"

_DDL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS molecules (
    id          TEXT PRIMARY KEY,   -- SHA256 of .xyz content
    name        TEXT,
    formula     TEXT,
    n_atoms     INTEGER,
    xyz_path    TEXT,
    created_at  DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calculations (
    id          TEXT PRIMARY KEY,   -- UUID v4
    molecule_id TEXT REFERENCES molecules(id),
    engine      TEXT NOT NULL,      -- gaussian | orca | xtb
    task        TEXT NOT NULL,      -- sp | opt | freq | opt+freq
    functional  TEXT,
    basis       TEXT,
    dispersion  TEXT,               -- d3bj | d4 | none
    energy_au   REAL,
    wall_time_s INTEGER,
    status      TEXT NOT NULL,      -- done | failed | running
    input_path  TEXT,
    output_path TEXT,
    created_at  DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS descriptors (
    id              TEXT PRIMARY KEY,
    calculation_id  TEXT NOT NULL REFERENCES calculations(id),
    atom_index      INTEGER NOT NULL,
    atom_symbol     TEXT NOT NULL,
    fukui_minus     REAL,
    fukui_plus      REAL,
    delta_f         REAL,
    homo_eV         REAL,
    lumo_eV         REAL,
    gap_eV          REAL,
    hardness        REAL,
    mu              REAL,
    omega           REAL
);

CREATE TABLE IF NOT EXISTS bond_descriptors (
    id              TEXT PRIMARY KEY,
    calculation_id  TEXT NOT NULL REFERENCES calculations(id),
    atom_a_idx      INTEGER NOT NULL,
    atom_b_idx      INTEGER NOT NULL,
    atom_a_symbol   TEXT,
    atom_b_symbol   TEXT,
    bond_order      REAL,
    bond_order_type TEXT,           -- wiberg | mayer | nlmo | multiwfn
    distance_A      REAL,
    bond_type       TEXT            -- covalent | ionic | dative | weak
);

CREATE TABLE IF NOT EXISTS adsorption (
    id              TEXT PRIMARY KEY,
    cluster_id      TEXT NOT NULL REFERENCES molecules(id),
    probe           TEXT NOT NULL,  -- H2 | CO2 | NH3
    site_atom_idx   INTEGER,
    config_index    INTEGER,
    e_ads_kcal      REAL,
    e_bsse_kcal     REAL,
    delta_bond_A    REAL,
    calc_id         TEXT REFERENCES calculations(id)
);

CREATE TABLE IF NOT EXISTS run_log (
    id          TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    action      TEXT NOT NULL,
    input_hash  TEXT,
    output_hash TEXT,
    status      TEXT NOT NULL,      -- ok | warning | error
    message     TEXT,
    timestamp   DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stability (
    id                TEXT PRIMARY KEY,
    molecule_id       TEXT NOT NULL REFERENCES molecules(id),
    engine            TEXT,
    e_mixed_au        REAL,
    e_rel_au          REAL,           -- E(mixed) − Σ c_i·E(pure_i)
    e_rel_kcal        REAL,
    e_rel_eV_per_atom REAL,
    n_atoms           INTEGER,
    references_json   TEXT,           -- [{label, formula, coefficient, e_au}]
    balanced          INTEGER,        -- 1 if stoichiometry balances, else 0
    energy_basis      TEXT,           -- E0 | E | H | G; NULL on older rows = E
    created_at        DATETIME DEFAULT (datetime('now'))
);

-- ── Schema v2 (traceability; additive) ──

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS systems (
    id          TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(id),
    name        TEXT NOT NULL,      -- e.g. "Al2TiO5 n=1"
    formula     TEXT,
    description TEXT,
    created_at  DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS structures (
    id           TEXT PRIMARY KEY,  -- SHA256 of canonical XYZ text (same scheme as molecules)
    system_id    TEXT REFERENCES systems(id),
    name         TEXT,
    formula      TEXT,
    n_atoms      INTEGER,
    charge       INTEGER DEFAULT 0,
    multiplicity INTEGER DEFAULT 1,
    xyz          TEXT,              -- canonical XYZ text (self-contained, no path dependency)
    cell_json    TEXT,              -- NULL for molecular/cluster structures
    created_at   DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,   -- SHA256 of file content
    path        TEXT NOT NULL,
    kind        TEXT,               -- output | input | wfn | gbw | cube | xyz | other
    size_bytes  INTEGER,
    created_at  DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS structure_files (
    id           TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES structures(id),
    file_id      TEXT NOT NULL REFERENCES files(id),
    role         TEXT               -- geometry | source | render | other
);

CREATE TABLE IF NOT EXISTS provenance (
    id               TEXT PRIMARY KEY,
    source_file      TEXT,
    file_hash        TEXT,           -- SHA256 of source_file
    parser           TEXT,
    parser_version   TEXT,
    command          TEXT,           -- CLI invocation that produced the record
    software_version TEXT,           -- CADMIA (+ engine) version
    timestamp        TEXT,
    warnings_json    TEXT
);

-- ── adsorption split into configs (geometry) + results (energetics) ──

CREATE TABLE IF NOT EXISTS adsorption_configs (
    id               TEXT PRIMARY KEY,
    structure_id     TEXT NOT NULL REFERENCES structures(id),   -- isolated cluster
    probe            TEXT NOT NULL,
    site_atom_idx    INTEGER,
    site_atom_symbol TEXT,
    config_index     INTEGER,
    orientation      TEXT,
    approach_angle   REAL,
    azimuthal_angle  REAL,
    placement_dist_A REAL,
    xyz_path         TEXT,
    created_at       DATETIME DEFAULT (datetime('now'))
);

-- A configuration IS the cluster, the probe, the site and the orientation. The
-- orientation string already carries mode, polar and azimuth ("bidentate_p20_a120"), which is
-- exactly what tells two placements apart physically — and what the filenames had been using
-- all along. `config_index` is the ordinal of a sweep, not a property of what the sweep
-- produced: it shifts whenever the clash filter rejects a different number of configurations,
-- so two rounds over the same cluster would collide on it. The index makes the database refuse
-- such a collision instead of silently handing back the wrong row.
CREATE UNIQUE INDEX IF NOT EXISTS ux_adsorption_config_identity
    ON adsorption_configs (structure_id, probe, site_atom_idx, orientation);

CREATE TABLE IF NOT EXISTS adsorption_results (
    id                 TEXT PRIMARY KEY,
    config_id          TEXT NOT NULL REFERENCES adsorption_configs(id),
    e_ads_kcal         REAL,
    e_bsse_kcal        REAL,
    e_def_cluster_kcal REAL,
    e_def_probe_kcal   REAL,
    e_int_kcal         REAL,
    delta_bond_A       REAL,
    activated          INTEGER,
    calc_id            TEXT REFERENCES calculations(id),
    provenance_id      TEXT REFERENCES provenance(id),
    created_at         DATETIME DEFAULT (datetime('now'))
);

-- The middle of the chain. `geometry` writes the starting
-- configuration and `analyze` writes the final energetics, but what happened in between —
-- which configurations survived the xTB screen, which collapsed onto which minimum, and what
-- skeleton the cluster relaxed to — lived only in a sidecar. Both screening stages record the
-- same shape of information, so they share one table with a `stage` discriminator.
--
-- The skeleton class belongs HERE and not in adsorption_configs: the same starting geometry
-- relaxes to different skeletons depending on the method (GFN2-xTB and DFT can disagree), so it
-- is a property of the relaxation, not of the placement.
CREATE TABLE IF NOT EXISTS screening_results (
    id               TEXT PRIMARY KEY,
    stage            TEXT NOT NULL,          -- prescreen (xTB) | rank (DFT)
    config_id        TEXT REFERENCES adsorption_configs(id),   -- NULL if geometry --db never ran
    source           TEXT NOT NULL,          -- input .xyz (prescreen) or output file (rank)
    optimized_xyz    TEXT,
    energy_hartree   REAL,
    rel_kcal         REAL,
    rank             INTEGER,
    kept             INTEGER,
    converged        INTEGER,
    duplicate_of     TEXT,                   -- source of the representative it collapsed onto
    cluster_skeleton TEXT,                   -- reference | isomer-N
    provenance_id    TEXT REFERENCES provenance(id),
    created_at       DATETIME DEFAULT (datetime('now'))
);

-- One row per file per stage: re-running a screen updates rather than piling up duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS ux_screening_stage_source
    ON screening_results (stage, source);

-- ── HPC jobs + workflow runs ──

CREATE TABLE IF NOT EXISTS hpc_jobs (
    id              TEXT PRIMARY KEY,
    calc_id         TEXT REFERENCES calculations(id),
    profile         TEXT,
    job_id          TEXT,            -- scheduler (SLURM) job id
    label           TEXT,
    engine          TEXT,
    status          TEXT,            -- pending | queued | running | done | failed | unknown
    remote_work_dir TEXT,
    remote_input    TEXT,
    remote_output   TEXT,
    created_at      DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    id          TEXT PRIMARY KEY,
    command     TEXT,                -- exact CLI invocation
    agent       TEXT,
    status      TEXT,                -- ok | warning | error
    ids_json    TEXT,                -- ids produced by the run (per table)
    timestamp   TEXT
);
"""

# Additive column migrations for pre-v2 databases (ALTER TABLE is idempotent via inspection).
_V2_COLUMNS = {
    "calculations": [("provenance_id", "TEXT REFERENCES provenance(id)")],
    "descriptors":  [("provenance_id", "TEXT REFERENCES provenance(id)")],
    "adsorption":   [("provenance_id", "TEXT REFERENCES provenance(id)")],
    "stability":    [("provenance_id", "TEXT REFERENCES provenance(id)"),
                     # Which energy e_rel is a difference of.
                     ("energy_basis", "TEXT")],
    # The skeleton class travels with the result that produced it.
    "adsorption_results": [("cluster_skeleton", "TEXT")],
}


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in _V2_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


# The file a project keeps its own database in. Looked for upwards from the working
# directory, so a campaign run from inside its own tree keeps its results apart from
# unrelated runs.
PROJECT_DB_NAME = "cadmia.db"


def resolve_db_path(explicit: "str | Path | None" = None,
                    start: "Path | None" = None) -> Path:
    """Where to persist: the explicit path, else the project's own database, else the global one.

    A database found by walking up from the working directory wins over the one in the home
    directory. That is what decides whether the database serves a project or becomes a drawer.
    Only an EXISTING project database is adopted — a new one is never created by surprise in
    whatever directory the user happened to be in.
    """
    if explicit:
        return Path(explicit)
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / ".cadmia" / PROJECT_DB_NAME
        if candidate.exists():
            return candidate
    return DEFAULT_DB


# Bumped whenever the schema changes in a way an older CADMIA cannot read. Stored in SQLite's
# own user_version so that a database written by a NEWER version is detected instead of being
# opened and half-understood: the column migrations only ever add what is missing, so without
# this a newer file looks merely "complete" to older code.
SCHEMA_VERSION = 2


def _stamp_schema_version(conn: sqlite3.Connection) -> None:
    """Record the schema version, or refuse a database written by a newer CADMIA."""
    found = conn.execute("PRAGMA user_version").fetchone()[0]
    if found > SCHEMA_VERSION:
        raise RuntimeError(
            f"This database was written with schema version {found} and this CADMIA understands "
            f"up to {SCHEMA_VERSION}. Upgrade CADMIA rather than writing to it: the migrations "
            "only add missing columns, so an older version would read it as if it were complete."
        )
    if found < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def init_db(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Create (or open) the database and initialize the schema.

    Returns an open connection. The caller is responsible for closing it.
    Prefer get_connection() for production code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 30 s instead of the 5 s default: a pipeline and an `analyze` writing at the same
    # time is the normal case here, and 5 s is short enough to fail on a slow disk.
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Set per-connection PRAGMAs explicitly rather than relying on the copy embedded
    # in the DDL script (foreign_keys is per-connection and not persisted).
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_DDL)
    _apply_column_migrations(conn)
    _stamp_schema_version(conn)
    conn.commit()
    return conn


@contextmanager
def get_connection(path: Path = DEFAULT_DB) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that opens the DB, commits on exit, and rolls back on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 30 s instead of the 5 s default: a pipeline and an `analyze` writing at the same
    # time is the normal case here, and 5 s is short enough to fail on a slow disk.
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
