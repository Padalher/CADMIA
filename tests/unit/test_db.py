"""Tests for cadmia.db (session schema + repository data-access + CLI persistence)."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cadmia.agents.cdft_agent import CDFTResult, SiteDescriptor
from cadmia.db import repository
from cadmia.db.session import get_connection, init_db
from cadmia.db.repository import (
    _formula_from_xyz,
    get_calculations,
    get_descriptors,
    get_run_log,
    insert_calculation,
    insert_descriptors_from_cdft,
    log_run,
    persist_cdft_result,
    upsert_molecule,
)

_TABLES = ["molecules", "calculations", "descriptors",
           "bond_descriptors", "adsorption", "run_log", "stability",
           # v2 (traceability)
           "projects", "systems", "structures", "structure_files", "files", "provenance",
           # v2 steps 4–5
           "adsorption_configs", "adsorption_results", "hpc_jobs", "workflow_runs"]

_REAL = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_cdft"
_HAS_DATA = (_REAL / "N.out").exists()


# ─────────────────────────────────────────────
# session / schema
# ─────────────────────────────────────────────

def test_init_db_creates_all_tables(tmp_path):
    db = tmp_path / "t.db"
    conn = init_db(db)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for t in _TABLES:
            assert t in names
    finally:
        conn.close()

def test_get_connection_commits(tmp_path):
    db = tmp_path / "t.db"
    init_db(db).close()
    with get_connection(db) as conn:
        log_run(conn, agent="x", action="a", status="ok")
    # reopen: the commit should have persisted
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0] == 1

def test_get_connection_rolls_back_on_error(tmp_path):
    db = tmp_path / "t.db"
    init_db(db).close()
    with pytest.raises(RuntimeError):
        with get_connection(db) as conn:
            log_run(conn, agent="x", action="a", status="ok")
            raise RuntimeError("boom")
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0] == 0


# ─────────────────────────────────────────────
# _formula_from_xyz
# ─────────────────────────────────────────────

def test_formula_hill_order_with_carbon():
    lines = ["O 0 0 0", "H 0 0 1", "C 0 0 2", "H 0 0 3"]
    # Hill: C first, H second, then alphabetical
    assert _formula_from_xyz(lines) == "CH2O"

def test_formula_without_carbon_is_alphabetical():
    lines = ["Ti 0 0 0", "Al 0 0 1", "O 0 0 2", "O 0 0 3"]
    assert _formula_from_xyz(lines) == "AlO2Ti"

def test_formula_empty():
    assert _formula_from_xyz([]) == ""


# ─────────────────────────────────────────────
# molecules
# ─────────────────────────────────────────────

@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "t.db")
    yield c
    c.close()

@pytest.fixture
def xyz_file(tmp_path):
    p = tmp_path / "mol.xyz"
    p.write_text("3\nTi2O\nTi 0 0 0\nO 1.95 0 0\nO -1.95 0 0\n")
    return p

def test_upsert_molecule_inserts(conn, xyz_file):
    mol_id = upsert_molecule(conn, xyz_file, name="ti2o")
    row = conn.execute("SELECT * FROM molecules WHERE id = ?", (mol_id,)).fetchone()
    assert row["n_atoms"] == 3
    assert row["formula"] == "O2Ti"

def test_upsert_molecule_idempotent(conn, xyz_file):
    id1 = upsert_molecule(conn, xyz_file)
    id2 = upsert_molecule(conn, xyz_file)
    assert id1 == id2
    assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1


# ─────────────────────────────────────────────
# calculations & descriptors
# ─────────────────────────────────────────────

def test_insert_and_get_calculation(conn, xyz_file):
    mol_id = upsert_molecule(conn, xyz_file)
    calc_id = insert_calculation(conn, mol_id, engine="gaussian", task="opt",
                                 output_path="/tmp/out.log", energy_au=-123.4)
    calcs = get_calculations(conn, mol_id)
    assert len(calcs) == 1
    assert calcs[0]["id"] == calc_id
    assert calcs[0]["engine"] == "gaussian"
    assert calcs[0]["energy_au"] == -123.4


def _fabricate_cdft() -> CDFTResult:
    sites = [
        SiteDescriptor(1, "Ti", (0.0, 0.0, 0.0), 0.3, 0.1, 0.2, "electrophilic", 1),
        SiteDescriptor(2, "O",  (1.95, 0.0, 0.0), 0.1, 0.4, -0.3, "nucleophilic", 2),
    ]
    return CDFTResult(
        cluster="ti2o", calculation_id="hash123", charge_scheme="mulliken",
        sites=sites,
        global_descriptors={"homo_eV": -8.0, "lumo_eV": -4.0, "gap_eV": 4.0,
                            "hardness_eV": 2.0, "mu_eV": -6.0, "omega_eV": 9.0},
        sources={},
    )

def test_insert_descriptors_from_cdft(conn, xyz_file):
    mol_id = upsert_molecule(conn, xyz_file)
    calc_id = insert_calculation(conn, mol_id, "gaussian", "cdft", "/tmp/o.log")
    ids = insert_descriptors_from_cdft(conn, _fabricate_cdft(), calc_id)
    assert len(ids) == 2
    rows = get_descriptors(conn, calc_id)
    assert len(rows) == 2
    assert rows[0]["atom_symbol"] == "Ti"
    assert rows[0]["delta_f"] == 0.2
    assert rows[0]["gap_eV"] == 4.0


# ─────────────────────────────────────────────
# run_log
# ─────────────────────────────────────────────

def test_log_run_and_filter(conn):
    log_run(conn, agent="cdft", action="run", status="ok", message="m1")
    log_run(conn, agent="validation", action="validate", status="fail", message="m2")
    log_run(conn, agent="cdft", action="persist", status="ok")
    assert len(get_run_log(conn)) == 3
    cdft_only = get_run_log(conn, agent="cdft")
    assert len(cdft_only) == 2
    assert all(e["agent"] == "cdft" for e in cdft_only)


# ─────────────────────────────────────────────
# persist_cdft_result (high-level orchestration)
# ─────────────────────────────────────────────

def test_persist_cdft_result_full_chain(conn):
    ids = persist_cdft_result(conn, _fabricate_cdft(), engine="gaussian",
                              output_path="/tmp/N.out")
    assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM calculations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0] == 1
    assert len(ids["descriptor_ids"]) == 2
    # molecule reconstructed from sites: 2 atoms, formula O Ti
    mol = conn.execute("SELECT * FROM molecules WHERE id = ?", (ids["molecule_id"],)).fetchone()
    assert mol["n_atoms"] == 2
    assert mol["formula"] == "OTi"

def test_persist_cdft_result_idempotent_molecule(conn):
    """Persisting the same geometry twice reuses the molecule but adds a calculation."""
    persist_cdft_result(conn, _fabricate_cdft(), engine="gaussian", output_path="/tmp/a.out")
    persist_cdft_result(conn, _fabricate_cdft(), engine="orca", output_path="/tmp/b.out")
    assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM calculations").fetchone()[0] == 2


# ─────────────────────────────────────────────
# Schema v2: structures mirror, provenance, migration
# ─────────────────────────────────────────────

def test_upsert_molecule_mirrors_structure(conn, xyz_file):
    mol_id = upsert_molecule(conn, xyz_file, name="ti2o")
    row = conn.execute("SELECT * FROM structures WHERE id = ?", (mol_id,)).fetchone()
    assert row is not None
    assert row["formula"] == "O2Ti"
    assert row["n_atoms"] == 3
    assert row["xyz"].startswith("3\n")

def test_persist_cdft_result_writes_provenance(conn, tmp_path):
    out = tmp_path / "N.out"
    out.write_text("dummy gaussian output")
    ids = persist_cdft_result(conn, _fabricate_cdft(), engine="gaussian", output_path=out)
    assert ids["provenance_id"]
    prov = conn.execute("SELECT * FROM provenance WHERE id = ?",
                        (ids["provenance_id"],)).fetchone()
    assert prov["source_file"].endswith("N.out")
    assert len(prov["file_hash"]) == 64
    # calculation and descriptors are linked to the same provenance row
    calc = conn.execute("SELECT provenance_id FROM calculations WHERE id = ?",
                        (ids["calculation_id"],)).fetchone()
    assert calc["provenance_id"] == ids["provenance_id"]
    descs = conn.execute("SELECT provenance_id FROM descriptors").fetchall()
    assert all(d["provenance_id"] == ids["provenance_id"] for d in descs)
    # structures mirror written too (same id as molecule)
    assert conn.execute("SELECT COUNT(*) FROM structures WHERE id = ?",
                        (ids["molecule_id"],)).fetchone()[0] == 1

def test_insert_provenance_from_core_object(conn, tmp_path):
    from cadmia.core import Provenance
    src = tmp_path / "x.out"
    src.write_text("data")
    prov = Provenance.from_file(src, parser="cdft", command="cadmia cdft --n x.out",
                                warnings=["w1"], software_version="cadmia 0.1.0")
    prov_id = repository.insert_provenance(conn, prov)
    row = conn.execute("SELECT * FROM provenance WHERE id = ?", (prov_id,)).fetchone()
    assert row["command"] == "cadmia cdft --n x.out"
    assert row["warnings_json"] == '["w1"]'
    assert row["software_version"] == "cadmia 0.1.0"

def test_init_db_migrates_pre_v2_database(tmp_path):
    """init_db on a v1 database (without provenance_id columns) adds them."""
    import sqlite3
    db = tmp_path / "old.db"
    old = sqlite3.connect(str(db))
    old.execute("CREATE TABLE calculations (id TEXT PRIMARY KEY, engine TEXT)")
    old.commit()
    old.close()
    conn = init_db(db)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(calculations)")}
        assert "provenance_id" in cols
    finally:
        conn.close()


# ─────────────────────────────────────────────
# CLI integration: cadmia cdft --db / cadmia validate --db
# ─────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_DATA, reason="1tial_cdft reference data not present")
def test_cli_cdft_db_persists(tmp_path):
    from cadmia.cli import main
    db = tmp_path / "cli.db"
    result = CliRunner().invoke(main, [
        "cdft", "--n", str(_REAL / "N.out"), "--np1", str(_REAL / "N+1.out"),
        "--nm1", str(_REAL / "N-1.out"), "--cluster", "Al2TiO5",
        "--no-table", "--db", "--db-path", str(db),
    ])
    assert result.exit_code == 0, result.output
    assert "Persisted" in result.output
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0] == 8


@pytest.mark.skipif(not _HAS_DATA, reason="1tial_cdft reference data not present")
def test_cli_validate_db_logs(tmp_path):
    from cadmia.cli import main
    db = tmp_path / "cli.db"
    result = CliRunner().invoke(main, [
        "validate", str(_REAL / "N.out"), "--no-table", "--db", "--db-path", str(db),
    ])
    assert result.exit_code == 0, result.output
    with get_connection(db) as conn:
        rows = get_run_log(conn, agent="validation")
    assert len(rows) == 1
    assert rows[0]["status"] == "pass"


# ─────────────────────────────────────────────
# Phase 2 persistence: geometry / calculation / benchmark
# ─────────────────────────────────────────────

_CLUSTER_XYZ = Path(__file__).resolve().parent.parent / "fixtures" / "Al2TiO5_cluster.xyz"


def test_persist_geometry_result(tmp_path):
    from cadmia.agents.geometry_agent import GeometryResult, AdsorptionConfig
    cfgs = [
        AdsorptionConfig(site_atom_idx=1, site_atom_symbol="Ti", config_index=0,
                         orientation="eta1", probe="NH3", approach_angle=0.0,
                         azimuthal_angle=0.0, placement_dist=2.1),
        AdsorptionConfig(site_atom_idx=5, site_atom_symbol="Al", config_index=1,
                         orientation="eta1", probe="NH3", approach_angle=30.0,
                         azimuthal_angle=90.0, placement_dist=2.0),
    ]
    result = GeometryResult(cluster="Al2TiO5", probe="NH3", n_sites=2, configs=cfgs)

    db = tmp_path / "geo.db"
    conn = init_db(db)
    try:
        ids = repository.persist_geometry_result(conn, result, cluster_xyz=_CLUSTER_XYZ)
        conn.commit()
    finally:
        conn.close()

    assert len(ids["adsorption_ids"]) == 2
    assert len(ids["config_ids"]) == 2
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM adsorption").fetchone()[0] == 2
        cfgs = conn.execute(
            "SELECT probe, site_atom_symbol, orientation, placement_dist_A "
            "FROM adsorption_configs ORDER BY config_index"
        ).fetchall()
        assert [c["site_atom_symbol"] for c in cfgs] == ["Ti", "Al"]
        assert cfgs[0]["orientation"] == "eta1"
        assert cfgs[0]["placement_dist_A"] == pytest.approx(2.1)
        assert conn.execute("SELECT COUNT(*) FROM run_log WHERE agent='geometry'").fetchone()[0] == 1


def test_persist_calculation_result(tmp_path):
    from cadmia.agents.calculation_agent import CalculationResult, JobRecord
    jobs = [
        JobRecord(label="job1", engine="orca", input_path=Path("/tmp/job1.inp"),
                  remote_input="/scratch/job1.inp", remote_output="/scratch/job1.out",
                  remote_work_dir="/scratch", job_id="1001", status="queued"),
        JobRecord(label="job2", engine="gaussian", input_path=Path("/tmp/job2.gjf"),
                  remote_input="/scratch/job2.gjf", remote_output="/scratch/job2.log",
                  remote_work_dir="/scratch", job_id="1002", status="running"),
    ]
    result = CalculationResult(profile="hpc-test", jobs=jobs)

    db = tmp_path / "calc.db"
    conn = init_db(db)
    try:
        ids = repository.persist_calculation_result(conn, result, task="opt")
        conn.commit()
    finally:
        conn.close()

    assert len(ids["calculation_ids"]) == 2
    assert len(ids["hpc_job_ids"]) == 2
    with get_connection(db) as conn:
        rows = conn.execute("SELECT engine, task, status FROM calculations ORDER BY engine").fetchall()
        assert [r["engine"] for r in rows] == ["gaussian", "orca"]
        assert all(r["task"] == "opt" for r in rows)
        jobs = conn.execute(
            "SELECT profile, job_id, status, calc_id FROM hpc_jobs ORDER BY job_id"
        ).fetchall()
        assert [j["job_id"] for j in jobs] == ["1001", "1002"]
        assert all(j["profile"] == "hpc-test" for j in jobs)
        assert {j["calc_id"] for j in jobs} == set(ids["calculation_ids"])
        assert conn.execute("SELECT COUNT(*) FROM run_log WHERE agent='calculation'").fetchone()[0] == 1


def test_persist_calculation_result_is_idempotent_on_resume(tmp_path):
    """Re-persisting a batch (a resumed `cadmia run --db`) must not duplicate rows."""
    from cadmia.agents.calculation_agent import CalculationResult, JobRecord

    def _batch(status):
        return CalculationResult(profile="hpc-test", jobs=[
            JobRecord(label="job1", engine="orca", input_path=Path("/tmp/job1.inp"),
                      remote_input="/scratch/job1.inp", remote_output="/scratch/job1.out",
                      remote_work_dir="/scratch", job_id="1001", status=status),
        ])

    db = tmp_path / "calc.db"
    conn = init_db(db)
    try:
        repository.persist_calculation_result(conn, _batch("running"), task="opt")
        repository.persist_calculation_result(conn, _batch("done"), task="opt")   # resume
        conn.commit()
    finally:
        conn.close()

    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM calculations").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hpc_jobs").fetchone()[0] == 1
        # the row reflects the *latest* persist, not the first
        assert conn.execute("SELECT status FROM calculations").fetchone()["status"] == "done"
        assert conn.execute("SELECT status FROM hpc_jobs").fetchone()["status"] == "done"


def test_insert_adsorption_result_upserts_on_config(conn):
    """A second result for the same config updates it in place, never duplicates."""
    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    cfg = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="H2", site_atom_idx=1, config_index=0,
    )
    first  = repository.insert_adsorption_result(conn, cfg, e_ads_kcal=-12.5)
    second = repository.insert_adsorption_result(conn, cfg, e_ads_kcal=-13.9)   # re-analyze
    assert first == second
    rows = conn.execute(
        "SELECT e_ads_kcal FROM adsorption_results WHERE config_id = ?", (cfg,)
    ).fetchall()
    assert len(rows) == 1                                   # one row, not two
    assert rows[0]["e_ads_kcal"] == pytest.approx(-13.9)    # refreshed to the latest


def test_persist_benchmark_result(tmp_path):
    from cadmia.agents.benchmark_agent import (
        BenchmarkResult, EngineResult, BenchmarkComparison,
    )
    g = EngineResult(engine="gaussian", output_path=Path("/tmp/g.log"),
                     energy_hartree=-1709.16, wall_time_s=120.0, validation_ok=True)
    o = EngineResult(engine="orca", output_path=Path("/tmp/o.out"),
                     energy_hartree=-1709.80, wall_time_s=90.0, validation_ok=True)
    comp = BenchmarkComparison(delta_e_kcal=0.05, rmsd_ang=0.002, timing_ratio=0.75,
                               engine_agreement=True, recommendation="agree")
    result = BenchmarkResult(cluster="Al2TiO5", params=None,
                             gaussian_result=g, orca_result=o, comparison=comp)

    db = tmp_path / "bench.db"
    conn = init_db(db)
    try:
        ids = repository.persist_benchmark_result(conn, result)
        conn.commit()
    finally:
        conn.close()

    assert len(ids["calculation_ids"]) == 2
    with get_connection(db) as conn:
        energies = sorted(r["energy_au"] for r in
                          conn.execute("SELECT energy_au FROM calculations").fetchall())
        assert energies == [pytest.approx(-1709.80), pytest.approx(-1709.16)]
        log = get_run_log(conn, agent="benchmark")
        assert len(log) == 1 and log[0]["status"] == "ok"


# ─────────────────────────────────────────────
# v2 steps 4–5: adsorption split, workflow_runs
# ─────────────────────────────────────────────

def test_adsorption_config_find_or_create(conn):
    """Identity is cluster + probe + site + orientation."""
    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    a = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="NH3", site_atom_idx=1, config_index=0,
        site_atom_symbol="Ti", orientation="eta1_p20_a120",
    )
    b = repository.insert_adsorption_config(          # same placement, different sweep ordinal
        conn, structure_id="s1", probe="NH3", site_atom_idx=1, config_index=7,
        orientation="eta1_p20_a120",
    )
    c = repository.insert_adsorption_config(          # another site
        conn, structure_id="s1", probe="NH3", site_atom_idx=2, orientation="eta1_p20_a120",
    )
    d = repository.insert_adsorption_config(          # another orientation on the same site
        conn, structure_id="s1", probe="NH3", site_atom_idx=1, orientation="eta1_p40_a120",
    )
    assert a == b, "the sweep ordinal must not create a second row for the same placement"
    assert c != a
    assert d != a, "two different orientations are two different configurations"
    assert conn.execute("SELECT COUNT(*) FROM adsorption_configs").fetchone()[0] == 3


def test_two_rounds_over_the_same_cluster_do_not_collide(conn):
    """The failure this key exists to prevent.

    Config 0 of site 1 in a 16-site sweep and config 0 of site 1 in a 32-site sweep are
    different placements. Keyed on the ordinal they collided: the SELECT found the old row,
    returned its id, and the new geometry — with its own xyz_path — was dropped in silence,
    leaving the new round's results hanging off the old round's configurations.
    """
    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    ronda_v6 = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="CO2", site_atom_idx=1, config_index=0,
        orientation="bidentate_p20_a0", xyz_path="v6/cfg0.xyz",
    )
    ronda_v7 = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="CO2", site_atom_idx=1, config_index=0,
        orientation="side_on_p60_a240", xyz_path="v7/cfg0.xyz",
    )

    assert ronda_v6 != ronda_v7
    paths = {r[0] for r in conn.execute("SELECT xyz_path FROM adsorption_configs")}
    assert paths == {"v6/cfg0.xyz", "v7/cfg0.xyz"}


def test_the_database_refuses_a_duplicate_identity(conn):
    """The unique index makes a collision an error, not a silently wrong row."""
    import sqlite3

    import pytest

    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    conn.execute(
        "INSERT INTO adsorption_configs (id, structure_id, probe, site_atom_idx, orientation)"
        " VALUES ('a', 's1', 'H2', 1, 'eta1_p0_a0')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO adsorption_configs (id, structure_id, probe, site_atom_idx, orientation)"
            " VALUES ('b', 's1', 'H2', 1, 'eta1_p0_a0')")


def test_insert_adsorption_result_links_config(conn):
    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    cfg = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="H2", site_atom_idx=1, config_index=0,
    )
    res = repository.insert_adsorption_result(
        conn, cfg, e_ads_kcal=-12.5, e_int_kcal=-15.0, activated=True,
    )
    row = conn.execute(
        "SELECT config_id, e_ads_kcal, e_int_kcal, activated FROM adsorption_results WHERE id = ?",
        (res,),
    ).fetchone()
    assert row["config_id"] == cfg
    assert row["e_ads_kcal"] == pytest.approx(-12.5)
    assert row["e_int_kcal"] == pytest.approx(-15.0)
    assert row["activated"] == 1


def test_insert_workflow_run(conn):
    run_id = repository.insert_workflow_run(
        conn, command="cadmia stability --mixed x.out --db", agent="stability",
        ids={"stability_id": "abc"},
    )
    row = conn.execute("SELECT * FROM workflow_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["agent"] == "stability"
    assert "stability --mixed" in row["command"]
    assert "abc" in row["ids_json"]
    assert row["status"] == "ok"


@pytest.mark.skipif(not _HAS_DATA, reason="1tial_cdft reference data not present")
def test_cli_cdft_records_workflow_run(tmp_path):
    from cadmia.cli import main
    db = tmp_path / "wf.db"
    result = CliRunner().invoke(main, [
        "cdft", "--n", str(_REAL / "N.out"), "--np1", str(_REAL / "N+1.out"),
        "--nm1", str(_REAL / "N-1.out"), "--cluster", "Al2TiO5",
        "--no-table", "--db", "--db-path", str(db),
    ])
    assert result.exit_code == 0, result.output
    with get_connection(db) as conn:
        row = conn.execute("SELECT agent, status, command, ids_json FROM workflow_runs").fetchone()
    assert row["agent"] == "cdft"
    assert row["status"] == "ok"
    assert "calculation_id" in row["ids_json"]


# ─────────────────────────────────────────────
# The middle of the provenance chain
# ─────────────────────────────────────────────

def test_prescreen_locates_the_config_geometry_created_instead_of_making_a_new_one(conn, tmp_path):
    cfg_xyz = tmp_path / "H2_site1_eta1_p0_a0.xyz"
    cfg_xyz.write_text("1\n\nTi 0 0 0\n", encoding="utf-8")
    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    cfg = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="H2", site_atom_idx=1,
        orientation="eta1_p0_a0", xyz_path=str(cfg_xyz.resolve()),
    )

    found = repository.find_adsorption_config_by_xyz(conn, str(cfg_xyz))
    assert found == cfg
    assert conn.execute("SELECT COUNT(*) FROM adsorption_configs").fetchone()[0] == 1


def test_the_screen_records_the_skeleton_class_and_the_duplicate_it_collapsed_onto(conn):
    repository.upsert_screening_result(
        conn, stage="prescreen", source="a.xyz", energy_hartree=-10.0, kept=True,
        converged=True, cluster_skeleton="reference")
    repository.upsert_screening_result(
        conn, stage="prescreen", source="b.xyz", energy_hartree=-10.0, kept=False,
        converged=True, duplicate_of="a.xyz", cluster_skeleton="isomer-1")

    rows = {r["source"]: r for r in conn.execute("SELECT * FROM screening_results")}
    assert rows["a.xyz"]["cluster_skeleton"] == "reference"
    assert rows["b.xyz"]["duplicate_of"] == "a.xyz"
    assert rows["b.xyz"]["kept"] == 0


def test_rescreening_updates_instead_of_piling_up(conn):
    first = repository.upsert_screening_result(
        conn, stage="prescreen", source="a.xyz", energy_hartree=-10.0, kept=False)
    second = repository.upsert_screening_result(
        conn, stage="prescreen", source="a.xyz", energy_hartree=-10.5, kept=True)

    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM screening_results").fetchone()[0] == 1
    row = conn.execute("SELECT energy_hartree, kept FROM screening_results").fetchone()
    assert row[0] == -10.5 and row[1] == 1


def test_the_two_stages_are_separate_rows_for_the_same_name(conn):
    """A prescreen verdict and a rank verdict on the same stem are different facts."""
    repository.upsert_screening_result(conn, stage="prescreen", source="a.xyz", kept=True)
    repository.upsert_screening_result(conn, stage="rank", source="a.xyz", kept=False)
    assert conn.execute("SELECT COUNT(*) FROM screening_results").fetchone()[0] == 2


def test_the_chain_joins_from_the_starting_geometry_to_the_final_result(conn, tmp_path):
    """The provenance of an adduct is a query, not a glob over filenames."""
    cfg_xyz = tmp_path / "CO2_site3_eta1_c_p20_a60.xyz"
    cfg_xyz.write_text("1\n\nTi 0 0 0\n", encoding="utf-8")
    repository.upsert_structure(conn, structure_id="s1", xyz_text="1\nx\nTi 0 0 0\n")
    cfg = repository.insert_adsorption_config(
        conn, structure_id="s1", probe="CO2", site_atom_idx=3,
        orientation="eta1_c_p20_a60", xyz_path=str(cfg_xyz.resolve()),
    )
    repository.upsert_screening_result(
        conn, stage="prescreen", source=str(cfg_xyz.resolve()), config_id=cfg,
        energy_hartree=-42.0, kept=True, cluster_skeleton="reference")
    repository.insert_adsorption_result(conn, cfg, e_ads_kcal=-18.4)

    row = conn.execute("""
        SELECT c.probe, c.site_atom_idx, c.orientation, s.cluster_skeleton, r.e_ads_kcal
        FROM adsorption_configs c
        JOIN screening_results  s ON s.config_id = c.id AND s.stage = 'prescreen'
        JOIN adsorption_results r ON r.config_id = c.id
        WHERE c.xyz_path = ?""", (str(cfg_xyz.resolve()),)).fetchone()

    assert row["probe"] == "CO2"
    assert row["site_atom_idx"] == 3
    assert row["orientation"] == "eta1_c_p20_a60"
    assert row["cluster_skeleton"] == "reference"
    assert row["e_ads_kcal"] == -18.4


def test_the_screen_still_records_when_geometry_never_persisted(conn):
    """Losing the screen's own verdict because an earlier stage was not persisted
    would defeat the purpose."""
    repository.upsert_screening_result(conn, stage="prescreen", source="orphan.xyz", kept=True)
    row = conn.execute("SELECT config_id, kept FROM screening_results").fetchone()
    assert row[0] is None and row[1] == 1


def test_prescreen_and_rank_write_nothing_without_db(tmp_path):
    """Regression: persistence stays opt-in until the default flips."""
    from cadmia.cli import main
    from cadmia.db.session import init_db

    db = tmp_path / "t.db"
    init_db(db).close()

    out = tmp_path / "ranked.xyz"
    src = tmp_path / "a.out"
    src.write_text("nothing parseable\n", encoding="utf-8")
    CliRunner().invoke(main, ["rank", str(src), "-o", str(out)])

    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM screening_results").fetchone()[0] == 0


def test_rank_db_records_each_output(tmp_path):
    """End to end through the CLI, which is the path that actually runs."""
    from cadmia.cli import main
    from cadmia.db.session import init_db

    db = tmp_path / "t.db"
    init_db(db).close()
    src = tmp_path / "a.out"
    src.write_text("not an output\n", encoding="utf-8")

    res = CliRunner().invoke(main, [
        "rank", str(src), "-o", str(tmp_path / "r.xyz"),
        "--db", "--db-path", str(db),
    ])
    assert res.exit_code == 1, res.output   # nothing could be ranked

    with get_connection(db) as conn:
        rows = conn.execute(
            "SELECT stage, source FROM screening_results").fetchall()
    assert len(rows) == 1
    assert rows[0]["stage"] == "rank"
    assert rows[0]["source"].endswith("a.out")


# ─────────────────────────────────────────────
# Persist by default
# ─────────────────────────────────────────────

def test_a_project_database_wins_over_the_global_one(tmp_path, monkeypatch):
    """What decides whether the database serves a project or becomes a drawer."""
    from cadmia.db import session

    project = tmp_path / "proyecto" / "sub" / "dir"
    project.mkdir(parents=True)
    own = tmp_path / "proyecto" / ".cadmia" / "cadmia.db"
    own.parent.mkdir(parents=True)
    own.write_text("", encoding="utf-8")

    monkeypatch.setattr(session, "DEFAULT_DB", tmp_path / "global.db")
    assert session.resolve_db_path(start=project) == own


def test_without_a_project_database_the_global_one_is_used(tmp_path, monkeypatch):
    from cadmia.db import session

    glob_db = tmp_path / "global.db"
    monkeypatch.setattr(session, "DEFAULT_DB", glob_db)
    lonely = tmp_path / "nada"
    lonely.mkdir()
    assert session.resolve_db_path(start=lonely) == glob_db


def test_a_project_database_is_adopted_only_if_it_already_exists(tmp_path, monkeypatch):
    """Never create one by surprise in whatever directory the user happened to be in."""
    from cadmia.db import session

    glob_db = tmp_path / "global.db"
    monkeypatch.setattr(session, "DEFAULT_DB", glob_db)
    here = tmp_path / "proyecto"
    (here / ".cadmia").mkdir(parents=True)          # the folder exists, the database does not
    assert session.resolve_db_path(start=here) == glob_db


def test_an_explicit_path_beats_everything(tmp_path):
    from cadmia.db import session

    assert session.resolve_db_path(tmp_path / "x.db") == tmp_path / "x.db"


def test_rank_persists_without_being_asked(tmp_path):
    """What used not to be saved unless asked is now saved unless forbidden."""
    from cadmia.cli import main
    from cadmia.db.session import init_db

    db = tmp_path / "t.db"
    init_db(db).close()
    src = tmp_path / "a.out"
    src.write_text("not an output\n", encoding="utf-8")

    res = CliRunner().invoke(main, ["rank", str(src), "-o", str(tmp_path / "r.xyz"),
                                    "--db-path", str(db)])          # no --db flag
    assert res.exit_code == 1, res.output   # nothing could be ranked
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM screening_results").fetchone()[0] == 1


def test_no_db_still_forbids_it(tmp_path):
    from cadmia.cli import main
    from cadmia.db.session import init_db

    db = tmp_path / "t.db"
    init_db(db).close()
    src = tmp_path / "a.out"
    src.write_text("not an output\n", encoding="utf-8")

    res = CliRunner().invoke(main, ["rank", str(src), "-o", str(tmp_path / "r.xyz"),
                                    "--no-db", "--db-path", str(db)])
    assert res.exit_code == 1, res.output   # nothing could be ranked
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM screening_results").fetchone()[0] == 0


def test_creating_a_database_is_announced(tmp_path):
    """A file that appears without warning is the other way of being silent."""
    from cadmia.cli import main

    db = tmp_path / "brand_new.db"
    src = tmp_path / "a.out"
    src.write_text("not an output\n", encoding="utf-8")

    res = CliRunner().invoke(main, ["rank", str(src), "-o", str(tmp_path / "r.xyz"),
                                    "--db-path", str(db)])
    assert res.exit_code == 1, res.output   # nothing could be ranked
    assert "creating" in res.output and "brand_new.db" in res.output
    assert db.exists()
