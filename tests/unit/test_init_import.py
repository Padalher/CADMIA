"""Tests for `cadmia init` (project scaffolding) and `cadmia import` (output ingestion)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.project import init_project
from cadmia.agents import import_agent

FIXTURES = Path(__file__).parent.parent / "fixtures"
_GAUSSIAN = FIXTURES / "1tial_cdft" / "N.out"
_ORCA = FIXTURES / "1tial_orca" / "1tial_N_orca.out"
_HAS_GAUSSIAN = _GAUSSIAN.exists()
_HAS_ORCA = _ORCA.exists()


# ─────────────────────────────────────────────
# init
# ─────────────────────────────────────────────

def test_init_creates_layout(tmp_path):
    summary = init_project(tmp_path / "proj", name="demo")
    root = Path(summary["root"])
    for sub in ("configs", "results", "reports"):
        assert (root / sub).is_dir()
    assert (root / "cadmia_project.yaml").exists()
    assert (root / ".cadmia" / "cadmia.db").exists()
    assert summary["name"] == "demo"


def test_init_db_is_the_one_commands_find(tmp_path):
    from cadmia.db.session import resolve_db_path
    summary = init_project(tmp_path / "proj")
    inside = Path(summary["root"]) / "configs"
    assert resolve_db_path(start=inside) == Path(summary["db"])


def test_init_manifest_imposes_no_level_of_theory(tmp_path):
    init_project(tmp_path / "p", name="x")
    text = (tmp_path / "p" / "cadmia_project.yaml").read_text(encoding="utf-8")
    assert "name: x" in text and "PBE0" not in text and "probe" not in text


def test_init_no_db(tmp_path):
    summary = init_project(tmp_path / "p", with_db=False)
    assert summary["db"] is None
    assert not (tmp_path / "p" / ".cadmia" / "cadmia.db").exists()


def test_init_is_idempotent(tmp_path):
    init_project(tmp_path / "p", name="x")
    summary = init_project(tmp_path / "p", name="x")        # second run keeps existing
    assert "cadmia_project.yaml" in summary["skipped"]
    assert "configs/" in summary["skipped"]


def test_init_name_defaults_to_dirname(tmp_path):
    summary = init_project(tmp_path / "MyStudy")
    assert summary["name"] == "MyStudy"


# ─────────────────────────────────────────────
# import
# ─────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_GAUSSIAN, reason="Gaussian reference output not present")
def test_import_gaussian_persists(tmp_path):
    db = tmp_path / "cadmia.db"
    result = import_agent.import_outputs([_GAUSSIAN], db_path=db,
                                         functional="PBE0", basis="def2-TZVP")
    assert result.n_ok == 1
    item = result.items[0]
    assert item.engine == "gaussian"
    assert item.energy_au is not None
    assert item.structure_id and item.calculation_id

    import sqlite3
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM calculations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0] == 1
    eng, func = conn.execute("SELECT engine, functional FROM calculations").fetchone()
    assert eng == "gaussian" and func == "PBE0"
    conn.close()


@pytest.mark.skipif(not _HAS_ORCA, reason="ORCA reference output not present")
def test_import_orca_detected(tmp_path):
    result = import_agent.import_outputs([_ORCA], db_path=tmp_path / "db.sqlite")
    assert result.n_ok == 1
    assert result.items[0].engine == "orca"


@pytest.mark.skipif(not _HAS_GAUSSIAN, reason="Gaussian reference output not present")
def test_import_is_idempotent_on_geometry(tmp_path):
    db = tmp_path / "cadmia.db"
    import_agent.import_outputs([_GAUSSIAN], db_path=db)
    import_agent.import_outputs([_GAUSSIAN], db_path=db)   # same geometry → same structure id
    import sqlite3
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0] == 1
    conn.close()


def test_import_bad_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / "garbage.out"
    bad.write_text("this is not a calculation output\n")
    result = import_agent.import_outputs([bad], db_path=tmp_path / "db.sqlite")
    assert result.n_ok == 0
    assert result.items[0].error is not None
    assert len(result.warnings) == 1
