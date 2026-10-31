"""Tests for cadmia.io.mayer (ORCA Mayer population/bond-order parser) and persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.io.mayer import parse_mayer

FIXTURES = Path(__file__).parent.parent / "fixtures"
_ORCA = FIXTURES / "1tial_orca" / "1tial_N_orca.out"
_GAUSSIAN = FIXTURES / "1tial_cdft" / "N.out"
_HAS_ORCA = _ORCA.exists()
_HAS_GAUSSIAN = _GAUSSIAN.exists()

_SAMPLE = """\
                      * MAYER POPULATION ANALYSIS *
                      *****************************

  NA   - Mulliken gross atomic population
  VA   - Mayer's total valence

  ATOM       NA         ZA         QA         VA         BVA        FA
  0 Ti    20.6794    22.0000     1.3206     4.1625     4.1625    -0.0000
  1 O      8.5637     8.0000    -0.5637     2.2410     2.2410     0.0000
  2 O      8.5945     8.0000    -0.5945     2.1223     2.1223    -0.0000

  Mayer bond orders larger than 0.100000
B(  0-Ti,  1-O ) :   2.0961 B(  0-Ti,  2-O ) :   0.7230
B(  1-O ,  2-O ) :   0.1500


------------------
HIRSHFELD ANALYSIS
------------------
"""


# ─────────────────────────────────────────────
# Parser (always-on synthetic block)
# ─────────────────────────────────────────────

def test_parse_sample_atoms(tmp_path):
    p = tmp_path / "s.out"
    p.write_text(_SAMPLE)
    r = parse_mayer(p)
    assert len(r.atoms) == 3
    ti = r.atoms[0]
    assert ti.index == 0 and ti.symbol == "Ti"
    assert ti.qa == pytest.approx(1.3206)
    assert ti.va == pytest.approx(4.1625)


def test_parse_sample_bonds_multi_per_line(tmp_path):
    p = tmp_path / "s.out"
    p.write_text(_SAMPLE)
    r = parse_mayer(p)
    assert r.threshold == pytest.approx(0.1)
    assert len(r.bonds) == 3                      # two on line 1, one on line 2
    b0 = r.bonds[0]
    assert (b0.atom_i, b0.symbol_i, b0.atom_j, b0.symbol_j) == (0, "Ti", 1, "O")
    assert b0.order == pytest.approx(2.0961)


def test_bonds_stop_at_blank(tmp_path):
    p = tmp_path / "s.out"
    p.write_text(_SAMPLE)
    r = parse_mayer(p)
    # The HIRSHFELD section must not leak into the bond list.
    assert all(b.symbol_i and b.symbol_j for b in r.bonds)


def test_no_mayer_block_returns_none(tmp_path):
    p = tmp_path / "plain.out"
    p.write_text("Some calculation output without a Mayer section.\n")
    assert parse_mayer(p) is None


# ─────────────────────────────────────────────
# Real ORCA fixtures
# ─────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_ORCA, reason="ORCA reference output not present")
def test_parse_real_orca():
    r = parse_mayer(_ORCA)
    assert len(r.atoms) == 8
    assert len(r.bonds) == 13
    assert r.threshold == pytest.approx(0.1)
    strongest = max(r.bonds, key=lambda b: b.order)
    assert {strongest.symbol_i, strongest.symbol_j} == {"Ti", "O"}
    assert strongest.order > 2.0


@pytest.mark.skipif(not _HAS_GAUSSIAN, reason="Gaussian reference output not present")
def test_gaussian_has_no_orca_mayer_block():
    assert parse_mayer(_GAUSSIAN) is None


# ─────────────────────────────────────────────
# Persistence + import hook
# ─────────────────────────────────────────────

def test_persist_bond_descriptors(tmp_path):
    import sqlite3
    from cadmia.db.session import init_db
    from cadmia.db import repository
    from cadmia.io.mayer import parse_mayer as _pm

    sample = tmp_path / "s.out"
    sample.write_text(_SAMPLE)
    bonds = _pm(sample).bonds
    db = tmp_path / "db.sqlite"
    conn = init_db(db)
    # A calculation row is required by the FK.
    ids = repository.persist_imported_calculation(
        conn, atoms=["Ti", "O", "O"], positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
        name="t", engine="orca",
    )
    inserted = repository.persist_bond_descriptors(
        conn, ids["calculation_id"], bonds, positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
    )
    conn.commit()
    assert len(inserted) == 3
    rows = conn.execute(
        "SELECT bond_order_type, distance_A FROM bond_descriptors ORDER BY bond_order DESC"
    ).fetchall()
    assert rows[0][0] == "mayer"
    assert rows[0][1] == pytest.approx(1.6, abs=1e-6)   # 0-Ti–1-O distance
    conn.close()


@pytest.mark.skipif(not _HAS_ORCA, reason="ORCA reference output not present")
def test_import_persists_mayer_bonds(tmp_path):
    import sqlite3
    from cadmia.agents import import_agent

    db = tmp_path / "db.sqlite"
    result = import_agent.import_outputs([_ORCA], db_path=db)
    assert result.items[0].n_bonds == 13
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM bond_descriptors").fetchone()[0] == 13
    conn.close()


# ─────────────────────────────────────────────
# CLI: -o decides the format by extension
# ─────────────────────────────────────────────

def test_bonds_cli_output_json_by_extension(tmp_path):
    import json

    from click.testing import CliRunner

    from cadmia.cli import main

    out = tmp_path / "s.out"
    out.write_text(_SAMPLE)
    jpath = tmp_path / "enlaces.json"
    res = CliRunner().invoke(main, ["bonds", str(out), "--top", "2", "-o", str(jpath)])

    assert res.exit_code == 0, res.output
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert len(data["bonds"]) == 3        # --top caps the screen table, never the data
    assert "JSON → " in res.output


def test_bonds_cli_output_text_by_extension(tmp_path):
    from click.testing import CliRunner

    from cadmia.cli import main

    out = tmp_path / "s.out"
    out.write_text(_SAMPLE)
    tpath = tmp_path / "enlaces.txt"
    res = CliRunner().invoke(main, ["bonds", str(out), "-o", str(tpath)])

    assert res.exit_code == 0, res.output
    text = tpath.read_text(encoding="utf-8")
    assert "Mayer bond orders" in text and "3 bonds, 3 atoms." in text
