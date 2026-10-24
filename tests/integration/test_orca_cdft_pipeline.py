"""Integration scaffold for the ORCA → CDFT pipeline (activates with real data).

This test exercises the *untested* link of CADMIA: running the CDFT Agent on real
ORCA output files. It stays skipped until the three ORCA SCF outputs are present.

To activate: drop the converged ORCA outputs for the Al2TiO5 system into
``1tial_cdft_orca/`` (sibling of ``1tial_cdft/``):

    1tial_cdft_orca/
        N.out      neutral  (charge 0,  closed shell)
        N+1.out    anion    (charge -1, open shell)
        N-1.out    cation   (charge +1, open shell)

Make sure each ORCA input printed atomic charges (Hirshfeld preferred:
``%output Print[P_Hirshfeld] 1 end``; Mulliken is the fallback).

The charge-sum invariants below are the key diagnostic: if the parser picks up the
spin-population column instead of the charge column on the open-shell ions (a known
risk for ORCA open-shell output), the sums will not match the molecular charge and
these tests fail loudly — turning a silent mis-parse into an obvious red flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_ORCA = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_orca"
_N    = _ORCA / "1tial_N_orca.out"
_NP1  = _ORCA / "1tial_anion_orca.out"
_NM1  = _ORCA / "1tial_cation_orca.out"

_HAS_ORCA = _N.exists() and _NP1.exists() and _NM1.exists()
_requires_orca = pytest.mark.skipif(
    not _HAS_ORCA,
    reason=f"Real ORCA outputs not present in {_ORCA}",
)

# Al2TiO5 reference system (1 Ti, 2 Al, 5 O = 8 atoms)
_N_ATOMS = 8
# Expected total molecular charge per state (used for the charge-sum invariant)
_EXPECTED_CHARGE = {_N: 0, _NP1: -1, _NM1: +1}


# ─────────────────────────────────────────────
# Engine detection
# ─────────────────────────────────────────────

@_requires_orca
@pytest.mark.parametrize("path", [_N, _NP1, _NM1])
def test_engine_detected_as_orca(path):
    from cadmia.agents.cdft_agent import _detect_engine
    assert _detect_engine(path) == "orca"


# ─────────────────────────────────────────────
# Charge-sum invariant (the key diagnostic)
# ─────────────────────────────────────────────

@_requires_orca
@pytest.mark.parametrize("path", [_N, _NP1, _NM1])
def test_charge_sum_matches_molecular_charge(path):
    """sum(atomic charges) must equal the molecular charge (±0.05).

    Catches the open-shell mis-parse where the spin-population column is read
    instead of the charge column.
    """
    from cadmia.io.orca import parse_orca_geometry
    from cadmia.io.electronic import parse_orca_electronic

    structure = parse_orca_geometry(path)
    state = parse_orca_electronic(path, structure)
    if not state.charges:
        pytest.skip(f"{path.name}: no atomic charges in output (add charge printing).")
    total = float(np.sum(state.charges))
    assert total == pytest.approx(_EXPECTED_CHARGE[path], abs=0.05), (
        f"{path.name}: sum(charges)={total:.4f} but molecular charge is "
        f"{_EXPECTED_CHARGE[path]}; scheme={state.charge_scheme}. "
        f"Likely the spin-population column was parsed instead of the charge."
    )


# ─────────────────────────────────────────────
# Full CDFT Agent pipeline on ORCA
# ─────────────────────────────────────────────

@pytest.fixture
def orca_result():
    from cadmia.agents import cdft_agent
    return cdft_agent.run(str(_N), str(_NP1), str(_NM1),
                          cluster="Al2TiO5", engine="orca")


@_requires_orca
def test_cdft_pipeline_shape(orca_result):
    assert len(orca_result.sites) == _N_ATOMS
    assert orca_result.charge_scheme in ("hirshfeld", "mulliken")
    ranks = sorted(s.rank for s in orca_result.sites)
    assert ranks == list(range(1, _N_ATOMS + 1))


@_requires_orca
def test_cdft_pipeline_global_descriptors(orca_result):
    gd = orca_result.global_descriptors
    assert gd["gap_eV"] is not None and gd["gap_eV"] > 0
    assert gd["homo_eV"] < gd["lumo_eV"]
    assert gd["hardness_eV"] > 0


@_requires_orca
def test_cdft_pipeline_has_fukui_when_charges_present(orca_result):
    """With charges in all three states, condensed Fukui must be computed."""
    deltas = [s.delta_f for s in orca_result.sites]
    if all(d is None for d in deltas):
        pytest.skip("No Fukui computed — charges missing in at least one ORCA state.")
    assert any(s.character == "electrophilic" for s in orca_result.sites)
    assert any(s.character == "nucleophilic" for s in orca_result.sites)


@_requires_orca
def test_cdft_pipeline_persists_to_db(orca_result, tmp_path):
    """The ORCA result flows through the DB persistence path like the Gaussian one."""
    from cadmia.db.session import init_db, get_connection
    from cadmia.db import repository
    db = tmp_path / "orca.db"
    conn = init_db(db)
    try:
        ids = repository.persist_cdft_result(conn, orca_result, engine="orca", output_path=str(_N))
        conn.commit()
    finally:
        conn.close()
    assert len(ids["descriptor_ids"]) == _N_ATOMS
    with get_connection(db) as c:
        assert c.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1


# ─────────────────────────────────────────────
# Validation Agent on the ORCA neutral
# ─────────────────────────────────────────────

@_requires_orca
def test_validation_on_orca_neutral():
    from cadmia.agents import validation_agent
    report = validation_agent.run(filepath=str(_N))
    assert report.engine == "orca"
    assert report.checks["termination"].status in ("pass", "fail")
    # A converged SCF must not report a failed convergence check
    assert report.checks["scf_convergence"].status != "fail"
