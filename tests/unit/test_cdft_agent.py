"""Tests for cadmia.agents.cdft_agent (the CADMIA CDFT Agent)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadmia.agents import cdft_agent
from cadmia.agents.cdft_agent import (
    CDFTResult,
    SiteDescriptor,
    _classify,
    _detect_engine,
    _file_hash,
    _r,
)

# Real Gaussian 16 data for Al2TiO5 (8 atoms) — N / N+1 / N-1 SCF outputs.
_DATA = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_cdft"
_HAS_DATA = (_DATA / "N.out").exists() and (_DATA / "N+1.out").exists() and (_DATA / "N-1.out").exists()
_requires_data = pytest.mark.skipif(not _HAS_DATA, reason="1tial_cdft reference data not present")


# ─────────────────────────────────────────────
# _classify
# ─────────────────────────────────────────────

def test_classify_electrophilic():
    assert _classify(0.5) == "electrophilic"

def test_classify_nucleophilic():
    assert _classify(-0.5) == "nucleophilic"

def test_classify_ambiphilic_below_threshold():
    assert _classify(0.001) == "ambiphilic"
    assert _classify(-0.001) == "ambiphilic"

def test_classify_none_is_unknown():
    assert _classify(None) == "unknown"

def test_classify_threshold_boundary():
    # |Δf| exactly at threshold is NOT ambiphilic (strict <)
    assert _classify(0.02) == "electrophilic"
    assert _classify(-0.02) == "nucleophilic"


# ─────────────────────────────────────────────
# _r (rounding helper)
# ─────────────────────────────────────────────

def test_r_rounds():
    assert _r(0.123456789) == 0.123457

def test_r_none():
    assert _r(None) is None

def test_r_custom_digits():
    assert _r(0.123456, 2) == 0.12


# ─────────────────────────────────────────────
# _detect_engine
# ─────────────────────────────────────────────

def test_detect_engine_orca(tmp_path):
    f = tmp_path / "orca.out"
    f.write_text("Some header\nFINAL SINGLE POINT ENERGY   -123.45\nmore\n")
    assert _detect_engine(f) == "orca"

def test_detect_engine_gaussian(tmp_path):
    f = tmp_path / "g.out"
    f.write_text(" Entering Gaussian System, Link 0\n SCF Done: ...\n")
    assert _detect_engine(f) == "gaussian"

def test_detect_engine_missing_defaults_gaussian(tmp_path):
    assert _detect_engine(tmp_path / "nope.out") == "gaussian"

@_requires_data
def test_detect_engine_real_file_is_gaussian():
    assert _detect_engine(_DATA / "N.out") == "gaussian"


# ─────────────────────────────────────────────
# _file_hash
# ─────────────────────────────────────────────

def test_file_hash_deterministic(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    h1 = _file_hash(a)
    h2 = _file_hash(a)
    assert h1 == h2
    assert len(h1) == 16

def test_file_hash_differs_by_content(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("hello")
    b = tmp_path / "b.txt"; b.write_text("world")
    assert _file_hash(a) != _file_hash(b)


# ─────────────────────────────────────────────
# run() — integration with real data
# ─────────────────────────────────────────────

@pytest.fixture
def real_result() -> CDFTResult:
    return cdft_agent.run(
        path_n=str(_DATA / "N.out"),
        path_np1=str(_DATA / "N+1.out"),
        path_nm1=str(_DATA / "N-1.out"),
        cluster="Al2TiO5",
    )


@_requires_data
def test_run_keeps_workflow_warnings_in_the_json(monkeypatch):
    import warnings
    original = cdft_agent.run_cdft_workflow

    def warning_workflow(*args, **kwargs):
        warnings.warn("Electron affinity is -0.100 eV", RuntimeWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(cdft_agent, "run_cdft_workflow", warning_workflow)
    result = cdft_agent.run(path_n=str(_DATA / "N.out"), path_np1=str(_DATA / "N+1.out"),
                            path_nm1=str(_DATA / "N-1.out"), cluster="Al2TiO5")
    assert json.loads(result.to_json())["warnings"] == ["Electron affinity is -0.100 eV"]


@_requires_data
def test_run_rejects_swapped_anion_and_cation():
    with pytest.raises(ValueError, match="N\\+1: total charge \\+1, expected -1"):
        cdft_agent.run(path_n=str(_DATA / "N.out"), path_np1=str(_DATA / "N-1.out"),
                       path_nm1=str(_DATA / "N+1.out"), cluster="Al2TiO5")


@_requires_data
def test_run_basic_shape(real_result):
    assert real_result.cluster == "Al2TiO5"
    assert real_result.charge_scheme == "mulliken"
    assert len(real_result.calculation_id) == 16
    assert len(real_result.sites) == 8


@_requires_data
def test_run_ranks_are_contiguous(real_result):
    ranks = [s.rank for s in real_result.sites]
    assert sorted(ranks) == list(range(1, 9))


@_requires_data
def test_run_sites_sorted_by_abs_delta_f(real_result):
    deltas = [abs(s.delta_f) for s in real_result.sites if s.delta_f is not None]
    assert deltas == sorted(deltas, reverse=True)


@_requires_data
def test_run_atom_index_is_1_based(real_result):
    indices = [s.atom_index for s in real_result.sites]
    assert min(indices) == 1
    assert max(indices) == 8


@_requires_data
def test_run_top_site_matches_reference(real_result):
    """Top-ranked site is the Al(5) electrophilic center (verified against Multiwfn)."""
    top = real_result.sites[0]
    assert top.atom_index == 5
    assert top.atom_symbol == "Al"
    assert top.character == "electrophilic"


@_requires_data
def test_run_acid_base_classification(real_result):
    """Top sites are labeled as Lewis acid (Al) / Lewis base (O), consistent with character."""
    top = real_result.sites[0]
    assert top.acid_base == "lewis_acid"   # Al(5) electrophilic center accepts electrons
    # every site's acid/base label must agree with its electrophilic/nucleophilic character
    _map = {"electrophilic": "lewis_acid", "nucleophilic": "lewis_base", "ambiphilic": "amphoteric"}
    for s in real_result.sites:
        assert s.acid_base == _map.get(s.character, s.acid_base)


@_requires_data
def test_run_acid_base_summary(real_result):
    ab = real_result.acid_base
    # strongest Lewis acid is a metal cation (Al), strongest base is oxygen — correct for an oxide
    assert ab["strongest_acid_site"]["atom_symbol"] == "Al"
    assert ab["strongest_base_site"]["atom_symbol"] == "O"
    assert ab["n_acid_sites"] >= 1 and ab["n_base_sites"] >= 1
    assert ab["hardness_eV"] == pytest.approx(real_result.global_descriptors["hardness_eV"])
    assert "HSAB" in ab["hsab_note"]


@_requires_data
def test_run_global_descriptors(real_result):
    gd = real_result.global_descriptors
    assert gd["gap_eV"] == pytest.approx(4.57, abs=0.05)
    assert gd["homo_eV"] < gd["lumo_eV"]
    assert gd["hardness_eV"] > 0


@_requires_data
def test_run_softness_key_is_labeled_ev_inv(real_result):
    """Softness is published as S = 1/η in eV^-1, under an honest key.

    The old key `softness_Ha` claimed Hartree^-1 while carrying the eV^-1 value (a factor of
    ~27.21 off its label): for TiAl it published 0.125 where a reader of "_Ha" expects the
    Multiwfn value 3.402.
    """
    gd = real_result.global_descriptors
    assert "softness_Ha" not in gd
    assert gd["softness_eV_inv"] == pytest.approx(1.0 / gd["hardness_eV"], abs=1e-4)
    assert gd["softness_eV_inv"] == pytest.approx(0.125, abs=0.005)   # eV^-1, not 3.402 Ha^-1


@_requires_data
def test_run_exports_electrodonating_accepting_powers(real_result):
    """ω⁻ and ω⁺ (GCV) are computed anyway; they must reach the global JSON."""
    gd = real_result.global_descriptors
    assert "omega_minus_eV" in gd and "omega_plus_eV" in gd
    # ω⁺ − ω⁻ = μ (electronic chemical potential) — the GCV identity
    assert gd["omega_plus_eV"] - gd["omega_minus_eV"] == pytest.approx(gd["mu_eV"], abs=1e-2)


@_requires_data
def test_run_sources_recorded(real_result):
    assert real_result.sources["N"].endswith("N.out")
    assert real_result.sources["N+1"].endswith("N+1.out")
    assert real_result.sources["N-1"].endswith("N-1.out")


@_requires_data
def test_run_explicit_engine(real_result):
    forced = cdft_agent.run(
        path_n=str(_DATA / "N.out"), path_np1=str(_DATA / "N+1.out"),
        path_nm1=str(_DATA / "N-1.out"), engine="gaussian",
    )
    assert len(forced.sites) == 8


def test_run_rejects_a_non_native_output_instead_of_silently_misparsing_it(tmp_path):
    """CDFT needs a native parser: without an explicit `engine`, a file that is neither ORCA
    nor Gaussian must fail with a clear message instead of silently falling back to Gaussian
    (the _detect_engine default meant for structure_from_file) and misparsing it.
    """
    other = tmp_path / "other_program.out"
    other.write_text("Some other quantum chemistry program\n Total energy = -193.1\n")
    with pytest.raises(ValueError, match="native parser"):
        cdft_agent.run(path_n=str(other), path_np1=str(other), path_nm1=str(other))


# ─────────────────────────────────────────────
# to_dict / to_json
# ─────────────────────────────────────────────

@_requires_data
def test_to_json_is_valid_and_complete(real_result):
    data = json.loads(real_result.to_json())
    assert data["cluster"] == "Al2TiO5"
    assert len(data["sites"]) == 8
    site = data["sites"][0]
    assert set(site) >= {"atom_index", "atom_symbol", "coordinates_ang",
                         "fukui_plus", "fukui_minus", "delta_f", "character", "rank"}
    assert len(site["coordinates_ang"]) == 3


@_requires_data
def test_to_dict_matches_to_json(real_result):
    assert real_result.to_dict() == json.loads(real_result.to_json())


