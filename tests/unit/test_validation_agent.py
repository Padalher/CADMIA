"""Tests for cadmia.agents.validation_agent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadmia.agents import validation_agent
from cadmia.agents.validation_agent import (
    CheckResult,
    ValidationReport,
    _check_bsse,
    _check_frequencies,
    _check_scf_convergence,
    _check_spin_contamination,
    _check_termination,
    _evaluate_spin,
    _parse_multiplicity,
)

_DATA = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_cdft"
_HAS_DATA = (_DATA / "N.out").exists()
_requires_data = pytest.mark.skipif(not _HAS_DATA, reason="1tial_cdft reference data not present")


# ─────────────────────────────────────────────
# _check_termination
# ─────────────────────────────────────────────

def test_termination_gaussian_normal():
    r = _check_termination([" Normal termination of Gaussian 16"], "gaussian")
    assert r.status == "pass"

def test_termination_gaussian_error():
    r = _check_termination([" Error termination via Lnk1e"], "gaussian")
    assert r.status == "fail"

def test_termination_gaussian_missing():
    r = _check_termination(["random line"], "gaussian")
    assert r.status == "fail"

def test_termination_orca_normal():
    r = _check_termination(["****ORCA TERMINATED NORMALLY****"], "orca")
    assert r.status == "pass"

def test_termination_orca_error():
    r = _check_termination(["****ORCA TERMINATED WITH ERROR****"], "orca")
    assert r.status == "fail"


# ─────────────────────────────────────────────
# _check_scf_convergence
# ─────────────────────────────────────────────

def test_scf_gaussian_converged():
    r = _check_scf_convergence([" SCF Done:  E(RPBE1PBE) = -123.4 A.U."], "gaussian")
    assert r.status == "pass"

def test_scf_gaussian_not_converged():
    r = _check_scf_convergence([" Convergence criterion not met."], "gaussian")
    assert r.status == "fail"

def test_scf_orca_converged():
    r = _check_scf_convergence(["FINAL SINGLE POINT ENERGY -123.4"], "orca")
    assert r.status == "pass"

def test_scf_orca_not_converged():
    r = _check_scf_convergence(["SCF NOT CONVERGED AFTER 125 CYCLES"], "orca")
    assert r.status == "fail"


# ─────────────────────────────────────────────
# _parse_multiplicity & spin
# ─────────────────────────────────────────────

def test_parse_multiplicity():
    assert _parse_multiplicity([" Charge =  0 Multiplicity = 3"]) == 3

def test_parse_multiplicity_absent():
    assert _parse_multiplicity(["no mult here"]) is None

def test_spin_singlet_clean():
    # multiplicity 1, no S**2 block → closed-shell pass
    r = _check_spin_contamination([" Multiplicity = 1", "SCF Done"], "gaussian")
    assert r.status == "pass"

def test_spin_skip_without_multiplicity():
    r = _check_spin_contamination(["no info"], "gaussian")
    assert r.status == "skip"

def test_spin_gaussian_contaminated_doublet():
    lines = [
        " Multiplicity = 2",
        " S**2 before annihilation     1.5000,   after     1.2000",
    ]
    r = _check_spin_contamination(lines, "gaussian")
    # expected S² for doublet = 0.75; 1.50 is >25% deviation → fail
    assert r.status == "fail"


def test_spin_gaussian_reads_the_value_before_annihilation():
    lines = [
        " Multiplicity = 2",
        " S**2 before annihilation     1.1000,   after     0.7600",
    ]
    assert _check_spin_contamination(lines, "gaussian").status == "fail"

def test_evaluate_spin_singlet_pass():
    assert _evaluate_spin(0.0, 0.0, 1).status == "pass"

def test_evaluate_spin_singlet_contaminated():
    assert _evaluate_spin(0.05, 0.0, 1).status == "warning"

def test_evaluate_spin_doublet_clean():
    assert _evaluate_spin(0.76, 0.75, 2).status == "pass"

def test_evaluate_spin_doublet_warning():
    # ~15% deviation
    assert _evaluate_spin(0.86, 0.75, 2).status == "warning"

def test_evaluate_spin_doublet_fail():
    # ~33% deviation
    assert _evaluate_spin(1.0, 0.75, 2).status == "fail"


# ─────────────────────────────────────────────
# _check_frequencies
# ─────────────────────────────────────────────

def test_frequencies_all_positive():
    r = _check_frequencies([" Frequencies --   100.0  200.0  300.0"], "gaussian")
    assert r.status == "pass"

def test_frequencies_imaginary():
    r = _check_frequencies([" Frequencies --   -50.0  200.0  300.0"], "gaussian")
    assert r.status == "fail"
    assert "imaginary" in r.message

def test_frequencies_none_is_skip():
    r = _check_frequencies(["no freq data"], "gaussian")
    assert r.status == "skip"

def test_frequencies_orca_positive():
    r = _check_frequencies(["   6:    145.23 cm**-1", "   7:    200.45 cm**-1"], "orca")
    assert r.status == "pass"


# ─────────────────────────────────────────────
# _check_bsse
# ─────────────────────────────────────────────

def test_bsse_present_gaussian():
    r = _check_bsse([" Counterpoise: corrected energy ="], "gaussian")
    assert r.status == "pass"

def test_bsse_absent_gaussian():
    r = _check_bsse(["no correction here"], "gaussian")
    assert r.status == "fail"

def test_bsse_present_orca():
    r = _check_bsse([" BSSE correction applied"], "orca")
    assert r.status == "pass"


# ─────────────────────────────────────────────
# ValidationReport serialisation
# ─────────────────────────────────────────────

def test_report_to_dict_and_json():
    report = ValidationReport(
        filepath="/tmp/x.out", engine="gaussian", overall="pass",
        checks={"termination": CheckResult("pass", "ok")},
    )
    d = report.to_dict()
    assert d["overall"] == "pass"
    assert d["checks"]["termination"]["status"] == "pass"
    assert json.loads(report.to_json()) == d

def test_report_passed_property():
    assert ValidationReport("f", "orca", "pass").passed is True
    assert ValidationReport("f", "orca", "warning").passed is False


# ─────────────────────────────────────────────
# run() — integration with real data
# ─────────────────────────────────────────────

@_requires_data
def test_run_real_file_passes():
    report = validation_agent.run(filepath=str(_DATA / "N.out"))
    assert report.engine == "gaussian"
    assert report.overall == "pass"
    assert report.checks["termination"].status == "pass"
    assert report.checks["scf_convergence"].status == "pass"
    assert report.checks["geometry_sanity"].status == "pass"

@_requires_data
def test_run_real_file_frequencies_skip():
    # N.out is an SCF (no freq) job → frequencies check is skipped
    report = validation_agent.run(filepath=str(_DATA / "N.out"))
    assert report.checks["frequencies"].status == "skip"

@_requires_data
def test_run_bsse_requested_fails_without_counterpoise():
    report = validation_agent.run(filepath=str(_DATA / "N.out"), check_bsse=True)
    assert report.checks["bsse"].status == "fail"
    assert report.overall == "fail"

@_requires_data
def test_run_bsse_not_requested_is_skip():
    report = validation_agent.run(filepath=str(_DATA / "N.out"), check_bsse=False)
    assert report.checks["bsse"].status == "skip"


def test_gaussian_freq_job_is_not_taken_for_an_unconverged_optimization(tmp_path):
    """A Freq job prints 'Berny optimization' and 'Step number 1' for its stationarity test."""
    freq = tmp_path / "freq.out"
    freq.write_text(" ------------------\n #P PBE1PBE Def2TZVP FREQ Geom=AllCheck\n ------------------\n"
                    " Berny optimization.\n Step number   1 out of a maximum of    2\n")
    assert validation_agent.geometry_converged(freq, "gaussian") is None
    opt = tmp_path / "opt.out"          # route wrapped in the middle of "Opt"
    opt.write_text(" ------------------\n #P PBE1PBE Def2TZVP Op\n t FREQ\n ------------------\n"
                   " Berny optimization.\n Step number   1 out of a maximum of  100\n")
    assert validation_agent.geometry_converged(opt, "gaussian") is False
