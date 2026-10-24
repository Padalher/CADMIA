"""Tests for automatic spin multiplicity (cadmia.core.spin) and its wiring into writers."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.core.spin import (electron_count, multiplicity_from_electrons,
                              transition_metals_present)
from cadmia.calculators.params import CalcParams, resolve_multiplicity
from cadmia.calculators.gaussian import write_gjf
from cadmia.calculators.orca import write_inp


def test_electron_count_neutral_and_charged():
    # H2O: 2·1 + 8 = 10 electrons
    assert electron_count(["O", "H", "H"], 0) == 10
    assert electron_count(["O", "H", "H"], -1) == 11   # anion adds one electron
    assert electron_count(["O", "H", "H"], +1) == 9    # cation removes one


def test_multiplicity_parity():
    assert multiplicity_from_electrons(["O", "H", "H"], 0) == 1   # 10 e- → singlet
    assert multiplicity_from_electrons(["O", "H", "H"], -1) == 2  # 11 e- → doublet
    assert multiplicity_from_electrons(["O", "H", "H"], +1) == 2  # 9 e-  → doublet


def test_unknown_element_raises():
    with pytest.raises(ValueError):
        electron_count(["Xx"], 0)


def test_transition_metals_flagged():
    assert transition_metals_present(["Ti", "O", "O"]) == ["Ti"]
    assert transition_metals_present(["Al", "O", "O"]) == []


def test_resolve_multiplicity_manual_vs_auto():
    atoms = ["O", "H", "H"]
    manual = CalcParams(multiplicity=3)
    assert resolve_multiplicity(manual, atoms) == 3      # manual wins
    auto = CalcParams(auto_multiplicity=True, charge=-1)
    assert resolve_multiplicity(auto, atoms) == 2        # odd electrons → doublet


def test_writers_honour_auto_multiplicity(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    # cation (charge +1) → 9 electrons → doublet, even though multiplicity field is 1
    params = CalcParams(task="sp", charge=1, auto_multiplicity=True, dispersion="d3bj")
    inp = write_inp(atoms, pos, params, tmp_path / "c.inp")
    assert "* xyz 1 2" in inp.read_text()
    gjf = write_gjf(atoms, pos, params, tmp_path / "c.gjf")
    assert "1 2" in gjf.read_text()


def test_writers_manual_multiplicity_unchanged(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    params = CalcParams(task="sp", charge=0, multiplicity=1)   # default: not auto
    inp = write_inp(atoms, pos, params, tmp_path / "c.inp")
    assert "* xyz 0 1" in inp.read_text()


def test_scf_maxiter_emitted_when_set(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    params = CalcParams(task="sp", scf_maxiter=512, dispersion="d3bj")
    inp = write_inp(atoms, pos, params, tmp_path / "c.inp").read_text()
    assert "%scf" in inp and "MaxIter 512" in inp
    gjf = write_gjf(atoms, pos, params, tmp_path / "c.gjf").read_text()
    assert "SCF=(MaxCycle=512)" in gjf


def test_scf_maxiter_absent_by_default(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    params = CalcParams(task="sp", dispersion="d3bj")   # scf_maxiter defaults to 0
    assert "%scf" not in write_inp(atoms, pos, params, tmp_path / "c.inp").read_text()
    assert "MaxCycle" not in write_gjf(atoms, pos, params, tmp_path / "c.gjf").read_text()


def test_geom_maxiter_emitted_only_for_optimizations(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    for task, expected in (("opt", True), ("opt+freq", True), ("freq", False), ("sp", False)):
        params = CalcParams(task=task, geom_maxiter=200, dispersion="d3bj")
        inp = write_inp(atoms, pos, params, tmp_path / "c.inp").read_text()
        assert ("%geom" in inp and "MaxIter 200" in inp) is expected, task


def test_geom_maxiter_goes_inside_gaussians_opt_keyword(tmp_path):
    """Gaussian has no %geom block: a second bare Opt would be a syntax error, and Opt=Tight
    has to absorb the limit rather than be replaced by it."""
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    def route(task):
        params = CalcParams(task=task, geom_maxiter=200, dispersion="d3bj")
        return write_gjf(atoms, pos, params, tmp_path / "c.gjf").read_text()
    assert "Opt=(MaxCycles=200)" in route("opt")
    assert "Opt=(Tight,MaxCycles=200) Freq" in route("opt_tight+freq")
    assert "Opt" not in route("sp")


def test_geom_maxiter_absent_by_default(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    params = CalcParams(task="opt", dispersion="d3bj")   # geom_maxiter defaults to 0
    assert "%geom" not in write_inp(atoms, pos, params, tmp_path / "c.inp").read_text()
    assert "MaxCycles" not in write_gjf(atoms, pos, params, tmp_path / "c.gjf").read_text()


def test_chelpg_fine_block_only_with_chelpg_keyword(tmp_path):
    atoms = ["O", "H", "H"]
    pos = np.zeros((3, 3))
    # requested → block appears
    p_yes = CalcParams(task="sp", extra_orca="CHELPG", chelpg_fine=True)
    inp = write_inp(atoms, pos, p_yes, tmp_path / "y.inp").read_text()
    assert "%chelpg" in inp and "GRID 0.20" in inp and "VDWRADII COSMO" in inp
    # flag on but no CHELPG keyword → no block
    p_no = CalcParams(task="sp", chelpg_fine=True)
    assert "%chelpg" not in write_inp(atoms, pos, p_no, tmp_path / "n.inp").read_text()
