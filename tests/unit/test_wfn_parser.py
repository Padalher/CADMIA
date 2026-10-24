"""Tests for Gaussian WFN file parser using real 1tial reference files."""
import pytest
from pathlib import Path

from cadmia.io.wfn import parse_wfn

BASE = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(scope="module")
def wfn_n():
    return parse_wfn(BASE / "N.wfn")


@pytest.fixture(scope="module")
def wfn_np1():
    return parse_wfn(BASE / "N+1.wfn")


@pytest.fixture(scope="module")
def wfn_nm1():
    return parse_wfn(BASE / "N-1.wfn")


# --- Total energies (verified against CDFT.txt) ---

def test_energy_n(wfn_n):
    assert wfn_n.total_energy == pytest.approx(-1710.163301887937, rel=1e-10)


def test_energy_np1(wfn_np1):
    assert wfn_np1.total_energy == pytest.approx(-1710.249267368876, rel=1e-10)


def test_energy_nm1(wfn_nm1):
    assert wfn_nm1.total_energy == pytest.approx(-1709.783384654412, rel=1e-10)


# --- HOMO energies ---

def test_homo_n(wfn_n):
    # MO 44 (last occupied, closed-shell)
    assert wfn_n.homo_hartree == pytest.approx(-0.304791, abs=1e-5)


def test_homo_np1(wfn_np1):
    # HOMO is highest-energy occupied MO (not necessarily last in file)
    assert wfn_np1.homo_hartree == pytest.approx(-0.028695, abs=1e-5)


def test_homo_nm1(wfn_nm1):
    assert wfn_nm1.homo_hartree == pytest.approx(-0.508138, abs=1e-5)


# --- Geometry ---

def test_n_nuclei(wfn_n):
    assert wfn_n.n_nuclei == 8


def test_atom_symbols(wfn_n):
    assert wfn_n.atoms[0] == "Ti"
    assert wfn_n.atoms.count("O") == 5
    assert wfn_n.atoms.count("Al") == 2


def test_ti_x_coordinate(wfn_n):
    # Ti bohr coords (2.24006438, ...) × 0.52917721 ≈ 1.185 Å
    assert wfn_n.coords_ang[0][0] == pytest.approx(1.185, abs=0.01)


# --- MO counts and occupations ---

def test_n_mo_neutral(wfn_n):
    assert wfn_n.n_mo == 44
    assert len(wfn_n.mo_occupations) == 44
    assert len(wfn_n.mo_energies) == 44


def test_closed_shell_occupations(wfn_n):
    assert all(abs(occ - 2.0) < 0.01 for occ in wfn_n.mo_occupations)


def test_n_mo_anion(wfn_np1):
    assert wfn_np1.n_mo == 89
    assert len(wfn_np1.mo_occupations) == 89


def test_open_shell_occupations_anion(wfn_np1):
    assert all(abs(occ - 1.0) < 0.01 for occ in wfn_np1.mo_occupations)


def test_n_mo_cation(wfn_nm1):
    assert wfn_nm1.n_mo == 87
    assert len(wfn_nm1.mo_occupations) == 87
