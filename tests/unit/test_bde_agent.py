"""Tests for cadmia.agents.bde_agent (bond dissociation energy)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.agents.bde_agent import compute_bde

_HARTREE_TO_KCAL = 627.509
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Ti": 22}


def _gauss(path: Path, atoms, energy):
    lines = [
        "", " Standard orientation:",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
    ]
    for i, (sym, x, y, z) in enumerate(atoms, 1):
        lines.append(f"   {i:>4}    {_Z[sym]:>6}           0     {x:11.6f} {y:11.6f} {z:11.6f}")
    lines.append(" ---------------------------------------------------------------------")
    lines.append(f" SCF Done:  E(RB3LYP) = {energy:.8f}     A.U. after   10 cycles")
    lines.append(" Alpha  occ. eigenvalues --  -0.30000")
    lines.append(" Alpha virt. eigenvalues --  -0.10000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def water(tmp_path):
    mol = tmp_path / "h2o.out"
    oh = tmp_path / "oh.out"
    h = tmp_path / "h.out"
    _gauss(mol, [("O", 0, 0, 0), ("H", 0.96, 0, 0), ("H", -0.24, 0.93, 0)], -76.0)
    _gauss(oh, [("O", 0, 0, 0), ("H", 0.96, 0, 0)], -75.5)
    _gauss(h, [("H", 0, 0, 0)], -0.4)
    return mol, oh, h


def test_bde_value(water):
    mol, oh, h = water
    r = compute_bde(mol, [oh, h])
    # BDE = E(OH) + E(H) - E(H2O) = -75.5 - 0.4 - (-76.0) = +0.1 Ha
    assert r.bde_kcal == pytest.approx(0.1 * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.bde_eV == pytest.approx(0.1 * 27.2114, abs=1e-3)
    assert r.bde_kcal > 0


def test_bde_atoms_conserved(water):
    mol, oh, h = water
    r = compute_bde(mol, [oh, h])
    assert r.atoms_conserved is True
    assert not any("mismatch" in w.lower() for w in r.warnings)


def test_bde_atom_mismatch_warns(water, tmp_path):
    mol, oh, h = water
    # Fragments that do not add up to H2O (missing one H)
    r = compute_bde(mol, [oh])
    assert r.atoms_conserved is False
    assert any("mismatch" in w.lower() for w in r.warnings)


def test_bde_missing_energy_skips(water, tmp_path):
    mol, oh, h = water
    # Valid geometry block but no "SCF Done" line -> energy parses as None.
    frag_no_e = tmp_path / "noe.out"
    lines = [
        " Standard orientation:",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
        "      1         1           0        0.000000    0.000000    0.000000",
        " ---------------------------------------------------------------------",
    ]
    frag_no_e.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = compute_bde(mol, [oh, frag_no_e])
    assert r.bde_kcal is None
    assert any("missing total energy" in w.lower() for w in r.warnings)
