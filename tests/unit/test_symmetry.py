"""Tests for cadmia.structure.structural.symmetry: symmetry from the geometry."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.structure.structural.symmetry import detect_symmetry

_T = 1.0 / np.sqrt(3.0)


@pytest.mark.parametrize("atoms, positions, group, sigma, chiral", [
    (["O", "H", "H"], [[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]], "C2v", 2, False),
    (["N", "H", "H", "H"], [[0, 0, 0.1], [0.94, 0, -0.27], [-0.47, 0.814, -0.27],
                           [-0.47, -0.814, -0.27]], "C3v", 3, False),
    (["C", "H", "H", "H", "H"], [[0, 0, 0], [_T, _T, _T], [-_T, -_T, _T], [-_T, _T, -_T],
                                 [_T, -_T, -_T]], "Td", 12, False),
    (["B", "F", "F", "F"], [[0, 0, 0], [1.3, 0, 0], [-0.65, 1.1258, 0],
                           [-0.65, -1.1258, 0]], "D3h", 6, False),
    (["O", "C", "O"], [[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]], "D∞h", 2, False),
    (["C", "O"], [[0, 0, 0], [1.13, 0, 0]], "C∞v", 1, False),
    # H2O2-like twist: a C2 axis and no improper operation, so chiral
    (["O", "O", "H", "H"], [[0, 0.7, 0], [0, -0.7, 0], [0.9, 0.9, 0.3], [-0.9, -0.9, 0.3]],
     "C2", 2, True),
    (["Ti", "O", "Al", "H"], [[0, 0, 0], [1.7, 0.2, 0.1], [-0.4, 2.1, 0.3], [0.3, -0.5, 1.9]],
     "C1", 1, True),
])
def test_point_group_sigma_and_chirality(atoms, positions, group, sigma, chiral):
    sym = detect_symmetry(atoms, positions)
    assert (sym.point_group, sym.sigma, sym.chiral) == (group, sigma, chiral)


def test_tolerance_decides_what_counts_as_symmetric():
    water = np.array([[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.762, -0.47]])   # 2 mÅ off C2v
    assert detect_symmetry(["O", "H", "H"], water).point_group == "C2v"
    assert detect_symmetry(["O", "H", "H"], water, tol=1e-4).point_group == "Cs"


def test_c2v_fixture_agrees_with_gaussian():
    from pathlib import Path
    from cadmia.io.electronic import load_electronic
    state, _ = load_electronic(Path(__file__).resolve().parents[1] / "fixtures" / "TiO2_monomer.out")
    sym = detect_symmetry(state.structure.atoms, state.structure.positions)
    assert (sym.point_group, sym.sigma) == ("C2v", 2) and state.rotational_symmetry_number == 2
