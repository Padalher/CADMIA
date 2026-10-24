"""Tests for cadmia.io."""
from pathlib import Path
import os
import tempfile

import numpy as np
import pytest

from cadmia.io.cube import parse_cube
from cadmia.io.xyz import get_chemical_formula, parse_xyz

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_parse_tio2_monomer():
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    assert s.n_atoms == 3
    assert s.atoms.count("Ti") == 1
    assert s.atoms.count("O") == 2
    assert "Ti1" in s.labels
    assert "O2" in s.labels
    assert "O3" in s.labels


def test_parse_al2o3_cluster():
    s = parse_xyz(FIXTURES / "Al2O3_cluster.xyz")
    assert s.n_atoms == 5
    assert s.atoms.count("Al") == 2
    assert s.atoms.count("O") == 3
    assert "Ti" not in s.atoms


def test_parse_al2tio5_cluster():
    s = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    assert s.n_atoms == 8
    assert "Ti" in s.atoms
    assert "Al" in s.atoms
    assert "O" in s.atoms


def test_tio2_bond_distances():
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    ti_pos = s.positions[s.atoms.index("Ti")]
    for i, atom in enumerate(s.atoms):
        if atom == "O":
            d = np.linalg.norm(s.positions[i] - ti_pos)
            assert abs(d - 1.62) < 1e-6


def test_parse_xyz_rejects_a_trajectory(tmp_path):
    frame = "2\nstep\nO 0 0 0\nO 1.2 0 0\n"
    path = tmp_path / "trj.xyz"
    path.write_text(frame + frame)
    with pytest.raises(ValueError, match="Pass a single geometry"):
        parse_xyz(path)
    path.write_text(frame + "\n\n")
    assert len(parse_xyz(path).atoms) == 2


def test_parse_xyz_rejects_invalid_element():
    content = "2\ntest\nXx 0 0 0\nYy 1 0 0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(content)
        tmppath = f.name
    try:
        with pytest.raises(ValueError, match="[Ii]nvalid|[Uu]nknown|[Ss]ymbol"):
            parse_xyz(tmppath)
    finally:
        os.unlink(tmppath)


def test_hill_notation_organic():
    assert get_chemical_formula(["C", "H", "H", "O"]) == "CH2O"


def test_hill_notation_inorganic():
    assert get_chemical_formula(["Ti", "O", "O"]) == "O2Ti"


def test_hill_notation_mixed():
    assert get_chemical_formula(["Al", "Al", "Ti", "O", "O", "O", "O", "O"]) == "Al2O5Ti"


def test_parse_cube_shape():
    grid = parse_cube(FIXTURES / "test_small.cube")
    assert grid.shape == (2, 2, 2)


def test_parse_cube_atom():
    grid = parse_cube(FIXTURES / "test_small.cube")
    assert grid.structure.n_atoms == 1
    assert grid.structure.atoms[0] == "Ti"


def test_parse_cube_origin():
    grid = parse_cube(FIXTURES / "test_small.cube")
    np.testing.assert_allclose(grid.origin, [0.0, 0.0, 0.0], atol=1e-6)


def test_parse_cube_step_vector():
    bohr = 0.52917721092
    grid = parse_cube(FIXTURES / "test_small.cube")
    np.testing.assert_allclose(grid.vx[0], bohr, rtol=1e-5)


def test_parse_cube_angstrom_units_negative_count(tmp_path):
    """Negative voxel counts mark Angstrom units → no Bohr→Å conversion applied."""
    cube = tmp_path / "ang.cube"
    cube.write_text(
        "comment\ncomment\n"
        "  1    0.000000    0.000000    0.000000\n"
        " -2    1.000000    0.000000    0.000000\n"
        " -2    0.000000    1.000000    0.000000\n"
        " -2    0.000000    0.000000    1.000000\n"
        " 22    0.000000    0.000000    0.000000    0.000000\n"
        "  0.001  0.002  0.003  0.004\n"
        "  0.005  0.006  0.007  0.008\n"
    )
    grid = parse_cube(cube)
    np.testing.assert_allclose(grid.vx[0], 1.0, rtol=1e-9)   # stays in Å, not 0.529
    assert grid.data.shape == (2, 2, 2)


def test_parse_cube_value_range():
    grid = parse_cube(FIXTURES / "test_small.cube")
    assert grid.vmax == pytest.approx(0.01, abs=1e-6)
    assert grid.vmin == pytest.approx(-0.01, abs=1e-6)


def test_parse_cube_returns_volumetric_grid():
    from cadmia.core import VolumetricGrid

    grid = parse_cube(FIXTURES / "test_small.cube")
    assert isinstance(grid, VolumetricGrid)
