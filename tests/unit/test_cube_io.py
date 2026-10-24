"""Tests for cube writing (io.cube.write_cube) and VolumetricGrid arithmetic."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.core import Structure, VolumetricGrid
from cadmia.io.cube import parse_cube, write_cube


def _grid(data, field_type="density"):
    s = Structure(atoms=["Ti", "O"],
                  positions=np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
                  labels=["Ti1", "O2"], structure_id="t", source_file="t")
    return VolumetricGrid(
        structure=s, origin=np.array([-1.0, -1.0, -1.0]),
        vx=np.array([0.3, 0.0, 0.0]), vy=np.array([0.0, 0.3, 0.0]),
        vz=np.array([0.0, 0.0, 0.3]), data=np.asarray(data, dtype=float),
        field_type=field_type,
    )


def test_write_cube_roundtrips_with_parse(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    grid = _grid(data)
    out = write_cube(grid, tmp_path / "g.cube", comment="roundtrip")
    back = parse_cube(out)
    assert back.shape == (2, 3, 4)
    np.testing.assert_allclose(back.data, data, atol=1e-4)
    np.testing.assert_allclose(back.origin, grid.origin, atol=1e-4)
    np.testing.assert_allclose(back.vx, grid.vx, atol=1e-4)
    assert back.structure.atoms == ["Ti", "O"]
    np.testing.assert_allclose(back.structure.positions, grid.structure.positions, atol=1e-4)


def test_parse_cube_negative_voxel_counts_are_angstrom(tmp_path):
    """Negative voxel counts mark Angstrom units: no Bohr→Å conversion is applied."""
    cube = tmp_path / "ang.cube"
    cube.write_text(
        "comment1\ncomment2\n"
        "    1    1.000000    0.000000    0.000000\n"   # 1 atom, origin at x=1.0 Å
        "   -2    0.500000    0.000000    0.000000\n"   # negative count → Angstrom
        "   -2    0.000000    0.500000    0.000000\n"
        "   -2    0.000000    0.000000    0.500000\n"
        "   22   22.000000    0.100000    0.200000    0.300000\n"
        "  1.0 2.0 3.0 4.0\n  5.0 6.0 7.0 8.0\n"
    )
    grid = parse_cube(cube)
    assert grid.shape == (2, 2, 2)
    # values kept verbatim, not scaled by BOHR_TO_ANG (~0.529)
    np.testing.assert_allclose(grid.origin, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(grid.vx, [0.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(grid.structure.positions[0], [0.1, 0.2, 0.3], atol=1e-9)


def test_grid_subtraction_and_scaling():
    a = _grid(np.full((2, 2, 2), 3.0))
    b = _grid(np.full((2, 2, 2), 1.0))
    diff = a - b
    assert diff.field_type == "difference"
    np.testing.assert_allclose(diff.data, 2.0)
    summed = a + b
    np.testing.assert_allclose(summed.data, 4.0)
    np.testing.assert_allclose(a.scaled(0.5).data, 1.5)


def test_grid_arithmetic_requires_alignment():
    a = _grid(np.zeros((2, 2, 2)))
    b = _grid(np.zeros((2, 2, 3)))   # different shape
    assert not a.same_grid_as(b)
    with pytest.raises(ValueError, match="voxel-aligned"):
        _ = a - b
