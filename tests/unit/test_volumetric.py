"""Tests for cadmia.structure.volumetric."""
from pathlib import Path

import numpy as np
import pytest

from cadmia.io.cube import parse_cube
from cadmia.structure.volumetric.cube_io import ISO_PRESETS, extract_isosurfaces, upsample_grid

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_upsample_shape():
    pytest.importorskip("scipy")
    grid = parse_cube(FIXTURES / "test_small.cube")
    up = upsample_grid(grid, zoom=3, sigma=0.5)
    assert up.shape == (6, 6, 6)


def test_upsample_step_vector_scaled():
    pytest.importorskip("scipy")
    grid = parse_cube(FIXTURES / "test_small.cube")
    up = upsample_grid(grid, zoom=3, sigma=0.5)
    np.testing.assert_allclose(up.vx, grid.vx / 3, rtol=1e-5)


def test_upsample_preserves_structure():
    pytest.importorskip("scipy")
    grid = parse_cube(FIXTURES / "test_small.cube")
    up = upsample_grid(grid, zoom=2, sigma=0.5)
    assert up.structure.n_atoms == grid.structure.n_atoms


def test_extract_isosurfaces_pos():
    pytest.importorskip("skimage")
    pytest.importorskip("scipy")
    grid = parse_cube(FIXTURES / "test_small.cube")
    up = upsample_grid(grid, zoom=3, sigma=0.3)
    iso_pos, iso_neg = extract_isosurfaces(up, iso_pos=0.005, iso_neg=-0.005)
    assert iso_pos is not None
    assert iso_pos.vertices.shape[1] == 3


def test_extract_isosurfaces_none_when_out_of_range():
    pytest.importorskip("skimage")
    pytest.importorskip("scipy")
    grid = parse_cube(FIXTURES / "test_small.cube")
    iso_pos, iso_neg = extract_isosurfaces(grid, iso_pos=999.0, iso_neg=-999.0)
    assert iso_pos is None
    assert iso_neg is None


def test_iso_presets_keys():
    expected = {"homo_lumo", "fukui", "dual", "density", "mep", "adndp"}
    assert expected.issubset(set(ISO_PRESETS.keys()))
