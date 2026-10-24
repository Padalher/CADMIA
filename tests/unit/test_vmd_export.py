"""Tests for cadmia.electronic.vmd_export (VMD/Tachyon .tcl generator)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadmia.core import Structure, VolumetricGrid
from cadmia.electronic.vmd_export import vmd_script, write_vmd_script

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _grid(data: np.ndarray) -> VolumetricGrid:
    s = Structure(atoms=["O"], positions=np.zeros((1, 3)), labels=["O1"],
                  structure_id="t", source_file="t.cube")
    return VolumetricGrid(structure=s, origin=np.zeros(3),
                          vx=np.array([0.2, 0.0, 0.0]), vy=np.array([0.0, 0.2, 0.0]),
                          vz=np.array([0.0, 0.0, 0.2]), data=data, field_type="orbital")


def test_signed_field_emits_two_isosurfaces():
    script, _ = vmd_script(_grid(np.linspace(-1, 1, 27).reshape(3, 3, 3)),
                           cube_path="orb.cube", isovalue=0.3)
    assert script.count("Isosurface") == 2
    assert "Isosurface 0.3 " in script
    assert "Isosurface -0.3 " in script
    assert "mol new" in script


def test_positive_field_emits_one_isosurface():
    script, _ = vmd_script(_grid(np.linspace(0.01, 1, 27).reshape(3, 3, 3)),
                           cube_path="dens.cube", isovalue=0.2)
    assert script.count("Isosurface") == 1


def test_default_isovalue_fraction():
    data = np.zeros((3, 3, 3))
    data[1, 1, 1] = 2.0          # max abs 2.0 → iso 0.30
    script, _ = vmd_script(_grid(data), cube_path="x.cube")
    assert "Isosurface 0.3 " in script


def test_builtin_renderer_by_default():
    script, _ = vmd_script(_grid(np.linspace(0.01, 1, 27).reshape(3, 3, 3)), cube_path="x.cube")
    assert "render TachyonInternal" in script
    assert "render_hires" in script


def test_external_tachyon_command():
    script, _ = vmd_script(_grid(np.linspace(0.01, 1, 27).reshape(3, 3, 3)),
                           cube_path="x.cube", tachyon="/opt/tachyon", resolution=(800, 600))
    assert "/opt/tachyon" in script
    assert "-res 800 600" in script


def test_cube_path_is_absolute(tmp_path):
    cube = tmp_path / "field.cube"
    script, _ = vmd_script(_grid(np.ones((3, 3, 3))), cube_path=cube)
    assert str(cube.resolve()) in script


def test_write_from_cube_file(tmp_path):
    out = tmp_path / "scene.tcl"
    warnings = write_vmd_script(FIXTURES / "test_small.cube", out)
    text = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "mol new" in text and "Isosurface" in text
    assert str((FIXTURES / "test_small.cube").resolve()) in text
    assert isinstance(warnings, list)


def test_write_grid_requires_cube_path():
    with pytest.raises(ValueError, match="cube_path is required"):
        # source_file is set to "t.cube" in _grid, so this actually has a path; use a blank one
        g = _grid(np.ones((3, 3, 3)))
        g.source_file = ""
        write_vmd_script(g, "out.tcl")
