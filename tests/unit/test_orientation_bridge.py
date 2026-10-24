"""Bridge tests: orientation flows into the POV-Ray scene and the VMD script."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cadmia.core import Structure
from cadmia.structure.rendering.orientation import Orientation
from cadmia.structure.rendering.povray import write_pov_scene


def _structure():
    return Structure(atoms=["Ti", "O"], positions=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
                     labels=["Ti1", "O2"], structure_id="t", source_file="t")


def _sphere_centers(pov_text):
    import re
    return np.array([[float(x) for x in m.group(1).split(",")]
                     for m in re.finditer(r"sphere \{ <([^>]+)>", pov_text)])


def test_povray_scene_respects_orientation(tmp_path):
    s = _structure()
    sphi = math.sin(math.pi / 4)
    # +90° about z: (1,0,0) → (0,1,0), (-1,0,0) → (0,-1,0)
    p = write_pov_scene(s, None, None, str(tmp_path / "o.pov"),
                        orientation=Orientation(0, 0, sphi, sphi))
    centers = _sphere_centers(p.read_text())
    np.testing.assert_allclose(sorted(centers[:, 1]), [-1.0, 1.0], atol=1e-5)   # rotated onto y
    np.testing.assert_allclose(centers[:, 0], [0.0, 0.0], atol=1e-5)            # x cleared


def test_povray_identity_orientation_is_unrotated(tmp_path):
    s = _structure()
    p = write_pov_scene(s, None, None, str(tmp_path / "i.pov"), orientation=Orientation())
    assert "<1.000000, 0.000000, 0.000000>" in p.read_text()


def test_vmd_script_emits_rotate_matrix():
    from cadmia.core import VolumetricGrid
    from cadmia.electronic.vmd_export import vmd_script
    g = VolumetricGrid(structure=_structure(), origin=np.zeros(3),
                       vx=np.array([0.2, 0, 0]), vy=np.array([0, 0.2, 0]),
                       vz=np.array([0, 0, 0.2]), data=np.zeros((2, 2, 2)), field_type="density")
    sphi = math.sin(math.pi / 4)
    script, _ = vmd_script(g, cube_path="x.cube", orientation=Orientation(0, 0, sphi, sphi))
    assert "molinfo top set rotate_matrix" in script
    # identity orientation emits no rotation line
    script2, _ = vmd_script(g, cube_path="x.cube", orientation=Orientation())
    assert "rotate_matrix" not in script2
