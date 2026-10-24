"""Tests for cadmia.structure.rendering."""
from pathlib import Path

import numpy as np
import pytest

from cadmia.structure.rendering.geometry import rotation_matrix, rotate_normals, rotate_positions
from cadmia.structure.rendering.geometry import align_to_inertia
from cadmia.structure.rendering.povray import RESOLUTION_PRESETS, check_dependencies


def test_check_dependencies_returns_dict():
    deps = check_dependencies()
    assert "povray" in deps
    assert "magick" in deps
    assert isinstance(deps["povray"], bool)
    assert isinstance(deps["magick"], bool)


def test_resolution_presets_keys():
    assert "preview" in RESOLUTION_PRESETS
    assert "publication" in RESOLUTION_PRESETS
    assert RESOLUTION_PRESETS["publication"][0] >= 2048


def test_rotation_matrix_identity():
    M = rotation_matrix(0.0, 0.0)
    np.testing.assert_allclose(M, np.eye(3), atol=1e-10)


def test_rotation_matrix_360():
    M = rotation_matrix(360.0, 360.0)
    np.testing.assert_allclose(M, np.eye(3), atol=1e-6)


def test_rotate_positions_preserves_shape():
    positions = np.random.randn(5, 3)
    com = positions.mean(axis=0)
    M = rotation_matrix(45.0, 90.0)
    rotated = rotate_positions(positions, com, M)
    assert rotated.shape == positions.shape


def test_rotate_normals_unit_length():
    normals = np.random.randn(10, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    M = rotation_matrix(30.0, 60.0)
    rotated = rotate_normals(normals, M)
    norms = np.linalg.norm(rotated, axis=1)
    np.testing.assert_allclose(norms, np.ones(10), atol=1e-10)


def test_write_pov_scene_creates_file(tmp_path):
    from cadmia.io.xyz import parse_xyz
    from cadmia.structure.rendering.povray import write_pov_scene

    fixtures = Path(__file__).parent.parent / "fixtures"
    s = parse_xyz(fixtures / "TiO2_monomer.xyz")
    out = str(tmp_path / "test_scene")
    write_pov_scene(s, None, None, out, cam_half_size=5.0)
    pov_file = Path(f"{out}.pov")
    assert pov_file.exists()
    content = pov_file.read_text()
    assert "#version 3.7" in content
    assert "camera" in content
    assert "sphere" in content


def test_align_to_inertia_preserves_mass_weighted_center(config):
    from cadmia.core import Structure

    structure = Structure(
        atoms=["Ti", "O", "O"],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        labels=["Ti1", "O2", "O3"],
        structure_id="heterogeneous",
        source_file="",
    )
    masses = np.array([config.get_atomic_mass(atom) for atom in structure.atoms])
    arithmetic_center = structure.positions.mean(axis=0)
    mass_center = np.average(structure.positions, axis=0, weights=masses)
    assert not np.allclose(arithmetic_center, mass_center)

    aligned, _, _ = align_to_inertia(structure, None, None, config=config)
    aligned_mass_center = np.average(aligned.positions, axis=0, weights=masses)
    np.testing.assert_allclose(aligned_mass_center, mass_center, atol=1e-10)
    assert aligned.metadata["inertia_uses_atomic_masses"] is True


def test_align_to_inertia_preserves_bonds_and_metadata(config):
    """Aligning only rotates: rebuilding the Structure by hand dropped bonds and symmetry."""
    from cadmia.core import Bond, Structure

    structure = Structure(
        atoms=["Ti", "O", "O"],
        positions=np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.0, 1.9, 0.0]]),
        labels=["Ti1", "O2", "O3"],
        structure_id="with_bonds",
        source_file="",
        bonds=[Bond(0, 1, 1.9), Bond(0, 2, 1.9)],
        point_group="C2v",
        charge=-1,
        multiplicity=3,
    )
    aligned, _, _ = align_to_inertia(structure, None, None, config=config)
    assert len(aligned.bonds) == 2
    assert aligned.point_group == "C2v"
    assert aligned.charge == -1 and aligned.multiplicity == 3
    # the rotation itself still happened
    assert aligned.metadata["inertia_uses_atomic_masses"] is True


def test_render_pov_reports_a_timeout(tmp_path, monkeypatch):
    """A pathological scene must not hang the program forever."""
    import subprocess

    from cadmia.structure.rendering import povray

    scene = tmp_path / "scene.pov"
    scene.write_text("// empty\n", encoding="utf-8")

    def _hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="povray", timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(povray.subprocess, "run", _hang)
    with pytest.raises(RuntimeError, match="exceeded"):
        povray.render_pov(str(scene), (512, 512), timeout_s=1)


def test_render_pov_passes_a_timeout_to_subprocess(tmp_path, monkeypatch):
    import subprocess

    from cadmia.structure.rendering import povray

    scene = tmp_path / "scene.pov"
    scene.write_text("// empty\n", encoding="utf-8")
    seen = {}

    def _run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(povray.subprocess, "run", _run)
    povray.render_pov(str(scene), (512, 512))
    assert seen.get("timeout") == povray.DEFAULT_RENDER_TIMEOUT_S
