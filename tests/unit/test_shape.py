"""Continuous shape measures and polyhedra connectivity."""
import numpy as np
import pytest

from cadmia.core import Structure
from cadmia.structure.structural.polyhedra import (Polyhedron, connectivity_summary,
                                                   polyhedra_connectivity)
from cadmia.structure.structural.shape import (_SHAPES, MAX_LIGANDS, available_shapes,
                                               continuous_shape_measure, shape_measures)

CENTER = np.zeros(3)


def _random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


@pytest.mark.parametrize("cn,name", [(cn, name) for cn, shapes in _SHAPES.items() for name in shapes])
def test_every_reference_scores_zero_against_itself(cn, name):
    reference = _SHAPES[cn][name]
    assert continuous_shape_measure(CENTER, reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_reproduces_published_shape_measures():
    """The only tests that can catch a wrong reference geometry.

    Everything else here is self-referential: a shape scores 0 against itself however badly its
    vertices are placed. These two values are tabulated (Alvarez et al., the SHAPE shape maps), so
    a reference built to the wrong proportions fails them. The trigonal prism was: built with a
    height of 2 instead of sqrt(3) it scores 17.51 against the octahedron, not 16.74.
    """
    square_vs_tetrahedron = continuous_shape_measure(CENTER, _SHAPES[4]["SP-4"], _SHAPES[4]["T-4"])
    assert square_vs_tetrahedron == pytest.approx(33.333, abs=0.01)

    octahedron_vs_prism = continuous_shape_measure(CENTER, _SHAPES[6]["OC-6"], _SHAPES[6]["TPR-6"])
    assert octahedron_vs_prism == pytest.approx(16.737, abs=0.01)


def test_trigonal_prism_reference_has_equal_edges():
    # The defining property of the reference prism: square lateral faces. Checked directly so the
    # geometry cannot drift back without the reason being obvious.
    prism = _SHAPES[6]["TPR-6"]
    triangular_edge = np.linalg.norm(prism[0] - prism[1])
    vertical_edge = np.linalg.norm(prism[0] - prism[3])
    assert triangular_edge == pytest.approx(vertical_edge, rel=1e-9)


def test_measure_is_invariant_to_rotation_translation_and_scale():
    # The measure must describe the arrangement, not where it sits or how big it is.
    reference = _SHAPES[6]["OC-6"]
    shift = np.array([3.0, -1.0, 8.0])
    moved = (reference * 2.7) @ _random_rotation(0).T + shift
    assert continuous_shape_measure(shift, moved, reference) == pytest.approx(0.0, abs=1e-9)


def test_central_atom_separates_pyramid_from_plane():
    # Without the centre in the point set these two are the same equilateral triangle, and no
    # measure over the ligands alone could tell them apart. This is why the centre is included.
    plane = _SHAPES[3]["TP-3"]
    pyramid = _SHAPES[3]["vT-3"]
    assert continuous_shape_measure(CENTER, plane, pyramid) > 1.0
    assert continuous_shape_measure(CENTER, pyramid, plane) > 1.0
    assert shape_measures(CENTER, plane).best_shape == "TP-3"
    assert shape_measures(CENTER, pyramid).best_shape == "vT-3"


def test_measure_grows_with_distortion():
    octahedron = _SHAPES[6]["OC-6"].copy()
    mild = octahedron.copy()
    mild[0] = mild[0] * 1.0 + np.array([0.0, 0.10, 0.0])
    severe = octahedron.copy()
    severe[0] = severe[0] * 1.0 + np.array([0.0, 0.45, 0.0])
    s_none = continuous_shape_measure(CENTER, octahedron, octahedron)
    s_mild = continuous_shape_measure(CENTER, mild, octahedron)
    s_severe = continuous_shape_measure(CENTER, severe, octahedron)
    assert s_none < s_mild < s_severe


def test_measure_is_defined_where_the_tabulated_indices_are_not():
    # A seesaw has no ideal cis angle, so Robinson's sigma^2 is NaN for it; the whole point of
    # the shape measure is that this environment still gets a number.
    seesaw = _SHAPES[4]["SS-4"]
    result = shape_measures(CENTER, seesaw)
    assert result is not None
    assert not np.isnan(result.best_value)
    assert set(result.measures) == set(available_shapes(4))


def test_unknown_coordination_number_returns_none_not_nan():
    assert shape_measures(CENTER, np.eye(3)[:1]) is None          # CN=1, no references
    assert available_shapes(1) == []


def test_rejects_mismatched_reference_and_too_many_ligands():
    with pytest.raises(ValueError, match="vertices"):
        continuous_shape_measure(CENTER, _SHAPES[4]["T-4"], _SHAPES[6]["OC-6"])
    too_many = np.random.default_rng(1).normal(size=(MAX_LIGANDS + 1, 3))
    with pytest.raises(ValueError, match="capped"):
        continuous_shape_measure(CENTER, too_many, too_many)


def _polyhedron(index: int, label: str, neighbors: list[int]) -> Polyhedron:
    return Polyhedron(
        center_index=index, center_label=label, center_element="Al",
        neighbor_indices=neighbors, neighbor_labels=[f"O{i}" for i in neighbors],
        neighbor_element="O", cn=len(neighbors), polyhedron_type="tetrahedron",
        mean_bond_length=1.8, bond_length_distortion=0.0, metal_displacement=0.0,
        mean_angle=109.5, angle_distortion=0.0,
    )


def test_connectivity_classifies_corner_edge_and_face():
    polyhedra = [
        _polyhedron(0, "Al1", [10, 11, 12, 13]),
        _polyhedron(1, "Al2", [13, 20, 21, 22]),          # 1 shared -> corner
        _polyhedron(2, "Al3", [12, 13, 30, 31]),          # 2 shared with Al1 -> edge
        _polyhedron(3, "Al4", [11, 12, 13, 40]),          # 3 shared with Al1 -> face
    ]
    links = polyhedra_connectivity(polyhedra)
    kinds = {(link.center_label_a, link.center_label_b): link.kind for link in links}
    assert kinds[("Al1", "Al2")] == "corner"
    assert kinds[("Al1", "Al3")] == "edge"
    assert kinds[("Al1", "Al4")] == "face"


def test_connectivity_reports_each_pair_once_and_skips_unshared():
    polyhedra = [
        _polyhedron(0, "Al1", [10, 11, 12]),
        _polyhedron(1, "Al2", [12, 20, 21]),
        _polyhedron(2, "Al3", [30, 31, 32]),              # shares nothing
    ]
    links = polyhedra_connectivity(polyhedra)
    assert len(links) == 1
    summary = connectivity_summary(links, polyhedra)
    assert summary["by_kind"] == {"corner": 1, "edge": 0, "face": 0}
    assert summary["isolated_centers"] == ["Al3"]


def test_connectivity_measures_the_metal_metal_distance():
    atoms = ["Al", "Al", "O", "O", "O", "O", "O"]
    structure = Structure(
        atoms=atoms,
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                            [0.0, 1.5, 0.0], [0.0, 0.0, 1.5], [3.0, 1.5, 0.0],
                            [3.0, 0.0, 1.5]]),
        labels=[f"{el}{i + 1}" for i, el in enumerate(atoms)],
        structure_id="pair",
        source_file="test",
    )
    polyhedra = [_polyhedron(0, "Al1", [2, 3, 4]), _polyhedron(1, "Al2", [2, 5, 6])]
    link = polyhedra_connectivity(polyhedra, structure)[0]
    assert link.center_distance == pytest.approx(3.0)
    assert link.shared_labels == [structure.labels[2]]
