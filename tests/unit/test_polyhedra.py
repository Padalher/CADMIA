"""Coordination-polyhedron descriptors, checked against ideal geometries.

Every ideal polyhedron built here must report zero angular distortion — that is the
property the previous implementation lacked: averaging cis and trans angles together
scored a perfect octahedron at 36.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cadmia.core import Structure
from cadmia.structure.structural.polyhedra import (
    angle_variance, classify_polyhedron, compute_polyhedron, convex_hull_volume,
    quadratic_elongation, tau_4, tau_5, _angles,
)


def _structure(positions) -> Structure:
    n = len(positions) - 1
    return Structure(
        atoms=["Ti"] + ["O"] * n,
        positions=np.asarray(positions, dtype=float),
        labels=["Ti1"] + [f"O{i}" for i in range(1, n + 1)],
        structure_id="ideal", source_file="memory",
    )


# Centre first, then ligands. All ideal — √3 exactly, so a rounded 1.732 does not show
# up as a distortion of the geometry rather than of the code.
_S3 = math.sqrt(3.0)
OCTAHEDRON = [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2]]
TETRAHEDRON = [[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
SQUARE_PLANAR = [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]]
SQUARE_PYRAMID = [[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2]]
TRIGONAL_BIPYRAMID = [[0, 0, 0], [2, 0, 0], [-1, _S3, 0], [-1, -_S3, 0], [0, 0, 2], [0, 0, -2]]
TRIANGLE = [[0, 0, 0], [2, 0, 0], [-1, _S3, 0], [-1, -_S3, 0]]


def _poly(positions):
    s = _structure(positions)
    return compute_polyhedron(0, list(range(1, len(positions))), s)


@pytest.mark.parametrize("positions,expected", [
    (OCTAHEDRON, "octahedron"),
    (TETRAHEDRON, "tetrahedron"),
    (SQUARE_PLANAR, "square_planar"),
    (SQUARE_PYRAMID, "square_pyramid"),
    (TRIGONAL_BIPYRAMID, "trigonal_bipyramid"),
    (TRIANGLE, "triangle"),
])
def test_ideal_polyhedra_are_classified_by_geometry(positions, expected):
    assert _poly(positions).polyhedron_type == expected


def test_square_planar_is_not_called_a_tetrahedron():
    """Both have CN=4; classifying on the count alone confused them."""
    assert _poly(SQUARE_PLANAR).polyhedron_type == "square_planar"
    assert _poly(TETRAHEDRON).polyhedron_type == "tetrahedron"


def test_trigonal_bipyramid_is_not_called_a_square_pyramid():
    assert _poly(TRIGONAL_BIPYRAMID).polyhedron_type == "trigonal_bipyramid"
    assert _poly(SQUARE_PYRAMID).polyhedron_type == "square_pyramid"


@pytest.mark.parametrize("positions", [
    OCTAHEDRON, TETRAHEDRON, SQUARE_PLANAR, SQUARE_PYRAMID, TRIANGLE,
])
def test_ideal_polyhedra_have_zero_angular_distortion(positions):
    assert _poly(positions).angle_distortion == pytest.approx(0.0, abs=1e-6)


def test_perfect_octahedron_reports_ninety_degrees():
    """Averaging the trans angles in gave 108° for a perfect octahedron."""
    p = _poly(OCTAHEDRON)
    assert p.mean_angle == pytest.approx(90.0, abs=1e-6)
    assert p.angle_distortion == pytest.approx(0.0, abs=1e-6)


def test_distortion_grows_with_real_distortion():
    """The old metric moved 0.27 out of 36 between a perfect and a distorted octahedron."""
    distorted = [[0, 0, 0], [2.2, 0, 0], [-1.8, 0, 0], [0, 2.1, 0], [0, -1.9, 0],
                 [0, 0.3, 2.0], [0, -0.3, -2.0]]
    assert _poly(OCTAHEDRON).angle_distortion == pytest.approx(0.0, abs=1e-6)
    assert _poly(distorted).angle_distortion > 10.0


def test_trigonal_bipyramid_distortion_is_nan_not_zero():
    """Its ideal cis angles are 90° and 120°: a single-reference variance is meaningless."""
    assert math.isnan(_poly(TRIGONAL_BIPYRAMID).angle_distortion)


def test_angle_variance_is_nan_for_unknown_geometry():
    assert math.isnan(angle_variance(np.array([90.0, 90.0]), "other_CN7"))


def test_tau5_separates_the_two_five_coordinate_geometries():
    tb = _angles(np.array([0.0, 0, 0]), np.asarray(TRIGONAL_BIPYRAMID[1:], dtype=float))
    sp = _angles(np.array([0.0, 0, 0]), np.asarray(SQUARE_PYRAMID[1:], dtype=float))
    assert tau_5(tb) == pytest.approx(1.0, abs=0.02)     # ideal trigonal bipyramid
    assert tau_5(sp) == pytest.approx(0.0, abs=0.02)     # ideal square pyramid


def test_tau4_separates_the_two_four_coordinate_geometries():
    td = _angles(np.array([0.0, 0, 0]), np.asarray(TETRAHEDRON[1:], dtype=float))
    sq = _angles(np.array([0.0, 0, 0]), np.asarray(SQUARE_PLANAR[1:], dtype=float))
    assert tau_4(td) == pytest.approx(1.0, abs=0.02)     # ideal tetrahedron
    assert tau_4(sq) == pytest.approx(0.0, abs=0.02)     # ideal square planar


def test_classify_without_angles_falls_back_to_coordination_number():
    assert classify_polyhedron(6) == "octahedron"
    assert classify_polyhedron(4) == "tetrahedron"
    assert classify_polyhedron(7) == "other_CN7"


def test_bond_length_descriptors_are_unchanged():
    """Baur's index and the mean bond length were already correct; keep them so."""
    p = _poly(OCTAHEDRON)
    assert p.mean_bond_length == pytest.approx(2.0)
    assert p.bond_length_distortion == pytest.approx(0.0, abs=1e-12)
    assert p.metal_displacement == pytest.approx(0.0, abs=1e-12)


def test_metal_displacement_ratio_is_size_independent():
    """The absolute displacement cannot be compared between polyhedra of different size."""
    small = [[0.2, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    large = [[0.4, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2]]
    ps, pl = _poly(small), _poly(large)
    # twice the absolute off-centering, but the same relative one
    assert pl.metal_displacement == pytest.approx(2 * ps.metal_displacement, rel=1e-6)
    assert pl.metal_displacement_ratio == pytest.approx(ps.metal_displacement_ratio, rel=1e-6)
    # and the ratio is actually the normalized quantity, not a placeholder
    assert ps.metal_displacement_ratio > 0.1
    assert ps.metal_displacement_ratio == pytest.approx(
        ps.metal_displacement / ps.mean_bond_length, rel=1e-9)


def test_ideal_polyhedron_is_centred():
    assert _poly(OCTAHEDRON).metal_displacement_ratio == pytest.approx(0.0, abs=1e-12)


# ── volume and quadratic elongation ──

def _ligands(positions):
    return np.asarray(positions[1:], dtype=float)


@pytest.mark.parametrize("positions,exact", [
    # V = k·R³ for the regular polyhedron with centre-to-vertex distance R (here R = 2).
    (OCTAHEDRON, 4 / 3 * 2.0 ** 3),
    (SQUARE_PYRAMID, 2 / 3 * 2.0 ** 3),
    (TRIGONAL_BIPYRAMID, math.sqrt(3) / 2 * 2.0 ** 3),
])
def test_hull_volume_matches_the_analytic_solid(positions, exact):
    assert convex_hull_volume(_ligands(positions)) == pytest.approx(exact, rel=1e-9)


def test_hull_volume_of_a_tetrahedron():
    """Circumradius √3 here, since the vertices are (±1, ±1, ±1)."""
    assert convex_hull_volume(_ligands(TETRAHEDRON)) == pytest.approx(
        8 / (9 * math.sqrt(3)) * _S3 ** 3, rel=1e-9)


def test_hull_volume_of_a_cube():
    """Six square faces: the case that breaks a naive per-triple sum, which counts each
    square four times."""
    a = 2.0
    cube = np.array([[x, y, z] for x in (0.0, a) for y in (0.0, a) for z in (0.0, a)])
    assert convex_hull_volume(cube) == pytest.approx(a ** 3, rel=1e-9)


@pytest.mark.parametrize("positions", [OCTAHEDRON, TETRAHEDRON, SQUARE_PYRAMID,
                                       TRIGONAL_BIPYRAMID])
def test_ideal_polyhedra_have_unit_quadratic_elongation(positions):
    assert _poly(positions).quadratic_elongation == pytest.approx(1.0, rel=1e-9)


def test_uniform_dilation_is_not_an_elongation():
    """A dilated octahedron is still regular: only its volume changes, and l₀ with it."""
    dilated = [[c * 1.10 for c in p] for p in OCTAHEDRON]
    assert _poly(dilated).quadratic_elongation == pytest.approx(1.0, rel=1e-9)


def test_elongation_sees_what_baur_index_cannot():
    """The reason this descriptor exists.

    All six bonds keep exactly the same length, so Baur's index — which normalises by the
    observed mean — is exactly zero. The polyhedron is nonetheless distorted: two
    equatorial ligands are bent by 20°, the hull shrinks, and ⟨λ⟩ (normalised by the
    equal-volume *ideal*) reports it.
    """
    t = math.radians(20.0)
    bent = [row[:] for row in OCTAHEDRON]
    bent[3] = [2 * math.sin(t), 2 * math.cos(t), 0.0]
    bent[4] = [-2 * math.sin(t), -2 * math.cos(t), 0.0]
    p = _poly(bent)
    assert p.bond_length_distortion == pytest.approx(0.0, abs=1e-12)
    assert p.quadratic_elongation > 1.04
    assert p.volume < 4 / 3 * 2.0 ** 3


def test_elongation_is_never_below_one_for_a_convex_polyhedron():
    """⟨λ⟩ ≥ 1 by construction; a value under 1 would mean the volume is wrong."""
    elongated = [row[:] for row in OCTAHEDRON]
    elongated[5] = [0.0, 0.0, 2.8]
    elongated[6] = [0.0, 0.0, -2.8]
    p = _poly(elongated)
    assert p.quadratic_elongation > 1.0
    assert p.bond_length_distortion > 0.0        # here Baur does see it too


@pytest.mark.parametrize("positions", [SQUARE_PLANAR, TRIANGLE])
def test_flat_polyhedra_have_no_volume_and_no_elongation(positions):
    """NaN and not 1.0: a flat set has no elongation *defined*, and 1.0 would claim it has
    none — the same rule that keeps σ² NaN for a trigonal bipyramid."""
    p = _poly(positions)
    assert math.isnan(p.volume)
    assert math.isnan(p.quadratic_elongation)


def test_elongation_is_nan_without_a_regular_reference():
    """A seesaw has a volume but no regular counterpart to compare against."""
    assert math.isnan(quadratic_elongation(np.array([2.0, 2.0, 2.0, 2.0]), 5.0, "seesaw"))
    assert math.isnan(quadratic_elongation(np.array([2.0]), float("nan"), "octahedron"))


def test_warns_when_ligands_cannot_be_inferred():
    """Silence cannot distinguish "no polyhedra" from "could not tell what the ligands are"."""
    from cadmia.structure.structural.analyzer import analyze_structure
    from cadmia.structure.structural.polyhedra import analyze_polyhedra

    # Two metals bonded to each other and nothing else: there are bonds, but none of
    # them is metal-ligand, so no ligand element can be inferred.
    s = Structure(atoms=["Ti", "Ti"], positions=np.array([[0.0, 0, 0], [2.5, 0, 0]]),
                  labels=["Ti1", "Ti2"], structure_id="metals", source_file="memory")
    result = analyze_structure(s)
    if not result.bonds:                     # nothing to warn about if no bond was found
        pytest.skip("bond detection found no Ti-Ti bond")
    with pytest.warns(UserWarning, match="no metal-ligand bonds"):
        assert analyze_polyhedra(s, result) == []


def test_no_warning_for_a_structure_without_metals(recwarn):
    """A probe molecule has no polyhedra to find; warning there would be noise."""
    from cadmia.structure.structural.analyzer import analyze_structure
    from cadmia.structure.structural.polyhedra import analyze_polyhedra

    s = Structure(atoms=["O", "H", "H"],
                  positions=np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
                  labels=["O1", "H2", "H3"], structure_id="water", source_file="memory")
    assert analyze_polyhedra(s, analyze_structure(s)) == []
    assert not [w for w in recwarn if "metal-ligand" in str(w.message)]
