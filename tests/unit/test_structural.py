"""Tests for cadmia.structure.structural."""
from dataclasses import replace
import math
from pathlib import Path

import pytest

from cadmia.io.xyz import parse_xyz
from cadmia.structure.structural.analyzer import (
    _self_consistent_ecn_and_distance,
    analyze_structure,
)
from cadmia.structure.structural.polyhedra import analyze_polyhedra, classify_polyhedron

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _reference_hoppe(distances, *, threshold=1e-6, max_iter=100, damping=6.0):
    d_av = min(distances)
    best_d_av = d_av
    best_error = float("inf")
    for _ in range(max(1, max_iter)):
        weights = [
            math.exp(1 - (distance / d_av) ** damping)
            for distance in distances
        ]
        weight_sum = sum(weights)
        new_d_av = (
            sum(distance * weight for distance, weight in zip(distances, weights))
            / weight_sum
        )
        if not math.isfinite(new_d_av) or new_d_av <= 0.0:
            break
        error = abs(new_d_av - d_av)
        if error < best_error:
            best_error = error
            best_d_av = new_d_av
        if error < threshold:
            best_d_av = new_d_av
            break
        d_av = new_d_av
    d_av = best_d_av
    weights = [
        math.exp(1 - (distance / d_av) ** damping)
        for distance in distances
    ]
    return sum(weights), d_av


def test_ecn_takes_every_atom_with_no_distance_cutoff():
    from cadmia.core import Structure
    atoms = ["Ti", "O", "Ti", "O", "Ti", "O", "O"]
    positions = [[0, 0, 0], [1.8, 0, 0], [3.6, 0, 0], [5.4, 0, 0], [7.2, 0, 0],
                 [0, 1.65, 0], [7.2, 1.65, 0]]
    structure = Structure(atoms=atoms, positions=positions,
                          labels=[f"{a}{i + 1}" for i, a in enumerate(atoms)],
                          structure_id="chain", source_file="chain.xyz")
    result = analyze_structure(structure)
    ti1 = 0
    positions = structure.positions
    distances = [
        math.dist(positions[ti1], positions[j])
        for j, element in enumerate(structure.atoms) if element == "Ti" and j != ti1
    ]
    assert max(distances) > 6.0
    expected_ecn, expected_d_av = _reference_hoppe(distances)
    descriptor = result.local_descriptors[ti1]
    assert descriptor.ecn_by_element["Ti"] == pytest.approx(expected_ecn)
    assert descriptor.effective_avg_distance_by_element["Ti"] == pytest.approx(expected_d_av)


def test_hoppe_effective_distance_uses_min_distance_initialization(config):
    distances = [1.80, 2.10, 2.35]
    ecn, d_av = _self_consistent_ecn_and_distance(distances, config)
    expected_ecn, expected_d_av = _reference_hoppe(distances)

    assert d_av == pytest.approx(expected_d_av)
    assert ecn == pytest.approx(expected_ecn)
    assert d_av != pytest.approx(sum(distances) / len(distances))


def test_hoppe_uses_best_iteration_when_max_iterations_reached(config):
    distances = [7.130506837307865, 7.177191409316593]
    limited_config = replace(
        config,
        ecn_parameters={
            **config.ecn_parameters,
            "max_iterations": 6,
            "convergence_threshold": 0.0,
        },
    )

    _, d_av = _self_consistent_ecn_and_distance(distances, limited_config)
    _, expected_d_av = _reference_hoppe(
        distances,
        threshold=0.0,
        max_iter=6,
    )

    last_d_av = min(distances)
    for _ in range(6):
        weights = [
            math.exp(1 - (distance / last_d_av) ** 6.0)
            for distance in distances
        ]
        last_d_av = (
            sum(distance * weight for distance, weight in zip(distances, weights))
            / sum(weights)
        )

    assert d_av == expected_d_av
    assert d_av != last_d_av


def test_tio2_no_al_bonds(config):
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    result = analyze_structure(s, config)
    bond_types = {b.bond_type for b in result.bonds}
    assert not any("Al" in bt for bt in bond_types), (
        f"TiO2 must not have Al bonds, found: {bond_types}"
    )


def test_al2o3_no_ti_bonds(config):
    s = parse_xyz(FIXTURES / "Al2O3_cluster.xyz")
    result = analyze_structure(s, config)
    bond_types = {b.bond_type for b in result.bonds}
    assert not any("Ti" in bt for bt in bond_types), (
        f"Al2O3 must not have Ti bonds, found: {bond_types}"
    )
    assert not any("Ti" in bt for bt in result.directed_bond_statistics)
    assert set(result.directed_bond_statistics) == {"Al-Al", "Al-O"}


def test_tio2_directed_statistics_do_not_include_al(config):
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    result = analyze_structure(s, config)
    assert not any("Al" in bt for bt in result.directed_bond_statistics)
    assert set(result.directed_bond_statistics) == {"Ti-Ti", "Ti-O"}


def test_homonuclear_directed_statistics_keep_unique_bond_count(config):
    s = parse_xyz(FIXTURES / "Al2O3_cluster.xyz")
    result = analyze_structure(s, config)
    unique_al_al = sum(1 for bond in result.bonds if bond.bond_type == "Al-Al")
    stats = result.directed_bond_statistics["Al-Al"]
    assert stats["bond_count"] == unique_al_al
    assert stats["directed_bond_count"] == unique_al_al * 2


def test_al2tio5_has_both_bonds(config):
    s = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    result = analyze_structure(s, config)
    bond_types = {b.bond_type for b in result.bonds}
    has_ti_o = any("Ti" in bt and "O" in bt for bt in bond_types)
    has_al_o = any("Al" in bt and "O" in bt for bt in bond_types)
    assert has_ti_o, f"Al2TiO5 must have Ti-O bonds, found: {bond_types}"
    assert has_al_o, f"Al2TiO5 must have Al-O bonds, found: {bond_types}"


def test_tio2_bond_distance(config):
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    result = analyze_structure(s, config)
    ti_o_bonds = [b for b in result.bonds if "Ti" in b.bond_type and "O" in b.bond_type]
    assert len(ti_o_bonds) == 2
    for bond in ti_o_bonds:
        assert abs(bond.distance - 1.62) < 0.01


def test_directed_ti_al_o_environment_statistics(config):
    s = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    result = analyze_structure(s, config)

    ti_descriptor = next(d for d in result.local_descriptors if d.element == "Ti")
    assert ti_descriptor.bonded_elements["O"] == 3
    assert ti_descriptor.bonded_elements["Al"] == 2
    assert ti_descriptor.ecn_by_element["O"] > 2.8
    assert abs(ti_descriptor.ecn_by_element["Al"] - 2.0) < 1e-6
    assert ti_descriptor.avg_distance_by_element["O"] > 0
    assert ti_descriptor.effective_avg_distance_by_element["O"] > 0

    directed = result.directed_bond_statistics
    assert directed["Ti-Ti"]["bond_count"] == 0
    assert directed["Ti-O"]["bond_count"] == 3
    assert directed["Ti-Al"]["bond_count"] == 2
    assert directed["Al-Al"]["bond_count"] == 0
    assert directed["Al-Ti"]["bond_count"] == 2
    assert directed["Al-O"]["bond_count"] == 4
    assert directed["Ti-O"]["mean_ecn_active_atoms"] > 2.8
    assert directed["Ti-O"]["average_d_av"] == pytest.approx(
        directed["Ti-O"]["mean_atom_effective_avg_distance"]
    )
    assert directed["Ti-O"]["average_d_av"] != pytest.approx(
        directed["Ti-O"]["mean_distance"]
    )


def test_classify_polyhedron():
    assert classify_polyhedron(4) == "tetrahedron"
    assert classify_polyhedron(6) == "octahedron"
    assert classify_polyhedron(5) == "square_pyramid"
    assert classify_polyhedron(7) == "other_CN7"


def test_polyhedra_tio2_monomer(config):
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    result = analyze_structure(s, config)
    polyhedra = analyze_polyhedra(s, result)
    assert len(polyhedra) == 0 or all(poly.cn >= 3 for poly in polyhedra)


def test_polyhedron_distortion_non_negative(config):
    s = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    result = analyze_structure(s, config)
    polyhedra = analyze_polyhedra(s, result)
    for poly in polyhedra:
        assert poly.bond_length_distortion >= 0
        assert poly.metal_displacement >= 0


# ─────────────────────────────────────────────
# Combined structural summary (CN + ECN + avg distances + angles)
# ─────────────────────────────────────────────

def test_structural_summary_combines_cn_ecn_and_angles():
    from cadmia.structure.structural.summary import structural_summary, site_coordination
    s = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    summ = structural_summary(s)
    assert summ["n_atoms"] == s.n_atoms
    # per-atom CN/ECN present
    assert summ["atoms"] and all({"cn", "ecn", "element"} <= set(a) for a in summ["atoms"])
    # average bond distances per type (Ti-O and Al-O exist in the mixed oxide)
    assert any(bt.startswith("Ti-O") for bt in summ["bond_statistics"])
    # polyhedra carry angles
    assert summ["polyhedra"], "expected at least one coordination polyhedron"
    assert all("mean_angle" in p for p in summ["polyhedra"])
    # site lookup returns that atom's coordination
    first = summ["atoms"][0]
    sc = site_coordination(summ, first["index"])
    assert sc["cn"] == first["cn"] and sc["ecn"] == first["ecn"]


def test_ceria_zirconia_metals_are_known_by_default():
    from cadmia.core import Structure
    from cadmia.structure.structural.morphology import metal_ligand_ring_analysis

    atoms = ["Ce", "Zr", "O", "O"]
    positions = [[0, 0, 0], [3.2, 0, 0], [1.6, 1.4, 0], [1.6, -1.4, 0]]
    rhomb = Structure(atoms=atoms, positions=positions,
                      labels=[f"{a}{i + 1}" for i, a in enumerate(atoms)],
                      structure_id="cezr", source_file="cezr.xyz")
    rings, _ = metal_ligand_ring_analysis(rhomb, analyze_structure(rhomb))
    assert len(rings) == 1
