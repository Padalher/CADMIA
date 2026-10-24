"""Bond angles, rings, global shape and relative coordination."""
import json
from pathlib import Path

import numpy as np
import pytest

from cadmia.core import Structure
from cadmia.io.xyz import parse_xyz
from cadmia.structure.structural.analyzer import analyze_structure
from cadmia.structure.structural.angles import (angle_between, angle_statistics, bond_angles,
                                                bridging_angles)
from cadmia.structure.structural.morphology import (MAX_RING_SIZE, coordination_outliers,
                                                    find_rings, global_shape, metal_ligand_rings,
                                                    ring_statistics)


def _structure(atoms, positions) -> Structure:
    return Structure(
        atoms=list(atoms),
        positions=np.asarray(positions, dtype=float),
        labels=[f"{el}{i + 1}" for i, el in enumerate(atoms)],
        structure_id="test",
        source_file="test",
    )


def test_angle_between_known_values():
    origin = [0.0, 0.0, 0.0]
    assert angle_between(origin, [1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle_between(origin, [1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
    assert angle_between(origin, [1, 0, 0], [1, 1, 0]) == pytest.approx(45.0)


def test_angle_between_is_nan_for_a_zero_length_arm():
    # An arccos on a zero vector is meaningless; NaN says so instead of inventing 90 degrees.
    assert np.isnan(angle_between([0, 0, 0], [0, 0, 0], [1, 0, 0]))


def test_triplet_label_is_direction_independent():
    # Al-O-Ti and Ti-O-Al are the same angle; they must not appear as two entries.
    bent = _structure(["O", "Al", "Ti"], [[0, 0, 0], [1.8, 0, 0], [0, 1.9, 0]])
    result = analyze_structure(bent)
    triplets = {item.triplet for item in bond_angles(bent, result)}
    assert "Al-O-Ti" in triplets
    assert "Ti-O-Al" not in triplets          # the reversed spelling must never appear
    assert all(label.count("-") == 2 for label in triplets)


def test_statistics_separate_cis_from_trans():
    # A 180 degree angle is a trans pair and must not be pooled with the cis ones: an ideal
    # octahedron has twelve angles at 90 and three at 180, and pooling them reports a mean of
    # 108 with a spread of 36 for a centre with no distortion at all.
    linear = _structure(["O", "Al", "Al"], [[0, 0, 0], [1.8, 0, 0], [-1.8, 0, 0]])
    result = analyze_structure(linear)
    stats = angle_statistics(bond_angles(linear, result))["Al-O-Al"]
    assert stats["count"] == 1
    assert stats["n_trans"] == 1
    assert stats["mean_trans"] == pytest.approx(180.0)
    assert np.isnan(stats["mean"])          # no cis angle to summarise


def test_statistics_std_is_nan_for_a_single_sample():
    # Zero would read as "perfectly uniform"; one angle simply has no spread to report.
    bent = _structure(["O", "Al", "Al"], [[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
    result = analyze_structure(bent)
    stats = angle_statistics(bond_angles(bent, result))["Al-O-Al"]
    assert stats["mean"] == pytest.approx(90.0)
    assert np.isnan(stats["std"])


def test_bridging_angles_are_the_metal_ligand_metal_subset():
    bridge = _structure(["O", "Al", "Al"], [[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
    result = analyze_structure(bridge)
    bridges = bridging_angles(bridge, result, metal_elements=["Al", "Ti"])
    assert len(bridges) == 1
    assert bridges[0].triplet == "Al-O-Al"
    assert bridges[0].angle == pytest.approx(90.0)
    # An angle centred on a metal is not a bridge, however many metals are around it.
    assert all(item.vertex_label.startswith("O") for item in bridges)


def _square_ring() -> Structure:
    # Al-O-Al-O, the four-membered motif, with a metal-metal contact across the diagonal.
    return _structure(
        ["Al", "O", "Al", "O"],
        [[0.0, 0.0, 0.0], [1.3, 1.3, 0.0], [2.6, 0.0, 0.0], [1.3, -1.3, 0.0]],
    )


def test_metal_ligand_rings_find_the_four_membered_motif():
    structure = _square_ring()
    result = analyze_structure(structure)
    rings = metal_ligand_rings(structure, result, metal_elements=["Al", "Ti"])
    assert len(rings) == 1
    assert rings[0].size == 4
    assert rings[0].composition == "Al2O2"


def test_full_graph_rings_are_short_circuited_by_metal_metal_contacts():
    # The reason metal_ligand_rings exists: across the whole graph the Al-Al diagonal turns the
    # four-membered ring into two triangles, and the framework motif disappears from the count.
    structure = _square_ring()
    result = analyze_structure(structure)
    assert any(structure.atoms[b.atom_i] == structure.atoms[b.atom_j] == "Al"
               for b in result.bonds), "the premise of this test is that Al-Al counts as a bond"
    assert min(ring.size for ring in find_rings(structure, result)) == 3
    assert min(ring.size for ring in
               metal_ligand_rings(structure, result, metal_elements=["Al", "Ti"])) == 4


def test_rings_are_counted_once_not_once_per_bond():
    structure = _square_ring()
    result = analyze_structure(structure)
    rings = metal_ligand_rings(structure, result, metal_elements=["Al", "Ti"])
    stats = ring_statistics(rings)
    assert stats["total"] == 1
    assert stats["by_size"] == {4: 1}
    assert stats["smallest"] == 4


def test_open_chain_closes_no_ring():
    chain = _structure(["Al", "O", "Al"], [[0, 0, 0], [1.8, 0, 0], [3.6, 0, 0]])
    result = analyze_structure(chain)
    assert metal_ligand_rings(chain, result, metal_elements=["Al", "Ti"]) == []
    assert ring_statistics([])["smallest"] == 0


def test_global_shape_separates_a_line_from_a_sphere():
    line = _structure(["O"] * 5, [[i * 1.5, 0.0, 0.0] for i in range(5)])
    assert global_shape(line).anisotropy == pytest.approx(1.0, abs=1e-9)
    # A regular tetrahedron is the smallest isotropic arrangement: all three moments equal.
    tetra = _structure(["O"] * 4, [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    shape = global_shape(tetra)
    assert shape.anisotropy == pytest.approx(0.0, abs=1e-9)
    assert shape.asphericity == pytest.approx(0.0, abs=1e-9)


def test_global_shape_radius_and_diameter():
    pair = _structure(["O", "O"], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    shape = global_shape(pair)
    assert shape.radius_of_gyration == pytest.approx(1.0)
    assert shape.max_diameter == pytest.approx(2.0)
    assert shape.n_atoms == 2


def test_global_shape_rejects_an_empty_structure():
    with pytest.raises(ValueError, match="no shape"):
        global_shape(_structure([], np.zeros((0, 3))))


def test_coordination_outliers_are_relative_to_the_element_not_absolute():
    # Two Al with the same ECN deviate by zero even if that ECN is low in absolute terms; the
    # point of the column is to find the atom that is unusual *for its own element here*.
    structure = _structure(
        ["Al", "O", "O", "O", "Al", "O"],
        [[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [0, 0, 1.8], [4.5, 0, 0], [4.5, 1.8, 0]],
    )
    result = analyze_structure(structure)
    rows = {row["label"]: row for row in coordination_outliers(result)}
    assert set(rows) == {atom for atom in structure.labels}
    for row in rows.values():
        assert row["deviation"] == pytest.approx(row["ecn"] - row["element_mean_ecn"])
    lonely = min(rows.values(), key=lambda row: row["deviation"])
    assert lonely["deviation"] < 0


def test_coordination_outlier_z_is_nan_for_a_single_atom_of_its_element():
    solo = _structure(["Ti", "O", "O"], [[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
    result = analyze_structure(solo)
    rows = {row["label"]: row for row in coordination_outliers(result)}
    assert np.isnan(rows["Ti1"]["z"])


def test_ring_search_respects_the_size_cap():
    # A six-membered ring exists and is found; cap the search below it and it must disappear,
    # which is what shows the cap is enforced rather than merely never exceeded.
    hexagon = _structure(
        ["Al", "O"] * 3,
        [[1.9 * np.cos(a), 1.9 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 6,
                                                                      endpoint=False)],
    )
    result = analyze_structure(hexagon)
    found = metal_ligand_rings(hexagon, result, metal_elements=["Al", "Ti"])
    assert [ring.size for ring in found] == [6]
    assert metal_ligand_rings(hexagon, result, max_size=5, metal_elements=["Al", "Ti"]) == []
    assert all(ring.size <= MAX_RING_SIZE for ring in found)


def test_metal_ligand_rings_do_not_miss_the_larger_window_of_a_dense_cluster():
    # An M-O octagonal prism: every bond lies on one of the eight square faces, so a per-bond
    # "shortest ring through each edge" search never reports the two octagons. The squares alone
    # have rank 8 and the cyclomatic number is 9, so an octagon is needed for a cycle basis; the
    # full relevant set (Vismara 1997) is the 8 squares plus both octagons.
    structure = parse_xyz(str(Path(__file__).resolve().parent.parent / "fixtures" / "ring_prism.xyz"))
    result = analyze_structure(structure)
    rings = metal_ligand_rings(structure, result, metal_elements=["Al", "Ti"])
    assert len(rings) == 10
    assert ring_statistics(rings)["by_size"] == {4: 8, 8: 2}


def test_all_minimal_rings_of_a_fused_cage_are_counted():
    # A cubane: eight vertices, twelve edges, six four-membered faces, and every edge lies on two
    # of them. Keeping only the first ring BFS reaches per edge covers all twelve edges with four
    # faces and undercounts. This is the case the ring search must not get wrong.
    a = 1.0
    corners = [[0, 0, 0], [2 * a, 0, 0], [0, 2 * a, 0], [2 * a, 2 * a, 0],
               [0, 0, 2 * a], [2 * a, 0, 2 * a], [0, 2 * a, 2 * a], [2 * a, 2 * a, 2 * a]]
    atoms = ["Al", "O", "O", "Al", "O", "Al", "Al", "O"]
    cage = _structure(atoms, corners)
    result = analyze_structure(cage)
    rings = metal_ligand_rings(cage, result, metal_elements=["Al", "Ti"])
    assert len(rings) == 6
    assert {ring.size for ring in rings} == {4}


# ── cadmia structure compare ────────────────────────────────────────────────

def _write_xyz(path, atoms, positions) -> str:
    lines = [str(len(atoms)), "test"]
    lines += [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in zip(atoms, positions)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_compare_separates_a_deformation_from_a_rearrangement(tmp_path):
    from click.testing import CliRunner

    from cadmia.cli import main

    atoms = ["Al", "O", "Al", "O"]
    square = [[0.0, 0.0, 0.0], [1.3, 1.3, 0.0], [2.6, 0.0, 0.0], [1.3, -1.3, 0.0]]
    reference = _write_xyz(tmp_path / "ref.xyz", atoms, square)
    # Bent: every atom moves, but no bond is made or broken.
    bent = _write_xyz(tmp_path / "bent.xyz", atoms,
                      [[0.0, 0.0, 0.0], [1.3, 1.25, 0.25], [2.6, 0.0, 0.0], [1.3, -1.25, -0.25]])
    # Torn apart: one oxygen leaves, so the connectivity itself changes.
    torn = _write_xyz(tmp_path / "torn.xyz", atoms,
                      [[0.0, 0.0, 0.0], [1.3, 1.3, 0.0], [2.6, 0.0, 0.0], [1.3, -6.0, 0.0]])

    result = CliRunner().invoke(main, ["structure", "compare", reference, bent, torn, "--json"])
    assert result.exit_code == 0, result.output
    rows = {Path(row["file"]).name: row for row in json.loads(result.output)["comparisons"]}
    assert rows["bent.xyz"]["same_skeleton"] is True
    assert rows["bent.xyz"]["rmsd_ang"] > 0.0
    assert rows["torn.xyz"]["same_skeleton"] is False


def test_compare_refuses_a_different_composition(tmp_path):
    from click.testing import CliRunner

    from cadmia.cli import main

    reference = _write_xyz(tmp_path / "ref.xyz", ["Al", "O"], [[0, 0, 0], [1.8, 0, 0]])
    other = _write_xyz(tmp_path / "other.xyz", ["Ti", "O"], [[0, 0, 0], [1.8, 0, 0]])
    result = CliRunner().invoke(main, ["structure", "compare", reference, other, "--json"])
    assert result.exit_code == 0, result.output
    row = json.loads(result.output)["comparisons"][0]
    assert row["comparable"] is False
    assert "composition" in row["reason"]
