"""Tests for cadmia.structure.reports."""
from dataclasses import replace
from pathlib import Path
import csv
import json

from cadmia.core import Structure
from cadmia.io.xyz import parse_xyz
from cadmia.structure.reports import write_batch_report, write_report
from cadmia.structure.reports.generator import _ordered_batch_results
from cadmia.structure.structural.analyzer import analyze_structure

FIXTURES = Path(__file__).parent.parent / "fixtures"
N1_XYZ = FIXTURES / "1tial_orca" / "1tial_opt.xyz"


def _result(config):
    structure = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    return analyze_structure(structure, config)


def _with_structure(result, structure_id, atoms):
    labels = [f"{element}{index + 1}" for index, element in enumerate(atoms)]
    structure = Structure(
        atoms=atoms,
        positions=[[0.0, 0.0, 0.0] for _ in atoms],
        labels=labels,
        structure_id=structure_id,
        source_file=f"{structure_id}.xyz",
    )
    return replace(result, structure=structure)


def test_write_text_report(config, tmp_path):
    import cadmia

    path = write_report(_result(config), tmp_path / "report.txt", config=config)
    content = path.read_text(encoding="utf-8")
    assert f"# CADMIA structure {cadmia.__version__}" in content
    assert "Pablo Daniel Alvarez Hernandez" in content
    assert "Universidad Juarez Autonoma de Tabasco" in content
    assert "CADMIA Structural Report" in content
    assert "=" * 80 + "\nCADMIA Structural Report\n" + "=" * 80 in content
    assert "=" * 80 + "\nLocal Descriptors\n" + "=" * 80 in content
    assert "Concepts and Parameter Definitions" in content
    assert "Analysis Parameters" in content
    assert "Bond cutoffs (max. distance counted as a bond" in content
    assert "Ti-O" in content and "A" in content.split("Bond cutoffs", 1)[1]
    assert "d_av" in content
    assert "MeanAux" in content
    assert "ECN_1d" in content
    assert "ECN_act" in content
    assert "\033[" not in content
    assert "(" in content
    assert "Ti-O" in content


def test_text_report_rings_declare_cyclomatic_completeness(tmp_path):
    structure = parse_xyz(N1_XYZ)
    result = analyze_structure(structure)
    path = write_report(result, tmp_path / "report.txt")
    content = path.read_text(encoding="utf-8")
    assert "Cyclomatic number (mu = E - V + C): 3" in content
    assert "Rank of the rings below: 3 (complete)" in content
    assert "3 ring(s): 3 of size 4." in content
    assert "(n-MR, n = size/2): 2-MR: 3." in content


def test_cation_neighborhood_counts_isolated_centers_as_zero(tmp_path):
    structure = parse_xyz(FIXTURES / "ring_prism.xyz")
    result = analyze_structure(structure)
    content = write_report(result, tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "Ti-Ti      2         0.000            2/2" in content
    assert "Al-Ti      6         1.333            0/6" in content
    assert "Ti14        Ti-Ti      0           4.96 A" in content
    payload = json.loads(write_report(result, tmp_path / "report.json").read_text(encoding="utf-8"))
    block = payload["cation_neighbors"]
    assert block["summary"]["Ti-Ti"]["isolated"] == 2
    assert len(block["bridges"]) == 16


def test_csv_carries_cation_neighborhood_and_rings(tmp_path):
    result = analyze_structure(parse_xyz(N1_XYZ))
    path = write_report(result, tmp_path / "report.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_type = {}
    for row in rows:
        by_type.setdefault(row["record_type"], []).append(row)
    ti_al = next(r for r in by_type["cation_neighbors"] if r["bond_type"] == "Ti-Al")
    assert float(ti_al["neighbors"]) == 2.0 and ti_al["isolated_count"] == "0"
    assert len(by_type["cation_bridge"]) == 3
    assert len(by_type["ring"]) == 3
    topology = by_type["ring_topology"][0]
    assert (topology["cyclomatic_number"], topology["ring_complete"]) == ("3", "True")


def test_config_metals_reach_neighborhood_and_rings(tmp_path):
    """The metals of a --config reach the rings and the neighbourhood, not only the header."""
    import copy
    from cadmia.structure.core.config import load_config
    from cadmia.structure.structural.morphology import metal_ligand_ring_analysis
    from cadmia.structure.structural.polyhedra import cation_neighbors

    cfg = copy.deepcopy(load_config())
    cfg.metal_elements = ["Al"]
    structure = parse_xyz(N1_XYZ)
    result = analyze_structure(structure, cfg)
    _, summary = cation_neighbors(structure, result)
    assert set(summary) == {"Al-Al"}
    _, topology = metal_ligand_ring_analysis(structure, result)
    assert topology.cyclomatic_number != 3        # 3 with the default metals (Ti and Al)


def _toy(atoms, positions):
    labels = [f"{a}{i + 1}" for i, a in enumerate(atoms)]
    return Structure(atoms=atoms, positions=positions, labels=labels, structure_id="toy",
                     source_file="toy.xyz")


def test_neighborhood_and_rings_with_no_metal_one_metal_and_two_fragments():
    from cadmia.structure.structural.morphology import metal_ligand_ring_analysis
    from cadmia.structure.structural.polyhedra import cation_neighbors

    no_metal = _toy(["O", "O"], [[0, 0, 0], [1.2, 0, 0]])
    one_metal = _toy(["Ti", "O", "O"], [[0, 0, 0], [1.65, 0, 0], [-1.65, 0, 0]])
    # two Ti2O2 rhombi 10 Å apart: one ring each, two components
    rhomb = [[0, 0, 0], [2.8, 0, 0], [1.4, 1.2, 0], [1.4, -1.2, 0]]
    two = _toy(["Ti", "Ti", "O", "O"] * 2,
               rhomb + [[x + 10.0, y, z] for x, y, z in rhomb])
    for structure, n_rings in ((no_metal, 0), (one_metal, 0), (two, 2)):
        result = analyze_structure(structure)
        bridges, summary = cation_neighbors(structure, result)
        rings, topology = metal_ligand_ring_analysis(structure, result)
        assert len(rings) == n_rings and topology.cyclomatic_number == n_rings
        assert topology.is_complete
    assert summary["Ti-Ti"]["mean_neighbors"] == 1.0 and summary["Ti-Ti"]["isolated"] == 0


def test_report_declares_the_point_group_detected_from_the_geometry(tmp_path):
    result = analyze_structure(parse_xyz(N1_XYZ))
    content = write_report(result, tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "Point group: Cs from the geometry (tolerance 0.01 A); symmetry number 1; achiral" in content
    payload = json.loads(write_report(result, tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["symmetry"]["point_group"] == "Cs"


def test_write_csv_report(config, tmp_path):
    path = write_report(_result(config), tmp_path / "report.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert any(row["record_type"] == "bond" for row in rows)
    assert any(row["record_type"] == "local_descriptor" for row in rows)
    assert any(row["record_type"] == "directed_bond_statistics" for row in rows)


def test_write_json_report(config, tmp_path):
    path = write_report(_result(config), tmp_path / "report.json", config=config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["structure"]["formula"] == "O2Ti"
    assert payload["structure"]["structure_id"]
    assert len(payload["bonds"]) == 2
    assert payload["directed_bond_statistics"]["Ti-O"]["bond_count"] == 2
    assert payload["local_descriptors"][0]["ecn_by_element"]


def _cluster_result(config):
    structure = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    return analyze_structure(structure, config)


def test_text_report_includes_polyhedra_angles(config, tmp_path):
    path = write_report(_cluster_result(config), tmp_path / "report.txt", config=config)
    content = path.read_text(encoding="utf-8")
    assert "Coordination Polyhedra (angles and distortion)" in content
    assert "angle_mean" in content
    # The Al2TiO5 cluster has metal centers with >= 3 ligands, so an angle is printed.
    assert "No metal center" not in content


def test_json_report_includes_polyhedra(config, tmp_path):
    path = write_report(_cluster_result(config), tmp_path / "report.json", config=config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["polyhedra"]
    for poly in payload["polyhedra"]:
        assert "mean_angle" in poly and "angle_distortion" in poly
        assert 0.0 < poly["mean_angle"] < 180.0


def test_csv_report_includes_polyhedron_rows(config, tmp_path):
    path = write_report(_cluster_result(config), tmp_path / "report.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    poly_rows = [row for row in rows if row["record_type"] == "polyhedron"]
    assert poly_rows
    assert all(row["mean_angle"] for row in poly_rows)


def test_csv_report_includes_metal_displacement_ratio(config, tmp_path):
    # The comparable (size-normalized) off-centering must reach the CSV, not only the JSON.
    path = write_report(_cluster_result(config), tmp_path / "report.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "metal_displacement_ratio" in reader.fieldnames
        poly_rows = [row for row in reader if row["record_type"] == "polyhedron"]
    assert poly_rows
    assert all(row["metal_displacement_ratio"] for row in poly_rows)


def test_text_report_polyhedra_header_has_displacement_ratio(config, tmp_path):
    path = write_report(_cluster_result(config), tmp_path / "report.txt", config=config)
    content = path.read_text(encoding="utf-8")
    assert "disp_r" in content


def test_text_report_no_polyhedra_for_small_molecule(config, tmp_path):
    # TiO2 monomer: Ti has only 2 O neighbors (CN 2 < 3) -> no polyhedron.
    path = write_report(_result(config), tmp_path / "report.txt", config=config)
    content = path.read_text(encoding="utf-8")
    assert "Coordination Polyhedra (angles and distortion)" in content
    assert "No metal center with 3 or more ligands." in content


def test_write_batch_text_report_is_global_by_default(config, tmp_path):
    results = [
        analyze_structure(parse_xyz(FIXTURES / "TiO2_monomer.xyz"), config),
        analyze_structure(parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz"), config),
    ]
    path = write_batch_report(results, tmp_path / "batch.txt", config=config)
    content = path.read_text(encoding="utf-8")
    assert "CADMIA Batch Structural Summary" in content
    assert (
        "=" * 80 + "\nCADMIA Batch Structural Summary\n" + "=" * 80
        in content
    )
    assert "# CADMIA structure" in content
    assert "Pablo Daniel Alvarez Hernandez" in content
    assert "Pair Summary by Structure" in content
    lines = content.splitlines()
    header_index = next(
        i for i, line in enumerate(lines)
        if "Structure" in line and "MeanAux" in line
    )
    assert lines[header_index + 1] == "-" * len(lines[header_index])
    assert "ECN_1d" in content
    assert "\033[" not in content
    assert content.count("Concepts and Parameter Definitions") == 1
    assert "CADMIA Structural Report" not in content


def test_batch_results_are_grouped_and_ordered_by_atom_count(config):
    base = _result(config)
    results = [
        _with_structure(base, "al2o3_2", ["Al"] * 4 + ["O"] * 6),
        _with_structure(base, "al2o3", ["Al"] * 2 + ["O"] * 3),
        _with_structure(base, "tio2_2", ["Ti"] * 2 + ["O"] * 4),
        _with_structure(base, "tio2", ["Ti"] + ["O"] * 2),
        _with_structure(
            base,
            "titania2alumina2a",
            ["Al"] * 4 + ["O"] * 10 + ["Ti"] * 2,
        ),
        _with_structure(
            base,
            "titania1alumina1a",
            ["Al"] * 2 + ["O"] * 5 + ["Ti"],
        ),
    ]

    ordered_ids = [
        result.structure.structure_id
        for result in _ordered_batch_results(results)
    ]

    assert ordered_ids == [
        "al2o3",
        "al2o3_2",
        "tio2",
        "tio2_2",
        "titania1alumina1a",
        "titania2alumina2a",
    ]


def test_batch_text_report_uses_present_elements_only(config, tmp_path):
    results = [analyze_structure(parse_xyz(FIXTURES / "Al2O3_cluster.xyz"), config)]
    path = write_batch_report(results, tmp_path / "al_only_batch.txt", config=config)
    content = path.read_text(encoding="utf-8")
    assert "Al-Al" in content
    assert "Al-O" in content
    assert "Ti-Ti" not in content
    assert "Ti-O" not in content


def test_write_batch_text_report_can_include_details(config, tmp_path):
    results = [
        analyze_structure(parse_xyz(FIXTURES / "TiO2_monomer.xyz"), config),
        analyze_structure(parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz"), config),
    ]
    path = write_batch_report(
        results,
        tmp_path / "batch_details.txt",
        config=config,
        include_details=True,
    )
    content = path.read_text(encoding="utf-8")
    assert "CADMIA Batch Structural Summary" in content
    assert content.count("CADMIA Structural Report") == 2
    assert content.count("Concepts and Parameter Definitions") == 1
