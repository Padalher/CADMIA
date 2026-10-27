"""Typed report generation."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import math
from math import gcd
from pathlib import Path
from typing import Iterable, Optional

from cadmia.io.results import fit_width, shown_path
from cadmia.structure.core.config import Config, load_config
from cadmia.structure.structural.analyzer import StructuralResult
from cadmia.structure.structural.angles import (angle_statistics, bond_angles,
                                                 bridging_angles)
from cadmia.structure.structural.morphology import (MAX_RING_SIZE, coordination_outliers, global_shape,
                                                    metal_ligand_ring_analysis, ring_statistics)
from cadmia.structure.structural.symmetry import detect_symmetry
from cadmia.structure.structural.polyhedra import (Polyhedron, analyze_polyhedra,
                                                   cation_neighbors, connectivity_summary,
                                                   polyhedra_connectivity)

TARGET_DIRECTED_PAIRS = ["Ti-Ti", "Ti-Al", "Ti-O", "Al-Al", "Al-Ti", "Al-O"]
REPORT_RULE_WIDTH = 80

def _program_header_lines() -> list[str]:
    import cadmia
    return [
        "# ======================================================",
        f"# CADMIA structure {cadmia.__version__}",
        "# Structural analysis for coordinate files",
        "#",
        "# Author:",
        "# Pablo Daniel Alvarez Hernandez",
        "#",
        "# Institution:",
        "# Division Academica de Ciencias Basicas (DACB)",
        "# Universidad Juarez Autonoma de Tabasco (UJAT)",
        "# ======================================================",
    ]


def _header_block(title: str, width: int = REPORT_RULE_WIDTH) -> list[str]:
    rule = "=" * width
    return [rule, title, rule]


PARAMETER_DEFINITIONS = [
    (
        "CN",
        "Coordination number from cutoff-filtered bonded neighbors. It counts EVERY "
        "neighbor, metals included.",
    ),
    (
        "CN_L",
        "Ligand count of a coordination polyhedron, which is NOT the same as CN: a "
        "polyhedron is metal-ligand by definition, so metal neighbors are excluded. A Ti "
        "bonded to 4 O and 2 Al has CN=6 and CN_L=4.",
    ),
    (
        "shape / CShM",
        "Closest reference polyhedron and its continuous shape measure (Pinsky & Avnir): "
        "the minimum normalized distance to that shape over rotation, scale and the "
        "ligand-to-vertex assignment, with the central atom included in the point set. 0 "
        "is an exact match and larger is more distorted. Unlike angle_dist and lambda it "
        "is defined for every coordination environment, not only the tabulated ideals.",
    ),
    (
        "Sharing",
        "How two polyhedra are joined, by shared ligands: corner (1), edge (2), face (3 "
        "or more).",
    ),
    (
        "ECN",
        "Hoppe self-consistent effective coordination number; it is the sum "
        "of distance weights around an atom or within a pair environment.",
    ),
    (
        "d_av",
        "Principal average bond length. It is the Hoppe "
        "self-consistent effective distance computed per central atom and "
        "neighbor element, then averaged over active centers for the "
        "interaction type.",
    ),
    (
        "ECN_act",
        "Mean pair ECN over the central atoms with a nonzero ECN toward the "
        "neighbor element. ECN has no distance cutoff, so that is every central "
        "atom, isolated ones included: for metal-metal pairs read it with its "
        "d_av, and take metal-metal coordination from the Cation Neighborhood. "
        "It is not averaged over the same atoms as 'active' (CN > 0) printed beside it.",
    ),
    (
        "ECN_1d",
        "One-decimal ECN display value rounded with Decimal ROUND_HALF_UP. "
        "Raw ECN values are still reported and used for calculations.",
    ),
    (
        "Bonds",
        "Number of unique cutoff-filtered bonds of that pair. Homonuclear "
        "pairs are not double-counted.",
    ),
    (
        "DirB",
        "Directed central-neighbor incidences used for CN/atom. Homonuclear "
        "pairs count both directions, so DirB is twice Bonds for "
        "homonuclear pairs.",
    ),
    (
        "MeanAux",
        "Auxiliary arithmetic mean of cutoff-filtered bond distances. It is "
        "kept for comparison but is not the principal average bond length.",
    ),
]


def _round_half_up(value: float, decimals: int = 1) -> float:
    quant = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(float(value))).quantize(quant, rounding=ROUND_HALF_UP))


def _ecn_rounded(value: float, decimals: int = 1) -> str:
    return f"{_round_half_up(value, decimals):.{decimals}f}"


def _ecn_with_rounded(value: float, decimals: int = 1) -> str:
    return f"{value:.3f} ({_ecn_rounded(value, decimals)})"


def _distance_rounded(value: float, decimals: int = 2) -> str:
    return f"{_round_half_up(value, decimals):.{decimals}f}"


def _d_av_field(value: float, width: int) -> str:
    return f"{_distance_rounded(value):<{width}}"


def _d_av_value(value: float, width: int = 0) -> str:
    text = _distance_rounded(value)
    if width:
        text = f"{text:>{width}}"
    return text


def _ecn_1d_field(value: float, width: int) -> str:
    return f"{_ecn_rounded(value):<{width}}"


def _molecular_weight(result: StructuralResult, config: Config) -> float:
    return sum(config.get_atomic_mass(atom) for atom in result.structure.atoms)


def _polyhedra(result: StructuralResult) -> list[Polyhedron]:
    """Coordination polyhedra (metal centers with >= 3 ligands) for angle reporting."""
    return analyze_polyhedra(result.structure, result)


def _connectivity_block(result: StructuralResult, polyhedra=None) -> dict:
    """Polyhedra sharing for the JSON report: the per-pair links plus their summary.

    Takes the polyhedra when the caller already has them. Recomputing costs a full continuous
    shape measure per centre, which is N! superpositions each.
    """
    polyhedra = _polyhedra(result) if polyhedra is None else polyhedra
    links = polyhedra_connectivity(polyhedra, result.structure)
    return {
        "summary": connectivity_summary(links, polyhedra),
        "links": [asdict(link) for link in links],
    }


def _cation_neighbor_block(result: StructuralResult) -> dict:
    bridges, summary = cation_neighbors(result.structure, result)
    return {"summary": summary, "bridges": [asdict(bridge) for bridge in bridges]}


def _symmetry_line(result: StructuralResult) -> str:
    sym = detect_symmetry(result.structure.atoms, result.structure.positions)
    return (f"Point group: {sym.point_group} from the geometry (tolerance {sym.tolerance} A); "
            f"symmetry number {sym.sigma}; {'chiral' if sym.chiral else 'achiral'}")


def _structure_blocks(result: StructuralResult) -> dict:
    """The per-structure JSON sections, with every expensive analysis evaluated once."""
    polyhedra = _polyhedra(result)
    return {
        "polyhedra": [asdict(poly) for poly in polyhedra],
        "polyhedra_connectivity": _connectivity_block(result, polyhedra),
        "cation_neighbors": _cation_neighbor_block(result),
        "bond_angles": _angle_block(result),
        "rings": _ring_block(result),
        "global_shape": asdict(global_shape(result.structure)),
        "symmetry": asdict(detect_symmetry(result.structure.atoms, result.structure.positions)),
        "coordination_outliers": coordination_outliers(result),
    }


def _is_metal(result: StructuralResult, index: int) -> bool:
    from cadmia.structure.core.config import load_config
    return result.structure.atoms[index] in set(result.metal_elements or load_config().metal_elements)


def _number(value: float, decimals: int = 1) -> str:
    """Format a float for a fixed-width column, with '-' for the ones that are not defined."""
    return "-" if value != value else f"{value:.{decimals}f}"


def _json_safe(value):
    """Replace NaN and infinities with null, recursively.

    ``json.dumps`` writes them as the bare tokens NaN and Infinity, which no strict JSON reader
    accepts: jq, most browsers and any other consumer reject the file. null is the representation
    of "not defined" that survives the round trip.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _angle_block(result: StructuralResult) -> dict:
    """Bond angles for the JSON report: the full list, the per-triplet stats and the bridges."""
    angles = bond_angles(result.structure, result)
    return {
        "statistics": angle_statistics(angles),
        "angles": [asdict(item) for item in angles],
        "bridging": [asdict(item) for item in bridging_angles(result.structure, result)],
    }


def _ring_block(result: StructuralResult) -> dict:
    """Rings of the metal-ligand subgraph, which is where the framework motif lives."""
    rings, topology = metal_ligand_ring_analysis(result.structure, result)
    return {"statistics": ring_statistics(rings), "topology": asdict(topology),
            "rings": [asdict(ring) for ring in rings]}


def _bond_rows(result: StructuralResult) -> list[dict]:
    return [
        {
            "record_type": "bond",
            "structure_id": result.structure.structure_id,
            "atom_i": bond.label_i,
            "atom_j": bond.label_j,
            "bond_type": bond.bond_type,
            "distance": f"{bond.distance:.8f}",
            "element": "",
            "cn": "",
            "ecn": "",
            "average_d_av": "",
            "avg_bond_distance": "",
            "effective_avg_bond_distance": "",
            "min_bond_distance": "",
            "max_bond_distance": "",
            "std_bond_distance": "",
            "bonded_elements": "",
            "ecn_by_element": "",
            "avg_distance_by_element": "",
            "effective_avg_distance_by_element": "",
            "bond_type_counts": "",
            "central_element": "",
            "neighbor_element": "",
            "central_atom_count": "",
            "active_atom_count": "",
            "bond_count": "",
            "directed_bond_count": "",
            "mean": "",
            "std": "",
            "min": "",
            "max": "",
        }
        for bond in result.bonds
    ]


def _local_rows(result: StructuralResult) -> list[dict]:
    rows = []
    for descriptor in result.local_descriptors:
        rows.append(
            {
                "record_type": "local_descriptor",
                "structure_id": result.structure.structure_id,
                "atom_i": descriptor.label,
                "atom_j": "",
                "bond_type": "",
                "distance": "",
                "element": descriptor.element,
                "cn": descriptor.cn,
                "ecn": f"{descriptor.ecn:.8f}",
                "average_d_av": "",
                "avg_bond_distance": f"{descriptor.avg_bond_distance:.8f}",
                "effective_avg_bond_distance": (
                    f"{descriptor.effective_avg_bond_distance:.8f}"
                ),
                "min_bond_distance": f"{descriptor.min_bond_distance:.8f}",
                "max_bond_distance": f"{descriptor.max_bond_distance:.8f}",
                "std_bond_distance": f"{descriptor.std_bond_distance:.8f}",
                "bonded_elements": json.dumps(
                    descriptor.bonded_elements,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "ecn_by_element": json.dumps(
                    descriptor.ecn_by_element,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "avg_distance_by_element": json.dumps(
                    descriptor.avg_distance_by_element,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "effective_avg_distance_by_element": json.dumps(
                    descriptor.effective_avg_distance_by_element,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "bond_type_counts": json.dumps(
                    descriptor.bond_type_counts,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            }
        )
    return rows


def _stat_rows(result: StructuralResult) -> list[dict]:
    rows = []
    for bond_type, stats in sorted(result.bond_statistics.items()):
        rows.append(
            {
                "record_type": "bond_statistics",
                "structure_id": result.structure.structure_id,
                "bond_type": bond_type,
                "bond_count": stats["count"],
                "average_d_av": "",
                "mean": f"{stats['mean']:.8f}",
                "std": f"{stats['std']:.8f}",
                "min": f"{stats['min']:.8f}",
                "max": f"{stats['max']:.8f}",
            }
        )

    for element, stats in sorted(result.ecn_statistics.items()):
        rows.append(
            {
                "record_type": "ecn_statistics",
                "structure_id": result.structure.structure_id,
                "element": element,
                "bond_count": stats["count"],
                "average_d_av": "",
                "mean": f"{stats['mean']:.8f}",
                "std": f"{stats['std']:.8f}",
                "min": f"{stats['min']:.8f}",
                "max": f"{stats['max']:.8f}",
            }
        )

    for pair_type, stats in sorted(result.directed_bond_statistics.items()):
        rows.append(
            {
                "record_type": "directed_bond_statistics",
                "structure_id": result.structure.structure_id,
                "bond_type": pair_type,
                "central_element": stats["central_element"],
                "neighbor_element": stats["neighbor_element"],
                "central_atom_count": stats["central_atom_count"],
                "active_atom_count": stats["active_atom_count"],
                "bond_count": stats["bond_count"],
                "directed_bond_count": stats["directed_bond_count"],
                "cn": f"{stats['mean_cn_per_atom']:.8f}",
                "ecn": f"{stats['mean_ecn_active_atoms']:.8f}",
                "average_d_av": f"{stats['average_d_av']:.8f}",
                "effective_avg_bond_distance": (
                    f"{stats['mean_atom_effective_avg_distance']:.8f}"
                ),
                "avg_bond_distance": f"{stats['mean_distance']:.8f}",
                "std": f"{stats['std_distance']:.8f}",
                "min": f"{stats['min_distance']:.8f}",
                "max": f"{stats['max_distance']:.8f}",
            }
        )
    return rows


def _polyhedra_rows(result: StructuralResult) -> list[dict]:
    rows = []
    for poly in _polyhedra(result):
        rows.append(
            {
                "record_type": "polyhedron",
                "structure_id": result.structure.structure_id,
                "atom_i": poly.center_label,
                "element": poly.center_element,
                "neighbor_element": poly.neighbor_element,
                "cn": poly.cn,
                "polyhedron_type": poly.polyhedron_type,
                "mean_bond_length": f"{poly.mean_bond_length:.8f}",
                "bond_length_distortion": f"{poly.bond_length_distortion:.8f}",
                "metal_displacement": f"{poly.metal_displacement:.8f}",
                "metal_displacement_ratio": f"{poly.metal_displacement_ratio:.8f}",
                "mean_angle": f"{poly.mean_angle:.8f}",
                "angle_distortion": f"{poly.angle_distortion:.8f}",
                "volume": f"{poly.volume:.8f}",
                "quadratic_elongation": f"{poly.quadratic_elongation:.8f}",
                "shape": poly.shape,
                "shape_measure": f"{poly.shape_measure:.8f}",
            }
        )
    return rows


def _cation_neighbor_rows(result: StructuralResult) -> list[dict]:
    structure_id = result.structure.structure_id
    bridges, summary = cation_neighbors(result.structure, result)
    rows = []
    for pair, row in summary.items():
        central, neighbor = pair.split("-", 1)
        rows.append({
            "record_type": "cation_neighbors",
            "structure_id": structure_id,
            "bond_type": pair,
            "central_element": central,
            "neighbor_element": neighbor,
            "central_atom_count": row["centers"],
            "neighbors": f"{row['mean_neighbors']:.8f}",
            "isolated_count": row["isolated"],
            "nearest_distance": f"{row['mean_nearest_distance']:.8f}",
        })
        for center in row["per_center"]:
            rows.append({
                "record_type": "cation_neighbor_center",
                "structure_id": structure_id,
                "atom_i": center["label"],
                "bond_type": pair,
                "element": central,
                "neighbor_element": neighbor,
                "neighbors": center["neighbors"],
                "nearest_distance": f"{center['nearest_distance']:.8f}",
            })
    for bridge in bridges:
        rows.append({
            "record_type": "cation_bridge",
            "structure_id": structure_id,
            "atom_i": bridge.label_a,
            "atom_j": bridge.label_b,
            "distance": f"{bridge.distance:.8f}",
            "sharing": bridge.kind,
            "shared_ligands": ";".join(bridge.shared_labels),
            "shared_mu": ";".join(str(mu) for mu in bridge.shared_mu),
        })
    return rows


def _ring_rows(result: StructuralResult) -> list[dict]:
    structure_id = result.structure.structure_id
    rings, topology = metal_ligand_ring_analysis(result.structure, result)
    rows = [{
        "record_type": "ring_topology",
        "structure_id": structure_id,
        "cyclomatic_number": topology.cyclomatic_number,
        "ring_rank": topology.rank,
        "ring_complete": topology.is_complete,
        "bond_count": topology.n_edges,
        "central_atom_count": topology.n_vertices,
    }]
    rows.extend({
        "record_type": "ring",
        "structure_id": structure_id,
        "ring_size": ring.size,
        "composition": ring.composition,
        "ring_atoms": "-".join(ring.atom_labels),
    } for ring in rings)
    return rows


def _csv_rows(result: StructuralResult) -> list[dict]:
    return (_bond_rows(result) + _local_rows(result) + _stat_rows(result)
            + _polyhedra_rows(result) + _cation_neighbor_rows(result) + _ring_rows(result))


def _csv_fieldnames() -> list[str]:
    return [
        "record_type",
        "structure_id",
        "atom_i",
        "atom_j",
        "bond_type",
        "distance",
        "element",
        "cn",
        "ecn",
        "average_d_av",
        "effective_avg_bond_distance",
        "avg_bond_distance",
        "min_bond_distance",
        "max_bond_distance",
        "std_bond_distance",
        "bonded_elements",
        "ecn_by_element",
        "avg_distance_by_element",
        "effective_avg_distance_by_element",
        "bond_type_counts",
        "central_element",
        "neighbor_element",
        "central_atom_count",
        "active_atom_count",
        "bond_count",
        "directed_bond_count",
        "mean",
        "std",
        "min",
        "max",
        "polyhedron_type",
        "mean_bond_length",
        "bond_length_distortion",
        "metal_displacement",
        "metal_displacement_ratio",
        "mean_angle",
        "angle_distortion",
        "volume",
        "quadratic_elongation",
        "shape",
        "shape_measure",
        "neighbors",
        "isolated_count",
        "nearest_distance",
        "sharing",
        "shared_ligands",
        "shared_mu",
        "cyclomatic_number",
        "ring_rank",
        "ring_complete",
        "ring_size",
        "composition",
        "ring_atoms",
    ]


def _text_report_lines(
    result: StructuralResult,
    config: Config,
    include_concepts: bool = True,
    include_program_header: bool = True,
) -> list[str]:
    cfg = config
    now = datetime.now()

    lines = [
        *(_program_header_lines() + [""] if include_program_header else []),
        *_header_block("CADMIA Structural Report"),
        f"Date: {now:%Y-%m-%d}",
        f"Time: {now:%H:%M:%S}",
        f"Structure ID: {result.structure.structure_id}",
        f"Source file: {shown_path(result.structure.source_file)}",
        f"Formula: {result.structure.formula}",
        f"Atoms: {result.structure.n_atoms}",
        f"Molecular weight: {_molecular_weight(result, cfg):.3f} g/mol",
        _symmetry_line(result),
        "",
        *_header_block("Analysis Parameters"),
    ]
    metals_present = sorted(set(result.structure.atoms) & set(cfg.metal_elements))
    lines.append(f"Metal elements present: {', '.join(metals_present) if metals_present else '(none)'}")
    lines.append("Bond cutoffs (max. distance counted as a bond, per pair present):")
    for bond_type in sorted(result.bond_statistics):
        element_i, element_j = bond_type.split("-", 1)
        lines.append(f"  {bond_type:<9} {_distance_rounded(cfg.get_bond_cutoff(element_i, element_j))} A")
    ecn_params = cfg.ecn_parameters
    lines.append(
        "Hoppe ECN: every atom pair, no distance cutoff; weight exp[1 - (d/d_av)^"
        f"{float(ecn_params.get('damping_factor', 6.0)):g}], d_av self-consistent from d_min, "
        f"tolerance {float(ecn_params.get('convergence_threshold', 1e-6)):g} A, "
        f"at most {int(ecn_params.get('max_iterations', 100))} iterations."
    )
    if include_concepts:
        lines.extend(["", *_concept_lines()])
    lines.extend(
        [
            "",
            *_header_block("Local Descriptors"),
            "Atom        Elem    CN      ECN       ECN_1d   d_av       MeanAux      Neighbors",
        ]
    )

    for descriptor in result.local_descriptors:
        neighbors = ", ".join(
            f"{element}:{count}"
            for element, count in sorted(descriptor.bonded_elements.items())
        ) or "-"
        lines.append(
            f"{descriptor.label:<11}{descriptor.element:<8}"
            f"{descriptor.cn:<8}{descriptor.ecn:<10.3f}"
            f"{_ecn_1d_field(descriptor.ecn, 9)}"
            f"{_d_av_field(descriptor.effective_avg_bond_distance, 11)}"
            f"{_distance_rounded(descriptor.avg_bond_distance):<13}{neighbors}"
        )

    lines.extend(["", *_header_block("Per-Atom ECN by Neighbor Element")])
    for descriptor in result.local_descriptors:
        if not descriptor.ecn_by_element:
            lines.append(f"{descriptor.label:<11}{descriptor.element:<8} -")
            continue
        for neighbor_element, ecn in sorted(descriptor.ecn_by_element.items()):
            avg = descriptor.avg_distance_by_element[neighbor_element]
            eff_avg = descriptor.effective_avg_distance_by_element[neighbor_element]
            cn = descriptor.bonded_elements.get(neighbor_element, 0)
            lines.append(
                f"{descriptor.label:<11}{descriptor.element}-{neighbor_element:<8}"
                f" CN={cn:<3} ECN={_ecn_with_rounded(ecn):>14} "
                f"d_av={_d_av_value(eff_avg, 8)} A "
                f"MeanAux={_distance_rounded(avg):>8} A"
            )

    lines.extend(
        ["", *_header_block("Bond Distance Statistics (Arithmetic, auxiliary)")]
    )
    for bond_type, stats in sorted(result.bond_statistics.items()):
        lines.append(
            f"{bond_type:<12} count={stats['count']} "
            f"MeanAux={_distance_rounded(stats['mean'])} "
            f"std={_distance_rounded(stats['std'])} "
            f"min={_distance_rounded(stats['min'])} "
            f"max={_distance_rounded(stats['max'])}"
        )

    lines.extend(["", *_header_block("Directed Pair Statistics (Central-Neighbor)")])
    lines.append("Pair      Centers Active Bonds Directed CN/atom  ECN_act (1d)  d_av (A) "
                 "MeanAux (A)")
    for pair_type, stats in sorted(result.directed_bond_statistics.items()):
        lines.append(
            f"{pair_type:<9}{stats['central_atom_count']:>7}{stats['active_atom_count']:>7}"
            f"{stats['bond_count']:>6}{stats['directed_bond_count']:>9}"
            f"{stats['mean_cn_per_atom']:>8.3f}  "
            f"{_ecn_with_rounded(stats['mean_ecn_active_atoms']):<14}"
            f"{_d_av_value(stats['mean_atom_effective_avg_distance']):>8}"
            f"{_distance_rounded(stats['mean_distance']):>12}"
        )

    polyhedra = _polyhedra(result)
    lines.extend(["", *_header_block("Coordination Polyhedra (angles and distortion)")])
    if not polyhedra:
        lines.append("No metal center with 3 or more ligands.")
    else:
        lines.append(
            "Center  CN_L Type                d_mean angle_mean angle_dist  volume  lambda"
            "  disp_r shape    CShM"
        )
        for poly in polyhedra:
            lines.append(
                f"{poly.center_label:<8}{poly.cn:<5}{poly.polyhedron_type:<19}"
                f"{_distance_rounded(poly.mean_bond_length):>7}"
                f"{poly.mean_angle:>11.1f}{poly.angle_distortion:>11.1f}"
                f"{poly.volume:>8.3f}{poly.quadratic_elongation:>8.4f}"
                f"{poly.metal_displacement_ratio:>8.4f} "
                f"{poly.shape:<9}{poly.shape_measure:.2f}"
            )

    links = polyhedra_connectivity(polyhedra, result.structure)
    lines.extend(["", *_header_block("Polyhedra Connectivity (shared ligands)")])
    if not polyhedra:
        lines.append("No polyhedra to connect.")
    elif not links:
        lines.append("No two polyhedra share a ligand: every center is isolated.")
    else:
        summary = connectivity_summary(links, polyhedra)
        kinds = summary["by_kind"]
        lines.append(
            f"{summary['links']} link(s) between {summary['polyhedra']} polyhedra: "
            f"{kinds['corner']} corner, {kinds['edge']} edge, {kinds['face']} face."
        )
        isolated = summary["isolated_centers"]
        lines.append("Isolated centers: " + (", ".join(isolated) if isolated else "none"))
        lines.append("Only metal centers with 3 or more ligands form a polyhedron, so a "
                     "two-coordinate edge atom is not counted here. Each linked pair, with the "
                     "mu of its shared ligands, is listed under Cation Neighborhood.")

    bridges, neighborhood = cation_neighbors(result.structure, result)
    lines.extend(["", *_header_block("Cation Neighborhood (via shared ligand)")])
    if not neighborhood:
        lines.append("Fewer than two metal atoms: no metal-metal neighborhood to report.")
    else:
        lines.append("Two metals are neighbors when one ligand is bonded to both, under the bond "
                     "cutoffs above.")
        lines.append("Means run over ALL centers of the element: an isolated center counts as "
                     "zero. Use this, not the Hoppe ECN, for metal-metal coordination: ECN has "
                     "no absolute distance scale and gives a nonzero value to a center that "
                     "shares no ligand with any metal of that element.")
        lines.append("")
        lines.append("Pair       Centers   Mean neighbors   Isolated   Mean nearest d(M-M)")
        for pair, row in neighborhood.items():
            lines.append(
                f"{pair:<11}{row['centers']:<10}{row['mean_neighbors']:<17.3f}"
                f"{str(row['isolated']) + '/' + str(row['centers']):<11}"
                f"{_distance_rounded(row['mean_nearest_distance'])} A"
            )
        lines.append("")
        lines.append("Center      Pair       Neighbors   Nearest d(M-M)")
        for pair, row in neighborhood.items():
            for center in row["per_center"]:
                lines.append(
                    f"{center['label']:<12}{pair:<11}{center['neighbors']:<12}"
                    f"{_distance_rounded(center['nearest_distance'])} A"
                )
        lines.append("")
        if bridges:
            lines.append("mu = number of metals bonded to the shared ligand. A mu_6 ligand "
                         "spreads its bond valence over six metals, so each bond to it is "
                         "weaker than to a mu_2 bridge.")
            lines.append("")
            lines.append("Pair              Sharing   d(M-M)   Shared ligands (mu)")
            for bridge in bridges:
                pair = f"{bridge.label_a}-{bridge.label_b}"
                shared = ", ".join(f"{label}(mu{mu})"
                                   for label, mu in zip(bridge.shared_labels, bridge.shared_mu))
                lines.append(f"{pair:<18}{bridge.kind:<10}"
                             f"{_distance_rounded(bridge.distance):<9}{shared}")
        else:
            lines.append("No ligand is bonded to two metals.")

    angles = bond_angles(result.structure, result)
    lines.extend(["", *_header_block("Bond Angles by Element Triplet")])
    if not angles:
        lines.append("No atom has two or more bonded neighbors.")
    else:
        stats = angle_statistics(angles)
        lines.append("Mean, Std, Min and Max describe the CIS angles only; trans pairs (above "
                     "150 degrees) are counted apart, because pooling them makes an undistorted "
                     "centre look distorted.")
        lines.append("")
        lines.append("Triplet        Count   Mean      Std      Min       Max      "
                     "nTrans  MeanTrans")
        for triplet, values in stats.items():
            lines.append(
                f"{triplet:<15}{values['count']:<8}{_number(values['mean']):<10}"
                f"{_number(values['std']):<9}{_number(values['min']):<10}"
                f"{_number(values['max']):<9}{values['n_trans']:<8}"
                f"{_number(values['mean_trans'])}"
            )
        contacts = sum(1 for item in angles
                       if _is_metal(result, item.left_index)
                       and _is_metal(result, item.vertex_index))
        if contacts:
            lines.append("")
            lines.append(f"{contacts} of these angles have a metal-metal contact as one arm. "
                         "Those contacts")
            lines.append("count as bonds under the distance cutoffs, but they are not "
                         "coordination angles.")

        bridges = bridging_angles(result.structure, result)
        lines.append("")
        if bridges:
            lines.append(
                f"Bridging metal-ligand-metal angles ({len(bridges)}): the hinge that lets two "
                "polyhedra"
            )
            lines.append("reorient without breaking a bond.")
            for item in sorted(bridges, key=lambda a: a.angle):
                pair = f"{item.left_label}-{item.vertex_label}-{item.right_label}"
                lines.append(f"  {pair:<24}{item.angle:.1f}")
        else:
            lines.append("No ligand bridges two metals.")

    rings, topology = metal_ligand_ring_analysis(result.structure, result)
    lines.extend(["", *_header_block("Rings (metal-ligand subgraph)")])
    completeness = ("complete" if topology.is_complete else
                    f"INCOMPLETE — rings longer than {MAX_RING_SIZE} atoms are not searched")
    lines.append(
        f"Cyclomatic number (mu = E - V + C): {topology.cyclomatic_number} "
        f"({topology.n_edges} edges, {topology.n_vertices} vertices, "
        f"{topology.n_components} component(s)). Rank of the rings below: "
        f"{topology.rank} ({completeness})."
    )
    if not rings:
        lines.append("The metal-ligand subgraph closes no ring.")
    else:
        stats = ring_statistics(rings)
        by_size = ", ".join(f"{count} of size {size}" for size, count in stats["by_size"].items())
        lines.append(f"{stats['total']} ring(s): {by_size}.")
        # The subgraph is bipartite (metal-ligand edges only), so every ring alternates and
        # holds size/2 metals: the n-MR count of zeolite and oxide-cluster nomenclature.
        by_mr = ", ".join(f"{size // 2}-MR: {count}" for size, count in stats["by_size"].items())
        lines.append(f"By metal atoms per ring (n-MR, n = size/2): {by_mr}.")
        composition = ", ".join(f"{name} x{count}"
                                for name, count in stats["by_composition"].items())
        lines.append(f"Compositions: {composition}")
        lines.append("")
        lines.append("Size   Composition   Atoms")
        for ring in rings:
            lines.append(f"{ring.size:<7}{ring.composition:<14}{'-'.join(ring.atom_labels)}")

    shape = global_shape(result.structure)
    lines.extend(["", *_header_block("Global Shape (gyration tensor)")])
    lines.append(f"Radius of gyration   {shape.radius_of_gyration:.3f} A")
    lines.append(
        "Principal moments    "
        + ", ".join(f"{value:.3f}" for value in shape.principal_moments)
        + " A^2"
    )
    lines.append(f"Asphericity          {shape.asphericity:.3f} A^2")
    lines.append(f"Acylindricity        {shape.acylindricity:.3f} A^2")
    lines.append(f"Shape anisotropy     {shape.anisotropy:.3f}   (0 spherical, 1 collinear)")
    lines.append(f"Max diameter         {shape.max_diameter:.3f} A")
    lines.append(f"Convex hull volume   {shape.hull_volume:.3f} A^3")

    outliers = coordination_outliers(result)
    lines.extend(["", *_header_block("Coordination Relative to the Element Mean")])
    if not outliers:
        lines.append("No atoms.")
    else:
        lines.append("The mean is the mean OF THIS STRUCTURE: a deviation of zero means "
                     "ordinary here,")
        lines.append("not ordinary in general. z is left blank below 4 atoms of the element, "
                     "where it")
        lines.append("is arithmetic rather than statistics.")
        lines.append("")
        lines.append("Atom        Elem    ECN      Elem mean   Deviation   z")
        for row in sorted(outliers, key=lambda item: item["deviation"]):
            z_value = row["z"]
            z_text = "-" if z_value != z_value else f"{z_value:.2f}"
            lines.append(
                f"{row['label']:<12}{row['element']:<8}{row['ecn']:<9.3f}"
                f"{row['element_mean_ecn']:<12.3f}{row['deviation']:<12.3f}{z_text}"
            )

    lines.extend(["", *_header_block("Individual Bonds")])
    for bond in sorted(result.bonds, key=lambda item: (item.bond_type, item.distance)):
        lines.append(
            f"{bond.bond_type:<12} {bond.label_i}-{bond.label_j}: "
            f"{_distance_rounded(bond.distance)} A"
        )
    return lines


def _concept_lines() -> list[str]:
    lines = _header_block("Concepts and Parameter Definitions")
    import textwrap
    for name, definition in PARAMETER_DEFINITIONS:
        # within the report's 80 columns, the term column kept clear
        lines += textwrap.wrap(f"{name:<15} {definition}", 80, subsequent_indent=" " * 16)
    return lines


def _formula_items(counter: Counter[str]) -> list[tuple[str, int]]:
    items = []
    for element in ("C", "H"):
        if element in counter:
            items.append((element, counter[element]))
    items.extend(
        (element, counter[element])
        for element in sorted(counter)
        if element not in ("C", "H")
    )
    return items


def _reduced_formula_key(result: StructuralResult) -> tuple[tuple[str, int], ...]:
    counter = Counter(result.structure.atoms)
    if not counter:
        return ()
    divisor = 0
    for count in counter.values():
        divisor = gcd(divisor, int(count))
    divisor = max(divisor, 1)
    return tuple((element, count // divisor) for element, count in _formula_items(counter))


def _ordered_batch_results(results: list[StructuralResult]) -> list[StructuralResult]:
    group_order: dict[tuple[tuple[str, int], ...], int] = {}
    ordered = []
    for index, result in enumerate(results):
        group_key = _reduced_formula_key(result)
        if group_key not in group_order:
            group_order[group_key] = len(group_order)
        ordered.append(
            (
                group_order[group_key],
                result.structure.n_atoms,
                index,
                result,
            )
        )
    return [result for _, _, _, result in sorted(ordered)]


def _batch_text_summary_lines(
    results: list[StructuralResult],
    config: Config,
) -> list[str]:
    results = _ordered_batch_results(results)
    pair_header = (
        f"{'Structure':<24}{'Formula':<14}{'Atoms':<7}{'Pair':<8}"
        f"{'Bonds':<8}{'DirB':<7}{'d_av':<7}"
        f"{'ECN_act':<9}{'ECN_1d':<8}{'MeanAux':<8}"
    )
    lines = [
        *_program_header_lines(),
        "",
        *_header_block("CADMIA Batch Structural Summary"),
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Structures analyzed: {len(results)}",
        "",
        *_header_block("Pair Summary by Structure"),
        pair_header,
        "-" * len(pair_header),
    ]

    for result in results:
        pair_names = [
            pair for pair in TARGET_DIRECTED_PAIRS
            if pair in result.directed_bond_statistics
        ]
        other_pairs = [
            pair for pair in sorted(result.directed_bond_statistics)
            if pair not in pair_names
        ]
        for pair in pair_names + other_pairs:
            stats = result.directed_bond_statistics[pair]
            lines.append(
                f"{result.structure.structure_id:<24}"
                f"{result.structure.formula:<14}"
                f"{result.structure.n_atoms:<7}"
                f"{pair:<8}"
                f"{stats['bond_count']:<8}"
                f"{stats['directed_bond_count']:<7}"
                f"{_d_av_field(stats['mean_atom_effective_avg_distance'], 7)}"
                f"{stats['mean_ecn_active_atoms']:<9.3f}"
                f"{_ecn_1d_field(stats['mean_ecn_active_atoms'], 8)}"
                f"{_distance_rounded(stats['mean_distance']):<8}"
            )

    lines.extend(
        [
            "",
            *_header_block("Structure Overview"),
            (
                f"{'Structure':<24}{'Formula':<14}{'Atoms':<7}"
                f"{'Weight':<12}{'ECN(mean)':<12}{'ECN_1d':<8}"
                f"{'d_av(mean)':<13}"
            ),
        ]
    )
    for result in results:
        local_stats = result.local_environment_statistics["all_atoms"]
        lines.append(
            f"{result.structure.structure_id:<24}"
            f"{result.structure.formula:<14}"
            f"{result.structure.n_atoms:<7}"
            f"{_molecular_weight(result, config):<12.3f}"
            f"{local_stats['ecn']['mean']:<12.3f}"
            f"{_ecn_1d_field(local_stats['ecn']['mean'], 8)}"
            f"{_d_av_field(local_stats['effective_avg_bond_distance']['mean'], 13)}"
        )
    lines.extend(["", *_concept_lines()])
    return lines


def write_text_report(
    result: StructuralResult,
    output_file: str | Path,
    config: Optional[Config] = None,
) -> Path:
    """Write a TXT report from a typed StructuralResult."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(structural_text(result, config) + "\n", encoding="utf-8")
    return path


def structural_text(result: StructuralResult, config: Optional[Config] = None,
                    program_header: bool = True) -> str:
    """The full text report of one structure; without the program header it reads as one
    section of a larger document."""
    return fit_width("\n".join(_text_report_lines(result, config or load_config(),
                                                   include_program_header=program_header)))


def write_csv_report(result: StructuralResult, output_file: str | Path) -> Path:
    """Write a CSV report with bond and local-descriptor rows."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _csv_rows(result)
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json_report(
    result: StructuralResult,
    output_file: str | Path,
    config: Optional[Config] = None,
) -> Path:
    """Write a JSON report from a typed StructuralResult."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**structural_payload(result, config),
               "report_text": structural_text(result, config, program_header=False)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
                    encoding="utf-8")
    return path


def structural_payload(result: StructuralResult, config: Optional[Config] = None) -> dict:
    """Every value of the structural report, as data (format ``cadmia-structural-report-v1``)."""
    cfg = config or load_config()
    structure_payload = result.structure.to_dict()
    structure_payload["formula"] = result.structure.formula
    payload = {
        "format": "cadmia-structural-report-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "structure": structure_payload,
        "molecular_weight": _molecular_weight(result, cfg),
        "bonds": [asdict(bond) for bond in result.bonds],
        "local_descriptors": [
            asdict(descriptor) for descriptor in result.local_descriptors
        ],
        "bond_statistics": result.bond_statistics,
        "ecn_statistics": result.ecn_statistics,
        "directed_bond_statistics": result.directed_bond_statistics,
        "local_environment_statistics": result.local_environment_statistics,
        **_structure_blocks(result),
    }
    return _json_safe(payload)


def write_batch_report(
    results: Iterable[StructuralResult],
    output_file: str | Path,
    output_format: Optional[str] = None,
    config: Optional[Config] = None,
    include_details: bool = False,
) -> Path:
    """Write one TXT, CSV, or JSON report containing multiple structures."""
    result_list = list(results)
    path = Path(output_file)
    fmt = (output_format or path.suffix.lstrip(".") or "txt").lower()
    cfg = config or load_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_results = _ordered_batch_results(result_list)

    if fmt == "txt":
        sections = ["\n".join(_batch_text_summary_lines(ordered_results, cfg))]
        if include_details:
            for result in ordered_results:
                sections.append(
                    "\n".join(
                        _text_report_lines(result, cfg, include_concepts=False)
                    )
                )
        path.write_text(fit_width("\n\n".join(sections)) + "\n", encoding="utf-8")
        return path

    if fmt == "csv":
        rows = []
        for result in ordered_results:
            rows.extend(_csv_rows(result))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
            writer.writeheader()
            writer.writerows(rows)
        return path

    if fmt == "json":
        payload = {
            "format": "cadmia-structural-batch-report-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "structure_count": len(ordered_results),
            "reports": [],
        }
        for result in ordered_results:
            structure_payload = result.structure.to_dict()
            structure_payload["formula"] = result.structure.formula
            payload["reports"].append(
                {
                    "structure": structure_payload,
                    "molecular_weight": _molecular_weight(result, cfg),
                    "bonds": [asdict(bond) for bond in result.bonds],
                    "local_descriptors": [
                        asdict(descriptor) for descriptor in result.local_descriptors
                    ],
                    "bond_statistics": result.bond_statistics,
                    "ecn_statistics": result.ecn_statistics,
                    "directed_bond_statistics": result.directed_bond_statistics,
                    "local_environment_statistics": result.local_environment_statistics,
                    **_structure_blocks(result),
                }
            )
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        return path

    raise ValueError(f"Unsupported report format: {fmt}")


def write_report(
    result: StructuralResult,
    output_file: str | Path,
    output_format: Optional[str] = None,
    config: Optional[Config] = None,
) -> Path:
    """Write TXT, CSV, or JSON according to output_format or file suffix."""
    path = Path(output_file)
    fmt = (output_format or path.suffix.lstrip(".") or "txt").lower()
    if fmt == "txt":
        return write_text_report(result, path, config=config)
    if fmt == "csv":
        return write_csv_report(result, path)
    if fmt == "json":
        return write_json_report(result, path, config=config)
    raise ValueError(f"Unsupported report format: {fmt}")
