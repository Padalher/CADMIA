"""Typed structural analysis."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cadmia.core import Structure
from cadmia.structure.core.config import Config, load_config


@dataclass
class BondInfo:
    atom_i: int
    atom_j: int
    label_i: str
    label_j: str
    bond_type: str
    distance: float


@dataclass
class NeighborInfo:
    atom_index: int
    label: str
    element: str
    bond_type: str
    distance: float


@dataclass
class LocalDescriptor:
    atom_index: int
    label: str
    element: str
    cn: int
    ecn: float
    avg_bond_distance: float
    bonded_elements: Dict[str, int]
    effective_avg_bond_distance: float
    min_bond_distance: float
    max_bond_distance: float
    std_bond_distance: float
    ecn_by_element: Dict[str, float]
    avg_distance_by_element: Dict[str, float]
    effective_avg_distance_by_element: Dict[str, float]
    bond_type_counts: Dict[str, int]
    neighbors: List[NeighborInfo]


@dataclass
class StructuralResult:
    structure: Structure
    bonds: List[BondInfo]
    local_descriptors: List[LocalDescriptor]
    bond_statistics: Dict
    ecn_statistics: Dict
    directed_bond_statistics: Dict
    local_environment_statistics: Dict
    # Metals of the config the analysis ran with, so later stages need not reload the default.
    metal_elements: List[str] = field(default_factory=list)


def _as_config(config: Optional[Config]) -> Config:
    if config is None:
        return load_config()
    return config


def distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def _canonical_bond_type(element1: str, element2: str, config: Config) -> Optional[str]:
    bond1 = f"{element1}-{element2}"
    bond2 = f"{element2}-{element1}"
    if bond1 in config.bond_cutoffs:
        return bond1
    if bond2 in config.bond_cutoffs:
        return bond2
    if config.get_bond_cutoff(element1, element2) > 0.0:
        return "-".join(sorted([element1, element2]))
    return None


def _is_valid_bond(
    element1: str,
    element2: str,
    distance: float,
    config: Config,
) -> bool:
    if config.is_excluded_bond(element1, element2):
        return False
    cutoff = config.get_bond_cutoff(element1, element2)
    return cutoff > 0.0 and float(distance) <= cutoff


def _self_consistent_ecn_and_distance(
    distances: List[float] | np.ndarray,
    config: Config,
) -> Tuple[float, float]:
    """Return Hoppe ECN and the self-consistent effective distance d_av."""
    distances_array = np.asarray(distances, dtype=float)
    distances_array = distances_array[distances_array > 0]
    if distances_array.size == 0:
        return 0.0, 0.0

    params = config.ecn_parameters
    max_iter = int(params.get("max_iterations", 100))
    threshold = float(params.get("convergence_threshold", 1e-6))
    damping = float(params.get("damping_factor", 6.0))
    d_avg = float(np.min(distances_array))
    best_d_avg = d_avg
    best_error = float("inf")

    for _ in range(max(1, max_iter)):
        weights = np.exp(1 - (distances_array / d_avg) ** damping)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            break
        new_d_avg = float(np.sum(distances_array * weights) / weight_sum)
        if not np.isfinite(new_d_avg) or new_d_avg <= 0.0:
            break
        error = abs(new_d_avg - d_avg)
        if error < best_error:
            best_error = error
            best_d_avg = new_d_avg
        if error < threshold:
            best_d_avg = new_d_avg
            break
        d_avg = new_d_avg

    d_avg = best_d_avg
    weights = np.exp(1 - (distances_array / d_avg) ** damping)
    return float(np.sum(weights)), float(d_avg)


def _self_consistent_ecn(distances: List[float], config: Config) -> float:
    ecn, _ = _self_consistent_ecn_and_distance(distances, config)
    return ecn


def _bond_stats(bonds: List[BondInfo]) -> Dict:
    distances_by_type = defaultdict(list)
    for bond in bonds:
        distances_by_type[bond.bond_type].append(bond.distance)

    stats = {}
    for bond_type, distances in distances_by_type.items():
        arr = np.asarray(distances, dtype=float)
        stats[bond_type] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return dict(sorted(stats.items()))


def _ecn_stats(descriptors: List[LocalDescriptor]) -> Dict:
    values_by_element = defaultdict(list)
    for descriptor in descriptors:
        values_by_element[descriptor.element].append(descriptor.ecn)

    stats = {}
    for element, values in values_by_element.items():
        arr = np.asarray(values, dtype=float)
        stats[element] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return dict(sorted(stats.items()))


def _local_environment_stats(descriptors: List[LocalDescriptor]) -> Dict:
    values_by_element = defaultdict(lambda: defaultdict(list))
    all_cn = []
    all_ecn = []
    all_avg = []
    all_eff_avg = []

    for descriptor in descriptors:
        all_cn.append(float(descriptor.cn))
        all_ecn.append(float(descriptor.ecn))
        if descriptor.cn > 0:
            all_avg.append(float(descriptor.avg_bond_distance))
            all_eff_avg.append(float(descriptor.effective_avg_bond_distance))

        element_values = values_by_element[descriptor.element]
        element_values["cn"].append(float(descriptor.cn))
        element_values["ecn"].append(float(descriptor.ecn))
        if descriptor.cn > 0:
            element_values["avg_bond_distance"].append(float(descriptor.avg_bond_distance))
            element_values["effective_avg_bond_distance"].append(
                float(descriptor.effective_avg_bond_distance)
            )

    def summarize(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    stats = {
        "all_atoms": {
            "count": len(descriptors),
            "coordination": summarize(all_cn),
            "ecn": summarize(all_ecn),
            "avg_bond_distance": summarize(all_avg),
            "effective_avg_bond_distance": summarize(all_eff_avg),
        },
        "by_element": {},
    }

    for element, values in sorted(values_by_element.items()):
        stats["by_element"][element] = {
            "count": len(values["cn"]),
            "coordination": summarize(values["cn"]),
            "ecn": summarize(values["ecn"]),
            "avg_bond_distance": summarize(values["avg_bond_distance"]),
            "effective_avg_bond_distance": summarize(
                values["effective_avg_bond_distance"]
            ),
        }
    return stats


def _directed_bond_stats(
    descriptors: List[LocalDescriptor],
    config: Config,
) -> Dict:
    descriptors_by_element = defaultdict(list)
    for descriptor in descriptors:
        descriptors_by_element[descriptor.element].append(descriptor)

    elements_present = sorted(descriptors_by_element)
    metal_elements = set(config.metal_elements)
    central_elements = [
        element for element in elements_present if element in metal_elements
    ] or elements_present
    neighbor_elements = elements_present

    stats = {}
    for central_element in central_elements:
        central_descriptors = descriptors_by_element[central_element]
        for neighbor_element in neighbor_elements:
            key = f"{central_element}-{neighbor_element}"
            cn_values = [
                float(descriptor.bonded_elements.get(neighbor_element, 0))
                for descriptor in central_descriptors
            ]
            ecn_values = [
                float(descriptor.ecn_by_element.get(neighbor_element, 0.0))
                for descriptor in central_descriptors
            ]
            avg_values = [
                float(descriptor.avg_distance_by_element[neighbor_element])
                for descriptor in central_descriptors
                if neighbor_element in descriptor.avg_distance_by_element
            ]
            eff_avg_values = [
                float(descriptor.effective_avg_distance_by_element[neighbor_element])
                for descriptor in central_descriptors
                if neighbor_element in descriptor.effective_avg_distance_by_element
            ]
            distances = [
                float(neighbor.distance)
                for descriptor in central_descriptors
                for neighbor in descriptor.neighbors
                if neighbor.element == neighbor_element
            ]

            arr_dist = np.asarray(distances, dtype=float)
            arr_ecn = np.asarray(ecn_values, dtype=float)
            arr_cn = np.asarray(cn_values, dtype=float)
            # active_ecn: atoms where ECN > 0 toward this neighbor element.
            # ECN is autoconsistent (Hoppe), computed over every atom — independent
            # of bond cutoffs. Filtering by CN > 0 would exclude atoms that genuinely
            # contribute to ECN but have no discrete bond, biasing the average.
            active_ecn = arr_ecn[arr_ecn > 0]
            directed_bond_count = int(arr_cn.sum())
            if central_element == neighbor_element:
                bond_count = directed_bond_count // 2
            else:
                bond_count = directed_bond_count
            # avg_distance_by_element is bond-based: filter by CN > 0 is correct.
            # effective_avg_distance_by_element is the Hoppe ECN d_av: use all
            # descriptors that carry the key, regardless of bond count.
            avg_values_active = [
                float(descriptor.avg_distance_by_element[neighbor_element])
                for descriptor in central_descriptors
                if descriptor.bonded_elements.get(neighbor_element, 0) > 0
                and neighbor_element in descriptor.avg_distance_by_element
            ]
            mean_atom_avg_distance = (
                float(np.mean(avg_values_active)) if avg_values_active else 0.0
            )
            average_d_av = float(np.mean(eff_avg_values)) if eff_avg_values else 0.0
            stats[key] = {
                "central_element": central_element,
                "neighbor_element": neighbor_element,
                "central_atom_count": len(central_descriptors),
                "active_atom_count": int(np.count_nonzero(arr_cn)),
                "bond_count": bond_count,
                "directed_bond_count": directed_bond_count,
                "mean_cn_per_atom": float(arr_cn.mean()) if arr_cn.size else 0.0,
                "mean_ecn_all_atoms": float(arr_ecn.mean()) if arr_ecn.size else 0.0,
                "mean_ecn_active_atoms": (
                    float(active_ecn.mean()) if active_ecn.size else 0.0
                ),
                "std_ecn_active_atoms": (
                    float(active_ecn.std()) if active_ecn.size else 0.0
                ),
                "mean_distance": float(arr_dist.mean()) if arr_dist.size else 0.0,
                "std_distance": float(arr_dist.std()) if arr_dist.size else 0.0,
                "min_distance": float(arr_dist.min()) if arr_dist.size else 0.0,
                "max_distance": float(arr_dist.max()) if arr_dist.size else 0.0,
                "mean_atom_avg_distance": mean_atom_avg_distance,
                "average_d_av": average_d_av,
                "mean_atom_effective_avg_distance": average_d_av,
            }

    return dict(sorted(stats.items()))


def analyze_structure(structure: Structure, config=None) -> StructuralResult:
    """
    Calculate bonds, Hoppe ECN, local descriptors, and bond statistics.

    CN uses explicit bond cutoffs per pair type (discrete, chemical criterion).
    ECN takes every other atom, with no distance cutoff and no bond-type restriction, as in
    Da Silva (J. Appl. Phys. 109, 023502, 2011): the Hoppe weight already suppresses distant
    atoms, and a cutoff only adds a parameter the result depends on.
    """
    cfg = _as_config(config)
    dist_matrix = distance_matrix(structure.positions)

    bonds = []
    cn_neighbors: dict = defaultdict(list)   # bond-cutoff neighbors → CN
    ecn_neighbors: dict = defaultdict(list)  # every other atom → ECN

    for i in range(structure.n_atoms):
        for j in range(i + 1, structure.n_atoms):
            d = float(dist_matrix[i, j])
            element_i = structure.atoms[i]
            element_j = structure.atoms[j]

            # --- CN / bonds: bond-type cutoff ---
            if _is_valid_bond(element_i, element_j, d, cfg):
                bond_type = _canonical_bond_type(element_i, element_j, cfg)
                if bond_type is not None:
                    bonds.append(BondInfo(
                        atom_i=i, atom_j=j,
                        label_i=structure.labels[i],
                        label_j=structure.labels[j],
                        bond_type=bond_type,
                        distance=d,
                    ))
                    cn_neighbors[i].append((j, d, bond_type))
                    cn_neighbors[j].append((i, d, bond_type))

            if d > 0.0:
                ecn_neighbors[i].append((j, d))
                ecn_neighbors[j].append((i, d))

    local_descriptors = []
    for i, atom in enumerate(structure.atoms):
        bonded = cn_neighbors.get(i, [])
        bonded_elements: dict = defaultdict(int)
        distances_by_element_bond: dict = defaultdict(list)
        bond_type_counts: dict = defaultdict(int)
        neighbor_details = []
        bond_distances = []

        for neighbor_idx, d, bond_type in bonded:
            bonded_elements[structure.atoms[neighbor_idx]] += 1
            distances_by_element_bond[structure.atoms[neighbor_idx]].append(d)
            bond_type_counts[bond_type] += 1
            bond_distances.append(d)
            neighbor_details.append(NeighborInfo(
                atom_index=neighbor_idx,
                label=structure.labels[neighbor_idx],
                element=structure.atoms[neighbor_idx],
                bond_type=bond_type,
                distance=d,
            ))

        avg = float(np.mean(bond_distances)) if bond_distances else 0.0
        min_distance = float(np.min(bond_distances)) if bond_distances else 0.0
        max_distance = float(np.max(bond_distances)) if bond_distances else 0.0
        std_distance = float(np.std(bond_distances)) if bond_distances else 0.0

        ecn_all = ecn_neighbors.get(i, [])
        ecn_distances_all = [d for _, d in ecn_all]
        ecn, effective_avg = _self_consistent_ecn_and_distance(ecn_distances_all, cfg)

        # ECN per element: filter global neighbors by element, self-consistent each
        ecn_distances_by_element: dict = defaultdict(list)
        for neighbor_idx, d in ecn_all:
            ecn_distances_by_element[structure.atoms[neighbor_idx]].append(d)

        ecn_by_element = {}
        avg_distance_by_element = {}
        effective_avg_distance_by_element = {}
        # avg_distance_by_element uses bond neighbors (CN context)
        for element, values in sorted(distances_by_element_bond.items()):
            avg_distance_by_element[element] = float(np.mean(values))
        # ECN and effective_avg use global neighbors (ECN context)
        for element, values in sorted(ecn_distances_by_element.items()):
            element_ecn, element_eff_avg = _self_consistent_ecn_and_distance(values, cfg)
            ecn_by_element[element] = element_ecn
            effective_avg_distance_by_element[element] = element_eff_avg
        # ensure avg_distance_by_element has an entry for any element seen via ECN
        for element, values in sorted(ecn_distances_by_element.items()):
            if element not in avg_distance_by_element:
                avg_distance_by_element[element] = float(np.mean(values))

        local_descriptors.append(
            LocalDescriptor(
                atom_index=i,
                label=structure.labels[i],
                element=atom,
                cn=len(bonded),
                ecn=ecn,
                avg_bond_distance=avg,
                bonded_elements=dict(sorted(bonded_elements.items())),
                effective_avg_bond_distance=effective_avg,
                min_bond_distance=min_distance,
                max_bond_distance=max_distance,
                std_bond_distance=std_distance,
                ecn_by_element=ecn_by_element,
                avg_distance_by_element=avg_distance_by_element,
                effective_avg_distance_by_element=effective_avg_distance_by_element,
                bond_type_counts=dict(sorted(bond_type_counts.items())),
                neighbors=sorted(neighbor_details, key=lambda item: item.distance),
            )
        )

    return StructuralResult(
        structure=structure,
        bonds=bonds,
        local_descriptors=local_descriptors,
        bond_statistics=_bond_stats(bonds),
        ecn_statistics=_ecn_stats(local_descriptors),
        directed_bond_statistics=_directed_bond_stats(local_descriptors, cfg),
        local_environment_statistics=_local_environment_stats(local_descriptors),
        metal_elements=list(cfg.metal_elements),
    )
