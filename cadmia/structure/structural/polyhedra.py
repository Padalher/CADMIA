"""Coordination-polyhedron descriptors."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import acos, degrees, isfinite, sqrt
from typing import Dict, List, Optional

import numpy as np

from cadmia.core import Structure
from cadmia.structure.core.config import load_config
from cadmia.structure.structural.analyzer import StructuralResult
from cadmia.structure.structural.shape import ShapeMeasure, shape_measures


@dataclass
class Polyhedron:
    """Coordination polyhedron of a central atom.

    ``mean_bond_length`` is in Å and ``bond_length_distortion`` is Baur's dimensionless
    index. ``mean_angle`` (degrees) and ``angle_distortion`` — Robinson's angle variance
    σ², in degrees² — are computed over the *cis* angles only, so both are 90.0 and 0.0
    for a perfect octahedron.

    ``metal_displacement`` is the absolute off-centering of the metal from its ligand
    centroid, in Å; ``metal_displacement_ratio`` is the same divided by the mean bond
    length, which is what makes it comparable between polyhedra of different size (a
    0.1 Å shift means much more in a short Al-O polyhedron than in a long one).

    ``volume`` (Å³) is the convex hull of the ligands and ``quadratic_elongation`` is
    Robinson's ⟨λ⟩ (dimensionless, 1.0 for an ideal polyhedron). ⟨λ⟩ normalises by the
    bond length of the *equal-volume ideal* polyhedron, whereas ``bond_length_distortion``
    normalises by the observed mean — so Baur's index is blind to a deformation that
    leaves the lengths evenly spread, and ⟨λ⟩ is not. Both are NaN for a flat or
    degenerate polyhedron, where an elongation is not defined.

    ``shape`` and ``shape_measure`` are the closest reference shape and its continuous
    shape measure (see :mod:`shape`), and ``shape_measures`` holds the measure against
    every reference of that coordination number. They exist because ``angle_distortion``
    and ``quadratic_elongation`` are only defined for the handful of types that have a
    tabulated ideal, which in a small cluster is the minority, so they are NaN for most
    centres. A CShM is defined for every centre whose
    coordination number has references at all, so it says how distorted a seesaw is
    instead of declining to answer.
    """

    center_index: int
    center_label: str
    center_element: str
    neighbor_indices: List[int]
    neighbor_labels: List[str]
    neighbor_element: str
    cn: int
    polyhedron_type: str
    mean_bond_length: float
    bond_length_distortion: float
    metal_displacement: float
    mean_angle: float
    angle_distortion: float
    metal_displacement_ratio: float = 0.0
    volume: float = float("nan")
    quadratic_elongation: float = float("nan")
    shape: str = ""
    shape_measure: float = float("nan")
    shape_measures: Optional[Dict[str, float]] = None


_TETRAHEDRAL_ANGLE = 109.4712206

# Reference for the angle variance, per polyhedron TYPE — not per coordination number:
# a square-planar and a tetrahedral center both have CN=4 but ideal cis angles of 90°
# and 109.47°. Each entry is (ideal cis angle in degrees, how many cis angles the ideal
# polyhedron has). A trigonal bipyramid is deliberately absent: its ideal angles are two
# different values (90° axial-equatorial, 120° equatorial-equatorial), so a variance
# against a single reference would be meaningless — σ² is reported as NaN there.
_IDEAL_CIS: dict[str, tuple[float, int]] = {
    "triangle":       (120.0, 3),
    "square_planar":  (90.0, 4),
    "tetrahedron":    (_TETRAHEDRAL_ANGLE, 6),
    "square_pyramid": (90.0, 8),
    "octahedron":     (90.0, 12),
}

# Volume of the REGULAR polyhedron of each type as V = k·R³, with R the centre-to-vertex
# distance. Inverting it gives the l₀ that Robinson's quadratic elongation needs: the bond
# length of the ideal polyhedron with the same volume as the real one.
#   octahedron         vertices at ±R on the axes           V = 4/3 R³
#   tetrahedron        circumradius R                       V = 8/(9√3) R³
#   square pyramid     half an octahedron                   V = 2/3 R³
#   trigonal bipyramid two pyramids on an equilateral base  V = √3/2 R³
# Flat types (triangle, square_planar) are absent: they have no volume, so no ⟨λ⟩. So is
# `seesaw`, which has no regular counterpart. Note this table covers the trigonal
# bipyramid while _IDEAL_CIS does not — a single ideal *angle* does not exist there, but a
# single ideal *volume* does.
_VOLUME_COEFF: dict[str, float] = {
    "tetrahedron":        8.0 / (9.0 * sqrt(3.0)),
    "square_pyramid":     2.0 / 3.0,
    "trigonal_bipyramid": sqrt(3.0) / 2.0,
    "octahedron":         4.0 / 3.0,
}

# Addison's tau_5 and Yang's tau_4 split geometries that share a coordination number.
# Midpoints of the 0..1 scales: tau_5 = 0 square pyramid, 1 trigonal bipyramid;
# tau_4 = 0 square planar, 1 tetrahedron (0.85 is the usual seesaw/tetrahedron cut).
_TAU5_MIDPOINT = 0.5
_TAU4_SQUARE_PLANAR = 0.25
_TAU4_TETRAHEDRON = 0.85


def classify_polyhedron(cn: int, angles: "Optional[np.ndarray]" = None) -> str:
    """Classify a coordination polyhedron.

    With *angles* (all ligand-center-ligand angles, degrees) CN=4 and CN=5 are told
    apart by geometry rather than by count: both a tetrahedron and a square-planar
    center have CN=4, and both a square pyramid and a trigonal bipyramid have CN=5.
    Without angles the classification falls back to the coordination number alone,
    which cannot distinguish them.
    """
    if cn == 6:
        return "octahedron"
    if cn == 3:
        return "triangle"
    if angles is not None and len(angles) >= 2:
        if cn == 4:
            t4 = tau_4(angles)
            if t4 <= _TAU4_SQUARE_PLANAR:
                return "square_planar"
            if t4 >= _TAU4_TETRAHEDRON:
                return "tetrahedron"
            return "seesaw"
        if cn == 5:
            return "trigonal_bipyramid" if tau_5(angles) > _TAU5_MIDPOINT else "square_pyramid"
    if cn == 4:
        return "tetrahedron"
    if cn == 5:
        return "square_pyramid"
    return f"other_CN{cn}"


def tau_4(angles: np.ndarray) -> float:
    """Yang's τ₄ for a 4-coordinate center: 1.00 tetrahedral, 0.00 square planar.

    τ₄ = [360° − (α + β)] / (360° − 2θ), with α, β the two largest angles and
    θ = 109.47°. Yang, Powell & Houser, *Dalton Trans.* 2007, 955.
    """
    a = np.sort(np.asarray(angles, dtype=float))[::-1]
    if a.size < 2:
        return float("nan")
    alpha, beta = float(a[0]), float(a[1])
    return (360.0 - (alpha + beta)) / (360.0 - 2 * _TETRAHEDRAL_ANGLE)


def tau_5(angles: np.ndarray) -> float:
    """Addison's τ₅ for a 5-coordinate center: 1.00 trigonal bipyramidal, 0.00 square pyramidal.

    τ₅ = (β − α) / 60°, with β ≥ α the two largest angles.
    Addison et al., *J. Chem. Soc., Dalton Trans.* 1984, 1349.
    """
    a = np.sort(np.asarray(angles, dtype=float))[::-1]
    if a.size < 2:
        return float("nan")
    return (float(a[0]) - float(a[1])) / 60.0


def _cis_angles(angles: np.ndarray, polyhedron_type: str) -> np.ndarray:
    """The smallest ligand-center-ligand angles: the *cis* (nearest-neighbour) set.

    An octahedron has 12 cis angles at 90° and 3 trans at 180°. Averaging all 15 gives
    108° for a *perfect* octahedron with a spread of 36°, so a flawless polyhedron would
    look maximally distorted — which is why only the cis set is used, as the
    crystallographic definitions intend. Unknown types keep every angle.
    """
    a = np.sort(np.asarray(angles, dtype=float))
    ideal = _IDEAL_CIS.get(polyhedron_type)
    if ideal is None:
        return a
    return a[:min(ideal[1], a.size)]


def angle_variance(angles: np.ndarray, polyhedron_type: str) -> float:
    """Robinson's angle variance σ² (deg²) over the cis angles; 0 for an ideal polyhedron.

    σ² = Σ(θᵢ − θ_ideal)² / (m − 1), over the cis angles of the polyhedron.
    Robinson, Gibbs & Ribbe, *Science* **172**, 567 (1971).

    NaN when the geometry has no single ideal cis angle (a trigonal bipyramid mixes 90°
    and 120°) or is unclassified: reporting a number there would invite comparing values
    that do not measure the same thing.
    """
    ideal = _IDEAL_CIS.get(polyhedron_type)
    if ideal is None:
        return float("nan")
    cis = _cis_angles(angles, polyhedron_type)
    if cis.size < 2:
        return float("nan")
    return float(np.sum((cis - ideal[0]) ** 2) / (cis.size - 1))


def convex_hull_volume(points: "np.ndarray") -> float:
    """Volume (Å³) of the convex hull of a small point set; NaN if it has none.

    Written here rather than taken from ``scipy.spatial`` on purpose: scipy lives in the
    ``[full]`` extra, and structural analysis is base functionality — the same command must
    not return different results depending on how CADMIA was installed. A coordination
    polyhedron has between 3 and ~8 vertices, so the cubic enumeration below is free.

    Supporting planes are found by brute force (every non-collinear triple whose plane
    leaves all points on one side) and grouped by the *set of points lying on them*, which
    merges the coplanar triples of a square face into a single face instead of counting it
    four times. Each face is then fanned into triangles and each triangle forms a
    tetrahedron with the hull centroid.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return float("nan")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = float(np.abs(centered).max())
    if scale == 0.0:
        return float("nan")
    tol = 1e-6 * scale
    if np.linalg.matrix_rank(centered, tol=tol) < 3:
        return float("nan")          # flat: a square-planar or triangular set has no volume

    faces: "dict[frozenset[int], np.ndarray]" = {}
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                normal = np.cross(pts[j] - pts[i], pts[k] - pts[i])
                length = float(np.linalg.norm(normal))
                if length <= tol * scale:
                    continue                                  # collinear triple
                normal = normal / length
                offset = float(np.dot(normal, pts[i]))
                if float(np.dot(normal, centroid)) > offset:  # orient outward
                    normal, offset = -normal, -offset
                signed = pts @ normal - offset
                if np.any(signed > tol):
                    continue                                  # not a supporting plane
                faces[frozenset(np.flatnonzero(np.abs(signed) <= tol).tolist())] = normal

    volume = 0.0
    for indices, normal in faces.items():
        face = pts[sorted(indices)]
        face_centroid = face.mean(axis=0)
        radial = face[0] - face_centroid
        radius = float(np.linalg.norm(radial))
        if radius <= tol:
            continue
        u = radial / radius
        v = np.cross(normal, u)
        local = face - face_centroid
        ordered = face[np.argsort(np.arctan2(local @ v, local @ u))]
        anchor = ordered[0] - centroid
        for m in range(1, len(ordered) - 1):
            volume += abs(float(np.linalg.det(
                np.stack([anchor, ordered[m] - centroid, ordered[m + 1] - centroid])
            ))) / 6.0
    return float(volume)


def quadratic_elongation(distances: "np.ndarray", volume: float, polyhedron_type: str) -> float:
    """Robinson's quadratic elongation ⟨λ⟩; 1.0 for an ideal polyhedron, NaN when undefined.

    ⟨λ⟩ = (1/n) Σ (lᵢ/l₀)², where l₀ is the centre-to-vertex distance of the **regular
    polyhedron of the same volume** — not the observed mean, which is what
    ``bond_length_distortion`` (Baur's index) uses and what makes the two indices measure
    different things. Robinson, Gibbs & Ribbe, *Science* **172**, 567 (1971).

    NaN for types with no regular reference (``seesaw``, unclassified) and for flat
    polyhedra, where the volume is not defined either.
    """
    coefficient = _VOLUME_COEFF.get(polyhedron_type)
    if coefficient is None or not isfinite(volume) or volume <= 0.0:
        return float("nan")
    lengths = np.asarray(distances, dtype=float)
    if lengths.size == 0:
        return float("nan")
    ideal_length = (volume / coefficient) ** (1.0 / 3.0)
    return float(np.mean((lengths / ideal_length) ** 2))


def _angles(center: np.ndarray, ligands: np.ndarray) -> np.ndarray:
    values = []
    for i in range(len(ligands)):
        for j in range(i + 1, len(ligands)):
            v1 = ligands[i] - center
            v2 = ligands[j] - center
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0.0 or n2 == 0.0:
                continue
            cosine = float(np.dot(v1, v2) / (n1 * n2))
            values.append(degrees(acos(float(np.clip(cosine, -1.0, 1.0)))))
    return np.asarray(values, dtype=float)


def compute_polyhedron(
    center_idx: int,
    neighbor_indices: List[int],
    structure: Structure,
) -> Polyhedron:
    """Calculate geometric descriptors for one coordination polyhedron."""
    if not neighbor_indices:
        raise ValueError("At least one neighbor is required.")

    center = structure.positions[center_idx]
    ligand_positions = structure.positions[neighbor_indices]
    distances = np.linalg.norm(ligand_positions - center, axis=1)
    mean_distance = float(distances.mean())
    if mean_distance == 0.0:
        distortion = 0.0
    else:
        distortion = float(np.mean(np.abs(distances - mean_distance) / mean_distance))

    centroid = ligand_positions.mean(axis=0)
    metal_displacement = float(np.linalg.norm(center - centroid))
    displacement_ratio = (metal_displacement / mean_distance) if mean_distance else 0.0

    cn = len(neighbor_indices)
    angle_values = _angles(center, ligand_positions)
    polyhedron_type = classify_polyhedron(cn, angle_values)
    cis = _cis_angles(angle_values, polyhedron_type)
    mean_angle = float(cis.mean()) if cis.size else 0.0
    angle_distortion = angle_variance(angle_values, polyhedron_type)
    volume = convex_hull_volume(ligand_positions)
    elongation = quadratic_elongation(distances, volume, polyhedron_type)

    neighbor_elements = sorted({structure.atoms[index] for index in neighbor_indices})
    neighbor_element = neighbor_elements[0] if len(neighbor_elements) == 1 else "mixed"

    shape = shape_measures(center, ligand_positions, center_index=center_idx)

    return Polyhedron(
        center_index=center_idx,
        center_label=structure.labels[center_idx],
        center_element=structure.atoms[center_idx],
        neighbor_indices=list(neighbor_indices),
        neighbor_labels=[structure.labels[index] for index in neighbor_indices],
        neighbor_element=neighbor_element,
        cn=cn,
        polyhedron_type=polyhedron_type,
        mean_bond_length=mean_distance,
        bond_length_distortion=distortion,
        metal_displacement=metal_displacement,
        mean_angle=mean_angle,
        angle_distortion=angle_distortion,
        metal_displacement_ratio=displacement_ratio,
        volume=volume,
        quadratic_elongation=elongation,
        shape=shape.best_shape if shape else "",
        shape_measure=shape.best_value if shape else float("nan"),
        shape_measures=dict(shape.measures) if shape else None,
    )


def analyze_polyhedra(
    structure: Structure,
    result: StructuralResult,
    metal_elements: Optional[List[str]] = None,
    ligand_elements: Optional[List[str]] = None,
) -> List[Polyhedron]:
    """Analyze coordination polyhedra for metal centers with at least 3 ligands."""
    if metal_elements is None:
        metal_elements = result.metal_elements or load_config().metal_elements
    metal_set = set(metal_elements)

    if ligand_elements is None:
        inferred = set()
        for bond in result.bonds:
            pair = [(bond.atom_i, structure.atoms[bond.atom_i]), (bond.atom_j, structure.atoms[bond.atom_j])]
            for center_idx, center_el in pair:
                if center_el not in metal_set:
                    continue
                other_idx = bond.atom_j if center_idx == bond.atom_i else bond.atom_i
                other_el = structure.atoms[other_idx]
                if other_el not in metal_set:
                    inferred.add(other_el)
        ligand_set = inferred
        # Only worth reporting when the structure has metals AND has bonds, yet none of
        # them is metal-ligand — that is the case where the caller expects polyhedra and
        # silently gets none. A probe molecule with no metal, or an isolated atom with no
        # bonds at all, has no polyhedron to find and warning there would be pure noise.
        # "No polyhedra" and "could not tell what the ligands are" are different answers,
        # and an empty list alone cannot distinguish them.
        if not ligand_set and result.bonds:
            metals_present = sorted({el for el in structure.atoms if el in metal_set})
            if metals_present:
                warnings.warn(
                    f"Found {', '.join(metals_present)} but no metal-ligand bonds, so no "
                    "ligand elements could be inferred and no polyhedra will be reported. "
                    "Pass ligand_elements explicitly if bond detection missed them.",
                    stacklevel=2,
                )
    else:
        ligand_set = set(ligand_elements)

    neighbors_by_center = {}
    for bond in result.bonds:
        for center_idx, other_idx in ((bond.atom_i, bond.atom_j), (bond.atom_j, bond.atom_i)):
            center_el = structure.atoms[center_idx]
            other_el = structure.atoms[other_idx]
            if center_el in metal_set and other_el in ligand_set:
                neighbors_by_center.setdefault(center_idx, []).append(other_idx)

    polyhedra = []
    for center_idx, neighbor_indices in sorted(neighbors_by_center.items()):
        unique_neighbors = sorted(set(neighbor_indices))
        if len(unique_neighbors) < 3:
            continue
        polyhedra.append(compute_polyhedron(center_idx, unique_neighbors, structure))
    return polyhedra


@dataclass
class PolyhedraLink:
    """How two coordination polyhedra are joined, by how many ligands they have in common.

    In oxide chemistry this is the structural description: the same stoichiometry built from
    corner-sharing or from edge-sharing polyhedra is a different material. ``kind`` is
    ``corner`` for one shared ligand, ``edge`` for two and ``face`` for three or more, which
    is the standard reading of a shared vertex, a shared edge and a shared face.

    ``center_distance`` is the metal-metal distance in Å. It is reported alongside because it
    is what makes the sharing visible in a plain structure file: corner sharing keeps the two
    metals far apart, edge sharing pulls them together, and face sharing closer still.
    """

    center_index_a: int
    center_index_b: int
    center_label_a: str
    center_label_b: str
    shared_indices: List[int]
    shared_labels: List[str]
    kind: str
    center_distance: float


_SHARING_KIND = {1: "corner", 2: "edge"}


def polyhedra_connectivity(polyhedra: List[Polyhedron],
                           structure: Optional[Structure] = None) -> List[PolyhedraLink]:
    """Pairwise sharing between coordination polyhedra, as corner/edge/face links.

    Only pairs with at least one ligand in common appear; a polyhedron that shares nothing is
    absent from the list rather than carrying an empty entry, so ``len(links)`` is the number
    of real connections. Pairs are reported once, with the lower centre index first.
    """
    links: List[PolyhedraLink] = []
    for i, first in enumerate(polyhedra):
        first_set = set(first.neighbor_indices)
        for second in polyhedra[i + 1:]:
            shared = sorted(first_set & set(second.neighbor_indices))
            if not shared:
                continue
            if structure is not None:
                distance = float(np.linalg.norm(
                    structure.positions[first.center_index] - structure.positions[second.center_index]))
                labels = [structure.labels[index] for index in shared]
            else:
                distance = float("nan")
                labels = [str(index) for index in shared]
            links.append(PolyhedraLink(
                center_index_a=first.center_index,
                center_index_b=second.center_index,
                center_label_a=first.center_label,
                center_label_b=second.center_label,
                shared_indices=shared,
                shared_labels=labels,
                kind=_SHARING_KIND.get(len(shared), "face"),
                center_distance=distance,
            ))
    return links


def connectivity_summary(links: List[PolyhedraLink],
                         polyhedra: List[Polyhedron]) -> Dict[str, object]:
    """Counts per sharing kind plus the centres that share nothing.

    ``isolated`` matters as much as the counts: a polyhedron connected to nothing is a centre
    whose ligands are all terminal, which in a cluster is where the reactive, undercoordinated
    sites tend to be.
    """
    counts: Dict[str, int] = {"corner": 0, "edge": 0, "face": 0}
    for link in links:
        counts[link.kind] = counts.get(link.kind, 0) + 1
    connected = {index for link in links for index in (link.center_index_a, link.center_index_b)}
    isolated = [p.center_label for p in polyhedra if p.center_index not in connected]
    return {
        "links": len(links),
        "by_kind": counts,
        "isolated_centers": isolated,
        "polyhedra": len(polyhedra),
    }


@dataclass
class CationBridge:
    """Two metals bonded to at least one common ligand, and how crowded each shared ligand is.

    ``shared_mu`` is the number of metals bonded to each shared ligand (the k of a mu_k bridge).
    It is reported because a mu_2 oxygen and a mu_6 oxygen are both "shared", yet the bond each
    metal makes to the second one carries a fraction of the valence of the first.
    """

    index_a: int
    index_b: int
    label_a: str
    label_b: str
    shared_indices: List[int]
    shared_labels: List[str]
    shared_mu: List[int]
    kind: str
    distance: float


def cation_neighbors(structure: Structure, result: StructuralResult,
                     metal_elements: Optional[List[str]] = None):
    """Metal-metal neighborhood through shared ligands, averaged over EVERY metal center.

    Unlike ``polyhedra_connectivity`` it does not require three ligands per center, and the
    per-pair means count the centers with no bridged neighbor as zero. Those are the two
    choices that let the mean say whether one element is dispersed in the other: averaging
    only the active centers hides exactly the atoms that are isolated.

    Returns ``(bridges, summary)``. ``summary`` is keyed by directed pair ``"A-B"`` and gives,
    for the centers of A, the mean number of B neighbors, the isolated count and the mean
    distance to the nearest B, which is defined whether or not a bridge exists.
    """
    if metal_elements is None:
        metal_elements = result.metal_elements or load_config().metal_elements
    metal_set = set(metal_elements)
    atoms = structure.atoms
    positions = np.asarray(structure.positions, dtype=float)

    metals_of_ligand: Dict[int, set] = {}
    for bond in result.bonds:
        if (atoms[bond.atom_i] in metal_set) == (atoms[bond.atom_j] in metal_set):
            continue
        metal, ligand = ((bond.atom_i, bond.atom_j) if atoms[bond.atom_i] in metal_set
                         else (bond.atom_j, bond.atom_i))
        metals_of_ligand.setdefault(ligand, set()).add(metal)

    shared_by_pair: Dict[tuple, List[int]] = {}
    for ligand, metals in metals_of_ligand.items():
        ordered = sorted(metals)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                shared_by_pair.setdefault((a, b), []).append(ligand)

    bridges = []
    for (a, b), shared in sorted(shared_by_pair.items()):
        shared = sorted(shared)
        bridges.append(CationBridge(
            index_a=a, index_b=b,
            label_a=structure.labels[a], label_b=structure.labels[b],
            shared_indices=shared,
            shared_labels=[structure.labels[index] for index in shared],
            shared_mu=[len(metals_of_ligand[index]) for index in shared],
            kind=_SHARING_KIND.get(len(shared), "face"),
            distance=float(np.linalg.norm(positions[a] - positions[b])),
        ))

    metal_indices = [i for i, element in enumerate(atoms) if element in metal_set]
    present = sorted({atoms[i] for i in metal_indices})
    partners: Dict[int, List[int]] = {i: [] for i in metal_indices}
    for bridge in bridges:
        partners[bridge.index_a].append(bridge.index_b)
        partners[bridge.index_b].append(bridge.index_a)

    summary: Dict[str, Dict[str, object]] = {}
    for element_a in present:
        centers = [i for i in metal_indices if atoms[i] == element_a]
        for element_b in present:
            others = [j for j in metal_indices if atoms[j] == element_b]
            if element_a == element_b and len(centers) < 2:
                continue
            counts = [sum(1 for j in partners[i] if atoms[j] == element_b) for i in centers]
            nearest = [min(float(np.linalg.norm(positions[i] - positions[j]))
                           for j in others if j != i) for i in centers]
            summary[f"{element_a}-{element_b}"] = {
                "centers": len(centers),
                "mean_neighbors": float(np.mean(counts)),
                "isolated": sum(1 for count in counts if count == 0),
                "mean_nearest_distance": float(np.mean(nearest)),
                "per_center": [
                    {"index": i, "label": structure.labels[i], "neighbors": count,
                     "nearest_distance": distance}
                    for i, count, distance in zip(centers, counts, nearest)
                ],
            }
    return bridges, summary
