"""Whole-structure descriptors: rings and overall shape.

Everything else in this package describes an atom or a pair of atoms. Two questions about the
cluster as a whole are left over, and both matter for a series where the only variable is size:

**What rings does the connectivity close?** In an oxide cluster the Al-O-Al-O four-membered
ring is the motif the framework is built from, and counting them says more about the topology
than any per-atom number. What is reported is Vismara's (1997) set of *relevant* cycles: every
cycle that is not a GF(2) sum of strictly shorter ones. That is stronger than "the shortest ring
through each bond" — a per-bond search misses a ring whose every edge also sits on some other,
shorter ring elsewhere, even when that longer ring is topologically necessary (its independence
only shows up at the level of the whole cycle space, not at any single edge). A densely bonded
cluster has exactly this shape: every bond belongs to a small 2- or 3-membered ring, so a
per-bond search never has a reason to report the larger ring the framework's windows are built
from, and undercounts.

Which bonds the search may use is not a detail. Over the *full* bond graph of these clusters
every ring comes out as a triangle, because a metal-metal contact short-circuits the
metal-oxygen ring it sits across: Al-O-Al closes through the Al-Al bond before the fourth atom
is reached. The framework motif only appears on the metal-ligand subgraph, which is why
``metal_ligand_rings`` exists and is what the report uses.

**What shape is it?** The gyration tensor gives the answer without any reference: its three
eigenvalues say whether the atoms fill a sphere, a disc or a rod, and the relative shape
anisotropy compresses that into one number between 0 (spherical) and 1 (collinear).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cadmia.core import Structure
from cadmia.structure.structural.analyzer import StructuralResult

# A ring larger than this is not a motif, it is a path around the whole cluster. The bound also
# keeps the per-bond search cheap on a structure whose graph is densely connected.
MAX_RING_SIZE = 12


@dataclass
class Ring:
    """One cycle of the bond graph, as the atoms it visits in order."""

    atom_indices: List[int]
    atom_labels: List[str]
    size: int
    composition: str


@dataclass
class RingTopology:
    """How complete a ring search is, independent of how many rings were kept.

    ``cyclomatic_number`` (mu = E - V + C) is the dimension of the graph's cycle space: the
    number of independent rings a *complete* search has to account for. ``rank`` is how many of
    those dimensions the returned rings actually span; ``is_complete`` says whether that matches
    mu. It can fall short only if ``max_size`` cut the search off before every dimension was
    reached — not a sign of a bug in the search itself.
    """

    n_vertices: int
    n_edges: int
    n_components: int
    cyclomatic_number: int
    rank: int
    is_complete: bool


@dataclass
class GlobalShape:
    """Shape of the whole atom cloud, from the gyration tensor.

    ``radius_of_gyration`` is in Å and ``principal_moments`` are its three eigenvalues in
    ascending order (Å²), so their sum is Rg². ``asphericity`` is 0 for any arrangement with
    tetrahedral or higher symmetry and grows as one axis stretches; ``acylindricity`` is 0
    whenever the two shorter axes are equal. ``anisotropy`` folds both into [0, 1]: 0 for a
    sphere, 1 for a straight line.

    ``max_diameter`` is the largest interatomic distance and ``hull_volume`` the convex hull of
    all atoms (Å³, NaN when they are coplanar).
    """

    n_atoms: int
    radius_of_gyration: float
    principal_moments: Tuple[float, float, float]
    asphericity: float
    acylindricity: float
    anisotropy: float
    max_diameter: float
    hull_volume: float


def _canonical_edges(bonds) -> List[Tuple[int, int]]:
    """Deduplicated (i, j) with i < j, in first-seen order — the graph's fixed edge indexing."""
    seen = set()
    edges: List[Tuple[int, int]] = []
    for bond in bonds:
        edge = (bond.atom_i, bond.atom_j) if bond.atom_i < bond.atom_j else (bond.atom_j, bond.atom_i)
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)
    return edges


def _adjacency_with_edges(edges: List[Tuple[int, int]], n_atoms: int) -> List[List[Tuple[int, int]]]:
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for edge_index, (i, j) in enumerate(edges):
        adjacency[i].append((j, edge_index))
        adjacency[j].append((i, edge_index))
    return adjacency


def _count_components(edges: List[Tuple[int, int]], n_atoms: int) -> int:
    """Connected components by union-find, counting every atom (isolated ones too)."""
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n_atoms)})


def _shortest_path_forest(adjacency: List[List[Tuple[int, int]]], source: int,
                          max_depth: int) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, int]]]]:
    """BFS by levels from ``source``, keeping every predecessor at the shortest depth.

    Every one, not the first: a bond that lies on two equally short routes to the same vertex is
    the normal case in a fused framework, and keeping only the first one BFS reaches would later
    hide a ring that only exists through the other route.
    """
    depth: Dict[int, int] = {source: 0}
    parents: Dict[int, List[Tuple[int, int]]] = {source: []}
    frontier = [source]
    level = 0
    while frontier and level < max_depth:
        level += 1
        nxt: List[int] = []
        for node in frontier:
            for neighbor, edge_index in adjacency[node]:
                if neighbor not in depth:
                    depth[neighbor] = level
                    parents[neighbor] = [(node, edge_index)]
                    nxt.append(neighbor)
                elif depth[neighbor] == level:
                    parents[neighbor].append((node, edge_index))
        frontier = nxt
    return depth, parents


def _enumerate_shortest_paths(parents: Dict[int, List[Tuple[int, int]]], source: int, vertex: int):
    """Every shortest source->vertex path, as (vertex sequence, edge-index sequence)."""
    def walk(node: int):
        if node == source:
            yield [source], []
            return
        for parent, edge_index in parents[node]:
            for path_vertices, path_edges in walk(parent):
                yield path_vertices + [node], path_edges + [edge_index]

    yield from walk(vertex)


def _horton_candidates(edges: List[Tuple[int, int]], n_atoms: int,
                       max_size: int) -> Dict[int, Tuple[int, List[int]]]:
    """Horton's (1987) candidate cycles: for every vertex and every edge, a shortest cycle.

    For source ``s`` and edge ``(x, y)``, the candidate is a shortest s->x path, the edge, and a
    shortest y->s path back, kept only when the two paths touch nowhere but ``s`` (otherwise it
    is not a simple cycle). Horton's theorem is that this set, taken over every vertex, is
    guaranteed to contain a minimum cycle basis; the same shortest-path construction is also
    Vismara's own candidate generator, which is why it is enough for the fuller *relevant* set
    computed in ``_vismara_relevant``.

    Keyed by the cycle's GF(2) edge vector, so two routes that close the same cycle collapse to
    one candidate.
    """
    adjacency = _adjacency_with_edges(edges, n_atoms)
    candidates: Dict[int, Tuple[int, List[int]]] = {}
    max_depth = max(max_size - 1, 0)

    for source in range(n_atoms):
        depth, parents = _shortest_path_forest(adjacency, source, max_depth)
        for edge_index, (x, y) in enumerate(edges):
            if x not in depth or y not in depth:
                continue
            length = depth[x] + depth[y] + 1
            if length < 3 or length > max_size:
                continue
            for vx, ex in _enumerate_shortest_paths(parents, source, x):
                for vy, ey in _enumerate_shortest_paths(parents, source, y):
                    if not set(vx[1:]).isdisjoint(vy[1:]):
                        continue  # the two arms touch away from the source: not a simple cycle
                    edge_mask = 1 << edge_index
                    for e in ex:
                        edge_mask ^= 1 << e
                    for e in ey:
                        edge_mask ^= 1 << e
                    if edge_mask not in candidates:
                        vertices = vx + list(reversed(vy))[:-1]
                        candidates[edge_mask] = (length, vertices)
    return candidates


def _gf2_reduce(vector: int, basis: Dict[int, int]) -> int:
    """Reduce ``vector`` against a row-echelon GF(2) basis; 0 means it is in the basis's span."""
    while vector:
        pivot = vector.bit_length() - 1
        row = basis.get(pivot)
        if row is None:
            return vector
        vector ^= row
    return 0


def _gf2_insert(vector: int, basis: Dict[int, int]) -> bool:
    """Reduce and, if independent, add to the basis. Returns whether the rank grew."""
    reduced = _gf2_reduce(vector, basis)
    if reduced:
        basis[reduced.bit_length() - 1] = reduced
        return True
    return False


def _vismara_relevant(candidates: Dict[int, Tuple[int, List[int]]], mu: int) -> List[Tuple[int, List[int]]]:
    """Vismara's (1997) relevant cycles: every candidate outside the span of strictly shorter ones.

    Processed by non-decreasing length, so "strictly shorter" has an unambiguous meaning: every
    cycle already accepted at a shorter length is frozen into one basis *before* any cycle of the
    next length is tested against it. That is what lets several equally-short alternatives all be
    relevant at once — a cubane's six square faces, say, even though only five of them are needed
    to span the cycle space — without one of them silently voting the others out.
    """
    by_length: Dict[int, List[Tuple[int, List[int]]]] = {}
    for edge_mask, (length, vertices) in candidates.items():
        by_length.setdefault(length, []).append((edge_mask, vertices))

    basis: Dict[int, int] = {}
    rank = 0
    relevant: List[Tuple[int, List[int]]] = []
    for length in sorted(by_length):
        if rank >= mu:
            break
        accepted_this_length: List[int] = []
        for edge_mask, vertices in by_length[length]:
            if _gf2_reduce(edge_mask, basis):
                relevant.append((edge_mask, vertices))
                accepted_this_length.append(edge_mask)
        for edge_mask in accepted_this_length:
            if _gf2_insert(edge_mask, basis):
                rank += 1
    return relevant


def ring_analysis(structure: Structure, result: StructuralResult,
                  max_size: int = MAX_RING_SIZE,
                  bonds=None) -> Tuple[List[Ring], RingTopology]:
    """Vismara's (1997) relevant cycles of the bond graph, and how complete that set is.

    ``bonds`` restricts the graph the search walks; it defaults to every detected bond. Pass a
    subset to ask a narrower question — see ``metal_ligand_ring_analysis``.
    """
    n_atoms = len(structure.atoms)
    bonds = list(result.bonds) if bonds is None else list(bonds)
    edges = _canonical_edges(bonds)
    n_components = _count_components(edges, n_atoms)
    mu = len(edges) - n_atoms + n_components

    if not edges or mu <= 0:
        topology = RingTopology(
            n_vertices=n_atoms, n_edges=len(edges), n_components=n_components,
            cyclomatic_number=max(mu, 0), rank=0, is_complete=mu <= 0,
        )
        return [], topology

    candidates = _horton_candidates(edges, n_atoms, max_size)
    relevant = _vismara_relevant(candidates, mu)
    rings: List[Ring] = []
    for edge_mask, vertices in relevant:
        elements = sorted(structure.atoms[index] for index in vertices)
        composition = "".join(
            f"{element}{elements.count(element)}" if elements.count(element) > 1 else element
            for element in sorted(set(elements))
        )
        rings.append(Ring(
            atom_indices=list(vertices),
            atom_labels=[structure.labels[index] for index in vertices],
            size=len(vertices),
            composition=composition,
        ))
    rings.sort(key=lambda ring: (ring.size, ring.atom_indices))

    basis: Dict[int, int] = {}
    rank = 0
    for edge_mask, _ in relevant:
        if _gf2_insert(edge_mask, basis):
            rank += 1
    topology = RingTopology(
        n_vertices=n_atoms, n_edges=len(edges), n_components=n_components,
        cyclomatic_number=mu, rank=rank, is_complete=rank == mu,
    )
    return rings, topology


def find_rings(structure: Structure, result: StructuralResult,
               max_size: int = MAX_RING_SIZE,
               bonds=None) -> List[Ring]:
    """Vismara's (1997) relevant cycles of the bond graph — see the module docstring."""
    rings, _ = ring_analysis(structure, result, max_size=max_size, bonds=bonds)
    return rings


def metal_ligand_ring_analysis(structure: Structure, result: StructuralResult,
                               max_size: int = MAX_RING_SIZE,
                               metal_elements: Optional[List[str]] = None,
                               ) -> Tuple[List[Ring], RingTopology]:
    """Rings and topology of the metal-ligand subgraph — see ``metal_ligand_rings``."""
    if metal_elements is None:
        from cadmia.structure.core.config import load_config
        metal_elements = result.metal_elements or load_config().metal_elements
    metals = set(metal_elements)
    bonds = [bond for bond in result.bonds
             if (structure.atoms[bond.atom_i] in metals) != (structure.atoms[bond.atom_j] in metals)]
    return ring_analysis(structure, result, max_size=max_size, bonds=bonds)


def metal_ligand_rings(structure: Structure, result: StructuralResult,
                       max_size: int = MAX_RING_SIZE,
                       metal_elements: Optional[List[str]] = None) -> List[Ring]:
    """Rings of the metal-ligand subgraph: metal-metal contacts are not walked.

    This is the framework topology a chemist means by "four-membered ring" in an oxide. Over
    the full graph the same structure reports only triangles, because every metal-oxygen ring
    is cut across by a metal-metal contact.
    """
    rings, _ = metal_ligand_ring_analysis(structure, result, max_size=max_size,
                                          metal_elements=metal_elements)
    return rings


def ring_statistics(rings: List[Ring]) -> Dict[str, object]:
    """How many rings of each size and of each composition."""
    by_size: Dict[int, int] = {}
    by_composition: Dict[str, int] = {}
    for ring in rings:
        by_size[ring.size] = by_size.get(ring.size, 0) + 1
        by_composition[ring.composition] = by_composition.get(ring.composition, 0) + 1
    return {
        "total": len(rings),
        "by_size": dict(sorted(by_size.items())),
        "by_composition": dict(sorted(by_composition.items())),
        "smallest": min((ring.size for ring in rings), default=0),
    }


def global_shape(structure: Structure) -> GlobalShape:
    """Gyration-tensor shape descriptors of the whole structure."""
    positions = np.asarray(structure.positions, dtype=float)
    n_atoms = len(positions)
    if n_atoms == 0:
        raise ValueError("an empty structure has no shape.")

    centered = positions - positions.mean(axis=0)
    tensor = (centered.T @ centered) / n_atoms
    moments = np.linalg.eigvalsh(tensor)          # eigvalsh returns them ascending
    l1, l2, l3 = (float(value) for value in moments)
    rg2 = l1 + l2 + l3
    asphericity = l3 - 0.5 * (l1 + l2)
    acylindricity = l2 - l1
    anisotropy = ((asphericity ** 2 + 0.75 * acylindricity ** 2) / (rg2 ** 2)) if rg2 > 0 else 0.0

    if n_atoms > 1:
        deltas = centered[:, None, :] - centered[None, :, :]
        max_diameter = float(np.sqrt((deltas ** 2).sum(axis=-1)).max())
    else:
        max_diameter = 0.0

    hull = _hull_volume(positions)

    return GlobalShape(
        n_atoms=n_atoms,
        radius_of_gyration=float(np.sqrt(rg2)),
        principal_moments=(l1, l2, l3),
        asphericity=float(asphericity),
        acylindricity=float(acylindricity),
        anisotropy=float(min(1.0, max(0.0, anisotropy))),
        max_diameter=max_diameter,
        hull_volume=float(hull),
    )


# The fallback hull enumerates every triple of points, which is fine for the 3 to 8 vertices of
# a coordination polyhedron and quadratic-times-linear for a whole structure: 100 atoms is 161700
# triples. Above this many atoms it is skipped rather than silently taking seconds, unless scipy
# is installed, in which case the proper O(n log n) hull is used and there is no limit.
HULL_BRUTE_FORCE_LIMIT = 40


def _hull_volume(positions: np.ndarray) -> float:
    """Convex hull volume of the whole structure (A^3), NaN when it is not worth computing."""
    n_atoms = len(positions)
    if n_atoms < 4:
        return float("nan")
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        if n_atoms > HULL_BRUTE_FORCE_LIMIT:
            return float("nan")
        from cadmia.structure.structural.polyhedra import convex_hull_volume
        return convex_hull_volume(positions)
    try:
        return float(ConvexHull(np.asarray(positions, dtype=float)).volume)
    except Exception:
        return float("nan")          # coplanar or degenerate: no volume to report


def coordination_outliers(result: StructuralResult,
                          min_atoms_for_z: int = 4) -> List[Dict[str, object]]:
    """Each atom's ECN against the mean ECN of its own element in this structure.

    An absolute ECN cannot say whether an atom is undercoordinated, because the answer depends
    on the element and on the cluster: the same ECN is low for an Al in a dense framework and
    normal for one on an edge. The deviation from the element's own mean does say it, and it
    makes an undercoordinated site visible without reading the whole table.

    The mean is the mean **of this structure**, so a deviation of zero means "ordinary here", not
    "ordinary in general": if every atom of an element is undercoordinated, none of them stands
    out. Read the absolute ``ecn`` alongside.

    ``z`` is reported only from ``min_atoms_for_z`` atoms of the element upward. Below that it is
    arithmetic, not statistics: with two atoms z is exactly +-0.707 whatever their ECN, and with
    three it cannot exceed 1.155, and a correlation read off so few distinct values means
    nothing.
    """
    by_element: Dict[str, List[float]] = {}
    for descriptor in result.local_descriptors:
        by_element.setdefault(descriptor.element, []).append(descriptor.ecn)

    rows: List[Dict[str, object]] = []
    for descriptor in result.local_descriptors:
        values = np.asarray(by_element[descriptor.element], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        deviation = descriptor.ecn - mean
        usable = values.size >= min_atoms_for_z and std > 0
        rows.append({
            "atom_index": descriptor.atom_index,
            "label": descriptor.label,
            "element": descriptor.element,
            "ecn": descriptor.ecn,
            "element_mean_ecn": mean,
            "element_count": int(values.size),
            "deviation": deviation,
            "z": (deviation / std) if usable else float("nan"),
        })
    return rows
