"""Bond angles over the whole connectivity, not only inside a coordination polyhedron.

:mod:`polyhedra` already averages the *cis* angles of each metal centre, which answers "how
regular is this polyhedron". It cannot answer the other half of the structural question: how
the polyhedra are oriented with respect to each other. That is carried by the angle at the
*bridging* atom — the M-O-M hinge — and by the full distribution of angles at each centre
rather than a single mean.

Both come from the same object. An angle is defined for every atom with two or more bonded
neighbours, so walking the bond list once yields the ligand-centre-ligand angles (O-Al-O) and
the metal-ligand-metal ones (Al-O-Al) in the same pass. Grouping them by the element triplet
is what separates the two families in the report.

The triplet label is written with the vertex in the middle and the two arms sorted, so
``Al-O-Ti`` and ``Ti-O-Al`` are the same entry: the angle does not know which arm came first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from cadmia.core import Structure
from cadmia.structure.structural.analyzer import StructuralResult

# Numerically an angle is undefined when either arm has no length. Below this separation the
# two atoms are the same point for practical purposes and the arccos is meaningless.
_MIN_ARM = 1e-8

# An angle this wide is a *trans* pair: two ligands on opposite sides of the centre. It must be
# summarised apart from the rest, because mixing the two families makes a PERFECT polyhedron look
# distorted. An ideal octahedron has twelve angles at 90 degrees and three at 180: pooled, that is
# a mean of 108 and a standard deviation of 36 for a centre with no distortion at all. The same
# trap is why polyhedra.py averages only the cis angles.
_TRANS_CUTOFF = 150.0


@dataclass
class BondAngle:
    """One i-j-k angle in degrees, with j the vertex."""

    vertex_index: int
    left_index: int
    right_index: int
    vertex_label: str
    left_label: str
    right_label: str
    triplet: str
    angle: float


def _triplet(left: str, vertex: str, right: str) -> str:
    arms = sorted((left, right))
    return f"{arms[0]}-{vertex}-{arms[1]}"


def angle_between(vertex: Sequence[float], left: Sequence[float],
                  right: Sequence[float]) -> float:
    """Angle left-vertex-right in degrees, or NaN if either arm has no length."""
    a = np.asarray(left, dtype=float) - np.asarray(vertex, dtype=float)
    b = np.asarray(right, dtype=float) - np.asarray(vertex, dtype=float)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _MIN_ARM or nb < _MIN_ARM:
        return float("nan")
    cosine = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def bond_angles(structure: Structure, result: StructuralResult, bonds=None) -> List[BondAngle]:
    """Every angle whose two arms are detected bonds, over all atoms of the structure.

    Ordered by vertex index and then by the two arms, so the list is reproducible. ``bonds``
    restricts which bonds may serve as arms; it defaults to every detected bond.

    Note that in an oxide cluster the bond list includes metal-metal contacts, so triplets like
    ``Al-Al-O`` appear. They are angles between detected contacts, not between chemical bonds in
    the usual sense; the report says how many there are so the table is not read as if every row
    described a coordination angle.
    """
    neighbors: Dict[int, List[int]] = {}
    for bond in (result.bonds if bonds is None else bonds):
        neighbors.setdefault(bond.atom_i, []).append(bond.atom_j)
        neighbors.setdefault(bond.atom_j, []).append(bond.atom_i)

    angles: List[BondAngle] = []
    for vertex in sorted(neighbors):
        arms = sorted(set(neighbors[vertex]))
        for position, left in enumerate(arms):
            for right in arms[position + 1:]:
                value = angle_between(structure.positions[vertex],
                                      structure.positions[left],
                                      structure.positions[right])
                angles.append(BondAngle(
                    vertex_index=vertex,
                    left_index=left,
                    right_index=right,
                    vertex_label=structure.labels[vertex],
                    left_label=structure.labels[left],
                    right_label=structure.labels[right],
                    triplet=_triplet(structure.atoms[left], structure.atoms[vertex],
                                     structure.atoms[right]),
                    angle=value,
                ))
    return angles


def angle_statistics(angles: List[BondAngle],
                     trans_cutoff: float = _TRANS_CUTOFF) -> Dict[str, Dict[str, float]]:
    """Per element triplet: the cis angles summarised, and the trans ones counted apart.

    ``mean``, ``std``, ``min`` and ``max`` describe the **cis** angles only (below
    ``trans_cutoff``); ``n_trans`` and ``mean_trans`` describe the rest. Pooling the two makes an
    undistorted centre look distorted — see the note on ``_TRANS_CUTOFF`` — so the spread here is
    only comparable between centres of the same coordination number, and is never by itself
    evidence of distortion. For that, use the continuous shape measure in :mod:`shape`.

    ``std`` is NaN for a single sample rather than 0.0: one angle has no spread to report, and
    zero would read as "perfectly uniform".
    """
    grouped: Dict[str, List[float]] = {}
    for item in angles:
        if not np.isnan(item.angle):
            grouped.setdefault(item.triplet, []).append(item.angle)

    stats: Dict[str, Dict[str, float]] = {}
    for triplet, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        cis = array[array < trans_cutoff]
        trans = array[array >= trans_cutoff]
        stats[triplet] = {
            "count": int(array.size),
            "n_trans": int(trans.size),
            "mean_trans": float(trans.mean()) if trans.size else float("nan"),
            "mean": float(cis.mean()) if cis.size else float("nan"),
            "std": float(cis.std(ddof=1)) if cis.size > 1 else float("nan"),
            "min": float(cis.min()) if cis.size else float("nan"),
            "max": float(cis.max()) if cis.size else float("nan"),
        }
    return stats


def bridging_angles(structure: Structure, result: StructuralResult,
                    metal_elements: Optional[Sequence[str]] = None) -> List[BondAngle]:
    """The metal-ligand-metal subset: the hinge angles that orient one polyhedron against another.

    A shared ligand between two polyhedra is a bridge, and the angle at it is the degree of
    freedom that lets a corner-sharing framework flex without breaking a bond. Reported apart
    from the rest because it answers a different question from the angles *inside* a polyhedron.
    """
    if metal_elements is None:
        from cadmia.structure.core.config import load_config
        metal_elements = result.metal_elements or load_config().metal_elements
    metals = set(metal_elements)
    return [item for item in bond_angles(structure, result)
            if structure.atoms[item.vertex_index] not in metals
            and structure.atoms[item.left_index] in metals
            and structure.atoms[item.right_index] in metals]
