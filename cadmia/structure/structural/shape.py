"""Continuous shape measures (CShM) for coordination environments.

The distortion indices in :mod:`polyhedra` compare a centre against the *ideal* of its own
type, and need a tabulated reference angle per type. A small cluster rarely sits near an
ideal, so most centres fall into shapes that have no entry and every index (Robinson's
sigma^2, <lambda>) comes back NaN.

A continuous shape measure asks a different question. Instead of "how far is this octahedron
from a perfect octahedron", it asks "how far is this arrangement from each candidate shape",
minimising over rotation, scale and the assignment of ligands to vertices. Every centre gets
a number against every reference, so nothing is ever undefined, and the smallest one both
names the shape and says how good the match is:

    S(Q, P) = 100 * min  sum_i |Q_i - s R P_perm(i)|^2 / sum_i |Q_i - Q_0|^2

S = 0 is an exact match and larger is worse. Reference: Pinsky & Avnir, Inorg. Chem. 1998,
37, 5575; Casanova et al., Chem. Eur. J. 2003, 9, 1281 (the SHAPE program).

Two implementation points worth stating, because both are easy to get wrong:

**The central atom is part of the point set.** Without it a trigonal pyramid and a trigonal
plane have the same ligands — an equilateral triangle — and no measure over the ligands alone
can tell them apart. Q and P therefore carry N+1 points, and only the N ligands are permuted.

**The rotation never has to be built.** With P normalised to unit size, the optimal scale is
the Kabsch trace, so the whole minimisation over rotation and scale collapses to

    S = 100 * (1 - (sum of singular values)^2 / |Q|^2)

with the singular values of P^T Q and the usual reflection correction. That is one 3x3 SVD
per permutation instead of an explicit superposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Beyond this many ligands the brute-force permutation search (N! superpositions) stops being
# free: 8 ligands is 40320 SVDs and still well under a second, 9 would be nine times that for
# a coordination number these clusters do not reach.
MAX_LIGANDS = 8


def _normalise(vertices: np.ndarray) -> np.ndarray:
    """Unit-distance vertices around a centre at the origin."""
    v = np.asarray(vertices, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _ring(n: int, z: float = 0.0) -> np.ndarray:
    """``n`` points evenly spaced on a circle at height ``z`` (before normalisation)."""
    angles = np.arange(n) * (2.0 * np.pi / n)
    return np.column_stack((np.cos(angles), np.sin(angles), np.full(n, z)))


_TETRAHEDRON = _normalise(np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                                    [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]))
_OCTAHEDRON = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
_TRIGONAL_BIPYRAMID = np.vstack((_ring(3), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
_PENTAGONAL_BIPYRAMID = np.vstack((_ring(5), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
# The reference trigonal prism has ALL its edges equal, so its lateral faces are squares. With
# the triangle at unit circumradius the triangular edge is sqrt(3), so the height must be sqrt(3)
# too: z = +-sqrt(3)/2. Using z = +-1 gives a prism 15% too tall, which is a different shape --
# it scores 17.51 against the octahedron where the literature value is 16.74.
_PRISM_HALF_HEIGHT = sqrt(3.0) / 2.0
_TRIGONAL_PRISM = _normalise(np.vstack((
    np.column_stack((_ring(3)[:, 0], _ring(3)[:, 1], np.full(3, _PRISM_HALF_HEIGHT))),
    np.column_stack((_ring(3)[:, 0], _ring(3)[:, 1], np.full(3, -_PRISM_HALF_HEIGHT))),
)))

# Reference shapes by coordination number, in the Alvarez naming. The "vacant" ones are literal
# vertex subsets of their parent polyhedron, which is what makes them unambiguous: a square
# pyramid here is an octahedron minus one vertex, not a pyramid with a fitted apex angle.
#
# That is also why they are named vOC-5 and vPBPY-6 rather than SHAPE's SPY-5 and PPY-6. Those
# two are the *spherical* pyramids, whose apex-centre-base angle is fitted (about 104.5 degrees
# for SPY-5) instead of inherited from the parent. Borrowing the name for a different shape
# would make the numbers look comparable to published ones when they are not.
_SHAPES: Dict[int, Dict[str, np.ndarray]] = {
    2: {
        "L-2": np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),          # linear
        "vT-2": _TETRAHEDRON[:2],                                       # bent (divacant tetrahedron)
    },
    3: {
        "TP-3": _ring(3),                                               # trigonal planar
        "vT-3": _TETRAHEDRON[:3],                                       # trigonal pyramid
        "TS-3": np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),   # T-shaped
    },
    4: {
        "T-4": _TETRAHEDRON,                                            # tetrahedron
        "SP-4": _ring(4),                                               # square planar
        "SS-4": _OCTAHEDRON[[0, 2, 4, 1]],                              # seesaw (cis-divacant OC)
        "vTBPY-4": _TRIGONAL_BIPYRAMID[[0, 1, 2, 3]],                   # axially vacant TBPY
    },
    5: {
        "TBPY-5": _TRIGONAL_BIPYRAMID,                                  # trigonal bipyramid
        "vOC-5": _OCTAHEDRON[:5],                                       # vacant octahedron
        "PP-5": _ring(5),                                               # pentagon
    },
    6: {
        "OC-6": _OCTAHEDRON,                                            # octahedron
        "TPR-6": _TRIGONAL_PRISM,                                       # trigonal prism
        "vPBPY-6": _PENTAGONAL_BIPYRAMID[:6],                           # vacant pentagonal bipyramid
        "HP-6": _ring(6),                                               # hexagon
    },
    7: {
        "PBPY-7": _PENTAGONAL_BIPYRAMID,                                # pentagonal bipyramid
        "HP-7": _ring(7),                                               # heptagon
    },
}


@dataclass
class ShapeMeasure:
    """The CShM of one centre against every reference shape of its coordination number.

    ``best_shape`` is the closest reference and ``best_value`` its measure; ``measures`` holds
    all of them, so a centre that sits between two shapes is visible as two similar values
    rather than hidden behind the winner.
    """

    center_index: int
    cn: int
    best_shape: str
    best_value: float
    measures: Dict[str, float]


def available_shapes(cn: int) -> List[str]:
    """Reference shape names defined for coordination number ``cn`` (empty when none)."""
    return sorted(_SHAPES.get(cn, {}))


def continuous_shape_measure(center: Sequence[float],
                             ligands: np.ndarray,
                             reference: np.ndarray) -> float:
    """CShM of one centre+ligand set against one reference shape, in [0, 100].

    ``reference`` holds the N ideal vertex directions around a centre at the origin; the
    centre itself is added here, so both point sets are N+1 long and only the ligands are
    permuted. 0 is an exact match.
    """
    ligands = np.asarray(ligands, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n = len(ligands)
    if n != len(reference):
        raise ValueError(f"reference has {len(reference)} vertices but the centre has {n} ligands.")
    if n > MAX_LIGANDS:
        raise ValueError(f"continuous shape measures are capped at {MAX_LIGANDS} ligands, got {n}.")

    Q = np.vstack((np.asarray(center, dtype=float), ligands))
    Q = Q - Q.mean(axis=0)
    q_norm2 = float((Q * Q).sum())
    if q_norm2 <= 0.0:
        return float("nan")

    # Centring and scaling are done once: every permutation is the same set of points, so the
    # centroid and the norm do not depend on the order. Only the rows are shuffled afterwards,
    # and row 0 — the centre — stays put.
    base = np.vstack((np.zeros(3), reference))
    base = base - base.mean(axis=0)
    p_norm = sqrt(float((base * base).sum()))
    if p_norm <= 0.0:
        return float("nan")
    base = base / p_norm

    best = 0.0
    for perm in permutations(range(1, n + 1)):
        P = base[(0,) + perm, :]
        # Kabsch trace with the reflection correction: the largest attainable <Q, R P> is the
        # sum of the singular values of P^T Q, with the smallest flipped if R would reflect.
        u, singular, vt = np.linalg.svd(P.T @ Q)
        if np.linalg.det(vt.T @ u.T) < 0:
            trace = float(singular[:-1].sum() - singular[-1])
        else:
            trace = float(singular.sum())
        best = max(best, trace)

    # |P| = 1, so the optimal scale is the trace itself and the residual is |Q|^2 - trace^2.
    return float(max(0.0, 100.0 * (1.0 - (best * best) / q_norm2)))


def shape_measures(center: Sequence[float], ligands: np.ndarray,
                   center_index: int = -1) -> Optional[ShapeMeasure]:
    """All CShM of one centre against the references of its coordination number.

    Returns None when no reference set is defined for that coordination number, which is the
    honest answer: unlike the tabulated indices, a defined shape never yields NaN.
    """
    ligands = np.asarray(ligands, dtype=float)
    cn = len(ligands)
    references = _SHAPES.get(cn)
    if not references or cn > MAX_LIGANDS:
        return None

    measures = {name: continuous_shape_measure(center, ligands, ref)
                for name, ref in references.items()}
    best_shape = min(measures, key=measures.__getitem__)
    return ShapeMeasure(center_index=center_index, cn=cn, best_shape=best_shape,
                        best_value=measures[best_shape], measures=measures)
