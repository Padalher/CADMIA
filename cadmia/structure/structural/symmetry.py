"""Point group, rotational symmetry number and chirality detected from the geometry.

Gaussian keeps the point group of its input geometry through an optimization and never raises
it. An input slightly off a symmetry therefore converges to a symmetric minimum that Gaussian
still treats as C1, and its G carries σ = 1, for example a C2 or C2h cluster that the program
labels C1. ``G`` is off by RT·ln σ (0.41 kcal/mol for σ = 2 at
298 K), and a C1 label also makes a structure look chiral. Detecting the symmetry from the final
geometry, with a declared tolerance, removes the dependence on how the input was written.

An operation is accepted when it maps every atom onto an atom of the same element within
``tol`` (Å). The tolerance is far below any interatomic distance, so the nearest image is the
partner, and the mapping must be a permutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from cadmia.core.elements import atomic_number

DEFAULT_TOLERANCE = 0.01   # Å
_MAX_ORDER = 6


@dataclass
class Symmetry:
    point_group: str
    sigma: int            # rotational symmetry number: order of the proper-rotation subgroup
    chiral: bool          # no improper operation (mirror, inversion, S2n)
    tolerance: float


def _maps(x: np.ndarray, groups: List[np.ndarray], op: np.ndarray, tol: float) -> bool:
    y = x @ op.T
    for idx in groups:
        d = np.linalg.norm(x[idx][:, None] - y[idx][None], axis=-1)
        nearest = d.argmin(axis=0)
        if d[nearest, np.arange(len(idx))].max() > tol or len(set(nearest)) != len(idx):
            return False
    return True


def _rotation(u: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _mirror(u: np.ndarray) -> np.ndarray:
    return np.eye(3) - 2.0 * np.outer(u, u)


def _candidate_axes(x: np.ndarray, inertia_axes: np.ndarray) -> List[np.ndarray]:
    # Cost: O(N²) candidate axes, each tested with O(N²) matching: ~1 s at 40 atoms. Prune
    # by the degeneracy of the inertia tensor if clusters of hundreds of atoms need it.
    vectors = list(inertia_axes.T) + list(x)
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            vectors += [x[i] + x[j], x[i] - x[j], np.cross(x[i], x[j])]
    axes: List[np.ndarray] = []
    for v in vectors:
        norm = np.linalg.norm(v)
        if norm < 1e-3:
            continue
        u = v / norm
        if not any(abs(u @ w) > 0.9995 for w in axes):
            axes.append(u)
    return axes


def detect_symmetry(atoms: Sequence[str], positions, tol: float = DEFAULT_TOLERANCE) -> Symmetry:
    """Schoenflies point group (molecular groups up to order-6 axes), σ and chirality."""
    x = np.asarray(positions, dtype=float)
    masses = np.array([atomic_number(a) or 1 for a in atoms], dtype=float)   # any per-element weight is invariant
    x = x - (masses[:, None] * x).sum(axis=0) / masses.sum()
    groups = [np.array([i for i, a in enumerate(atoms) if a == e]) for e in sorted(set(atoms))]
    if len(atoms) == 1:
        return Symmetry("Kh", 1, False, tol)
    inertia = sum(m * (r @ r * np.eye(3) - np.outer(r, r)) for m, r in zip(masses, x))
    moments, principal = np.linalg.eigh(inertia)
    if moments[0] < 1e-6 * moments[-1]:                       # linear: C∞ axis
        if _maps(x, groups, -np.eye(3), tol):
            return Symmetry("D∞h", 2, False, tol)
        return Symmetry("C∞v", 1, False, tol)
    axes = _candidate_axes(x, principal)

    rotations = []          # (order, axis)
    mirrors = []            # normals
    s2n = []                # (2n, axis)
    for u in axes:
        order = next((n for n in range(_MAX_ORDER, 1, -1)
                      if _maps(x, groups, _rotation(u, 2 * np.pi / n), tol)), 1)
        if order > 1:
            rotations.append((order, u))
        if _maps(x, groups, _mirror(u), tol):
            mirrors.append(u)
        for n in range(_MAX_ORDER, 1, -1):
            if _maps(x, groups, _mirror(u) @ _rotation(u, np.pi / n), tol):
                s2n.append((2 * n, u))
                break
    inversion = _maps(x, groups, -np.eye(3), tol)
    sigma = 1 + sum(order - 1 for order, _ in rotations)
    chiral = not (mirrors or inversion or s2n)
    return Symmetry(_schoenflies(rotations, mirrors, s2n, inversion), sigma, chiral, tol)


def _schoenflies(rotations, mirrors, s2n, inversion) -> str:
    c3 = sum(1 for order, _ in rotations if order == 3)
    c4 = sum(1 for order, _ in rotations if order == 4)
    if c3 >= 2:                                     # cubic groups
        if c4:
            return "Oh" if inversion else "O"
        if inversion:
            return "Th"
        return "Td" if mirrors else "T"
    if not rotations:
        if mirrors:
            return "Cs"
        return "Ci" if inversion else "C1"
    n, axis = max(rotations, key=lambda r: r[0])
    perp_c2 = sum(1 for order, u in rotations if order == 2 and abs(u @ axis) < 1e-3)
    sigma_h = any(abs(abs(m @ axis) - 1) < 1e-3 for m in mirrors)
    sigma_v = any(abs(m @ axis) < 1e-3 for m in mirrors)
    if perp_c2 >= n:
        if sigma_h:
            return f"D{n}h"
        return f"D{n}d" if sigma_v else f"D{n}"
    if sigma_h:
        return f"C{n}h"
    if sigma_v:
        return f"C{n}v"
    if any(abs(abs(u @ axis) - 1) < 1e-3 and order == 2 * n for order, u in s2n):
        return f"S{2 * n}"
    return f"C{n}"
