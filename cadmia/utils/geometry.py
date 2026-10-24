"""Molecular geometry utilities: RMSD, alignment, Kabsch rotation."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimum RMSD between two (N, 3) coordinate arrays after optimal rotation and translation.

    Uses the Kabsch algorithm: centers both structures, computes the optimal rotation
    matrix via SVD, and returns the resulting RMSD in the same units as the input.

    Parameters
    ----------
    P, Q : (N, 3) arrays of Cartesian coordinates. Must have the same shape.

    Returns
    -------
    float — RMSD after optimal superposition.

    Raises
    ------
    ValueError — if P and Q have different shapes or are not (N, 3).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {P.shape}")

    # Center
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    # Covariance matrix
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)

    # Handle improper rotation (reflection)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, math.copysign(1.0, d)])
    R = Vt.T @ D @ U.T

    P_rot = Pc @ R.T
    diff = P_rot - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def _superposition(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch rotation + translation that best maps P onto Q (proper rotations only)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, math.copysign(1.0, d)])
    R = Vt.T @ D @ U.T
    return R, Q.mean(axis=0) - P.mean(axis=0) @ R.T


def _greedy_pairing(dist: np.ndarray) -> List[int]:
    """One-to-one pairing of two equal-size point sets, closest available couple first.

    Greedy is not guaranteed to minimize the largest pair, so it can only over-report the
    worst distance — which for a duplicate test errs toward "different", never toward a
    wrong merge.
    """
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    pair = [-1] * dist.shape[0]
    taken: set = set()
    for i, j in order:
        if pair[i] != -1 or int(j) in taken:
            continue
        pair[int(i)] = int(j)
        taken.add(int(j))
    return pair


def kabsch_deviations(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Per-atom displacement (Å) after optimal superposition, atom order as given.

    The quantity :func:`kabsch_max_deviation` reduces to its maximum. Kept separate because
    the profile itself is the observable when the question is *which* atoms moved, not
    *whether* any did — the substrate's response to adsorption, say.

    Rigid translations and rotations give zero: the superposition removes them first.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {P.shape}")
    P_rot, _ = align(P, Q)       # align() returns P in Q's own frame
    return np.linalg.norm(P_rot - Q, axis=1)


def kabsch_max_deviation(P: np.ndarray, Q: np.ndarray,
                         symbols: "Sequence[str] | None" = None,
                         symbols_q: "Sequence[str] | None" = None) -> float:
    """Largest single-atom displacement (Å) after optimal superposition.

    The size-independent companion to :func:`kabsch_rmsd`. An RMSD averages over every
    atom, so the tolerance it needs depends on how many atoms are in the structure: in a
    cluster + probe, the probe has to move `tol · sqrt(N_total / n_probe)` before the mean
    notices, which grows with the cluster. The largest
    displacement asks the question the dedup actually means: does *every* atom coincide?

    Parameters
    ----------
    P, Q : (N, 3) arrays of Cartesian coordinates. Must have the same shape.
    symbols : optional element symbols. When given, atoms are paired by ELEMENT rather
        than by list position, so the same structure written with its atoms in a
        different order still compares as identical. `symbols_q` defaults to `symbols`.
        Returns inf when the two do not even have the same composition.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {P.shape}")

    deviations = kabsch_paired_deviations(P, Q, symbols, symbols_q)
    return float(deviations.max()) if deviations.size else float("inf")


def kabsch_paired_deviations(P: np.ndarray, Q: np.ndarray,
                             symbols: "Sequence[str] | None" = None,
                             symbols_q: "Sequence[str] | None" = None) -> np.ndarray:
    """Per-atom displacement (Å) after superposition, pairing atoms by ELEMENT when told how.

    Split out so that a maximum and an RMSD of the same comparison cannot disagree about which
    atom is which. Pairing by list position and by element give different answers whenever the
    two files write their atoms in a different order, and a caller that took the maximum from
    one and the mean from the other would report a large RMSD next to a zero deviation.

    Returns an empty array when the two do not have the same composition, or when too few atoms
    agree position by position to fit a frame on — the cases where no pairing is meaningful.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {P.shape}")

    if symbols is None:
        return kabsch_deviations(P, Q)

    sp = list(symbols)
    sq = list(symbols_q) if symbols_q is not None else sp
    if len(sp) != len(P) or len(sq) != len(Q):
        raise ValueError("symbols must have one entry per atom")
    if sorted(sp) != sorted(sq):
        return np.empty(0)             # different composition: not the same structure

    # The rotation is fitted on the atoms whose element already agrees position by
    # position — in a cluster + probe that is the cluster, which pins the frame. If too
    # few agree there is no reliable frame and the two count as different.
    anchor = [k for k in range(len(sp)) if sp[k] == sq[k]]
    if len(anchor) < min(3, len(sp)):
        return np.empty(0)    # too few atoms agree to fit a frame on
    R, t = _superposition(P[anchor], Q[anchor])

    # Fit on the atoms that already agree, pair the rest by element, then re-fit on that
    # pairing: the first frame is only as good as the agreeing atoms, and the few that do
    # not agree are exactly the ones that tilt it.
    mapping = [0] * len(sp)
    for _ in range(2):
        P_rot = P @ R.T + t
        for element in set(sp):
            rows = [k for k in range(len(sp)) if sp[k] == element]
            cols = [k for k in range(len(sq)) if sq[k] == element]
            d = np.linalg.norm(P_rot[rows][:, None, :] - Q[cols][None, :, :], axis=-1)
            for r, c in zip(rows, _greedy_pairing(d)):
                mapping[r] = cols[c]
        R, t = _superposition(P, Q[mapping])

    P_rot = P @ R.T + t
    return np.linalg.norm(P_rot - Q[mapping], axis=1)


def align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return P superimposed onto Q and the resulting RMSD.

    Applies optimal rotation and translation (Kabsch) to P so that it best
    overlaps with Q. Q is not modified.

    Returns
    -------
    (P_aligned, rmsd) — aligned copy of P and the RMSD after alignment.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")

    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)

    Pc = P - P_centroid
    Qc = Q - Q_centroid

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, math.copysign(1.0, d)])
    R = Vt.T @ D @ U.T

    P_rot = Pc @ R.T + Q_centroid
    diff = (P_rot - Q)
    rmsd = float(np.sqrt((diff ** 2).sum() / len(P)))
    return P_rot, rmsd
