"""Nuclear Fukui function and its second derivative.

Every local descriptor CADMIA computes so far is a derivative of the electron density with
respect to N at FIXED nuclei. Conceptual DFT has a second axis this ignores: the nuclear
Fukui function

    Φ_A = (∂F_A/∂N)_v = -(∂μ/∂R_A)_N          [Cohen, Ganduglia-Pirovano & Kudrnovský,
                                               J. Chem. Phys. 101, 8988 (1994)]

which measures how the FORCE on nucleus A changes when the system gains or loses an
electron. It is a Maxwell relation between the electronic and the nuclear response, it has
been published for three decades, and no common tool emits it.

Its second derivative, Φ⁽²⁾_A = (∂²F_A/∂N²)_v = ∂³E/∂N²∂R_A, is the nuclear analogue of the
dual descriptor, obtained as the second difference F(N+1) - 2F(N) + F(N-1) from the very same
three gradients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from cadmia.io.engrad import EngradResult

_GEOMETRY_TOL_BOHR = 1e-4   # the three states must be single points on ONE geometry


@dataclass
class NuclearFukuiResult:
    """Nuclear Fukui function Φ_A and, with the neutral, Φ⁺, Φ⁻ and Φ⁽²⁾."""

    phi:            np.ndarray            # (N, 3) central difference, Eh/bohr per electron
    phi_plus:       Optional[np.ndarray] = None   # F(N+1) - F(N), needs the neutral
    phi_minus:      Optional[np.ndarray] = None   # F(N) - F(N-1)
    phi_dual:       Optional[np.ndarray] = None   # (N,3) ∂²F/∂N², the nuclear analogue
                                                  # of the dual descriptor; needs the neutral
    magnitude:      Optional[np.ndarray] = None   # (N,) ‖Φ_A‖
    warnings:       List[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.magnitude is None:
            self.magnitude = np.linalg.norm(self.phi, axis=1)


def _check_same_geometry(a: EngradResult, b: EngradResult, label: str,
                         warnings: List[str]) -> None:
    if a.n_atoms != b.n_atoms:
        raise ValueError(
            f"{label}: {a.n_atoms} atoms vs {b.n_atoms}; the charged states must be single "
            "points on the SAME geometry as the neutral")
    if list(a.atomic_numbers) != list(b.atomic_numbers):
        raise ValueError(f"{label}: the atom order differs between the two files")
    drift = float(np.abs(a.positions_bohr - b.positions_bohr).max())
    if drift > _GEOMETRY_TOL_BOHR:
        warnings.append(
            f"{label}: the geometries differ by up to {drift:.2e} bohr. Φ is defined at "
            "fixed nuclei; a relaxed charged state makes it meaningless.")


def nuclear_fukui(
    anion: EngradResult,
    cation: EngradResult,
    neutral: Optional[EngradResult] = None,
) -> NuclearFukuiResult:
    """Φ_A from the charged states' gradients.

    Parameters
    ----------
    anion   : .engrad of the N+1 state, at the neutral's geometry.
    cation  : .engrad of the N-1 state, at the same geometry.
    neutral : optional .engrad of N. Only needed for the one-sided Φ⁺ and Φ⁻, which differ
              from each other — the derivative is discontinuous at integer N, exactly as it
              is for the electronic Fukui function. Reported separately rather than averaged
              away.

    Returns Φ in Eh/bohr per electron.
    """
    warnings: List[str] = []
    _check_same_geometry(anion, cation, "anion vs cation", warnings)

    f_plus, f_minus = anion.force(), cation.force()
    phi = 0.5 * (f_plus - f_minus)          # central difference over ΔN = 2

    phi_p = phi_m = None
    if neutral is not None:
        _check_same_geometry(neutral, anion, "neutral vs anion", warnings)
        f_0 = neutral.force()
        residual = float(np.linalg.norm(f_0, axis=1).max())
        phi_p = f_plus - f_0
        phi_m = f_0 - f_minus

    # ∂²F_A/∂N² = F(N+1) - 2F(N) + F(N-1) = Φ⁺ - Φ⁻, a third
    # derivative of E — ∂³E/∂N²∂R_A — the nuclear analogue of the dual descriptor, and it
    # falls out of the same three gradients at no extra cost.
    phi_dual = (phi_p - phi_m) if phi_p is not None else None

    if neutral is not None and phi_dual is not None:
        # The residual force cancels in the central difference but enters Φ⁽²⁾ with weight −2,
        # so what matters is not its absolute size but its size NEXT TO Φ⁽²⁾. An absolute 1e-2
        # cut lets a 5e-3 residual through unremarked while it still supplies a fifth of the
        # number. Over a geometry that is not the functional's own minimum, ‖2F(N)‖ can be
        # most of ‖Φ⁽²⁾‖.
        scale = float(np.linalg.norm(phi_dual, axis=1).mean())
        weight = (2.0 * float(np.linalg.norm(f_0, axis=1).mean()) / scale) if scale > 0 else 0.0
        if weight > 0.10:
            warnings.append(
                f"the neutral is not a stationary point: its residual force contributes about "
                f"{weight:.0%} of Φ⁽²⁾ (max {residual:.2e} Eh/bohr). Φ⁽²⁾ there measures the "
                "relaxation the geometry still owes as much as the response to charge — "
                "re-optimize at this level of theory before reading it.")
        elif residual > 1e-2:
            warnings.append(
                f"the neutral state carries a residual force of {residual:.2e} Eh/bohr: its "
                "geometry is not a stationary point, so Φ mixes the charge response with "
                "the relaxation the neutral itself still owes.")

    return NuclearFukuiResult(phi=phi, phi_plus=phi_p, phi_minus=phi_m,
                              phi_dual=phi_dual, warnings=warnings)


def directional_phi(phi: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Φ projected on a direction: Φ_A·û, one axis per atom or one shared axis.

    Keeps the SIGN, which the magnitude ‖Φ_A‖ throws away: a nucleus pulled INTO the cluster
    when it gains an electron is the opposite case to one pushed out, and the two say opposite
    things about whether the site will yield to an approaching probe.

    The natural axis is the site's own outward normal, which is also the direction the probe
    is made to approach along. Fix the axis before correlating, never after seeing the result.
    """
    u = np.asarray(axis, dtype=float)
    if u.ndim == 1:
        return phi @ (u / np.linalg.norm(u))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", phi, u)
