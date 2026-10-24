"""Vibrational entropy, harmonic and quasi-RRHO.

Harmonic oscillator entropy diverges as ln(1/ν) when ν → 0, so modes below ~100 cm⁻¹ (hindered
rotations, large-amplitude motions of a cluster) inflate S and make G unreliable. Grimme's
quasi-RRHO (Chem. Eur. J. 18 (2012) 9955) interpolates, mode by mode, between the harmonic
entropy and that of a free rotor of the same frequency:

    S = w·S_V + (1 − w)·S_R,   w = 1 / (1 + (ν0/ν)^α),   ν0 = 100 cm⁻¹, α = 4

with the rotor moment μ = h / (8π²cν) damped by B_av = 1e-44 kg·m² (μ' = μB_av / (μ + B_av)).
These are the values ORCA 6 applies by default, so a Gaussian output corrected here and an ORCA
one read as printed share one scheme. Only S changes; ZPE and H are left as the engine printed.
"""
from __future__ import annotations

from math import exp, log, pi, sqrt
from typing import Iterable, Optional

_H = 6.62607015e-34       # J·s
_KB = 1.380649e-23        # J/K
_C = 2.99792458e10        # cm/s
_R = 8.314462618          # J/(mol·K)
_HARTREE_J_MOL = 2625499.6394799   # J/mol per Hartree

GRIMME_NU0_CM1 = 100.0
GRIMME_ALPHA = 4
GRIMME_B_AV = 1e-44       # kg·m²


def _s_harmonic(nu: float, t: float) -> float:
    x = _H * _C * nu / (_KB * t)
    return _R * (x / (exp(x) - 1.0) - log(1.0 - exp(-x)))


def _s_free_rotor(nu: float, t: float) -> float:
    mu = _H / (8.0 * pi ** 2 * _C * nu)
    mu_eff = mu * GRIMME_B_AV / (mu + GRIMME_B_AV)
    return _R * (0.5 + log(sqrt(8.0 * pi ** 3 * mu_eff * _KB * t / _H ** 2)))


def vibrational_entropy(freqs_cm1: Iterable[float], temperature: float,
                        scheme: str = "harmonic") -> float:
    """S_vib in J/(mol·K) over the real modes; imaginary (negative) ones never enter."""
    total = 0.0
    for nu in freqs_cm1:
        if nu <= 0.0:
            continue
        s_v = _s_harmonic(nu, temperature)
        if scheme == "harmonic":
            total += s_v
        elif scheme == "grimme":
            w = 1.0 / (1.0 + (GRIMME_NU0_CM1 / nu) ** GRIMME_ALPHA)
            total += w * s_v + (1.0 - w) * _s_free_rotor(nu, temperature)
        else:
            raise ValueError(f"Unknown entropy scheme {scheme!r}; expected 'harmonic' or 'grimme'.")
    return total


def gibbs_quasi_rrho(state) -> Optional[float]:
    """G with Grimme's quasi-RRHO vibrational entropy (Hartree), or None if it cannot be had.

    An output whose engine already applied it (ORCA's default) is returned as printed; a
    harmonic one (Gaussian) is corrected by −T·(S_grimme − S_harmonic) over its own frequencies.
    """
    g = state.gibbs_free_energy_hartree
    if g is None:
        return None
    if state.entropy_scheme == "grimme":
        return g
    if state.entropy_scheme != "harmonic" or not state.frequencies_cm1 or not state.temperature_kelvin:
        return None
    t = state.temperature_kelvin
    delta_s = (vibrational_entropy(state.frequencies_cm1, t, "grimme")
               - vibrational_entropy(state.frequencies_cm1, t, "harmonic"))
    return g - t * delta_s / _HARTREE_J_MOL
