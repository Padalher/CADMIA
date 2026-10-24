"""Global electronic descriptors."""
from __future__ import annotations

from dataclasses import dataclass

from cadmia.core.units import HARTREE_TO_EV  # re-exported; canonical home is core.units


@dataclass
class GlobalElectronic:
    """Global electronic descriptors derived from HOMO and LUMO."""

    homo_ev: float
    lumo_ev: float
    gap_ev: float
    ip_ev: float
    ea_ev: float
    mu_ev: float
    chi_ev: float
    eta_ev: float
    softness: float
    omega_ev: float
    omega_minus_ev: float   # electrodonating power ω⁻ (Gázquez-Cedillo-Vela)
    omega_plus_ev: float    # electroaccepting power ω⁺ (Gázquez-Cedillo-Vela)


def electrodonating_accepting_powers(ip_ev: float, ea_ev: float) -> tuple[float, float]:
    """Electrodonating (ω⁻) and electroaccepting (ω⁺) powers of Gázquez, Cedillo, Vela (2007).

    ω⁻ = (3I + A)² / (16(I − A)) measures the propensity to donate charge;
    ω⁺ = (I + 3A)² / (16(I − A)) the propensity to accept it. Both use η = I − A
    (no ½, matching the project's Multiwfn convention), and satisfy the exact identity
    ω⁺ − ω⁻ = μ = −(I + A)/2, a self-consistency check.

    Returns ``(omega_minus_ev, omega_plus_ev)``.  Raises ``ValueError`` if I ≤ A.
    """
    i, a = float(ip_ev), float(ea_ev)
    eta = i - a
    if eta <= 0.0:
        raise ValueError("Electrodonating/accepting powers require IP > EA (positive hardness).")
    omega_minus = (3.0 * i + a) ** 2 / (16.0 * eta)
    omega_plus = (i + 3.0 * a) ** 2 / (16.0 * eta)
    return omega_minus, omega_plus


def compute_global_descriptors(homo_ev: float, lumo_ev: float) -> GlobalElectronic:
    """Compute the Koopmans-approximated global reactivity descriptors from HOMO and LUMO.

    Koopmans' theorem gives IP ≈ −ε_HOMO and EA ≈ −ε_LUMO, so the hardness uses the same
    finite-difference definition as the SCF/ΔSCF route — η = IP − EA = LUMO − HOMO — WITHOUT the
    Parr-Pearson ½ factor, matching Multiwfn (and the project's CDFT global descriptors). η(K) is
    therefore directly comparable to η(FD); they differ only in method (frontier orbitals vs ΔSCF).
    """
    gap = float(lumo_ev - homo_ev)
    if gap <= 0.0:
        raise ValueError("LUMO must be greater than HOMO to produce a positive gap.")

    ip = -float(homo_ev)
    ea = -float(lumo_ev)
    mu = (float(homo_ev) + float(lumo_ev)) / 2.0
    chi = -mu
    eta = gap                      # η = IP − EA = LUMO − HOMO (no ½, Multiwfn convention)
    softness = 1.0 / eta
    omega = (mu**2) / (2.0 * eta)
    omega_minus, omega_plus = electrodonating_accepting_powers(ip, ea)

    return GlobalElectronic(
        homo_ev=float(homo_ev),
        lumo_ev=float(lumo_ev),
        gap_ev=gap,
        ip_ev=ip,
        ea_ev=ea,
        mu_ev=mu,
        chi_ev=chi,
        eta_ev=eta,
        softness=softness,
        omega_ev=omega,
        omega_minus_ev=omega_minus,
        omega_plus_ev=omega_plus,
    )
