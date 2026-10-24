"""Two-species charge transfer within conceptual DFT (Parr–Pearson, one parabola).

When two closed subsystems A and B are brought into contact, electrons flow from the
species with the higher chemical potential (lower electronegativity) to the one with the
lower chemical potential, until the potentials equalize. The amount transferred and the
associated second-order stabilization follow from modelling each subsystem's energy as a
parabola in its electron number.

Sign/convention. CADMIA uses η = I − A throughout (no ½ factor, Multiwfn convention), which
makes the electrophilicity index ω = μ²/(2η) exact. The energy of a subsystem displaced by
δ electrons is then E(δ) = E₀ + μ·δ + ½·η·δ². Letting ΔN electrons flow A → B and minimizing
the total energy gives, self-consistently with that ω:

    ΔN = (μ_A − μ_B) / (η_A + η_B)              electrons transferred A → B  (>0 ⇒ A donates)
    ΔE = −(μ_A − μ_B)² / (2(η_A + η_B))         second-order stabilization  (≤ 0)

Because these η are I − A (twice the ∂²E/∂N² hardness), the denominators carry no extra factor
of 2: numerically ΔN equals the textbook (μ_A − μ_B)/(2(η_A + η_B)) written with the ½-hardness.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChargeTransfer:
    """Charge transfer between two species A and B (Parr–Pearson, η = I − A convention)."""

    label_a: str
    label_b: str
    mu_a_ev: float
    eta_a_ev: float
    mu_b_ev: float
    eta_b_ev: float
    delta_n: float        # electrons transferred A → B (>0 ⇒ A donates to B)
    delta_e_ev: float     # second-order charge-transfer stabilization (≤ 0)
    donor: str            # label of the electron donor
    acceptor: str         # label of the electron acceptor


def compute_charge_transfer(
    mu_a_ev: float,
    eta_a_ev: float,
    mu_b_ev: float,
    eta_b_ev: float,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> ChargeTransfer:
    """Charge transferred from A to B and its stabilization energy (eV).

    ``eta_a_ev`` and ``eta_b_ev`` are the hardnesses in the η = I − A convention (both must be
    positive). ``delta_n`` is the number of electrons flowing A → B: positive means A donates.
    """
    eta_a, eta_b = float(eta_a_ev), float(eta_b_ev)
    if eta_a <= 0.0 or eta_b <= 0.0:
        raise ValueError("Charge transfer requires positive hardness for both species.")

    mu_a, mu_b = float(mu_a_ev), float(mu_b_ev)
    denom = eta_a + eta_b
    delta_n = (mu_a - mu_b) / denom
    delta_e = -((mu_a - mu_b) ** 2) / (2.0 * denom)

    if delta_n >= 0.0:
        donor, acceptor = label_a, label_b
    else:
        donor, acceptor = label_b, label_a

    return ChargeTransfer(
        label_a=label_a,
        label_b=label_b,
        mu_a_ev=mu_a,
        eta_a_ev=eta_a,
        mu_b_ev=mu_b,
        eta_b_ev=eta_b,
        delta_n=delta_n,
        delta_e_ev=delta_e,
        donor=donor,
        acceptor=acceptor,
    )
