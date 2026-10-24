"""Conceptual DFT calculators."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from cadmia.core.electronic_descriptors import GlobalElectronic
from cadmia.core.electronic_descriptors import HARTREE_TO_EV
from cadmia.core.electronic_descriptors import electrodonating_accepting_powers

# E(HOMO) of tetracyanoethylene (TCE) at B3LYP/6-31G* — Multiwfn reference.
# Used as the zero for the nucleophilicity index N = HOMO(molecule) - HOMO(TCE).
# Override via homo_tce_ev in compute_finite_difference_descriptors.
HOMO_TCE_EV: float = -9.1212  # Multiwfn built-in value (B3LYP/6-31G*)


@dataclass
class FukuiResult:
    """Condensed Fukui functions and local CDFT descriptors by atom."""

    atoms: List[str]
    labels: List[str]
    # Core Fukui functions
    f_plus: Optional[List[float]]          # electrophilic f+
    f_minus: Optional[List[float]]         # nucleophilic f-
    f_zero: Optional[List[float]]          # radical f0 = (f+ + f-)/2
    dual: Optional[List[float]]            # CDD = f+ - f-
    # Local softnesses (S * f)
    local_softness_plus: Optional[List[float]]   # s+ = S * f+
    local_softness_minus: Optional[List[float]]  # s- = S * f-
    local_electrophilicity: Optional[List[float]]  # ω * f+  (e·eV)
    # --- extended descriptors (default None, populated by full workflow) ---
    softness_zero: Optional[List[float]] = None          # s0 = S * f0
    local_nucleophilicity: Optional[List[float]] = None  # N_index * f-  (e·eV)
    relative_electrophilicity: Optional[List[float]] = None  # s+/s-
    relative_nucleophilicity: Optional[List[float]] = None   # s-/s+
    # η·f0. NOT a local hardness: η is a global constant here, so this is f0 rescaled and
    # its per-atom ranking is identical to f0 by construction. The consistent definitions
    # of local hardness give η(r) = η, and the η·f(r) form is the one the literature
    # discarded as a site descriptor. Kept for continuity with Multiwfn output, under a
    # name that does not invite reading it as a physical property.
    f_zero_scaled_by_hardness: Optional[List[float]] = None   # η * f0  (eV·e)
    donicity: Optional[List[float]] = None               # IP * f-  (eV·e)
    accepting_capacity: Optional[List[float]] = None     # EA * f+  (eV·e)


@dataclass
class FiniteDifferenceCDFT:
    """Global CDFT descriptors from E(N), E(N+1), and E(N-1)."""

    energy_n_hartree: float
    energy_np1_hartree: float
    energy_nm1_hartree: float
    ip_ev: float
    ea_ev: float
    mu_ev: float
    chi_ev: float
    hardness_ev: float
    softness: float
    omega_ev: float
    omega_minus_ev: float   # electrodonating power ω⁻ (Gázquez-Cedillo-Vela)
    omega_plus_ev: float    # electroaccepting power ω⁺ (Gázquez-Cedillo-Vela)
    # --- optional fields populated when HOMO/LUMO values are provided ---
    homo_n_ev: Optional[float] = None
    homo_np1_ev: Optional[float] = None
    homo_nm1_ev: Optional[float] = None
    lumo_n_ev: Optional[float] = None
    nucleophilicity_ev: Optional[float] = None      # HOMO(N) - HOMO(TCE)
    nucleophilicity_hartree: Optional[float] = None
    homo_tce_ev: Optional[float] = None             # reference value used


def compute_finite_difference_descriptors(
    energy_n_hartree: float,
    energy_np1_hartree: float,
    energy_nm1_hartree: float,
    homo_n_ev: Optional[float] = None,
    homo_np1_ev: Optional[float] = None,
    homo_nm1_ev: Optional[float] = None,
    lumo_n_ev: Optional[float] = None,
    homo_tce_ev: Optional[float] = None,
) -> FiniteDifferenceCDFT:
    """Compute global CDFT descriptors from vertical finite differences.

    E(N+1) is the anion; E(N-1) is the cation.  Hardness convention matches
    Multiwfn's "Hardness (=fundamental gap)": η = IP - EA.

    When HOMO/LUMO values are supplied, the nucleophilicity index
    N = HOMO(N) - HOMO(TCE) is computed using *homo_tce_ev* as reference
    (defaults to the Multiwfn B3LYP/6-31G* value for TCE).
    """
    ip_ev = (float(energy_nm1_hartree) - float(energy_n_hartree)) * HARTREE_TO_EV
    ea_ev = (float(energy_n_hartree) - float(energy_np1_hartree)) * HARTREE_TO_EV
    hardness_ev = ip_ev - ea_ev
    if hardness_ev <= 0.0:
        raise ValueError("Finite-difference hardness must be positive: IP must exceed EA.")

    # η > 0 is satisfied trivially even when EA < 0, so it does not check the one assumption
    # everything drawn from the N+1 state rests on: that the anion is BOUND. With a basis without
    # diffuse functions an unbound anion is confined artificially, E(N+1) comes out plausible, and
    # f+, Δf, ω+, s+ and the nuclear Φ+ are all computed from a state that does not exist.
    if ea_ev <= 0.0:
        warnings.warn(
            f"Electron affinity is {ea_ev:.3f} eV: the N+1 state is NOT bound, so everything "
            "derived from it (f+, the dual descriptor, omega+, the one-sided nuclear Fukui) has "
            "no physical meaning. Check with a basis carrying diffuse functions.",
            RuntimeWarning, stacklevel=2)
    elif ea_ev < 0.5:
        warnings.warn(
            f"Electron affinity is only {ea_ev:.3f} eV: the anion is weakly bound and everything "
            "derived from it is sensitive to the basis set. Worth confirming with diffuse "
            "functions before quoting f+ or the dual descriptor.",
            RuntimeWarning, stacklevel=2)

    mu_ev = -(ip_ev + ea_ev) / 2.0
    chi_ev = -mu_ev
    softness = 1.0 / hardness_ev
    omega_ev = (mu_ev**2) / (2.0 * hardness_ev)
    omega_minus_ev, omega_plus_ev = electrodonating_accepting_powers(ip_ev, ea_ev)

    tce = float(homo_tce_ev) if homo_tce_ev is not None else HOMO_TCE_EV
    nucleophilicity_ev = None
    nucleophilicity_hartree = None
    if homo_n_ev is not None:
        nucleophilicity_ev = float(homo_n_ev) - tce
        nucleophilicity_hartree = nucleophilicity_ev / HARTREE_TO_EV

    return FiniteDifferenceCDFT(
        energy_n_hartree=float(energy_n_hartree),
        energy_np1_hartree=float(energy_np1_hartree),
        energy_nm1_hartree=float(energy_nm1_hartree),
        ip_ev=ip_ev,
        ea_ev=ea_ev,
        mu_ev=mu_ev,
        chi_ev=chi_ev,
        hardness_ev=hardness_ev,
        softness=softness,
        omega_ev=omega_ev,
        omega_minus_ev=omega_minus_ev,
        omega_plus_ev=omega_plus_ev,
        homo_n_ev=homo_n_ev,
        homo_np1_ev=homo_np1_ev,
        homo_nm1_ev=homo_nm1_ev,
        lumo_n_ev=lumo_n_ev,
        nucleophilicity_ev=nucleophilicity_ev,
        nucleophilicity_hartree=nucleophilicity_hartree,
        homo_tce_ev=tce,
    )


def _validate_lengths(atoms, labels, *charge_sets):
    n_atoms = len(atoms)
    if len(labels) != n_atoms:
        raise ValueError("atoms and labels must have the same length.")
    for charges in charge_sets:
        if charges is not None and len(charges) != n_atoms:
            raise ValueError("All charge arrays must have one value per atom.")


def _warn_if_not_normalized(values: Optional[List[float]], name: str) -> None:
    if values is None:
        return
    # Condensed Fukui values sum to 1 WITH SIGN (charge conservation); individual atoms may be
    # negative (common with Mulliken), so the signed sum — not Σ|f| — is the normalization check.
    total = float(np.sum(values))
    if abs(total - 1.0) > 0.01:
        warnings.warn(
            f"{name} sum is {total:.6f}, expected near 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )


def _as_list(values: np.ndarray) -> List[float]:
    return [float(value) for value in values]


def compute_condensed_fukui(
    atoms: List[str],
    labels: List[str],
    charges_N: List[float],
    charges_Np1: Optional[List[float]] = None,
    charges_Nm1: Optional[List[float]] = None,
    global_descriptors: Optional[GlobalElectronic] = None,
    nucleophilicity_ev: Optional[float] = None,
    hardness_ev: Optional[float] = None,
    ip_ev: Optional[float] = None,
    ea_ev: Optional[float] = None,
    softness_override: Optional[float] = None,
    omega_ev_override: Optional[float] = None,
) -> FukuiResult:
    """Calculate condensed Fukui functions and local CDFT descriptors.

    Core Fukui functions (f+, f-, f0, CDD) are always computed when the
    relevant charge arrays are provided.

    Extended local descriptors require additional parameters:
    - ``global_descriptors``: enables s+, s-, ω·f+ (local electrophilicity)
    - ``nucleophilicity_ev``: enables s0, N·f- (local nucleophilicity), s+/s-, s-/s+
    - ``hardness_ev``: enables η·f0 (local hardness)
    - ``ip_ev``: enables IP·f- (donicity)
    - ``ea_ev``: enables EA·f+ (accepting capacity)
    """
    _validate_lengths(atoms, labels, charges_N, charges_Np1, charges_Nm1)
    q_n = np.asarray(charges_N, dtype=float)

    f_plus = None
    if charges_Np1 is not None:
        f_plus = _as_list(q_n - np.asarray(charges_Np1, dtype=float))
        _warn_if_not_normalized(f_plus, "f_plus")

    f_minus = None
    if charges_Nm1 is not None:
        f_minus = _as_list(np.asarray(charges_Nm1, dtype=float) - q_n)
        _warn_if_not_normalized(f_minus, "f_minus")

    f_zero = None
    dual = None
    if f_plus is not None and f_minus is not None:
        plus = np.asarray(f_plus, dtype=float)
        minus = np.asarray(f_minus, dtype=float)
        f_zero = _as_list((plus + minus) / 2.0)
        dual = _as_list(plus - minus)

    local_softness_plus = None
    local_softness_minus = None
    local_electrophilicity = None
    softness_zero = None
    local_nucleophilicity = None
    relative_electrophilicity = None
    relative_nucleophilicity = None
    f_zero_scaled_by_hardness = None
    donicity = None
    accepting_capacity = None

    # Descriptors that require global softness / omega
    # softness_override / omega_ev_override take priority over global_descriptors
    # (used to pass FD-based values instead of Koopmans approximations)
    _need_s_omega = global_descriptors is not None or softness_override is not None
    if _need_s_omega:
        S = (softness_override if softness_override is not None
             else getattr(global_descriptors, "softness", None))
        omega = (omega_ev_override if omega_ev_override is not None
                 else getattr(global_descriptors, "omega_ev", None))

        if S is not None and f_plus is not None:
            local_softness_plus = _as_list(S * np.asarray(f_plus, dtype=float))
        if omega is not None and f_plus is not None:
            local_electrophilicity = _as_list(omega * np.asarray(f_plus, dtype=float))

        if S is not None and f_minus is not None:
            minus = np.asarray(f_minus, dtype=float)
            local_softness_minus = _as_list(S * minus)

        if S is not None and f_zero is not None:
            zero = np.asarray(f_zero, dtype=float)
            softness_zero = _as_list(S * zero)

        # Relative electrophilicity/nucleophilicity
        if local_softness_plus is not None and local_softness_minus is not None:
            sp = np.asarray(local_softness_plus, dtype=float)
            sm = np.asarray(local_softness_minus, dtype=float)
            rel_e = np.where(sm != 0.0, sp / sm, np.nan)
            rel_n = np.where(sp != 0.0, sm / sp, np.nan)
            relative_electrophilicity = [float(v) for v in rel_e]
            relative_nucleophilicity = [float(v) for v in rel_n]
            nan_atoms = int(np.sum(np.isnan(rel_e) | np.isnan(rel_n)))
            if nan_atoms:
                warnings.warn(
                    f"Relative electrophilicity/nucleophilicity undefined for {nan_atoms} "
                    "atom(s) where s+ or s- is zero; stored as NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # Local nucleophilicity N * f-
    if nucleophilicity_ev is not None and f_minus is not None:
        minus = np.asarray(f_minus, dtype=float)
        local_nucleophilicity = _as_list(float(nucleophilicity_ev) * minus)

    # η·f0 — see the field's comment: it is f0 rescaled, not a local hardness
    if hardness_ev is not None and f_zero is not None:
        zero = np.asarray(f_zero, dtype=float)
        f_zero_scaled_by_hardness = _as_list(float(hardness_ev) * zero)

    # Donicity IP * f-
    if ip_ev is not None and f_minus is not None:
        minus = np.asarray(f_minus, dtype=float)
        donicity = _as_list(float(ip_ev) * minus)

    # Accepting capacity EA * f+
    if ea_ev is not None and f_plus is not None:
        plus = np.asarray(f_plus, dtype=float)
        accepting_capacity = _as_list(float(ea_ev) * plus)

    return FukuiResult(
        atoms=list(atoms),
        labels=list(labels),
        f_plus=f_plus,
        f_minus=f_minus,
        f_zero=f_zero,
        dual=dual,
        local_softness_plus=local_softness_plus,
        local_softness_minus=local_softness_minus,
        local_electrophilicity=local_electrophilicity,
        softness_zero=softness_zero,
        local_nucleophilicity=local_nucleophilicity,
        relative_electrophilicity=relative_electrophilicity,
        relative_nucleophilicity=relative_nucleophilicity,
        f_zero_scaled_by_hardness=f_zero_scaled_by_hardness,
        donicity=donicity,
        accepting_capacity=accepting_capacity,
    )
