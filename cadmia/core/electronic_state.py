"""ElectronicState — electronic properties associated with a Structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from cadmia.core.provenance import Provenance
from cadmia.core.structure import Structure


@dataclass
class ElectronicState:
    """Electronic properties associated with a Structure."""

    structure: Structure
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    energy_hartree: Optional[float] = None
    charges: Optional[List[float]] = None
    charge_scheme: str = "mulliken"
    dipole_debye: Optional[np.ndarray] = None
    source_file: str = ""
    # Orbital eigenvalues (eV) — alpha and beta spin channels
    spin_sq: Optional[float] = None       # ⟨S²⟩ of an open-shell SCF; None if closed shell
    alpha_eigenvalues_ev: Optional[List[float]] = None
    beta_eigenvalues_ev: Optional[List[float]] = None
    # Frontier orbitals per spin channel (eV). homo_ev/lumo_ev combine them as max/min, which
    # for an open shell gives a cross-spin gap; these keep the two channels apart.
    homo_alpha_ev: Optional[float] = None
    lumo_alpha_ev: Optional[float] = None
    homo_beta_ev: Optional[float] = None
    lumo_beta_ev: Optional[float] = None
    # Thermochemistry — only populated when a frequency calculation is detected
    zpe_correction_hartree: Optional[float] = None
    thermal_energy_correction_hartree: Optional[float] = None
    thermal_enthalpy_correction_hartree: Optional[float] = None
    thermal_gibbs_correction_hartree: Optional[float] = None
    electronic_zpe_energy_hartree: Optional[float] = None
    enthalpy_hartree: Optional[float] = None
    gibbs_free_energy_hartree: Optional[float] = None
    entropy_term_hartree: Optional[float] = None   # T·S (ORCA "Final entropy term")
    temperature_kelvin: Optional[float] = None
    frequencies_cm1: Optional[List[float]] = None
    # Vibrational entropy the engine used for G: "harmonic" (Gaussian), "grimme" (ORCA's
    # quasi-RRHO, read from its output), None when there is no G or the output does not say.
    entropy_scheme: Optional[str] = None
    # Rotational symmetry number the engine used in its G ("Rotational symmetry number" in
    # Gaussian, "Symmetry Number" in ORCA), and the symmetry CADMIA detects from the geometry
    # (a cadmia.structure.structural.symmetry.Symmetry, set by the agents that need it).
    rotational_symmetry_number: Optional[int] = None
    geometry_symmetry: Optional[object] = None
    # Set only by readers that need to mark a lower verification level: None means "native parser, no extra marking needed".
    provenance: Optional[Provenance] = None

    @property
    def homo_lumo_gap_ev(self) -> Optional[float]:
        if self.homo_ev is not None and self.lumo_ev is not None:
            return self.lumo_ev - self.homo_ev
        return None

    @property
    def gap_alpha_ev(self) -> Optional[float]:
        if self.homo_alpha_ev is not None and self.lumo_alpha_ev is not None:
            return self.lumo_alpha_ev - self.homo_alpha_ev
        return None

    @property
    def gap_beta_ev(self) -> Optional[float]:
        if self.homo_beta_ev is not None and self.lumo_beta_ev is not None:
            return self.lumo_beta_ev - self.homo_beta_ev
        return None

    def to_dict(self) -> Dict:
        return {
            "structure_id": self.structure.structure_id,
            "homo_ev": self.homo_ev,
            "lumo_ev": self.lumo_ev,
            "gap_ev": self.homo_lumo_gap_ev,
            "energy_hartree": self.energy_hartree,
            "charges": self.charges,
            "charge_scheme": self.charge_scheme,
            "source_file": str(self.source_file),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
