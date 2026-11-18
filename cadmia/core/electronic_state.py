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
    spin_sq: Optional[float] = None
    alpha_eigenvalues_ev: Optional[List[float]] = None
    beta_eigenvalues_ev: Optional[List[float]] = None
    homo_alpha_ev: Optional[float] = None
    lumo_alpha_ev: Optional[float] = None
    homo_beta_ev: Optional[float] = None
    lumo_beta_ev: Optional[float] = None
    zpe_correction_hartree: Optional[float] = None
    thermal_energy_correction_hartree: Optional[float] = None
    thermal_enthalpy_correction_hartree: Optional[float] = None
    thermal_gibbs_correction_hartree: Optional[float] = None
    electronic_zpe_energy_hartree: Optional[float] = None
    enthalpy_hartree: Optional[float] = None
    gibbs_free_energy_hartree: Optional[float] = None
    entropy_term_hartree: Optional[float] = None
    temperature_kelvin: Optional[float] = None
    frequencies_cm1: Optional[List[float]] = None
    entropy_scheme: Optional[str] = None
    rotational_symmetry_number: Optional[int] = None
    geometry_symmetry: Optional[object] = None
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
