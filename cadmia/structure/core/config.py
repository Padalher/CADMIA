"""Typed configuration loading for structural analysis."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Elements already warned about (missing radius) — avoids repeating the notice per atom pair.
_WARNED_MISSING_RADII: set[str] = set()

# Covalent radii (Angstrom) from Alvarez 2008, DOI:10.1039/b801115j.
# Used as fallback when an element is absent from the YAML config.
_FALLBACK_COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.61,
    "Fe": 1.52, "Co": 1.50, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50,
    "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80,
    "Cm": 1.69, "Bk": 1.68, "Cf": 1.68, "Es": 1.65, "Fm": 1.67,
    "Md": 1.73, "No": 1.76, "Lr": 1.61,
}

# Standard atomic weights (g/mol) from IUPAC 2021.
# Used as fallback when an element is absent from the YAML config.
_FALLBACK_ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905,
    "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,
    "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.045,
    "Lu": 174.967, "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Tl": 204.383, "Pb": 207.2, "Bi": 208.980, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.038,
    "Pa": 231.036, "U": 238.029, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
    "Md": 258.0, "No": 259.0, "Lr": 262.0,
}


@dataclass
class Config:
    """Atomic data and analysis parameters used across structural analysis."""

    atomic_masses: Dict[str, float] = field(default_factory=dict)
    covalent_radii: Dict[str, float] = field(default_factory=dict)
    metallic_radii: Dict[str, float] = field(default_factory=dict)
    bond_cutoffs: Dict[str, float] = field(default_factory=dict)
    excluded_bonds: List[str] = field(default_factory=list)
    metal_elements: List[str] = field(default_factory=list)
    ecn_parameters: Dict[str, Any] = field(default_factory=dict)
    metallic_bond_tolerance: float = 1.25
    default_bond_tolerance: float = 1.20
    source_file: str = ""

    def get_atomic_mass(self, element: str) -> float:
        mass = self.atomic_masses.get(element)
        if mass is not None:
            return float(mass)
        return float(_FALLBACK_ATOMIC_MASSES.get(element, 0.0))

    def get_bond_cutoff(self, element1: str, element2: str) -> float:
        bond1 = f"{element1}-{element2}"
        bond2 = f"{element2}-{element1}"
        explicit = self.bond_cutoffs.get(bond1, self.bond_cutoffs.get(bond2))
        if explicit is not None:
            return float(explicit)

        metal1 = self.metallic_radii.get(element1)
        metal2 = self.metallic_radii.get(element2)
        if metal1 is not None and metal2 is not None:
            return (float(metal1) + float(metal2)) * self.metallic_bond_tolerance

        radius1 = self.covalent_radii.get(element1, _FALLBACK_COVALENT_RADII.get(element1))
        radius2 = self.covalent_radii.get(element2, _FALLBACK_COVALENT_RADII.get(element2))
        if radius1 is None or radius2 is None:
            for el, r in ((element1, radius1), (element2, radius2)):
                if r is None and el not in _WARNED_MISSING_RADII:
                    _WARNED_MISSING_RADII.add(el)
                    warnings.warn(
                        f"No covalent/metallic radius for element '{el}'; its bonds will not "
                        f"be detected. Add it to atomic_data.yaml (covalent_radii).",
                        stacklevel=2,
                    )
            return 0.0
        return (float(radius1) + float(radius2)) * self.default_bond_tolerance

    def is_excluded_bond(self, element1: str, element2: str) -> bool:
        bond1 = f"{element1}-{element2}"
        bond2 = f"{element2}-{element1}"
        return bond1 in self.excluded_bonds or bond2 in self.excluded_bonds

    def is_metal(self, element: str) -> bool:
        return element in self.metal_elements


def default_config_path() -> Path:
    """Return the bundled atomic-data config path (shipped inside the package).

    The YAML lives next to this module so it is found both in an editable checkout and
    in an installed wheel/sdist (declared as package data in pyproject.toml).
    """
    return Path(__file__).resolve().parent / "atomic_data.yaml"


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """Load atomic data from YAML into a typed Config object."""
    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(
        atomic_masses=data.get("atomic_masses", {}),
        covalent_radii=data.get("covalent_radii", {}),
        metallic_radii=data.get("metallic_radii", {}),
        bond_cutoffs=data.get("bond_cutoffs", {}),
        excluded_bonds=data.get("excluded_bonds", []),
        metal_elements=data.get("metal_elements", []),
        ecn_parameters=data.get(
            "ecn_parameters",
            {
                "max_iterations": 100,
                "convergence_threshold": 1e-6,
                "damping_factor": 6.0,
            },
        ),
        metallic_bond_tolerance=float(data.get("metallic_bond_tolerance", 1.25)),
        default_bond_tolerance=float(data.get("default_bond_tolerance", 1.20)),
        source_file=str(path.resolve()),
    )
