"""Parser for Gaussian WFN (wavefunction) files."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from cadmia.core.units import BOHR_TO_ANG

_FORTRAN_D = re.compile(r'[Dd]')


@dataclass
class WFNData:
    """Data extracted from a Gaussian .wfn file."""

    title: str
    n_mo: int
    n_primitives: int
    n_nuclei: int
    atoms: List[str]
    coords_ang: List[Tuple[float, float, float]]
    nuclear_charges: List[float]
    mo_occupations: List[float]
    mo_energies: List[float]    # Hartree
    total_energy: Optional[float]

    # Occupation cutoff separating occupied from virtual MOs (WFN occupations are ~0/1/2,
    # so 0.01 cleanly excludes numerically-zero virtual occupations).
    _OCC_CUTOFF = 0.01

    @property
    def homo_hartree(self) -> Optional[float]:
        """Highest occupied MO energy in Hartree."""
        occupied = [e for occ, e in zip(self.mo_occupations, self.mo_energies) if occ > self._OCC_CUTOFF]
        return max(occupied) if occupied else None

    @property
    def lumo_hartree(self) -> Optional[float]:
        """Lowest unoccupied MO energy (None when no virtual MOs stored)."""
        virtual = [e for occ, e in zip(self.mo_occupations, self.mo_energies) if occ <= self._OCC_CUTOFF]
        return min(virtual) if virtual else None


def parse_wfn(filepath: str | Path) -> WFNData:
    """Parse a Gaussian .wfn file and return a WFNData object.

    Extracts geometry, MO occupations/energies, and total energy.
    MO coefficients are skipped (not needed for CDFT global descriptors).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"WFN file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty WFN file: {path}")

    title = lines[0].strip()

    # --- header ---
    header_m = re.search(
        r'(\d+)\s+MOL ORBITALS\s+(\d+)\s+PRIMITIVES\s+(\d+)\s+NUCLEI',
        lines[1],
    )
    if not header_m:
        raise ValueError(f"Cannot parse WFN header (line 2): {lines[1]!r}")
    n_mo   = int(header_m.group(1))
    n_prim = int(header_m.group(2))
    n_nuclei = int(header_m.group(3))

    # --- atoms ---
    atom_re = re.compile(
        r'^\s*([A-Za-z]+)\s+\d+\s+\(CENTRE\s+\d+\)\s+'
        r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+CHARGE\s*=\s*([-\d.]+)'
    )
    atoms: List[str] = []
    coords_ang: List[Tuple[float, float, float]] = []
    nuclear_charges: List[float] = []

    for i in range(n_nuclei):
        m = atom_re.match(lines[2 + i])
        if not m:
            raise ValueError(f"Cannot parse atom line {3+i}: {lines[2+i]!r}")
        atoms.append(m.group(1).capitalize())
        coords_ang.append((
            float(m.group(2)) * BOHR_TO_ANG,
            float(m.group(3)) * BOHR_TO_ANG,
            float(m.group(4)) * BOHR_TO_ANG,
        ))
        nuclear_charges.append(float(m.group(5)))

    # --- MO blocks ---
    # Skip CENTRE ASSIGNMENTS / TYPE ASSIGNMENTS / EXPONENTS until first 'MO ' line
    mo_re = re.compile(
        r'^MO\s+\d+\s+.*?OCC NO\s*=\s*([-\d.]+)\s+ORB\.\s*ENERGY\s*=\s*([-\d.Ee+Dd-]+)'
    )
    end_re = re.compile(r'TOTAL ENERGY\s*=\s*([-\d.Ee+Dd-]+)')

    mo_occupations: List[float] = []
    mo_energies: List[float] = []
    total_energy: Optional[float] = None

    # Each MO header is followed by exactly ceil(n_prim / 5) coefficient lines
    n_coef_lines = math.ceil(n_prim / 5)

    idx = 2 + n_nuclei
    while idx < len(lines):
        line = lines[idx]
        mo_m = mo_re.match(line)
        if mo_m:
            mo_occupations.append(float(mo_m.group(1)))
            # MO energy may use Fortran D notation
            mo_energies.append(float(_FORTRAN_D.sub('E', mo_m.group(2))))
            # Skip coefficient lines
            idx += n_coef_lines + 1
            continue

        end_m = end_re.search(line)
        if end_m:
            total_energy = float(_FORTRAN_D.sub('E', end_m.group(1)))

        idx += 1

    return WFNData(
        title=title,
        n_mo=n_mo,
        n_primitives=n_prim,
        n_nuclei=n_nuclei,
        atoms=atoms,
        coords_ang=coords_ang,
        nuclear_charges=nuclear_charges,
        mo_occupations=mo_occupations,
        mo_energies=mo_energies,
        total_energy=total_energy,
    )
