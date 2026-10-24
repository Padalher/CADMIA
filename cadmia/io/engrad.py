"""ORCA .engrad — energy and nuclear gradient at a fixed geometry.

The gradient is what the nuclear Fukui function is built from, and ORCA writes it in a
plain fixed-layout file rather than inside the .out: four comment-delimited blocks with the
atom count, the total energy, the 3N gradient components in Eh/bohr, and the atomic numbers
with the coordinates in bohr.

The file stores the GRADIENT ∂E/∂R, not the force. Everything downstream wants the force
F = -∂E/∂R, so `force()` does that conversion once, in one place, instead of leaving every
caller to remember the sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from cadmia.core.units import BOHR_TO_ANG

# Alias kept for the callers that already import this name. The VALUE now comes from core.units:
# two constants that disagree in the 10th decimal are harmless numerically and a trap to maintain.
BOHR_TO_ANGSTROM = BOHR_TO_ANG


@dataclass
class EngradResult:
    """One ORCA .engrad file: energy, gradient and the geometry they were evaluated at."""

    n_atoms:        int
    energy_hartree: float
    gradient:       np.ndarray      # (N, 3) in Eh/bohr — ∂E/∂R, NOT the force
    atomic_numbers: List[int]
    positions_bohr: np.ndarray      # (N, 3)
    source_file:    str

    def force(self) -> np.ndarray:
        """Force on each nucleus, F = -∂E/∂R, in Eh/bohr."""
        return -self.gradient

    @property
    def positions_angstrom(self) -> np.ndarray:
        return self.positions_bohr * BOHR_TO_ANGSTROM


def _numeric_lines(lines: List[str]) -> List[str]:
    """Drop the '#' comment banners ORCA uses to delimit the blocks."""
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def parse_engrad(filepath: str | Path) -> EngradResult:
    """Read an ORCA .engrad file.

    Raises ValueError on a file whose declared atom count does not match the numbers it
    carries: a truncated gradient silently read as a shorter one would put the remaining
    components on the wrong atoms, and every derived quantity would be wrong without a word.
    """
    path = Path(filepath)
    try:
        rows = _numeric_lines(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError as exc:
        raise ValueError(f"could not read the .engrad file {path}: {exc}") from exc

    if len(rows) < 2:
        raise ValueError(f"{path}: not an ORCA .engrad file (no numeric blocks found)")

    try:
        n_atoms = int(rows[0])
        energy = float(rows[1])
    except ValueError as exc:
        raise ValueError(f"{path}: could not read the atom count and energy: {exc}") from exc
    if n_atoms <= 0:
        raise ValueError(f"{path}: atom count is {n_atoms}")

    expected = 2 + 3 * n_atoms + n_atoms
    if len(rows) != expected:
        raise ValueError(
            f"{path}: declares {n_atoms} atoms, which needs {expected} numeric lines "
            f"(count + energy + 3N gradient + N geometry) but the file has {len(rows)}"
        )

    try:
        flat = [float(v) for v in rows[2:2 + 3 * n_atoms]]
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric value in the gradient block: {exc}") from exc
    gradient = np.asarray(flat, dtype=float).reshape(n_atoms, 3)

    numbers: List[int] = []
    coords: List[List[float]] = []
    for ln in rows[2 + 3 * n_atoms:]:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: malformed geometry line {ln!r}")
        try:
            numbers.append(int(float(parts[0])))
            coords.append([float(x) for x in parts[1:4]])
        except ValueError as exc:
            raise ValueError(f"{path}: malformed geometry line {ln!r}: {exc}") from exc

    return EngradResult(
        n_atoms=n_atoms,
        energy_hartree=energy,
        gradient=gradient,
        atomic_numbers=numbers,
        positions_bohr=np.asarray(coords, dtype=float),
        source_file=str(path),
    )
