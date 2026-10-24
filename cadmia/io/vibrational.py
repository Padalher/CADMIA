"""Parse ORCA vibrational analysis: frequencies, IR intensities and normal modes.

Populated only by a frequency (``! Freq``) run. Frequencies come from the
``VIBRATIONAL FREQUENCIES`` block, IR intensities (km/mol) from the ``IR SPECTRUM``
block, and the Cartesian displacement vectors from the ``NORMAL MODES`` block.
The first six modes of a non-linear molecule are translations/rotations (≈ 0 cm⁻¹);
imaginary modes are reported as negative wavenumbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

_FREQ_LINE = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1")
_IR_DATA = re.compile(r"^\s*(\d+):\s+(.*)")
_SCALING = re.compile(r"Scaling factor for frequencies\s*=\s*([\d.]+)")
_FLOAT = re.compile(r"-?\d+\.\d+")


# Modes with |ν| below this are translations/rotations or Hessian numerical noise, not
# genuine vibrations: ORCA/Gaussian project the six trans-rot modes to ~0. The SAME tolerance
# bounds both the imaginary test and the real-mode filter, so the [−tol, tol] dead zone is
# symmetric — a −5 cm⁻¹ artifact is not miscounted as an imaginary (saddle-point) mode while a
# +5 cm⁻¹ one is silently dropped.
_TRANSROT_TOL_CM1 = 10.0


@dataclass
class VibrationalMode:
    """One normal mode: wavenumber, IR intensity and per-atom displacement."""

    index: int
    frequency_cm1: float
    ir_intensity_km_mol: Optional[float] = None
    displacements: Optional[np.ndarray] = None  # shape (n_atoms, 3)

    @property
    def is_imaginary(self) -> bool:
        return self.frequency_cm1 < -_TRANSROT_TOL_CM1


@dataclass
class VibrationalAnalysis:
    """Full vibrational analysis of one structure."""

    modes: List[VibrationalMode] = field(default_factory=list)
    n_atoms: int = 0
    scaling_factor: float = 1.0
    source_file: str = ""

    @property
    def imaginary_modes(self) -> List[VibrationalMode]:
        return [m for m in self.modes if m.is_imaginary]

    @property
    def real_modes(self) -> List[VibrationalMode]:
        """Vibrational modes above the translation/rotation threshold (10 cm⁻¹)."""
        return [m for m in self.modes if m.frequency_cm1 > _TRANSROT_TOL_CM1]

    @property
    def ir_active_modes(self) -> List[VibrationalMode]:
        return [m for m in self.real_modes if m.ir_intensity_km_mol]


def _last_index(lines: List[str], matches: Callable[[str], bool]) -> Optional[int]:
    """Index of the LAST line satisfying *matches*, or None.

    A re-run or a concatenated output carries several vibrational sections. The converged
    one is the last, and the three parsers below must agree on which section they read:
    frequencies from one block and displacements from another would describe different
    geometries — the failure this whole module has to avoid.
    """
    found = None
    for i, line in enumerate(lines):
        if matches(line):
            found = i
    return found


def _parse_frequencies(lines: List[str]) -> tuple[dict, float]:
    """Return {mode_index: frequency_cm1} and the scaling factor of the last block."""
    start = _last_index(lines, lambda l: "VIBRATIONAL FREQUENCIES" in l)
    if start is None:
        return {}, 1.0
    freqs: dict = {}
    scaling = 1.0
    for line in lines[start + 1:]:
        ms = _SCALING.search(line)
        if ms:
            scaling = float(ms.group(1))
        m = _FREQ_LINE.match(line)
        if m:
            freqs[int(m.group(1))] = float(m.group(2))
        elif "NORMAL MODES" in line:
            break
    return freqs, scaling


_IR_INT_COL = 2  # IR SPECTRUM row values: freq, eps, Int (km/mol), T**2, ...


def _parse_ir_intensities(lines: List[str]) -> dict:
    """Return {mode_index: ir_intensity_km_mol} from the IR SPECTRUM block.

    Scanning forward from the header only meets IR data rows (``idx: freq ... Int ...``)
    followed by unrelated text; ``_IR_DATA`` ignores headers/rulers/blank lines.
    """
    intensities: dict = {}
    start = _last_index(lines, lambda l: l.strip() == "IR SPECTRUM")
    if start is None:
        return intensities
    for line in lines[start + 1:]:
        m = _IR_DATA.match(line)
        if m:
            values = _FLOAT.findall(m.group(2))
            if len(values) > _IR_INT_COL:
                intensities[int(m.group(1))] = float(values[_IR_INT_COL])
        elif intensities:
            break
    return intensities


def _parse_normal_modes(lines: List[str], n_coords: int) -> dict:
    """Return {mode_index: np.ndarray(n_coords)} from the last NORMAL MODES block."""
    modes: dict = {}
    start = _last_index(lines, lambda l: "NORMAL MODES" in l)
    if start is None or n_coords <= 0:
        return modes
    col_indices: List[int] = []
    for line in lines[start + 1:]:
        stripped = line.split()
        if not stripped:
            continue
        # A header row is all integers (the mode-column labels for this block).
        if all(tok.lstrip("-").isdigit() for tok in stripped) and "." not in line:
            col_indices = [int(tok) for tok in stripped]
            for c in col_indices:
                modes.setdefault(c, [None] * n_coords)
            continue
        # A data row: leading coordinate index, then one float per column.
        if col_indices and stripped[0].isdigit() and "." in line:
            coord = int(stripped[0])
            vals = _FLOAT.findall(line)
            if 0 <= coord < n_coords:
                for c, v in zip(col_indices, vals):
                    if c in modes:
                        modes[c][coord] = float(v)
        elif modes and "IR SPECTRUM" in line:
            break
    return {k: np.array(v, dtype=float) for k, v in modes.items()
            if all(x is not None for x in v)}


def parse_orca_vibrational(filepath: str | Path) -> VibrationalAnalysis:
    """Parse frequencies, IR intensities and normal modes from an ORCA output."""
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    freqs, scaling = _parse_frequencies(lines)
    if not freqs:
        return VibrationalAnalysis(source_file=str(path))

    n_coords = max(freqs) + 1            # ORCA prints 3N modes (incl. trans/rot)
    n_atoms = n_coords // 3
    intensities = _parse_ir_intensities(lines)
    displacements = _parse_normal_modes(lines, n_coords)

    modes: List[VibrationalMode] = []
    for idx in sorted(freqs):
        disp = displacements.get(idx)
        modes.append(VibrationalMode(
            index=idx,
            frequency_cm1=freqs[idx],
            ir_intensity_km_mol=intensities.get(idx),
            displacements=disp.reshape(n_atoms, 3) if disp is not None else None,
        ))
    return VibrationalAnalysis(
        modes=modes, n_atoms=n_atoms, scaling_factor=scaling, source_file=str(path),
    )
