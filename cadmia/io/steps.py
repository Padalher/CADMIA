"""Every step of a geometry optimization, read from an ORCA or Gaussian output.

A step is one geometry with the energy and the nuclear forces evaluated there. The forces
are what the engine prints as gradient (ORCA, ``CARTESIAN GRADIENT``, Eh/bohr, F = -dE/dR)
or as force (Gaussian, ``Forces (Hartrees/Bohr)``), converted once to eV/Angstrom.

ORCA: each ``GEOMETRY OPTIMIZATION CYCLE`` is one step, with the coordinates, energy and
gradient it prints. The final evaluation ORCA runs at the stationary point is left out: it
has no gradient and its energy comes from a different grid, so mixing it with the cycles'
energies would put two kinds of number in one column.

Gaussian: each ``Forces (Hartrees/Bohr)`` table is one step, paired with the last input-frame
orientation block (``Input`` or ``Z-Matrix``, never ``Standard``) and the last ``SCF Done``
energy printed before it. A log that chains several jobs
(opt, then freq) gives one step per table.

A single point is one step; it needs a gradient, otherwise there is nothing to write.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from cadmia.core.elements import ATOMIC_NUMBER_TO_SYMBOL
from cadmia.core.units import BOHR_TO_ANG, HARTREE_TO_EV
from cadmia.io.gaussian import _parse_charge_mult_pointgroup
from cadmia.io.orca import _extract_cartesian_blocks, _parse_orca_charge_mult_pg
from cadmia.io.xyz import extxyz_block

_EH_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / BOHR_TO_ANG

_ORCA_CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+\d+")
_ORCA_FINAL_EVALUATION = "FINAL ENERGY EVALUATION AT THE STATIONARY POINT"
_ORCA_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_NUMBER = r"(-?\d+\.\d+)"
_GAUSSIAN_NUMBER_RE = re.compile(r"^-?\d+\.\d+$")
_ORCA_GRADIENT_ROW_RE = re.compile(rf"^\s*\d+\s+[A-Za-z]{{1,3}}\s*:\s*{_NUMBER}\s+{_NUMBER}\s+{_NUMBER}\s*$")
_GAUSSIAN_ENERGY_RE = re.compile(r"SCF Done:\s+E\([^()\n]{0,64}\)\s+=\s+(-?\d+\.\d+)")
_GAUSSIAN_FORCES_HEADER = "Forces (Hartrees/Bohr)"
_GAUSSIAN_FORCES_ROW_RE = re.compile(rf"^\s*\d+\s+\d+\s+{_NUMBER}\s+{_NUMBER}\s+{_NUMBER}\s*$")
_GAUSSIAN_INPUT_FRAMES = ("Input orientation:", "Z-Matrix orientation:")
_GAUSSIAN_FORCES_HEADER_LINES = 2


@dataclass
class OptimizationStep:
    """One geometry of an optimization: positions in Å, energy in Eh, forces in Eh/bohr."""

    atoms: List[str]
    positions: np.ndarray
    energy_hartree: Optional[float]
    forces_eh_bohr: Optional[np.ndarray]


@dataclass
class OptimizationSteps:
    """The steps of one output, in the order the engine computed them."""

    engine: str
    charge: int
    multiplicity: int
    steps: List[OptimizationStep]


def _orca_gradient(lines: List[str], header: int, hi: int) -> List[List[float]]:
    rows = []
    for i in range(header + 1, hi):
        m = _ORCA_GRADIENT_ROW_RE.match(lines[i])
        if m:
            rows.append([float(v) for v in m.groups()])
        elif rows:
            break
    return rows


def _orca_segments(lines: List[str]) -> List[tuple]:
    """``(lo, hi)`` line ranges, one per optimization cycle, or the whole output if it has none.

    A cycle ends at the next cycle header or at the next final evaluation, whichever comes
    first, so an output with several jobs keeps every cycle of every job.
    """
    markers = sorted((i, bool(_ORCA_CYCLE_RE.search(ln)))
                     for i, ln in enumerate(lines)
                     if _ORCA_CYCLE_RE.search(ln) or _ORCA_FINAL_EVALUATION in ln)
    if not any(is_cycle for _, is_cycle in markers):
        return [(0, len(lines))]
    ends = [i for i, _ in markers[1:]] + [len(lines)]
    return [(lo, hi) for (lo, is_cycle), hi in zip(markers, ends) if is_cycle]


def _first_in(indexes: List[int], lo: int, hi: int) -> Optional[int]:
    k = bisect_left(indexes, lo)
    return k if k < len(indexes) and indexes[k] < hi else None


def _checked(values, where: str, what: str) -> np.ndarray:
    array = np.array(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{where}: {what} is not a finite number")
    return array


def parse_orca_steps(path: str | Path) -> OptimizationSteps:
    """Steps of an ORCA output: one per optimization cycle (a single point is one step).

    A gradient block whose rows do not match the atoms of its step is an error, not a
    shorter gradient: the missing rows would otherwise put the forces on the wrong atoms.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = _extract_cartesian_blocks(lines)
    block_at = [b[0] for b in blocks]
    energies = [(i, float(m.group(1))) for i, ln in enumerate(lines)
                if (m := _ORCA_ENERGY_RE.search(ln))]
    energy_at = [i for i, _ in energies]
    gradient_at = [i for i, ln in enumerate(lines) if ln.strip() == "CARTESIAN GRADIENT"]
    steps = []
    for lo, hi in _orca_segments(lines):
        k = _first_in(block_at, lo, hi)
        if k is None:
            continue
        where = f"ORCA step {len(steps) + 1}"
        _, atoms, raw = blocks[k]
        positions = _checked(raw, where, "a coordinate")
        e = _first_in(energy_at, lo, hi)
        g = _first_in(gradient_at, lo, hi)
        forces = None
        if g is not None:
            rows = _orca_gradient(lines, gradient_at[g], hi)
            if len(rows) != len(atoms):
                raise ValueError(f"{where}: the gradient has {len(rows)} rows for "
                                 f"{len(atoms)} atoms (truncated or malformed output)")
            forces = -_checked(rows, where, "a gradient component")
        energy = float(_checked(energies[e][1], where, "the energy")) if e is not None else None
        steps.append(OptimizationStep(atoms, positions, energy, forces))
    charge, multiplicity, _ = _parse_orca_charge_mult_pg(lines)
    return OptimizationSteps("orca", charge, multiplicity, steps)


def _input_frame_blocks(lines: List[str]) -> List[tuple]:
    """Coordinates the engine printed in the frame of the input (``Input`` or ``Z-Matrix``).

    The Gaussian forces table is written in that frame, not in the standard orientation.
    """
    blocks = []
    total = len(lines)
    i = 0
    while i < total:
        if not any(name in lines[i] for name in _GAUSSIAN_INPUT_FRAMES):
            i += 1
            continue
        start = i
        cursor = i + 1
        dashes = 0
        while cursor < total and dashes < 2 and not any(
                name in lines[cursor] for name in _GAUSSIAN_INPUT_FRAMES):
            dashes += lines[cursor].strip().startswith("----")
            cursor += 1
        if dashes < 2:
            i = cursor
            continue
        atoms, positions = [], []
        while cursor < total and not lines[cursor].strip().startswith("----"):
            parts = lines[cursor].split()
            cursor += 1
            if len(parts) < 6 or not (parts[0].isdigit() and parts[1].isdigit()):
                continue
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(parts[1]))
            if symbol is None:
                continue
            if not all(_GAUSSIAN_NUMBER_RE.match(v) for v in parts[-3:]):
                raise ValueError(f"Gaussian line {cursor}: a coordinate is not a finite number")
            atoms.append(symbol)
            positions.append([float(v) for v in parts[-3:]])
        if atoms:
            blocks.append((start, atoms, positions))
        i = cursor
    return blocks


def _gaussian_forces(lines: List[str], header_index: int) -> np.ndarray:
    rows = []
    for i in range(header_index + 1 + _GAUSSIAN_FORCES_HEADER_LINES, len(lines)):
        m = _GAUSSIAN_FORCES_ROW_RE.match(lines[i])
        if not m:
            break
        rows.append([float(v) for v in m.groups()])
    return np.array(rows)


def parse_gaussian_steps(path: str | Path) -> OptimizationSteps:
    """Steps of a Gaussian output: one per ``Forces (Hartrees/Bohr)`` table.

    Each table is paired with the last ``Input orientation`` or ``Z-Matrix orientation``
    block before it, the frame the forces are printed in, and with the last ``SCF Done``
    energy before it. A table with no such block is an error: the standard orientation is
    rotated with respect to the forces and would give the right numbers on the wrong axes.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = _input_frame_blocks(lines)
    block_at = [b[0] for b in blocks]
    energies = [(i, float(m.group(1))) for i, ln in enumerate(lines)
                if (m := _GAUSSIAN_ENERGY_RE.search(ln))]
    energy_at = [i for i, _ in energies]
    steps = []
    for i, line in enumerate(lines):
        if _GAUSSIAN_FORCES_HEADER not in line:
            continue
        where = f"Gaussian step {len(steps) + 1}"
        k = bisect_left(block_at, i) - 1
        if k < 0:
            raise ValueError(f"{where}: no Input or Z-Matrix orientation before the Forces "
                             "table (the forces are not in the standard orientation)")
        _, atoms, positions = blocks[k]
        forces = _gaussian_forces(lines, i)
        if len(forces) != len(atoms):
            raise ValueError(f"{where}: the forces table has {len(forces)} rows for "
                             f"{len(atoms)} atoms")
        e = bisect_left(energy_at, i) - 1
        energy = float(_checked(energies[e][1], where, "the energy")) if e >= 0 else None
        steps.append(OptimizationStep(atoms, _checked(positions, where, "a coordinate"),
                                      energy, _checked(forces, where, "a force component")))
    charge, multiplicity, _ = _parse_charge_mult_pointgroup(lines)
    return OptimizationSteps("gaussian", charge, multiplicity, steps)


def parse_steps(path: str | Path, engine: Optional[str] = None) -> OptimizationSteps:
    """Steps of a Gaussian/ORCA output; the engine is detected from the content."""
    from cadmia.agents.cdft_agent import _detect_engine

    p = Path(path)
    parse = parse_orca_steps if (engine or _detect_engine(p)) == "orca" else parse_gaussian_steps
    result = parse(p)
    if not any(s.forces_eh_bohr is not None for s in result.steps):
        raise ValueError(f"{p.name} carries no gradient to write: the engine printed no "
                         "forces (ORCA CARTESIAN GRADIENT, Gaussian Forces table)")
    return result


def steps_extxyz(result: OptimizationSteps, label: str) -> str:
    """Extended XYZ, one block per step: ``step``, ``forces`` in eV/Å (``nan`` without gradient)."""
    blocks = []
    for number, step in enumerate(result.steps, start=1):
        if step.forces_eh_bohr is None:
            forces = np.full((len(step.atoms), 3), np.nan)
        else:
            forces = step.forces_eh_bohr * _EH_PER_BOHR_TO_EV_PER_ANG
        info = {"label": label,
                "energy": None if step.energy_hartree is None else f"{step.energy_hartree:.8f}",
                "energy_unit": "Eh", "step": number, "forces_unit": "eV/Ang",
                "charge": result.charge, "mult": result.multiplicity,
                "engine": result.engine}
        blocks.append(extxyz_block(step.atoms, step.positions, info,
                                   {"forces": ("R", [list(row) for row in forces])}))
    return "".join(blocks)
