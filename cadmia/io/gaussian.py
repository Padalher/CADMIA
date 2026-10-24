"""Gaussian geometry parser."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from cadmia.core import Structure
from cadmia.core.elements import ATOMIC_NUMBER_TO_SYMBOL
from cadmia.io.xyz import get_atom_labels


def _extract_orientation_blocks(lines: List[str]) -> List[Tuple[int, List[str], list]]:
    blocks = []
    for start_index, line in enumerate(lines):
        if "Standard orientation:" not in line and "Input orientation:" not in line:
            continue

        cursor = start_index + 1
        dash_count = 0
        while cursor < len(lines):
            if lines[cursor].strip().startswith("----"):
                dash_count += 1
                if dash_count == 2:
                    cursor += 1
                    break
            cursor += 1
        if dash_count < 2:
            continue

        atoms = []
        positions = []
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped or stripped.startswith("----"):
                break
            parts = stripped.split()
            if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(parts[1]))
                if symbol is not None:
                    try:
                        positions.append([float(value) for value in parts[-3:]])
                    except ValueError:
                        cursor += 1
                        continue
                    atoms.append(symbol)
            cursor += 1

        if atoms:
            blocks.append((start_index, atoms, positions))
    return blocks


def _parse_charge_mult_pointgroup(lines: List[str]):
    """Return (charge, multiplicity, point_group) from Gaussian output lines."""
    charge = 0
    multiplicity = 1
    point_group: Optional[str] = None
    _cm_re = re.compile(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)')
    _pg_re = re.compile(r'Full point group\s+(\S+)')
    for line in lines:
        m = _cm_re.search(line)
        if m:
            charge = int(m.group(1))
            multiplicity = int(m.group(2))
        m2 = _pg_re.search(line)
        if m2:
            point_group = m2.group(1)
    return charge, multiplicity, point_group


def parse_gaussian_geometry(filepath: str | Path) -> Structure:
    """Read the final supported Gaussian geometry from a .log/.out file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Gaussian output file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = _extract_orientation_blocks(lines)
    if not blocks:
        raise ValueError(f"No Gaussian orientation block found in {path}")

    charge, multiplicity, point_group = _parse_charge_mult_pointgroup(lines)
    _, atoms, positions = max(blocks, key=lambda block: block[0])
    return Structure(
        atoms=atoms,
        positions=np.array(positions, dtype=float),
        labels=get_atom_labels(atoms),
        structure_id=path.stem,
        source_file=str(path.resolve()),
        charge=charge,
        multiplicity=multiplicity,
        point_group=point_group,
    )
