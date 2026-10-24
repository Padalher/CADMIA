"""ORCA geometry parser."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from cadmia.core import Structure
from cadmia.io.xyz import get_atom_labels, normalize_element_symbol


def _parse_symbol_coordinate_line(line: str) -> Optional[Tuple[str, list]]:
    parts = line.split()
    for position in (0, 1):
        if len(parts) < position + 4:
            continue
        symbol = normalize_element_symbol(parts[position])
        if symbol is None:
            continue
        try:
            coords = [
                float(parts[position + 1]),
                float(parts[position + 2]),
                float(parts[position + 3]),
            ]
        except ValueError:
            continue
        return symbol, coords
    return None


def _extract_cartesian_blocks(lines: List[str]) -> List[Tuple[int, List[str], list]]:
    blocks = []
    for start_index, line in enumerate(lines):
        header = line.upper()
        if "CARTESIAN COORDINATES" not in header:
            continue
        if "ANGSTROEM" not in header and "ANGSTROM" not in header:
            continue

        cursor = start_index + 1
        atoms = []
        positions = []
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped:
                if atoms:
                    break
                cursor += 1
                continue
            if stripped.startswith("----"):
                cursor += 1
                continue
            parsed = _parse_symbol_coordinate_line(stripped)
            if parsed is None:
                if atoms:
                    break
                cursor += 1
                continue
            symbol, coords = parsed
            atoms.append(symbol)
            positions.append(coords)
            cursor += 1

        if atoms:
            blocks.append((start_index, atoms, positions))
    return blocks


def _parse_orca_charge_mult_pg(lines: List[str]):
    """Return (charge, multiplicity, point_group) from ORCA output lines."""
    charge = 0
    multiplicity = 1
    point_group: Optional[str] = None
    _charge_re = re.compile(r'Total Charge\s+Charge\s+\.*\s+(-?\d+)', re.IGNORECASE)
    _mult_re   = re.compile(r'Multiplicity\s+Mult\s+\.*\s+(\d+)', re.IGNORECASE)
    _pg_re     = re.compile(r'(?:Point Group|Symmetry Group used)\s+(\S+)', re.IGNORECASE)
    for line in lines:
        m = _charge_re.search(line)
        if m:
            charge = int(m.group(1))
        m = _mult_re.search(line)
        if m:
            multiplicity = int(m.group(1))
        m = _pg_re.search(line)
        if m:
            point_group = m.group(1)
    return charge, multiplicity, point_group


def parse_orca_geometry(filepath: str | Path) -> Structure:
    """Read Cartesian coordinates from an ORCA .out file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ORCA output file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = _extract_cartesian_blocks(lines)
    if not blocks:
        raise ValueError(f"No ORCA Cartesian coordinate block found in {path}")

    charge, multiplicity, point_group = _parse_orca_charge_mult_pg(lines)
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
