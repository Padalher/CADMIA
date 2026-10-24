"""XYZ parsers that return typed CADMIA structures."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from cadmia.core import Structure

from cadmia.core.elements import (  # re-exported for backward compatibility
    PERIODIC_TABLE, ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER, VALID_SYMBOLS,
    hill_formula,
)
SYMBOL_RE = re.compile(r"^[A-Za-z]{1,3}$")


def normalize_element_symbol(symbol: str) -> Optional[str]:
    """Normalize and validate an element symbol."""
    if SYMBOL_RE.match(symbol) is None:
        return None
    normalized = symbol[0].upper() + symbol[1:].lower()
    if normalized not in VALID_SYMBOLS:
        return None
    return normalized


def get_atom_labels(atoms: Iterable[str]) -> List[str]:
    """Generate labels using the global atom index: Ti1, O2, O3."""
    return [f"{atom}{index}" for index, atom in enumerate(atoms, start=1)]


def get_chemical_formula(atoms: list[str]) -> str:
    """Chemical formula in Hill notation (delegates to the canonical core helper)."""
    return hill_formula(atoms)


def _structure_id(path: Path, comment: str, index: Optional[int] = None) -> str:
    tokens = [token for token in re.split(r"\s+", comment.strip()) if token]
    candidate = tokens[-1] if tokens else path.stem
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_")
    if not candidate:
        candidate = path.stem
    if index is not None:
        return f"{candidate}_{index:04d}"
    return candidate


def _parse_xyz_block(
    lines: list[str],
    start: int,
    path: Path,
    block_index: Optional[int] = None,
) -> Tuple[Structure, int]:
    if start >= len(lines):
        raise ValueError("Missing XYZ atom count.")
    try:
        n_atoms = int(lines[start].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count: {lines[start].strip()}") from exc

    comment_index = start + 1
    if comment_index >= len(lines):
        raise ValueError("Missing XYZ comment line.")
    comment = lines[comment_index].strip()

    atom_start = start + 2
    atom_end = atom_start + n_atoms
    if len(lines) < atom_end:
        raise ValueError(
            f"Expected {n_atoms} atoms but found {max(0, len(lines) - atom_start)}"
        )

    atoms = []
    positions = []
    for line_number, line in enumerate(lines[atom_start:atom_end], start=atom_start + 1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"Line {line_number}: expected at least 4 columns, found {len(parts)}"
            )
        element = normalize_element_symbol(parts[0])
        if element is None:
            raise ValueError(f"Line {line_number}: invalid element symbol: {parts[0]}")
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid coordinates") from exc
        atoms.append(element)
        positions.append([x, y, z])

    source = str(path.resolve())
    structure = Structure(
        atoms=atoms,
        positions=np.array(positions, dtype=float),
        labels=get_atom_labels(atoms),
        structure_id=_structure_id(path, comment, block_index),
        source_file=source,
        metadata={"comment": comment},
    )
    return structure, atom_end


def parse_xyz(filepath: str | Path) -> Structure:
    """
    Read a standard XYZ file and return a Structure.

    Element symbols are validated against an internal periodic table.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"XYZ file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    structure, cursor = _parse_xyz_block(lines, 0, path)
    # Taking the first block of a trajectory would hand back its starting geometry.
    if any(line.strip() for line in lines[cursor:]):
        raise ValueError(
            f"{path}: lines follow the {len(structure.atoms)} atoms the header declares — a "
            "trajectory, several geometries or a wrong atom count. Pass a single geometry.")
    return structure


def parse_xyz_database(filepath: str | Path) -> list[Structure]:
    """Read concatenated XYZ blocks and return one Structure per block."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"XYZ database not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    structures = []
    cursor = 0

    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        structure, cursor = _parse_xyz_block(
            lines,
            cursor,
            path,
            block_index=len(structures) + 1,
        )
        structures.append(structure)

    if not structures:
        raise ValueError(f"No XYZ structures found in {path}")
    return structures
