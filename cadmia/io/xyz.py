"""XYZ parsers that return typed CADMIA structures."""
from __future__ import annotations

import re
import shlex
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


EXTXYZ_FORMAT = "extxyz-cadmia-1"
_BASE_PROPERTIES = "species:S:1:pos:R:3"


def comment_info(comment: str) -> dict:
    """The ``key=value`` pairs of a comment line (extended XYZ, or CADMIA's older lines)."""
    try:
        tokens = shlex.split(comment)
    except ValueError:                      # an unbalanced quote: read it word by word
        tokens = comment.split()
    return dict(t.split("=", 1) for t in tokens if "=" in t and not t.startswith("="))


def extxyz_columns(info: dict) -> List[Tuple[str, str]]:
    """The per-atom columns a ``Properties=`` value declares after species and position."""
    fields = info.get("Properties", "").split(":")
    cols = [(fields[i], fields[i + 1]) for i in range(0, len(fields) - 2, 3)]
    return [c for c in cols if c[0] not in ("species", "pos")]


def extxyz_comment(info: dict, columns: Iterable[Tuple[str, str]] = ()) -> str:
    """Comment line of an extended XYZ block, format ``extxyz-cadmia-1``.

    ``energy`` goes first after the format, in Hartree, because readers take the first number
    called energy. Values with spaces are quoted; ``None`` values are left out. *columns* are
    the extra per-atom columns as ``(name, type)``, type ``S``, ``R`` or ``I``.
    """
    from cadmia import __version__
    info = {**info, "program": f"CADMIA {__version__}"}
    keys = ["energy"] + [k for k in info if k != "energy"]
    parts = [f"format={EXTXYZ_FORMAT}"]
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        text = ("T" if value else "F") if isinstance(value, bool) else str(value)
        if key == "label":                  # viewers split the line on spaces and skip quotes
            text = re.sub(r"[\s\"'=]+", "_", text)
        if not text or any(c.isspace() or c in "\"'=" for c in text):
            text = '"' + text.replace('"', "'") + '"'
        parts.append(f"{key}={text}")
    props = _BASE_PROPERTIES + "".join(f":{name}:{kind}:1" for name, kind in columns)
    return " ".join(parts + [f"Properties={props}"])


def extxyz_block(atoms: Iterable[str], positions, info: dict,
                 columns: Optional[dict] = None) -> str:
    """One extended XYZ block. *columns* maps a column name to ``(type, values)``; a missing
    real value is written ``nan``, never 0."""
    atoms = list(atoms)
    columns = columns or {}

    def cell(kind, v):
        if kind == "R":
            return "nan" if v is None or v != v else f"{float(v):.6f}"
        return str(v)

    lines = [str(len(atoms)), extxyz_comment(info, [(n, k) for n, (k, _) in columns.items()])]
    for i, (sym, pos) in enumerate(zip(atoms, positions)):
        extra = "".join(f" {cell(k, vals[i])}" for k, vals in columns.values())
        x, y, z = (float(c) for c in pos)
        lines.append(f"{sym:<3} {x:14.8f} {y:14.8f} {z:14.8f}{extra}")
    return "\n".join(lines) + "\n"


def _structure_id(path: Path, comment: str, index: Optional[int] = None) -> str:
    tokens = [token for token in re.split(r"\s+", _SPIN_TOKEN_RE.sub("", comment).strip()) if token]
    candidate = comment_info(comment).get("label") or (tokens[-1] if tokens else path.stem)
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
    spin = _spin_tokens(comment)
    structure = Structure(
        atoms=atoms,
        positions=np.array(positions, dtype=float),
        labels=get_atom_labels(atoms),
        structure_id=_structure_id(path, comment, block_index),
        source_file=source,
        metadata={"comment": comment, "info": comment_info(comment),
                  **({"spin_given": True} if "multiplicity" in spin else {})},
        **spin,
    )
    return structure, atom_end


_SPIN_TOKEN_RE = re.compile(r"(?<!\S)(charge|mult)=(-?\d+)(?!\S)")


def _spin_tokens(comment: str) -> dict:
    """``charge=`` and ``mult=`` written in the comment line (by `cadmia geometry`, or by hand).

    An XYZ file has no other place for them, and an adduct with O2 is a triplet that the
    electron-count parity would turn into a singlet.
    """
    found = dict(_SPIN_TOKEN_RE.findall(comment))
    spin = {}
    if "charge" in found:
        spin["charge"] = int(found["charge"])
    if "mult" in found and int(found["mult"]) >= 1:
        spin["multiplicity"] = int(found["mult"])
    return spin


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
