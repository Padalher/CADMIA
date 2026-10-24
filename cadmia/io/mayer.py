"""Parser for ORCA's Mayer population analysis (`Print[P_Mayer] 1`).

Extracts the per-atom Mayer valences (NA/ZA/QA/VA/BVA/FA) and the bond-order list from the
``* MAYER POPULATION ANALYSIS *`` block. Built against real ORCA 6 output; the block format
is identical for closed- and open-shell runs. Gaussian does not print this block, so
``parse_mayer`` returns None when the block is absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_HEADER = "MAYER POPULATION ANALYSIS"
_ATOM_TABLE_HEADER = "ATOM"
_BOND_HEADER = "Mayer bond orders larger than"

# B(  0-Ti,  1-O ) :   2.0961   (several per line)
_BOND_RE = re.compile(
    r"B\(\s*(\d+)\s*-\s*([A-Za-z]{1,3})\s*,\s*(\d+)\s*-\s*([A-Za-z]{1,3})\s*\)\s*:\s*(-?\d+\.\d+)"
)


@dataclass
class MayerAtom:
    index:  int
    symbol: str
    na: float   # Mulliken gross atomic population
    za: float   # total nuclear charge
    qa: float   # Mulliken gross atomic charge
    va: float   # Mayer total valence
    bva: float  # Mayer bonded valence
    fa: float   # Mayer free valence


@dataclass
class MayerBond:
    atom_i:   int
    symbol_i: str
    atom_j:   int
    symbol_j: str
    order:    float


@dataclass
class MayerResult:
    atoms:     List[MayerAtom] = field(default_factory=list)
    bonds:     List[MayerBond] = field(default_factory=list)
    threshold: float = 0.0
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "atoms": [a.__dict__ for a in self.atoms],
            "bonds": [b.__dict__ for b in self.bonds],
            "source_file": self.source_file,
        }


def _parse_atom_table(lines: List[str], start: int) -> tuple[List[MayerAtom], int]:
    atoms: List[MayerAtom] = []
    i = start
    while i < len(lines):
        parts = lines[i].split()
        # Expect: index symbol + 6 floats
        if len(parts) == 8 and parts[0].isdigit():
            try:
                atoms.append(MayerAtom(
                    index=int(parts[0]), symbol=parts[1],
                    na=float(parts[2]), za=float(parts[3]), qa=float(parts[4]),
                    va=float(parts[5]), bva=float(parts[6]), fa=float(parts[7]),
                ))
                i += 1
                continue
            except ValueError:
                break
        if atoms:           # table ended after at least one row
            break
        i += 1              # still searching for the first data row
    return atoms, i


def parse_mayer(filepath: "str | Path") -> Optional[MayerResult]:
    """Parse the Mayer block of an ORCA output; return None if it is absent.

    The last Mayer block in the file is used (matches the converged final analysis).
    """
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    starts = [i for i, ln in enumerate(lines) if _HEADER in ln]
    if not starts:
        return None
    block_start = starts[-1]

    result = MayerResult(source_file=str(path.resolve()))

    # Atom table: find the "ATOM ... FA" header after the block start.
    table_hdr = None
    for i in range(block_start, len(lines)):
        if lines[i].lstrip().startswith(_ATOM_TABLE_HEADER) and "VA" in lines[i] and "FA" in lines[i]:
            table_hdr = i
            break
    if table_hdr is not None:
        result.atoms, _ = _parse_atom_table(lines, table_hdr + 1)

    # Bond-order list.
    for i in range(block_start, len(lines)):
        if _BOND_HEADER in lines[i]:
            m = re.search(r"larger than\s+(-?\d+\.\d+)", lines[i])
            if m:
                result.threshold = float(m.group(1))
            j = i + 1
            while j < len(lines):
                found = _BOND_RE.findall(lines[j])
                if not found:
                    if result.bonds:            # list ended
                        break
                    j += 1
                    continue
                for ai, si, aj, sj, order in found:
                    result.bonds.append(
                        MayerBond(int(ai), si, int(aj), sj, float(order))
                    )
                j += 1
            break

    return result
