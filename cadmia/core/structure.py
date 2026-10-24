"""Structure — the central structural representation of the CADMIA suite.

Parsers return Structure objects; analyzers, agents and the database consume them. Holds a
flat (atoms[str], positions[ndarray]) representation; richer Atom/Bond/Cell views are optional
add-ons that build on this without replacing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from cadmia.core.atom import Atom
from cadmia.core.bond import Bond
from cadmia.core.cell import Cell


@dataclass
class Structure:
    """Internal representation of a molecular or cluster structure.

    Canonical storage is flat (``atoms`` symbols + ``positions`` array) for fast numerical
    work; ``bonds``/``cell`` are optional, and ``Atom`` is available as an on-demand view
    (``atom()``, ``to_atoms()``, ``from_atoms()``) without changing the storage.
    """

    atoms: List[str]
    positions: np.ndarray
    labels: List[str]
    structure_id: str
    source_file: str
    charge: int = 0
    multiplicity: int = 1
    point_group: Optional[str] = None
    energy: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    bonds: List[Bond] = field(default_factory=list)
    cell: Optional[Cell] = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3), received: {self.positions.shape}"
            )
        n_atoms = len(self.atoms)
        if len(self.positions) != n_atoms:
            raise ValueError(
                f"atoms ({n_atoms}) and positions ({len(self.positions)}) "
                "must have the same length."
            )
        if len(self.labels) != n_atoms:
            raise ValueError(f"labels ({len(self.labels)}) must have length {n_atoms}.")

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def formula(self) -> str:
        """Chemical formula in Hill notation."""
        from cadmia.core.elements import hill_formula
        return hill_formula(self.atoms)

    def symbols(self) -> List[str]:
        return list(self.atoms)

    def coordinates_array(self) -> np.ndarray:
        return self.positions

    def distance(self, i: int, j: int) -> float:
        """Distance (Å) between atoms i and j (0-based indices)."""
        return float(np.linalg.norm(self.positions[i] - self.positions[j]))

    def to_xyz(self, comment: str = "") -> str:
        """Render the structure as XYZ text."""
        lines = [str(self.n_atoms), comment or self.structure_id or ""]
        for sym, (x, y, z) in zip(self.atoms, self.positions):
            lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_xyz(cls, text: str, *, structure_id: str = "", source_file: str = "") -> "Structure":
        """Parse a single-structure XYZ block (self-contained; for full I/O use cadmia.io.xyz)."""
        raw = text.splitlines()
        n = int(raw[0].strip())
        atoms, positions = [], []
        for line in raw[2:2 + n]:
            parts = line.split()
            atoms.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return cls(atoms=atoms, positions=np.array(positions, dtype=float),
                   labels=list(atoms), structure_id=structure_id, source_file=source_file)

    def atom(self, i: int) -> Atom:
        """Return atom i as an Atom view (0-based)."""
        x, y, z = (float(c) for c in self.positions[i])
        return Atom(index=i, symbol=self.atoms[i], position=(x, y, z))

    def to_atoms(self) -> List[Atom]:
        """Rich per-atom view of the structure (built on demand from the flat arrays)."""
        return [self.atom(i) for i in range(self.n_atoms)]

    @classmethod
    def from_atoms(cls, atoms: List[Atom], *, structure_id: str = "", source_file: str = "",
                   charge: int = 0, multiplicity: int = 1) -> "Structure":
        """Build a Structure from a list of Atom objects."""
        symbols = [a.symbol for a in atoms]
        positions = np.array([a.position for a in atoms], dtype=float)
        return cls(atoms=symbols, positions=positions, labels=list(symbols),
                   structure_id=structure_id, source_file=source_file,
                   charge=charge, multiplicity=multiplicity)

    def to_dict(self) -> Dict:
        return {
            "atoms": self.atoms,
            "positions": self.positions.tolist(),
            "labels": self.labels,
            "structure_id": self.structure_id,
            "source_file": str(self.source_file),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "point_group": self.point_group,
            "energy": self.energy,
            "metadata": self.metadata,
            "bonds": [b.to_dict() for b in self.bonds],
            "cell": self.cell.to_dict() if self.cell else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Structure":
        payload = dict(data)
        payload["positions"] = np.array(payload["positions"])
        if payload.get("bonds"):
            payload["bonds"] = [Bond.from_dict(b) for b in payload["bonds"]]
        if payload.get("cell"):
            payload["cell"] = Cell.from_dict(payload["cell"])
        return cls(**payload)
