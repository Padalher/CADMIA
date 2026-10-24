"""Cell — periodic unit cell. Prepared for future CIF/POSCAR/periodic structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Cell:
    lattice_vectors: Optional[np.ndarray] = None   # (3, 3), rows are a/b/c vectors
    periodic:        bool = False
    metadata:        dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lattice_vectors is not None:
            self.lattice_vectors = np.asarray(self.lattice_vectors, dtype=float)
            if self.lattice_vectors.shape != (3, 3):
                raise ValueError("lattice_vectors must have shape (3, 3).")

    def volume(self) -> Optional[float]:
        if self.lattice_vectors is None:
            return None
        return float(abs(np.linalg.det(self.lattice_vectors)))

    def cart_to_frac(self, coords: np.ndarray) -> np.ndarray:
        if self.lattice_vectors is None:
            raise ValueError("No lattice_vectors set.")
        return np.asarray(coords, dtype=float) @ np.linalg.inv(self.lattice_vectors)

    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        if self.lattice_vectors is None:
            raise ValueError("No lattice_vectors set.")
        return np.asarray(frac, dtype=float) @ self.lattice_vectors

    def to_dict(self) -> dict:
        return {
            "lattice_vectors": self.lattice_vectors.tolist() if self.lattice_vectors is not None else None,
            "periodic": self.periodic,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cell":
        lv = d.get("lattice_vectors")
        return cls(lattice_vectors=np.array(lv) if lv is not None else None,
                   periodic=d.get("periodic", False), metadata=d.get("metadata", {}))
