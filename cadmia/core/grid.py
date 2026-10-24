"""VolumetricGrid — a 3D scalar field such as density, orbital, or MEP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cadmia.core.structure import Structure


@dataclass
class VolumetricGrid:
    """Three-dimensional scalar field such as density, orbital, or MEP."""

    structure: Structure
    origin: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    data: np.ndarray
    field_type: str
    source_file: str = ""

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.vx = np.asarray(self.vx, dtype=float)
        self.vy = np.asarray(self.vy, dtype=float)
        self.vz = np.asarray(self.vz, dtype=float)
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 3:
            raise ValueError(f"data must be 3D, received: {self.data.ndim}D")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def vmin(self) -> float:
        return float(self.data.min())

    @property
    def vmax(self) -> float:
        return float(self.data.max())

    # ── Grid arithmetic (for Fukui/dual/Δρ maps from density cubes) ──
    def same_grid_as(self, other: "VolumetricGrid", tol: float = 1e-4) -> bool:
        """True when two grids share voxel layout (shape, origin, step vectors).

        Density-difference maps (Fukui f±, dual Δf, deformation Δρ) are only meaningful
        when the operands are voxel-aligned; the cube generator must emit every species
        on one common grid.
        """
        return (
            self.shape == other.shape
            and np.allclose(self.origin, other.origin, atol=tol)
            and np.allclose(self.vx, other.vx, atol=tol)
            and np.allclose(self.vy, other.vy, atol=tol)
            and np.allclose(self.vz, other.vz, atol=tol)
        )

    def _require_aligned(self, other: "VolumetricGrid") -> None:
        if not self.same_grid_as(other):
            raise ValueError(
                "grids are not voxel-aligned (shape/origin/step mismatch); generate every "
                "cube on one common grid before combining them."
            )

    def _with_data(self, data: np.ndarray, field_type: str) -> "VolumetricGrid":
        return VolumetricGrid(
            structure=self.structure, origin=self.origin, vx=self.vx, vy=self.vy, vz=self.vz,
            data=data, field_type=field_type, source_file="",
        )

    def __sub__(self, other: "VolumetricGrid") -> "VolumetricGrid":
        self._require_aligned(other)
        return self._with_data(self.data - other.data, "difference")

    def __add__(self, other: "VolumetricGrid") -> "VolumetricGrid":
        self._require_aligned(other)
        return self._with_data(self.data + other.data, self.field_type)

    def scaled(self, k: float) -> "VolumetricGrid":
        return self._with_data(self.data * float(k), self.field_type)
