"""Geometry helpers for static rendering."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from cadmia.core import Structure
from cadmia.structure.core.config import Config, load_config
from cadmia.structure.volumetric.cube_io import Isosurface


def rotation_matrix(ax: float, ay: float) -> np.ndarray:
    """Return a 3D rotation matrix for rotations around x and y in degrees."""
    x = np.deg2rad(ax)
    y = np.deg2rad(ay)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ],
        dtype=float,
    )
    return ry @ rx


def rotate_positions(positions, com, M) -> np.ndarray:
    """Rotate positions around a center of mass."""
    positions = np.asarray(positions, dtype=float)
    com = np.asarray(com, dtype=float)
    matrix = np.asarray(M, dtype=float)
    return (positions - com) @ matrix.T + com


def rotate_normals(normals, M) -> np.ndarray:
    """Rotate normals and preserve unit length."""
    normals = np.asarray(normals, dtype=float)
    matrix = np.asarray(M, dtype=float)
    rotated = normals @ matrix.T
    lengths = np.linalg.norm(rotated, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return rotated / lengths


def _rotate_iso(surface, com, matrix):
    if surface is None:
        return None
    return Isosurface(
        vertices=rotate_positions(surface.vertices, com, matrix),
        faces=surface.faces.copy(),
        normals=rotate_normals(surface.normals, matrix),
    )


def _atomic_masses(structure: Structure, config: Config) -> np.ndarray:
    masses = np.array(
        [config.get_atomic_mass(element) for element in structure.atoms],
        dtype=float,
    )
    if np.any(masses <= 0.0):
        missing = sorted(
            {
                element
                for element, mass in zip(structure.atoms, masses)
                if mass <= 0.0
            }
        )
        raise ValueError(f"Missing atomic masses for: {', '.join(missing)}")
    return masses


def _center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.average(np.asarray(positions, dtype=float), axis=0, weights=masses)


def _inertia_tensor(centered: np.ndarray, masses: np.ndarray) -> np.ndarray:
    tensor = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for mass, vector in zip(masses, centered):
        tensor += mass * ((np.dot(vector, vector) * identity) - np.outer(vector, vector))
    return tensor


def align_to_inertia(structure: Structure, pos_iso, neg_iso, config: Config = None):
    """Align a structure and optional isosurfaces to principal inertia axes."""
    cfg = config or load_config()
    positions = np.asarray(structure.positions, dtype=float)
    masses = _atomic_masses(structure, cfg)
    com = _center_of_mass(positions, masses)
    centered = positions - com
    if len(positions) < 2:
        matrix = np.eye(3)
        moments = np.zeros(3, dtype=float)
    else:
        moments, vectors = np.linalg.eigh(_inertia_tensor(centered, masses))
        if np.linalg.det(vectors) < 0:
            vectors[:, 0] *= -1.0
        matrix = vectors.T

    metadata = dict(structure.metadata)
    metadata["inertia_center_of_mass"] = com.tolist()
    metadata["inertia_moments"] = moments.tolist()
    metadata["inertia_uses_atomic_masses"] = True

    # `replace` rather than rebuilding field by field: a rotation changes the positions
    # and nothing else, and listing the fields by hand silently dropped bonds,
    # point_group and cell every time a structure was aligned.
    rotated_structure = replace(
        structure,
        positions=rotate_positions(positions, com, matrix),
        metadata=metadata,
    )
    return rotated_structure, _rotate_iso(pos_iso, com, matrix), _rotate_iso(neg_iso, com, matrix)
