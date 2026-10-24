"""Volumetric grid utilities."""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from cadmia.core import VolumetricGrid


class Isosurface(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


def grid_to_real(grid: VolumetricGrid, vertices: np.ndarray) -> np.ndarray:
    """Convert grid-index vertices to real-space Angstrom coordinates."""
    vertices = np.asarray(vertices, dtype=float)
    return (
        grid.origin
        + vertices[:, 0:1] * grid.vx
        + vertices[:, 1:2] * grid.vy
        + vertices[:, 2:3] * grid.vz
    )


def upsample_grid(
    grid: VolumetricGrid,
    zoom: int = 3,
    sigma: float = 1.0,
) -> VolumetricGrid:
    """Upsample a VolumetricGrid with cubic interpolation and Gaussian smoothing."""
    if zoom <= 0:
        raise ValueError("zoom must be a positive integer.")

    from scipy.ndimage import gaussian_filter, zoom as ndi_zoom

    data = ndi_zoom(grid.data, zoom=zoom, order=3)
    if sigma and sigma > 0:
        data = gaussian_filter(data, sigma=sigma)

    return VolumetricGrid(
        structure=grid.structure,
        origin=grid.origin.copy(),
        vx=grid.vx / zoom,
        vy=grid.vy / zoom,
        vz=grid.vz / zoom,
        data=data,
        field_type=grid.field_type,
        source_file=grid.source_file,
    )


def _extract_one(grid: VolumetricGrid, level: Optional[float], invert_normals: bool):
    if level is None:
        return None
    if level < grid.vmin or level > grid.vmax:
        return None

    from skimage.measure import marching_cubes

    try:
        vertices, faces, normals, _ = marching_cubes(grid.data, level=level)
    except ValueError:
        return None

    real_vertices = grid_to_real(grid, vertices)
    normals = np.asarray(normals, dtype=float)
    if invert_normals:
        normals = -normals
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    normals = normals / lengths
    return Isosurface(
        vertices=real_vertices,
        faces=np.asarray(faces, dtype=int),
        normals=normals,
    )


def extract_isosurfaces(
    grid: VolumetricGrid,
    iso_pos: Optional[float],
    iso_neg: Optional[float],
) -> Tuple[Optional[Isosurface], Optional[Isosurface]]:
    """Extract positive and negative isosurfaces with marching cubes."""
    pos = _extract_one(grid, iso_pos, invert_normals=False)
    neg = _extract_one(grid, iso_neg, invert_normals=True)
    return pos, neg


ISO_PRESETS = {
    "homo_lumo": (0.040, -0.040),
    "fukui": (0.010, -0.010),
    "dual": (0.005, -0.005),
    "density": (0.002, None),
    "mep": (0.050, -0.050),
    "adndp": (10.000, -26.9),
}
"""Reference isovalues by field type; callers may override them."""
