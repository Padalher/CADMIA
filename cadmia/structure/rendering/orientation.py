"""Orientation bridge — carry a viewer's camera orientation into the renderers.

The interactive viewer (`cadmia view`, 3Dmol.js) lets you rotate/zoom a structure to expose
the relevant sites, then export its orientation with `viewer.getView()`. 3Dmol's view array is
``[cx, cy, cz, zoom, qx, qy, qz, qw]``: a center, a zoom, and a **quaternion** for the rotation
applied to the scene. This module turns that quaternion into a 3×3 rotation matrix and applies it
to the atomic positions, so the POV-Ray / VMD scene is rendered in exactly the pose you framed
("what you orient is what you render").

Convention: we rotate the *geometry* about its center and keep the camera fixed on +z, which is
the simplest way to reproduce a 3Dmol view in an orthographic renderer. The sign convention of
the quaternion is validated visually on a first render; `invert=True` flips it if a build differs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Orientation:
    """A viewer camera orientation: rotation quaternion (x, y, z, w) + zoom."""
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    zoom: float = 1.0

    @property
    def is_identity(self) -> bool:
        return abs(self.qx) < 1e-9 and abs(self.qy) < 1e-9 and abs(self.qz) < 1e-9


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit-normalize (x, y, z, w) and return the 3×3 rotation matrix.

    Identity quaternion (0, 0, 0, 1) → identity matrix. Standard formula.
    """
    n = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def orientation_from_3dmol_view(view: List[float]) -> Orientation:
    """Parse a 3Dmol.js ``getView()`` array → Orientation.

    Accepts the 8-element ``[cx, cy, cz, zoom, qx, qy, qz, qw]`` form (and tolerates a longer
    array by reading the first 8). The center is not needed here (we rotate about the structure's
    own centroid), only the zoom and the quaternion.
    """
    v = list(view)
    if len(v) < 8:
        raise ValueError(f"3Dmol view array must have at least 8 elements, got {len(v)}.")
    zoom, qx, qy, qz, qw = v[3], v[4], v[5], v[6], v[7]
    return Orientation(qx=qx, qy=qy, qz=qz, qw=qw, zoom=zoom)


def load_orientation(path: "str | Path") -> Orientation:
    """Load an orientation JSON written by the viewer's export button.

    Accepts ``{"view": [...]}`` (raw 3Dmol array) or ``{"qx":..,"qy":..,"qz":..,"qw":..,"zoom":..}``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "view" in data:
        return orientation_from_3dmol_view(data["view"])
    return Orientation(qx=data.get("qx", 0.0), qy=data.get("qy", 0.0), qz=data.get("qz", 0.0),
                       qw=data.get("qw", 1.0), zoom=data.get("zoom", 1.0))


def rotation_matrix(orientation: Orientation, *, invert: bool = False) -> np.ndarray:
    """3×3 rotation matrix for an orientation (transpose when invert=True)."""
    R = quat_to_rotation_matrix(orientation.qx, orientation.qy, orientation.qz, orientation.qw)
    return R.T if invert else R


def apply_orientation(positions: np.ndarray, orientation: Orientation, *,
                      center: "np.ndarray | None" = None, invert: bool = False) -> np.ndarray:
    """Rotate positions (N×3) about *center* (default: their centroid) by the orientation."""
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        return pos
    c = pos.mean(axis=0) if center is None else np.asarray(center, dtype=float)
    R = rotation_matrix(orientation, invert=invert)
    return (pos - c) @ R.T + c


def vmd_rotate_matrix(orientation: Orientation, *, invert: bool = False) -> str:
    """Tcl literal for ``molinfo top set rotate_matrix`` (a single 4×4 row-major matrix).

    VMD applies this as the view rotation, so the loaded molecule shows the framed pose.
    """
    R = rotation_matrix(orientation, invert=invert)
    m = np.eye(4)
    m[:3, :3] = R
    rows = " ".join("{" + " ".join(f"{v:.6f}" for v in row) + "}" for row in m)
    return "{{" + rows + "}}"
