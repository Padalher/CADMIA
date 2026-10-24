"""POV-Ray scene writer and renderer."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from cadmia.core import Structure
from cadmia.structure.volumetric.cube_io import ISO_PRESETS

RESOLUTION_PRESETS = {
    "preview": (512, 512),
    "screen": (1024, 1024),
    "publication": (2560, 2560),
    "ultra": (5120, 5120),
}

POV_COLORS = {
    "H": "rgb <1.0, 1.0, 1.0>",
    "C": "rgb <0.45, 0.45, 0.45>",
    "N": "rgb <0.18, 0.31, 0.97>",
    "O": "rgb <1.0, 0.05, 0.05>",
    "Al": "rgb <0.75, 0.65, 0.65>",
    "Ti": "rgb <0.75, 0.76, 0.78>",
}

RADII = {
    "H": 0.25,
    "C": 0.35,
    "N": 0.35,
    "O": 0.35,
    "Al": 0.55,
    "Ti": 0.60,
}


def check_dependencies() -> dict:
    """Check whether POV-Ray and ImageMagick are available in PATH."""
    return {
        "povray": shutil.which("povray") is not None,
        "magick": (shutil.which("magick") or shutil.which("convert")) is not None,
    }


# A publication/ultra render is legitimately slow, so the budget is generous; the point
# is that a pathological scene cannot hang the program with no way back.
DEFAULT_RENDER_TIMEOUT_S = 3600


def render_pov(filename: str, resolution: Tuple[int, int],
               timeout_s: int = DEFAULT_RENDER_TIMEOUT_S) -> bool:
    """Render a .pov file with POV-Ray.

    Raises RuntimeError if POV-Ray fails or exceeds *timeout_s* seconds.
    """
    path = Path(filename)
    w, h = resolution
    output_name = f"{path.stem}.png"
    try:
        result = subprocess.run(
            [
                "povray",
                "+A0.1",
                "+AM2",
                "+R3",
                f"+W{w}",
                f"+H{h}",
                f"+O{output_name}",
                path.name,
            ],
            cwd=str(path.parent) if str(path.parent) else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"POV-Ray exceeded {timeout_s}s rendering {path.name} at {w}x{h} and was "
            f"stopped. Try a smaller --resolution preset."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"POV-Ray failed with code {result.returncode}:\n{result.stderr}"
        )
    return True


def compute_camera_size(
    structure: Structure,
    pos_iso=None,
    neg_iso=None,
    zoom: float = 0.7,
) -> float:
    """Compute camera half-size including atoms and isosurface vertices."""
    all_points = [structure.positions]
    if pos_iso is not None:
        all_points.append(pos_iso.vertices)
    if neg_iso is not None:
        all_points.append(neg_iso.vertices)
    points = np.vstack(all_points)
    com = structure.positions.mean(axis=0)
    bsphere = np.max(np.linalg.norm(points - com, axis=1))
    return float(bsphere * 1.5 * zoom)


def _vec(point) -> str:
    return f"<{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}>"


def _sphere_lines(structure: Structure) -> list[str]:
    lines = []
    for atom, position in zip(structure.atoms, structure.positions):
        color = POV_COLORS.get(atom, "rgb <0.5, 0.5, 0.5>")
        radius = RADII.get(atom, 0.45)
        lines.append(
            f"sphere {{ {_vec(position)}, {radius:.3f} "
            f"texture {{ pigment {{ color {color} }} finish {{ phong 0.35 }} }} }}"
        )
    return lines


def _mesh_lines(surface, color: str, name: str) -> list[str]:
    if surface is None:
        return []
    lines = [f"// {name}", "mesh2 {"]
    lines.append(f"  vertex_vectors {{ {len(surface.vertices)},")
    for vertex in surface.vertices:
        lines.append(f"    {_vec(vertex)},")
    lines.append("  }")
    lines.append(f"  face_indices {{ {len(surface.faces)},")
    for face in surface.faces:
        lines.append(f"    <{int(face[0])}, {int(face[1])}, {int(face[2])}>,")
    lines.append("  }")
    lines.append(
        f"  texture {{ pigment {{ color {color} transmit 0.35 }} "
        "finish { phong 0.2 } }}"
    )
    lines.append("}")
    return lines


def write_pov_scene(
    structure: Structure,
    pos_iso,
    neg_iso,
    filename: str,
    cam_half_size: Optional[float] = None,
    orientation=None,
) -> Path:
    """Write a POV-Ray scene file without requiring POV-Ray to be installed.

    When *orientation* (a `cadmia.structure.rendering.orientation.Orientation`) is given, the
    geometry and isosurfaces are rotated about the centroid so the scene reproduces the pose
    framed in the interactive viewer (the camera stays on +z). This is the orientation bridge.
    """
    base = Path(filename)
    path = base if base.suffix.lower() == ".pov" else base.with_suffix(".pov")
    path.parent.mkdir(parents=True, exist_ok=True)

    if orientation is not None and not getattr(orientation, "is_identity", False):
        center = structure.positions.mean(axis=0)
        structure = _rotated_structure(structure, orientation, center)
        pos_iso = _rotated_isosurface(pos_iso, orientation, center)
        neg_iso = _rotated_isosurface(neg_iso, orientation, center)

    com = structure.positions.mean(axis=0)
    if cam_half_size is None:
        cam_half_size = compute_camera_size(structure, pos_iso, neg_iso)
    camera_location = com + np.array([0.0, 0.0, max(cam_half_size, 1.0) * 4.0])

    lines = [
        "#version 3.7;",
        "global_settings { assumed_gamma 1.0 }",
        "background { color rgb <1, 1, 1> }",
        f"camera {{ orthographic location {_vec(camera_location)} look_at {_vec(com)} "
        f"right <{cam_half_size:.6f}, 0, 0> up <0, {cam_half_size:.6f}, 0> }}",
        f"light_source {{ {_vec(camera_location + np.array([2.0, 2.0, 2.0]))} color rgb <1,1,1> }}",
        "",
    ]
    lines.extend(_sphere_lines(structure))
    lines.extend(_mesh_lines(pos_iso, "rgb <0.1, 0.3, 1.0>", "positive isosurface"))
    lines.extend(_mesh_lines(neg_iso, "rgb <1.0, 0.2, 0.1>", "negative isosurface"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rotated_structure(structure: Structure, orientation, center) -> Structure:
    """Return a copy of *structure* with positions rotated by *orientation* about *center*."""
    from dataclasses import replace
    from cadmia.structure.rendering.orientation import apply_orientation
    new_pos = apply_orientation(structure.positions, orientation, center=center)
    return replace(structure, positions=new_pos)


def _rotated_isosurface(surface, orientation, center):
    """Rotate an isosurface's vertices (faces unchanged); None passes through.

    Isosurface is a NamedTuple, so use ._replace (not dataclasses.replace).
    """
    if surface is None:
        return None
    from cadmia.structure.rendering.orientation import apply_orientation
    new_verts = apply_orientation(surface.vertices, orientation, center=center)
    return surface._replace(vertices=new_verts)
