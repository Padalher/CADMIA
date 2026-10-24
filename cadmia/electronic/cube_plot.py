"""Interactive HTML viewer for volumetric .cube data (orbitals, density, MEP).

Self-contained generator: the page embeds the grid as JSON and loads plotly.js from its
CDN, so no Python plotting/marching-cubes dependency is required — plotly's Isosurface
trace builds the surface in the browser (same exportable-HTML philosophy as the DOS/PDOS
plots). Works for the axis-aligned grids that Gaussian/ORCA cube files use; per-point
positions are computed from origin + i·vx + j·vy + k·vz, so skewed grids also place
correctly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from cadmia.core import VolumetricGrid

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

# CPK-ish element colors and relative marker sizes (px) for the atom overlay.
_ELEMENT_COLORS = {
    "H": "#dddddd", "C": "#2c2c2c", "N": "#1f77b4", "O": "#d62728",
    "Al": "#bfa6a0", "Si": "#daa520", "Ti": "#8c8c9e", "S": "#e6c200",
    "Fe": "#e06633", "Cu": "#c88033", "Zn": "#7d80b0",
}
_DEFAULT_COLOR = "#9467bd"

_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{cdn}" charset="utf-8"></script>
</head>
<body>
<div id="plot" style="width:100%;height:90vh;"></div>
{controls}
<script>
var data = {data};
var layout = {layout};
Plotly.newPlot("plot", data, layout, {{responsive: true}});
{slider_js}
</script>
</body>
</html>
"""


def _downsample_stride(shape, max_points: int) -> int:
    nx, ny, nz = shape
    stride = 1
    while len(range(0, nx, stride)) * len(range(0, ny, stride)) * len(range(0, nz, stride)) > max_points:
        stride += 1
    return stride


def _grid_points(grid: VolumetricGrid, stride: int):
    nx, ny, nz = grid.data.shape
    ii = np.arange(0, nx, stride)
    jj = np.arange(0, ny, stride)
    kk = np.arange(0, nz, stride)
    I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
    pos = (grid.origin
           + I[..., None] * grid.vx
           + J[..., None] * grid.vy
           + K[..., None] * grid.vz)
    values = grid.data[np.ix_(ii, jj, kk)]
    return pos[..., 0].ravel(), pos[..., 1].ravel(), pos[..., 2].ravel(), values.ravel()


def _isosurface_trace(x, y, z, v, *, isomin, isomax, colorscale, name, opacity):
    return {
        "type": "isosurface",
        "x": x, "y": y, "z": z, "value": v,
        "isomin": isomin, "isomax": isomax,
        "surface": {"count": 1, "show": True},
        "caps": {"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        "showscale": False, "colorscale": colorscale, "opacity": opacity,
        "name": name, "showlegend": True,
    }


def _atom_trace(grid: VolumetricGrid):
    atoms = grid.structure.atoms
    pos = np.asarray(grid.structure.positions, dtype=float)
    if len(atoms) == 0:
        return None
    colors = [_ELEMENT_COLORS.get(a, _DEFAULT_COLOR) for a in atoms]
    return {
        "type": "scatter3d", "mode": "markers+text",
        "x": [round(float(p), 4) for p in pos[:, 0]],
        "y": [round(float(p), 4) for p in pos[:, 1]],
        "z": [round(float(p), 4) for p in pos[:, 2]],
        "text": list(atoms), "textposition": "top center",
        "marker": {"size": 6, "color": colors, "line": {"width": 0.5, "color": "#222"}},
        "name": "atoms", "showlegend": True,
    }


def _controls_and_js(max_abs: float, iso: float, pos_idx: int, neg_idx: int) -> tuple[str, str]:
    """Build the iso slider markup + a JS handler that restyles the surfaces in place."""
    if max_abs <= 0:
        return "", ""
    smin = max(1e-4, round(max_abs * 0.005, 6))
    smax = round(max_abs * 0.99, 6)
    step = round(max_abs / 200.0, 6) or 1e-4
    controls = (
        '<div style="padding:6px 14px;font-family:sans-serif;font-size:14px;">'
        'isovalue = <span id="isoLabel">{iso:g}</span>&nbsp;'
        '<input type="range" id="isoSlider" min="{smin:g}" max="{smax:g}" step="{step:g}" '
        'value="{iso:g}" style="width:60%;vertical-align:middle;"></div>'
    ).format(iso=iso, smin=smin, smax=smax, step=step)
    js = (
        "var POS=%d, NEG=%d, s=document.getElementById('isoSlider');"
        "if(s){s.addEventListener('input',function(){"
        "var v=parseFloat(this.value);"
        "document.getElementById('isoLabel').textContent=v.toPrecision(3);"
        "if(POS>=0)Plotly.restyle('plot',{isomin:[v]},[POS]);"
        "if(NEG>=0)Plotly.restyle('plot',{isomax:[-v]},[NEG]);"
        "});}" % (pos_idx, neg_idx)
    )
    return controls, js


def cube_html(
    grid: VolumetricGrid,
    *,
    title: str = "Volumetric field",
    isovalue: Optional[float] = None,
    max_points: int = 60000,
    opacity: float = 0.5,
    interactive: bool = True,
) -> tuple[str, list]:
    """Render *grid* as a self-contained interactive HTML page.

    Returns (html, warnings). When the field has both signs (orbitals, MEP) two
    isosurfaces are drawn at ±isovalue; otherwise a single positive surface is drawn.
    isovalue defaults to 15% of the field's maximum absolute value. With *interactive*
    a slider lets you retune the isovalue live in the browser (no regeneration).
    """
    warnings: list = []
    vmin, vmax = grid.vmin, grid.vmax
    max_abs = max(abs(vmin), abs(vmax))
    if max_abs == 0:
        warnings.append("Field is uniformly zero; nothing to contour.")
    iso = isovalue if isovalue is not None else round(0.15 * max_abs, 6)
    if iso <= 0 and max_abs > 0:
        iso = round(0.15 * max_abs, 6)

    # Off-axis (non-orthogonal) check on the step vectors.
    for vec, axis in ((grid.vx, "x"), (grid.vy, "y"), (grid.vz, "z")):
        off = float(np.linalg.norm(vec) - abs(vec[{"x": 0, "y": 1, "z": 2}[axis]]))
        if off > 1e-6:
            warnings.append("Grid step vectors are not axis-aligned; surface placement is approximate.")
            break

    stride = _downsample_stride(grid.data.shape, max_points)
    if stride > 1:
        warnings.append(
            f"Grid {grid.data.shape} downsampled by stride {stride} to stay under "
            f"{max_points} points (raise --max-points for finer surfaces)."
        )
    x, y, z, v = _grid_points(grid, stride)
    x = [round(float(c), 3) for c in x]
    y = [round(float(c), 3) for c in y]
    z = [round(float(c), 3) for c in z]
    v = [round(float(c), 6) for c in v]

    traces = []
    pos_idx = neg_idx = -1
    has_negative = vmin < -iso
    if max_abs > 0:
        pos_idx = len(traces)
        traces.append(_isosurface_trace(
            x, y, z, v, isomin=iso, isomax=float(vmax),
            colorscale=[[0, "#1f77b4"], [1, "#1f77b4"]],
            name=f"+{iso:g}", opacity=opacity,
        ))
        if has_negative:
            neg_idx = len(traces)
            traces.append(_isosurface_trace(
                x, y, z, v, isomin=float(vmin), isomax=-iso,
                colorscale=[[0, "#d62728"], [1, "#d62728"]],
                name=f"−{iso:g}", opacity=opacity,
            ))
    atom_trace = _atom_trace(grid)
    if atom_trace:
        traces.append(atom_trace)

    controls, slider_js = ("", "")
    if interactive:
        controls, slider_js = _controls_and_js(max_abs, iso, pos_idx, neg_idx)

    layout = {
        "title": {"text": f"{title}  (iso = ±{iso:g})" if has_negative else f"{title}  (iso = {iso:g})"},
        "scene": {
            "xaxis": {"title": "x (Å)"}, "yaxis": {"title": "y (Å)"}, "zaxis": {"title": "z (Å)"},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "template": "plotly_white",
    }
    html = _PAGE.format(
        title=title, cdn=_PLOTLY_CDN,
        data=json.dumps(traces), layout=json.dumps(layout),
        controls=controls, slider_js=slider_js,
    )
    return html, warnings


def write_cube_html(
    source: "str | Path | VolumetricGrid",
    filepath: "str | Path",
    *,
    title: Optional[str] = None,
    isovalue: Optional[float] = None,
    max_points: int = 60000,
    field_type: str = "other",
) -> list:
    """Parse a .cube (or accept a VolumetricGrid) and write an interactive HTML viewer.

    Returns the list of warnings produced during rendering.
    """
    if isinstance(source, VolumetricGrid):
        grid = source
    else:
        from cadmia.io.cube import parse_cube
        grid = parse_cube(source, field_type=field_type)
    if title is None:
        title = Path(grid.source_file).name if grid.source_file else "Volumetric field"
    html, warnings = cube_html(grid, title=title, isovalue=isovalue, max_points=max_points)
    Path(filepath).write_text(html, encoding="utf-8")
    return warnings
