"""Interactive structure viewer (3Dmol.js) — preview, highlight, and export orientation.

Self-contained HTML page (3Dmol.js from CDN, geometry/cube embedded) that lets you rotate,
zoom and pan a structure of considerable size, highlight the relevant atoms (CDFT top sites,
the binding site, any atoms of interest), optionally overlay a .cube isosurface (HOMO/LUMO/
Fukui/dual/MEP/Δρ), and **export the camera orientation** with a button. That orientation feeds
the renderers (`cadmia.structure.rendering.orientation`) so the POV-Ray / VMD figure shows
exactly the framed pose — "what you orient is what you render".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from cadmia.core import Structure

_CDN = "https://3Dmol.org/build/3Dmol-min.js"

_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{cdn}"></script>
<style>
 body {{ margin:0; font-family:sans-serif; }}
 #bar {{ padding:8px 14px; background:#f4f4f4; border-bottom:1px solid #ddd; }}
 #viewer {{ width:100%; height:88vh; position:relative; }}
 button {{ font-size:14px; padding:6px 12px; cursor:pointer; }}
 #status {{ margin-left:12px; color:#444; font-size:13px; }}
</style>
</head>
<body>
<div id="bar">
  <strong>{title}</strong>
  <button onclick="exportView()">Export orientation (.json)</button>
  <span id="status">Rotate and frame the structure, then export the orientation for rendering.</span>
</div>
<div id="viewer"></div>
<script>
var xyz = {xyz_json};
var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
viewer.addModel(xyz, "xyz");
viewer.setStyle({{}}, {{stick: {{radius: 0.12}}, sphere: {{scale: 0.28}}}});
{highlight_js}
{cube_js}
viewer.zoomTo();
viewer.render();

function exportView() {{
  var v = viewer.getView();
  var payload = {{viewer: "3dmol", view: v}};
  var blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "{orient_name}";
  a.click();
  document.getElementById("status").textContent =
    "Orientation exported → {orient_name}.  Use it with: cadmia structure render ... --orientation {orient_name}";
}}
</script>
</body>
</html>
"""

_HIGHLIGHT_COLORS = ["#e6c200", "#2ca02c", "#d62728", "#1f77b4", "#9467bd"]


def _highlight_js(highlight: List[int], labels: Optional[List[str]],
                  structure: Structure) -> str:
    """3Dmol calls to recolor/enlarge highlighted atoms (0-based) and label them."""
    if not highlight:
        return ""
    lines = []
    for n, idx in enumerate(highlight):
        color = _HIGHLIGHT_COLORS[n % len(_HIGHLIGHT_COLORS)]
        lines.append(
            f'viewer.setStyle({{index: {idx}}}, '
            f'{{stick: {{radius: 0.12}}, sphere: {{scale: 0.42, color: "{color}"}}}});'
        )
        if 0 <= idx < structure.n_atoms:
            pos = structure.positions[idx]
            label = labels[n] if labels and n < len(labels) else f"{structure.atoms[idx]}{idx + 1}"
            lines.append(
                f'viewer.addLabel("{label}", {{position: {{x: {pos[0]:.4f}, y: {pos[1]:.4f}, '
                f'z: {pos[2]:.4f}}}, backgroundColor: "{color}", fontColor: "black", '
                f'fontSize: 12, backgroundOpacity: 0.85}});'
            )
    return "\n".join(lines)


def _cube_js(cube_text: Optional[str], isovalue: Optional[float]) -> str:
    """3Dmol calls to overlay a .cube isosurface (±isovalue for signed fields)."""
    if not cube_text:
        return ""
    iso = isovalue if isovalue is not None else 0.02
    data = json.dumps(cube_text)
    return (
        f'var cube = {data};\n'
        f'viewer.addVolumetricData(cube, "cube", {{isoval: {iso:g}, color: "blue", opacity: 0.85}});\n'
        f'viewer.addVolumetricData(cube, "cube", {{isoval: {-iso:g}, color: "red", opacity: 0.85}});'
    )


def structure_viewer_html(
    structure: Structure,
    *,
    title: str = "CADMIA viewer",
    highlight: Optional[List[int]] = None,
    highlight_labels: Optional[List[str]] = None,
    cube_text: Optional[str] = None,
    isovalue: Optional[float] = None,
    orientation_filename: str = "orientation.json",
) -> str:
    """Build a self-contained interactive 3Dmol.js page for *structure*.

    highlight: 0-based atom indices to color/enlarge/label (e.g. CDFT top sites, binding site).
    cube_text: raw .cube contents to overlay as an isosurface (optional).
    """
    return _PAGE.format(
        title=title, cdn=_CDN,
        xyz_json=json.dumps(structure.to_xyz(comment=title)),
        highlight_js=_highlight_js(highlight or [], highlight_labels, structure),
        cube_js=_cube_js(cube_text, isovalue),
        orient_name=orientation_filename,
    )


def write_structure_viewer(
    source: "str | Path | Structure",
    filepath: "str | Path",
    *,
    title: Optional[str] = None,
    highlight: Optional[List[int]] = None,
    highlight_labels: Optional[List[str]] = None,
    cube: "str | Path | None" = None,
    isovalue: Optional[float] = None,
) -> Path:
    """Write an interactive viewer HTML from a structure (or a .xyz / Gaussian-ORCA output)."""
    if isinstance(source, Structure):
        structure = source
    else:
        from cadmia.io import structure_from_file
        structure = structure_from_file(source)
    cube_text = Path(cube).read_text(encoding="utf-8") if cube else None
    if title is None:
        title = structure.structure_id or "CADMIA viewer"
    html = structure_viewer_html(
        structure, title=title, highlight=highlight, highlight_labels=highlight_labels,
        cube_text=cube_text, isovalue=isovalue,
        orientation_filename=f"{Path(filepath).stem}_orientation.json",
    )
    out = Path(filepath)
    out.write_text(html, encoding="utf-8")
    return out
