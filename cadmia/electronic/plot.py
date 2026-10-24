"""Interactive HTML plots for DOS/PDOS curves.

Self-contained generator: the page embeds the curves as JSON and loads plotly.js
from its CDN, so no Python plotting dependency is required (same exportable-HTML
philosophy as the `cadmia structure` plots, without importing plotly).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from cadmia.electronic.dos import DOSResult, PDOSResult

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

_ELEMENT_COLORS = {
    "Ti": "#8c8c9e", "Al": "#c5c5c5", "O": "#d62728",
    "H": "#7f7f7f", "C": "#2c2c2c", "N": "#1f77b4",
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{cdn}" charset="utf-8"></script>
</head>
<body>
<div id="plot" style="width:100%;height:92vh;"></div>
<script>
var data = {data};
var layout = {layout};
Plotly.newPlot("plot", data, layout, {{responsive: true}});
</script>
</body>
</html>
"""


def _frontier_decorations(homo_ev: Optional[float], lumo_ev: Optional[float],
                          gap_ev: Optional[float]) -> tuple[list, list]:
    shapes, annotations = [], []
    for x, label in ((homo_ev, "HOMO"), (lumo_ev, "LUMO")):
        if x is None:
            continue
        shapes.append({
            "type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1,
            "line": {"color": "#444", "width": 1, "dash": "dash"},
        })
        annotations.append({
            "x": x, "yref": "paper", "y": 1.02, "showarrow": False,
            "text": f"{label} = {x:.2f} eV", "font": {"size": 11},
        })
    if gap_ev is not None and homo_ev is not None and lumo_ev is not None:
        annotations.append({
            "x": (homo_ev + lumo_ev) / 2.0, "yref": "paper", "y": 0.96,
            "showarrow": False, "text": f"gap = {gap_ev:.2f} eV",
            "font": {"size": 12},
        })
    return shapes, annotations


# Opening view around the frontier orbitals; the core levels stay in the data, one zoom-out away.
_VIEW_BELOW_HOMO_EV, _VIEW_ABOVE_LUMO_EV = 20.0, 10.0


def _layout(title: str, ylabel: str, shapes: list, annotations: list,
            homo: Optional[float] = None, lumo: Optional[float] = None) -> dict:
    xaxis = {"title": {"text": "Energy (eV)"}}
    if homo is not None and lumo is not None:
        xaxis["range"] = [homo - _VIEW_BELOW_HOMO_EV, lumo + _VIEW_ABOVE_LUMO_EV]
    return {
        "title": {"text": title},
        "xaxis": xaxis,
        "yaxis": {"title": {"text": ylabel}, "rangemode": "tozero"},
        "shapes": shapes,
        "annotations": annotations,
        "template": "plotly_white",
        "legend": {"orientation": "h", "y": -0.15},
    }


def _render(title: str, traces: list, layout: dict) -> str:
    return _PAGE.format(
        title=title, cdn=_PLOTLY_CDN,
        data=json.dumps(traces), layout=json.dumps(layout),
    )


def dos_html(dos: DOSResult, *, title: str = "Density of states") -> str:
    """Build a self-contained interactive HTML page for a total-DOS curve."""
    energy = dos.energy_ev.tolist()
    traces = [{
        "x": energy, "y": dos.dos_total.tolist(), "mode": "lines",
        "name": "DOS total", "line": {"color": "#1f77b4", "width": 2},
    }]
    if dos.is_spin_polarized:
        traces.append({
            "x": energy, "y": dos.dos_alpha.tolist(), "mode": "lines",
            "name": "alpha", "line": {"color": "#2ca02c", "width": 1, "dash": "dot"},
        })
        traces.append({
            "x": energy, "y": dos.dos_beta.tolist(), "mode": "lines",
            "name": "beta", "line": {"color": "#d62728", "width": 1, "dash": "dot"},
        })
    shapes, annotations = _frontier_decorations(dos.homo_ev, dos.lumo_ev, dos.gap_ev)
    return _render(title, traces,
                   _layout(title, "DOS (states/eV)", shapes, annotations, dos.homo_ev, dos.lumo_ev))


def pdos_html(pdos: PDOSResult, *, title: str = "Projected density of states") -> str:
    """Build a self-contained interactive HTML page for element-projected DOS."""
    energy = pdos.energy_ev.tolist()
    traces = [{
        "x": energy, "y": pdos.total.tolist(), "mode": "lines",
        "name": "DOS total", "line": {"color": "#333333", "width": 1, "dash": "dash"},
    }]
    fallback = iter(_FALLBACK_COLORS * 4)
    for element in sorted(pdos.series):
        color = _ELEMENT_COLORS.get(element) or next(fallback)
        traces.append({
            "x": energy, "y": pdos.series[element].tolist(), "mode": "lines",
            "name": element, "line": {"color": color, "width": 2},
        })
    shapes, annotations = _frontier_decorations(pdos.homo_ev, pdos.lumo_ev, pdos.gap_ev)
    return _render(title, traces,
                   _layout(title, "PDOS (states/eV)", shapes, annotations, pdos.homo_ev, pdos.lumo_ev))


def ir_html(spectrum, *, title: str = "IR spectrum") -> str:
    """Build a self-contained interactive HTML page for a broadened IR spectrum.

    The broadened curve is drawn as a line; each IR-active mode is overlaid as a
    stick at its scaled wavenumber. By convention IR plots run high→low wavenumber.
    """
    x = spectrum.wavenumber_cm1.tolist()
    traces = [{
        "x": x, "y": spectrum.intensity.tolist(), "mode": "lines",
        "name": f"{spectrum.lineshape} (FWHM {spectrum.fwhm_cm1:g})",
        "line": {"color": "#1f77b4", "width": 2},
    }]
    sticks_x, sticks_y = [], []
    for pos, height in zip(spectrum.peak_positions, spectrum.peak_intensities):
        sticks_x += [float(pos), float(pos), None]
        sticks_y += [0.0, float(height), None]
    if sticks_x:
        traces.append({
            "x": sticks_x, "y": sticks_y, "mode": "lines", "name": "modes",
            "line": {"color": "#d62728", "width": 1}, "yaxis": "y2",
        })
    scale_note = "" if spectrum.scaling_factor == 1.0 else f"  (scale ×{spectrum.scaling_factor:g})"
    layout = {
        "title": {"text": title + scale_note},
        "xaxis": {"title": {"text": "Wavenumber (cm⁻¹)"}, "autorange": "reversed"},
        "yaxis": {"title": {"text": "Intensity (arb. units)"}, "rangemode": "tozero"},
        "yaxis2": {"overlaying": "y", "side": "right", "rangemode": "tozero",
                   "title": {"text": "IR intensity (km/mol)"}, "showgrid": False},
        "template": "plotly_white",
        "legend": {"orientation": "h", "y": -0.15},
    }
    return _render(title, traces, layout)


def write_ir_html(spectrum, filepath: str | Path, *, title: str = "IR spectrum") -> Path:
    path = Path(filepath)
    path.write_text(ir_html(spectrum, title=title), encoding="utf-8")
    return path


def write_dos_html(dos: DOSResult, filepath: str | Path,
                   *, title: str = "Density of states") -> Path:
    path = Path(filepath)
    path.write_text(dos_html(dos, title=title), encoding="utf-8")
    return path


def write_pdos_html(pdos: PDOSResult, filepath: str | Path,
                    *, title: str = "Projected density of states") -> Path:
    path = Path(filepath)
    path.write_text(pdos_html(pdos, title=title), encoding="utf-8")
    return path
