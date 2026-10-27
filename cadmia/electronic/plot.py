"""Interactive HTML plots for DOS/PDOS curves.

Self-contained generator: the page embeds the curves as JSON and loads plotly.js
from its CDN, so no Python plotting dependency is required (same exportable-HTML
philosophy as the `cadmia structure` plots, without importing plotly).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from cadmia.electronic.dos import DOSResult, PDOSResult

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

ELEMENT_COLORS = {
    "Ti": "#8c8c9e", "Al": "#c5c5c5", "O": "#d62728",
    "H": "#7f7f7f", "C": "#2c2c2c", "N": "#1f77b4",
}
FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{cdn}" charset="utf-8"></script>
</head>
<body>
<div id="plot" style="width:100%;max-width:1100px;height:620px;margin:0 auto;"></div>
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
    # HOMO's label sits left of its line and LUMO's right of its own, so the two never meet however
    # small the gap; the gap gets its own bracket lower down, between the lines.
    shapes, annotations = [], []
    for x, label, side in ((homo_ev, "HOMO", "right"), (lumo_ev, "LUMO", "left")):
        if x is None:
            continue
        shapes.append({
            "type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1,
            "line": {"color": "#444", "width": 1, "dash": "dash"},
        })
        annotations.append({
            "x": x, "yref": "paper", "y": 0.99, "yanchor": "top", "xanchor": side,
            "xshift": -6 if side == "right" else 6, "showarrow": False,
            "text": f"<b>{label}</b><br>{x:.2f} eV", "font": {"size": 17}, "align": side,
        })
    if gap_ev is not None and homo_ev is not None and lumo_ev is not None:
        shapes.append({
            "type": "line", "x0": homo_ev, "x1": lumo_ev, "yref": "paper", "y0": 0.80, "y1": 0.80,
            "line": {"color": "#444", "width": 1.5},
        })
        annotations.append({
            "x": (homo_ev + lumo_ev) / 2.0, "yref": "paper", "y": 0.81, "yanchor": "bottom",
            "showarrow": False, "text": f"<b>gap</b> {gap_ev:.2f} eV", "font": {"size": 17},
        })
    return shapes, annotations


# Readable on a full-width browser window (plotly's 12 px default is lost on a large plot).
_FONT_PX = 17

# Opening view around the frontier orbitals; the core levels stay in the data, one zoom-out away.
VIEW_BELOW_HOMO_EV, VIEW_ABOVE_LUMO_EV = 20.0, 10.0


def _layout(title: str, ylabel: str, shapes: list, annotations: list,
            homo: Optional[float] = None, lumo: Optional[float] = None,
            energy: Optional[list] = None) -> dict:
    # no zero line on x: a solid line at 0 eV reads as a Fermi level, and there is none
    xaxis = {"title": {"text": "Energy (eV)"}, "zeroline": False}
    if homo is not None and lumo is not None:
        lo, hi = homo - VIEW_BELOW_HOMO_EV, lumo + VIEW_ABOVE_LUMO_EV
        if energy:      # never past the ends of the curve
            lo, hi = max(lo, energy[0]), min(hi, energy[-1])
        xaxis["range"] = [lo, hi]
    return {
        "title": {"text": title},
        "xaxis": xaxis,
        "yaxis": {"title": {"text": ylabel}, "rangemode": "tozero"},
        "shapes": shapes,
        "annotations": annotations,
        "template": "plotly_white",
        "font": {"size": _FONT_PX},
        "legend": {"orientation": "h", "y": -0.15},
    }


def _render(title: str, traces: list, layout: dict) -> str:
    return _PAGE.format(
        title=title, cdn=_PLOTLY_CDN,
        data=json.dumps(traces), layout=json.dumps(layout),
    )


def _compact(energy: np.ndarray, *curves: np.ndarray) -> tuple[list, list]:
    """Energies and curves as plain lists for the page. The core levels of a heavy atom leave
    thousands of eV where every curve is exactly zero: those stretches keep only their end
    points (a straight line between them is the same zero), and values are rounded as in the
    .dat files, which together cut the page from megabytes to kilobytes."""
    keep = np.any([c != 0 for c in curves], axis=0)
    keep = keep | np.roll(keep, 1) | np.roll(keep, -1)
    keep[[0, -1]] = True
    return np.round(energy[keep], 4).tolist(), [np.round(c[keep], 6).tolist() for c in curves]


def dos_html(dos: DOSResult, *, title: str = "Density of states") -> str:
    """Build a self-contained interactive HTML page for a total-DOS curve."""
    curves = [dos.dos_total] + ([dos.dos_alpha, dos.dos_beta] if dos.is_spin_polarized else [])
    energy, (total, *spins) = _compact(dos.energy_ev, *curves)
    traces = [{
        "x": energy, "y": total, "mode": "lines",
        "name": "DOS total", "line": {"color": "#1f77b4", "width": 2},
    }]
    for name, y, color in zip(("alpha", "beta"), spins, ("#2ca02c", "#d62728")):
        traces.append({
            "x": energy, "y": y, "mode": "lines",
            "name": name, "line": {"color": color, "width": 1, "dash": "dot"},
        })
    shapes, annotations = _frontier_decorations(dos.homo_ev, dos.lumo_ev, dos.gap_ev)
    return _render(title, traces,
                   _layout(title, "DOS (states/eV)", shapes, annotations, dos.homo_ev, dos.lumo_ev,
                           energy))


def pdos_html(pdos: PDOSResult, *, title: str = "Projected density of states") -> str:
    """Build a self-contained interactive HTML page for element-projected DOS."""
    elements = sorted(pdos.series)
    energy, (total, *projected) = _compact(pdos.energy_ev, pdos.total,
                                           *(pdos.series[e] for e in elements))
    traces = [{
        "x": energy, "y": total, "mode": "lines",
        "name": "DOS total", "line": {"color": "#333333", "width": 1, "dash": "dash"},
    }]
    fallback = iter(FALLBACK_COLORS * 4)
    for element, y in zip(elements, projected):
        color = ELEMENT_COLORS.get(element) or next(fallback)
        traces.append({
            "x": energy, "y": y, "mode": "lines",
            "name": element, "line": {"color": color, "width": 2},
        })
    shapes, annotations = _frontier_decorations(pdos.homo_ev, pdos.lumo_ev, pdos.gap_ev)
    return _render(title, traces,
                   _layout(title, "PDOS (states/eV)", shapes, annotations, pdos.homo_ev, pdos.lumo_ev,
                           energy))


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
        "font": {"size": _FONT_PX},
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
