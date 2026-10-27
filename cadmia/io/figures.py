"""Shared setup of the static, publication-ready figures (matplotlib, an optional dependency).

One style for every static figure CADMIA writes — electronic (DOS, PDOS, IR) and structural
(ECN trends, heatmaps, histograms) — so that a set of figures reads as one. Figures are built
with ``matplotlib.figure.Figure`` inside an ``rc_context``: no pyplot, so neither the backend
nor the global rcParams of the calling session (a Jupyter notebook, say) are touched.
"""
from __future__ import annotations

import functools
from pathlib import Path

FIGURE_FORMATS = ("png", "svg", "pdf")
FIGURE_WIDTHS_IN = {"single": 3.4, "double": 6.5}      # one- and two-column journal widths
_DPI = 300
_STYLE = {"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "legend.title_fontsize": 7,
          "axes.linewidth": 0.8, "lines.linewidth": 1.2,
          "axes.spines.top": False, "axes.spines.right": False,
          "svg.fonttype": "none", "pdf.fonttype": 42}


def resolve_format(fmt: "str | None", output: "str | Path | None", allowed, default: str) -> str:
    """--format if given; else the suffix of -o when it names an allowed format; else default."""
    if fmt:
        return fmt
    suffix = Path(output).suffix.lower().lstrip(".") if output else ""
    return suffix if suffix in allowed else default


def static_figure(func):
    """Run a figure writer inside the CADMIA matplotlib style; RuntimeError if it is missing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import matplotlib
        except ImportError as exc:
            raise RuntimeError("Static figures need matplotlib: pip install 'cadmia[full]'") from exc
        with matplotlib.rc_context(_STYLE):
            return func(*args, **kwargs)
    return wrapper


def new_figure(width: str = "single", aspect: float = 0.62):
    """An empty Figure of the chosen column width and height = aspect × width."""
    from matplotlib.figure import Figure
    w = FIGURE_WIDTHS_IN[width]
    return Figure(figsize=(w, w * aspect), layout="constrained")


def save_figure(fig, path: "str | Path") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    return path
