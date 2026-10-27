"""Static, publication-ready figures of DOS, PDOS and IR spectra (matplotlib).

The interactive HTML pages (`plot.py`) are for exploring; these are for a paper or a poster:
vector (SVG, PDF) or 300-dpi PNG, one journal column wide (3.4 in) or two (6.5 in), no chart
junk. matplotlib is an optional dependency (``pip install "cadmia[full]"``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from cadmia.electronic.dos import DOSResult, PDOSResult
from cadmia.electronic.plot import ELEMENT_COLORS, FALLBACK_COLORS
from cadmia.electronic.plot import VIEW_ABOVE_LUMO_EV, VIEW_BELOW_HOMO_EV
from cadmia.io.figures import new_figure, save_figure, static_figure


def _frontier(ax, homo: Optional[float], lumo: Optional[float]) -> None:
    for e, label in ((homo, "HOMO"), (lumo, "LUMO")):
        if e is not None:
            ax.axvline(e, color="#555555", linestyle="--", linewidth=0.8)
            ax.annotate(label, (e, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -10),
                        textcoords="offset points", fontsize=7, color="#555555")
    if homo is not None and lumo is not None:
        xs = [line.get_xdata() for line in ax.get_lines() if len(line.get_xdata()) > 2]
        lo = max(homo - VIEW_BELOW_HOMO_EV, min(float(x.min()) for x in xs))
        hi = min(lumo + VIEW_ABOVE_LUMO_EV, max(float(x.max()) for x in xs))
        ax.set_xlim(lo, hi)
        # the y range of what is on view, not of the core peaks left outside it
        ys = [y[(x >= lo) & (x <= hi)] for line in ax.get_lines()
              for x, y in [line.get_data()] if len(x) > 2]
        top = max((float(abs(v).max()) for v in ys if v.size), default=0.0)
        if top > 0:
            bottom = -top * 1.08 if any(float(v.min()) < 0 for v in ys if v.size) else 0
            ax.set_ylim(bottom, top * 1.08)


@static_figure
def write_dos_figure(dos: DOSResult, filepath: "str | Path", *, title: str = "",
                     width: str = "single") -> Path:
    fig = new_figure(width)
    ax = fig.subplots()
    if dos.is_spin_polarized:
        ax.plot(dos.energy_ev, dos.dos_alpha, color="#1f77b4", label="α")
        ax.plot(dos.energy_ev, -dos.dos_beta, color="#d62728", label="β")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(frameon=False)
    else:
        ax.plot(dos.energy_ev, dos.dos_total, color="#1f77b4")
        ax.set_ylim(bottom=0)
    _frontier(ax, dos.homo_ev, dos.lumo_ev)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (states/eV)")
    if title:
        ax.set_title(title)
    return save_figure(fig, filepath)


@static_figure
def write_pdos_figure(pdos: PDOSResult, filepath: "str | Path", *, title: str = "",
                      width: str = "single") -> Path:
    fig = new_figure(width)
    ax = fig.subplots()
    ax.plot(pdos.energy_ev, pdos.total, color="#333333", linestyle="--", linewidth=1.0,
            label="total")
    fallback = iter(FALLBACK_COLORS * 4)
    for element in sorted(pdos.series):
        ax.plot(pdos.energy_ev, pdos.series[element], label=element,
                color=ELEMENT_COLORS.get(element) or next(fallback))
    ax.set_ylim(bottom=0)
    _frontier(ax, pdos.homo_ev, pdos.lumo_ev)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("PDOS (states/eV)")
    ax.legend(frameon=False, ncols=min(5, len(pdos.series) + 1))
    if title:
        ax.set_title(title)
    return save_figure(fig, filepath)


@static_figure
def write_ir_figure(spectrum, filepath: "str | Path", *, title: str = "",
                    width: str = "single") -> Path:
    """Broadened spectrum (left axis, arbitrary units) with each IR-active mode as a stick on
    its own right axis in km/mol, high to low wavenumber."""
    fig = new_figure(width)
    ax = fig.subplots()
    ax.plot(spectrum.wavenumber_cm1, spectrum.intensity, color="#1f77b4")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)")
    sticks = ax.twinx()
    sticks.vlines(spectrum.peak_positions, 0, spectrum.peak_intensities,
                  color="#d62728", linewidth=0.8)
    sticks.set_ylim(bottom=0)
    sticks.set_ylabel("IR intensity (km/mol)")
    sticks.spines["right"].set_visible(True)
    ax.invert_xaxis()
    scale = "" if spectrum.scaling_factor == 1.0 else f"  (scaled ×{spectrum.scaling_factor:g})"
    if title or scale:
        ax.set_title(title + scale)
    return save_figure(fig, filepath)
