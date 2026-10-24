"""IR spectrum from ORCA normal-mode IR intensities via line broadening.

Each IR-active mode contributes a peak at its (optionally scaled) wavenumber with
area proportional to its km/mol intensity, broadened by a Lorentzian (default, the
physical lineshape) or Gaussian profile. The result is a continuous curve suitable
for plotting against an experimental spectrum.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from cadmia.io.vibrational import VibrationalAnalysis, parse_orca_vibrational


@dataclass
class IRSpectrum:
    wavenumber_cm1: np.ndarray   # x grid (cm⁻¹)
    intensity: np.ndarray        # broadened curve (arb. units)
    peak_positions: np.ndarray   # scaled wavenumbers of the IR-active modes
    peak_intensities: np.ndarray  # km/mol of those modes
    scaling_factor: float
    fwhm_cm1: float
    lineshape: str
    # Imaginary modes carry no IR intensity, so they drop out of the broadening on their own and
    # the spectrum of a saddle point comes out looking like a minimum's. Counted here so the
    # caller can say what the curve does not show.
    n_imaginary: int = 0


# Both profiles integrate to 1, so a band's AREA is its km/mol intensity and its height is
# whatever the width makes it. Normalising to height 1 instead — as this did — means two spectra
# computed with different --fwhm cannot be compared on intensity at all, and the area stops being
# the quantity the experiment measures.
def _lorentzian(grid: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return (gamma / np.pi) / ((grid - center) ** 2 + gamma ** 2)


def _gaussian(grid: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((grid - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def compute_ir_spectrum(
    analysis: VibrationalAnalysis,
    *,
    scale: float = 1.0,
    fwhm: float = 16.0,
    lineshape: str = "lorentzian",
    n_points: int = 4000,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> IRSpectrum:
    """Broaden the IR-active modes into a continuous spectrum.

    Parameters
    ----------
    scale : empirical frequency scaling factor applied to every wavenumber
        (e.g. ~0.96 for hybrid DFT; 1.0 = unscaled).
    fwhm : full width at half maximum of each peak (cm⁻¹).
    lineshape : ``"lorentzian"`` (default) or ``"gaussian"``.
    """
    if fwhm <= 0:
        # gamma = 0 makes the Lorentzian 0/0 at the peak and the Gaussian divide by zero:
        # a whole spectrum of NaN, with nothing to say where it came from.
        raise ValueError(f"fwhm must be positive, got {fwhm}.")

    active = analysis.ir_active_modes
    n_imaginary = len(getattr(analysis, "imaginary_modes", ()) or ())
    positions = np.array([m.frequency_cm1 * scale for m in active], dtype=float)
    weights = np.array([m.ir_intensity_km_mol for m in active], dtype=float)

    if positions.size:
        lo = x_min if x_min is not None else max(0.0, positions.min() - 200.0)
        hi = x_max if x_max is not None else positions.max() + 200.0
    else:
        lo = x_min if x_min is not None else 0.0
        hi = x_max if x_max is not None else 4000.0
    grid = np.linspace(lo, hi, n_points)

    profile = _gaussian if lineshape == "gaussian" else _lorentzian
    curve = np.zeros_like(grid)
    for center, weight in zip(positions, weights):
        curve += weight * profile(grid, center, fwhm)

    return IRSpectrum(
        wavenumber_cm1=grid,
        intensity=curve,
        peak_positions=positions,
        peak_intensities=weights,
        scaling_factor=scale,
        fwhm_cm1=fwhm,
        lineshape=lineshape,
        n_imaginary=n_imaginary,
    )


def ir_spectrum_from_output(
    filepath: str | Path,
    *,
    scale: float = 1.0,
    fwhm: float = 16.0,
    lineshape: str = "lorentzian",
    n_points: int = 4000,
) -> IRSpectrum:
    analysis = parse_orca_vibrational(filepath)
    return compute_ir_spectrum(
        analysis, scale=scale, fwhm=fwhm, lineshape=lineshape, n_points=n_points,
    )


def _peak_table(spectrum: IRSpectrum) -> List[dict]:
    return [
        {"wavenumber_cm1": float(p), "ir_intensity_km_mol": float(w)}
        for p, w in sorted(
            zip(spectrum.peak_positions, spectrum.peak_intensities),
            key=lambda t: -t[1],
        )
    ]


def write_ir_spectrum(spectrum: IRSpectrum, filepath: str | Path, fmt: str = "dat") -> None:
    """Write the broadened curve as .dat/.csv (two columns) or the peaks as .json."""
    path = Path(filepath)
    if fmt == "json":
        import json
        payload = {
            "scaling_factor": spectrum.scaling_factor,
            "fwhm_cm1": spectrum.fwhm_cm1,
            "lineshape": spectrum.lineshape,
            "peaks": _peak_table(spectrum),
            "wavenumber_cm1": spectrum.wavenumber_cm1.tolist(),
            "intensity": spectrum.intensity.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    sep = "," if fmt == "csv" else "  "
    header = "wavenumber_cm-1" + sep + "intensity"
    rows = [header] + [
        f"{x:.4f}{sep}{y:.6e}"
        for x, y in zip(spectrum.wavenumber_cm1, spectrum.intensity)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
