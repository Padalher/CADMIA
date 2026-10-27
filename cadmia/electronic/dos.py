"""Density of States (DOS) calculation from orbital eigenvalues."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DOSResult:
    """DOS curves for one or two spin channels.

    ``dos_alpha``/``dos_beta`` are per spin channel (one state per orbital); ``dos_total`` is
    states/eV over both spins, so in a closed shell every spatial orbital counts twice.
    """

    energy_ev: np.ndarray
    dos_alpha: np.ndarray
    dos_beta: Optional[np.ndarray] = None
    n_alpha_occ: int = 0
    n_beta_occ: int = 0
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None

    @property
    def is_spin_polarized(self) -> bool:
        return self.dos_beta is not None

    @property
    def dos_total(self) -> np.ndarray:
        if self.dos_beta is not None:
            return self.dos_alpha + self.dos_beta
        return 2.0 * self.dos_alpha

    @property
    def gap_ev(self) -> Optional[float]:
        if self.homo_ev is not None and self.lumo_ev is not None:
            return self.lumo_ev - self.homo_ev
        return None


_GRID_STEPS_PER_SIGMA = 4   # coarser than sigma/4 and the peaks are aliased, not broadened
_TAIL_SIGMAS = 6


def _energy_grid(e_min: float, e_max: float, n_points: int, sigma: float) -> np.ndarray:
    """n_points is a minimum: the core orbitals of a heavy atom stretch the range to thousands
    of eV, and a fixed count would leave the valence peaks with a handful of points each."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    needed = int(np.ceil((e_max - e_min) * _GRID_STEPS_PER_SIGMA / sigma)) + 1
    return np.linspace(e_min, e_max, max(n_points, needed))


def _broaden(eigenvalues_ev, weights, energy_grid: np.ndarray, sigma: float) -> np.ndarray:
    """Sum of normalized Gaussians, each evaluated only within ±6 sigma of its eigenvalue."""
    dos = np.zeros_like(energy_grid)
    for e, w in zip(eigenvalues_ev, weights):
        lo, hi = np.searchsorted(energy_grid, (e - _TAIL_SIGMAS * sigma, e + _TAIL_SIGMAS * sigma))
        dos[lo:hi] += w * np.exp(-0.5 * ((energy_grid[lo:hi] - e) / sigma) ** 2)
    return dos / (sigma * np.sqrt(2 * np.pi))


def _gaussian_dos(
    eigenvalues_ev: List[float],
    energy_grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    return _broaden(eigenvalues_ev, [1.0] * len(eigenvalues_ev), energy_grid, sigma)


def compute_dos(
    alpha_eigenvalues_ev: List[float],
    beta_eigenvalues_ev: Optional[List[float]] = None,
    n_alpha_occ: int = 0,
    n_beta_occ: int = 0,
    homo_ev: Optional[float] = None,
    lumo_ev: Optional[float] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    n_points: int = 2000,
    sigma: float = 0.1,
    padding_ev: float = 2.0,
) -> DOSResult:
    """Compute DOS via Gaussian broadening of orbital eigenvalues.

    Parameters
    ----------
    alpha_eigenvalues_ev:
        All alpha orbital eigenvalues (occupied + virtual) in eV.
    beta_eigenvalues_ev:
        Beta orbital eigenvalues for open-shell systems. None for closed-shell.
    n_alpha_occ / n_beta_occ:
        Number of occupied alpha/beta orbitals (informational).
    homo_ev / lumo_ev:
        HOMO and LUMO energies in eV.
    energy_range:
        ``(e_min, e_max)`` for the energy grid in eV.  Defaults to
        ``[min(eigenvalues) - padding_ev, max(eigenvalues) + padding_ev]``.
    n_points:
        Minimum number of grid points; more are used when needed for a step of sigma/4.
    sigma:
        Gaussian broadening width (eV). Typical: 0.05–0.3 eV.
    padding_ev:
        Extra energy range beyond the outermost eigenvalue; never less than the 6-sigma tail
        of the Gaussian, so a wide broadening is not cut off at the edge of the grid.
    """
    all_eigs = list(alpha_eigenvalues_ev)
    if beta_eigenvalues_ev:
        all_eigs += list(beta_eigenvalues_ev)

    if energy_range is None:
        padding_ev = max(padding_ev, _TAIL_SIGMAS * sigma)
        e_min = min(all_eigs) - padding_ev
        e_max = max(all_eigs) + padding_ev
    else:
        e_min, e_max = energy_range

    grid = _energy_grid(e_min, e_max, n_points, sigma)
    dos_alpha = _gaussian_dos(alpha_eigenvalues_ev, grid, sigma)
    dos_beta = _gaussian_dos(beta_eigenvalues_ev, grid, sigma) if beta_eigenvalues_ev else None

    return DOSResult(
        energy_ev=grid,
        dos_alpha=dos_alpha,
        dos_beta=dos_beta,
        n_alpha_occ=n_alpha_occ,
        n_beta_occ=n_beta_occ,
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
    )


def dos_from_output(
    filepath: str | Path,
    *,
    sigma: float = 0.1,
    n_points: int = 2000,
    energy_range: Optional[Tuple[float, float]] = None,
) -> DOSResult:
    """Compute the total DOS directly from a Gaussian or ORCA output file.

    Reads all orbital eigenvalues (occupied + virtual) from the output and broadens
    them. Raises ValueError when the output carries no eigenvalue list.
    """
    from cadmia.io.electronic import parse_gaussian_electronic, parse_orca_electronic
    from cadmia.io.gaussian import parse_gaussian_geometry
    from cadmia.io.orca import parse_orca_geometry

    p = Path(filepath)
    try:
        structure = parse_gaussian_geometry(p)
        state = parse_gaussian_electronic(p, structure)
    except ValueError:
        structure = parse_orca_geometry(p)
        state = parse_orca_electronic(p, structure)

    alpha = state.alpha_eigenvalues_ev
    if not alpha:
        raise ValueError(
            f"No orbital eigenvalues found in {p.name} — DOS needs an ORBITAL ENERGIES / "
            "eigenvalue block in the output."
        )
    beta = state.beta_eigenvalues_ev
    homo = state.homo_ev

    # Counting occupation by "eigenvalue below the HOMO" is only valid within one spin channel.
    # In an open-shell system the HOMO is max(alpha, beta), which is usually the alpha one, so
    # counting beta levels below it sweeps in beta VIRTUAL orbitals that happen to lie lower and
    # reports more beta electrons than the system has.
    #
    # The occupation counts are known exactly where the eigenvalues are read, so they are taken
    # from there. The threshold below is only the fallback for a reader that does not supply them.
    na = nb = None
    try:
        from cadmia.io.electronic import _parse_orca_orbital_energies
        with open(p, "r", encoding="utf-8", errors="replace") as handle:
            _, _, na, nb = _parse_orca_orbital_energies(handle.read().splitlines())
    except Exception:
        na = nb = None
    if not na:
        na = sum(1 for e in alpha if homo is not None and e <= homo + 1e-6)
        nb = sum(1 for e in (beta or []) if homo is not None and e <= homo + 1e-6)
    return compute_dos(
        alpha, beta, n_alpha_occ=na, n_beta_occ=nb,
        homo_ev=homo, lumo_ev=state.lumo_ev,
        energy_range=energy_range, n_points=n_points, sigma=sigma,
    )


def _row(values, sep: str) -> str:
    """Energy to 1e-4 eV, curves to 6 significant digits: the empty stretches of the grid
    (zeros) are one character wide instead of eight."""
    return sep.join([f"{values[0]:.4f}"] + [f"{v:.6g}" for v in values[1:]])


def write_dos(
    dos: DOSResult,
    filepath: str | Path,
    fmt: str = "dat",
) -> None:
    """Write DOS data to a file.

    Parameters
    ----------
    dos:
        Output of :func:`compute_dos`.
    filepath:
        Destination path.
    fmt:
        ``"dat"`` (space-delimited, default) or ``"csv"``.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    sep = "," if fmt == "csv" else "  "
    if dos.is_spin_polarized:
        header = f"Energy(eV){sep}DOS_alpha{sep}DOS_beta{sep}DOS_total"
        rows = np.column_stack([dos.energy_ev, dos.dos_alpha, dos.dos_beta, dos.dos_total])
    else:
        header = f"Energy(eV){sep}DOS"
        rows = np.column_stack([dos.energy_ev, dos.dos_total])

    lines = ["# CADMIA DOS output"]
    if dos.homo_ev is not None:
        lines.append(f"# HOMO = {dos.homo_ev:.4f} eV")
    if dos.lumo_ev is not None:
        lines.append(f"# LUMO = {dos.lumo_ev:.4f} eV")
    if dos.gap_ev is not None:
        lines.append(f"# Gap  = {dos.gap_ev:.4f} eV")
    lines.append("# DOS in states/eV, both spins")
    lines.append(f"# grid step = {dos.energy_ev[1] - dos.energy_ev[0]:.4f} eV; the Gaussian broadening is set at call time")
    lines.append(header)
    for row in rows:
        lines.append(_row(row, sep))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────
# PDOS (projected DOS by element, from ORCA per-MO populations)
# ─────────────────────────────────────────────

@dataclass
class PDOSResult:
    """Element-projected DOS in states/eV, both spins (a closed-shell orbital counts twice)."""

    energy_ev: np.ndarray
    series: "dict[str, np.ndarray]"        # element -> projected DOS
    total: np.ndarray
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    spin_polarized: bool = False

    @property
    def gap_ev(self) -> Optional[float]:
        if self.homo_ev is not None and self.lumo_ev is not None:
            return self.lumo_ev - self.homo_ev
        return None


def _weighted_dos(mos: List[dict], weights: List[float],
                  energy_grid: np.ndarray, sigma: float) -> np.ndarray:
    return _broaden([mo["energy_ev"] for mo in mos], weights, energy_grid, sigma)


def pdos_from_output(
    filepath: str | Path,
    *,
    sigma: float = 0.1,
    n_points: int = 2000,
    energy_range: Optional[Tuple[float, float]] = None,
) -> PDOSResult:
    """Element-projected DOS from an ORCA output with per-MO Loewdin populations.

    Requires the run to include ``%output Print[P_OrbPopMO_L] 1 end``; raises
    ValueError otherwise. Element weights come from the printed percentages
    (entries below ORCA's 0.1% threshold are missing, so projections are a very
    close lower bound of the exact Loewdin decomposition).
    """
    from cadmia.io.electronic import parse_orca_mo_populations

    pops = parse_orca_mo_populations(filepath)
    alpha, beta = pops["alpha"], pops["beta"]
    if not alpha:
        raise ValueError(
            f"No 'LOEWDIN ORBITAL POPULATIONS PER MO' block in {Path(filepath).name} — "
            "re-run ORCA with: %output Print[P_OrbPopMO_L] 1 end"
        )

    mos_all = alpha + (beta or [])
    energies = [mo["energy_ev"] for mo in mos_all]
    occupied = [mo["energy_ev"] for mo in mos_all if mo["occ"] > 0.5]
    virtual = [mo["energy_ev"] for mo in mos_all if mo["occ"] <= 0.5]
    homo = max(occupied) if occupied else None
    lumo = min(virtual) if virtual else None

    if energy_range is None:
        energy_range = (min(energies) - _TAIL_SIGMAS * sigma, max(energies) + _TAIL_SIGMAS * sigma)
    grid = _energy_grid(energy_range[0], energy_range[1], n_points, sigma)

    spin_factor = 1.0 if beta else 2.0
    elements = sorted({elem for mo in mos_all
                       for (_, elem, _) in mo["contributions"]})
    series = {}
    negative_elems = []
    for elem in elements:
        weights = [
            spin_factor * sum(pct for (ai, el, orb), pct in mo["contributions"].items()
                              if el == elem) / 100.0
            for mo in mos_all
        ]
        if any(w < 0 for w in weights):
            negative_elems.append(elem)
        series[elem] = _weighted_dos(mos_all, weights, grid, sigma)
    if negative_elems:
        warnings.warn(
            f"Negative Loewdin populations in the PDOS of {', '.join(negative_elems)}: "
            "the Loewdin projection can go slightly negative, so that element's curve may "
            "dip below zero — this is a known artifact, not a bug.",
            stacklevel=2,
        )

    total = _weighted_dos(mos_all, [spin_factor] * len(mos_all), grid, sigma)
    return PDOSResult(energy_ev=grid, series=series, total=total,
                      homo_ev=homo, lumo_ev=lumo, spin_polarized=beta is not None)


def write_pdos(pdos: PDOSResult, filepath: str | Path, fmt: str = "dat") -> None:
    """Write the PDOS curves (one column per element + total) to .dat or .csv."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if fmt == "csv" else "  "
    elements = sorted(pdos.series)
    header = sep.join(["Energy(eV)"] + [f"PDOS_{e}" for e in elements] + ["DOS_total"])
    rows = np.column_stack([pdos.energy_ev] + [pdos.series[e] for e in elements] + [pdos.total])
    lines = ["# CADMIA PDOS output (Loewdin per-MO projections, states/eV, both spins)"]
    if pdos.homo_ev is not None:
        lines.append(f"# HOMO = {pdos.homo_ev:.4f} eV")
    if pdos.lumo_ev is not None:
        lines.append(f"# LUMO = {pdos.lumo_ev:.4f} eV")
    if pdos.gap_ev is not None:
        lines.append(f"# Gap  = {pdos.gap_ev:.4f} eV")
    lines.append(header)
    for row in rows:
        lines.append(_row(row, sep))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
