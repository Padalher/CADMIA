"""MEP (Molecular Electrostatic Potential) analysis from Gaussian cube files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cadmia.core import VolumetricGrid
from cadmia.core.units import HARTREE_TO_EV


@dataclass
class MEPStats:
    """Statistical summary of a molecular electrostatic potential grid."""

    vmin: float
    vmax: float
    vmean: float
    vstd: float
    # Isosurface-restricted stats (only computed when density cube is provided)
    surface_vmin: Optional[float] = None
    surface_vmax: Optional[float] = None
    surface_vmean: Optional[float] = None
    surface_positive_fraction: Optional[float] = None
    surface_negative_fraction: Optional[float] = None
    # Where the surface extremes are (Å) and the atom nearest to each
    surface_vmin_position: Optional[tuple] = None
    surface_vmax_position: Optional[tuple] = None
    surface_vmin_atom: Optional[str] = None
    surface_vmax_atom: Optional[str] = None
    # Derived reactivity indicators
    nucleophilic_site_mep: Optional[float] = None   # most negative MEP on surface
    electrophilic_site_mep: Optional[float] = None  # most positive MEP on surface
    isovalue_used: Optional[float] = None
    # Surface area breakdown (Å²)
    surface_area_ang2: Optional[float] = None
    positive_area_ang2: Optional[float] = None
    negative_area_ang2: Optional[float] = None
    # Politzer-Murray second moments and indices, all area-weighted over the surface
    sigma2_plus: Optional[float] = None    # centred variance over the V > 0 part
    sigma2_minus: Optional[float] = None   # centred variance over the V < 0 part
    sigma2_total: Optional[float] = None   # σ²+ + σ²-
    nu_balance: Optional[float] = None     # σ²+ × σ²- / σ²_tot²
    pi_au: Optional[float] = None          # internal charge separation (a.u.)
    mpi_ev: Optional[float] = None         # molecular polarity index (eV)

    def __str__(self) -> str:
        lines = [
            " MEP Statistics",
            f"   Global min : {self.vmin:>12.6f} a.u.",
            f"   Global max : {self.vmax:>12.6f} a.u.",
            f"   Global mean: {self.vmean:>12.6f} a.u.",
            f"   Global std : {self.vstd:>12.6f} a.u.",
        ]
        if self.surface_vmin is not None:
            lines += [
                f"   --- On density isosurface (rho = {self.isovalue_used:.4f} a.u.) ---",
                f"   Surface min : {self.surface_vmin:>12.6f} a.u.  (most nucleophilic site)"
                + _where(self.surface_vmin_position, self.surface_vmin_atom),
                f"   Surface max : {self.surface_vmax:>12.6f} a.u.  (most electrophilic site)"
                + _where(self.surface_vmax_position, self.surface_vmax_atom),
                f"   Surface mean: {self.surface_vmean:>12.6f} a.u.",
                f"   Positive fraction: {self.surface_positive_fraction*100:>6.2f} %  (electrophilic region)",
                f"   Negative fraction: {self.surface_negative_fraction*100:>6.2f} %  (nucleophilic region)",
            ]
        if self.surface_area_ang2 is not None:
            lines += [
                f"   Surface area:            {self.surface_area_ang2:>10.5f} Angstrom^2",
                f"   Positive surface area:   {self.positive_area_ang2:>10.5f} Angstrom^2",
                f"   Negative surface area:   {self.negative_area_ang2:>10.5f} Angstrom^2",
            ]
        if self.sigma2_total is not None:
            lines += [
                f"   Overall variance σ²_tot: {self.sigma2_total:>12.8f} a.u.^2",
                f"   Positive variance σ²+:   {self.sigma2_plus:>12.8f} a.u.^2",
                f"   Negative variance σ²-:   {self.sigma2_minus:>12.8f} a.u.^2",
                f"   Balance of charges ν:    {self.nu_balance:>12.8f}",
                f"   Internal charge sep. Pi: {self.pi_au:>12.8f} a.u.",
                f"   Molecular polarity MPI:  {self.mpi_ev:>10.5f} eV",
            ]
        return "\n".join(lines)


def _where(position, atom) -> str:
    if position is None:
        return ""
    return f"  at ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) A, next to {atom}"


def _surface_points(rho: np.ndarray, field: np.ndarray, isovalue: float, steps: np.ndarray):
    """Points where a grid edge crosses rho = isovalue: the field there, the surface area each
    point stands for, and its position in grid units.

    A surface patch dA with unit normal n crosses |n_j|·dA/A_j edges along axis j (A_j = area of
    the cell face perpendicular to j), so weighting every crossing by w = 1 / Σ_j |n_j|/A_j makes
    Σ f·w estimate ∫ f dA whatever the orientation. Unweighted crossings over-count the patches
    tilted to the three axes by up to √3, a bias that shows first in σ²_tot.
    """
    inside = rho >= isovalue
    grad = np.gradient(rho, *steps)
    face = np.array([steps[1] * steps[2], steps[0] * steps[2], steps[0] * steps[1]])
    values, normals, positions = [], [], []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        cross = inside[lo] != inside[hi]
        r0 = rho[lo][cross]
        t = (isovalue - r0) / (rho[hi][cross] - r0)       # one end is >= isovalue, the other not
        lerp = lambda f: f[lo][cross] + t * (f[hi][cross] - f[lo][cross])
        values.append(lerp(field))
        normals.append(np.stack([lerp(g) for g in grad], axis=-1))
        index = np.argwhere(cross).astype(float)
        index[:, axis] += t
        positions.append(index)
    values, normals, positions = map(np.concatenate, (values, normals, positions))
    norm = np.linalg.norm(normals, axis=1)
    keep = np.isfinite(values) & (norm > 0)
    unit = np.abs(normals[keep]) / norm[keep, None]
    return values[keep], 1.0 / (unit / face).sum(axis=1), positions[keep]


def analyze_mep(
    mep_grid: VolumetricGrid,
    density_grid: Optional[VolumetricGrid] = None,
    isovalue: float = 0.001,
) -> MEPStats:
    """Compute statistical descriptors of a MEP grid.

    Parameters
    ----------
    mep_grid:
        MEP cube (in atomic units, Hartree/e).
    density_grid:
        Electron density cube used to define the molecular surface.
        When provided, MEP statistics are also computed on the isosurface.
        Must share the same grid dimensions as *mep_grid*.
    isovalue:
        Electron density isovalue (a.u.) defining the molecular surface.
        Typical value: 0.001 a.u. (Bader surface) or 0.002 a.u.

    On the surface every statistic is area-weighted, the way Multiwfn's quantitative surface
    analysis computes them; both agree on oxide clusters.
    """
    mep = mep_grid.data

    # Remove pathological large values (e.g. near nuclei in MEP cubes)
    finite_mask = np.isfinite(mep) & (np.abs(mep) < 1e6)
    mep_clean = mep[finite_mask]

    stats = MEPStats(
        vmin=float(mep_clean.min()),
        vmax=float(mep_clean.max()),
        vmean=float(mep_clean.mean()),
        vstd=float(mep_clean.std()),
    )
    if density_grid is None:
        return stats

    rho = density_grid.data
    if rho.shape != mep.shape:
        raise ValueError(
            f"MEP grid shape {mep.shape} does not match density grid shape {rho.shape}."
        )
    axes = np.array([mep_grid.vx, mep_grid.vy, mep_grid.vz])     # step vectors, already in Å
    steps = np.linalg.norm(axes, axis=1)
    if not np.allclose(axes, np.diag(steps), atol=1e-6 * steps.max()):
        raise ValueError("The MEP surface analysis needs a grid with axes along x, y and z.")

    v, w, index = _surface_points(rho, np.where(finite_mask, mep, np.nan), isovalue, steps)
    if v.size == 0:
        raise ValueError(
            f"The density isosurface rho = {isovalue} does not cross the grid; the cube may "
            "be too small or the isovalue out of range.")

    pos, neg = v > 0, v < 0
    area = w.sum()
    mean = np.average(v, weights=w)
    mean_pos = np.average(v[pos], weights=w[pos]) if pos.any() else 0.0
    mean_neg = np.average(v[neg], weights=w[neg]) if neg.any() else 0.0
    s2p = float(np.average((v[pos] - mean_pos) ** 2, weights=w[pos])) if pos.any() else 0.0
    s2m = float(np.average((v[neg] - mean_neg) ** 2, weights=w[neg])) if neg.any() else 0.0
    s2t = s2p + s2m

    xyz = mep_grid.origin + index * steps
    atoms = np.asarray(mep_grid.structure.positions, dtype=float)
    labels = mep_grid.structure.labels or mep_grid.structure.atoms

    def site(k):
        if not len(atoms):
            return tuple(float(c) for c in xyz[k]), None
        return (tuple(float(c) for c in xyz[k]),
                labels[int(np.argmin(np.linalg.norm(atoms - xyz[k], axis=1)))])

    kmin, kmax = int(np.argmin(v)), int(np.argmax(v))
    stats.isovalue_used = isovalue
    stats.surface_vmin, stats.surface_vmax = float(v[kmin]), float(v[kmax])
    stats.surface_vmin_position, stats.surface_vmin_atom = site(kmin)
    stats.surface_vmax_position, stats.surface_vmax_atom = site(kmax)
    stats.surface_vmean = float(mean)
    stats.surface_positive_fraction = float(w[pos].sum() / area)
    stats.surface_negative_fraction = float(w[neg].sum() / area)
    stats.nucleophilic_site_mep = stats.surface_vmin
    stats.electrophilic_site_mep = stats.surface_vmax
    stats.surface_area_ang2 = float(area)
    stats.positive_area_ang2 = float(w[pos].sum())
    stats.negative_area_ang2 = float(w[neg].sum())
    # Politzer: centred variances, each over its own part; nu = s2+ s2- / s2tot^2;
    # Pi = mean |V - mean(V)| and MPI = mean |V| (eV) are different quantities.
    stats.sigma2_plus, stats.sigma2_minus, stats.sigma2_total = s2p, s2m, s2t
    stats.nu_balance = (s2p * s2m) / s2t ** 2 if s2t > 0 else 0.0
    stats.pi_au = float(np.average(np.abs(v - mean), weights=w))
    stats.mpi_ev = float(np.average(np.abs(v), weights=w)) * HARTREE_TO_EV
    return stats


def write_mep_report(
    stats: MEPStats,
    filepath: str | Path,
) -> None:
    """Write a plain-text MEP analysis report."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(stats) + "\n", encoding="utf-8")
