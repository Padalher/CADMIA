"""Density Agent — generate density/HOMO/LUMO/MEP cubes and derive Fukui/dual/Δρ maps.

Drives ``orca_plot`` (`cadmia/calculators/orca_plot_runner.py`) to emit Gaussian cubes from a
.gbw, then uses the pure grid arithmetic in `cadmia/electronic/density_maps.py` to build the
Fukui f±/f0, dual-descriptor Δf and deformation-density Δρ fields, written with
`cadmia/io/cube.write_cube`. Cube generation needs ORCA (workstation); the map arithmetic is
engine-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cadmia.calculators.orca_plot_runner import (
    ngrid_for_step,
    probe_boundaries,
    run_orca_plot,
)
from cadmia.core import VolumetricGrid
from cadmia.core.units import BOHR_TO_ANG
from cadmia.electronic.density_maps import deformation_density, fukui_maps
from cadmia.io.cube import parse_cube, write_cube


def cube_summary(grid: VolumetricGrid) -> Dict[str, float]:
    """min/max and the volume integral of a field, integrated in Bohr³.

    orca_plot writes its cubes in atomic units (densities in e/Bohr³) and parse_cube converts the
    grid vectors Bohr→Å, so the voxel volume is in Å³; we divide by BOHR_TO_ANG³ to integrate in
    Bohr³. For a density field this makes ``volume_integral`` ≈ electrons (∫ρ ≈ N, ∫f± ≈ 1,
    ∫Δf ≈ 0); for an orbital or MEP cube it is just the raw volume integral, not a physical count.
    (Assumes orca_plot/CADMIA cubes, i.e. atomic-unit data on a Bohr grid.)
    """
    voxel_ang3 = abs(float(np.linalg.det(np.array([grid.vx, grid.vy, grid.vz]))))
    voxel_bohr3 = voxel_ang3 / BOHR_TO_ANG ** 3
    total = float(grid.data.sum())
    return {"field_type": grid.field_type, "shape": tuple(grid.shape),
            "min": grid.vmin, "max": grid.vmax,
            "volume_integral": total * voxel_bohr3}


def resolve_grid(gbw: "str | Path", step: float, orca_plot_path: str) -> Tuple[int, int, int]:
    """Per-axis interval count for a uniform ``step`` (Bohr), from the box orca_plot would use.

    Derived once from a reference .gbw and reused across the species of a Fukui/Δρ set, so all
    cubes share the exact same grid and stay voxel-aligned for the subtraction.
    """
    return ngrid_for_step(probe_boundaries(gbw, orca_plot_path=orca_plot_path), step)


def generate_species_cubes(
    gbw: "str | Path", kinds: List[str], out_dir: "str | Path", *,
    step: float = 0.20, grid: "Tuple[int, int, int] | None" = None,
    homo_index: Optional[int] = None, orca_plot_path: str = "orca_plot",
) -> Dict[str, Path]:
    """Generate the requested cubes (density/homo/lumo/mep) for one species via orca_plot.

    The grid is sized for a uniform ``step`` (Bohr); pass a pre-resolved ``grid`` to force the same
    resolution across a multi-species set (Fukui/Δρ).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if grid is None:
        grid = resolve_grid(gbw, step, orca_plot_path)
    stem = Path(gbw).stem
    written: Dict[str, Path] = {}
    for kind in kinds:
        mo_index = None
        if kind == "homo":
            mo_index = homo_index
        elif kind == "lumo":
            mo_index = None if homo_index is None else homo_index + 1
        dst = out / f"{stem}_{kind}.cube"
        run_orca_plot(gbw, kind, dst, ngrid=grid, mo_index=mo_index,
                      orca_plot_path=orca_plot_path)
        written[kind] = dst
    return written


def fukui_maps_from_gbw(
    n_gbw: "str | Path", np1_gbw: "str | Path", nm1_gbw: "str | Path",
    out_dir: "str | Path", *, step: float = 0.20, orca_plot_path: str = "orca_plot",
) -> Dict[str, Path]:
    """Density cubes of N/N+1/N-1 → f+, f-, f0 and dual Δf maps (written to out_dir).

    N/N+1/N-1 share the geometry (only the charge differs); the per-axis grid is sized once from
    the N state for a uniform ``step`` and reused, so the density cubes are voxel-aligned and the
    differences are valid.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    grid = resolve_grid(n_gbw, step, orca_plot_path)
    rho_n = parse_cube(generate_species_cubes(n_gbw, ["density"], out, grid=grid,
                                              orca_plot_path=orca_plot_path)["density"], "density")
    rho_np1 = parse_cube(generate_species_cubes(np1_gbw, ["density"], out, grid=grid,
                                                orca_plot_path=orca_plot_path)["density"], "density")
    rho_nm1 = parse_cube(generate_species_cubes(nm1_gbw, ["density"], out, grid=grid,
                                                orca_plot_path=orca_plot_path)["density"], "density")
    maps = fukui_maps(rho_n, rho_np1, rho_nm1)
    written: Dict[str, Path] = {}
    for name, grid in maps.items():
        dst = out / f"{name}.cube"
        write_cube(grid, dst, comment=grid.field_type)
        written[name] = dst
    return written


def deformation_from_gbw(
    adduct_gbw: "str | Path", cluster_gbw: "str | Path", probe_gbw: "str | Path",
    out_dir: "str | Path", *, step: float = 0.20, orca_plot_path: str = "orca_plot",
) -> Path:
    """Density cubes of adduct/cluster/probe → deformation density Δρ (written to out_dir).

    The cluster and probe densities must come from single points at the adduct geometry; the
    per-axis grid is sized once from the adduct for a uniform ``step`` and reused, so the three
    cubes are voxel-aligned. `deformation_density` raises if they are not.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    grid = resolve_grid(adduct_gbw, step, orca_plot_path)
    rho_ad = parse_cube(generate_species_cubes(adduct_gbw, ["density"], out, grid=grid,
                                               orca_plot_path=orca_plot_path)["density"], "density")
    rho_cl = parse_cube(generate_species_cubes(cluster_gbw, ["density"], out, grid=grid,
                                               orca_plot_path=orca_plot_path)["density"], "density")
    rho_pr = parse_cube(generate_species_cubes(probe_gbw, ["density"], out, grid=grid,
                                               orca_plot_path=orca_plot_path)["density"], "density")
    drho = deformation_density(rho_ad, rho_cl, rho_pr)
    dst = out / "deformation.cube"
    write_cube(drho, dst, comment="deformation")
    return dst
