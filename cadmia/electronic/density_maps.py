"""Fukui / dual-descriptor / deformation-density MAPS from electron-density cubes.

The condensed Fukui functions (`cadmia/cdft/calculator.py`) work on per-atom charges; these are
the same descriptors as *volumetric fields*, built by subtracting electron-density cubes of the
N / N+1 / N-1 states (and, for Δρ, of the adduct / cluster / probe). Every input cube must be on
one common, voxel-aligned grid — see `VolumetricGrid.same_grid_as` — which the cube generator
(`orca_plot`) guarantees by emitting all species on a fixed grid.

Sign conventions match the condensed code:
    f+  (electrophilic)  = ρ(N+1) − ρ(N)
    f-  (nucleophilic)   = ρ(N)   − ρ(N-1)
    f0  (radical)        = (f+ + f-) / 2
    Δf  (dual)           = f+ − f-
    Δρ  (deformation)    = ρ(adduct) − ρ(cluster) − ρ(probe)
"""
from __future__ import annotations

from typing import Dict

from cadmia.core import VolumetricGrid


def fukui_maps(rho_n: VolumetricGrid, rho_np1: VolumetricGrid,
               rho_nm1: VolumetricGrid) -> Dict[str, VolumetricGrid]:
    """Return the Fukui f+, f-, f0 and dual Δf fields from N / N+1 / N-1 density cubes."""
    f_plus = rho_np1 - rho_n          # alignment checked by __sub__
    f_minus = rho_n - rho_nm1
    f_zero = (f_plus + f_minus).scaled(0.5)
    dual = f_plus - f_minus
    # field_type matches the dict key / written filename / CLI label so the volumetric vocabulary
    # is single-sourced (these are the grid fields, distinct from the condensed Fukui descriptors).
    f_plus.field_type = "f_plus"
    f_minus.field_type = "f_minus"
    f_zero.field_type = "f0"
    dual.field_type = "dual"
    return {"f_plus": f_plus, "f_minus": f_minus, "f0": f_zero, "dual": dual}


def deformation_density(rho_adduct: VolumetricGrid, rho_cluster: VolumetricGrid,
                        rho_probe: VolumetricGrid) -> VolumetricGrid:
    """Return the deformation (interaction) density Δρ = ρ_adduct − ρ_cluster − ρ_probe.

    The cluster and probe densities must be computed at the adduct geometry on the adduct grid
    (ghost-atom / common-grid convention) so the three cubes are voxel-aligned.
    """
    rho_adduct._require_aligned(rho_cluster)
    rho_adduct._require_aligned(rho_probe)
    data = rho_adduct.data - rho_cluster.data - rho_probe.data
    return rho_adduct._with_data(data, "deformation")
