"""Tests for cadmia.electronic.density_maps (Fukui / dual / Δρ volumetric maps)."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.core import Structure, VolumetricGrid
from cadmia.electronic.density_maps import deformation_density, fukui_maps


def _grid(value, *, shape=(2, 2, 2), field_type="density"):
    s = Structure(atoms=["O"], positions=np.zeros((1, 3)), labels=["O1"],
                  structure_id="t", source_file="t")
    return VolumetricGrid(
        structure=s, origin=np.zeros(3), vx=np.array([0.2, 0.0, 0.0]),
        vy=np.array([0.0, 0.2, 0.0]), vz=np.array([0.0, 0.0, 0.2]),
        data=np.full(shape, float(value)), field_type=field_type,
    )


def test_fukui_maps_signs_and_labels():
    rho_n = _grid(5.0)
    rho_np1 = _grid(7.0)   # f+ = ρ(N+1) − ρ(N) = 2
    rho_nm1 = _grid(4.0)   # f- = ρ(N) − ρ(N-1) = 1
    m = fukui_maps(rho_n, rho_np1, rho_nm1)
    np.testing.assert_allclose(m["f_plus"].data, 2.0)
    np.testing.assert_allclose(m["f_minus"].data, 1.0)
    np.testing.assert_allclose(m["f0"].data, 1.5)        # (2+1)/2
    np.testing.assert_allclose(m["dual"].data, 1.0)      # 2 − 1
    assert m["f_plus"].field_type == "f_plus"
    assert m["dual"].field_type == "dual"


def test_deformation_density():
    rho_ad = _grid(10.0)
    rho_cl = _grid(6.0)
    rho_pr = _grid(3.0)
    drho = deformation_density(rho_ad, rho_cl, rho_pr)
    np.testing.assert_allclose(drho.data, 1.0)           # 10 − 6 − 3
    assert drho.field_type == "deformation"


def test_maps_require_aligned_grids():
    rho_n = _grid(5.0, shape=(2, 2, 2))
    rho_np1 = _grid(7.0, shape=(2, 2, 3))   # misaligned
    rho_nm1 = _grid(4.0, shape=(2, 2, 2))
    with pytest.raises(ValueError, match="voxel-aligned"):
        fukui_maps(rho_n, rho_np1, rho_nm1)


def test_cube_summary_integral_is_in_bohr3():
    # orca_plot writes e/Bohr³ but grid vectors are stored in Å; the integral must use the voxel
    # volume in Bohr³ to come out in electrons (Å³ would underestimate it by BOHR_TO_ANG³ ≈ 0.148).
    from cadmia.agents.density_agent import cube_summary
    from cadmia.core.units import BOHR_TO_ANG

    grid = _grid(1.0, shape=(2, 2, 2))             # ρ = 1 e/Bohr³, 8 voxels, vx=vy=vz=0.2 Å
    s = cube_summary(grid)
    voxel_ang3 = 0.2 ** 3
    expected = 8 * voxel_ang3 / BOHR_TO_ANG ** 3
    assert s["volume_integral"] == pytest.approx(expected)
    assert s["volume_integral"] > 8 * voxel_ang3   # larger than the naive Å³ integral
