"""Tests for the MEP analysis module and its `cadmia electronic mep` CLI command."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from cadmia.cli import main
from cadmia.core import Structure, VolumetricGrid
from cadmia.core.units import HARTREE_TO_EV
from cadmia.electronic.mep import MEPStats, analyze_mep, write_mep_report
from cadmia.io.cube import parse_cube

_CUBE = Path(__file__).resolve().parents[1] / "fixtures" / "test_small.cube"


# ─────────────────────────────────────────────
# Politzer-Murray statistics on the isosurface
# ─────────────────────────────────────────────

def _grid(values, field_type, step=0.5, origin=(0.0, 0.0, 0.0), atoms=None):
    """A cubic grid of *step* Å per side."""
    atoms = atoms or [("H", (0.0, 0.0, 0.0))]
    s = Structure(atoms=[a for a, _ in atoms], positions=np.array([p for _, p in atoms]),
                  labels=[f"{a}{k}" for k, (a, _) in enumerate(atoms, 1)],
                  structure_id="t", source_file="")
    return VolumetricGrid(structure=s, origin=np.asarray(origin, dtype=float),
                          vx=np.array([step, 0.0, 0.0]),
                          vy=np.array([0.0, step, 0.0]),
                          vz=np.array([0.0, 0.0, step]),
                          data=np.asarray(values, dtype=float), field_type=field_type)


_R, _STEP, _ISO = 2.0, 0.05, 0.001


@pytest.fixture(scope="module")
def sphere():
    """rho = iso·exp(R − r): its isosurface is the sphere |r| = R. MEP V = z on it.

    On a sphere z is uniform in [−R, R] (Archimedes), so the exact answers are: area 4πR²,
    mean 0, half the area positive, sigma²± = R²/12, Pi = MPI/eV = R/2.
    """
    ax = np.arange(-2.6, 2.6, _STEP) + _STEP / 2       # no grid plane exactly at z = 0
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = _ISO * np.exp(_R - np.sqrt(x ** 2 + y ** 2 + z ** 2))
    atoms = [("O", (0.0, 0.0, 1.0)), ("Ti", (0.0, 0.0, -1.0))]
    origin = (ax[0],) * 3
    return (_grid(z, "mep", _STEP, origin, atoms), _grid(rho, "density", _STEP, origin, atoms))


def test_mep_surface_area_and_averages_match_the_analytic_sphere(sphere):
    st = analyze_mep(*sphere, isovalue=_ISO)
    assert st.surface_area_ang2 == pytest.approx(4 * np.pi * _R ** 2, rel=5e-3)
    assert st.surface_vmean == pytest.approx(0.0, abs=5e-3)
    assert st.surface_positive_fraction == pytest.approx(0.5, abs=5e-3)
    assert st.positive_area_ang2 + st.negative_area_ang2 == pytest.approx(st.surface_area_ang2)
    assert st.sigma2_plus == pytest.approx(_R ** 2 / 12, rel=1e-2)
    assert st.sigma2_minus == pytest.approx(_R ** 2 / 12, rel=1e-2)
    assert st.nu_balance == pytest.approx(0.25, rel=1e-2)
    assert st.pi_au == pytest.approx(_R / 2, rel=1e-2)
    assert st.mpi_ev == pytest.approx(_R / 2 * HARTREE_TO_EV, rel=1e-2)


def test_mep_surface_statistics_are_area_weighted_not_per_point(sphere):
    """Grid edges cross a surface patch |n_x| + |n_y| + |n_z| times per unit area (in units of
    the cell face), so unweighted crossings over-count the patches tilted to the axes. With
    V = |x| + |y| + |z| the exact surface mean is 3R/2; per point it comes out high."""
    from cadmia.electronic.mep import _surface_points
    mep_grid, rho_grid = sphere
    ax = mep_grid.origin[0] + _STEP * np.arange(mep_grid.data.shape[0])
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    v, w, _ = _surface_points(rho_grid.data, np.abs(x) + np.abs(y) + np.abs(z), _ISO, np.full(3, _STEP))
    weighted, per_point = np.average(v, weights=w) - 1.5 * _R, np.mean(v) - 1.5 * _R
    assert abs(weighted) < 1e-3 * _R
    assert per_point > 5 * abs(weighted)


def test_mep_extremes_lie_on_the_surface_and_name_their_atom(sphere):
    st = analyze_mep(*sphere, isovalue=_ISO)
    assert st.surface_vmax == pytest.approx(_R, rel=2e-3)
    assert st.surface_vmin == pytest.approx(-_R, rel=2e-3)
    assert np.linalg.norm(st.surface_vmax_position) == pytest.approx(_R, rel=2e-3)
    assert (st.surface_vmax_atom, st.surface_vmin_atom) == ("O1", "Ti2")
    assert "next to O1" in str(st)


def test_mep_vacuum_spikes_and_nuclear_cusps_stay_out(sphere):
    mep_grid, rho_grid = sphere
    spiked = mep_grid.data.copy()
    spiked[rho_grid.data < _ISO / 50] = 99.0             # vacuum, far outside the surface
    spiked[rho_grid.data > _ISO * 50] = 99.0             # interior, near the nuclei
    st = analyze_mep(_grid(spiked, "mep", _STEP, mep_grid.origin, [("H", (0, 0, 0))]),
                     rho_grid, isovalue=_ISO)
    assert st.surface_vmax == pytest.approx(_R, rel=2e-3)


def test_mep_without_a_surface_on_the_grid_is_an_error_not_the_interior():
    rho = np.full((4, 4, 4), 0.01)                        # all inside: the isovalue never crossed
    with pytest.raises(ValueError, match="does not cross the grid"):
        analyze_mep(_grid(np.zeros((4, 4, 4)), "mep"), _grid(rho, "density"), isovalue=0.001)


def test_mep_rejects_a_grid_with_tilted_axes(sphere):
    mep_grid, rho_grid = sphere
    tilted = VolumetricGrid(structure=mep_grid.structure, origin=mep_grid.origin,
                            vx=np.array([_STEP, 0.01, 0.0]), vy=mep_grid.vy, vz=mep_grid.vz,
                            data=mep_grid.data, field_type="mep")
    with pytest.raises(ValueError, match="axes along x, y and z"):
        analyze_mep(tilted, rho_grid, isovalue=_ISO)


def test_mep_rejects_a_density_grid_of_another_shape():
    with pytest.raises(ValueError, match="does not match density grid shape"):
        analyze_mep(_grid(np.zeros((2, 2, 2)), "mep"), _grid(np.ones((2, 2, 3)) * 0.01, "density"))


def test_mep_without_density_leaves_surface_stats_empty():
    """The bare MEP path must not invent surface statistics."""
    st = analyze_mep(_grid(np.array([[[0.1, -0.1], [0.2, -0.2]]]), "mep"))
    assert st.surface_area_ang2 is None and st.mpi_ev is None and st.isovalue_used is None


def test_analyze_mep_basic_stats():
    grid = parse_cube(_CUBE)
    stats = analyze_mep(grid)
    assert isinstance(stats, MEPStats)
    assert stats.vmin <= stats.vmean <= stats.vmax
    assert str(stats)  # __str__ renders


def test_write_mep_report(tmp_path):
    grid = parse_cube(_CUBE)
    stats = analyze_mep(grid)
    out = tmp_path / "mep.txt"
    write_mep_report(stats, out)
    assert out.exists() and out.read_text(encoding="utf-8").strip()


def test_cli_electronic_mep_stdout():
    grid = parse_cube(_CUBE)
    expected = analyze_mep(grid)
    result = CliRunner().invoke(main, ["electronic", "mep", str(_CUBE)])
    assert result.exit_code == 0
    # Assert the numbers reach stdout: `assert result.output.strip()` also passed on a
    # truncated traceback, which is exactly what a CLI smoke test must not do.
    assert f"{expected.vmin:.6f}" in result.output
    assert f"{expected.vmax:.6f}" in result.output


def test_cli_electronic_mep_to_file(tmp_path):
    out = tmp_path / "report.txt"
    result = CliRunner().invoke(main, ["electronic", "mep", str(_CUBE), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "MEP report written to" in result.output
