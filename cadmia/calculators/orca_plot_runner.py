"""orca_plot runner — generate Gaussian .cube files from an ORCA wavefunction (.gbw).

Shells out to ORCA's ``orca_plot`` (ships with ORCA; no Python dependency). Used by
``cadmia density`` to produce the electron-density / HOMO / LUMO / MEP cubes that feed the
Fukui, dual-descriptor and deformation-density maps (`cadmia/electronic/density_maps.py`).

Grid alignment (critical for the difference maps):
- Fukui f±/f0/Δf subtract the densities of N / N+1 / N-1. Those three share the SAME geometry
  (only the charge differs), so with the same grid interval count orca_plot emits voxel-aligned
  cubes automatically — the subtraction is valid.
- The deformation density Δρ subtracts cluster and probe densities computed at the adduct
  geometry; those must be generated on the same grid as the adduct (same atoms/positions +
  same intervals). The caller checks alignment with ``VolumetricGrid.same_grid_as`` and errors
  clearly if it does not hold.

The non-interactive interface feeds orca_plot its menu selections on stdin. The menu codes
below follow ORCA 6.x; expose them as constants so a different build is a one-line fix. This
runner is gated on ``orca_plot`` being on PATH — it is validated with a real ORCA installation, not here.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# orca_plot interactive menu codes — verified against ORCA 6.1.0.
# Tunable if a build differs (the whole interaction is just these constants + _build_plot_script).
_MENU_SET_TYPE = "1"      # 1 → choose plot type
_MENU_SET_ORBITAL = "2"   # 2 → choose orbital index (MO plots only)
_MENU_SET_NGRID = "4"     # 4 → set number of grid intervals
_MENU_GENERATE = "11"     # 11 → generate the plot to file
_MENU_EXIT = "12"         # 12 → exit
# Gaussian cube is already the default output format in ORCA 6.x ("the default volumetric
# output format is Cube"), so we do NOT select it explicitly: the format submenu was a source
# of stdin desync (an unanswered sub-prompt shifts every later token → runaway grid / hang).

# Plot-type codes under "Enter type of plot" (ORCA 6.1.0 submenu).
_PLOT_TYPE = {
    "density": "2",    # (scf) electron density
    "spin":    "3",    # (scf) spin density
    "homo":    "1",    # molecular orbitals
    "lumo":    "1",    # molecular orbitals
    "mep":     "43",   # electrostatic potential (ORCA 6.1.0; was "6" — that is a TD-DFT menu entry)
}
# Selecting a (scf) density first prints "The default name of the density would be: <stem>.scfp"
# and asks "Is this the one you want (y/n)?" — answer 'y' to accept the default density, otherwise
# the next token is swallowed by that prompt and the whole menu desyncs.
_DENSITY_KINDS = {"density", "spin"}
# The MEP entry does NOT ask y/n: it lists the saved densities and asks "Enter Name for an STATE
# Density", so it needs the name itself (<stem>.scfp). Answering 'y' there selects nothing and the
# plot is generated with no density assigned.
_NAMED_DENSITY_KINDS = {"mep"}
_VALID_KINDS = tuple(_PLOT_TYPE)


class OrcaPlotError(RuntimeError):
    """orca_plot executable missing or the run failed."""


@dataclass
class OrcaPlotResult:
    """One generated cube."""
    cube_path: Path
    kind: str


def orca_plot_available(orca_plot_path: str = "orca_plot") -> bool:
    """True when the orca_plot executable is on PATH."""
    return shutil.which(orca_plot_path) is not None


def _grid_arg(ngrid: "int | Tuple[int, int, int]") -> str:
    """Menu-4 argument: one integer (same N on the three axes) or ``NX NY NZ`` per axis.

    orca_plot's "number of grid intervals" accepts either form; passing NX NY NZ is how we get a
    uniform step on a non-cubic box (N per axis derived from the box length and a target step).
    """
    if isinstance(ngrid, (tuple, list)):
        nx, ny, nz = ngrid
        return f"{int(nx)} {int(ny)} {int(nz)}"
    return str(int(ngrid))


def _build_plot_script(kind: str, *, ngrid: "int | Tuple[int, int, int]" = 80,
                       mo_index: Optional[int] = None,
                       density_name: Optional[str] = None) -> str:
    """Build the orca_plot stdin menu sequence for one cube (pure function; testable).

    ``kind`` ∈ {density, spin, homo, lumo, mep}. ``ngrid`` is either a single interval count or an
    ``(nx, ny, nz)`` triple (uniform step). For MO plots (homo/lumo) ``mo_index`` is the 0-based
    orbital number; the caller derives it from the parsed occupation.
    """
    if kind not in _PLOT_TYPE:
        raise OrcaPlotError(f"unknown plot kind {kind!r}; expected one of {_VALID_KINDS}.")
    lines = [
        _MENU_SET_NGRID, _grid_arg(ngrid),         # grid resolution (cube is the default format)
        _MENU_SET_TYPE, _PLOT_TYPE[kind],          # plot type
    ]
    if kind in _DENSITY_KINDS:
        lines.append("y")                          # accept the default saved density (<stem>.scfp)
    if kind in _NAMED_DENSITY_KINDS:
        if not density_name:
            raise OrcaPlotError(f"{kind} plot requires density_name (e.g. '<stem>.scfp').")
        lines.append(density_name)
    if kind in ("homo", "lumo"):
        if mo_index is None:
            raise OrcaPlotError(f"{kind} plot requires mo_index (0-based orbital number).")
        lines += [_MENU_SET_ORBITAL, str(int(mo_index))]
    lines += [_MENU_GENERATE, _MENU_EXIT]
    return "\n".join(lines) + "\n"


_BOUNDARY_RE = re.compile(
    r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\(\s*([xyz])\s+direction\s*\)")


def _parse_boundaries(text: str) -> Optional[Dict[str, Tuple[float, float]]]:
    """Extract the (min, max) grid limits per axis (Bohr) from orca_plot's menu dump (testable).

    orca_plot prints, e.g., ``-7.000000  8.738548 (z direction)`` for each axis; the box is the
    atomic bounding box plus a fixed 7 Bohr margin per side. Returns None if all three axes are
    not found.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    for lo, hi, axis in _BOUNDARY_RE.findall(text):
        bounds[axis] = (float(lo), float(hi))
    return bounds if {"x", "y", "z"} <= bounds.keys() else None


def _orca_plot_complaint(text: str) -> str:
    """Turn orca_plot's own stdout into the reason it produced no grid box.

    It reports a .gbw written by an incompatible ORCA build plainly ("either corrupt or from a
    different ORCA version", then an Unknown HFTYP error) and then prints no boundaries at all.
    Reporting that verbatim beats blaming the menu, which sends the reader to the wrong place.
    """
    if "different ORCA version" in text or "Unknown HFTYP" in text:
        return ("orca_plot says the file is corrupt or was written by a different ORCA build, so "
                "it never reached the menu. Re-run the single point with the ORCA that is on this "
                "machine, or run orca_plot where the .gbw was produced.")
    return "orca_plot printed no grid box (unexpected menu output for this ORCA version)."


def probe_boundaries(
    gbw_path: "str | Path", *, orca_plot_path: str = "orca_plot", timeout_s: int = 120,
) -> Dict[str, Tuple[float, float]]:
    """Read the grid box orca_plot would use for a .gbw, without generating any cube.

    Enters interactive mode and exits immediately; orca_plot prints the box on entry. Used to size
    a per-axis grid for a uniform step (`ngrid_for_step`).
    """
    gbw = Path(gbw_path).resolve()
    if shutil.which(orca_plot_path) is None:
        raise OrcaPlotError(
            f"orca_plot executable '{orca_plot_path}' not found on PATH. It ships with ORCA.")
    try:
        proc = subprocess.run([orca_plot_path, str(gbw), "-i"], input=_MENU_EXIT + "\n",
                              capture_output=True, text=True, timeout=timeout_s, cwd=gbw.parent)
    except subprocess.TimeoutExpired as exc:
        raise OrcaPlotError(f"orca_plot timed out reading the grid box of {gbw.name}") from exc
    bounds = _parse_boundaries(proc.stdout)
    if bounds is None:
        raise OrcaPlotError(
            f"could not read the grid boundaries from orca_plot for {gbw.name}. "
            + _orca_plot_complaint(proc.stdout))
    return bounds


def ngrid_for_step(boundaries: Dict[str, Tuple[float, float]], step: float) -> Tuple[int, int, int]:
    """Per-axis interval count that lands the grid on ~``step`` Bohr, given the box (Bohr).

    ``N_i = round(L_i / step)`` per axis → a near-uniform step on a non-cubic box (this is what
    keeps the difference-map lobes smooth instead of faceted; the residual per-axis spread is the
    rounding, well under 1%).
    """
    if step <= 0:
        raise OrcaPlotError(f"grid step must be > 0 Bohr, got {step}.")
    ns = []
    for axis in ("x", "y", "z"):
        lo, hi = boundaries[axis]
        ns.append(max(1, int(round((hi - lo) / step))))
    return (ns[0], ns[1], ns[2])


def run_orca_plot(
    gbw_path: "str | Path",
    kind: str,
    out_cube: "str | Path",
    *,
    ngrid: "int | Tuple[int, int, int]" = 80,
    mo_index: Optional[int] = None,
    orca_plot_path: str = "orca_plot",
    timeout_s: int = 21600,
) -> OrcaPlotResult:
    """Generate one .cube from a .gbw with orca_plot. Gated on orca_plot being installed.

    orca_plot writes the cube next to the .gbw using its own naming; we move it to ``out_cube``.

    ``timeout_s`` guards against a hang, not against a slow plot, so it is generous: the
    electrostatic potential costs far more per grid point than a density (a few tens of atoms on a
    fine grid can take hours), and there is nothing to gain from killing a cube that is
    still making progress.
    """
    gbw = Path(gbw_path).resolve()
    if not gbw.exists():
        raise OrcaPlotError(f"gbw file not found: {gbw}")
    if shutil.which(orca_plot_path) is None:
        raise OrcaPlotError(
            f"orca_plot executable '{orca_plot_path}' not found on PATH. It ships with ORCA; "
            "run this on the machine where ORCA is installed (or pass --orca-plot-path)."
        )

    if kind in _NAMED_DENSITY_KINDS and not gbw.with_suffix(".densitiesinfo").exists():
        raise OrcaPlotError(
            f"{kind} needs the saved density next to the wavefunction: {gbw.stem}.densities and "
            f"{gbw.stem}.densitiesinfo are missing beside {gbw.name}. They are written by the "
            "single point that produced the .gbw; copy them along with it.")

    script = _build_plot_script(kind, ngrid=ngrid, mo_index=mo_index,
                                density_name=f"{gbw.stem}.scfp")
    cmd = [orca_plot_path, str(gbw), "-i"]
    try:
        proc = subprocess.run(cmd, input=script, capture_output=True, text=True,
                              timeout=timeout_s, cwd=gbw.parent)
    except subprocess.TimeoutExpired as exc:
        raise OrcaPlotError(f"orca_plot timed out after {timeout_s}s on {gbw.name}") from exc
    if proc.returncode != 0:
        raise OrcaPlotError(f"orca_plot failed on {gbw.name} (rc={proc.returncode}): "
                            f"{proc.stderr.strip()[:200]}")

    # orca_plot writes <gbw_stem>.<kind>.cube (naming varies); pick the newest .cube it produced.
    produced = sorted(gbw.parent.glob(f"{gbw.stem}*.cube"), key=lambda p: p.stat().st_mtime)
    if not produced:
        raise OrcaPlotError(f"orca_plot produced no .cube for {gbw.name}; check the menu codes "
                            "for this ORCA version (orca_plot_runner._MENU_*).")
    dst = Path(out_cube)
    dst.parent.mkdir(parents=True, exist_ok=True)
    produced[-1].replace(dst)
    return OrcaPlotResult(cube_path=dst, kind=kind)
