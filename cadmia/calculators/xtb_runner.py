"""xTB runner — fast GFN-xTB geometry optimization for pre-screening adsorption configs.

Shells out to the standalone ``xtb`` binary (Grimme group; free, no Python dependency,
offline). Used by the prescreen agent to cheaply optimize and rank the many starting
geometries from the Geometry Agent before the expensive DFT stage. GFN2-xTB already
includes D4 dispersion, so the screen and the (D4) DFT stage share the same dispersion
physics.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# "TOTAL ENERGY   -123.456789 Eh" in the xtb stdout; also in the xtbopt.xyz comment.
_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_XYZ_COMMENT_ENERGY_RE = re.compile(r"energy:\s*(-?\d+\.\d+)")

# Geometry-optimization verdict, as xtb prints it:
#   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS ***
#   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 1 ITERATIONS ***
# Both patterns are anchored on "GEOMETRY OPTIMIZATION" on purpose: a run that
# fails to converge the geometry still prints "*** convergence criteria satisfied
# after N iterations ***" for every SCF cycle, so matching on "converge" alone
# reads a failed optimization as a successful one.
_GEOM_CONVERGED_RE = re.compile(r"GEOMETRY\s+OPTIMIZATION\s+CONVERGED", re.IGNORECASE)
_GEOM_FAILED_RE = re.compile(r"FAILED\s+TO\s+CONVERGE\s+GEOMETRY\s+OPTIMIZATION", re.IGNORECASE)


class XTBError(RuntimeError):
    """xTB executable missing or run failed."""


@dataclass
class XTBResult:
    """Outcome of one xTB optimization."""
    energy_hartree: float
    optimized_xyz: Path
    converged: bool = True


def xtb_available() -> bool:
    """True when the xtb executable is on PATH."""
    return shutil.which("xtb") is not None


def _parse_convergence(stdout: str) -> Optional[bool]:
    """Read xtb's geometry-optimization verdict; None when it printed neither.

    The failure banner wins if both appear: on a restarted optimization the log
    can carry an earlier success followed by the run that actually gave up.
    """
    if _GEOM_FAILED_RE.search(stdout):
        return False
    if _GEOM_CONVERGED_RE.search(stdout):
        return True
    return None


def _parse_energy(stdout: str, opt_xyz: Path) -> Optional[float]:
    matches = _ENERGY_RE.findall(stdout)
    if matches:
        return float(matches[-1])
    # Fallback: the optimized .xyz comment line carries "energy: <Eh>".
    if opt_xyz.exists():
        lines = opt_xyz.read_text(encoding="utf-8").splitlines()
        if len(lines) > 1:
            m = _XYZ_COMMENT_ENERGY_RE.search(lines[1])
            if m:
                return float(m.group(1))
    return None


def run_xtb_opt(
    xyz_path: "str | Path",
    *,
    charge: int = 0,
    multiplicity: int = 1,
    gfn: int = 2,
    workdir: "str | Path | None" = None,
    xtb_path: str = "xtb",
    timeout_s: int = 1800,
    threads: int = 1,
    fix_atoms: int = 0,
) -> XTBResult:
    """Optimize a geometry with GFN-xTB and return (energy, optimized .xyz).

    Parameters
    ----------
    xyz_path     : input geometry (.xyz).
    charge       : total molecular charge.
    multiplicity : spin multiplicity (2S+1); xtb receives --uhf = multiplicity-1.
    gfn          : GFN parameterization (0, 1, or 2; default 2 = GFN2-xTB with D4).
    workdir      : scratch dir (xtb writes xtbopt.xyz etc. here). If None a temp dir is
                   created and **left in place** — the returned .xyz lives inside it, so
                   it cannot be deleted here; the caller owns its removal. Pass a workdir
                   to control that (the prescreen agent does, and cleans it per config).
    xtb_path     : xtb executable name/path.
    threads      : OpenMP threads for xtb (default 1; some builds SIGSEGV with >1).
    fix_atoms    : hold atoms 1..fix_atoms rigid (xtb ``$constrain``); 0 relaxes everything.
    """
    src = Path(xyz_path).resolve()
    if not src.exists():
        raise XTBError(f"input geometry not found: {src}")
    if shutil.which(xtb_path) is None:
        raise XTBError(
            f"xtb executable '{xtb_path}' not found on PATH. Install xtb "
            "(https://github.com/grimme-lab/xtb) to enable pre-screening."
        )

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="cadmia_xtb_"))
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [xtb_path, str(src), "--opt", "--gfn", str(gfn),
           "--chrg", str(charge), "--uhf", str(max(0, multiplicity - 1))]
    if fix_atoms > 0:
        # $constrain holds every distance among those atoms (measured: 0.013 A at most on Al2TiO5
        # + CO2). $fix does not: with the default ANC optimizer it let the cluster move 1.6 A.
        (tmp / "xcontrol").write_text(
            f"$constrain\n force constant=5.0\n atoms: 1-{fix_atoms}\n$end\n", encoding="utf-8")
        cmd += ["--input", "xcontrol"]
    # Thread control: several xtb builds (e.g. the Ubuntu apt 6.6.1) SIGSEGV with
    # >1 OpenMP thread, so default to single-threaded and override any inherited
    # OMP_NUM_THREADS. A small thread stack is the other usual SIGSEGV cause.
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(max(1, threads))
    env["MKL_NUM_THREADS"] = str(max(1, threads))
    env.setdefault("OMP_STACKSIZE", "4G")
    try:
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        raise XTBError(f"xtb timed out after {timeout_s}s on {src.name}") from exc

    opt_xyz = tmp / "xtbopt.xyz"
    energy = _parse_energy(proc.stdout, opt_xyz)
    if proc.returncode != 0 and energy is None:
        raise XTBError(f"xtb failed on {src.name} (rc={proc.returncode}): "
                       f"{proc.stderr.strip()[:200]}")
    if energy is None:
        raise XTBError(f"could not parse xtb energy for {src.name}")

    # Convergence comes from xtb's own verdict, not from the presence of
    # xtbopt.xyz: xtb writes that file with the last geometry even when the
    # optimization ran out of cycles (and exits 0 doing so). Absence of a verdict
    # counts as unconverged — a config wrongly flagged unconverged only adds a
    # warning, while one wrongly flagged converged sends a geometry that is not a
    # minimum into the DFT stage silently.
    # A nonzero exit code with a parseable energy is accepted (xtb reports one for runs
    # that ran out of cycles) but never as converged: whatever made it fail is reason
    # enough not to treat the geometry as a minimum.
    converged = proc.returncode == 0 and _parse_convergence(proc.stdout) is True
    if not opt_xyz.exists():
        # No optimized geometry at all; fall back to the input.
        return XTBResult(energy_hartree=energy, optimized_xyz=src, converged=False)
    return XTBResult(energy_hartree=energy, optimized_xyz=opt_xyz, converged=converged)
