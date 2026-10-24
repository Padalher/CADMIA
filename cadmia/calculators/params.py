"""Shared calculation parameters for Gaussian and ORCA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


_VALID_TASKS = {"sp", "opt", "freq", "opt+freq", "opt_tight", "opt_tight+freq"}
_VALID_DISPERSIONS = {"d3bj", "d4", "none", ""}


@dataclass
class CalcParams:
    """Engine-agnostic DFT calculation parameters.

    Attributes
    ----------
    functional         : DFT functional keyword (e.g. "PBE0", "B3LYP", "wB97X-D3").
    basis              : Basis set name (e.g. "def2-TZVP", "6-311+G(d,p)").
    dispersion         : Empirical dispersion — "d4" (default), "d3bj", or "none".
                         D4's charge-dependent C6 suits mixed-oxide / charged species;
                         ORCA emits it natively (! D4). Override per calculation as needed.
    task               : Calculation type — "sp", "opt", "freq", or "opt+freq".
    charge             : Total molecular charge.
    multiplicity       : Spin multiplicity (2S+1) used when auto_multiplicity is False.
    auto_multiplicity  : When True, the input writers derive the multiplicity from the
                         electron-count parity (Σ Z − charge): even → 1, odd → 2. The
                         explicit ``multiplicity`` is then ignored. Manual override =
                         leave this False and set ``multiplicity``.
    nprocs             : Number of parallel processes / CPU cores.
    mem_gaussian_gb    : Memory for Gaussian (%mem, in GB).
    mem_orca_mb_core   : Memory per core for ORCA (%maxcore, in MB).
    extra_gaussian     : Additional Gaussian route keywords (appended verbatim).
    extra_orca         : Additional ORCA simple input keywords (appended verbatim).
    orca_output_flags  : ORCA ``Print[<flag>] 1`` lines to emit inside a ``%output`` block
                         (e.g. "P_Mayer", "P_OrbPopMO_L", "P_Hirshfeld"). Usually set via a
                         preset (see cadmia.calculators.presets).
    scf_maxiter        : SCF iteration limit. 0 (default) keeps the engine default (no block
                         emitted); >0 writes ORCA ``%scf MaxIter N end`` / Gaussian
                         ``SCF=(MaxCycle=N)``. Useful to help open-shell ions converge.
    geom_maxiter       : Geometry optimization step limit. 0 (default) keeps the engine
                         default, which for ORCA is max(50, 3N) — 50 steps for a small
                         adduct. That is not enough when the optimization has to break and
                         remake bonds in the substrate: those runs stop at the cap with a
                         provisional energy, and they are precisely the low-energy ones.
                         >0 writes ORCA ``%geom MaxIter N end`` / Gaussian
                         ``Opt=(MaxCycles=N)``. Ignored when the task has no optimization.
    chelpg_fine        : When True, ORCA emits a fine ``%chelpg`` block (GRID 0.20, RMAX 3.50,
                         VDWRADII COSMO, DIPOLE FALSE); only meaningful when the CHELPG keyword
                         is requested. ORCA-only (ignored for Gaussian).
    """

    functional:      str  = "PBE0"
    basis:           str  = "def2-TZVP"
    dispersion:      str  = "d4"
    task:            str  = "opt"
    charge:          int  = 0
    multiplicity:    int  = 1
    auto_multiplicity: bool = False
    nprocs:          int  = 8
    mem_gaussian_gb: int  = 16
    mem_orca_mb_core: int = 2000
    extra_gaussian:  str  = ""
    extra_orca:      str  = ""
    orca_output_flags: Tuple[str, ...] = ()
    scf_maxiter:     int  = 0
    geom_maxiter:    int  = 0
    chelpg_fine:     bool = False

    def __post_init__(self) -> None:
        self.dispersion = self.dispersion.lower()
        self.task       = self.task.lower()
        if self.task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS}, got '{self.task}'")
        if self.dispersion not in _VALID_DISPERSIONS:
            raise ValueError(f"dispersion must be one of {_VALID_DISPERSIONS}, got '{self.dispersion}'")
        if self.multiplicity < 1:
            raise ValueError(f"multiplicity must be >= 1, got {self.multiplicity}")
        if self.scf_maxiter < 0:
            raise ValueError(f"scf_maxiter must be >= 0, got {self.scf_maxiter}")
        if self.geom_maxiter < 0:
            raise ValueError(f"geom_maxiter must be >= 0, got {self.geom_maxiter}")


def resolve_multiplicity(params: "CalcParams", atoms) -> int:
    """Multiplicity to write for *atoms*: parity-derived when auto, else the explicit value."""
    if params.auto_multiplicity:
        from cadmia.core.spin import multiplicity_from_electrons
        return multiplicity_from_electrons(atoms, params.charge)
    return params.multiplicity
