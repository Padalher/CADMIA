"""ORCA 6 input file writer (.inp)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from cadmia.calculators.params import CalcParams, resolve_multiplicity

# ORCA keyword mappings
_DISPERSION_KW = {
    "d3bj": "D3BJ",
    "d4":   "D4",
    "none": "",
    "":     "",
}

_TASK_KW = {
    "sp":             "SP",
    "opt":            "Opt",
    "freq":           "Freq",
    "opt+freq":       "Opt Freq",
    "opt_tight":      "TightOpt",
    "opt_tight+freq": "TightOpt Freq",
}


def write_inp(
    atoms: List[str],
    positions: "list | np.ndarray",
    params: CalcParams,
    output_path: "str | Path",
    title: str = "",
) -> Path:
    """Write an ORCA 6 input file (.inp).

    Parameters
    ----------
    atoms       : element symbols, length N.
    positions   : (N, 3) coordinate array in Ångström.
    params      : CalcParams describing the level of theory.
    output_path : destination file path.
    title       : optional comment (written as a # line after the simple input).
    """
    out = Path(output_path)

    disp_kw = _DISPERSION_KW.get(params.dispersion, "")
    task_kw = _TASK_KW.get(params.task, "SP")

    # Simple input line: ! functional basis [dispersion] task [extra]
    simple_parts = [params.functional, params.basis]
    if disp_kw:
        simple_parts.append(disp_kw)
    simple_parts.append(task_kw)
    if params.extra_orca:
        simple_parts.append(params.extra_orca)
    simple_input = "! " + " ".join(simple_parts)

    lines: List[str] = [simple_input, ""]

    if title:
        lines += [f"# {title}", ""]

    # Parallel and memory blocks
    lines += [
        f"%pal nprocs {params.nprocs} end",
        f"%maxcore {params.mem_orca_mb_core}",
    ]

    # SCF iteration limit (helps open-shell ions converge); 0 keeps the ORCA default.
    if params.scf_maxiter > 0:
        lines += ["%scf", f"  MaxIter {params.scf_maxiter}", "end"]

    # Geometry step limit; 0 keeps the ORCA default of max(50, 3N).
    if params.geom_maxiter > 0 and "opt" in params.task:
        lines += ["%geom", f"  MaxIter {params.geom_maxiter}", "end"]

    # Fine CHELPG grid, only when CHELPG charges were actually requested.
    if params.chelpg_fine and "chelpg" in params.extra_orca.lower():
        lines += [
            "%chelpg",
            "  GRID 0.20",
            "  RMAX 3.50",
            "  VDWRADII COSMO",
            "  DIPOLE FALSE",
            "end",
        ]

    # Analysis output requests (Mayer / per-MO Löwdin / Hirshfeld …), usually from a preset.
    if params.orca_output_flags:
        lines.append("%output")
        for flag in params.orca_output_flags:
            lines.append(f"  Print[{flag}] 1")
        lines.append("end")

    lines += [
        "",
        f"* xyz {params.charge} {resolve_multiplicity(params, atoms)}",
    ]

    for sym, pos in zip(atoms, positions):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        lines.append(f"{sym:<4} {x:14.8f} {y:14.8f} {z:14.8f}")

    lines += ["*", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
