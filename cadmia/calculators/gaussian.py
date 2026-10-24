"""Gaussian 16 input file writer (.gjf)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from cadmia.calculators.params import CalcParams, resolve_multiplicity

# Gaussian keyword mappings
_DISPERSION_KW = {
    "d3bj": "EmpiricalDispersion=GD3BJ",
    "none": "",
    "":     "",
}


def _dispersion_keyword(dispersion: str) -> str:
    """Route keyword for *dispersion*, or raise if Gaussian cannot apply it.

    Gaussian 16 implements GD2/GD3/GD3BJ and Petersson-Frisch damping only: **there is no
    native D4** (it needs Grimme's external ``dftd4``). CADMIA defaults to D4 because ORCA
    is the primary engine, so this writer refuses it rather than emit a keyword the engine
    rejects at startup — or, worse, silently drop the dispersion and change the level of
    theory behind the caller's back.
    """
    if dispersion == "d4":
        raise ValueError(
            "dispersion 'd4' is not available in Gaussian 16 (it implements "
            "GD2/GD3/GD3BJ/PFD only). Use --dispersion d3bj, or --engine orca, "
            "where D4 is supported."
        )
    return _DISPERSION_KW.get(dispersion, "")


# ORCA/Turbomole spellings that Gaussian does not recognize. CADMIA's defaults are written
# in the ORCA vocabulary because ORCA is the primary engine, so the Gaussian writer has to
# translate rather than pass them through and let l1.exe fail on the route card.
_GAUSSIAN_NAME = {
    # Same functional, different name: Gaussian calls PBE0 "PBE1PBE".
    "pbe0":       "PBE1PBE",
    "wb97x-d":    "wB97XD",
    "m06-2x":     "M062X",
    # The def2 family is internal to Gaussian 16 but spelled without the hyphen.
    "def2-svp":   "Def2SVP",
    "def2-svpp":  "Def2SVPP",
    "def2-tzvp":  "Def2TZVP",
    "def2-tzvpp": "Def2TZVPP",
    "def2-qzvp":  "Def2QZVP",
    "def2-qzvpp": "Def2QZVPP",
}


def _gaussian_name(token: str) -> str:
    """Gaussian's spelling for a functional or basis CADMIA names the ORCA way."""
    return _GAUSSIAN_NAME.get(token.strip().lower(), token)


_TASK_KW = {
    "sp":             "",
    "opt":            "Opt",
    "freq":           "Freq",
    "opt+freq":       "Opt Freq",
    "opt_tight":      "Opt=Tight",
    "opt_tight+freq": "Opt=Tight Freq",
}


def write_gjf(
    atoms: List[str],
    positions: "list | np.ndarray",
    params: CalcParams,
    output_path: "str | Path",
    title: str = "",
    chk_name: str = "",
) -> Path:
    """Write a Gaussian 16 input file (.gjf).

    Parameters
    ----------
    atoms       : element symbols, length N.
    positions   : (N, 3) coordinate array in Ångström.
    params      : CalcParams describing the level of theory.
    output_path : destination file path.
    title       : comment line in the job title block (default: stem of output_path).
    chk_name    : %chk= filename stem (default: stem of output_path).
    """
    out = Path(output_path)
    stem = chk_name or out.stem
    title = title or stem

    disp_kw   = _dispersion_keyword(params.dispersion)
    task_kw   = _TASK_KW.get(params.task, "")
    if params.geom_maxiter > 0 and task_kw.startswith("Opt"):
        # Gaussian carries the step limit inside the Opt keyword, not in a block of its own,
        # so Opt=Tight has to become Opt=(Tight,MaxCycles=N) rather than gain a second Opt.
        head, _, rest = task_kw.partition(" ")
        inner = "Tight," if head == "Opt=Tight" else ""
        task_kw = " ".join(filter(None, [f"Opt=({inner}MaxCycles={params.geom_maxiter})", rest]))

    # Build route card. The space after '#' is required: Gaussian reads the character right
    # after it as the print level (N/P/T), so "#PBE1PBE/..." is parsed as print level P plus
    # a stray "BE1PBE" and dies on the '/' with a syntax error.
    route_parts = [f"# {_gaussian_name(params.functional)}/{_gaussian_name(params.basis)}"]
    if task_kw:
        route_parts.append(task_kw)
    if disp_kw:
        route_parts.append(disp_kw)
    if params.scf_maxiter > 0:
        route_parts.append(f"SCF=(MaxCycle={params.scf_maxiter})")
    if params.extra_gaussian:
        route_parts.append(params.extra_gaussian)
    route = " ".join(route_parts)

    lines: List[str] = [
        f"%chk={stem}.chk",
        f"%nprocshared={params.nprocs}",
        f"%mem={params.mem_gaussian_gb}GB",
        route,
        "",
        title,
        "",
        f"{params.charge} {resolve_multiplicity(params, atoms)}",
    ]

    for sym, pos in zip(atoms, positions):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        lines.append(f"{sym:<4} {x:14.8f} {y:14.8f} {z:14.8f}")

    # Gaussian requires two blank lines at the end
    lines += ["", ""]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
