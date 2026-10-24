"""Calculation presets — bundle the print/output flags a downstream analysis needs.

A bare DFT input gives energies and orbitals, but the CADMIA analysis also reads Mayer
bond orders, per-MO Löwdin populations (PDOS), and Hirshfeld/CHELPG charges. Requesting
them by hand on every job is error-prone, so a *preset* turns a one-word choice into the
exact ORCA ``%output`` block and simple-input keywords (or Gaussian ``pop=`` list) that
make the output self-sufficient for the report.

Presets only *add* requests; the level of theory (functional/basis/dispersion/task) stays
whatever the caller set. Apply one with :func:`apply_preset`, which returns a new
``CalcParams`` (the original is left untouched).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

from cadmia.calculators.params import CalcParams


@dataclass(frozen=True)
class CalculationPreset:
    """A named bundle of analysis-oriented output requests.

    orca_print_flags     : ORCA ``Print[<flag>] 1`` lines emitted inside ``%output``.
    orca_simple_keywords : extra ORCA simple-input keywords (e.g. ``CHELPG``).
    gaussian_pop         : population keywords for Gaussian ``pop=(...)``.
    description          : one-line human description (for --help / listings).
    """
    name:                 str
    orca_print_flags:     Tuple[str, ...] = ()
    orca_simple_keywords: Tuple[str, ...] = ()
    gaussian_pop:         Tuple[str, ...] = ()
    description:          str = ""


PRESETS: Dict[str, CalculationPreset] = {
    "none": CalculationPreset(
        name="none",
        description="No extra output (level of theory only).",
    ),
    "cdft": CalculationPreset(
        name="cdft",
        orca_print_flags=("P_Hirshfeld",),
        orca_simple_keywords=("CHELPG",),
        gaussian_pop=("Hirshfeld", "CHELPG"),
        description="Charges for condensed Fukui (Hirshfeld + CHELPG).",
    ),
    "analysis": CalculationPreset(
        name="analysis",
        orca_print_flags=("P_Hirshfeld", "P_Mayer", "P_OrbPopMO_L"),
        orca_simple_keywords=("CHELPG",),
        gaussian_pop=("Hirshfeld", "CHELPG"),
        description="Full property set: Hirshfeld + CHELPG charges, Mayer bond orders, "
                    "and per-MO Löwdin populations for PDOS.",
    ),
    "accurate": CalculationPreset(
        name="accurate",
        # 'analysis' property set + the recommended ORCA accuracy/efficiency block for hybrid DFT:
        # def2/J + RIJCOSX (RI-J Coulomb fit + COSX exchange), TightSCF and DefGrid3. AIM/NPA are
        # NOT ORCA simple-input keywords (QTAIM/NPA are post-processing on the density/.gbw via
        # Multiwfn / NBO·JANPA), so they are intentionally excluded.
        orca_print_flags=("P_Hirshfeld", "P_Mayer", "P_OrbPopMO_L"),
        orca_simple_keywords=("def2/J", "RIJCOSX", "TightSCF", "DefGrid3", "CHELPG"),
        gaussian_pop=("Hirshfeld", "CHELPG"),
        description="All-in-one: the full analysis property set plus the recommended ORCA "
                    "accuracy/efficiency block (def2/J, RIJCOSX, TightSCF, DefGrid3). Note: "
                    "RIJCOSX is an RI approximation, so energies differ slightly from a "
                    "non-RIJCOSX reference.",
    ),
}


def get_preset(name: str) -> CalculationPreset:
    """Look up a preset by name; raise ValueError listing the choices if unknown."""
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown preset '{name}'; choose one of {sorted(PRESETS)}")


def _merge_keywords(existing: str, extra: Tuple[str, ...]) -> str:
    """Append *extra* keywords to a space-separated string, skipping duplicates (case-insensitive)."""
    present = {tok.lower() for tok in existing.split()}
    added = [kw for kw in extra if kw.lower() not in present]
    return " ".join(([existing] if existing else []) + added)


def apply_preset(params: CalcParams, name: str, *, engine: str = "orca") -> CalcParams:
    """Return a copy of *params* with the named preset's output requests merged in.

    For ORCA the print flags accumulate in ``orca_output_flags`` and the simple keywords
    in ``extra_orca``; for Gaussian the population keywords are folded into ``extra_gaussian``
    as ``pop=(...)``. ``none`` returns *params* unchanged.
    """
    preset = get_preset(name)
    if preset.name == "none":
        return params

    if engine == "gaussian":
        if not preset.gaussian_pop:
            return params
        # _merge_keywords compares whole tokens, so an existing "pop=hirshfeld" does not look
        # like a duplicate of "pop=(hirshfeld,mbs)" and Gaussian would receive the keyword twice.
        # Fold both sets into one pop=(...) instead.
        wanted = list(preset.gaussian_pop)
        kept = []
        for token in params.extra_gaussian.split():
            if token.lower().startswith("pop="):
                inner = token.split("=", 1)[1].strip("()")
                wanted = [w for w in wanted if w.lower() not in
                          {p.strip().lower() for p in inner.split(",")}]
                wanted = [p.strip() for p in inner.split(",") if p.strip()] + wanted
            else:
                kept.append(token)
        pop_kw = f"pop=({','.join(wanted)})"
        return replace(params, extra_gaussian=" ".join(kept + [pop_kw]).strip())

    flags = tuple(dict.fromkeys(params.orca_output_flags + preset.orca_print_flags))
    return replace(
        params,
        orca_output_flags=flags,
        extra_orca=_merge_keywords(params.extra_orca, preset.orca_simple_keywords),
    )
