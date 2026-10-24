"""Lewis acid/base reading of reactive sites from CDFT descriptors.

A Lewis acid accepts an electron pair → high f+ and dual descriptor Δf > 0
(susceptible to nucleophilic attack). A Lewis base donates one → high f- and
Δf < 0. Hardness η places the species on the HSAB hard/soft axis. These labels
are a chemistry-facing reading of descriptors the CDFT workflow already produces;
no new SCF data is introduced. See docs/descriptor_dictionary.md §4.
"""
from __future__ import annotations

from typing import Iterable, Optional

LEWIS_ACID = "lewis_acid"
LEWIS_BASE = "lewis_base"
AMPHOTERIC = "amphoteric"
DUAL_ROLE = "dual_role"
UNKNOWN = "unknown"

# |Δf| below this → no dominant acid/base character, for a system of _REFERENCE_ATOMS atoms.
DEFAULT_THRESHOLD = 0.02
# f sums to 1 over all atoms, so the mean per atom is 1/N and an absolute cut means a
# different thing in a cluster of 8 than in one of 32: 0.02 is 16 % of the mean in the first
# and 64 % in the second. The threshold is therefore expressed relative to that mean, with
# 0.02 kept as the value it takes for an 8-atom system and as an absolute floor below which
# nothing is worth calling a character.
_REFERENCE_ATOMS = 8


def scaled_threshold(n_atoms: Optional[int], base: float = DEFAULT_THRESHOLD) -> float:
    """The |Δf| cut for a system of *n_atoms*, scaled to its mean Fukui per atom."""
    if not n_atoms or n_atoms <= 0:
        return base
    return max(base * _REFERENCE_ATOMS / n_atoms, base / 4.0)


def classify_acid_base(delta_f: Optional[float], threshold: float = DEFAULT_THRESHOLD,
                       fukui_plus: Optional[float] = None,
                       fukui_minus: Optional[float] = None) -> str:
    """Label a site from its dual descriptor Δf = f+ − f−.

    Δf is a RELATIVE quantity — does this atom accept more than it donates? — so a small |Δf|
    has two very different meanings. An oxygen with f+ ≈ f− ≈ 0.10 exchanges a lot of charge in
    both directions; one with f+ ≈ f− ≈ 0 exchanges none.
    When the two Fukui functions are supplied they are told apart: `dual_role` for the first,
    `amphoteric` for the second.

    A negative f+ cannot make an acid, nor a negative f− a base: condensed Fukui functions can
    come out slightly negative, and a site with f+ = −0.03 and f− = −0.08 has Δf = +0.05 and was
    being labelled a Lewis acid with no electron-accepting power at all.
    """
    if delta_f is None:
        return UNKNOWN
    if abs(delta_f) >= threshold:
        if delta_f > 0:
            return LEWIS_ACID if (fukui_plus is None or fukui_plus > 0) else UNKNOWN
        return LEWIS_BASE if (fukui_minus is None or fukui_minus > 0) else UNKNOWN
    if fukui_plus is not None and fukui_minus is not None:
        if min(fukui_plus, fukui_minus) >= threshold:
            return DUAL_ROLE
    return AMPHOTERIC


def _hsab_note(hardness_ev: Optional[float]) -> str:
    if hardness_ev is None:
        return "Hardness unavailable; HSAB hard/soft matching not assessed."
    return (
        f"Global hardness η = {hardness_ev:.2f} eV. HSAB is comparative: match this "
        f"against the probe/reactant hardness (hard prefers hard, soft prefers soft); "
        f"a larger η means a harder, less polarizable species."
    )


def summarize(sites: Iterable, hardness_ev: Optional[float] = None) -> dict:
    """Headline acid/base reading: strongest Lewis acid and base sites + HSAB context.

    ``sites`` is any iterable of objects exposing ``atom_index``, ``atom_symbol``,
    ``fukui_plus``, ``fukui_minus``, ``delta_f`` and (optionally) ``acid_base``.
    Acid strength is read from f+ (electron-accepting power), base strength from
    f- (electron-donating power).
    """
    sites = list(sites)
    acids, bases = [], []
    for s in sites:
        label = getattr(s, "acid_base", None) or classify_acid_base(
            getattr(s, "delta_f", None),
            fukui_plus=getattr(s, "fukui_plus", None),
            fukui_minus=getattr(s, "fukui_minus", None))
        if label == LEWIS_ACID:
            acids.append(s)
        elif label == LEWIS_BASE:
            bases.append(s)

    def _entry(s, strength) -> dict:
        e = {
            "atom_index": s.atom_index,
            "atom_symbol": s.atom_symbol,
            "strength": round(strength, 6) if strength is not None else None,
            "delta_f": round(s.delta_f, 6) if s.delta_f is not None else None,
        }
        cn = getattr(s, "cn", None)
        ecn = getattr(s, "ecn", None)
        if cn is not None:
            e["cn"] = cn
        if ecn is not None:
            e["ecn"] = round(ecn, 4)
        return e

    top_acid = max(acids, key=lambda s: (s.fukui_plus or 0.0), default=None)
    top_base = max(bases, key=lambda s: (s.fukui_minus or 0.0), default=None)

    summary = {
        "n_acid_sites": len(acids),
        "n_base_sites": len(bases),
        "strongest_acid_site": _entry(top_acid, top_acid.fukui_plus) if top_acid else None,
        "strongest_base_site": _entry(top_base, top_base.fukui_minus) if top_base else None,
        "hardness_eV": round(hardness_ev, 6) if hardness_ev is not None else None,
        "hsab_note": _hsab_note(hardness_ev),
    }
    note = _coordination_note(top_acid, sites)
    if note:
        summary["acid_coordination_note"] = note
    return summary


# An undercoordinated metal cation is a stronger Lewis acid. We judge "under-
# coordinated" relative to the mean ECN of the same element in the same cluster
# (data-driven, not an absolute cutoff). The 0.1-ECN margin is a presentation
# tolerance so near-equal coordination is not flagged as unsaturated.
_ECN_MARGIN = 0.1


def _coordination_note(site, sites) -> Optional[str]:
    if site is None:
        return None
    ecn = getattr(site, "ecn", None)
    if ecn is None:
        return None
    el = site.atom_symbol
    same = [getattr(s, "ecn", None) for s in sites
            if s.atom_symbol == el and getattr(s, "ecn", None) is not None]
    if len(same) < 2:
        return f"{el}({site.atom_index}) ECN = {ecn:.1f}."
    mean = sum(same) / len(same)
    if ecn < mean - _ECN_MARGIN:
        return (f"{el}({site.atom_index}) is coordinatively unsaturated "
                f"(ECN {ecn:.1f} < {mean:.1f} average for {el}) → stronger Lewis acid.")
    return f"{el}({site.atom_index}) ECN {ecn:.1f} ≈ {mean:.1f} average for {el}."
