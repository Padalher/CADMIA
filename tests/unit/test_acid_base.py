"""Tests for cadmia.cdft.acid_base — Lewis acid/base reading from CDFT descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cadmia.cdft.acid_base import (
    AMPHOTERIC,
    LEWIS_ACID,
    LEWIS_BASE,
    UNKNOWN,
    classify_acid_base,
    summarize,
)


def test_classify_acid_when_dual_positive():
    assert classify_acid_base(0.30) == LEWIS_ACID

def test_classify_base_when_dual_negative():
    assert classify_acid_base(-0.30) == LEWIS_BASE

def test_classify_amphoteric_below_threshold():
    assert classify_acid_base(0.001) == AMPHOTERIC
    assert classify_acid_base(-0.001) == AMPHOTERIC

def test_classify_none_is_unknown():
    assert classify_acid_base(None) == UNKNOWN

def test_classify_threshold_boundary_strict():
    # |Δf| exactly at threshold is NOT amphoteric (strict <), matching the CDFT site rule
    assert classify_acid_base(0.02) == LEWIS_ACID
    assert classify_acid_base(-0.02) == LEWIS_BASE

def test_custom_threshold():
    assert classify_acid_base(0.05, threshold=0.1) == AMPHOTERIC


@dataclass
class _Site:
    atom_index: int
    atom_symbol: str
    fukui_plus: Optional[float]
    fukui_minus: Optional[float]
    delta_f: Optional[float]
    acid_base: str = ""
    cn: Optional[int] = None
    ecn: Optional[float] = None


def _sites():
    return [
        _Site(5, "Al", 0.35, 0.11, 0.24, LEWIS_ACID),   # strong acid
        _Site(7, "Al", 0.20, 0.08, 0.12, LEWIS_ACID),   # weaker acid
        _Site(2, "O", 0.07, 0.32, -0.25, LEWIS_BASE),   # strong base
        _Site(3, "O", 0.05, 0.12, -0.07, LEWIS_BASE),   # weaker base
        _Site(1, "Ti", 0.10, 0.10, 0.0, AMPHOTERIC),
    ]


def test_summarize_picks_strongest_acid_by_fplus():
    s = summarize(_sites(), hardness_ev=7.74)
    assert s["n_acid_sites"] == 2
    assert s["strongest_acid_site"]["atom_index"] == 5
    assert s["strongest_acid_site"]["atom_symbol"] == "Al"
    assert s["strongest_acid_site"]["strength"] == 0.35

def test_summarize_picks_strongest_base_by_fminus():
    s = summarize(_sites(), hardness_ev=7.74)
    assert s["n_base_sites"] == 2
    assert s["strongest_base_site"]["atom_index"] == 2
    assert s["strongest_base_site"]["atom_symbol"] == "O"
    assert s["strongest_base_site"]["strength"] == 0.32

def test_summarize_hsab_note_includes_hardness():
    s = summarize(_sites(), hardness_ev=7.74)
    assert s["hardness_eV"] == 7.74
    assert "7.74" in s["hsab_note"] and "HSAB" in s["hsab_note"]

def test_summarize_without_hardness():
    s = summarize(_sites())
    assert s["hardness_eV"] is None
    assert "unavailable" in s["hsab_note"].lower()

def test_summarize_falls_back_to_delta_f_when_label_absent():
    sites = [_Site(5, "Al", 0.35, 0.11, 0.24), _Site(2, "O", 0.07, 0.32, -0.25)]
    s = summarize(sites)
    assert s["n_acid_sites"] == 1 and s["n_base_sites"] == 1

def test_summarize_empty():
    s = summarize([])
    assert s["n_acid_sites"] == 0 and s["n_base_sites"] == 0
    assert s["strongest_acid_site"] is None and s["strongest_base_site"] is None


# ── coordination refinement (low ECN → stronger Lewis acid) ──

def _sites_with_coordination():
    return [
        _Site(5, "Al", 0.35, 0.11, 0.24, LEWIS_ACID, cn=3, ecn=2.5),   # undercoordinated Al
        _Site(7, "Al", 0.20, 0.08, 0.12, LEWIS_ACID, cn=6, ecn=5.5),   # saturated Al
        _Site(2, "O", 0.07, 0.32, -0.25, LEWIS_BASE, cn=2, ecn=2.0),
    ]


def test_entry_includes_cn_ecn():
    s = summarize(_sites_with_coordination())
    acid = s["strongest_acid_site"]
    assert acid["atom_index"] == 5 and acid["cn"] == 3 and acid["ecn"] == 2.5

def test_coordination_note_flags_unsaturated():
    s = summarize(_sites_with_coordination())
    note = s["acid_coordination_note"]
    # Al(5) ECN 2.5 is below the Al average (4.0) → flagged as stronger Lewis acid
    assert "unsaturated" in note and "stronger Lewis acid" in note

def test_coordination_note_near_average():
    sites = [
        _Site(5, "Al", 0.35, 0.11, 0.24, LEWIS_ACID, cn=5, ecn=3.0),
        _Site(7, "Al", 0.34, 0.11, 0.24, LEWIS_ACID, cn=5, ecn=3.0),
    ]
    note = summarize(sites)["acid_coordination_note"]
    assert "≈" in note and "unsaturated" not in note

def test_coordination_note_single_atom_of_element():
    sites = [_Site(5, "Al", 0.35, 0.11, 0.24, LEWIS_ACID, cn=4, ecn=3.7)]
    note = summarize(sites)["acid_coordination_note"]
    assert "ECN = 3.7" in note

def test_no_coordination_note_without_ecn():
    s = summarize(_sites())  # the plain sites have no cn/ecn
    assert "acid_coordination_note" not in s
