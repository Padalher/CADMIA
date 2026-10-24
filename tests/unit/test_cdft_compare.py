"""Reproducible CADMIA vs Multiwfn comparison (independence evidence).

Runs CADMIA's native CDFT workflow on the 1tial fixtures and lines it up against the Multiwfn
reference report (CDFT.txt). Asserts agreement within a tight tolerance and regenerates the
Markdown comparison table at docs/cdft_vs_multiwfn.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.cdft.compare import comparison_markdown, compare_to_multiwfn
from cadmia.cdft.workflow import CDFTInputState, run_cdft_workflow

BASE = Path(__file__).parents[1] / "fixtures"
MW_REPORT = BASE / "1tial_cdft" / "CDFT.txt"

# Hirshfeld charges from Multiwfn (same as test_cdft_validation; 8 atoms Ti O O O Al O Al O)
Q_N = [0.7399, -0.4274, -0.4476, -0.4475, 0.7065, -0.3911, 0.7065, -0.4393]
Q_NP1 = [0.6351, -0.4939, -0.4855, -0.4854, 0.3542, -0.4085, 0.3543, -0.4701]
Q_NM1 = [0.9157, -0.1673, -0.3338, -0.3337, 0.8261, -0.3527, 0.8261, -0.3803]


@pytest.fixture(scope="module")
def comparison():
    sn = CDFTInputState.from_wfn(BASE / "N.wfn", charges=Q_N, charge_scheme="hirshfeld")
    snp1 = CDFTInputState.from_wfn(BASE / "N+1.wfn", charges=Q_NP1, charge_scheme="hirshfeld")
    snm1 = CDFTInputState.from_wfn(BASE / "N-1.wfn", charges=Q_NM1, charge_scheme="hirshfeld")
    result = run_cdft_workflow(sn, snp1, snm1)
    return compare_to_multiwfn(result, MW_REPORT)


def test_global_descriptors_match_multiwfn(comparison):
    by_name = {r.name: r for r in comparison.global_rows}
    # IP, EA, η, ω, N all agree with Multiwfn within 0.01 eV
    for name in ("IP (eV)", "EA (eV)", "η (eV)", "ω (eV)", "N (eV)"):
        assert by_name[name].abs_diff is not None
        assert by_name[name].abs_diff < 0.01, f"{name} deviates: {by_name[name]}"


def test_condensed_fukui_match_multiwfn(comparison):
    # every condensed Fukui (f+, f-, Δf) for every atom agrees within 0.001
    assert comparison.fukui_rows
    worst = max(r.abs_diff for r in comparison.fukui_rows if r.abs_diff is not None)
    assert worst < 0.001, f"worst Fukui deviation = {worst}"


def test_overall_max_deviation_small(comparison):
    assert comparison.max_abs_diff < 0.01


def test_published_comparison_is_current(comparison):
    md = comparison_markdown(comparison, title="CADMIA vs Multiwfn — Al2TiO5 (PBE0/def2-TZVP)")
    assert "Maximum absolute deviation" in md
    published = Path(__file__).parents[2] / "docs" / "cdft_vs_multiwfn.md"
    assert published.read_text(encoding="utf-8") == md, "regenerate docs/cdft_vs_multiwfn.md"
