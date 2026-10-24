"""Tests for cadmia.display terminal rendering (analysis thermochemistry rows)."""
from __future__ import annotations

from cadmia.agents.analysis_agent import AnalysisResult
from cadmia.display import print_analysis_result, _fallback_analysis_print


def _result_with_thermo():
    return AnalysisResult(
        cluster="Al2TiO5", probe="CO2", site_atom_index=6, site_atom_symbol="O",
        e_ads_kcal=-30.000, delta_h_ads_kcal=-28.800,
        delta_g_ads_kcal=-17.500, minus_t_delta_s_kcal=11.300,
    )


def test_fallback_prints_thermochemistry(capsys):
    _fallback_analysis_print(_result_with_thermo())
    out = capsys.readouterr().out
    assert "ΔH_ads" in out and "-28.800" in out
    assert "ΔG_ads" in out and "-17.500" in out
    assert "11.300" in out


def test_rich_table_prints_thermochemistry(capsys):
    print_analysis_result(_result_with_thermo())
    out = capsys.readouterr().out
    assert "H_ads" in out and "G_ads" in out
    assert "-28.800" in out and "-17.500" in out


def test_no_thermo_rows_when_absent(capsys):
    _fallback_analysis_print(AnalysisResult(cluster="Al2TiO5", probe="CO2", e_ads_kcal=-30.0))
    out = capsys.readouterr().out
    assert "ΔG_ads" not in out


def test_e_ads_declares_missing_bsse(capsys):
    """Without a BSSE correction, E_ads is labeled 'no BSSE' — never presented as corrected."""
    _fallback_analysis_print(AnalysisResult(cluster="Al2TiO5", probe="CO2", e_ads_kcal=-30.0))
    assert "no BSSE" in capsys.readouterr().out


def test_e_ads_no_caveat_when_bsse_present(capsys):
    _fallback_analysis_print(
        AnalysisResult(cluster="Al2TiO5", probe="CO2", e_ads_kcal=-30.0, e_ads_bsse_kcal=-27.5))
    out = capsys.readouterr().out
    assert "no BSSE" not in out and "E_ads(BSSE)" in out
