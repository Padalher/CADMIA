"""Tests for calculation presets (analysis output requests) and their effect on inputs."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.calculators.params import CalcParams
from cadmia.calculators.presets import apply_preset, get_preset, PRESETS
from cadmia.calculators.orca import write_inp
from cadmia.calculators.gaussian import write_gjf


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        get_preset("does-not-exist")


def test_none_preset_leaves_params_untouched():
    p = CalcParams(extra_orca="TightSCF")
    out = apply_preset(p, "none", engine="orca")
    assert out is p
    assert out.orca_output_flags == ()


def test_analysis_preset_sets_orca_flags_and_chelpg():
    p = apply_preset(CalcParams(), "analysis", engine="orca")
    assert set(p.orca_output_flags) == {"P_Hirshfeld", "P_Mayer", "P_OrbPopMO_L"}
    assert "CHELPG" in p.extra_orca


def test_cdft_preset_is_charges_only():
    p = apply_preset(CalcParams(), "cdft", engine="orca")
    assert p.orca_output_flags == ("P_Hirshfeld",)
    assert "P_Mayer" not in p.orca_output_flags


def test_apply_preset_does_not_duplicate_existing_keyword():
    p = apply_preset(CalcParams(extra_orca="CHELPG"), "analysis", engine="orca")
    assert p.extra_orca.lower().split().count("chelpg") == 1


def test_orca_input_emits_output_block(tmp_path):
    p = apply_preset(CalcParams(task="sp"), "analysis", engine="orca")
    inp = write_inp(["O", "H", "H"], np.zeros((3, 3)), p, tmp_path / "m.inp")
    text = inp.read_text(encoding="utf-8")
    assert "%output" in text and "Print[P_Mayer] 1" in text and "end" in text
    assert "CHELPG" in text.splitlines()[0]   # simple-input line


def test_orca_input_without_flags_has_no_output_block(tmp_path):
    inp = write_inp(["O", "H", "H"], np.zeros((3, 3)), CalcParams(task="sp"), tmp_path / "m.inp")
    assert "%output" not in inp.read_text(encoding="utf-8")


def test_gaussian_preset_folds_into_pop(tmp_path):
    p = apply_preset(CalcParams(task="sp", dispersion="d3bj"), "analysis", engine="gaussian")
    assert "pop=(" in p.extra_gaussian and "Hirshfeld" in p.extra_gaussian
    gjf = write_gjf(["O", "H", "H"], np.zeros((3, 3)), p, tmp_path / "m.gjf")
    assert "pop=(" in gjf.read_text(encoding="utf-8")
