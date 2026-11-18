"""Tests for the orca_plot driver (script building + gating). Execution needs a real ORCA installation and is not run here."""
from __future__ import annotations

import pytest

from cadmia.calculators.orca_plot_runner import (
    OrcaPlotError,
    _build_plot_script,
    _parse_boundaries,
    ngrid_for_step,
    orca_plot_available,
    run_orca_plot,
)


def test_build_script_density():
    s = _build_plot_script("density", ngrid=60)
    lines = s.strip().splitlines()
    assert lines[0] == "4" and lines[1] == "60"
    assert lines[2] == "1" and lines[3] == "2"
    assert "y" in lines
    assert lines[-2] == "11" and lines[-1] == "12"
    assert "5" not in lines


def test_build_script_per_axis_grid():
    lines = _build_plot_script("density", ngrid=(118, 101, 88)).strip().splitlines()
    assert lines[0] == "4" and lines[1] == "118 101 88"


_MENU_DUMP = """
Resolution     ... 40 40 40
Boundaries     ...    -7.000000     7.000000 (x direction)
                      -7.000000     7.000000 (y direction)
                      -7.000000     8.738548 (z direction)
       4 - Enter number of grid intervals
"""


def test_parse_boundaries_reads_three_axes():
    b = _parse_boundaries(_MENU_DUMP)
    assert b is not None
    assert b["x"] == (-7.0, 7.0) and b["z"] == (-7.0, 8.738548)


def test_parse_boundaries_none_when_incomplete():
    assert _parse_boundaries("Boundaries ... -7.0 7.0 (x direction)\n") is None


def test_ngrid_for_step_uniform_on_noncubic_box():
    b = _parse_boundaries(_MENU_DUMP)
    nx, ny, nz = ngrid_for_step(b, 0.20)
    assert (nx, ny, nz) == (70, 70, 79)
    for lo_hi, n in ((b["x"], nx), (b["z"], nz)):
        step = (lo_hi[1] - lo_hi[0]) / n
        assert abs(step - 0.20) < 0.01


def test_ngrid_for_step_rejects_nonpositive():
    b = _parse_boundaries(_MENU_DUMP)
    with pytest.raises(OrcaPlotError, match="step must be"):
        ngrid_for_step(b, 0.0)


def test_build_script_density_y_before_generate():
    lines = _build_plot_script("density", ngrid=40).strip().splitlines()
    assert lines.index("y") < lines.index("11")


def test_build_script_mep_uses_esp_code():
    lines = _build_plot_script("mep", ngrid=40, density_name="n.scfp").strip().splitlines()
    assert "43" in lines and "6" not in lines


def test_build_script_mep_names_the_density_instead_of_answering_yes():
    lines = _build_plot_script("mep", ngrid=40, density_name="n.scfp").strip().splitlines()
    assert "y" not in lines
    assert lines.index("n.scfp") == lines.index("43") + 1
    assert lines.index("n.scfp") < lines.index("11")


def test_build_script_mep_needs_density_name():
    with pytest.raises(OrcaPlotError, match="density_name"):
        _build_plot_script("mep", ngrid=40)


def test_build_script_homo_needs_mo_index():
    with pytest.raises(OrcaPlotError, match="mo_index"):
        _build_plot_script("homo")
    s = _build_plot_script("homo", mo_index=41)
    assert "2" in s.splitlines() and "41" in s.splitlines()


def test_build_script_rejects_unknown_kind():
    with pytest.raises(OrcaPlotError, match="unknown plot kind"):
        _build_plot_script("banana")


def test_run_orca_plot_gated_when_missing(tmp_path):
    gbw = tmp_path / "x.gbw"; gbw.write_text("not a real gbw")
    if orca_plot_available():
        pytest.skip("orca_plot present; gating path not exercised")
    with pytest.raises(OrcaPlotError, match="not found on PATH"):
        run_orca_plot(gbw, "density", tmp_path / "out.cube")


def test_cadmia_density_cli_gated_when_missing(tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main
    if orca_plot_available():
        pytest.skip("orca_plot present; gating path not exercised")
    gbw = tmp_path / "N.gbw"; gbw.write_text("x")
    res = CliRunner().invoke(main, ["density", "--gbw", str(gbw), "-o", str(tmp_path / "cubes")])
    assert res.exit_code != 0
    assert "orca_plot" in res.output and "not found" in res.output


def _gbw(tmp_path, name):
    p = tmp_path / name; p.write_text("x"); return str(p)


@pytest.mark.parametrize("args, msg", [
    (["--adduct", "A", "--cluster", "C"], "all three"),
    (["--adduct", "A", "--cluster", "C", "--probe", "P", "--np1", "N1"], "don't mix"),
    (["--gbw", "G", "--np1", "N1"], "both"),
    (["--np1", "N1", "--nm1", "M1"], "--gbw"),
])
def test_density_cli_mode_inference_errors(monkeypatch, tmp_path, args, msg):
    from click.testing import CliRunner
    from cadmia.cli import main
    monkeypatch.setattr(
        "cadmia.calculators.orca_plot_runner.orca_plot_available", lambda *a, **k: True)
    files = {"A": "ad.gbw", "C": "cl.gbw", "P": "pr.gbw", "G": "N.gbw", "N1": "an.gbw", "M1": "ca.gbw"}
    resolved = [_gbw(tmp_path, files[a]) if a in files else a for a in args]
    res = CliRunner().invoke(main, ["density", *resolved, "-o", str(tmp_path / "cubes")])
    assert res.exit_code != 0 and msg in res.output


def test_an_incompatible_gbw_is_reported_as_such_not_as_a_menu_problem():
    from cadmia.calculators.orca_plot_runner import _orca_plot_complaint

    said = _orca_plot_complaint(
        "WARNING!!!!!!!\nYour GBWFile is either corrupt or from a different ORCA version!\n"
        "Error (TORCAFlags): Unknown HFTYP. THis is a bug\n")
    assert "different ORCA build" in said
    assert "Re-run the single point" in said
    assert "unexpected menu output" not in said


def test_a_silent_orca_plot_still_blames_the_menu():
    from cadmia.calculators.orca_plot_runner import _orca_plot_complaint

    assert "menu" in _orca_plot_complaint("Entering interactive generation of plots ...\n")
