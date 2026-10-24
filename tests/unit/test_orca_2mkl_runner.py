"""Tests for the orca_2mkl → Molden export driver (command building + gating + rename).

Execution against a real orca_2mkl is validated with a real installation; here we mock it so the
rename/move logic and the gating are exercised offline.
"""
from __future__ import annotations

import subprocess

import pytest

from cadmia.calculators import orca_2mkl_runner as mod
from cadmia.calculators.orca_2mkl_runner import (
    OrcaToMoldenError,
    _build_2mkl_cmd,
    gbw_to_molden,
    orca_2mkl_available,
)


def test_build_cmd_molden():
    cmd = _build_2mkl_cmd("cluster", "molden")
    assert cmd == ["orca_2mkl", "cluster", "-molden"]


def test_build_cmd_custom_path():
    cmd = _build_2mkl_cmd("N", "molden", orca_2mkl_path="/opt/orca/orca_2mkl")
    assert cmd[0] == "/opt/orca/orca_2mkl" and cmd[1] == "N" and cmd[2] == "-molden"


def test_build_cmd_rejects_unknown_format():
    with pytest.raises(OrcaToMoldenError, match="unknown format"):
        _build_2mkl_cmd("x", "banana")


def test_gbw_to_molden_gated_when_missing(tmp_path):
    gbw = tmp_path / "x.gbw"; gbw.write_text("not a real gbw")
    if orca_2mkl_available():
        pytest.skip("orca_2mkl present; gating path not exercised")
    with pytest.raises(OrcaToMoldenError, match="not found on PATH"):
        gbw_to_molden(gbw)


def test_gbw_to_molden_missing_file(tmp_path):
    with pytest.raises(OrcaToMoldenError, match="gbw file not found"):
        gbw_to_molden(tmp_path / "nope.gbw")


def test_gbw_to_molden_renames_input_to_molden(tmp_path, monkeypatch):
    """A successful run moves orca_2mkl's <stem>.molden.input to <stem>.molden."""
    gbw = tmp_path / "cluster.gbw"; gbw.write_text("gbw")

    monkeypatch.setattr(mod.shutil, "which", lambda _p: "/fake/orca_2mkl")

    def fake_run(cmd, **kwargs):
        # orca_2mkl writes <stem>.molden.input next to the .gbw.
        (gbw.parent / "cluster.molden.input").write_text("[Molden Format]\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    res = gbw_to_molden(gbw)
    assert res.molden_path == gbw.with_suffix(".molden")
    assert res.molden_path.exists()
    assert not (gbw.parent / "cluster.molden.input").exists()   # moved, not left behind


def test_gbw_to_molden_custom_output_dir(tmp_path, monkeypatch):
    gbw = tmp_path / "N.gbw"; gbw.write_text("gbw")
    out = tmp_path / "molden" / "N.molden"
    monkeypatch.setattr(mod.shutil, "which", lambda _p: "/fake/orca_2mkl")

    def fake_run(cmd, **kwargs):
        (gbw.parent / "N.molden.input").write_text("x")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    res = gbw_to_molden(gbw, out)
    assert res.molden_path == out and out.exists()


def test_gbw_to_molden_raises_when_no_output_produced(tmp_path, monkeypatch):
    gbw = tmp_path / "N.gbw"; gbw.write_text("gbw")
    monkeypatch.setattr(mod.shutil, "which", lambda _p: "/fake/orca_2mkl")
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    with pytest.raises(OrcaToMoldenError, match="produced no Molden file"):
        gbw_to_molden(gbw)


def test_cadmia_molden_cli_gated_when_missing(tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main
    if orca_2mkl_available():
        pytest.skip("orca_2mkl present; gating path not exercised")
    gbw = tmp_path / "cluster.gbw"; gbw.write_text("x")
    res = CliRunner().invoke(main, ["molden", str(gbw)])
    assert res.exit_code != 0
    assert "orca_2mkl" in res.output and "not found" in res.output


def test_cadmia_molden_cli_requires_files():
    from click.testing import CliRunner
    from cadmia.cli import main
    res = CliRunner().invoke(main, ["molden"])
    assert res.exit_code != 0
    assert "one or more .gbw" in res.output
