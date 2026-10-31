"""Tests for the local (workstation) execution backend — no real engine needed."""
from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

from cadmia.calculators import local_runner
from cadmia.calculators.local_runner import LocalRunnerError, find_engine, run_job
from cadmia.calculators.params import CalcParams
from cadmia.agents.calculation_agent import CalculationResult, run_local

_GAUSSIAN_OK = """\
 SCF Done:  E(RPBE1PBE) =  -100.00000000     A.U. after   10 cycles
 Normal termination of Gaussian 16
"""

# The fake engine is a #!/bin/sh script, which Windows cannot execute.
_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="fake engine is a POSIX shell script")


def _fake_orca(tmp_path, body="echo 'FINAL SINGLE POINT ENERGY -100.0'"):
    """Create a fake ORCA binary big enough to pass the screen-reader check."""
    binary = tmp_path / "bin" / "orca"
    binary.parent.mkdir(exist_ok=True)
    script = f"#!/bin/sh\n{body}\n" + "#" * local_runner._MIN_ENGINE_SIZE
    binary.write_text(script)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


# ─────────────────────────────────────────────
# find_engine
# ─────────────────────────────────────────────

def test_find_engine_explicit_dir(tmp_path):
    binary = _fake_orca(tmp_path)
    assert find_engine("orca", str(binary.parent)) == binary

def test_find_engine_explicit_file(tmp_path):
    binary = _fake_orca(tmp_path)
    assert find_engine("orca", str(binary)) == binary

def test_find_engine_missing_path(tmp_path):
    with pytest.raises(LocalRunnerError, match="not found"):
        find_engine("orca", str(tmp_path / "nope"))

def test_find_engine_rejects_screen_reader(tmp_path, monkeypatch):
    fake = tmp_path / "orca"
    fake.write_text("#!/bin/sh\n")          # tiny → GNOME screen reader heuristic
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr("shutil.which", lambda name: str(fake))
    with pytest.raises(LocalRunnerError, match="screen reader"):
        find_engine("orca")


# ─────────────────────────────────────────────
# run_job
# ─────────────────────────────────────────────

@_posix_only
def test_run_job_orca_writes_output(tmp_path):
    binary = _fake_orca(tmp_path)
    inp = tmp_path / "job.inp"
    inp.write_text("! PBE0\n")
    out = tmp_path / "job.out"
    ok, wall = run_job(inp, "orca", out, engine_path=str(binary))
    assert ok
    assert "FINAL SINGLE POINT ENERGY" in out.read_text(encoding="utf-8")
    assert wall >= 0

@_posix_only
def test_run_job_failure_reports_false(tmp_path):
    binary = _fake_orca(tmp_path, body="exit 1")
    inp = tmp_path / "bad.inp"
    inp.write_text("! PBE0\n")
    ok, _ = run_job(inp, "orca", tmp_path / "bad.out", engine_path=str(binary))
    assert not ok


@_posix_only
def test_run_job_timeout_kills_process_tree(tmp_path):
    """A timeout must reap the engine's whole process group, not just the direct child.

    The fake engine backgrounds a grandchild (as ORCA's %pal MPI ranks would be) that
    writes a marker after a delay; if the group is killed on timeout the marker never
    appears, even though the grandchild was reparented away from the direct child.
    """
    marker = tmp_path / "orphan.marker"
    body = f"( sleep 2; echo alive > '{marker}' ) &\nsleep 2\n"
    binary = _fake_orca(tmp_path, body=body)
    inp = tmp_path / "slow.inp"
    inp.write_text("! PBE0\n")

    ok, _ = run_job(inp, "orca", tmp_path / "slow.out",
                    engine_path=str(binary), timeout_s=1)
    assert not ok
    time.sleep(2.5)                          # past when the orphan would have written
    assert not marker.exists()               # grandchild died with the group


# ─────────────────────────────────────────────
# run_local (agent queue)
# ─────────────────────────────────────────────

def test_run_local_queue_from_inputs(tmp_path, monkeypatch):
    for name in ("a.inp", "b.inp"):
        (tmp_path / name).write_text("! PBE0 def2-TZVP\n")

    def fake_run_job(inp, engine, output, *, engine_path=None, timeout_s=None):
        output.write_text("FINAL SINGLE POINT ENERGY -1.0\n****ORCA TERMINATED NORMALLY****\n")
        return True, 1.0
    monkeypatch.setattr(local_runner, "run_job", fake_run_job)

    result = run_local(tmp_path, engine="orca", output_dir=tmp_path,
                       validate_outputs=False)
    assert result.profile == "local"
    assert [j.status for j in result.jobs] == ["done", "done"]
    assert [j.job_id for j in result.jobs] == ["local-1", "local-2"]
    assert (tmp_path / "cadmia_jobs.json").exists()

def test_run_local_generates_inputs_from_xyz(tmp_path, monkeypatch):
    xyz = tmp_path / "config0.xyz"
    xyz.write_text("2\nconfig0\nTi 0.0 0.0 0.0\nO 1.7 0.0 0.0\n")

    seen = []
    def fake_run_job(inp, engine, output, *, engine_path=None, timeout_s=None):
        seen.append(inp)
        output.write_text("ok\n")
        return True, 1.0
    monkeypatch.setattr(local_runner, "run_job", fake_run_job)

    params = CalcParams(task="opt", nprocs=12, mem_orca_mb_core=3000)
    result = run_local(tmp_path, params, output_dir=tmp_path, validate_outputs=False)
    assert result.jobs[0].status == "done"
    inp_text = seen[0].read_text(encoding="utf-8")
    assert "%pal nprocs 12 end" in inp_text
    assert "%maxcore 3000" in inp_text
    assert "PBE0" in inp_text and "def2-TZVP" in inp_text and "D4" in inp_text

def test_run_local_stops_queue_when_engine_missing(tmp_path, monkeypatch):
    for name in ("a.inp", "b.inp"):
        (tmp_path / name).write_text("! PBE0\n")
    def no_engine(*a, **k):
        raise LocalRunnerError("'orca' not found in PATH")
    monkeypatch.setattr(local_runner, "run_job", no_engine)

    result = run_local(tmp_path, engine="orca", output_dir=tmp_path)
    assert result.jobs[0].status == "failed"
    assert len(result.jobs) == 1            # queue stopped, b.inp never attempted
    assert any("not found" in w for w in result.warnings)

def test_run_local_continues_after_failed_job(tmp_path, monkeypatch):
    for name in ("a.inp", "b.inp"):
        (tmp_path / name).write_text("! PBE0\n")
    def flaky(inp, engine, output, *, engine_path=None, timeout_s=None):
        output.write_text("boom\n")
        return (inp.stem != "a"), 1.0
    monkeypatch.setattr(local_runner, "run_job", flaky)

    result = run_local(tmp_path, engine="orca", output_dir=tmp_path,
                       validate_outputs=False)
    assert [j.status for j in result.jobs] == ["failed", "done"]
    assert result.n_failed == 1


def _counting_run_job(calls):
    """A fake run_job that records the stems it is asked to run and writes a valid output."""
    def fake(inp, engine, output, *, engine_path=None, timeout_s=None):
        calls.append(inp.stem)
        output.write_text("FINAL SINGLE POINT ENERGY -1.0\n"
                          "****ORCA TERMINATED NORMALLY****\n")
        return True, 1.0
    return fake


def test_run_local_resume_skips_completed_jobs(tmp_path, monkeypatch):
    (tmp_path / "a.inp").write_text("! PBE0\n")
    calls: list[str] = []
    monkeypatch.setattr(local_runner, "run_job", _counting_run_job(calls))

    run_local(tmp_path, engine="orca", output_dir=tmp_path, validate_outputs=False)
    assert calls == ["a"]

    result = run_local(tmp_path, engine="orca", output_dir=tmp_path, validate_outputs=False)
    assert calls == ["a"]                           # not recomputed
    assert [j.status for j in result.jobs] == ["done"]
    assert any("already computed" in w for w in result.warnings)


def test_run_local_overwrite_recomputes(tmp_path, monkeypatch):
    (tmp_path / "a.inp").write_text("! PBE0\n")
    calls: list[str] = []
    monkeypatch.setattr(local_runner, "run_job", _counting_run_job(calls))

    run_local(tmp_path, engine="orca", output_dir=tmp_path, validate_outputs=False)
    run_local(tmp_path, engine="orca", output_dir=tmp_path, validate_outputs=False,
              overwrite=True)
    assert calls == ["a", "a"]                      # forced recompute


def test_run_local_resume_recomputes_when_output_missing(tmp_path, monkeypatch):
    (tmp_path / "a.inp").write_text("! PBE0\n")
    calls: list[str] = []
    monkeypatch.setattr(local_runner, "run_job", _counting_run_job(calls))

    run_local(tmp_path, engine="orca", output_dir=tmp_path, validate_outputs=False)
    (tmp_path / "a.out").unlink()                   # output lost after the run
    run_local(tmp_path, engine="orca", output_dir=tmp_path, validate_outputs=False)
    assert calls == ["a", "a"]                      # missing output → recompute


def test_calculation_result_load_rejects_corrupt_state(tmp_path):
    corrupt = tmp_path / "cadmia_jobs.json"
    corrupt.write_text('{"profile": "local", "jobs":')
    with pytest.raises(ValueError, match="cadmia_jobs.json"):
        CalculationResult.load(corrupt)


def test_cli_run_command(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main

    (tmp_path / "a.inp").write_text("! PBE0\n")
    def fake_run_job(inp, engine, output, *, engine_path=None, timeout_s=None):
        output.write_text("FINAL SINGLE POINT ENERGY -1.0\n")
        return True, 1.0
    monkeypatch.setattr(local_runner, "run_job", fake_run_job)

    db = tmp_path / "run.db"
    result = CliRunner().invoke(main, [
        "run", "--jobs", str(tmp_path), "--engine", "orca",
        "-o", str(tmp_path), "--db", "--db-path", str(db),
    ])
    assert result.exit_code == 0, result.output
    assert "Persisted" in result.output
    import sqlite3
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM hpc_jobs WHERE profile='local'").fetchone()[0] == 1
    conn.close()
