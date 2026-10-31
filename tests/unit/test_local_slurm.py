"""Tests for the local SLURM backend: local profiles, conditional modules, runner."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cadmia.config import HPCProfile
from cadmia.calculators.params import CalcParams
from cadmia.calculators.slurm import write_slurm_script
from cadmia.calculators.local_slurm_runner import LocalSlurmRunner
from cadmia.calculators.hpc_runner import HPCRunner, HPCConnectionError, marker_status
from cadmia.agents.calculation_agent import _make_runner


# ─────────────────────────────────────────────
# Local profiles
# ─────────────────────────────────────────────

def test_local_profile_allows_empty_host_user():
    p = HPCProfile(name="ws", host="", user="", work_dir="/scratch/ws", local=True)
    assert p.local is True
    assert p.work_dir == "/scratch/ws"


def test_local_profile_still_requires_work_dir():
    with pytest.raises(ValueError, match="work_dir must not be empty"):
        HPCProfile(name="ws", host="", user="", work_dir="", local=True)


def test_remote_profile_still_requires_host():
    with pytest.raises(ValueError, match="host must not be empty"):
        HPCProfile(name="c", host="", user="u", work_dir="/w")


def test_local_profile_roundtrip(tmp_path):
    p = HPCProfile(name="ws", host="", user="", work_dir="/scratch/ws",
                   local=True, orca_path="/opt/orca6", orca_module="", gaussian_module="")
    p.save(tmp_path)
    loaded = HPCProfile.load("ws", directory=tmp_path)
    assert loaded.local is True
    assert loaded.orca_path == "/opt/orca6"


def test_old_profile_without_local_defaults_false(tmp_path):
    # A YAML written before the 'local' field existed must still load.
    (tmp_path / "legacy.yaml").write_text(
        "name: legacy\nhost: h\nuser: u\nwork_dir: /w\npartition: general\n",
        encoding="utf-8",
    )
    loaded = HPCProfile.load("legacy", directory=tmp_path)
    assert loaded.local is False


# ─────────────────────────────────────────────
# Conditional `module load`
# ─────────────────────────────────────────────

@pytest.fixture
def params():
    return CalcParams(nprocs=4, mem_gaussian_gb=8, mem_orca_mb_core=1000)


def test_slurm_omits_orca_module_when_empty(params):
    profile = HPCProfile(name="ws", host="", user="", work_dir="/w",
                         local=True, orca_module="", orca_path="/opt/orca6")
    s = write_slurm_script("/w/in.inp", "/w/in.out", "orca", params, profile)
    assert "module load" not in s
    assert "export PATH=/opt/orca6:$PATH" in s
    assert "/opt/orca6/orca /w/in.inp > /w/in.out" in s


def test_slurm_omits_gaussian_module_when_empty(params):
    profile = HPCProfile(name="ws", host="", user="", work_dir="/w",
                         local=True, gaussian_module="")
    s = write_slurm_script("/w/in.gjf", "/w/in.log", "gaussian", params, profile)
    assert "module load" not in s
    assert "g16 < /w/in.gjf > /w/in.log" in s


def test_slurm_keeps_module_when_set(params):
    profile = HPCProfile(name="c", host="h", user="u", work_dir="/w",
                         orca_module="orca/5.0.4")
    s = write_slurm_script("/w/in.inp", "/w/in.out", "orca", params, profile)
    assert "module load orca/5.0.4" in s


# ─────────────────────────────────────────────
# Runner selection
# ─────────────────────────────────────────────

def test_make_runner_local():
    p = HPCProfile(name="ws", host="", user="", work_dir="/w", local=True)
    assert isinstance(_make_runner(p), LocalSlurmRunner)


def test_make_runner_remote():
    p = HPCProfile(name="c", host="h", user="u", work_dir="/w")
    assert isinstance(_make_runner(p), HPCRunner)


# ─────────────────────────────────────────────
# LocalSlurmRunner
# ─────────────────────────────────────────────

@pytest.fixture
def local_profile():
    return HPCProfile(name="ws", host="", user="", work_dir="/scratch/ws",
                      local=True, orca_module="", orca_path="/opt/orca6")


def test_local_ping_checks_sbatch(local_profile):
    runner = LocalSlurmRunner(local_profile)
    with patch("shutil.which", return_value="/usr/bin/sbatch"):
        assert runner.ping() is True
    with patch("shutil.which", return_value=None):
        assert runner.ping() is False


def test_local_submit_writes_script_and_parses_jobid(tmp_path):
    profile = HPCProfile(name="ws", host="", user="", work_dir=str(tmp_path),
                         local=True, orca_module="", orca_path="/opt/orca6")
    runner = LocalSlurmRunner(profile)
    inp = tmp_path / "mol.inp"
    inp.write_text("! PBE0 def2-SVP SP\n")
    params = CalcParams(nprocs=4, mem_orca_mb_core=1000)

    mock = MagicMock(returncode=0, stdout="Submitted batch job 4242", stderr="")
    with patch("subprocess.run", return_value=mock):
        job_id, out_path = runner.submit(inp, "orca", params, label="mol")

    assert job_id == "4242"
    work = tmp_path / "mol"
    assert (work / "mol.inp").exists()        # input staged
    assert (work / "mol.sh").exists()          # SLURM script written
    assert out_path == str(work / "mol.out")
    script = (work / "mol.sh").read_text(encoding="utf-8")
    assert "module load" not in script


def test_local_submit_raises_on_sbatch_failure(tmp_path):
    profile = HPCProfile(name="ws", host="", user="", work_dir=str(tmp_path),
                         local=True, orca_module="")
    runner = LocalSlurmRunner(profile)
    inp = tmp_path / "mol.inp"
    inp.write_text("! PBE0\n")
    mock = MagicMock(returncode=1, stdout="", stderr="sbatch: error: partition")
    with patch("subprocess.run", return_value=mock):
        with pytest.raises(HPCConnectionError, match="sbatch failed"):
            runner.submit(inp, "orca", CalcParams(), label="mol")


def test_local_check_status_maps_state(local_profile):
    runner = LocalSlurmRunner(local_profile)
    mock = MagicMock(returncode=0, stdout="RUNNING", stderr="")
    with patch("subprocess.run", return_value=mock):
        assert runner.check_status("4242") == "running"


# ─────────────────────────────────────────────
# Completion marker fallback (SLURM without accounting)
# ─────────────────────────────────────────────

def test_marker_status_parsing():
    assert marker_status("...\nORCA TERMINATED NORMALLY\nCADMIA_DONE:0\n") == "done"
    assert marker_status("boom\nCADMIA_DONE:1\n") == "failed"
    assert marker_status("no marker here") is None


def test_local_check_status_uses_marker_when_no_squeue_no_sacct(tmp_path, local_profile):
    runner = LocalSlurmRunner(local_profile)
    out = tmp_path / "h2o.out"
    out.write_text("ORCA TERMINATED NORMALLY\nCADMIA_DONE:0\n")
    empty = MagicMock(returncode=0, stdout="", stderr="")   # squeue + sacct both empty
    with patch("subprocess.run", return_value=empty):
        assert runner.check_status("1", str(out)) == "done"


def test_local_check_status_marker_failed(tmp_path, local_profile):
    runner = LocalSlurmRunner(local_profile)
    out = tmp_path / "h2o.out"
    out.write_text("some error\nCADMIA_DONE:1\n")
    empty = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=empty):
        assert runner.check_status("1", str(out)) == "failed"


def test_local_check_status_unknown_when_no_marker(tmp_path, local_profile):
    runner = LocalSlurmRunner(local_profile)
    out = tmp_path / "h2o.out"
    out.write_text("running, no marker yet\n")
    empty = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=empty):
        assert runner.check_status("1", str(out)) == "unknown"


def test_local_download_copies_file(tmp_path, local_profile):
    runner = LocalSlurmRunner(local_profile)
    src = tmp_path / "src.out"
    src.write_text("RESULT")
    dst = tmp_path / "sub" / "dst.out"
    runner.download(str(src), dst)
    assert dst.read_text(encoding="utf-8") == "RESULT"


def test_local_download_noop_when_same_path(tmp_path, local_profile):
    runner = LocalSlurmRunner(local_profile)
    f = tmp_path / "x.out"
    f.write_text("R")
    runner.download(str(f), f)   # must not raise or truncate
    assert f.read_text(encoding="utf-8") == "R"


def test_local_download_missing_raises(tmp_path, local_profile):
    runner = LocalSlurmRunner(local_profile)
    with pytest.raises(HPCConnectionError, match="output not found"):
        runner.download(str(tmp_path / "ghost.out"), tmp_path / "dst.out")


def test_script_exits_with_the_engine_status_not_the_tidy_loop(params):
    """The tidy block must not decide whether the job succeeded.

    A bash script exits with the status of its last command, and the tidy loop ends in
    `[ -f X.xyz ] && mv ...`, which returns 1 when the file is absent. ORCA only writes
    <stem>.xyz for an optimization, so without an explicit exit every single point and every
    frequency job finished with status 1 and SLURM recorded it as FAILED — which check_status
    reads from sacct before it ever looks at the CADMIA_DONE marker.
    """
    profile = HPCProfile(name="ws", host="", user="", work_dir="/w", local=True,
                         orca_path="/opt/orca6")
    s = write_slurm_script("/w/in.inp", "/w/in.out", "orca", params, profile, tidy=True)

    body = [line for line in s.splitlines() if line.strip()]
    assert body[-1].strip() == "exit $_rc"
    assert 'echo "CADMIA_DONE:$_rc"' in s            # the marker reports the engine too
    # The status is captured right after the engine runs, before anything can overwrite $?.
    assert s.index("_rc=$?") < s.index("for _ext in")


def test_tidy_can_still_be_turned_off(params):
    profile = HPCProfile(name="ws", host="", user="", work_dir="/w", local=True)
    s = write_slurm_script("/w/in.inp", "/w/in.out", "orca", params, profile, tidy=False)
    assert "for _ext in" not in s
    assert s.splitlines()[-1].strip() == "exit $_rc"
