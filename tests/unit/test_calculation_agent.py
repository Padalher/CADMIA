"""Tests for HPC profile management, SLURM script generation, and HPCRunner."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cadmia.calculators.params import CalcParams
from cadmia.calculators.slurm import write_slurm_script
from cadmia.config import HPCProfile, DEFAULT_PROFILES_DIR
from cadmia.agents.calculation_agent import (
    JobRecord,
    CalculationResult,
    _engine_from_input,
    _collect_inputs,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_profile() -> HPCProfile:
    return HPCProfile(
        name="testcluster",
        host="hpc.example.com",
        user="researcher",
        work_dir="/scratch/researcher/cadmia",
        partition="compute",
        account="myproject",
        gaussian_module="gaussian/16",
        orca_module="orca/5.0.4",
        orca_path="/opt/orca/5.0.4",
        max_time="12:00:00",
        nprocs=8 if False else None,   # not a param — just for reference
    )


@pytest.fixture
def sample_profile_clean() -> HPCProfile:
    return HPCProfile(
        name="testcluster",
        host="hpc.example.com",
        user="researcher",
        work_dir="/scratch/researcher/cadmia",
        partition="compute",
        account="myproject",
        gaussian_module="gaussian/16",
        orca_module="orca/5.0.4",
        orca_path="/opt/orca/5.0.4",
        max_time="12:00:00",
    )


@pytest.fixture
def simple_params() -> CalcParams:
    return CalcParams(nprocs=4, mem_gaussian_gb=8, mem_orca_mb_core=1000)


# ─────────────────────────────────────────────
# HPCProfile
# ─────────────────────────────────────────────

def test_profile_creation(sample_profile_clean):
    p = sample_profile_clean
    assert p.name == "testcluster"
    assert p.host == "hpc.example.com"
    assert p.partition == "compute"


def test_profile_invalid_name():
    with pytest.raises(ValueError, match="invalid characters"):
        HPCProfile(name="my cluster!", host="h", user="u", work_dir="/w")


def test_profile_empty_host():
    with pytest.raises(ValueError, match="host must not be empty"):
        HPCProfile(name="ok", host="", user="u", work_dir="/w")


def test_profile_save_load_roundtrip(tmp_path, sample_profile_clean):
    profile = sample_profile_clean
    saved = profile.save(tmp_path)
    assert saved.exists()
    loaded = HPCProfile.load(profile.name, directory=tmp_path)
    assert loaded.name == profile.name
    assert loaded.host == profile.host
    assert loaded.work_dir == profile.work_dir
    assert loaded.orca_path == profile.orca_path


def test_profile_save_is_yaml(tmp_path, sample_profile_clean):
    sample_profile_clean.save(tmp_path)
    content = (tmp_path / "testcluster.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert data["host"] == "hpc.example.com"
    assert data["partition"] == "compute"


def test_profile_load_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Profile 'ghost' not found"):
        HPCProfile.load("ghost", directory=tmp_path)


def test_profile_list_profiles(tmp_path, sample_profile_clean):
    HPCProfile(name="alpha", host="a", user="u", work_dir="/w").save(tmp_path)
    HPCProfile(name="beta",  host="b", user="u", work_dir="/w").save(tmp_path)
    names = HPCProfile.list_profiles(tmp_path)
    assert "alpha" in names
    assert "beta"  in names


def test_profile_list_empty_dir(tmp_path):
    assert HPCProfile.list_profiles(tmp_path) == []


def test_profile_extra_sbatch(tmp_path):
    p = HPCProfile(name="x", host="h", user="u", work_dir="/w",
                   extra_sbatch=["--qos=high", "--mail-type=END"])
    saved = p.save(tmp_path)
    loaded = HPCProfile.load("x", directory=tmp_path)
    assert loaded.extra_sbatch == ["--qos=high", "--mail-type=END"]


# ─────────────────────────────────────────────
# SLURM script generation
# ─────────────────────────────────────────────

@pytest.fixture
def profile_for_slurm():
    return HPCProfile(
        name="test", host="h", user="u",
        work_dir="/scratch/u/cadmia",
        partition="general",
        account="proj",
        gaussian_module="gaussian/16",
        orca_module="orca/5",
        orca_path="/opt/orca/5",
        max_time="24:00:00",
    )


def test_slurm_gaussian_has_shebang(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.gjf", "/work/input.log",
                           "gaussian", simple_params, profile_for_slurm)
    assert s.startswith("#!/bin/bash")


def test_slurm_gaussian_sbatch_directives(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.gjf", "/work/input.log",
                           "gaussian", simple_params, profile_for_slurm)
    assert "#SBATCH --partition=general" in s
    assert "#SBATCH --account=proj" in s
    assert f"#SBATCH --ntasks-per-node={simple_params.nprocs}" in s
    assert "#SBATCH --time=24:00:00" in s


def test_slurm_gaussian_load_module(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.gjf", "/work/input.log",
                           "gaussian", simple_params, profile_for_slurm)
    assert "module load gaussian/16" in s


def test_slurm_gaussian_run_command(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.gjf", "/work/input.log",
                           "gaussian", simple_params, profile_for_slurm)
    assert "g16 < /work/input.gjf > /work/input.log" in s


def test_slurm_gaussian_has_completion_marker(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.gjf", "/work/input.log",
                           "gaussian", simple_params, profile_for_slurm)
    assert "CADMIA_DONE" in s


def test_slurm_orca_run_command(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.inp", "/work/input.out",
                           "orca", simple_params, profile_for_slurm)
    assert "orca /work/input.inp > /work/input.out" in s
    assert "module load orca/5" in s


def test_slurm_orca_path_exported(profile_for_slurm, simple_params):
    s = write_slurm_script("/work/input.inp", "/work/input.out",
                           "orca", simple_params, profile_for_slurm)
    assert "export PATH=/opt/orca/5" in s


def test_slurm_extra_sbatch(simple_params):
    profile = HPCProfile(
        name="x", host="h", user="u", work_dir="/w",
        partition="gpu", extra_sbatch=["--gres=gpu:1", "--qos=express"],
    )
    s = write_slurm_script("/w/in.gjf", "/w/in.log",
                           "gaussian", simple_params, profile)
    assert "#SBATCH --gres=gpu:1" in s
    assert "#SBATCH --qos=express" in s


def test_slurm_no_account_when_empty(simple_params):
    profile = HPCProfile(name="x", host="h", user="u", work_dir="/w", account="")
    s = write_slurm_script("/w/in.gjf", "/w/in.log",
                           "gaussian", simple_params, profile)
    assert "--account=" not in s


def test_slurm_invalid_engine_raises(profile_for_slurm, simple_params):
    with pytest.raises(ValueError, match="engine must be"):
        write_slurm_script("/w/in.gjf", "/w/in.log",
                           "turbomole", simple_params, profile_for_slurm)


# ─────────────────────────────────────────────
# HPCRunner — SSH command construction (mocked)
# ─────────────────────────────────────────────

from cadmia.calculators.hpc_runner import HPCRunner, HPCConnectionError


@pytest.fixture
def runner(sample_profile_clean):
    return HPCRunner(sample_profile_clean)


def test_runner_ssh_base_includes_host(runner):
    cmd = runner._ssh_base()
    assert "researcher@hpc.example.com" in cmd


def test_runner_ssh_base_batch_mode(runner):
    cmd = runner._ssh_base()
    assert "BatchMode=yes" in " ".join(cmd)


def test_runner_ssh_base_custom_port():
    p = HPCProfile(name="x", host="h", user="u", work_dir="/w", ssh_port=2222)
    r = HPCRunner(p)
    cmd = r._ssh_base()
    assert "-p" in cmd
    assert "2222" in cmd


def test_runner_ssh_base_with_key():
    p = HPCProfile(name="x", host="h", user="u", work_dir="/w", ssh_key="/home/u/.ssh/id_rsa")
    r = HPCRunner(p)
    cmd = r._ssh_base()
    assert "-i" in cmd


def test_runner_ping_success(runner):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "cadmia_ping"
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        assert runner.ping() is True


def test_runner_ping_failure(runner):
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 10)):
        assert runner.ping() is False


def test_runner_upload_raises_on_failure(runner, tmp_path):
    local = tmp_path / "input.gjf"
    local.write_text("fake gjf")
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Permission denied"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(HPCConnectionError, match="scp upload failed"):
            runner.upload(local, "/remote/input.gjf")


def test_runner_download_raises_on_failure(runner, tmp_path):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "No such file"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(HPCConnectionError, match="scp download failed"):
            runner.download("/remote/output.log", tmp_path / "output.log")


def test_runner_check_status_maps_slurm_states(runner):
    for slurm_state, expected in [
        ("PENDING", "queued"), ("RUNNING", "running"),
        ("COMPLETED", "done"), ("FAILED", "failed"), ("CANCELLED", "failed"),
    ]:
        mock_result = MagicMock(returncode=0, stdout=slurm_state, stderr="")
        with patch("subprocess.run", return_value=mock_result):
            assert runner.check_status("12345") == expected


def test_runner_check_status_absent_from_queue(runner):
    """When squeue returns empty and sacct says COMPLETED, status is 'done'."""
    def side_effect(cmd, **kwargs):
        m = MagicMock()
        m.returncode = 0
        if "squeue" in " ".join(cmd):
            m.stdout = ""
            m.stderr = ""
        else:
            m.stdout = "COMPLETED"
            m.stderr = ""
        return m
    with patch("subprocess.run", side_effect=side_effect):
        assert runner.check_status("12345") == "done"


def test_runner_list_jobs(runner):
    mock_result = MagicMock(
        returncode=0,
        stdout="12345 cadmia_job1 RUNNING 01:23\n12346 cadmia_job2 PENDING 00:00",
        stderr="",
    )
    with patch("subprocess.run", return_value=mock_result):
        jobs = runner.list_jobs()
    assert len(jobs) == 2
    assert jobs[0]["job_id"] == "12345"
    assert jobs[0]["state"] == "running"
    assert jobs[1]["state"] == "queued"


# ─────────────────────────────────────────────
# _engine_from_input, _collect_inputs
# ─────────────────────────────────────────────

_GAUSSIAN_INPUT = ("%chk=m.chk\n%nprocshared=8\n%mem=16GB\n#PBE0/def2-TZVP Opt\n"
                   "\nm\n\n0 1\nO   0.00   0.00   0.00\n\n")
_ORCA_INPUT = ("! PBE0 def2-TZVP D4 Opt\n\n# m\n\n%pal nprocs 8 end\n"
               "\n* xyz 0 1\nO   0.00   0.00   0.00\n*\n")


def test_engine_from_input_gaussian_named_inp(tmp_path):
    """The reason the engine is read from the content: Gaussian is fed through stdin and
    does not care about the extension, so its inputs are commonly called .inp too — which
    the extension alone cannot tell apart from an ORCA input."""
    f = tmp_path / "mol.inp"
    f.write_text(_GAUSSIAN_INPUT)
    assert _engine_from_input(f) == "gaussian"


def test_engine_from_input_orca_named_inp(tmp_path):
    f = tmp_path / "mol.inp"
    f.write_text(_ORCA_INPUT)
    assert _engine_from_input(f) == "orca"


def test_engine_from_input_gaussian_named_gjf(tmp_path):
    f = tmp_path / "mol.gjf"
    f.write_text(_GAUSSIAN_INPUT)
    assert _engine_from_input(f) == "gaussian"


def test_engine_from_input_orca_leading_comment_is_not_a_route(tmp_path):
    """`#` starts a Gaussian route AND an ORCA comment, so it must never decide alone."""
    f = tmp_path / "c.inp"
    f.write_text("# a comment first\n! PBE0 def2-SVP SP\n\n* xyzfile 0 1 mol.xyz\n")
    assert _engine_from_input(f) == "orca"


def test_engine_from_input_bare_gaussian_route(tmp_path):
    """A Gaussian input with no %-lines at all still resolves from its route card."""
    f = tmp_path / "bare.inp"
    f.write_text("#P B3LYP/6-31G* Opt\n\ntitle\n\n0 1\nO 0.0 0.0 0.0\n\n")
    assert _engine_from_input(f) == "gaussian"


def test_engine_from_input_unrecognised_raises(tmp_path):
    f = tmp_path / "mol.inp"
    f.write_text("just some text, no engine markers\n")
    with pytest.raises(ValueError, match="Cannot detect the engine"):
        _engine_from_input(f)


def test_collect_inputs_single_file(tmp_path):
    f = tmp_path / "mol.gjf"
    f.write_text("gjf")
    result = _collect_inputs(f)
    assert result == [f]


def test_collect_inputs_directory(tmp_path):
    (tmp_path / "a.gjf").write_text("g")
    (tmp_path / "b.inp").write_text("o")
    (tmp_path / "c.txt").write_text("x")
    result = _collect_inputs(tmp_path)
    names = {f.name for f in result}
    assert "a.gjf" in names
    assert "b.inp" in names
    assert "c.txt" not in names


def test_collect_inputs_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No .gjf or .inp files"):
        _collect_inputs(tmp_path)


# ─────────────────────────────────────────────
# CalculationResult serialisation
# ─────────────────────────────────────────────

def test_calculation_result_save_load(tmp_path):
    rec = JobRecord(
        label="mol", engine="gaussian",
        input_path=tmp_path / "mol.gjf",
        remote_input="/r/mol.gjf",
        remote_output="/r/mol.log",
        remote_work_dir="/r",
        job_id="99999",
        status="done",
        output_path=tmp_path / "mol.log",
        validation_ok=True,
        validation_msg="OK",
    )
    cr = CalculationResult(profile="testcluster", jobs=[rec])
    save_path = tmp_path / "jobs.json"
    cr.save(save_path)

    loaded = CalculationResult.load(save_path)
    assert loaded.profile == "testcluster"
    assert len(loaded.jobs) == 1
    assert loaded.jobs[0].job_id == "99999"
    assert loaded.jobs[0].status == "done"
    assert loaded.jobs[0].validation_ok is True


def test_calculation_result_to_json(tmp_path):
    rec = JobRecord(
        label="mol", engine="orca",
        input_path=tmp_path / "mol.inp",
        remote_input="/r/mol.inp",
        remote_output="/r/mol.out",
        remote_work_dir="/r",
        job_id="11111",
        status="running",
    )
    cr = CalculationResult(profile="myhpc", jobs=[rec])
    data = json.loads(cr.to_json())
    assert data["profile"] == "myhpc"
    assert data["n_total"] == 1
    assert data["n_done"] == 0
    assert data["jobs"][0]["engine"] == "orca"


def test_calculation_result_n_done_failed():
    jobs = [
        JobRecord("a", "gaussian", Path("a.gjf"), "", "", "", status="done"),
        JobRecord("b", "gaussian", Path("b.gjf"), "", "", "", status="done"),
        JobRecord("c", "orca",     Path("c.inp"), "", "", "", status="failed"),
    ]
    cr = CalculationResult(profile="x", jobs=jobs)
    assert cr.n_done   == 2
    assert cr.n_failed == 1



def test_input_stems_keep_the_origin_through_rank_skeleton_labels():
    from cadmia.agents.calculation_agent import _input_stems
    from cadmia.core import Structure

    def block(comment):
        return Structure(atoms=["O"], positions=[[0.0, 0.0, 0.0]], labels=["O1"],
                         structure_id="b", source_file="b.xyz", metadata={"comment": comment})

    ref = "rank 1  +0.000 kcal/mol  E=-1.000000 Eh  orca  site3_eta1_p30  [reference]"
    iso = "rank 2  +1.200 kcal/mol  E=-0.998000 Eh  orca  site5_eta2_p0  [isomer-1]"
    source = Path("ranked_by_energy.xyz")
    assert _input_stems(source, [block(ref), block(iso)]) == ["site3_eta1_p30", "site5_eta2_p0"]
    assert _input_stems(source, [block(ref)]) == ["site3_eta1_p30"]
    assert _input_stems(Path("cluster.xyz"), [block("any comment")]) == ["cluster"]
