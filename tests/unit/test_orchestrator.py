"""Tests for the Orchestrator Agent (cadmia pipeline) — real engine-free stages,
mocked engine for the calculation/analysis path."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.agents import orchestrator
from cadmia.agents.orchestrator import (
    DONE, BLOCKED, SKIPPED, STAGES, PipelineState, StageState, plan, run_pipeline,
)

_ORCA = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_orca"
_N, _NP1, _NM1 = (_ORCA / "1tial_N_orca.out",
                  _ORCA / "1tial_anion_orca.out",
                  _ORCA / "1tial_cation_orca.out")
_HAS_DATA = _N.exists()
_VALID_ORCA = "FINAL SINGLE POINT ENERGY -1.0\n****ORCA TERMINATED NORMALLY****\n"
_GAUSSIAN = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_cdft"
_needs_data = pytest.mark.skipif(not _HAS_DATA, reason="1tial_orca fixtures not present")


# ─────────────────────────────────────────────
# PipelineState / plan — no data needed
# ─────────────────────────────────────────────

def test_pipeline_state_roundtrip(tmp_path):
    state = PipelineState(cluster="X", probe="NH3", workdir=str(tmp_path))
    state.inputs = {"n": "N.out"}
    state.stages["validate"].status = DONE
    path = state.save()
    loaded = PipelineState.load(path)
    assert loaded.cluster == "X"
    assert loaded.inputs["n"] == "N.out"
    assert loaded.stages["validate"].status == DONE
    assert set(loaded.stages) == set(STAGES)


def test_pipeline_state_load_rejects_corrupt_manifest(tmp_path):
    # A power cut mid-write leaves a truncated manifest; --resume must fail with an
    # actionable error naming the file, not a raw JSONDecodeError.
    manifest = tmp_path / "pipeline.json"
    manifest.write_text('{"cluster": "X", "stages":')
    with pytest.raises(ValueError, match="pipeline.json"):
        PipelineState.load(manifest)

def test_plan_full_run():
    state = PipelineState(cluster="X", probe="NH3", workdir=".")
    assert plan(state, until=None, resume=False) == list(STAGES)

def test_plan_until_stops():
    state = PipelineState(cluster="X", probe="NH3", workdir=".")
    assert plan(state, until="geometry", resume=False) == ["validate", "cdft", "geometry"]

def test_plan_resume_skips_done():
    state = PipelineState(cluster="X", probe="NH3", workdir=".")
    for s in ("validate", "cdft", "geometry"):
        state.stages[s].status = DONE
    assert plan(state, until=None, resume=True) == ["run", "analyze", "stability", "report"]

def test_run_pipeline_rejects_unknown_until(tmp_path):
    with pytest.raises(ValueError, match="Unknown stage"):
        run_pipeline(n="a", np1="b", nm1="c", workdir=tmp_path, until="nope")


def _skipped_first_leg(tmp_path):
    """Manifest of a first run that finished with analyze/stability SKIPPED (no
    probe_free/refs) and a report already built."""
    state = PipelineState(cluster="X", probe="NH3", workdir=str(tmp_path))
    state.inputs = {"n": "N.out", "np1": "A.out", "nm1": "C.out",
                    "refs": [], "probe_free": None}
    for s in ("validate", "cdft", "geometry", "run", "report"):
        state.stages[s].status = DONE
    state.stages["analyze"].status = SKIPPED
    state.stages["stability"].status = SKIPPED
    state.save()
    return state

def test_resume_reopens_skipped_stages_with_new_inputs(tmp_path):
    """Regression: supplying --probe-free/--ref at resume must re-open the stages
    that were SKIPPED for lack of them (and the report), not leave them SKIPPED."""
    _skipped_first_leg(tmp_path)
    state = run_pipeline(n=None, np1=None, nm1=None, workdir=tmp_path, resume=True,
                         probe_free="NH3.out", refs=["TiO2.out", "Al2O3.out"],
                         dry_run=True, on_event=lambda s, m: None)
    assert state.stages["analyze"].status == "pending"
    assert state.stages["stability"].status == "pending"
    assert state.stages["report"].status == "pending"
    assert plan(state, until=None, resume=True) == ["analyze", "stability", "report"]

def test_resume_reopens_only_analyze_when_only_probe_free(tmp_path):
    _skipped_first_leg(tmp_path)
    state = run_pipeline(n=None, np1=None, nm1=None, workdir=tmp_path, resume=True,
                         probe_free="NH3.out", dry_run=True, on_event=lambda s, m: None)
    assert state.stages["analyze"].status == "pending"
    assert state.stages["stability"].status == SKIPPED   # no refs → stays skipped
    assert state.stages["report"].status == "pending"

def test_resume_without_new_inputs_changes_nothing(tmp_path):
    _skipped_first_leg(tmp_path)
    state = run_pipeline(n=None, np1=None, nm1=None, workdir=tmp_path, resume=True,
                         dry_run=True, on_event=lambda s, m: None)
    assert state.stages["analyze"].status == SKIPPED
    assert state.stages["report"].status == DONE
    assert plan(state, until=None, resume=True) == []

def test_dry_run_writes_nothing(tmp_path):
    events = []
    state = run_pipeline(n="a", np1="b", nm1="c", workdir=tmp_path / "wd",
                         dry_run=True, on_event=lambda s, m: events.append((s, m)))
    assert not (tmp_path / "wd").exists()       # dry-run creates no working dir
    assert len(events) == len(STAGES)


# ─────────────────────────────────────────────
# Engine-free stages with real data: clean stop at run
# ─────────────────────────────────────────────

@_needs_data
def test_pipeline_blocks_at_run_without_engine(tmp_path):
    wd = tmp_path / "pipe"
    state = run_pipeline(
        n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
        workdir=wd, engine="orca", engine_path=str(tmp_path / "no_orca"),
    )
    assert state.stages["validate"].status == DONE
    assert state.stages["cdft"].status == DONE
    assert state.stages["geometry"].status == DONE
    assert state.stages["run"].status == BLOCKED
    assert state.stages["analyze"].status == "pending"
    # geometry produced configs and a cdft.json on disk
    assert Path(state.stages["cdft"].outputs["cdft_json"]).exists()
    assert (wd / "configs").is_dir()
    assert len(list((wd / "configs").glob("*.xyz"))) == state.stages["geometry"].outputs["n_configs"]
    assert (wd / "pipeline.json").exists()

@_needs_data
def test_pipeline_until_geometry(tmp_path):
    state = run_pipeline(
        n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
        workdir=tmp_path / "pipe", until="geometry",
    )
    assert state.stages["geometry"].status == DONE
    assert state.stages["run"].status == "pending"      # never attempted


# ─────────────────────────────────────────────
# Resume with a mocked engine: run → (analyze skipped) → report
# ─────────────────────────────────────────────

@_needs_data
def test_resume_with_mocked_engine_completes(tmp_path, monkeypatch):
    wd = tmp_path / "pipe"
    # First leg: engine-free, blocks at run.
    run_pipeline(n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
                 workdir=wd, engine="orca", engine_path=str(tmp_path / "absent"))

    # Second leg: pretend ORCA is available.
    from cadmia.calculators import local_runner
    monkeypatch.setattr(local_runner, "find_engine",
                        lambda engine, path=None: Path("/fake/orca"))

    def fake_run_job(inp, engine, output, *, engine_path=None, timeout_s=None):
        output.write_text(_VALID_ORCA)
        return True, 1.0
    monkeypatch.setattr(local_runner, "run_job", fake_run_job)

    state = run_pipeline(n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
                         workdir=wd, engine="orca", engine_path="/fake/orca",
                         resume=True)

    assert state.stages["run"].status == DONE
    assert state.stages["run"].outputs["n_done"] > 0
    assert state.stages["analyze"].status == SKIPPED   # no probe_free supplied
    assert state.stages["stability"].status == SKIPPED  # no refs supplied
    assert state.stages["report"].status == DONE
    assert Path(state.stages["report"].outputs["report_md"]).exists()
    assert state.blocked_stage is None


@_needs_data
def test_analyze_stage_maps_configs(tmp_path, monkeypatch):
    """The analyze stage must hand each adduct its geometry config_index."""
    wd = tmp_path / "pipe"
    run_pipeline(n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
                 workdir=wd, engine="orca", engine_path=str(tmp_path / "absent"))

    from cadmia.calculators import local_runner
    monkeypatch.setattr(local_runner, "find_engine",
                        lambda engine, path=None: Path("/fake/orca"))
    monkeypatch.setattr(local_runner, "run_job",
                        lambda inp, engine, output, *, engine_path=None, timeout_s=None:
                        (output.write_text(_VALID_ORCA), (True, 1.0))[1])

    seen = []
    class _FakeAnalysis:
        def to_json(self):
            return "{}"
    def fake_analyze(adduct, cluster, probe_free, **kw):
        seen.append(kw.get("config_index"))
        return _FakeAnalysis()
    from cadmia.agents import analysis_agent
    monkeypatch.setattr(analysis_agent, "analyze", fake_analyze)

    probe_free = tmp_path / "NH3.out"
    probe_free.write_text("dummy\n")
    state = run_pipeline(n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
                         workdir=wd, engine="orca", engine_path="/fake/orca",
                         probe_free=str(probe_free), resume=True)

    assert state.stages["analyze"].status == DONE
    assert len(state.stages["analyze"].outputs["analyses"]) > 0
    assert all(idx is not None for idx in seen)      # every adduct got its config_index
    assert (wd / "analyses").is_dir()


def _fake_engine(monkeypatch, output_text):
    from cadmia.calculators import local_runner
    engines = []
    def fake_find(engine, path=None):
        engines.append(engine)
        return Path("/fake/engine")
    monkeypatch.setattr(local_runner, "find_engine", fake_find)
    monkeypatch.setattr(local_runner, "run_job",
                        lambda inp, engine, output, *, engine_path=None, timeout_s=None:
                        (output.write_text(output_text), (True, 1.0))[1])
    return engines


@pytest.mark.skipif(not (_GAUSSIAN / "N.out").exists(), reason="1tial_cdft fixtures not present")
def test_run_detects_the_engine_from_the_outputs(tmp_path, monkeypatch):
    """Without --engine, Gaussian outputs must not launch ORCA."""
    from cadmia.calculators.params import CalcParams
    engines = _fake_engine(monkeypatch, "x\n")
    run_pipeline(n=_GAUSSIAN / "N.out", np1=_GAUSSIAN / "N+1.out", nm1=_GAUSSIAN / "N-1.out",
                 cluster="Al2TiO5", probe="NH3", workdir=tmp_path / "pipe",
                 params=CalcParams(dispersion="d3bj"), until="run")
    assert engines and set(engines) == {"gaussian"}


@pytest.mark.skipif(not (_GAUSSIAN / "N.out").exists(), reason="1tial_cdft fixtures not present")
def test_run_fails_cleanly_on_params_the_engine_cannot_take(tmp_path, monkeypatch):
    """Gaussian has no D4: the stage fails with the reason instead of raising."""
    from cadmia.calculators.params import CalcParams
    _fake_engine(monkeypatch, "x\n")
    state = run_pipeline(n=_GAUSSIAN / "N.out", np1=_GAUSSIAN / "N+1.out",
                         nm1=_GAUSSIAN / "N-1.out", cluster="Al2TiO5", probe="NH3",
                         workdir=tmp_path / "pipe", params=CalcParams(dispersion="d4"),
                         until="run")
    assert state.stages["run"].status == "failed" and "d4" in state.stages["run"].note


@_needs_data
def test_run_fails_when_every_output_fails_validation(tmp_path, monkeypatch):
    _fake_engine(monkeypatch, "truncated\n")
    state = run_pipeline(n=_N, np1=_NP1, nm1=_NM1, cluster="Al2TiO5", probe="NH3",
                         workdir=tmp_path / "pipe", engine="orca", until="run")
    st = state.stages["run"]
    assert st.status == "failed" and "failed validation" in st.note
    assert st.outputs["n_failed_validation"] == st.outputs["n_done"] > 0


# ─────────────────────────────────────────────
# Validation gate
# ─────────────────────────────────────────────

def test_validate_failure_stops_pipeline(tmp_path, monkeypatch):
    from cadmia.agents import validation_agent

    class _Rep:
        overall = "fail"
    monkeypatch.setattr(validation_agent, "run",
                        lambda filepath, engine=None, check_bsse=False: _Rep())

    n = tmp_path / "n.out"; n.write_text("x")
    state = run_pipeline(n=n, np1=n, nm1=n, workdir=tmp_path / "pipe", engine="orca")
    assert state.stages["validate"].status == "failed"
    assert state.stages["cdft"].status == "pending"   # chain stopped at the gate
