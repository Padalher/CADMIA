"""Orchestrator Agent — chain the specialized agents into one resumable pipeline.

Runs the reactivity → geometry → calculation → analysis → stability → report
chain in dependency order, persisting a manifest (``pipeline.json``) so a run can
be stopped and resumed. The calculation stage needs a quantum-chemistry engine;
when none is reachable it is marked ``blocked`` and the pipeline stops cleanly,
so the engine-free stages (validate → cdft → geometry) can run on any machine and
the rest resume later on the workstation.

Deterministic only: no natural-language parsing, no LLM. Every stage decision is
recorded in the manifest (and, with a database, in ``workflow_runs``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cadmia.core.sidecar import atomic_write_text, read_json

# Pipeline stages in dependency order.
STAGES: tuple[str, ...] = (
    "validate", "cdft", "geometry", "run", "analyze", "stability", "report",
)

# Stage status vocabulary.
PENDING, DONE, SKIPPED, BLOCKED, FAILED = (
    "pending", "done", "skipped", "blocked", "failed",
)

_MANIFEST_NAME = "pipeline.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageState:
    name:    str
    status:  str = PENDING
    outputs: Dict[str, Any] = field(default_factory=dict)
    note:    str = ""
    updated: str = ""


@dataclass
class PipelineState:
    """Persisted plan + progress of one pipeline run."""
    cluster:  str
    probe:    str
    workdir:  str
    inputs:   Dict[str, Any] = field(default_factory=dict)
    stages:   Dict[str, StageState] = field(default_factory=dict)
    created:  str = ""

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = {name: StageState(name=name) for name in STAGES}
        if not self.created:
            self.created = _now()

    # ── serialization ──
    def to_dict(self) -> dict:
        return {
            "cluster": self.cluster,
            "probe":   self.probe,
            "workdir": self.workdir,
            "created": self.created,
            "inputs":  self.inputs,
            "stages":  {k: asdict(v) for k, v in self.stages.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else Path(self.workdir) / _MANIFEST_NAME
        return atomic_write_text(path, self.to_json())

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        data = read_json(path)
        known = {f.name for f in fields(StageState)}
        stages = {
            k: StageState(**{kk: vv for kk, vv in v.items() if kk in known})
            for k, v in data.get("stages", {}).items()
        }
        for name in STAGES:                       # tolerate manifests from other versions
            stages.setdefault(name, StageState(name=name))
        return cls(
            cluster=data.get("cluster", ""), probe=data.get("probe", ""),
            workdir=data.get("workdir", "."), inputs=data.get("inputs", {}),
            stages=stages, created=data.get("created", ""),
        )

    # ── queries ──
    @property
    def completed(self) -> List[str]:
        return [n for n in STAGES if self.stages[n].status in (DONE, SKIPPED)]

    @property
    def blocked_stage(self) -> Optional[str]:
        for n in STAGES:
            if self.stages[n].status in (BLOCKED, FAILED):
                return n
        return None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def detect_engine(n_output: "str | Path") -> str:
    """Engine of the run: the one that wrote the N output."""
    from cadmia.agents.cdft_agent import _detect_engine
    return _detect_engine(Path(n_output))


def _extract_cluster_xyz(n_output: Path, dest: Path, engine: Optional[str]) -> Path:
    """Write the cluster geometry from the N SCF output, preserving atom order.

    Using the output (not a hand-made .xyz) guarantees the same atom ordering as the
    CDFT sites, which is what the geometry agent needs.
    """
    from cadmia.agents.cdft_agent import _detect_engine
    eng = engine or _detect_engine(n_output)
    if eng == "orca":
        from cadmia.io.orca import parse_orca_geometry as parse
    else:
        from cadmia.io.gaussian import parse_gaussian_geometry as parse
    structure = parse(n_output)
    atomic_write_text(dest, structure.to_xyz(comment=dest.stem))
    return dest


def _emit(on_event: Optional[Callable[[str, str], None]], stage: str, msg: str) -> None:
    if on_event is not None:
        on_event(stage, msg)


# ─────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────

def _stage_validate(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.agents import validation_agent
    files = [ctx["n"], ctx["np1"], ctx["nm1"], *ctx["refs"]]
    verdicts, failed = {}, []
    for f in files:
        report = validation_agent.run(filepath=str(f), engine=ctx["engine"])
        verdicts[Path(f).name] = report.overall
        if report.overall == "fail":
            failed.append(Path(f).name)
    st = state.stages["validate"]
    st.outputs = {"verdicts": verdicts}
    if failed:
        st.status, st.note = FAILED, f"SCF validation failed: {', '.join(failed)}"
    else:
        st.status, st.note = DONE, "all base states pass"
    return st


def _stage_cdft(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.agents import cdft_agent
    result = cdft_agent.run(
        path_n=ctx["n"], path_np1=ctx["np1"], path_nm1=ctx["nm1"],
        cluster=state.cluster, engine=ctx["engine"],
    )
    cdft_json = ctx["workdir"] / "cdft.json"
    atomic_write_text(cdft_json, result.to_json())
    st = state.stages["cdft"]
    st.status = DONE
    st.outputs = {"cdft_json": str(cdft_json),
                  "top_site": result.sites[0].atom_symbol if result.sites else None}
    st.note = f"{len(result.sites)} sites ranked"
    if result.warnings:
        st.note += "; " + " | ".join(result.warnings)
    return st


def _stage_geometry(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.agents import geometry_agent
    cdft_json = Path(state.stages["cdft"].outputs["cdft_json"])
    cluster_xyz = ctx["workdir"] / f"{state.cluster or 'cluster'}.xyz"
    _extract_cluster_xyz(Path(ctx["n"]), cluster_xyz, ctx["engine"])
    configs_dir = ctx["workdir"] / "configs"
    result = geometry_agent.run(
        cdft_json, cluster_xyz, state.probe,
        output_dir=configs_dir, cluster_label=state.cluster,
    )
    config_map = {
        Path(c.xyz_path).stem: {"config_index": c.config_index,
                                "site_atom_symbol": c.site_atom_symbol}
        for c in result.configs if c.xyz_path
    }
    st = state.stages["geometry"]
    st.status = DONE
    st.outputs = {"cluster_xyz": str(cluster_xyz), "configs_dir": str(configs_dir),
                  "n_configs": len(result.configs), "config_map": config_map}
    st.note = f"{len(result.configs)} {state.probe} configs"
    return st


def _stage_run(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.calculators.local_runner import LocalRunnerError, find_engine
    from cadmia.agents import calculation_agent

    st = state.stages["run"]
    configs_dir = Path(state.stages["geometry"].outputs["configs_dir"])
    eng = ctx["engine"]

    try:
        find_engine(eng, ctx["engine_path"])
    except LocalRunnerError as exc:
        st.status = BLOCKED
        st.note = (f"no reachable {eng} engine ({exc}). Resume on a machine with the "
                   f"engine: cadmia pipeline --resume --workdir {state.workdir} "
                   f"--engine-path /path/to/engine")
        return st

    ads_dir = ctx["workdir"] / "adsorption"
    try:
        result = calculation_agent.run_local(
            configs_dir, ctx["params"], engine=eng, output_dir=ads_dir,
            engine_path=ctx["engine_path"], timeout_s=ctx["timeout_s"],
            overwrite=ctx.get("regenerate_inputs", False),
        )
    except ValueError as exc:
        st.status, st.note = FAILED, str(exc)
        return st
    # "done" only means the engine exited. Validation is what says the numbers are usable, and
    # an adduct that failed it still goes through: dropping it here would hide a result the user
    # may want to look at. What must not happen is that it contributes a number in silence, and
    # that is guarded where the number is computed — analyze refuses an energy from a geometry
    # that did not converge. Here it is only made visible.
    rejected = [j.label for j in result.jobs
                if j.status == "done" and j.output_path and j.validation_ok is False]
    outputs = {j.label: str(j.output_path) for j in result.jobs
               if j.status == "done" and j.output_path}
    st.outputs = {"adsorption_dir": str(ads_dir), "outputs": outputs,
                  "n_done": result.n_done, "n_failed": result.n_failed,
                  "n_failed_validation": len(rejected), "warnings": result.warnings}
    if result.n_done == 0:
        st.status, st.note = FAILED, "no adsorption job completed"
    elif len(rejected) == result.n_done:
        st.status = FAILED
        st.note = (f"all {result.n_done} output(s) failed validation; check the {eng} "
                   f"outputs in {ads_dir}")
    else:
        st.status = DONE
        st.note = f"{result.n_done} done, {result.n_failed} failed"
        if rejected:
            st.note += (f"; {len(rejected)} failed validation and are analyzed anyway: "
                        + ", ".join(sorted(rejected)[:6]) + (" …" if len(rejected) > 6 else ""))
    if result.warnings:
        # the spin and charge each input got, among others: lost here, nobody would see them
        st.note += f"; {len(result.warnings)} warning(s): " + " | ".join(result.warnings[:3]) + (
            " …" if len(result.warnings) > 3 else "")
    return st


def _stage_analyze(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.agents import analysis_agent
    st = state.stages["analyze"]
    if not ctx["probe_free"]:
        st.status = SKIPPED
        st.note = "no --probe-free output; adsorption energetics need the isolated probe"
        return st

    cdft_json = state.stages["cdft"].outputs["cdft_json"]
    config_map = state.stages["geometry"].outputs.get("config_map", {})
    outputs = state.stages["run"].outputs.get("outputs", {})
    analyses_dir = ctx["workdir"] / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    written, failed = [], []
    for label, adduct in sorted(outputs.items()):
        cfg = config_map.get(label, {})
        try:
            result = analysis_agent.analyze(
                adduct, ctx["n"], ctx["probe_free"],
                probe=state.probe, bsse=ctx["bsse"], cdft_json=cdft_json,
                config_index=cfg.get("config_index"), cluster_label=state.cluster,
            )
        except Exception as exc:                  # one bad adduct must not sink the batch
            failed.append(f"{label}: {exc}")
            continue
        dest = analyses_dir / f"{label}_analysis.json"
        atomic_write_text(dest, result.to_json())
        written.append(str(dest))

    st.status = DONE if written else FAILED
    st.outputs = {"analyses": written}
    st.note = f"{len(written)} analyzed" + (f", {len(failed)} failed" if failed else "")
    return st


def _stage_stability(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.agents import stability_agent
    st = state.stages["stability"]
    if not ctx["refs"]:
        st.status, st.note = SKIPPED, "no --ref outputs supplied"
        return st
    result = stability_agent.compute_stability(
        ctx["n"], list(ctx["refs"]), mixed_label=state.cluster,
    )
    stability_json = ctx["workdir"] / "stability.json"
    atomic_write_text(stability_json, result.to_json())
    st.status = DONE
    st.outputs = {"stability_json": str(stability_json),
                  "e_rel_kcal": result.e_rel_kcal}
    st.note = (f"E_rel({result.energy_basis}) = {result.e_rel_kcal:.3f} kcal/mol"
               if result.e_rel_kcal is not None
               else "E_rel not computed; see the stability warnings")
    return st


def _stage_report(state: PipelineState, ctx: dict) -> StageState:
    from cadmia.agents import report_agent
    cdft_json = state.stages["cdft"].outputs.get("cdft_json")
    analyses = state.stages["analyze"].outputs.get("analyses", [])
    stability_json = state.stages["stability"].outputs.get("stability_json")

    report = report_agent.build_report(
        cdft=cdft_json, analyses=analyses, stability=stability_json,
        title=f"{state.cluster or 'CADMIA'} pipeline",
    )
    report_md = ctx["workdir"] / "report.md"
    written = report.to_files(report_md, latex=ctx["latex"])
    st = state.stages["report"]
    st.status = DONE
    st.outputs = {"report_md": written["markdown"], "latex": written["latex"]}
    st.note = f"{len(report.correlations)} correlation(s)"
    return st


_STAGE_FUNCS: Dict[str, Callable[[PipelineState, dict], StageState]] = {
    "validate":  _stage_validate,
    "cdft":      _stage_cdft,
    "geometry":  _stage_geometry,
    "run":       _stage_run,
    "analyze":   _stage_analyze,
    "stability": _stage_stability,
    "report":    _stage_report,
}


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def _params_record(params) -> dict:
    """The calculation parameters as the manifest stores them (JSON round-trip, so that a
    saved record and a fresh one compare equal)."""
    from dataclasses import asdict
    return json.loads(json.dumps(asdict(params)))


# resources, not level of theory: changing them on resume is fine
_RESOURCE_PARAMS = ("nprocs", "mem_gaussian_gb", "mem_orca_mb_core")


def plan(state: PipelineState, *, until: Optional[str], resume: bool) -> List[str]:
    """Return the stages that would execute, given resume/until (no side effects)."""
    limit = STAGES.index(until) if until else len(STAGES) - 1
    out = []
    for i, name in enumerate(STAGES):
        if i > limit:
            break
        if resume and state.stages[name].status in (DONE, SKIPPED):
            continue
        out.append(name)
    return out


def run_pipeline(
    *,
    n: "str | Path",
    np1: "str | Path",
    nm1: "str | Path",
    cluster: str = "",
    probe: str = "NH3",
    refs: Optional[List["str | Path"]] = None,
    probe_free: "str | Path | None" = None,
    workdir: "str | Path" = ".",
    engine: Optional[str] = None,
    engine_path: Optional[str] = None,
    params: Any = None,
    bsse: bool = False,
    latex: bool = False,
    timeout_s: Optional[int] = None,
    until: Optional[str] = None,
    resume: bool = False,
    dry_run: bool = False,
    on_event: Optional[Callable[[str, str], None]] = None,
    params_explicit: bool = True,
) -> PipelineState:
    """Execute (or plan) the full CADMIA pipeline, resuming from a saved manifest.

    With ``resume=True`` an existing ``pipeline.json`` in *workdir* is loaded and
    completed stages are skipped. ``until`` stops after the named stage. ``dry_run``
    reports the plan without executing. A ``blocked`` or ``failed`` stage stops the
    run; the manifest is saved after every stage so progress is never lost.

    The level of theory is saved in the manifest. On resume it is the saved one unless
    *params_explicit*; a different one is accepted only while no calculation has run, and
    then the inputs are written again. Once outputs exist it is refused, because the study
    would mix two levels of theory.
    """
    from cadmia.calculators.params import CalcParams

    if until is not None and until not in STAGES:
        raise ValueError(f"Unknown stage '{until}'. Choices: {', '.join(STAGES)}")

    work = Path(workdir)
    manifest_path = work / _MANIFEST_NAME

    regenerate_inputs = False
    wanted = _params_record(params or CalcParams())
    has_calcs = (work / "adsorption" / "cadmia_jobs.json").exists()
    if not resume and (has_calcs or manifest_path.exists() and
                       PipelineState.load(manifest_path).stages["run"].status in (DONE, FAILED)):
        raise ValueError(f"{work} already holds the calculations of a pipeline: continue it with "
                         f"--resume, or start a new one in another --workdir.")
    if resume and manifest_path.exists():
        state = PipelineState.load(manifest_path)
        saved = state.inputs.get("params")
        has_calcs = has_calcs or state.stages["run"].status in (DONE, FAILED)
        if saved and not params_explicit:
            known = {f.name for f in fields(CalcParams)}
            params = CalcParams(**{k: v for k, v in saved.items() if k in known})
        elif saved and wanted != saved:
            changed = ", ".join(f"{k} {saved.get(k)!r} → {wanted.get(k)!r}"
                                for k in sorted(set(saved) | set(wanted))
                                if saved.get(k) != wanted.get(k) and k not in _RESOURCE_PARAMS)
            if changed and has_calcs:
                raise ValueError(
                    f"The level of theory differs from the one this pipeline started with "
                    f"({changed}), and calculations at that level already exist. Resume without "
                    f"the level options to keep it, or start a new pipeline in another --workdir.")
            regenerate_inputs = bool(changed)
        if params_explicit or not (saved or has_calcs):
            # a manifest from 0.3.0 holds no level: with outputs present, defaults are no record
            state.inputs["params"] = wanted
        # Optional inputs may only become available at resume time (the isolated probe
        # optimized later, or the stability references). When one arrives, re-open the
        # stage that was skipped for lack of it — and the report that consumes it —
        # so plan() picks them up instead of skipping them as already SKIPPED.
        reopen: List[str] = []
        if probe_free:
            state.inputs["probe_free"] = str(probe_free)
            if state.stages["analyze"].status == SKIPPED:
                reopen += ["analyze", "report"]
        if refs:
            state.inputs["refs"] = [str(r) for r in refs]
            if state.stages["stability"].status == SKIPPED:
                reopen += ["stability", "report"]
        for name in reopen:
            st = state.stages[name]
            st.status, st.outputs, st.note = PENDING, {}, "re-opened (new input at resume)"
    else:
        state = PipelineState(cluster=cluster, probe=probe.upper(), workdir=str(work))
        state.inputs = {
            "n": str(n), "np1": str(np1), "nm1": str(nm1),
            "refs": [str(r) for r in (refs or [])],
            "probe_free": str(probe_free) if probe_free else None,
            "params": wanted,
        }

    ctx = {
        "n": state.inputs["n"], "np1": state.inputs["np1"], "nm1": state.inputs["nm1"],
        "refs": state.inputs.get("refs", []),
        "probe_free": state.inputs.get("probe_free"),
        "workdir": work, "engine": engine or detect_engine(state.inputs["n"]),
        "engine_path": engine_path,
        "params": params or CalcParams(), "bsse": bsse, "latex": latex,
        "timeout_s": timeout_s, "regenerate_inputs": regenerate_inputs,
    }

    todo = plan(state, until=until, resume=resume)

    if dry_run:
        for name in STAGES:
            mark = "RUN" if name in todo else ("skip" if state.stages[name].status
                                               in (DONE, SKIPPED) else "—")
            _emit(on_event, name, f"[{mark}]")
        return state

    work.mkdir(parents=True, exist_ok=True)
    for name in todo:
        _emit(on_event, name, "running")
        st = _STAGE_FUNCS[name](state, ctx)
        st.updated = _now()
        state.stages[name] = st
        state.save(manifest_path)
        _emit(on_event, name, f"{st.status}: {st.note}")
        if st.status in (BLOCKED, FAILED):
            break
    return state
