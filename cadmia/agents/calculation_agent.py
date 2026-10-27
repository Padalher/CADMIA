"""Calculation Agent — submit, monitor, download, and validate DFT jobs on HPC."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple

from cadmia.agents.validation_agent import validate_output_summary as _validate_output
from cadmia.calculators.params import CalcParams
from cadmia.config import HPCProfile
from cadmia.core.sidecar import atomic_write_text, file_lock, read_json

# What `prepare` accepts as a source: inputs an engine can already run, and the files that
# carry a geometry to build one from.
_READY_SUFFIXES  = {".inp", ".gjf"}
_OUTPUT_SUFFIXES = {".out", ".log"}

# ─────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────

@dataclass
class JobRecord:
    """State of one submitted HPC job."""
    label:           str
    engine:          str                    # gaussian | orca
    input_path:      Path                   # local input file
    remote_input:    str                    # remote path to input
    remote_output:   str                    # remote path to output
    remote_work_dir: str
    job_id:          str        = ""
    status:          str        = "pending" # pending|queued|running|done|failed|unknown
    output_path:     Optional[Path] = None  # local output after download
    validation_ok:   Optional[bool] = None
    validation_msg:  str             = ""

    def to_dict(self) -> dict:
        return {
            "label":           self.label,
            "engine":          self.engine,
            "job_id":          self.job_id,
            "status":          self.status,
            "input_path":      str(self.input_path),
            "remote_output":   self.remote_output,
            "output_path":     str(self.output_path) if self.output_path else None,
            "validation_ok":   self.validation_ok,
            "validation_msg":  self.validation_msg,
        }


@dataclass
class PrepareResult:
    """Inputs ready for an engine run, split by how each one got there.

    Unpacks as ``inputs, warnings`` so callers that only need the queue (`run_local`)
    stay unchanged; `generated` vs `passed_through` is what lets the CLI report what it
    actually wrote instead of counting every path as a new input.
    """
    inputs:         List[Path] = field(default_factory=list)
    warnings:       List[str]  = field(default_factory=list)
    generated:      List[Path] = field(default_factory=list)
    passed_through: List[Path] = field(default_factory=list)

    def __iter__(self):
        return iter((self.inputs, self.warnings))


@dataclass
class CalculationResult:
    """Output of the Calculation Agent for a batch of jobs."""
    profile:  str
    jobs:     List[JobRecord] = field(default_factory=list)
    warnings: List[str]       = field(default_factory=list)

    @property
    def n_done(self) -> int:
        return sum(1 for j in self.jobs if j.status == "done")

    @property
    def n_failed(self) -> int:
        return sum(1 for j in self.jobs if j.status == "failed")

    def to_json(self) -> str:
        return json.dumps(
            {
                "profile":  self.profile,
                "n_total":  len(self.jobs),
                "n_done":   self.n_done,
                "n_failed": self.n_failed,
                "warnings": self.warnings,
                "jobs":     [j.to_dict() for j in self.jobs],
            },
            indent=2,
        )

    def save(self, path: Path) -> None:
        """Persist job state to a JSON file (used by cadmia monitor)."""
        atomic_write_text(path, self.to_json())

    @classmethod
    def load(cls, path: Path) -> "CalculationResult":
        """Restore job state from a previously saved JSON file."""
        data = read_json(path)
        jobs = []
        for j in data.get("jobs", []):
            jobs.append(JobRecord(
                label=j["label"],
                engine=j["engine"],
                input_path=Path(j["input_path"]),
                remote_input="",    # not persisted; not needed for monitoring
                remote_output=j["remote_output"],
                remote_work_dir="",
                job_id=j["job_id"],
                status=j["status"],
                output_path=Path(j["output_path"]) if j["output_path"] else None,
                validation_ok=j.get("validation_ok"),
                validation_msg=j.get("validation_msg", ""),
            ))
        return cls(profile=data["profile"], jobs=jobs, warnings=data.get("warnings", []))


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_runner(profile: "HPCProfile"):
    """Return the right SLURM backend for *profile*: local (no SSH) or remote SSH.

    Both backends expose the same interface, so submit()/monitor() select by
    profile alone — a `local` profile runs SLURM on this machine, any other goes
    over SSH.
    """
    if getattr(profile, "local", False):
        from cadmia.calculators.local_slurm_runner import LocalSlurmRunner
        return LocalSlurmRunner(profile)
    from cadmia.calculators.hpc_runner import HPCRunner
    return HPCRunner(profile)


# Markers that identify an input's engine, mirroring the output autodetection of
# `cdft_agent._detect_engine`. The extension cannot decide it: `.inp` is the usual name for
# an ORCA input, but Gaussian is fed through stdin and does not care about the extension,
# so many groups call its inputs `.inp` too. The content, on the other hand, is unambiguous.
# (`#` is NOT a marker: it starts a Gaussian route AND an ORCA comment.)
_GAUSSIAN_INPUT_MARKERS = ("%chk", "%nprocshared", "%mem", "--link1--")
_ORCA_INPUT_MARKERS     = ("%pal", "%maxcore", "* xyz", "*xyz")


def _engine_from_input(path: Path) -> str:
    """Detect 'gaussian' or 'orca' from the content of an engine input file."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000].lower()
    except OSError as exc:
        raise ValueError(f"Cannot read {path.name} to detect its engine: {exc}")

    if any(m in head for m in _GAUSSIAN_INPUT_MARKERS):
        return "gaussian"
    if any(m in head for m in _ORCA_INPUT_MARKERS):
        return "orca"
    # Fallback for inputs carrying none of the markers above (a bare Gaussian route, an ORCA
    # input reading its geometry from an xyzfile). The `!` simple-input line is checked over
    # the whole head BEFORE any `#`, because an ORCA input may open with a `#` comment.
    lines = [line.strip() for line in head.splitlines()]
    if any(line.startswith("!") for line in lines):
        return "orca"
    if any(line.startswith("#") for line in lines):
        return "gaussian"
    raise ValueError(
        f"Cannot detect the engine of {path.name} from its content. Pass --engine "
        "explicitly (an ORCA input carries `!`/%pal/%maxcore; a Gaussian one %chk/%mem)."
    )


def _collect_inputs(jobs_source: Path | str) -> List[Path]:
    """Return list of .gjf/.inp files from a file path or directory."""
    p = Path(jobs_source)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.gjf")) + sorted(p.glob("*.inp"))
        if not files:
            raise ValueError(f"No .gjf or .inp files found in {p}")
        return files
    raise FileNotFoundError(f"jobs source not found: {p}")


def _resolve_job_files(
    jobs_source: "Path | str | List[Path | str]",
) -> Tuple[List[Path], List[str]]:
    """Resolve a jobs source into a list of files, returning (files, warnings).

    Accepts a single .gjf/.inp/.xyz file, a directory, or an explicit list. In a
    directory, ready engine inputs (.inp/.gjf) take precedence over .xyz geometries;
    if both are present the .xyz are ignored and a warning is emitted (they would
    otherwise duplicate the hand-made inputs).
    """
    warnings: List[str] = []
    if isinstance(jobs_source, (str, Path)):
        p = Path(jobs_source)
        if p.is_dir():
            ready = sorted(p.glob("*.gjf")) + sorted(p.glob("*.inp"))
            xyz = sorted(p.glob("*.xyz"))
            if ready and xyz:
                warnings.append(f"{p}: using {len(ready)} .inp/.gjf input(s); "
                                f"ignoring {len(xyz)} .xyz file(s).")
            files = ready or xyz
            if not files:
                raise ValueError(f"No .gjf/.inp/.xyz files found in {p}")
            from cadmia.core.sidecar import check_input_directory
            warnings += check_input_directory(p)
        elif p.is_file():
            files = [p]
        else:
            raise FileNotFoundError(f"jobs source not found: {p}")
    else:
        files = [Path(f) for f in jobs_source]
    return files, warnings


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def prepare(
    jobs_source: "Path | str | List[Path | str]",
    params: Optional[CalcParams] = None,
    *,
    engine: Optional[str] = None,
    output_dir: "Path | str" = ".",
    overwrite: bool = False,
) -> PrepareResult:
    """Generate engine inputs (.inp/.gjf) from geometries WITHOUT running them.

    Every source that carries a geometry is converted to an engine input with *params*:
    a .xyz (e.g. the adsorption configs from `cadmia geometry`), or an engine output
    (.out/.log), whose optimized geometry is extracted so each stage of the workflow can
    chain from the previous one's result without a detour through `cadmia xyz`. A
    concatenated .xyz — what `cadmia rank` writes — yields ONE input per block. Ready
    .inp/.gjf files are passed through unchanged, and any other extension is an error
    rather than a silent pass-through.

    This is the batch primitive reused by `run_local`, and exposed directly through
    `cadmia prepare` so inputs can be inspected or submitted to a queue before any
    calculation runs.

    Parameters
    ----------
    jobs_source : a .xyz/.out/.log/.inp/.gjf file, a directory, or a list of paths.
    params      : CalcParams for the generated inputs (default: CalcParams()).
    engine      : "orca" or "gaussian" for the inputs written (default: orca). The parser
                  for a source .out/.log is auto-detected from the file, not from this.
    output_dir  : directory where generated inputs are written.
    overwrite   : regenerate an input even if the target file already exists.
    """
    params  = params or CalcParams()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files, warnings = _resolve_job_files(jobs_source)
    result = PrepareResult(warnings=warnings)

    eng = engine or "orca"
    # Both engines get `.inp`: ORCA takes the filename as its argument and Gaussian is fed
    # through stdin, so the extension is CADMIA's convention, not a requirement — and `.gjf`
    # is a GaussView habit rather than what groups actually use. The engine of a ready input
    # is read from its content (`_engine_from_input`), which the extension cannot tell.
    ext = ".inp"
    for f in files:
        suffix = f.suffix.lower()
        if suffix in _READY_SUFFIXES:
            result.inputs.append(f)
            result.passed_through.append(f)
            continue

        structures = _structures_from_source(f)
        for stem, structure in zip(_input_stems(f, structures), structures):
            target = out_dir / f"{stem}{ext}"
            if target.exists() and not overwrite:
                result.warnings.append(
                    f"{target.name} already exists; kept (use overwrite to regenerate).")
                result.inputs.append(target)
                continue
            own, warning = _structure_spin(structure, params, target.name)
            if warning:
                result.warnings.append(warning)
            written = _write_input(structure, eng, own, target)
            result.inputs.append(written)
            result.generated.append(written)
    return result


def prepare_cdft(
    xyz: "Path | str",
    params: Optional[CalcParams] = None,
    *,
    engine: str = "orca",
    output_dir: "Path | str" = ".",
    charge: int = 0,
    mult: Optional[int] = 1,
    mult_cation: Optional[int] = 2,
    mult_anion: Optional[int] = 2,
    overwrite: bool = False,
) -> Tuple[List[Path], List[str]]:
    """Write the three single-point inputs of a CDFT analysis from ONE geometry.

    Finite-difference CDFT (condensed Fukui, global descriptors) needs the neutral
    (N), the anion (N+1 electrons) and the cation (N-1 electrons) computed at the
    SAME, fixed geometry — so all three are single points (task='sp'), regardless of
    the task in *params*. Feed it an already optimized neutral geometry.

    Charges: neutral = *charge* (default 0), anion = charge-1, cation = charge+1.
    Multiplicities default to a closed-shell neutral (1) with doublet radical ions
    (2); override with *mult* / *mult_anion* / *mult_cation* for other electron counts
    (e.g. an open-shell neutral). Pass ``None`` for any of them to derive the
    multiplicity automatically from the electron-count parity of that charged state.
    Returns ``(input_paths, warnings)`` ordered [N, anion, cation] to match
    `cadmia cdft --n/--np1/--nm1`.
    """
    from dataclasses import replace
    from cadmia.io import structure_from_file

    params  = params or CalcParams()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = Path(xyz)
    if not xyz.is_file():
        raise FileNotFoundError(f"geometry not found: {xyz}")
    if xyz.suffix.lower() != ".xyz" and xyz.suffix.lower() not in _OUTPUT_SUFFIXES:
        raise ValueError(f"prepare_cdft expects a single geometry (.xyz) or an engine "
                         f"output (.out/.log), got '{xyz.name}'")

    structure = structure_from_file(xyz)
    ext = ".inp"                       # one extension for both engines, as in prepare()
    states = [
        ("N",      charge,     mult),
        ("anion",  charge - 1, mult_anion),
        ("cation", charge + 1, mult_cation),
    ]

    # The spin the geometry carries (an XYZ with mult=, an engine output) sets N under auto, and
    # the ions one electron away from it: m − 1, which both adding and removing one can reach.
    if structure.metadata.get("spin_given") and int(structure.charge or 0) == charge:
        m_n = int(structure.multiplicity)
        states = [(label, q, m if m is not None else (m_n if label == "N" else max(m_n - 1, 2)))
                  for label, q, m in states]

    inputs:   List[Path] = []
    warnings: List[str]  = []
    if any(m is None for _, _, m in states):
        from cadmia.core.spin import transition_metals_present
        tms = transition_metals_present(structure.atoms)
        if tms:
            warnings.append(
                "automatic multiplicity uses the lowest spin consistent with the electron "
                f"count; {'/'.join(tms)} may have a high-spin ground state — set --mult "
                "explicitly if needed."
            )
    for label, q, m in states:
        sp     = replace(params, task="sp", charge=q, multiplicity=(m if m else 1),
                         auto_multiplicity=(m is None))
        target = out_dir / f"{xyz.stem}_{label}{ext}"
        if target.exists() and not overwrite:
            warnings.append(f"{target.name} already exists; kept (use overwrite to regenerate).")
            inputs.append(target)
            continue
        if engine == "gaussian":
            from cadmia.calculators.gaussian import write_gjf
            write_gjf(structure.atoms, structure.positions, sp, target,
                      title=f"{xyz.stem} {label}")
        else:
            from cadmia.calculators.orca import write_inp
            write_inp(structure.atoms, structure.positions, sp, target,
                      title=f"{xyz.stem} {label}")
        inputs.append(target)
    return inputs, warnings


def submit(
    jobs_source: "Path | str | List[Path | str]",
    profile_name: str,
    params: Optional[CalcParams] = None,
    *,
    engine: Optional[str] = None,
    output_dir: "Path | str" = ".",
    wait: bool = False,
    poll_interval: int = 60,
    validate_on_download: bool = True,
    profiles_dir: Optional[Path] = None,
) -> CalculationResult:
    """Submit one or more input files to HPC via SLURM and optionally wait.

    Parameters
    ----------
    jobs_source          : path to a .gjf/.inp file, a directory, or a list of paths.
    profile_name         : name of the HPC profile to use (see `cadmia profile`).
    params               : CalcParams for nprocs/mem in SLURM script (default: CalcParams()).
    engine               : force engine; auto-detected from extension if None.
    output_dir           : local directory for downloaded output files.
    wait                 : if True, block until all jobs finish, then download.
    poll_interval        : seconds between status checks when wait=True.
    validate_on_download : run ValidationAgent after each output is downloaded.
    profiles_dir         : override profile directory (used in tests).
    """
    from cadmia.config import DEFAULT_PROFILES_DIR
    profile = HPCProfile.load(profile_name, directory=profiles_dir or DEFAULT_PROFILES_DIR)
    runner  = _make_runner(profile)
    params  = params or CalcParams()

    if isinstance(jobs_source, (str, Path)):
        input_files = _collect_inputs(jobs_source)
    else:
        input_files = [Path(f) for f in jobs_source]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = CalculationResult(profile=profile_name)

    for inp in input_files:
        eng  = engine or _engine_from_input(inp)
        work = f"{profile.work_dir}/{inp.stem}"
        try:
            job_id, remote_out = runner.submit(inp, eng, params, label=inp.stem,
                                               remote_work_dir=work)
            record = JobRecord(
                label=inp.stem, engine=eng, input_path=inp,
                remote_input=f"{work}/{inp.name}",
                remote_output=remote_out,
                remote_work_dir=work,
                job_id=job_id, status="queued",
            )
            result.jobs.append(record)
        except Exception as exc:
            result.warnings.append(f"{inp.name}: submit failed — {exc}")
            result.jobs.append(JobRecord(
                label=inp.stem, engine=eng, input_path=inp,
                remote_input="", remote_output="", remote_work_dir=work,
                status="failed",
            ))

    # Persist initial state
    result.save(out_dir / "cadmia_jobs.json")

    if wait:
        _wait_and_download(result, runner, out_dir, poll_interval, validate_on_download)
        result.save(out_dir / "cadmia_jobs.json")

    return result


def _completed_jobs(jobs_path: Path) -> dict:
    """Map ``label -> JobRecord`` for jobs a previous run already finished successfully.

    Lets `run_local` resume a batch instead of recomputing it: a job whose output still
    exists and did not fail validation is carried forward untouched. A corrupt or missing
    state file is treated as "no prior progress" — never fatal here, since the worst case is
    recomputing (which is exactly what a first run does).
    """
    if not jobs_path.exists():
        return {}
    try:
        prior = CalculationResult.load(jobs_path)
    except (ValueError, KeyError, OSError):
        return {}
    done = {}
    for j in prior.jobs:
        if (j.status == "done" and j.validation_ok is not False
                and j.output_path and Path(j.output_path).exists()):
            done[j.label] = j
    return done


def run_local(
    jobs_source: "Path | str | List[Path | str]",
    params: Optional[CalcParams] = None,
    *,
    engine: Optional[str] = None,
    output_dir: "Path | str" = ".",
    engine_path: Optional[str] = None,
    validate_outputs: bool = True,
    timeout_s: Optional[int] = None,
    tidy: bool = True,
    overwrite: bool = False,
) -> CalculationResult:
    """Run input files sequentially on THIS machine (workstation queue, no SSH/SLURM).

    Accepts .inp/.gjf inputs directly, or .xyz files (e.g. the adsorption configs
    written by the geometry agent) — those are converted to engine inputs with
    *params* first. Jobs run one at a time so each gets the cores/memory declared
    in its input (%pal/%maxcore, %nprocshared/%mem). State is saved to
    ``cadmia_jobs.json`` after every job, so progress survives interruptions.

    Resumable: a job whose output already exists in *output_dir* and passed (per the
    ``cadmia_jobs.json`` left by the earlier run) is skipped, so a re-run after a power
    cut only computes what is left — hours of DFT are never redone by accident. Pass
    ``overwrite=True`` to force every job (and regenerate its input) from scratch.
    """
    from cadmia.calculators import local_runner

    params  = params or CalcParams()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = out_dir / "cadmia_jobs.json"

    inputs, warnings = prepare(
        jobs_source, params, engine=engine, output_dir=out_dir, overwrite=overwrite,
    )

    prior_done = {} if overwrite else _completed_jobs(jobs_path)

    result = CalculationResult(profile="local")
    result.warnings.extend(warnings)
    for i, inp in enumerate(inputs, start=1):
        eng = engine or _engine_from_input(inp)
        ext = ".log" if eng == "gaussian" else ".out"
        output = inp.with_suffix(ext)
        done = prior_done.get(inp.stem)
        if done is not None and output.exists():
            result.jobs.append(done)
            result.warnings.append(
                f"{inp.stem}: already computed; kept (use overwrite to recompute).")
            result.save(jobs_path)
            continue
        record = JobRecord(
            label=inp.stem, engine=eng, input_path=inp,
            remote_input="", remote_output="", remote_work_dir=str(inp.parent),
            job_id=f"local-{i}", status="running",
        )
        result.jobs.append(record)
        result.save(jobs_path)
        try:
            ok, wall = local_runner.run_job(
                inp, eng, output, engine_path=engine_path, timeout_s=timeout_s,
            )
        except local_runner.LocalRunnerError as exc:
            record.status = "failed"
            result.warnings.append(f"{inp.name}: {exc}")
            result.save(jobs_path)
            break   # engine missing — no point trying the rest of the queue
        record.output_path = output
        record.status = "done" if ok else "failed"
        if not ok:
            result.warnings.append(f"{inp.name}: engine exited with an error "
                                   f"after {wall:.0f}s — see {output.name}")
        elif validate_outputs:
            v_ok, msg = _validate_output(output, eng)
            record.validation_ok, record.validation_msg = v_ok, msg
            if not v_ok:
                result.warnings.append(f"Validation failed for {inp.stem}: {msg}")
        if ok and tidy:
            from cadmia.calculators.tidy import tidy_job_outputs
            tidy_job_outputs(out_dir, inp.stem)
        result.save(jobs_path)
    return result


def _structures_from_source(f: Path):
    """Every geometry held by *f*: one per .xyz block, or the final geometry of an output."""
    suffix = f.suffix.lower()
    if suffix == ".xyz":
        from cadmia.io.xyz import parse_xyz_database
        return parse_xyz_database(f)
    if suffix in _OUTPUT_SUFFIXES:
        from cadmia.io import structure_from_file
        structure = structure_from_file(f)
        structure.metadata["spin_given"] = True   # the engine's own charge and spin
        return [structure]
    raise ValueError(
        f"{f.name}: cannot prepare an input from a '{f.suffix}' file. Give a geometry "
        f"(.xyz), an engine output (.out/.log), or a ready engine input (.inp/.gjf)."
    )


def _stem_from_comment(comment: str) -> str:
    """The origin stem `cadmia rank` records in a block comment, or "" if it is not one.

    rank writes ``rank 1  +0.000 kcal/mol  E=-1.234 Eh  orca  <origin stem>``, followed by
    ``[<skeleton class>]`` when it grouped the outputs by skeleton.
    """
    tokens = [t for t in comment.split() if not (t.startswith("[") and t.endswith("]"))]
    if len(tokens) < 2 or tokens[0] != "rank":
        return ""
    stem = Path(tokens[-1]).name
    return stem if stem not in ("", ".", "..") else ""


def _input_stems(source: Path, structures: list) -> List[str]:
    """One input stem per structure, keeping the original name wherever there is one.

    A single-geometry source keeps its own stem, unless it is a `cadmia rank` block. In a
    concatenated .xyz each block keeps the stem of the output it came from: those names are long, but they carry the site, the
    adsorption mode and the angles, which is exactly what makes a candidate identifiable
    downstream. Blocks from any other concatenated .xyz fall back to a numbered stem.
    """
    if len(structures) == 1:
        return [_stem_from_comment((structures[0].metadata or {}).get("comment", ""))
                or source.stem]

    stems: List[str] = []
    used: set = set()
    for i, structure in enumerate(structures, start=1):
        stem = _stem_from_comment((structure.metadata or {}).get("comment", ""))
        if not stem or stem in used:
            stem = f"{stem or source.stem}_{i:03d}"
        used.add(stem)
        stems.append(stem)
    return stems


COUNTERPOISE_TERMS = ("cluster_own", "cluster_full", "probe_own", "probe_full")


def prepare_counterpoise(
    adduct: "Path | str",
    cluster: "Path | str",
    probe_free: "Path | str",
    params: Optional[CalcParams] = None,
    *,
    output_dir: "Path | str" = ".",
    overwrite: bool = False,
) -> Tuple[List[Path], List[str]]:
    """The four ORCA single points of the Boys–Bernardi counterpoise correction of one adduct.

    Each fragment (the cluster, the probe) at its geometry IN THE ADDUCT, once in its own basis
    and once in the full basis of the adduct, the partner's atoms as ghosts (``O:`` in ORCA). The
    adduct holds the cluster's atoms first, as `cadmia geometry` writes it; the fragments' charge
    and multiplicity are those of the isolated cluster and probe outputs. `cadmia analyze --bsse`
    reads the four outputs, named ``<adduct>_cp_<term>.out``, next to the adduct or here.
    Returns ``(input_paths, warnings)``.
    """
    from cadmia.io import structure_from_file
    from cadmia.calculators.orca import write_inp

    params = params or CalcParams()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ad, cl, pr = (structure_from_file(Path(p)) for p in (adduct, cluster, probe_free))
    n_cl = len(cl.atoms)
    if sorted(ad.atoms[:n_cl]) != sorted(cl.atoms) or sorted(ad.atoms[n_cl:]) != sorted(pr.atoms):
        raise ValueError(f"{Path(adduct).name}: its first {n_cl} atoms are not the cluster and the "
                         "rest the probe; the counterpoise fragments cannot be told apart.")
    atoms, pos = list(ad.atoms), ad.positions
    ghost = [f"{a}:" for a in atoms]
    terms = {
        "cluster_own":  (atoms[:n_cl], pos[:n_cl], cl),
        "cluster_full": (atoms[:n_cl] + ghost[n_cl:], pos, cl),
        "probe_own":    (atoms[n_cl:], pos[n_cl:], pr),
        "probe_full":   (ghost[:n_cl] + atoms[n_cl:], pos, pr),
    }
    inputs: List[Path] = []
    warnings: List[str] = []
    stem = Path(adduct).stem
    for term in COUNTERPOISE_TERMS:
        fatoms, fpos, frag = terms[term]
        sp = replace(params, task="sp", charge=int(frag.charge or 0),
                     multiplicity=int(frag.multiplicity or 1), auto_multiplicity=False)
        target = out_dir / f"{stem}_cp_{term}.inp"
        if target.exists() and not overwrite:
            warnings.append(f"{target.name} already exists; kept (use overwrite to regenerate).")
        else:
            write_inp(fatoms, fpos, sp, target, title=f"{stem} counterpoise {term}")
        inputs.append(target)
    return inputs, warnings


def _structure_spin(structure, params: CalcParams, name: str) -> "tuple[CalcParams, str]":
    """Under ``--mult auto``, the multiplicity the structure itself carries (an XYZ comment with
    ``mult=``, as geometry/prescreen/rank write, or an engine output's own) wins over the
    electron-count parity, which only knows the lowest spin. Only for the same charge: a spin
    read for another charge would be an impossible pair. Returns the params and a warning."""
    if not (params.auto_multiplicity and structure.metadata.get("spin_given")):
        return params, ""
    if int(structure.charge or 0) != params.charge:
        return params, (f"{name}: the geometry carries charge {structure.charge} and multiplicity "
                        f"{structure.multiplicity}, but the input is written with charge "
                        f"{params.charge}; the multiplicity comes from the electron count. Pass "
                        f"--mult to set it.")
    return replace(params, multiplicity=int(structure.multiplicity), auto_multiplicity=False), ""


def _write_input(structure, engine: str, params: CalcParams, target: Path) -> Path:
    """Write one engine input for *structure* at *target*."""
    if engine == "orca":
        from cadmia.calculators.orca import write_inp
        return write_inp(structure.atoms, structure.positions, params,
                         target, title=target.stem)
    from cadmia.calculators.gaussian import write_gjf
    return write_gjf(structure.atoms, structure.positions, params,
                     target, title=target.stem)


def monitor(
    jobs_file: "Path | str",
    profile_name: str,
    *,
    output_dir: "Path | str" = ".",
    poll_interval: int = 60,
    validate_on_download: bool = True,
    profiles_dir: Optional[Path] = None,
    once: bool = False,
) -> CalculationResult:
    """Resume monitoring jobs recorded in *jobs_file*.

    Polls SLURM status for all non-finished jobs, downloads and validates
    completed outputs. Saves updated state to *jobs_file* after each cycle.

    Parameters
    ----------
    jobs_file            : path to ``cadmia_jobs.json`` written by submit().
    profile_name         : HPC profile name.
    output_dir           : local directory for downloaded outputs.
    poll_interval        : seconds between polling cycles.
    validate_on_download : run ValidationAgent on each downloaded output.
    once                 : if True, poll once then return (useful for CLI --once flag).
    """
    jobs_path = Path(jobs_file)
    from cadmia.config import DEFAULT_PROFILES_DIR
    profile = HPCProfile.load(profile_name, directory=profiles_dir or DEFAULT_PROFILES_DIR)
    runner  = _make_runner(profile)
    out_dir = Path(output_dir)

    # Hold an exclusive lock across load→poll→save so a second `cadmia monitor` on the same
    # jobs file cannot overwrite this one's updates (a `done` reverting to `running`).
    with file_lock(jobs_path):
        result = CalculationResult.load(jobs_path)
        result.profile = profile_name
        _wait_and_download(result, runner, out_dir, poll_interval, validate_on_download, once=once)
        result.save(jobs_path)
    return result


def _wait_and_download(
    result: CalculationResult,
    runner: "HPCRunner",
    out_dir: Path,
    poll_interval: int,
    validate_on_download: bool,
    once: bool = False,
) -> None:
    """Poll until all pending/running jobs finish, then download."""
    from cadmia.calculators.hpc_runner import HPCConnectionError

    active_statuses = {"queued", "running", "pending", "unknown"}

    while True:
        active = [j for j in result.jobs if j.status in active_statuses and j.job_id]
        if not active:
            break

        for job in active:
            try:
                status = runner.check_status(job.job_id, job.remote_output)
                job.status = status
                if status == "done":
                    _download_job(job, runner, out_dir, validate_on_download, result)
                elif status == "failed":
                    result.warnings.append(f"Job {job.job_id} ({job.label}) failed on HPC.")
            except HPCConnectionError as exc:
                result.warnings.append(f"Status check error for {job.job_id}: {exc}")

        if once:
            break

        remaining = [j for j in result.jobs if j.status in active_statuses and j.job_id]
        if not remaining:
            break
        time.sleep(poll_interval)


def _download_job(
    job: JobRecord,
    runner: "HPCRunner",
    out_dir: Path,
    validate: bool,
    result: CalculationResult,
) -> None:
    """Download a completed job's output and optionally validate it."""
    from cadmia.calculators.hpc_runner import HPCConnectionError

    ext     = ".log" if job.engine == "gaussian" else ".out"
    local   = out_dir / f"{job.label}{ext}"
    try:
        runner.download(job.remote_output, local)
        job.output_path = local
        job.status = "done"
    except HPCConnectionError as exc:
        result.warnings.append(f"Download failed for {job.label}: {exc}")
        job.status = "failed"
        return

    if validate:
        ok, msg = _validate_output(local, job.engine)
        job.validation_ok  = ok
        job.validation_msg = msg
        if not ok:
            result.warnings.append(
                f"Validation failed for {job.label}: {msg}"
            )
