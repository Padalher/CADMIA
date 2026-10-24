"""Benchmark Agent — systematic comparison of Gaussian 16 vs ORCA 6 for the same system."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cadmia.agents.validation_agent import validate_output_summary as _validate_output
from cadmia.calculators.gaussian import write_gjf
from cadmia.calculators.orca import write_inp
from cadmia.calculators.params import CalcParams
from cadmia.core.sidecar import write_json_sidecar
from cadmia.core.units import HARTREE_TO_KCAL as _HARTREE_TO_KCAL
from cadmia.io.electronic import _parse_gaussian_energy, _parse_orca_energy
from cadmia.utils.geometry import kabsch_rmsd

# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────

_DE_KCAL_THRESHOLD   = 0.1     # flag if |ΔE| > 0.1 kcal/mol
_RMSD_ANG_THRESHOLD  = 0.01    # flag if geometry RMSD > 0.01 Å

# Modes with |freq| ≤ this are treated as translational/rotational and excluded
_NONTRIVIAL_FREQ_THRESHOLD = 10.0   # cm⁻¹

_N_FREQ_COMPARE = 6   # compare first N non-trivial modes


# ─────────────────────────────────────────────
# Output parsers (private)
# ─────────────────────────────────────────────

def _parse_gaussian_frequencies(lines: List[str]) -> List[float]:
    """Parse all vibrational frequencies (cm⁻¹) from Gaussian output."""
    freqs: List[float] = []
    for line in lines:
        # Not startswith("Frequencies --"): hpmodes prints "Frequencies ---" for the
        # same modes, so both blocks match and every frequency is counted twice.
        if re.match(r"\s*Frequencies\s+--\s", line):
            freqs.extend(float(x) for x in line.split("--", 1)[1].split())
    return freqs


def _parse_orca_frequencies(lines: List[str]) -> List[float]:
    """Parse all vibrational frequencies (cm⁻¹) from ORCA output.

    ORCA output format:
        0:         0.00 cm**-1
        ...
        6:       145.23 cm**-1
    Skips near-zero modes (|freq| <= _NONTRIVIAL_FREQ_THRESHOLD).
    """
    # The LAST block, not the first. An opt+freq run prints one per geometry, and the rest of
    # CADMIA (io/vibrational.py) reads the last one: taking the first here made the same file
    # give two different frequency sets depending on which code path asked.
    freqs: List[float] = []
    in_block = False
    pat = re.compile(r"^\s*\d+:\s+([-\d.]+)\s+cm\*\*-1")
    for line in lines:
        if "VIBRATIONAL FREQUENCIES" in line:
            in_block = True
            freqs = []                     # a new block replaces the previous one
            continue
        if in_block:
            m = pat.match(line)
            if m:
                freqs.append(float(m.group(1)))
            elif freqs:
                in_block = False           # block ended; keep reading in case another follows
    return freqs


def _nontrivial(freqs: List[float]) -> List[float]:
    """Return frequencies with |freq| > threshold (vibrational modes only)."""
    return [f for f in freqs if abs(f) > _NONTRIVIAL_FREQ_THRESHOLD]


def _parse_gaussian_wall_time(lines: List[str]) -> Optional[float]:
    """Return elapsed wall time in seconds from Gaussian output.

    Looks for: 'Elapsed time:  X days  Y hours  Z minutes  W.w seconds.'
    """
    pat = re.compile(
        r"Elapsed time:\s+(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes\s+([\d.]+)\s+seconds"
    )
    for line in lines:
        m = pat.search(line)
        if m:
            d, h, mn, s = m.groups()
            return int(d) * 86400 + int(h) * 3600 + int(mn) * 60 + float(s)
    return None


def _parse_orca_wall_time(lines: List[str]) -> Optional[float]:
    """Return wall time in seconds from ORCA output.

    Looks for: 'TOTAL RUN TIME: X days Y hours Z minutes W seconds N msec'
    """
    pat = re.compile(
        r"TOTAL RUN TIME:\s+(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes\s+(\d+)\s+seconds\s+(\d+)\s+msec"
    )
    for line in lines:
        m = pat.search(line)
        if m:
            d, h, mn, s, ms = m.groups()
            return int(d) * 86400 + int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000
    return None


# ─────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────

@dataclass
class EngineResult:
    """Parsed results from a single engine's output file."""
    engine:          str                   # "gaussian" | "orca"
    output_path:     Path
    energy_hartree:  Optional[float]       = None
    frequencies_cm1: Optional[List[float]] = None
    geometry:        Optional[np.ndarray]  = None   # (N, 3) optimized positions
    atoms:           Optional[List[str]]   = None
    wall_time_s:     Optional[float]       = None
    validation_ok:   bool                  = False
    validation_msg:  str                   = ""

    def to_dict(self) -> dict:
        return {
            "engine":         self.engine,
            "output_path":    str(self.output_path),
            "energy_hartree": self.energy_hartree,
            "wall_time_s":    self.wall_time_s,
            "frequencies_cm1_first6": (
                _nontrivial(self.frequencies_cm1)[:_N_FREQ_COMPARE]
                if self.frequencies_cm1 else None
            ),
            "validation_ok":  self.validation_ok,
            "validation_msg": self.validation_msg,
        }


@dataclass
class BenchmarkComparison:
    """Cross-engine comparison metrics."""
    delta_e_kcal:     Optional[float] = None   # |E_gaussian − E_orca| × 627.509
    rmsd_ang:         Optional[float] = None   # Kabsch RMSD of optimized geometries
    timing_ratio:     Optional[float] = None   # wall_time_orca / wall_time_gaussian
    freq_rmsd_cm1:    Optional[float] = None   # RMSD of first 6 non-trivial modes
    engine_agreement: bool            = False
    recommendation:   str             = "inconclusive"
    flags:            List[str]       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "delta_e_kcal":     self.delta_e_kcal,
            "rmsd_ang":         self.rmsd_ang,
            "timing_ratio":     self.timing_ratio,
            "freq_rmsd_cm1":    self.freq_rmsd_cm1,
            "engine_agreement": self.engine_agreement,
            "recommendation":   self.recommendation,
            "flags":            self.flags,
        }


@dataclass
class BenchmarkResult:
    """Full output of the Benchmark Agent.

    ``params`` is None when the result comes from compare() on pre-computed
    outputs: the level of theory is not reliably recoverable from output files,
    so it is reported as unknown rather than guessed.
    """
    cluster:         str
    params:          Optional[CalcParams]
    gaussian_input:  Optional[Path]             = None
    orca_input:      Optional[Path]             = None
    gaussian_result: Optional[EngineResult]     = None
    orca_result:     Optional[EngineResult]     = None
    comparison:      Optional[BenchmarkComparison] = None
    warnings:        List[str]                  = field(default_factory=list)
    output_json:     Optional[str]              = None   # sidecar written next to the inputs

    def to_json(self) -> str:
        data: dict = {
            "cluster": self.cluster,
            "params": {
                "functional":   self.params.functional,
                "basis":        self.params.basis,
                "dispersion":   self.params.dispersion,
                "task":         self.params.task,
                "charge":       self.params.charge,
                "multiplicity": self.params.multiplicity,
            } if self.params else None,
            "gaussian_input": str(self.gaussian_input) if self.gaussian_input else None,
            "orca_input":     str(self.orca_input)     if self.orca_input     else None,
            "gaussian_result": self.gaussian_result.to_dict() if self.gaussian_result else None,
            "orca_result":     self.orca_result.to_dict()     if self.orca_result     else None,
            "comparison":      self.comparison.to_dict()      if self.comparison      else None,
            "output_json":     self.output_json,
            "warnings":        self.warnings,
        }
        return json.dumps(data, indent=2)

    def to_markdown(self) -> str:
        p = self.params
        if p is not None:
            theory_lines = [
                f"**Level of theory**: {p.functional}/{p.basis} "
                f"{'+ ' + p.dispersion.upper() if p.dispersion not in ('none', '') else 'no dispersion'}  ",
                f"**Task**: {p.task}  ",
                f"**Charge / Mult**: {p.charge} / {p.multiplicity}",
            ]
        else:
            theory_lines = ["**Level of theory**: unknown (parsed from pre-computed outputs)  "]

        lines = [
            f"## CADMIA Benchmark Report — {self.cluster}",
            "",
            *theory_lines,
            "",
            "| Metric | Gaussian | ORCA | Δ / ratio |",
            "|--------|----------|------|-----------|",
        ]

        g = self.gaussian_result
        o = self.orca_result
        c = self.comparison

        def _fmt(v: Optional[float], decimals: int = 6) -> str:
            return f"{v:.{decimals}f}" if v is not None else "—"

        lines.append(
            f"| Energy (Ha) | {_fmt(g.energy_hartree if g else None)} "
            f"| {_fmt(o.energy_hartree if o else None)} "
            f"| {_fmt(c.delta_e_kcal if c else None, 4)} kcal/mol |"
        )
        lines.append(
            f"| Geometry RMSD (Å) | — | — | {_fmt(c.rmsd_ang if c else None, 4)} |"
        )
        lines.append(
            f"| Wall time (s) | {_fmt(g.wall_time_s if g else None, 1)} "
            f"| {_fmt(o.wall_time_s if o else None, 1)} "
            f"| {_fmt(c.timing_ratio if c else None, 3)}× |"
        )
        lines.append(
            f"| Freq RMSD (cm⁻¹) | — | — | {_fmt(c.freq_rmsd_cm1 if c else None, 2)} |"
        )

        if c:
            agree_str = "✓ YES" if c.engine_agreement else "✗ NO"
            lines += [
                "",
                f"**Engine agreement**: {agree_str}  ",
                f"**Recommendation**: {c.recommendation}  ",
            ]
            if c.flags:
                lines.append("")
                lines.append("**Flags**:")
                for flag in c.flags:
                    lines.append(f"- {flag}")

        if self.warnings:
            lines += ["", "**Warnings**:"]
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# Comparison logic
# ─────────────────────────────────────────────

def _compare(g: EngineResult, o: EngineResult) -> BenchmarkComparison:
    flags: List[str] = []

    # ΔE
    delta_e: Optional[float] = None
    if g.energy_hartree is not None and o.energy_hartree is not None:
        delta_e = abs(g.energy_hartree - o.energy_hartree) * _HARTREE_TO_KCAL
        if delta_e > _DE_KCAL_THRESHOLD:
            flags.append(f"|ΔE| = {delta_e:.4f} kcal/mol > threshold ({_DE_KCAL_THRESHOLD})")

    # Geometry RMSD
    rmsd: Optional[float] = None
    if g.geometry is not None and o.geometry is not None:
        if g.geometry.shape == o.geometry.shape:
            rmsd = kabsch_rmsd(g.geometry, o.geometry)
            if rmsd > _RMSD_ANG_THRESHOLD:
                flags.append(f"Geometry RMSD = {rmsd:.4f} Å > threshold ({_RMSD_ANG_THRESHOLD})")
        else:
            flags.append(
                f"Geometry shape mismatch: Gaussian {g.geometry.shape} vs ORCA {o.geometry.shape}"
            )

    # Timing ratio
    timing_ratio: Optional[float] = None
    if g.wall_time_s and o.wall_time_s and g.wall_time_s > 0:
        timing_ratio = o.wall_time_s / g.wall_time_s

    # Frequency RMSD (first N non-trivial modes)
    freq_rmsd: Optional[float] = None
    if g.frequencies_cm1 and o.frequencies_cm1:
        gf = _nontrivial(g.frequencies_cm1)[:_N_FREQ_COMPARE]
        of = _nontrivial(o.frequencies_cm1)[:_N_FREQ_COMPARE]
        if len(gf) == len(of) == _N_FREQ_COMPARE:
            arr = np.array(gf) - np.array(of)
            freq_rmsd = float(np.sqrt((arr ** 2).mean()))

    # Validation status
    if not g.validation_ok:
        flags.append(f"Gaussian validation failed: {g.validation_msg}")
    if not o.validation_ok:
        flags.append(f"ORCA validation failed: {o.validation_msg}")

    agreement = (
        (delta_e is not None and delta_e <= _DE_KCAL_THRESHOLD)
        and (rmsd is None or rmsd <= _RMSD_ANG_THRESHOLD)
        and g.validation_ok and o.validation_ok
    )

    # Engine recommendation
    if not agreement:
        recommendation = "inconclusive"
    elif timing_ratio is None:
        recommendation = "either"
    elif timing_ratio < 1.0:
        recommendation = "orca"
    else:
        recommendation = "gaussian"

    return BenchmarkComparison(
        delta_e_kcal=delta_e,
        rmsd_ang=rmsd,
        timing_ratio=timing_ratio,
        freq_rmsd_cm1=freq_rmsd,
        engine_agreement=agreement,
        recommendation=recommendation,
        flags=flags,
    )


def _parse_engine_result(output_path: Path, engine: str) -> EngineResult:
    """Parse energy, geometry, frequencies, and wall time from a calculation output."""
    from cadmia.io.gaussian import parse_gaussian_geometry
    from cadmia.io.orca    import parse_orca_geometry

    lines = output_path.read_text(encoding="utf-8", errors="replace").splitlines()

    result = EngineResult(engine=engine, output_path=output_path)

    if engine == "gaussian":
        result.energy_hartree  = _parse_gaussian_energy(lines)
        result.frequencies_cm1 = _parse_gaussian_frequencies(lines)
        result.wall_time_s     = _parse_gaussian_wall_time(lines)
        try:
            struct = parse_gaussian_geometry(output_path)
            result.geometry = struct.positions
            result.atoms    = struct.atoms
        except (OSError, ValueError, IndexError, KeyError):
            pass  # optional geometry enrichment; energy/frequencies already parsed

    elif engine == "orca":
        result.energy_hartree  = _parse_orca_energy(lines)
        result.frequencies_cm1 = _parse_orca_frequencies(lines)
        result.wall_time_s     = _parse_orca_wall_time(lines)
        try:
            struct = parse_orca_geometry(output_path)
            result.geometry = struct.positions
            result.atoms    = struct.atoms
        except (OSError, ValueError, IndexError, KeyError):
            pass  # optional geometry enrichment; energy/frequencies already parsed

    return result


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate(
    xyz_path: "str | Path",
    params: CalcParams,
    output_dir: "str | Path",
    cluster_label: str = "",
) -> BenchmarkResult:
    """Generate Gaussian (.gjf) and ORCA (.inp) input files for the same geometry.

    Parameters
    ----------
    xyz_path      : cluster .xyz file.
    params        : CalcParams defining the level of theory.
    output_dir    : directory where .gjf and .inp files are written.
    cluster_label : label override (default: stem of xyz_path).
    """
    from cadmia.agents.geometry_agent import _read_xyz  # reuse xyz reader

    xyz_path = Path(xyz_path)
    atoms, coords, _ = _read_xyz(xyz_path)
    positions = np.array([c for c in coords])

    label = cluster_label or xyz_path.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gjf_path = out_dir / f"{label}_gaussian.gjf"
    inp_path = out_dir / f"{label}_orca.inp"

    write_gjf(atoms, positions, params, gjf_path, title=f"{label} — Gaussian")
    write_inp(atoms, positions, params, inp_path, title=f"{label} — ORCA")

    result = BenchmarkResult(
        cluster=label,
        params=params,
        gaussian_input=gjf_path,
        orca_input=inp_path,
    )
    # The level of theory the two inputs share is what `benchmark compare` is comparing at;
    # the sidecar records it so the comparison never has to be guessed from the outputs.
    result.output_json = str(out_dir / "benchmark.json")
    _, err = write_json_sidecar(result.output_json, result.to_json(),
                                artifacts=[p for p in (gjf_path, inp_path) if p])
    if err:
        result.output_json = None
        result.warnings.append(err)
    return result


def compare(
    gaussian_out: "str | Path",
    orca_out: "str | Path",
    *,
    engine_gaussian: str = "gaussian",
    engine_orca: str = "orca",
    cluster_label: str = "",
) -> BenchmarkResult:
    """Compare already-computed outputs from Gaussian and ORCA.

    Calls the Validation Agent on both files, extracts ΔE, geometry RMSD,
    frequency comparison, and timing ratio.

    Parameters
    ----------
    gaussian_out    : path to the Gaussian output file (.log or .out).
    orca_out        : path to the ORCA output file (.out).
    engine_gaussian : engine type for gaussian_out (default "gaussian").
    engine_orca     : engine type for orca_out (default "orca").
    cluster_label   : label for the cluster; inferred from gaussian_out stem if omitted.
    """
    g_path = Path(gaussian_out)
    o_path = Path(orca_out)
    label = cluster_label or g_path.stem

    # Level of theory is not reliably recoverable from output files — report unknown.
    result = BenchmarkResult(cluster=label, params=None)

    # Parse each output
    g_res = _parse_engine_result(g_path, engine_gaussian)
    o_res = _parse_engine_result(o_path, engine_orca)

    # Validate
    g_ok, g_msg = _validate_output(g_path, engine_gaussian)
    o_ok, o_msg = _validate_output(o_path, engine_orca)
    g_res.validation_ok  = g_ok
    g_res.validation_msg = g_msg
    o_res.validation_ok  = o_ok
    o_res.validation_msg = o_msg

    result.gaussian_result = g_res
    result.orca_result     = o_res
    result.comparison      = _compare(g_res, o_res)

    return result


def run(
    xyz_path: "str | Path",
    params: CalcParams,
    output_dir: "str | Path",
    *,
    gaussian_out: "str | Path | None" = None,
    orca_out: "str | Path | None" = None,
    cluster_label: str = "",
) -> BenchmarkResult:
    """Generate inputs and optionally compare pre-computed results.

    Parameters
    ----------
    xyz_path     : cluster .xyz file.
    params       : level of theory.
    output_dir   : directory for input files.
    gaussian_out : if provided, parse and validate this Gaussian output.
    orca_out     : if provided, parse and validate this ORCA output.
    cluster_label: label override.
    """
    result = generate(xyz_path, params, output_dir, cluster_label)

    if gaussian_out and orca_out:
        cmp = compare(gaussian_out, orca_out, cluster_label=result.cluster)
        result.gaussian_result = cmp.gaussian_result
        result.orca_result     = cmp.orca_result
        result.comparison      = cmp.comparison
        result.warnings.extend(cmp.warnings)

    return result
