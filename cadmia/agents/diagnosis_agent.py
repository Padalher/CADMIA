"""Diagnosis Agent — deterministic symptom extraction for problematic DFT calculations.

Scans a Gaussian/ORCA output for convergence pathologies (error termination, SCF
failure/oscillation, variational energy collapse, negative or near-zero HOMO-LUMO gap,
CP-SCF response failure, imaginary frequencies, spin contamination) and produces a structured Diagnosis. The optional LLM layer
(cadmia.llm.diagnostician) turns the Diagnosis into fix suggestions; this module is
pure parsing and rules — no network, no LLM.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from cadmia.agents.cdft_agent import _detect_engine
from cadmia.agents import validation_agent
from cadmia.agents.benchmark_agent import _parse_gaussian_frequencies
from cadmia.agents.rank_agent import _count_imaginary
from cadmia.core.units import HARTREE_TO_EV
from cadmia.io.electronic import (
    _parse_gaussian_energy,
    _parse_orca_energy,
    _parse_orca_orbital_energies,
    _read_lines,
)


@dataclass
class Symptom:
    code:     str            # e.g. scf_not_converged, negative_gap
    severity: str            # critical | warning
    message:  str
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnosis:
    filepath:            str
    engine:              str
    terminated_normally: Optional[bool] = None
    scf_converged:       Optional[bool] = None
    energy_hartree:      Optional[float] = None
    homo_ev:             Optional[float] = None
    lumo_ev:             Optional[float] = None
    gap_ev:              Optional[float] = None
    multiplicity:        Optional[int] = None
    s_squared:           Optional[float] = None
    scf_n_iterations:    Optional[int] = None
    n_imaginary:         Optional[int] = None   # None = the output carries no frequencies
    scf_trace_tail:      List[float] = field(default_factory=list)   # last SCF energies (Ha)
    symptoms:            List[Symptom] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.symptoms

    def to_dict(self) -> dict:
        d = asdict(self)
        d["healthy"] = self.healthy
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─────────────────────────────────────────────
# SCF iteration traces
# ─────────────────────────────────────────────

_ORCA_ITER_RE = re.compile(r"^\s*(\d+)\s+(-\d+\.\d{4,})")
_GAUSSIAN_ITER_RE = re.compile(r"^\s*E=\s*(-?\d+\.\d+)")

_SCF_BLOCK_END = ("SCF CONVERGED", "SCF NOT CONVERGED", "TOTAL SCF ENERGY",
                  "SCF SUCCESS", "ERROR")


def _scf_trace(lines: List[str], engine: str) -> List[float]:
    """Energies (Ha) of the SCF iterations; empty when the output has no iteration table."""
    trace: List[float] = []
    if engine == "orca":
        in_block = False
        for line in lines:
            up = line.upper()
            # D-I-I-S / S-O-S-C-F blocks, each headed by "Iteration  Energy (Eh)  Delta-E".
            # One SCF may span several blocks, so the trace accumulates across headers.
            if "ITERATION" in up and "DELTA-E" in up:
                in_block = True
                continue
            if not in_block:
                continue
            if any(marker in up for marker in _SCF_BLOCK_END):
                in_block = False
                continue
            m = _ORCA_ITER_RE.match(line)
            if m:
                trace.append(float(m.group(2)))
        return trace

    for line in lines:                       # gaussian (scf=verbose outputs)
        m = _GAUSSIAN_ITER_RE.match(line)
        if m:
            trace.append(float(m.group(1)))
    return trace


# ─────────────────────────────────────────────
# Frontier orbitals (lightweight, no Structure required)
# ─────────────────────────────────────────────

def _frontier_ev(lines: List[str], engine: str):
    """(homo_ev, lumo_ev) from eigenvalues; open shell: HOMO = max(alpha, beta) HOMO."""
    if engine == "orca":
        alpha, beta, na, nb = _parse_orca_orbital_energies(lines)
        if not alpha:
            return None, None
        homos = [alpha[na - 1]] if na else []
        lumos = [alpha[na]] if na < len(alpha) else []
        if beta:
            if nb:
                homos.append(beta[nb - 1])
            if nb < len(beta):
                lumos.append(beta[nb])
        return (max(homos) if homos else None,
                min(lumos) if lumos else None)

    occ: List[float] = []
    virt: List[float] = []
    beta_occ: List[float] = []
    beta_virt: List[float] = []
    in_block = False
    for line in lines:
        tail = line.split("--", 1)[-1]
        is_alpha_occ = "Alpha  occ. eigenvalues" in line or "Alpha occ. eigenvalues" in line
        is_alpha_virt = "Alpha virt. eigenvalues" in line
        is_beta_occ = "Beta  occ. eigenvalues" in line or "Beta occ. eigenvalues" in line
        is_beta_virt = "Beta virt. eigenvalues" in line
        # A Gaussian job can print several eigenvalue dumps -- an Opt Freq prints one per step.
        # Concatenating them takes the HOMO from the last dump and the LUMO from the first,
        # because the alpha-virtual list starts filling before the final alpha-occupied one
        # replaces it. Every dump begins with "Alpha occ", so that is where the reset goes; the
        # official parser in io/electronic.py already does this and diagnose did not.
        if is_alpha_occ and not in_block:
            occ, virt, beta_occ, beta_virt = [], [], [], []
        in_block = is_alpha_occ or is_alpha_virt or is_beta_occ or is_beta_virt
        if is_alpha_occ:
            occ.extend(float(v) for v in tail.split())
        elif is_alpha_virt:
            virt.extend(float(v) for v in tail.split())
        elif is_beta_occ:
            beta_occ.extend(float(v) for v in tail.split())
        elif is_beta_virt:
            beta_virt.extend(float(v) for v in tail.split())
    if not occ:
        return None, None
    homos = [occ[-1]] + ([beta_occ[-1]] if beta_occ else [])
    lumos = ([virt[0]] if virt else []) + ([beta_virt[0]] if beta_virt else [])
    return (max(homos) * HARTREE_TO_EV,
            min(lumos) * HARTREE_TO_EV if lumos else None)


# ─────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────

_OSCILLATION_HA = 1.0      # amplitude of the last SCF energies considered pathological
_COLLAPSE_HA = 10.0        # drop below the first iteration considered variational collapse
_SMALL_GAP_EV = 0.5
_TRACE_TAIL = 15


def _is_cpscf_failure(line: str) -> bool:
    return ("error termination in SCF RESPONSE" in line
            or ("CP-SCF Solver" in line and "did not converge" in line))


def diagnose(filepath: "str | Path", engine: Optional[str] = None) -> Diagnosis:
    """Extract convergence symptoms from a Gaussian/ORCA output file."""
    path = Path(filepath)
    detected = engine or _detect_engine(path)
    lines = _read_lines(path)
    report = validation_agent.run(path, engine=detected)

    diag = Diagnosis(filepath=str(path.resolve()), engine=detected)
    symptoms: List[Symptom] = []

    term = report.checks["termination"]
    diag.terminated_normally = term.status == "pass"
    if term.status == "fail":
        code = ("truncated_output" if "incomplete or truncated" in term.message
                else "error_termination")
        symptoms.append(Symptom(code, "critical", term.message))

    # The response module (orca_scfresp) builds the analytic Hessian; when its CP-SCF
    # equations fail the run aborts, but the geometry and the reference SCF that precede
    # it are intact, so the fix is a finer grid / numerical frequencies, not a rerun.
    cpscf = next((line.strip() for line in lines if _is_cpscf_failure(line)), None)
    if detected == "orca" and cpscf:
        symptoms.append(Symptom(
            "cpscf_not_converged", "critical",
            "The CP-SCF response solver did not converge: the analytic Hessian "
            "(frequencies, polarizability) failed. Geometry and reference SCF are "
            "unaffected — retry with a finer integration grid and a tighter SCF, or "
            "fall back to numerical frequencies.",
            evidence=cpscf[:200],
        ))

    geo = report.checks["geometry_optimization"]
    if geo.status == "fail":
        symptoms.append(Symptom("geometry_not_converged", "critical", geo.message))
    elif geo.status == "warning":
        symptoms.append(Symptom("geometry_not_converged", "warning", geo.message))

    scf = report.checks["scf_convergence"]
    diag.scf_converged = (True if scf.status == "pass"
                          else False if scf.status == "fail" else None)
    if scf.status == "fail":
        symptoms.append(Symptom("scf_not_converged", "critical", scf.message))

    trace = _scf_trace(lines, detected)
    if trace:
        diag.scf_n_iterations = len(trace)
        diag.scf_trace_tail = trace[-_TRACE_TAIL:]
        tail = diag.scf_trace_tail
        amplitude = max(tail) - min(tail)
        if diag.scf_converged is False and amplitude > _OSCILLATION_HA:
            symptoms.append(Symptom(
                "scf_oscillation", "warning",
                f"SCF energy oscillates with amplitude {amplitude:.3f} Ha "
                f"over the last {len(tail)} iterations.",
                evidence=f"last energies (Ha): {', '.join(f'{e:.4f}' for e in tail[-5:])}",
            ))
        # Only on a non-converged SCF: in a healthy run the first-iteration energy is
        # naturally far above the converged one, and that gap scales with system size,
        # so the absolute 10 Ha threshold would false-positive on large clusters.
        drop = trace[0] - min(trace)
        if diag.scf_converged is False and drop > _COLLAPSE_HA:
            symptoms.append(Symptom(
                "energy_collapse", "critical",
                f"SCF energy fell {drop:.1f} Ha below the first iteration without "
                "converging — likely variational collapse (unphysical state).",
                evidence=f"first {trace[0]:.4f} Ha, lowest {min(trace):.4f} Ha",
            ))
        diag.energy_hartree = trace[-1]

    # Prefer the engine's final energy (includes dispersion and post-SCF terms),
    # so the value matches what benchmark/stability report; trace[-1] is the fallback.
    final = (_parse_orca_energy(lines) if detected == "orca"
             else _parse_gaussian_energy(lines))
    if final is not None:
        diag.energy_hartree = final

    try:
        homo, lumo = _frontier_ev(lines, detected)
    except (ValueError, IndexError):
        homo = lumo = None
    diag.homo_ev, diag.lumo_ev = homo, lumo
    if homo is not None and lumo is not None:
        diag.gap_ev = lumo - homo
        if diag.gap_ev < 0:
            symptoms.append(Symptom(
                "negative_gap", "critical",
                f"Negative HOMO-LUMO gap ({diag.gap_ev:.2f} eV) — occupied orbitals above "
                "virtuals (wrong occupation / variational collapse).",
                evidence=f"HOMO {homo:.3f} eV, LUMO {lumo:.3f} eV",
            ))
        elif diag.gap_ev < _SMALL_GAP_EV:
            symptoms.append(Symptom(
                "small_gap", "warning",
                f"Near-degenerate frontier orbitals (gap {diag.gap_ev:.2f} eV) — "
                "SCF prone to oscillation between configurations.",
                evidence=f"HOMO {homo:.3f} eV, LUMO {lumo:.3f} eV",
            ))

    # Counted through the Rank Agent's helper so both commands always report the same
    # number for the same file. None means the output has no frequencies at all, which is
    # normal for a single point and therefore a fact, not a symptom.
    diag.n_imaginary = _count_imaginary(
        path, SimpleNamespace(frequencies_cm1=_parse_gaussian_frequencies(lines)), detected)
    if diag.n_imaginary == 1:
        symptoms.append(Symptom(
            "imaginary_frequencies", "warning",
            "One imaginary frequency — a first-order saddle point. Legitimate for a "
            "transition state; for an intended minimum, reoptimize along that mode.",
            evidence="1 imaginary mode",
        ))
    elif diag.n_imaginary and diag.n_imaginary > 1:
        symptoms.append(Symptom(
            "imaginary_frequencies", "critical",
            f"{diag.n_imaginary} imaginary frequencies — not a minimum. Several of them on "
            "a structure whose duplicates come out clean usually means a noisy Hessian "
            "(raise the integration grid) rather than a different structure.",
            evidence=f"{diag.n_imaginary} imaginary modes",
        ))

    diag.multiplicity = validation_agent._parse_multiplicity(lines)
    spin = report.checks["spin_contamination"]
    if spin.status in ("fail", "warning"):
        m = re.search(r"⟨S²⟩ = ([\d.]+)", spin.message)
        if m:
            diag.s_squared = float(m.group(1))
        symptoms.append(Symptom(
            "spin_contamination",
            "critical" if spin.status == "fail" else "warning",
            spin.message,
        ))

    diag.symptoms = symptoms
    return diag
