"""Validation Agent — scientific quality control for DFT calculation outputs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cadmia.agents.cdft_agent import _detect_engine
from cadmia.io.electronic import _parse_gaussian_spin_sq
from cadmia.utils.chemistry import bond_distance_range, expected_spin_sq

# Spin contamination thresholds: relative deviation from expected ⟨S²⟩
_SPIN_WARN_REL    = 0.10   # > 10%  → warning
_SPIN_FAIL_REL    = 0.25   # > 25%  → fail
_SPIN_ABS_SINGLET = 0.02   # singlet: any ⟨S²⟩ > 0.02 → warning


@dataclass
class CheckResult:
    status: str    # "pass" | "fail" | "warning" | "skip"
    message: str

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message}


@dataclass
class ValidationReport:
    filepath: str
    engine: str
    overall: str   # "pass" | "fail" | "warning"
    checks: Dict[str, CheckResult] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_dict(self) -> dict:
        return {
            "file": self.filepath,
            "engine": self.engine,
            "overall": self.overall,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
        }

    @property
    def passed(self) -> bool:
        return self.overall == "pass"


# ─────────────────────────────────────────────
# Reading helpers
# ─────────────────────────────────────────────

def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _parse_multiplicity(lines: List[str]) -> Optional[int]:
    """Spin multiplicity from either engine's header.

    Gaussian writes ``Multiplicity = 2`` and ORCA writes
    ``Multiplicity           Mult            ....    2``. Matching only the first form left every
    ORCA calculation with no multiplicity, so the spin-contamination check skipped silently on
    exactly the open-shell states it exists for.
    """
    for line in lines:
        m = (re.search(r"Multiplicity\s*=\s*(\d+)", line)
             or re.search(r"Multiplicity\s+Mult\s+\.+\s*(\d+)", line))
        if m:
            return int(m.group(1))
    return None


# ─────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────

def _check_termination(lines: List[str], engine: str) -> CheckResult:
    if engine == "gaussian":
        for line in reversed(lines):
            if "Normal termination" in line:
                return CheckResult("pass", f"Normal termination: {line.strip()}")
            if "Error termination" in line:
                return CheckResult("fail", f"Error termination: {line.strip()}")
        return CheckResult("fail", "No termination marker found — file may be incomplete or truncated.")

    # ORCA
    for line in reversed(lines):
        if "ORCA TERMINATED NORMALLY" in line:
            return CheckResult("pass", "Normal termination (ORCA).")
        if "****ORCA TERMINATED WITH ERROR****" in line:
            return CheckResult("fail", "ORCA terminated with error.")
    return CheckResult("fail", "No ORCA termination marker found — file may be incomplete or truncated.")


def _check_scf_convergence(lines: List[str], engine: str) -> CheckResult:
    if engine == "gaussian":
        converged = False
        for line in lines:
            if "SCF Done:" in line:
                converged = True
            if "Convergence criterion not met" in line:
                return CheckResult("fail", f"SCF did not converge: {line.strip()}")
        if converged:
            return CheckResult("pass", "SCF converged.")
        for line in lines:
            if "TOTAL ENERGY" in line.upper() or "E(R" in line or "E(U" in line:
                return CheckResult("pass", "SCF converged (energy found in output).")
        return CheckResult("warning", "No explicit SCF convergence marker found.")

    # ORCA
    for line in lines:
        if "SCF NOT CONVERGED" in line.upper():
            return CheckResult("fail", "SCF did not converge (ORCA).")
    for line in lines:
        if "SCF CONVERGED" in line.upper():
            return CheckResult("pass", "SCF converged (ORCA).")
    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            return CheckResult("pass", "SCF converged — final energy found (ORCA).")
    return CheckResult("warning", "No SCF convergence marker found (ORCA).")


def _check_spin_contamination(lines: List[str], engine: str) -> CheckResult:
    mult = _parse_multiplicity(lines)
    if mult is None:
        return CheckResult("skip", "Could not determine spin multiplicity.")

    s2_expected = expected_spin_sq(mult)

    if engine == "gaussian":
        # Before annihilation: the value after it hides the contamination of the energy itself.
        s2 = _parse_gaussian_spin_sq(lines)
        if s2 is None:
            if mult == 1:
                return CheckResult("pass", "Closed-shell singlet — no spin contamination.")
            return CheckResult("skip", "No S**2 block found (multiplicity > 1 without UHF/ROHF).")
        return _evaluate_spin(s2, s2_expected, mult)

    # ORCA
    s2_val = None
    for line in lines:
        if "<S**2>" in line or "Expectation value of <S**2>" in line.lower():
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if nums:
                try:
                    s2_val = float(nums[-1])
                except ValueError:
                    pass
    if s2_val is None:
        if mult == 1:
            return CheckResult("pass", "Closed-shell singlet — no spin contamination.")
        return CheckResult("skip", "⟨S²⟩ not found in ORCA output.")
    return _evaluate_spin(s2_val, s2_expected, mult)


def _evaluate_spin(s2: float, s2_expected: float, mult: int) -> CheckResult:
    label = f"⟨S²⟩ = {s2:.4f}, expected {s2_expected:.4f} (mult={mult})"
    if s2_expected == 0.0:
        if s2 > _SPIN_ABS_SINGLET:
            return CheckResult("warning", f"{label} — spin contamination detected in singlet.")
        return CheckResult("pass", label)
    dev_rel = abs(s2 - s2_expected) / s2_expected
    if dev_rel > _SPIN_FAIL_REL:
        return CheckResult("fail", f"{label} — relative deviation {dev_rel:.1%} exceeds limit ({_SPIN_FAIL_REL:.0%}).")
    if dev_rel > _SPIN_WARN_REL:
        return CheckResult("warning", f"{label} — relative deviation {dev_rel:.1%} (warning threshold {_SPIN_WARN_REL:.0%}).")
    return CheckResult("pass", f"{label} — deviation {dev_rel:.1%}.")


def _check_frequencies(lines: List[str], engine: str) -> CheckResult:
    freqs: List[float] = []

    if engine == "gaussian":
        for line in lines:
            # "Frequencies --" is also a substring of hpmodes' "Frequencies ---", so a
            # job with hpmodes reported every frequency twice.
            if re.match(r"\s*Frequencies\s+--\s", line):
                raw = line.split("--", 1)[-1]
                for tok in raw.split():
                    try:
                        freqs.append(float(tok))
                    except ValueError:
                        pass
    else:  # ORCA
        for line in lines:
            if "cm**-1" in line or "cm-1" in line:
                m = re.search(r"([-+]?\d+\.\d+)\s+cm", line)
                if m:
                    freqs.append(float(m.group(1)))

    if not freqs:
        return CheckResult("skip", "No frequency data in file (SP or non-freq calculation).")

    # The same tolerance io/vibrational.py uses, so `validate` and `analyze` cannot disagree
    # about the same file. Any f < 0 counted a -3 cm-1 numerical artifact as a saddle point,
    # while the thermochemistry path already treated it as a translation/rotation residue.
    from cadmia.io.vibrational import _TRANSROT_TOL_CM1
    imaginary = [f for f in freqs if f < -_TRANSROT_TOL_CM1]
    if not imaginary:
        return CheckResult(
            "pass",
            f"{len(freqs)} frequencies computed — none imaginary "
            f"(below -{_TRANSROT_TOL_CM1:.0f} cm⁻¹).")

    vals = ", ".join(f"{f:.1f}" for f in imaginary[:5])
    return CheckResult(
        "fail",
        f"{len(imaginary)} imaginary frequency(ies): {vals} cm⁻¹ — geometry is not a true minimum.",
    )


def _gaussian_routes(lines: List[str]) -> List[str]:
    """Every route section (one per Link1 job), its wrapped lines joined back."""
    routes = []
    for i, line in enumerate(lines):
        if line.startswith(" #") and (i == 0 or lines[i - 1].startswith(" ---")):
            route = []
            for part in lines[i:]:
                if part.startswith(" ---"):
                    break
                route.append(part[1:])
            routes.append("".join(route))
    return routes


def _check_geometry_optimization(lines: List[str], engine: str) -> CheckResult:
    """Did the geometry optimization actually converge?

    An optimization that exhausts its cycle budget gives up, and the engine still reports a
    normal termination — ORCA additionally skips the Freq stage, so the output looks like a
    plain Opt. Without this check the unconverged geometry and its energy flow into ranking
    and adsorption energetics as if they were a minimum.
    """
    text = "\n".join(lines)
    if engine == "orca":
        if "GEOMETRY OPTIMIZATION CYCLE" not in text:
            return CheckResult("skip", "No geometry optimization in file (single point).")
        if "did not converge but reached the maximum" in text:
            n = text.count("GEOMETRY OPTIMIZATION CYCLE")
            return CheckResult(
                "fail",
                f"Geometry optimization ran out of cycles after {n} steps without converging — "
                "the final geometry is not a stationary point and its energy is not comparable "
                "to converged ones. ORCA skips the Freq stage in this case and still terminates "
                "normally.",
            )
        if "THE OPTIMIZATION HAS CONVERGED" in text:
            return CheckResult("pass", "Geometry optimization converged.")
        return CheckResult("warning", "Geometry optimization present but no convergence "
                                      "statement found — the run may have been interrupted.")

    routes = _gaussian_routes(lines)
    if routes and not any(re.search(r"\bopt\b", r, re.IGNORECASE) for r in routes):
        # A Freq job prints "Berny optimization" and "Step number 1" for its stationarity test.
        return CheckResult("skip", "No geometry optimization in file (no Opt in the route).")
    if "Berny optimization" not in text and "Step number" not in text:
        return CheckResult("skip", "No geometry optimization in file (single point).")
    if "Optimization completed" in text:
        return CheckResult("pass", "Geometry optimization converged.")
    if "Number of steps exceeded" in text or "Optimization stopped" in text:
        return CheckResult("fail", "Geometry optimization stopped without converging — the "
                                   "final geometry is not a stationary point.")
    return CheckResult("warning", "Geometry optimization present but no convergence statement "
                                  "found — the run may have been interrupted.")


def _check_geometry_sanity(filepath: Path, engine: str) -> CheckResult:
    try:
        if engine == "gaussian":
            from cadmia.io.gaussian import parse_gaussian_geometry
            structure = parse_gaussian_geometry(filepath)
        else:
            from cadmia.io.orca import parse_orca_geometry
            structure = parse_orca_geometry(filepath)
    except Exception as exc:
        return CheckResult("warning", f"Could not parse geometry for sanity check: {exc}")

    pos = structure.positions
    atoms = structure.atoms
    n = len(atoms)
    clashes: List[str] = []

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            d_min, _ = bond_distance_range(atoms[i], atoms[j])
            if d < d_min:
                clashes.append(
                    f"{atoms[i]}({i+1})–{atoms[j]}({j+1}): {d:.3f} Å < {d_min:.3f} Å"
                )

    if clashes:
        detail = "; ".join(clashes[:3])
        extra = f" (+{len(clashes) - 3} more)" if len(clashes) > 3 else ""
        return CheckResult("fail", f"{len(clashes)} abnormally short distance(s): {detail}{extra}")

    n_pairs = n * (n - 1) // 2
    return CheckResult("pass", f"Geometry sanity OK — {n_pairs} pairs checked, no clashes.")


def _check_bsse(lines: List[str], engine: str) -> CheckResult:
    if engine == "gaussian":
        for line in lines:
            if "counterpoise" in line.lower():
                return CheckResult("pass", "Counterpoise (BSSE) correction found in output.")
        return CheckResult(
            "fail",
            "No counterpoise correction found — required for adsorption calculations.",
        )
    # ORCA
    for line in lines:
        if "BSSE" in line or "counterpoise" in line.lower():
            return CheckResult("pass", "BSSE correction found in output (ORCA).")
    return CheckResult(
        "fail",
        "No BSSE correction found — required for adsorption calculations.",
    )


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run(
    filepath: str | Path,
    engine: Optional[str] = None,
    check_bsse: bool = False,
) -> ValidationReport:
    """Validate a DFT output file.

    Parameters
    ----------
    filepath:
        Path to the Gaussian or ORCA .out/.log file.
    engine:
        'gaussian' or 'orca'. Auto-detected from file content when omitted.
    check_bsse:
        Enable BSSE check. Should only be True for adsorption calculations.
    """
    path = Path(filepath)
    detected = engine or _detect_engine(path)
    lines = _read_lines(path)

    checks: Dict[str, CheckResult] = {
        "termination":        _check_termination(lines, detected),
        "scf_convergence":    _check_scf_convergence(lines, detected),
        "spin_contamination": _check_spin_contamination(lines, detected),
        "geometry_optimization": _check_geometry_optimization(lines, detected),
        "frequencies":        _check_frequencies(lines, detected),
        "geometry_sanity":    _check_geometry_sanity(path, detected),
        "bsse":               _check_bsse(lines, detected) if check_bsse
                              else CheckResult("skip", "BSSE check not requested (adsorption calculations only)."),
    }

    statuses = {c.status for c in checks.values()}
    if "fail" in statuses:
        overall = "fail"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "pass"

    return ValidationReport(
        filepath=str(path.resolve()),
        engine=detected,
        overall=overall,
        checks=checks,
    )


def validate_output_summary(output_path: "str | Path", engine: Optional[str] = None) -> "tuple[bool, str]":
    """Run validation and return (passed, summary of fail/warning checks).

    Shared by the benchmark and calculation agents. 'passed' is True for pass or warning.
    """
    try:
        report = run(filepath=str(output_path), engine=engine)
        ok = report.overall in ("pass", "warning")
        msg = "; ".join(
            f"{name}: {r.message}"
            for name, r in report.checks.items()
            if r.status in ("fail", "warning")
        ) or "OK"
        return ok, msg
    except Exception as exc:
        return False, str(exc)


def geometry_converged(filepath: "str | Path", engine: Optional[str] = None) -> Optional[bool]:
    """Did the optimization in this output converge? None when the file holds no optimization.

    Thin wrapper over the check :func:`run` already applies, for callers that need this one
    answer and not a whole report. Worth reusing instead of grepping: ORCA wraps its failure
    message across two lines, so the obvious one-line grep for it reports every unconverged
    run as fine. An optimization present but with no convergence statement counts as False —
    unknown is not the same as converged.
    """
    path = Path(filepath)
    detected = engine or _detect_engine(path)
    result = _check_geometry_optimization(_read_lines(path), detected)
    if result.status == "skip":
        return None
    return result.status == "pass"
