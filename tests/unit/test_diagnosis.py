"""Tests for cadmia.agents.diagnosis_agent + cadmia.llm.diagnostician — no network."""
from __future__ import annotations

import json

import pytest

from cadmia.agents.diagnosis_agent import diagnose, _scf_trace, _frontier_ev
from cadmia.llm import LLMResponse, Provider, suggest_fixes


_DIVERGING_ORCA = """\
                                 O   R   C   A

Multiplicity = 2
----------------------------------------D-I-I-S--------------------------------------------
Iteration    Energy (Eh)           Delta-E    RMSDP     MaxDP
   0   -1709.10000000   0.000000000     0.10        0.01
   1   -1709.30000000  -0.200000000     0.09        0.01
   2   -1722.50000000 -13.200000000     0.50        0.10
   3   -1721.80000000   0.700000000     0.40        0.09
   4   -1722.90000000  -1.100000000     0.55        0.11
SCF NOT CONVERGED AFTER 512 CYCLES

ORBITAL ENERGIES
  NO   OCC          E(Eh)            E(eV)
   0   1.0000     -10.000000      -272.1140
   1   1.0000      -0.100000        -2.7211
   2   0.0000      -0.300000        -8.1634

****ORCA TERMINATED WITH ERROR****
"""

_HEALTHY_ORCA = """\
                                 O   R   C   A

----------------------------------------D-I-I-S--------------------------------------------
Iteration    Energy (Eh)           Delta-E    RMSDP     MaxDP
   0   -1709.10000000   0.000000000     0.10        0.01
   1   -1709.15000000  -0.050000000     0.01        0.001
   2   -1709.16000000  -0.010000000     0.001       0.0001
SCF CONVERGED AFTER 3 CYCLES

ORBITAL ENERGIES
  NO   OCC          E(Eh)            E(eV)
   0   2.0000     -10.000000      -272.1140
   1   2.0000      -0.335000        -9.1212
   2   0.0000      -0.150000        -4.0823

FINAL SINGLE POINT ENERGY      -1709.16000000

                   ****ORCA TERMINATED NORMALLY****
"""


@pytest.fixture
def diverging(tmp_path):
    p = tmp_path / "cation.out"
    p.write_text(_DIVERGING_ORCA)
    return p


@pytest.fixture
def healthy(tmp_path):
    p = tmp_path / "neutral.out"
    p.write_text(_HEALTHY_ORCA)
    return p


# ─────────────────────────────────────────────
# symptom extraction
# ─────────────────────────────────────────────

def test_diverging_cation_symptoms(diverging):
    diag = diagnose(diverging)
    assert diag.engine == "orca"
    assert diag.terminated_normally is False
    assert diag.scf_converged is False
    codes = {s.code for s in diag.symptoms}
    assert {"error_termination", "scf_not_converged",
            "energy_collapse", "scf_oscillation", "negative_gap"} <= codes
    assert not diag.healthy
    assert diag.scf_n_iterations == 5
    assert diag.energy_hartree == pytest.approx(-1722.9)
    assert diag.gap_ev == pytest.approx(-5.4423, abs=1e-3)
    assert diag.multiplicity == 2

def test_diverging_severities(diverging):
    diag = diagnose(diverging)
    by_code = {s.code: s for s in diag.symptoms}
    assert by_code["energy_collapse"].severity == "critical"
    assert by_code["negative_gap"].severity == "critical"
    assert by_code["scf_oscillation"].severity == "warning"

def test_healthy_output_no_symptoms(healthy):
    diag = diagnose(healthy)
    assert diag.healthy
    assert diag.terminated_normally is True
    assert diag.scf_converged is True
    assert diag.gap_ev == pytest.approx(5.0389, abs=1e-3)
    assert diag.energy_hartree == pytest.approx(-1709.16)

def test_energy_prefers_final_over_scf_trace(tmp_path):
    # ORCA: FINAL SINGLE POINT ENERGY includes the dispersion correction, so it
    # differs from the last SCF iteration — diagnose must report the final one
    # (consistent with benchmark/stability).
    p = tmp_path / "disp.out"
    p.write_text(_HEALTHY_ORCA.replace(
        "FINAL SINGLE POINT ENERGY      -1709.16000000",
        "FINAL SINGLE POINT ENERGY      -1709.16923320",
    ))
    diag = diagnose(p)
    assert diag.energy_hartree == pytest.approx(-1709.1692332)
    assert diag.scf_trace_tail[-1] != diag.energy_hartree

def test_diagnosis_json_roundtrip(diverging):
    diag = diagnose(diverging)
    data = json.loads(diag.to_json())
    assert data["healthy"] is False
    assert data["engine"] == "orca"
    assert len(data["symptoms"]) == len(diag.symptoms)
    assert data["scf_trace_tail"][-1] == pytest.approx(-1722.9)


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def test_scf_trace_orca():
    trace = _scf_trace(_DIVERGING_ORCA.splitlines(), "orca")
    assert trace == [-1709.1, -1709.3, -1722.5, -1721.8, -1722.9]

def test_frontier_open_shell_uses_max_homo():
    lines = """\
ORBITAL ENERGIES
                 SPIN UP ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000     -10.000000      -272.1140
   1   1.0000      -0.400000       -10.8846
   2   0.0000      -0.100000        -2.7211
                 SPIN DOWN ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000     -10.100000      -274.8351
   1   0.0000      -0.200000        -5.4423
""".splitlines()
    homo, lumo = _frontier_ev(lines, "orca")
    assert homo == pytest.approx(-10.8846)   # max(alpha_HOMO, beta_HOMO)
    assert lumo == pytest.approx(-5.4423)    # min(alpha_LUMO, beta_LUMO)

def test_frontier_gaussian_closed_shell():
    lines = [
        " Alpha  occ. eigenvalues --  -10.00000  -0.33523",
        " Alpha virt. eigenvalues --   -0.15000   0.20000",
    ]
    homo, lumo = _frontier_ev(lines, "gaussian")
    assert homo == pytest.approx(-0.33523 * 27.2114, abs=1e-3)
    assert lumo == pytest.approx(-0.15 * 27.2114, abs=1e-3)


# ─────────────────────────────────────────────
# LLM suggestions (FakeProvider)
# ─────────────────────────────────────────────

class FakeProvider(Provider):
    name = "fake"
    def __init__(self):
        self.last_prompt = None
        self.last_system = None
    def complete(self, prompt, *, system=None, max_tokens=None):
        self.last_prompt = prompt
        self.last_system = system
        return LLMResponse(text="1. Usa MOREAD desde el .gbw del neutro.",
                           model="fake-1", provider="fake")

def test_suggest_fixes_sends_diagnosis_json(diverging):
    diag = diagnose(diverging)
    fake = FakeProvider()
    result = suggest_fixes(diag, language="es", provider=fake)
    sent = json.loads(fake.last_prompt)
    assert sent["engine"] == "orca"
    assert "MOREAD" in fake.last_system
    assert "español" in fake.last_system
    assert "MOREAD" in result.text

def test_suggest_fixes_bad_language(diverging):
    with pytest.raises(ValueError, match="Unsupported language"):
        suggest_fixes(diagnose(diverging), language="de", provider=FakeProvider())


# ─────────────────────────────────────────────
# CLI: cadmia diagnose
# ─────────────────────────────────────────────

def test_cli_diagnose_no_llm(diverging, tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main

    out_json = tmp_path / "diag.json"
    result = CliRunner().invoke(main, ["diagnose", str(diverging),
                                       "--no-llm", "-o", str(out_json)])
    assert result.exit_code == 0, result.output
    assert "scf_not_converged" in result.output
    assert "negative_gap" in result.output
    assert json.loads(out_json.read_text())["engine"] == "orca"

def test_cli_diagnose_healthy_skips_llm(healthy):
    from click.testing import CliRunner
    from cadmia.cli import main

    result = CliRunner().invoke(main, ["diagnose", str(healthy)])
    assert result.exit_code == 0, result.output
    assert "No convergence symptoms detected" in result.output
    assert "Suggested strategies" not in result.output

def test_cli_diagnose_with_fake_llm(diverging, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main
    import cadmia.llm.diagnostician as diag_mod

    monkeypatch.setattr(diag_mod, "get_provider", lambda config=None: FakeProvider())
    result = CliRunner().invoke(main, ["diagnose", str(diverging)])
    assert result.exit_code == 0, result.output
    assert "Suggested strategies" in result.output
    assert "MOREAD" in result.output

def test_cli_diagnose_without_key_degrades(diverging, monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(main, ["diagnose", str(diverging)])
    assert result.exit_code == 0, result.output
    assert "LLM suggestions unavailable" in result.output
    assert "scf_not_converged" in result.output


# ─────────────────────────────────────────────
# CP-SCF response failure (analytic Hessian)
# ─────────────────────────────────────────────

_CPSCF_ORCA = """\
                                 O   R   C   A

----------------------------------------D-I-I-S--------------------------------------------
Iteration    Energy (Eh)           Delta-E    RMSDP     MaxDP
   0   -1709.10000000   0.000000000     0.10        0.01
   1   -1709.16000000  -0.060000000     0.001       0.0001
SCF CONVERGED AFTER 2 CYCLES

ORBITAL ENERGIES
  NO   OCC          E(Eh)            E(eV)
   0   2.0000     -10.000000      -272.1140
   1   2.0000      -0.328000        -8.9253
   2   0.0000      -0.118000        -3.2109

FINAL SINGLE POINT ENERGY      -1709.16000000

[file orca_tools/DRIVERS/shark_driver_cpscf_solver.cpp, line 1401, Process 6]: \
Error (SHARK/CP-SCF Solver): Unfortunately, the calculation did not converge. \
Try a different solver or more iterations in case it was close

ORCA finished by error termination in SCF RESPONSE
[file orca_tools/qcmsg.cpp, line 394]:
  .... aborting the run
"""


def test_cpscf_failure_is_reported_and_geometry_data_survives(tmp_path):
    p = tmp_path / "bidentate.out"
    p.write_text(_CPSCF_ORCA)
    d = diagnose(p)

    codes = {s.code for s in d.symptoms}
    assert "cpscf_not_converged" in codes
    # The reference SCF converged and its numbers are still usable: that is the whole
    # point of separating this symptom from a generic error termination.
    assert d.scf_converged is True
    assert d.energy_hartree == pytest.approx(-1709.16)
    assert d.gap_ev == pytest.approx(5.714, abs=1e-3)
    assert "small_gap" not in codes


def test_healthy_output_has_no_cpscf_symptom(healthy):
    assert "cpscf_not_converged" not in {s.code for s in diagnose(healthy).symptoms}


# ─────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────

def test_cli_diagnose_batch_reports_each_and_skips_the_llm(diverging, healthy, tmp_path,
                                                           monkeypatch):
    from click.testing import CliRunner
    from cadmia.cli import main
    import cadmia.llm.diagnostician as diag_mod

    # The provider is available: the batch must still not call it, or triaging a campaign
    # would fan out into one paid request per output.
    monkeypatch.setattr(diag_mod, "get_provider", lambda config=None: FakeProvider())
    out = tmp_path / "batch.json"
    result = CliRunner().invoke(main, ["diagnose", str(diverging), str(healthy),
                                       "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "cation.out" in result.output and "neutral.out" in result.output
    assert "2 output(s) diagnosed, 1 with symptoms." in result.output
    assert "Suggested strategies" not in result.output

    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and len(payload) == 2
    assert {d["healthy"] for d in payload} == {True, False}


def test_cli_diagnose_single_file_json_stays_an_object(diverging, tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main

    out = tmp_path / "one.json"
    result = CliRunner().invoke(main, ["diagnose", str(diverging), "--no-llm", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert isinstance(json.loads(out.read_text()), dict)


# ─────────────────────────────────────────────
# Imaginary frequencies
# ─────────────────────────────────────────────

def _orca_with_freqs(*freqs: float) -> str:
    """A converged ORCA output carrying a vibrational block with the given frequencies."""
    modes = "\n".join(f"   {i}:   {f:>10.2f} cm**-1" for i, f in enumerate(freqs))
    return _HEALTHY_ORCA.replace(
        "                   ****ORCA TERMINATED NORMALLY****",
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n\n"
        f"{modes}\n\n"
        "NORMAL MODES\n\n"
        "                   ****ORCA TERMINATED NORMALLY****",
    )


def test_several_imaginary_frequencies_are_critical(tmp_path):
    p = tmp_path / "eight.out"
    p.write_text(_orca_with_freqs(0, 0, 0, 0, 0, 0, -145.2, -88.1, 320.6))
    d = diagnose(p)

    assert d.n_imaginary == 2
    sym = next(s for s in d.symptoms if s.code == "imaginary_frequencies")
    assert sym.severity == "critical"


def test_one_imaginary_frequency_is_only_a_warning(tmp_path):
    # A first-order saddle point is a legitimate transition state, not a broken run.
    p = tmp_path / "ts.out"
    p.write_text(_orca_with_freqs(0, 0, 0, 0, 0, 0, -410.7, 220.1, 315.9))
    d = diagnose(p)

    assert d.n_imaginary == 1
    assert next(s for s in d.symptoms if s.code == "imaginary_frequencies").severity == "warning"


def test_a_clean_minimum_reports_zero_and_stays_healthy(tmp_path):
    p = tmp_path / "min.out"
    p.write_text(_orca_with_freqs(0, 0, 0, 0, 0, 0, 145.2, 220.1, 315.9))
    d = diagnose(p)

    assert d.n_imaginary == 0
    assert d.healthy


def test_no_frequencies_is_a_fact_not_a_symptom(healthy):
    # A single point has no frequencies by design; flagging that would mark every SCF-only
    # run (CDFT included) as unhealthy.
    d = diagnose(healthy)
    assert d.n_imaginary is None
    assert d.healthy


# ─────────────────────────────────────────────
# Geometry optimization convergence
# ─────────────────────────────────────────────

def _orca_opt(converged: bool, cycles: int = 3) -> str:
    """A normally-terminated ORCA optimization that did or did not converge."""
    body = "\n".join(f"*** GEOMETRY OPTIMIZATION CYCLE {i + 1} ***" for i in range(cycles))
    tail = ("THE OPTIMIZATION HAS CONVERGED" if converged else
            "The optimization did not converge but reached the maximum "
            "number of optimization cycles.")
    return _HEALTHY_ORCA.replace(
        "                   ****ORCA TERMINATED NORMALLY****",
        f"{body}\n{tail}\n\n                   ****ORCA TERMINATED NORMALLY****",
    )


def test_optimization_out_of_cycles_is_critical_despite_normal_termination(tmp_path):
    # The exact trap: ORCA gives up, skips Freq, and still prints TERMINATED NORMALLY, so
    # every other check passes and the unconverged energy flows into rank/analyze.
    p = tmp_path / "stalled.out"
    p.write_text(_orca_opt(converged=False, cycles=54))
    d = diagnose(p)

    assert d.terminated_normally is True
    assert d.scf_converged is True
    sym = next(s for s in d.symptoms if s.code == "geometry_not_converged")
    assert sym.severity == "critical"
    assert "54" in sym.message


def test_converged_optimization_is_clean(tmp_path):
    p = tmp_path / "opt.out"
    p.write_text(_orca_opt(converged=True))
    assert diagnose(p).healthy


def test_single_point_has_no_optimization_to_judge(healthy):
    # No optimization in the file at all — must not be reported as a failed one.
    from cadmia.agents import validation_agent
    assert validation_agent.run(healthy).checks["geometry_optimization"].status == "skip"
    assert diagnose(healthy).healthy


def test_gaussian_optimization_out_of_steps(tmp_path):
    from cadmia.agents import validation_agent
    p = tmp_path / "g.log"
    p.write_text("Gaussian, Inc.\n Berny optimization.\n Step number 64\n"
                 " Number of steps exceeded,  NStep=  64\n Normal termination of Gaussian 16\n")
    check = validation_agent.run(p, engine="gaussian").checks["geometry_optimization"]
    assert check.status == "fail"
