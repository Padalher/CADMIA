"""Tests for workflow stages (`cadmia prepare -s <stage>`)."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cadmia.calculators.params import CalcParams
from cadmia.calculators.stages import STAGES, apply_stage, get_stage
from cadmia.cli import main

_WATER = """3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""


@pytest.fixture
def water_xyz(tmp_path) -> Path:
    p = tmp_path / "mol.xyz"
    p.write_text(_WATER)
    return p


def _header(inp_path: Path) -> str:
    return inp_path.read_text(encoding="utf-8").splitlines()[0]


# --- unit: apply_stage -------------------------------------------------------

def test_stage_tokens_are_the_eight_expected():
    assert set(STAGES) == {"screen-opt", "screen-freq", "opt", "freq",
                           "opt+freq", "char", "cdft", "bsse"}


def test_apply_stage_screen_opt_is_bare_opt_in_svp():
    params, stage = apply_stage(CalcParams(), "screen-opt", engine="orca")
    assert params.task == "opt"
    assert params.basis == "def2-SVP"                   # cheap basis for the screen
    assert "RIJCOSX" in params.extra_orca and "def2/J" in params.extra_orca
    assert "TightSCF" not in params.extra_orca          # screen stays cheap
    assert not params.orca_output_flags                 # no analysis
    assert stage.preset == "none"


def test_apply_stage_screen_freq_matches_screen_opt_level():
    # A Freq is only meaningful on a geometry optimized at the SAME level, and the level
    # includes the SCF threshold and the grid: screen-freq must stay as loose as screen-opt,
    # or the screen-opt minima come back with spurious imaginary modes.
    freq, _ = apply_stage(CalcParams(), "screen-freq", engine="orca")
    opt, _  = apply_stage(CalcParams(), "screen-opt", engine="orca")
    assert freq.task == "freq"
    assert freq.basis == opt.basis == "def2-SVP"
    assert freq.extra_orca == opt.extra_orca
    assert "TightSCF" not in freq.extra_orca and "DefGrid3" not in freq.extra_orca
    assert not freq.orca_output_flags


def test_apply_stage_production_freq_stays_tight():
    params, _ = apply_stage(CalcParams(), "freq", engine="orca")
    assert "TightSCF" in params.extra_orca and "DefGrid3" in params.extra_orca


def test_apply_stage_optfreq_combines_opt_and_freq():
    params, _ = apply_stage(CalcParams(), "opt+freq", engine="orca")
    assert params.task == "opt+freq"          # plain Opt: TightOpt never converges a free rotor
    assert not params.orca_output_flags


def test_apply_stage_explicit_basis_kept():
    params, _ = apply_stage(CalcParams(basis="def2-TZVPP"), "screen-opt",
                            engine="orca", set_basis=False)
    assert params.basis == "def2-TZVPP"                 # explicit --basis wins


def test_apply_stage_opt_keeps_the_accuracy_block_no_analysis():
    params, _ = apply_stage(CalcParams(), "opt", engine="orca")
    assert params.task == "opt"
    for kw in ("RIJCOSX", "def2/J", "TightSCF", "DefGrid3"):
        assert kw in params.extra_orca
    assert not params.orca_output_flags


def test_apply_stage_char_carries_full_analysis():
    params, _ = apply_stage(CalcParams(), "char", engine="orca")
    assert params.task == "sp"
    assert set(params.orca_output_flags) == {"P_Hirshfeld", "P_Mayer", "P_OrbPopMO_L"}
    assert "CHELPG" in params.extra_orca                 # from the analysis preset


def test_apply_stage_respects_explicit_preset():
    params, _ = apply_stage(CalcParams(), "char", engine="orca", set_preset=False)
    assert params.task == "sp"                           # stage still owns the task
    assert not params.orca_output_flags                  # preset skipped (kept explicit)


def test_apply_stage_skips_orca_keywords_for_gaussian():
    params, _ = apply_stage(CalcParams(), "opt", engine="gaussian")
    assert "RIJCOSX" not in params.extra_orca            # ORCA-only, not added


def test_get_stage_unknown_raises():
    with pytest.raises(ValueError):
        get_stage("nope")


# --- CLI: prepare <stage> ----------------------------------------------------

def test_cli_prepare_stage_screen_opt(water_xyz, tmp_path):
    out = tmp_path / "screen"
    res = CliRunner().invoke(main, ["prepare", "screen-opt", str(water_xyz), "-o", str(out)])
    assert res.exit_code == 0, res.output
    header = _header(out / "mol.inp")
    assert "Opt" in header and "RIJCOSX" in header and "def2-SVP" in header
    assert "TightOpt" not in header and "Print[" not in (out / "mol.inp").read_text(encoding="utf-8")


def test_cli_prepare_stage_optfreq(water_xyz, tmp_path):
    out = tmp_path / "optfreq"
    res = CliRunner().invoke(main, ["prepare", "opt+freq", str(water_xyz), "-o", str(out)])
    assert res.exit_code == 0, res.output
    header = _header(out / "mol.inp")
    assert "Opt Freq" in header and "def2-TZVP" in header
    assert "TightOpt" not in header
    for kw in ("RIJCOSX", "def2/J", "TightSCF", "DefGrid3"):   # accuracy block still applies
        assert kw in header


def test_cli_prepare_stage_screen_freq(water_xyz, tmp_path):
    out = tmp_path / "screenfreq"
    res = CliRunner().invoke(main, ["prepare", "screen-freq", str(water_xyz), "-o", str(out)])
    assert res.exit_code == 0, res.output
    header = _header(out / "mol.inp")
    assert "Freq" in header and "def2-SVP" in header
    assert "TightSCF" not in header and "Opt" not in header


def test_cli_prepare_reports_generated_not_passed_through(water_xyz, tmp_path):
    ready = tmp_path / "ready.inp"
    ready.write_text("! PBE0 def2-TZVP Opt\n")
    res = CliRunner().invoke(main, ["prepare", "screen-opt", str(ready),
                                    "-o", str(tmp_path / "o")])
    assert res.exit_code == 0, res.output
    assert "0 input(s) generated" in res.output
    assert "1 ready input(s) passed through" in res.output


def test_cli_prepare_unsupported_extension_fails_cleanly(tmp_path):
    stray = tmp_path / "notes.txt"
    stray.write_text("not a geometry")
    res = CliRunner().invoke(main, ["prepare", "screen-opt", str(stray),
                                    "-o", str(tmp_path / "o")])
    assert res.exit_code != 0
    assert "cannot prepare an input from a '.txt' file" in res.output


def test_cli_prepare_basis_override(water_xyz, tmp_path):
    # screen-opt defaults to def2-SVP; an explicit --basis must win.
    out = tmp_path / "svpover"
    res = CliRunner().invoke(main, ["prepare", "screen-opt", str(water_xyz),
                                    "-o", str(out), "--basis", "def2-TZVP"])
    assert res.exit_code == 0, res.output
    assert "def2-TZVP" in _header(out / "mol.inp") and "def2-SVP" not in _header(out / "mol.inp")


def test_cli_prepare_stage_char_has_output_block(water_xyz, tmp_path):
    out = tmp_path / "char"
    res = CliRunner().invoke(main, ["prepare", "char", str(water_xyz), "-o", str(out)])
    assert res.exit_code == 0, res.output
    text = (out / "mol.inp").read_text(encoding="utf-8")
    assert "SP" in _header(out / "mol.inp")
    for flag in ("P_Hirshfeld", "P_Mayer", "P_OrbPopMO_L"):
        assert flag in text


def test_cli_prepare_stage_cdft_writes_triplet(water_xyz, tmp_path):
    out = tmp_path / "cdft"
    res = CliRunner().invoke(main, ["prepare", "cdft", str(water_xyz), "-o", str(out)])
    assert res.exit_code == 0, res.output
    names = sorted(p.name for p in out.glob("*.inp"))
    assert names == ["mol_N.inp", "mol_anion.inp", "mol_cation.inp"]


def test_cli_prepare_no_task_option(water_xyz, tmp_path):
    # --task was removed; the stage owns the task.
    res = CliRunner().invoke(main, ["prepare", "opt", str(water_xyz),
                                    "-o", str(tmp_path / "o"), "--task", "sp"])
    assert res.exit_code != 0
    assert "no such option" in res.output.lower()


def test_cli_prepare_explicit_preset_overrides_stage(water_xyz, tmp_path):
    # screen carries no analysis; --preset analysis must add the property block back.
    out = tmp_path / "pov"
    res = CliRunner().invoke(main, ["prepare", "screen-opt", str(water_xyz),
                                    "-o", str(out), "--preset", "analysis"])
    assert res.exit_code == 0, res.output
    assert "P_Mayer" in (out / "mol.inp").read_text(encoding="utf-8")


def test_cli_prepare_multiple_geometries(water_xyz, tmp_path):
    other = tmp_path / "mol2.xyz"
    other.write_text(_WATER)
    out = tmp_path / "multi"
    res = CliRunner().invoke(main, ["prepare", "opt", str(water_xyz), str(other), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert {p.name for p in out.glob("*.inp")} == {"mol.inp", "mol2.inp"}


def test_cli_prepare_requires_geometries(tmp_path):
    res = CliRunner().invoke(main, ["prepare", "opt"])
    assert res.exit_code != 0
    assert "give one or more geometries" in res.output


def test_cli_prepare_unknown_stage_errors():
    res = CliRunner().invoke(main, ["prepare", "bogus", "x.xyz"])
    assert res.exit_code != 0
    assert "bogus" in res.output or "invalid choice" in res.output.lower()


def test_cli_prepare_cdft_command_is_gone():
    res = CliRunner().invoke(main, ["prepare-cdft", "--help"])
    assert res.exit_code != 0                    # no such command anymore


def test_gaussian_pop_keywords_are_folded_into_one(tmp_path):
    """Gaussian must receive a single pop=(...), not two.

    _merge_keywords compares whole tokens, so an existing `pop=hirshfeld` did not look like a
    duplicate of the preset's `pop=(Hirshfeld,CHELPG)` and both ended up in the route section.
    """
    from cadmia.calculators.params import CalcParams
    from cadmia.calculators.presets import apply_preset

    params = CalcParams(nprocs=4, mem_gaussian_gb=8, mem_orca_mb_core=1000,
                        extra_gaussian="pop=hirshfeld scf=tight")
    merged = apply_preset(params, "analysis", engine="gaussian").extra_gaussian

    assert merged.lower().count("pop=") == 1
    assert "scf=tight" in merged                       # unrelated keywords survive
    inner = merged.split("pop=(", 1)[1].rstrip(")").lower()
    assert "hirshfeld" in inner and "chelpg" in inner  # neither request is dropped
    assert inner.count("hirshfeld") == 1               # and it is not repeated inside
