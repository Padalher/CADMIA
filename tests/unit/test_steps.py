"""`cadmia xyz --steps`: every step of an optimization as extended XYZ with energy and forces."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from cadmia.cli import main
from cadmia.core.units import BOHR_TO_ANG, HARTREE_TO_EV
from cadmia.io.steps import OptimizationStep, OptimizationSteps, parse_steps, steps_extxyz
from cadmia.io.xyz import comment_info, extxyz_columns, parse_xyz_database

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_OPT = _FIXTURES / "1tial_d4" / "1tial.out"
_GAUSSIAN = _FIXTURES / "TiO2_monomer.out"
_ORCA_NO_GRADIENT = _FIXTURES / "1tial_orca" / "1tial_N_orca.out"


def _orca_lines():
    return _OPT.read_text(encoding="utf-8").splitlines()


def test_orca_steps_are_the_cycles_with_gradient():
    lines = _orca_lines()
    cycles = sum("GEOMETRY OPTIMIZATION CYCLE" in ln for ln in lines)
    gradients = sum(ln.strip() == "CARTESIAN GRADIENT" for ln in lines)
    energies = [float(ln.split()[-1]) for ln in lines if ln.startswith("FINAL SINGLE POINT ENERGY")]
    result = parse_steps(_OPT)
    assert len(result.steps) == cycles == gradients == 6
    assert all(s.forces_eh_bohr is not None for s in result.steps)
    assert [s.energy_hartree for s in result.steps] == energies[:6]
    assert len(energies) == 7
    assert result.charge == 0 and result.multiplicity == 1


def test_orca_force_is_minus_the_gradient_of_the_output():
    lines = _orca_lines()
    first = next(i for i, ln in enumerate(lines) if ln.strip() == "CARTESIAN GRADIENT")
    gradient_x = float(lines[first + 3].split()[3])
    step = parse_steps(_OPT).steps[0]
    assert step.forces_eh_bohr[0, 0] == pytest.approx(-gradient_x)


def test_cli_steps_writes_units_in_ev_per_angstrom(tmp_path):
    out = tmp_path / "opt_steps.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_OPT), "--steps", "-o", str(out)])
    assert res.exit_code == 0, res.output
    text = out.read_text(encoding="utf-8")
    comments = [ln for ln in text.splitlines() if ln.startswith("format=")]
    assert len(comments) == 6
    info = comment_info(comments[0])
    assert info["step"] == "1" and info["forces_unit"] == "eV/Ang" and info["energy_unit"] == "Eh"
    assert info["Properties"] == "species:S:1:pos:R:3:forces:R:3"
    assert extxyz_columns(info) == [("forces", "R", 3)]
    assert comment_info(comments[-1])["step"] == "6"

    step = parse_steps(_OPT).steps[0]
    first_atom = text.splitlines()[2].split()
    expected = step.forces_eh_bohr[0, 0] * HARTREE_TO_EV / BOHR_TO_ANG
    assert float(first_atom[4]) == pytest.approx(expected, abs=1e-6)
    assert float(first_atom[4]) == pytest.approx(0.015218, abs=1e-6)

    assert all(len(ln.split()) == 7 for ln in text.splitlines() if re.match(r"^[A-Z][a-z]? ", ln))
    assert len(parse_xyz_database(out)) == 6


def test_a_step_without_gradient_is_written_with_nan_forces():
    step = OptimizationStep(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]), -1.1, None)
    text = steps_extxyz(OptimizationSteps("orca", 0, 1, [step]), "h2")
    atoms = text.splitlines()[2:]
    assert [ln.split()[4:] for ln in atoms] == [["nan"] * 3] * 2
    assert "Properties=species:S:1:pos:R:3:forces:R:3" in text


def test_cli_steps_default_output_sits_next_to_the_source(tmp_path):
    source = tmp_path / "run.out"
    source.write_text(_OPT.read_text(encoding="utf-8"), encoding="utf-8")
    res = CliRunner().invoke(main, ["xyz", str(source), "--steps"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "run_steps.xyz").exists()


def test_steps_without_any_gradient_is_an_error(tmp_path):
    out = tmp_path / "x_steps.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_ORCA_NO_GRADIENT), "--steps", "-o", str(out)])
    assert res.exit_code != 0 and "no gradient" in res.output
    assert not out.exists()


def test_without_steps_the_command_is_unchanged(tmp_path):
    out = tmp_path / "last.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_OPT), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_text(encoding="utf-8").splitlines()[0].strip() == "8"
    assert "step=" not in out.read_text(encoding="utf-8")


def test_gaussian_forces_are_used_as_printed():
    result = parse_steps(_GAUSSIAN)
    first = result.steps[0]
    assert first.atoms == ["Ti", "O", "O"]
    assert first.energy_hartree == pytest.approx(-999.651664736)
    assert first.forces_eh_bohr[0, 2] == pytest.approx(-0.000073432)
    assert math.isfinite(float(np.sum(first.forces_eh_bohr)))


_GAUSSIAN_LOG = """ Charge =  0 Multiplicity = 1
                          Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          1           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    0.740000
 ---------------------------------------------------------------------
                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          1           0        0.000000    0.000000   -0.370000
      2          1           0        0.000000    0.000000    0.370000
 ---------------------------------------------------------------------
 SCF Done:  E(RHF) =  -1.100000000     A.U. after    5 cycles
 ***** Axes restored to original set *****
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        1           0.000000000    0.000000000    0.010000000
      2        1           0.000000000    0.000000000   -0.010000000
 -------------------------------------------------------------------
"""


def test_gaussian_forces_are_paired_with_the_input_orientation(tmp_path):
    log = tmp_path / "h2.log"
    log.write_text(_GAUSSIAN_LOG, encoding="utf-8")
    step = parse_steps(log).steps[0]
    assert step.positions[1].tolist() == [0.0, 0.0, 0.74]
    assert step.positions[0].tolist() == [0.0, 0.0, 0.0]
    assert step.forces_eh_bohr[0, 2] == pytest.approx(0.01)


def test_gaussian_with_only_the_standard_orientation_is_an_error(tmp_path):
    log = tmp_path / "h2.log"
    text = _GAUSSIAN_LOG.replace("Input orientation", "Other orientation")
    log.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Input or Z-Matrix"):
        parse_steps(log, engine="gaussian")


def test_cli_gaussian_force_column_is_the_printed_force_in_ev_per_angstrom(tmp_path):
    out = tmp_path / "ti_steps.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_GAUSSIAN), "--steps", "-o", str(out)])
    assert res.exit_code == 0, res.output
    atom1 = out.read_text(encoding="utf-8").splitlines()[2].split()
    assert atom1[0] == "Ti"
    printed = -0.000073432
    assert float(atom1[6]) == pytest.approx(printed * HARTREE_TO_EV / BOHR_TO_ANG, abs=1e-6)
    assert float(atom1[6]) == pytest.approx(printed * 51.4221, rel=1e-3)


def test_orca_gradient_truncated_midway_is_an_error(tmp_path):
    lines = _orca_lines()
    header = [i for i, ln in enumerate(lines) if ln.strip() == "CARTESIAN GRADIENT"][1]
    cut = tmp_path / "cut.out"
    cut.write_text("\n".join(lines[:header + 6]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ORCA step 2: the gradient has 3 rows for 8 atoms"):
        parse_steps(cut)
    res = CliRunner().invoke(main, ["xyz", str(cut), "--steps", "-o", str(tmp_path / "o.xyz")])
    assert res.exit_code != 0 and "3 rows for 8 atoms" in res.output
    assert not (tmp_path / "o.xyz").exists()


def test_orca_non_finite_values_are_errors(tmp_path):
    text = _OPT.read_text(encoding="utf-8")
    bad_gradient = tmp_path / "g.out"
    bad_gradient.write_text(text.replace("-0.000295941", "nan", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="ORCA step 1: the gradient has 7 rows for 8 atoms"):
        parse_steps(bad_gradient)
    bad_geometry = tmp_path / "x.out"
    lines = text.splitlines()
    cycle = next(i for i, ln in enumerate(lines) if "GEOMETRY OPTIMIZATION CYCLE" in ln)
    first_atom = next(i for i in range(cycle, len(lines)) if lines[i].startswith("  Ti "))
    lines[first_atom] = lines[first_atom].replace("1.185391", "inf", 1)
    bad_geometry.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ORCA step 1"):
        parse_steps(bad_geometry)


def test_orca_with_two_jobs_keeps_every_cycle(tmp_path):
    text = _OPT.read_text(encoding="utf-8")
    both = tmp_path / "two.out"
    both.write_text(text + text, encoding="utf-8")
    assert len(parse_steps(both).steps) == 12


def test_unreadable_or_unwritable_paths_are_clean_errors(tmp_path):
    res = CliRunner().invoke(main, ["xyz", str(_OPT), "--steps", "-o", str(tmp_path / "no" / "o.xyz")])
    assert res.exit_code != 0 and "cannot write" in res.output
    assert "Traceback" not in res.output


def test_label_backslash_is_replaced_in_the_comment_line():
    from cadmia.io.xyz import extxyz_comment
    assert "label=a_b" in extxyz_comment({"label": "a\\b"})


def test_many_consecutive_orientation_headers_are_read_in_linear_time_without_phantoms(tmp_path):
    import time

    from cadmia.io.steps import _input_frame_blocks

    header = "                          Input orientation:"
    lines = [header] * 20000 + _GAUSSIAN_LOG.splitlines()[1:]
    start = time.perf_counter()
    blocks = _input_frame_blocks(lines)
    assert time.perf_counter() - start < 2.0
    assert len(blocks) == 1
    assert blocks[0][0] == 20000 and blocks[0][1] == ["H", "H"]


def test_long_scf_done_line_without_closing_parenthesis_is_fast():
    import time

    from cadmia.io.steps import _GAUSSIAN_ENERGY_RE

    line = "SCF Done:  " + "E(" * 50000
    start = time.perf_counter()
    assert _GAUSSIAN_ENERGY_RE.search(line) is None
    assert time.perf_counter() - start < 2.0


def test_non_finite_energy_and_forces_are_errors(tmp_path):
    huge = "9" * 400 + ".0"
    orca = tmp_path / "e.out"
    orca.write_text(_OPT.read_text(encoding="utf-8").replace("-1710.170757736594", "-" + huge, 1),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="ORCA step 1: the energy is not a finite number"):
        parse_steps(orca)
    gaussian = tmp_path / "e.log"
    gaussian.write_text(_GAUSSIAN_LOG.replace("-1.100000000", "-" + huge), encoding="utf-8")
    with pytest.raises(ValueError, match="Gaussian step 1: the energy is not a finite number"):
        parse_steps(gaussian)
    forces = tmp_path / "f.log"
    forces.write_text(_GAUSSIAN_LOG.replace("0.010000000", huge, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Gaussian step 1: a force component"):
        parse_steps(forces)
    gradient = tmp_path / "g.out"
    gradient.write_text(_OPT.read_text(encoding="utf-8").replace("-0.000295941", huge, 1),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="ORCA step 1: a gradient component"):
        parse_steps(gradient)


def test_orca_charge_and_multiplicity_lines_are_read_in_linear_time():
    import time

    from cadmia.io.orca import _parse_orca_charge_mult_pg

    pathological = ["Total Charge" + " " * 50000 + "x", "Multiplicity" + " " * 50000 + "x"]
    start = time.perf_counter()
    assert _parse_orca_charge_mult_pg(pathological)[:2] == (0, 1)
    assert time.perf_counter() - start < 2.0
    normal = [" Total Charge           Charge          ....    -1",
              " Multiplicity           Mult            ....    2"]
    assert _parse_orca_charge_mult_pg(normal)[:2] == (-1, 2)
