"""Tests for cadmia benchmark stack: params, calculators, utils/geometry, benchmark_agent."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from cadmia.calculators.params import CalcParams
from cadmia.calculators.gaussian import write_gjf
from cadmia.calculators.orca import write_inp
from cadmia.utils.geometry import kabsch_rmsd, kabsch_max_deviation, align
from cadmia.agents.benchmark_agent import (
    BenchmarkResult,
    _parse_gaussian_energy,
    _parse_orca_energy,
    _parse_gaussian_frequencies,
    _parse_orca_frequencies,
    _parse_gaussian_wall_time,
    _parse_orca_wall_time,
    _nontrivial,
    generate,
    compare,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def simple_atoms():
    return ["Ti", "O", "O"]

@pytest.fixture
def simple_positions():
    return np.array([[0.0, 0.0, 0.0], [1.95, 0.0, 0.0], [-1.95, 0.0, 0.0]])

@pytest.fixture
def simple_params():
    return CalcParams(functional="PBE0", basis="def2-TZVP", dispersion="d3bj",
                      task="opt", nprocs=4, mem_gaussian_gb=8, mem_orca_mb_core=1000)

@pytest.fixture
def simple_xyz(tmp_path, simple_atoms, simple_positions):
    p = tmp_path / "cluster.xyz"
    lines = ["3", "Ti2O test", "Ti   0.00000000   0.00000000   0.00000000",
             "O    1.95000000   0.00000000   0.00000000",
             "O   -1.95000000   0.00000000   0.00000000"]
    p.write_text("\n".join(lines))
    return p


# ─────────────────────────────────────────────
# CalcParams
# ─────────────────────────────────────────────

def test_calcparams_defaults():
    p = CalcParams()
    assert p.functional == "PBE0"
    assert p.dispersion == "d4"
    assert p.task == "opt"

def test_calcparams_invalid_task():
    with pytest.raises(ValueError, match="task must be one of"):
        CalcParams(task="nbo")

def test_calcparams_invalid_dispersion():
    with pytest.raises(ValueError, match="dispersion must be one of"):
        CalcParams(dispersion="d3abc")

def test_calcparams_normalizes_case():
    p = CalcParams(dispersion="D3BJ", task="OPT+FREQ")
    assert p.dispersion == "d3bj"
    assert p.task == "opt+freq"


# ─────────────────────────────────────────────
# Gaussian input writer
# ─────────────────────────────────────────────

def test_write_gjf_creates_file(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.gjf"
    result = write_gjf(simple_atoms, simple_positions, simple_params, out)
    assert result.exists()

def test_write_gjf_nprocs_mem(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert "%nprocshared=4" in content
    assert "%mem=8GB" in content

def test_write_gjf_route_card(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    # Gaussian's own spelling: PBE0 is PBE1PBE and the def2 family loses the hyphen.
    assert "# PBE1PBE/Def2TZVP" in content
    assert "Opt" in content
    assert "EmpiricalDispersion=GD3BJ" in content

def test_write_gjf_charge_mult(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert "0 1" in content

def test_write_gjf_coordinates(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert "Ti" in content
    assert "1.95000000" in content

def test_write_gjf_ends_with_blank_lines(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert content.endswith("\n\n")

def test_write_gjf_no_dispersion(tmp_path, simple_atoms, simple_positions):
    params = CalcParams(dispersion="none")
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, params, out)
    content = out.read_text()
    assert "EmpiricalDispersion" not in content

def test_write_gjf_refuses_d4(tmp_path, simple_atoms, simple_positions):
    """Gaussian 16 implements GD2/GD3/GD3BJ/PFD only — there is no native D4. Since d4 is
    the CADMIA default (ORCA is the primary engine), the writer used to emit
    `EmpiricalDispersion=GD4` and every .gjf built with defaults was an input G16 rejects
    at startup. Refusing beats emitting a broken keyword, and beats silently dropping the
    dispersion (which would change the level of theory without saying so)."""
    params = CalcParams(task="sp")                      # dispersion defaults to d4
    with pytest.raises(ValueError, match=r"'d4' is not available in Gaussian 16"):
        write_gjf(simple_atoms, simple_positions, params, tmp_path / "d4.gjf")


def test_write_gjf_d3bj_is_emitted(tmp_path, simple_atoms, simple_positions):
    params = CalcParams(task="sp", dispersion="d3bj")
    out = write_gjf(simple_atoms, simple_positions, params, tmp_path / "d3bj.gjf")
    assert "EmpiricalDispersion=GD3BJ" in out.read_text()


def test_write_gjf_no_dispersion_emits_none(tmp_path, simple_atoms, simple_positions):
    params = CalcParams(task="sp", dispersion="none")
    out = write_gjf(simple_atoms, simple_positions, params, tmp_path / "no.gjf")
    assert "EmpiricalDispersion" not in out.read_text()


def test_write_inp_still_takes_d4(tmp_path, simple_atoms, simple_positions):
    """The refusal is Gaussian-only: D4 is valid in ORCA and stays the default there."""
    from cadmia.calculators.orca import write_inp
    out = write_inp(simple_atoms, simple_positions, CalcParams(task="sp"), tmp_path / "o.inp")
    assert " D4 " in out.read_text().splitlines()[0]


def test_write_gjf_freq_task(tmp_path, simple_atoms, simple_positions):
    params = CalcParams(task="opt+freq", dispersion="d3bj")   # G16 has no native D4
    out = tmp_path / "test.gjf"
    write_gjf(simple_atoms, simple_positions, params, out)
    content = out.read_text()
    assert "Opt" in content and "Freq" in content


# ─────────────────────────────────────────────
# ORCA input writer
# ─────────────────────────────────────────────

def test_write_inp_creates_file(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.inp"
    result = write_inp(simple_atoms, simple_positions, simple_params, out)
    assert result.exists()

def test_write_inp_simple_input_line(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.inp"
    write_inp(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    first_line = content.splitlines()[0]
    assert first_line.startswith("!")
    assert "PBE0" in first_line
    assert "def2-TZVP" in first_line
    assert "D3BJ" in first_line
    assert "Opt" in first_line

def test_write_inp_parallel_block(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.inp"
    write_inp(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert "%pal nprocs 4 end" in content
    assert "%maxcore 1000" in content

def test_write_inp_xyz_block(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.inp"
    write_inp(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert "* xyz 0 1" in content
    assert content.rstrip().endswith("*")

def test_write_inp_coordinates(tmp_path, simple_atoms, simple_positions, simple_params):
    out = tmp_path / "test.inp"
    write_inp(simple_atoms, simple_positions, simple_params, out)
    content = out.read_text()
    assert "Ti" in content
    assert "1.95000000" in content

def test_write_inp_no_dispersion(tmp_path, simple_atoms, simple_positions):
    params = CalcParams(dispersion="none")
    out = tmp_path / "test.inp"
    write_inp(simple_atoms, simple_positions, params, out)
    simple_line = out.read_text().splitlines()[0]
    assert "D3BJ" not in simple_line
    assert "D4" not in simple_line

def test_write_inp_sp_task(tmp_path, simple_atoms, simple_positions):
    params = CalcParams(task="sp")
    out = tmp_path / "test.inp"
    write_inp(simple_atoms, simple_positions, params, out)
    assert "SP" in out.read_text().splitlines()[0]


# ─────────────────────────────────────────────
# Kabsch RMSD
# ─────────────────────────────────────────────

def test_rmsd_identical_structures():
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    assert kabsch_rmsd(P, P) < 1e-10


def test_rmsd_pure_translation_is_zero():
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = P + np.array([5.0, 3.0, -2.0])
    assert kabsch_rmsd(P, Q) < 1e-10


def test_rmsd_pure_rotation_is_zero():
    P = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    angle = math.pi / 3
    R = np.array([[math.cos(angle), -math.sin(angle), 0],
                  [math.sin(angle),  math.cos(angle), 0],
                  [0,                0,               1]])
    Q = (P @ R.T)
    assert kabsch_rmsd(P, Q) < 1e-10


def test_rmsd_known_displacement():
    """Diatomic stretched by 1 Å: after Kabsch alignment the RMSD is exactly 0.5 Å.

    P = [(0,0,0), (1,0,0)], Q = [(0,0,0), (2,0,0)].
    After centering: Pc = [(-0.5,0,0),(0.5,0,0)], Qc = [(-1,0,0),(1,0,0)].
    Kabsch gives R=I; diff = [(0.5,0,0),(-0.5,0,0)]; RMSD = sqrt(0.5/2) = 0.5.
    """
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    Q = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert abs(kabsch_rmsd(P, Q) - 0.5) < 1e-8


def test_rmsd_shape_mismatch_raises():
    P = np.zeros((3, 3))
    Q = np.zeros((4, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        kabsch_rmsd(P, Q)


def test_align_returns_rmsd():
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    Q = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    P_aligned, rmsd = align(P, Q)
    assert rmsd >= 0
    assert P_aligned.shape == P.shape


# ─────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────

def test_parse_gaussian_energy():
    lines = [
        " SCF Done:  E(RPBE1PBE) =  -1234.56789012  A.U. after    9 cycles",
        " SCF Done:  E(RPBE1PBE) =  -1234.56790000  A.U. after   10 cycles",
    ]
    e = _parse_gaussian_energy(lines)
    assert e is not None
    assert abs(e - (-1234.56790000)) < 1e-8


def test_parse_orca_energy():
    lines = [
        "FINAL SINGLE POINT ENERGY       -1234.567890123",
        "FINAL SINGLE POINT ENERGY       -1234.567891456",
    ]
    e = _parse_orca_energy(lines)
    assert e is not None
    assert abs(e - (-1234.567891456)) < 1e-9


def test_parse_gaussian_frequencies():
    lines = [
        " Frequencies --   100.23  200.45  350.12",
        " Frequencies --   500.00  600.75",
    ]
    freqs = _parse_gaussian_frequencies(lines)
    assert len(freqs) == 5
    assert abs(freqs[0] - 100.23) < 1e-6


def test_parse_orca_frequencies():
    lines = [
        "VIBRATIONAL FREQUENCIES",
        "----",
        "   0:         0.00 cm**-1",
        "   1:         0.00 cm**-1",
        "   2:         0.01 cm**-1",
        "   3:         0.00 cm**-1",
        "   4:         0.00 cm**-1",
        "   5:         0.00 cm**-1",
        "   6:       145.23 cm**-1",
        "   7:       200.45 cm**-1",
        "",
    ]
    freqs = _parse_orca_frequencies(lines)
    nontrivial = _nontrivial(freqs)
    assert 145.23 in [round(f, 2) for f in nontrivial]
    assert 200.45 in [round(f, 2) for f in nontrivial]
    # Near-zero modes excluded
    assert all(abs(f) > 10 for f in nontrivial)


def test_parse_gaussian_wall_time():
    lines = [
        " Elapsed time:       0 days  1 hours 30 minutes 45.6 seconds.",
    ]
    t = _parse_gaussian_wall_time(lines)
    assert t is not None
    assert abs(t - (3600 + 1800 + 45.6)) < 1e-6


def test_parse_orca_wall_time():
    lines = [
        "TOTAL RUN TIME: 0 days 0 hours 45 minutes 30 seconds 500 msec",
    ]
    t = _parse_orca_wall_time(lines)
    assert t is not None
    assert abs(t - (45 * 60 + 30 + 0.5)) < 1e-6


def test_parse_energy_returns_none_on_empty():
    assert _parse_gaussian_energy([]) is None
    assert _parse_orca_energy([]) is None


# ─────────────────────────────────────────────
# generate()
# ─────────────────────────────────────────────

def test_generate_creates_both_files(tmp_path, simple_xyz, simple_params):
    result = generate(simple_xyz, simple_params, tmp_path / "inputs")
    assert isinstance(result, BenchmarkResult)
    assert result.gaussian_input is not None and result.gaussian_input.exists()
    assert result.orca_input     is not None and result.orca_input.exists()


def test_generate_gjf_has_coordinates(tmp_path, simple_xyz, simple_params):
    result = generate(simple_xyz, simple_params, tmp_path / "inputs")
    content = result.gaussian_input.read_text()
    assert "Ti" in content and "1.95" in content


def test_generate_inp_has_coordinates(tmp_path, simple_xyz, simple_params):
    result = generate(simple_xyz, simple_params, tmp_path / "inputs")
    content = result.orca_input.read_text()
    assert "Ti" in content and "1.95" in content


def test_generate_cluster_label_override(tmp_path, simple_xyz, simple_params):
    result = generate(simple_xyz, simple_params, tmp_path / "inputs", cluster_label="mycluster")
    assert result.cluster == "mycluster"
    assert result.gaussian_input.name.startswith("mycluster")
    assert result.orca_input.name.startswith("mycluster")


def test_generate_to_json(tmp_path, simple_xyz, simple_params):
    import json
    result = generate(simple_xyz, simple_params, tmp_path / "inputs")
    data = json.loads(result.to_json())
    assert data["cluster"] == simple_xyz.stem
    assert data["gaussian_input"] is not None
    assert data["orca_input"] is not None


# ─────────────────────────────────────────────
# compare() — smoke test with real Gaussian files
# ─────────────────────────────────────────────

_GAUSSIAN_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_cdft" / "N.out"

@pytest.mark.skipif(not _GAUSSIAN_OUT.exists(), reason="1tial_cdft data not present")
def test_compare_same_file_gives_zero_delta_e():
    """Comparing a Gaussian file against itself should give ΔE = 0."""
    result = compare(_GAUSSIAN_OUT, _GAUSSIAN_OUT,
                     engine_gaussian="gaussian", engine_orca="gaussian")
    assert result.comparison is not None
    if result.comparison.delta_e_kcal is not None:
        assert result.comparison.delta_e_kcal < 1e-10


@pytest.mark.skipif(not _GAUSSIAN_OUT.exists(), reason="1tial_cdft data not present")
def test_compare_returns_benchmark_result():
    result = compare(_GAUSSIAN_OUT, _GAUSSIAN_OUT,
                     engine_gaussian="gaussian", engine_orca="gaussian")
    assert isinstance(result, BenchmarkResult)
    assert result.gaussian_result is not None
    assert result.orca_result is not None


@pytest.mark.skipif(not _GAUSSIAN_OUT.exists(), reason="1tial_cdft data not present")
def test_compare_to_markdown():
    result = compare(_GAUSSIAN_OUT, _GAUSSIAN_OUT,
                     engine_gaussian="gaussian", engine_orca="gaussian")
    md = result.to_markdown()
    assert "CADMIA Benchmark Report" in md
    assert "Metric" in md


def test_kabsch_max_deviation_is_zero_for_a_rigid_motion():
    """Rotating and translating a structure changes no distance, so nothing deviates."""
    import numpy as np
    rng = np.random.default_rng(1)
    P = rng.normal(size=(12, 3))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta),  np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    assert kabsch_max_deviation(P, P @ R.T + np.array([3.0, -1.0, 2.0])) < 1e-9


def test_kabsch_max_deviation_does_not_dilute_with_system_size():
    """One displaced atom keeps its weight as the rest of the system grows; the RMSD does not.

    This is the whole point of the measure: the dedup tolerance must mean the same thing
    on a 4-atom cluster and on a 50-atom one.
    """
    import numpy as np
    rng = np.random.default_rng(2)
    base = rng.normal(scale=3.0, size=(60, 3))
    shifted = base.copy()
    shifted[0] += np.array([0.8, 0.0, 0.0])
    small = slice(0, 6)
    assert kabsch_rmsd(base[small], shifted[small]) > 2.5 * kabsch_rmsd(base, shifted)
    assert kabsch_max_deviation(base, shifted) > 0.5


def test_kabsch_max_deviation_pairs_atoms_by_element_when_asked():
    """The same structure written in a different atom order is still the same structure.

    Probe modes write their atoms in different orders — o_down gives O,C,O where side_on
    gives C,O,O — so pairing by list position compared a carbon against an oxygen and
    reported 2.2 Å for two geometries that coincide within 0.02 Å.
    """
    import numpy as np
    coords = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.6, 0.0],
                       [3.0, 0.0, 0.0], [4.16, 0.0, 0.0], [1.84, 0.0, 0.0]])
    syms = ["Ti", "O", "O", "C", "O", "O"]
    order = [0, 1, 2, 4, 3, 5]                      # C and one O swap places in the list
    assert kabsch_max_deviation(coords, coords[order]) > 1.0
    assert kabsch_max_deviation(coords, coords[order], syms, [syms[i] for i in order]) < 1e-6


def test_kabsch_max_deviation_refuses_a_different_composition():
    import numpy as np
    P = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.6, 0.0]])
    assert kabsch_max_deviation(P, P, ["Ti", "O", "O"], ["Ti", "O", "N"]) == float("inf")
