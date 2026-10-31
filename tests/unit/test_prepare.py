"""Tests for calculation_agent.prepare — batch input generation without running."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cadmia.calculators.params import CalcParams
from cadmia.agents.calculation_agent import prepare, prepare_cdft, _resolve_job_files


def _write_xyz(path: Path) -> Path:
    path.write_text(
        "3\nwater\n"
        "O    0.00000000    0.00000000    0.00000000\n"
        "H    0.75800000    0.58700000    0.00000000\n"
        "H   -0.75800000    0.58700000    0.00000000\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def params() -> CalcParams:
    return CalcParams(functional="PBE0", basis="def2-SVP", charge=0,
                      multiplicity=1, nprocs=4, mem_orca_mb_core=1500)


def test_prepare_xyz_to_orca_inp(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    out = tmp_path / "inputs"
    inputs, warnings = prepare(xyz, params, engine="orca", output_dir=out)
    assert warnings == []
    assert len(inputs) == 1
    inp = inputs[0]
    assert inp.name == "mol.inp"
    text = inp.read_text(encoding="utf-8")
    assert "! PBE0 def2-SVP" in text
    assert "%pal nprocs 4 end" in text
    assert "* xyz 0 1" in text
    assert "O    " in text


def test_prepare_xyz_to_gaussian_input(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    out = tmp_path / "inputs"
    # d3bj, not the d4 default: Gaussian 16 has no native D4 and the writer refuses it.
    inputs, _ = prepare(xyz, replace(params, dispersion="d3bj"), engine="gaussian",
                        output_dir=out)
    # `.inp` for both engines: Gaussian reads stdin and ignores the extension, so `.gjf`
    # bought nothing. The engine is recovered from the content, not from the name.
    assert inputs[0].name == "mol.inp"
    assert inputs[0].exists()
    assert "%chk" in inputs[0].read_text(encoding="utf-8")


def test_prepare_default_engine_is_orca(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    inputs, _ = prepare(xyz, params, output_dir=tmp_path / "o")
    assert inputs[0].suffix == ".inp"


def test_prepare_charge_mult_applied(tmp_path):
    xyz = _write_xyz(tmp_path / "ion.xyz")
    params = CalcParams(charge=-1, multiplicity=2)
    inputs, _ = prepare(xyz, params, engine="orca", output_dir=tmp_path / "o")
    assert "* xyz -1 2" in inputs[0].read_text(encoding="utf-8")


def test_prepare_passes_through_existing_input(tmp_path, params):
    inp = tmp_path / "ready.inp"
    inp.write_text("! PBE0 def2-TZVP Opt\n")
    inputs, warnings = prepare(inp, params, output_dir=tmp_path / "o")
    assert inputs == [inp]
    assert warnings == []


def test_prepare_skips_existing_target(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    out = tmp_path / "o"
    prepare(xyz, params, engine="orca", output_dir=out)
    target = out / "mol.inp"
    target.write_text("HAND-EDITED")
    inputs, warnings = prepare(xyz, params, engine="orca", output_dir=out)
    assert target.read_text(encoding="utf-8") == "HAND-EDITED"
    assert any("already exists" in w for w in warnings)
    assert inputs == [target]


def test_prepare_overwrite_regenerates(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    out = tmp_path / "o"
    target = out / "mol.inp"
    out.mkdir()
    target.write_text("STALE")
    inputs, warnings = prepare(xyz, params, engine="orca",
                               output_dir=out, overwrite=True)
    assert "! PBE0" in target.read_text(encoding="utf-8")
    assert warnings == []


def test_prepare_directory_of_xyz(tmp_path, params):
    d = tmp_path / "configs"
    d.mkdir()
    _write_xyz(d / "a.xyz")
    _write_xyz(d / "b.xyz")
    inputs, _ = prepare(d, params, engine="orca", output_dir=tmp_path / "o")
    names = sorted(p.name for p in inputs)
    assert names == ["a.inp", "b.inp"]


def test_prepare_directory_prefers_inputs_warns_on_xyz(tmp_path, params):
    d = tmp_path / "mix"
    d.mkdir()
    (d / "ready.inp").write_text("! PBE0 def2-TZVP Opt\n")
    _write_xyz(d / "geom.xyz")
    inputs, warnings = prepare(d, params, output_dir=tmp_path / "o")
    assert [p.name for p in inputs] == ["ready.inp"]
    assert any("ignoring" in w and ".xyz" in w for w in warnings)


# --- sources that carry a geometry: outputs and concatenated .xyz ------------

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ORCA_OUT     = FIXTURES / "1tial_orca" / "1tial_N_orca.out"
GAUSSIAN_OUT = FIXTURES / "1tial_cdft" / "N.out"


def _write_concatenated_xyz(path: Path, stems: list[str]) -> Path:
    """A ranked .xyz the way `cadmia rank` writes it: one block per candidate."""
    blocks = []
    for rk, stem in enumerate(stems, start=1):
        blocks.append(
            f"2\nrank {rk}  +{rk - 1:.3f} kcal/mol  E=-100.{rk:06d} Eh  orca  {stem}\n"
            "O    0.00000000    0.00000000    0.00000000\n"
            "H    0.75800000    0.58700000    0.00000000\n"
        )
    path.write_text("".join(blocks), encoding="utf-8")
    return path


@pytest.mark.parametrize("source", [ORCA_OUT, GAUSSIAN_OUT])
def test_prepare_generates_an_input_from_an_engine_output(tmp_path, params, source):
    # The engine of the source is auto-detected; --engine only picks what we write.
    result = prepare(source, params, engine="orca", output_dir=tmp_path / "o")
    assert [p.name for p in result.generated] == [f"{source.stem}.inp"]
    body = result.generated[0].read_text(encoding="utf-8")
    assert "! PBE0" in body
    assert body.count("Ti") == 1 and body.count("\nO ") + body.count("\nAl ") > 0


def test_prepare_from_output_keeps_the_optimized_geometry_and_atom_order(tmp_path, params):
    from cadmia.io import structure_from_file

    result = prepare(ORCA_OUT, params, engine="orca", output_dir=tmp_path / "o")
    body = result.generated[0].read_text(encoding="utf-8")
    block = body.split("* xyz")[1].split("\n*")[0].strip().splitlines()[1:]

    expected = structure_from_file(ORCA_OUT)
    assert [line.split()[0] for line in block] == expected.atoms
    for line, position in zip(block, expected.positions):
        assert [float(v) for v in line.split()[1:4]] == pytest.approx(list(position))


def test_prepare_concatenated_xyz_yields_one_input_per_block(tmp_path, params):
    stems = ["cluster_CO2_site14_bidentate_p60_a0_xtbopt",
             "cluster_CO2_site09_eta1_c_p30_a0_xtbopt"]
    ranked = _write_concatenated_xyz(tmp_path / "ranked.xyz", stems)
    result = prepare(ranked, params, engine="orca", output_dir=tmp_path / "o")
    # The origin stem is kept: it carries site/mode/angles, which is what identifies a candidate.
    assert [p.name for p in result.generated] == [f"{s}.inp" for s in stems]


def test_prepare_concatenated_xyz_without_rank_comments_numbers_the_blocks(tmp_path, params):
    path = tmp_path / "traj.xyz"
    path.write_text(
        "2\nframe one\nO 0.0 0.0 0.0\nH 0.7 0.6 0.0\n"
        "2\nframe two\nO 0.0 0.0 0.0\nH 0.8 0.6 0.0\n",
        encoding="utf-8",
    )
    result = prepare(path, params, engine="orca", output_dir=tmp_path / "o")
    assert [p.name for p in result.generated] == ["traj_001.inp", "traj_002.inp"]


def test_prepare_concatenated_xyz_disambiguates_repeated_stems(tmp_path, params):
    ranked = _write_concatenated_xyz(tmp_path / "ranked.xyz", ["same", "same"])
    result = prepare(ranked, params, engine="orca", output_dir=tmp_path / "o")
    assert [p.name for p in result.generated] == ["same.inp", "same_002.inp"]


def test_prepare_single_block_xyz_keeps_the_file_stem(tmp_path, params):
    # A plain .xyz must not gain a _001 suffix just because the reader is multi-block now.
    xyz = _write_xyz(tmp_path / "mol.xyz")
    result = prepare(xyz, params, engine="orca", output_dir=tmp_path / "o")
    assert [p.name for p in result.generated] == ["mol.inp"]


def test_prepare_rejects_an_unsupported_extension(tmp_path, params):
    stray = tmp_path / "notes.txt"
    stray.write_text("not a geometry")
    with pytest.raises(ValueError, match=r"cannot prepare an input from a '\.txt' file"):
        prepare(stray, params, output_dir=tmp_path / "o")


def test_prepare_result_separates_generated_from_passed_through(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    ready = tmp_path / "ready.inp"
    ready.write_text("! PBE0 def2-TZVP Opt\n")
    result = prepare([xyz, ready], params, engine="orca", output_dir=tmp_path / "o")
    assert [p.name for p in result.generated] == ["mol.inp"]
    assert result.passed_through == [ready]
    inputs, warnings = result           # still unpacks for run_local
    assert len(inputs) == 2 and warnings == []


def test_prepare_cdft_accepts_an_engine_output(tmp_path, params):
    inputs, _ = prepare_cdft(ORCA_OUT, params, output_dir=tmp_path / "o")
    assert [p.name for p in inputs] == [f"{ORCA_OUT.stem}_{s}.inp"
                                        for s in ("N", "anion", "cation")]


def test_resolve_job_files_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_job_files(tmp_path / "nope.xyz")


def test_resolve_job_files_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No .gjf/.inp/.xyz"):
        _resolve_job_files(tmp_path)


def test_prepare_list_of_paths(tmp_path, params):
    a = _write_xyz(tmp_path / "a.xyz")
    b = _write_xyz(tmp_path / "b.xyz")
    inputs, _ = prepare([a, b], params, engine="orca", output_dir=tmp_path / "o")
    assert len(inputs) == 2


# ─────────────────────────────────────────────
# prepare_cdft — three single points from one geometry
# ─────────────────────────────────────────────

def test_prepare_cdft_writes_three_inputs(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    out = tmp_path / "cdft"
    inputs, warnings = prepare_cdft(xyz, params, engine="orca", output_dir=out)
    assert warnings == []
    names = [p.name for p in inputs]
    assert names == ["mol_N.inp", "mol_anion.inp", "mol_cation.inp"]


def test_prepare_cdft_charges_and_mults(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    inputs, _ = prepare_cdft(xyz, params, engine="orca", output_dir=tmp_path / "o")
    n, anion, cation = (p.read_text(encoding="utf-8") for p in inputs)
    assert "* xyz 0 1" in n          # neutral: charge 0, singlet
    assert "* xyz -1 2" in anion     # anion N+1: charge -1, doublet
    assert "* xyz 1 2" in cation     # cation N-1: charge +1, doublet


def test_prepare_cdft_custom_charge_and_open_shell(tmp_path):
    xyz = _write_xyz(tmp_path / "rad.xyz")
    params = CalcParams()
    inputs, _ = prepare_cdft(xyz, params, engine="orca", output_dir=tmp_path / "o",
                             charge=1, mult=2, mult_anion=1, mult_cation=1)
    n, anion, cation = (p.read_text(encoding="utf-8") for p in inputs)
    assert "* xyz 1 2" in n          # neutral here is the +1 cation, doublet
    assert "* xyz 0 1" in anion      # +1 electron -> charge 0, singlet
    assert "* xyz 2 1" in cation     # -1 electron -> charge +2, singlet


def test_prepare_cdft_auto_multiplicity(tmp_path, params):
    # water: 10 e- neutral (singlet), 11 anion / 9 cation (both doublet) — by parity
    xyz = _write_xyz(tmp_path / "mol.xyz")
    inputs, warnings = prepare_cdft(xyz, params, engine="orca", output_dir=tmp_path / "o",
                                    mult=None, mult_anion=None, mult_cation=None)
    n, anion, cation = (p.read_text(encoding="utf-8") for p in inputs)
    assert "* xyz 0 1" in n
    assert "* xyz -1 2" in anion
    assert "* xyz 1 2" in cation
    # water has no transition metals → no high-spin warning
    assert not any("high-spin" in w for w in warnings)


def test_prepare_cdft_auto_multiplicity_warns_on_transition_metal(tmp_path, params):
    xyz = tmp_path / "ti.xyz"
    xyz.write_text("2\nTiO\nTi 0 0 0\nO 1.6 0 0\n", encoding="utf-8")
    _, warnings = prepare_cdft(xyz, params, engine="orca", output_dir=tmp_path / "o",
                               mult=None, mult_anion=None, mult_cation=None)
    assert any("high-spin" in w and "Ti" in w for w in warnings)


def test_prepare_cdft_forces_single_point(tmp_path):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    params = CalcParams(task="opt")   # should be overridden to sp
    inputs, _ = prepare_cdft(xyz, params, engine="orca", output_dir=tmp_path / "o")
    for p in inputs:
        text = p.read_text(encoding="utf-8")
        assert " Opt" not in text
        assert "SP" in text


def test_prepare_cdft_gaussian(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    inputs, _ = prepare_cdft(xyz, replace(params, dispersion="d3bj"),
                             engine="gaussian", output_dir=tmp_path / "o")
    assert [p.name for p in inputs] == ["mol_N.inp", "mol_anion.inp", "mol_cation.inp"]
    assert "0 1" in inputs[0].read_text(encoding="utf-8")


def test_prepare_cdft_rejects_a_ready_input(tmp_path, params):
    inp = tmp_path / "mol.inp"
    inp.write_text("! PBE0\n")
    with pytest.raises(ValueError, match=r"single geometry"):
        prepare_cdft(inp, params, output_dir=tmp_path / "o")


def test_prepare_cdft_missing_file(tmp_path, params):
    with pytest.raises(FileNotFoundError):
        prepare_cdft(tmp_path / "ghost.xyz", params, output_dir=tmp_path / "o")


def test_prepare_cdft_skips_existing(tmp_path, params):
    xyz = _write_xyz(tmp_path / "mol.xyz")
    out = tmp_path / "o"
    prepare_cdft(xyz, params, engine="orca", output_dir=out)
    (out / "mol_N.inp").write_text("HAND")
    inputs, warnings = prepare_cdft(xyz, params, engine="orca", output_dir=out)
    assert (out / "mol_N.inp").read_text(encoding="utf-8") == "HAND"
    assert any("already exists" in w for w in warnings)
    inputs2, warnings2 = prepare_cdft(xyz, params, engine="orca",
                                      output_dir=out, overwrite=True)
    assert "! PBE0" in (out / "mol_N.inp").read_text(encoding="utf-8")
