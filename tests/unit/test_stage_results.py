"""Inside a project every stage saves itself to results/, and `cadmia report` assembles the output."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cadmia.cli import main
from cadmia.io.results import find_project_root, load_stage_results, save_stage_result

D = Path(__file__).resolve().parents[1] / "fixtures" / "1tial_d4"


@pytest.fixture
def project(tmp_path, monkeypatch):
    runner = CliRunner()
    assert runner.invoke(main, ["init", str(tmp_path / "p")]).exit_code == 0
    monkeypatch.chdir(tmp_path / "p")
    return runner, tmp_path / "p"


def _ok(runner, args):
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    return result


def test_outside_a_project_nothing_is_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_project_root() is None
    assert save_stage_result("cdft", {"a": 1}, "text") is None


def test_every_stage_saves_itself_and_report_assembles_the_output(project):
    runner, root = project
    assert (root / "results").is_dir()
    _ok(runner, ["structure", "report", str(D / "1tial_N.out"), "-o", str(root / "s.txt")])
    _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                 "--nm1", str(D / "1tial_cation.out"), "--cluster", "Al2TiO5"])
    _ok(runner, ["analyze", str(D / "1tial_CO2.out"), "--cluster", str(D / "1tial.out"),
                 "--probe-free", str(D / "co2.out"), "--cdft", "results/cdft.json"])
    _ok(runner, ["stability", "--mixed", str(D / "1tial.out"), "--ref", str(D / "TiO2.out"),
                 "--ref", str(D / "Al2O3.out")])

    stages = load_stage_results(root)
    assert list(stages) == ["structure", "cdft", "analysis", "stability"]
    cdft = stages["cdft"][0]
    # one file: the ranking that geometry/analyze read and the full report correlate reads
    assert cdft["sites"] and cdft["atoms"] and cdft["global_fd"]["ip_ev"] == pytest.approx(10.0968, abs=1e-4)
    assert "Global CDFT descriptors" in cdft["report_text"]

    _ok(runner, ["report"])
    out = (root / "reports" / "cadmia.out").read_text(encoding="utf-8")
    order = ["please cite", "MODULE 1  ·  STRUCTURE", "Rings (metal-ligand subgraph)",
             "MODULE 2  ·  CDFT", "Global CDFT descriptors", "MODULE 3  ·  ADSORPTION",
             "Probe activation", "MODULE 4  ·  STABILITY", "Relative stability against the references (",
             "Normal termination"]
    assert [out.index(s) for s in order] == sorted(out.index(s) for s in order)
    assert "Author:" not in out and "```" not in out
    assert "CDFT       1tial_N.out" in out


def test_cdft_report_alone_is_written_from_the_saved_result(project):
    runner, root = project
    _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                 "--nm1", str(D / "1tial_cation.out"), "--no-db"])
    saved = json.loads((root / "results" / "cdft.json").read_text(encoding="utf-8"))
    _ok(runner, ["cdft", "--report", "again.txt"])
    assert (root / "again.txt").read_text(encoding="utf-8").strip() == saved["report_text"].strip()


def test_cdft_report_alone_needs_a_saved_result(project):
    runner, _ = project
    result = runner.invoke(main, ["cdft", "--report", "x.txt"])
    assert result.exit_code != 0 and "results/cdft.json" in result.output


def test_report_in_an_empty_project_says_what_to_run(project):
    runner, _ = project
    result = runner.invoke(main, ["report"])
    assert result.exit_code != 0 and "run a module" in result.output


def test_o2_configs_carry_the_triplet_into_the_inputs(tmp_path):
    from cadmia.agents import calculation_agent, geometry_agent
    from cadmia.calculators.params import CalcParams
    from cadmia.io.xyz import parse_xyz

    runner = CliRunner()
    cdft = tmp_path / "cdft.json"
    _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                 "--nm1", str(D / "1tial_cation.out"), "-o", str(cdft), "--no-db"])
    geometry_agent.run(str(cdft), str(D / "1tial_N.out"), "O2", n_sites=1, n_configs=1,
                       polar_angles=[0], output_dir=tmp_path / "configs")
    xyz = sorted((tmp_path / "configs").glob("*.xyz"))[0]
    assert parse_xyz(xyz).multiplicity == 3

    result = calculation_agent.prepare(str(xyz), engine="orca", output_dir=str(tmp_path / "inp"),
                                       params=CalcParams(auto_multiplicity=True))
    assert "* xyz 0 3" in Path(result.inputs[0]).read_text()


def test_a_spin_read_for_another_charge_is_not_used(tmp_path):
    from cadmia.agents import calculation_agent
    from cadmia.calculators.params import CalcParams

    result = calculation_agent.prepare(str(D / "1tial_cation.out"), engine="orca",
                                       output_dir=str(tmp_path), params=CalcParams(auto_multiplicity=True))
    assert "* xyz 0 1" in Path(result.inputs[0]).read_text()
    assert any("charge 1" in w for w in result.warnings)
    result = calculation_agent.prepare(str(D / "1tial_cation.out"), engine="orca", overwrite=True,
                                       output_dir=str(tmp_path), params=CalcParams(charge=1, auto_multiplicity=True))
    assert "* xyz 1 2" in Path(result.inputs[0]).read_text()


def test_batch_analyze_survives_one_broken_adduct(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    broken = tmp_path / "broken.out"
    broken.write_text("not an engine output\n")
    good = tmp_path / "good.out"
    good.write_bytes((D / "1tial_CO2.out").read_bytes())
    res = CliRunner().invoke(main, ["analyze", str(good), str(broken), "--cluster", str(D / "1tial.out"),
                                    "--probe-free", str(D / "co2.out"), "-o", "ads", "--no-db"])
    assert res.exit_code == 0, res.output
    assert "Analyzed 1 of 2" in res.output and (tmp_path / "ads" / "analysis_good.json").exists()


def test_structure_report_relative_output_lands_in_the_current_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _ok(CliRunner(), ["structure", "report", str(D / "1tial_N.out"), "-o", "rel.txt"])
    assert (tmp_path / "rel.txt").exists() and not (D / "rel.txt").exists()


def test_counterpoise_inputs_and_bsse(tmp_path, monkeypatch):
    from cadmia.agents import analysis_agent
    monkeypatch.chdir(tmp_path)
    _ok(CliRunner(), ["prepare", "bsse", str(D / "1tial_CO2.out"), "--cluster", str(D / "1tial.out"),
                      "--probe-free", str(D / "co2.out"), "-o", "cp"])
    full = (tmp_path / "cp" / "1tial_CO2_cp_cluster_full.inp").read_text()
    own = (tmp_path / "cp" / "1tial_CO2_cp_probe_own.inp").read_text()
    assert "C:" in full and "O:" in full and "* xyz 0 1" in full
    assert "\nTi " not in own and own.count("\nO ") == 2
    # four fake outputs: BSSE = (E_cl_own − E_cl_full) + (E_pr_own − E_pr_full)
    energies = {"cluster_own": -10.0, "cluster_full": -10.002, "probe_own": -5.0, "probe_full": -5.001}
    real = analysis_agent._parse_output
    monkeypatch.setattr(analysis_agent, "_parse_output", lambda f: (
        type("S", (), {"energy_hartree": energies[Path(f).stem.split("_cp_")[1]]})(), "orca")
        if "_cp_" in str(f) else real(f))
    for term in energies:
        (tmp_path / f"1tial_CO2_cp_{term}.out").write_text("x")
    assert analysis_agent._counterpoise_bsse_hartree(D / "1tial_CO2.out") == pytest.approx(0.003)


def test_report_alone_reads_every_stage_from_results(project):
    runner, root = project
    _ok(runner, ["structure", "report", str(D / "1tial_N.out"), "-o", str(root / "s.txt")])
    _ok(runner, ["analyze", str(D / "1tial_CO2.out"), "--cluster", str(D / "1tial.out"),
                 "--probe-free", str(D / "co2.out"), "--no-db"])
    _ok(runner, ["stability", "--mixed", str(D / "1tial.out"), "--ref", str(D / "TiO2.out"),
                 "--ref", str(D / "Al2O3.out"), "--no-db"])
    for args, marker in ((["structure", "-r", "st.txt"], "Rings (metal-ligand"),
                         (["analyze", "-r", "an.txt"], "Probe activation"),
                         (["stability", "--report", "sb.txt"], "Relative stability against"),
                         (["cdft", "-r", "cd.txt"], None)):
        if marker is None:
            _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                         "--nm1", str(D / "1tial_cation.out"), "--no-db"])
            marker = "Global CDFT descriptors"
        _ok(runner, args)
        assert marker in (root / args[-1]).read_text(encoding="utf-8")
    res = runner.invoke(main, ["stability"])
    assert res.exit_code != 0 and "--mixed and --ref are required" in res.output


def test_spin_tokens_do_not_name_the_structure_and_charge_alone_is_no_spin(tmp_path):
    from cadmia.io.xyz import parse_xyz
    f = tmp_path / "c.xyz"
    f.write_text("1\nAl2TiO5_CO2_site2 charge=0 mult=3\nO 0 0 0\n")
    s = parse_xyz(f)
    assert s.structure_id == "Al2TiO5_CO2_site2" and s.metadata["spin_given"]
    f.write_text("1\nsite2 charge=1\nO 0 0 0\n")
    assert "spin_given" not in parse_xyz(f).metadata


def test_same_adduct_name_in_two_folders_is_kept_apart(project, tmp_path):
    runner, root = project
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "site5.out").write_bytes((D / "1tial_CO2.out").read_bytes())
        _ok(runner, ["analyze", str(tmp_path / folder / "site5.out"), "--cluster", str(D / "1tial.out"),
                     "--probe-free", str(D / "co2.out"), "--no-db"])
    saved = sorted(p.name for p in (root / "results" / "analysis").glob("*.json"))
    assert saved == ["site5-2.json", "site5.json"]
    assert json.loads((root / "results" / "analysis" / "site5-2.json").read_text())["config_id"] == "site5-2"


def test_a_cdft_of_other_files_is_saved_as_a_copy_not_over_the_first(project):
    runner, root = project
    _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                 "--nm1", str(D / "1tial_cation.out"), "--no-db"])
    G = D.parent / "1tial_cdft"
    for _ in range(2):          # the second run of the same files updates its copy, no third one
        _ok(runner, ["cdft", "--n", str(G / "N.out"), "--np1", str(G / "N+1.out"),
                     "--nm1", str(G / "N-1.out"), "--no-db"])
    assert sorted(p.name for p in (root / "results").glob("cdft*.json")) == ["cdft-2.json", "cdft.json"]
    first = json.loads((root / "results" / "cdft.json").read_text())
    assert first["sources"]["N"].endswith("1tial_d4/1tial_N.out")
    stages = load_stage_results(root)
    assert [d["_file"] for d in stages["cdft"]] == ["results/cdft.json", "results/cdft-2.json"]


def test_report_stops_when_results_mix_clusters(project):
    runner, root = project
    _ok(runner, ["stability", "--mixed", str(D / "1tial.out"), "--ref", str(D / "TiO2.out"),
                 "--ref", str(D / "Al2O3.out"), "--no-db"])
    data = json.loads((root / "results" / "stability.json").read_text())
    _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                 "--nm1", str(D / "1tial_cation.out"), "--no-db"])
    (root / "results" / "stability.json").write_text(json.dumps({**data, "formula": "Al4O8Ti2"}))
    res = runner.invoke(main, ["report"])
    assert res.exit_code != 0 and "stability.json: Al4O8Ti2" in res.output


def test_a_new_pipeline_refuses_a_workdir_with_calculations(tmp_path):
    from cadmia.agents import orchestrator
    (tmp_path / "adsorption").mkdir()
    (tmp_path / "adsorption" / "cadmia_jobs.json").write_text("{}")
    with pytest.raises(ValueError, match="--resume"):
        orchestrator.run_pipeline(n=D / "1tial_N.out", np1=D / "1tial_anion.out",
                                  nm1=D / "1tial_cation.out", workdir=tmp_path, dry_run=True)


def test_bsse_from_real_orca_single_points_and_a_real_gaussian_counterpoise(tmp_path):
    import shutil
    from cadmia.agents import analysis_agent
    from cadmia.core.units import HARTREE_TO_KCAL
    for f in [D / "1tial_CO2.out", *(D.parent / "1tial_d4_cp").glob("*.out")]:
        shutil.copy(f, tmp_path)
    orca = analysis_agent._counterpoise_bsse_hartree(tmp_path / "1tial_CO2.out")
    assert orca == pytest.approx(0.002370571, abs=1e-8)
    g16 = D.parent / "gaussian_cp" / "1tial_CO2_cp.out"
    assert analysis_agent._parse_bsse_hartree(g16) == pytest.approx(0.002428048502)
    assert analysis_agent._adduct_energy_is_cp_corrected(g16)
    state, engine = analysis_agent._parse_output(g16)
    # Gaussian's corrected energy is the complex plus the BSSE, not minus
    assert engine == "gaussian" and state.energy_hartree == pytest.approx(-1898.67578709 + 0.002428048502, abs=1e-7)
    # the two engines agree on the basis-set error within 0.05 kcal/mol
    assert abs(orca - 0.002428048502) * HARTREE_TO_KCAL < 0.05


def test_report_takes_originals_by_default_and_copies_on_request(project):
    runner, root = project
    G = D.parent / "1tial_cdft"
    _ok(runner, ["cdft", "--n", str(D / "1tial_N.out"), "--np1", str(D / "1tial_anion.out"),
                 "--nm1", str(D / "1tial_cation.out"), "--no-db"])
    _ok(runner, ["cdft", "--n", str(G / "N.out"), "--np1", str(G / "N+1.out"),
                 "--nm1", str(G / "N-1.out"), "--no-db"])
    res = _ok(runner, ["report"])
    out = (root / "reports" / "cadmia.out").read_text(encoding="utf-8")
    assert "cdft-2.json" in res.output and "of 2" not in out
    assert "MODULE 2  ·  CDFT" in out and "MODULE 1" not in out      # numbers are fixed
    for args in (["--cdft", "all"], ["--cdft", "results/cdft.json", "results/cdft-2.json"]):
        _ok(runner, ["report", *args, "-o", "both.out"])
        assert "CDFT 2 of 2: cdft-2" in (root / "both.out").read_text(encoding="utf-8")


def test_text_reports_print_relative_paths_unless_the_user_asks_for_absolute(project, monkeypatch):
    from cadmia.io import results
    runner, root = project
    monkeypatch.setattr(results, "CONFIG_PATH", root / "no_config.yaml")
    inside = root / "outs" / "1tial_N.out"
    inside.parent.mkdir()
    inside.write_bytes((D / "1tial_N.out").read_bytes())
    assert results.shown_path(inside) == "outs/1tial_N.out"
    assert results.shown_path(D / "1tial_N.out") == "1tial_N.out"      # outside: the name only
    from cadmia.agents.text_report import analysis_report_text
    assert f" Adduct  : 1tial_N.out" in analysis_report_text({"adduct_path": str(D / "1tial_N.out")})
    _ok(runner, ["structure", "report", str(inside), "-o", "s.txt"])
    assert "Source file: outs/1tial_N.out" in (root / "s.txt").read_text()
    _ok(runner, ["report"])
    out = (root / "reports" / "cadmia.out").read_text(encoding="utf-8")
    assert f"Project         : {root.name}\n" in out and "relative to the project" in out
    (root / "absolute.yaml").write_text("other_key: 1\n")
    monkeypatch.setattr(results, "CONFIG_PATH", root / "absolute.yaml")
    _ok(runner, ["paths", "absolute"])
    assert "other_key: 1" in (root / "absolute.yaml").read_text()
    assert "report paths: absolute" in _ok(runner, ["paths"]).output
    assert results.shown_path(inside) == str(inside.resolve())
    _ok(runner, ["report"])
    assert "check before sharing" in (root / "reports" / "cadmia.out").read_text(encoding="utf-8")


def test_text_reports_wrap_prose_at_80_columns_and_leave_tables_alone():
    from cadmia.io.results import fit_width
    prose = "word " * 30 + "https://doi.org/10.5281/zenodo.23608477"
    table = "Al-O   " + "  ".join(["1.000"] * 14)
    lines = fit_width(f"  {prose}\n{table}").split("\n")
    assert all(len(l) <= 80 and l.startswith("  ") for l in lines[:-1])
    assert lines[-2].endswith("https://doi.org/10.5281/zenodo.23608477")
    assert lines[-1] == table


def test_structure_report_in_a_project_writes_nothing_beside_the_input(project, tmp_path):
    runner, root = project
    src = tmp_path / "src"
    src.mkdir()
    (src / "1tial_N.out").write_bytes((D / "1tial_N.out").read_bytes())
    _ok(runner, ["structure", "report", str(src / "1tial_N.out")])
    assert sorted(p.name for p in src.iterdir()) == ["1tial_N.out"]
    assert (root / "results" / "structure").is_dir()
