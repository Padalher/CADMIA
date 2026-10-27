"""Tests for cadmia.structure.cli."""
import json
from pathlib import Path

import pytest

from cadmia.structure.cli import _ecn_table_plot_output_path, _plot_output_path, main

FIXTURES = Path(__file__).parent.parent / "fixtures"
ROOT = Path(__file__).resolve().parents[2]

def test_cli_structure_writes_report(tmp_path):
    out = tmp_path / "structure.json"
    code = main(
        [
            "structure",
            str(FIXTURES / "TiO2_monomer.xyz"),
            "-o",
            str(out),
            "--format",
            "json",
        ]
    )
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["bonds"]) == 2
    assert payload["directed_bond_statistics"]["Ti-O"]["bond_count"] == 2


def test_cli_structure_writes_batch_report(tmp_path):
    out = tmp_path / "structures.json"
    code = main(
        [
            "structure",
            str(FIXTURES / "TiO2_monomer.xyz"),
            str(FIXTURES / "Al2TiO5_cluster.xyz"),
            "-o",
            str(out),
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["structure_count"] == 2
    assert payload["reports"][1]["directed_bond_statistics"]["Al-O"]["bond_count"] == 4


def test_cli_structure_directory_rerun_skips_its_own_report(tmp_path):
    import shutil
    shutil.copy(FIXTURES / "TiO2_monomer.xyz", tmp_path)
    for _ in range(2):
        assert main(["structure", str(tmp_path), "-o", str(tmp_path / "structural_report.txt")]) == 0


def test_cli_plot_writes_html(tmp_path):
    pytest.importorskip("plotly")
    out = tmp_path / "molecule.html"
    code = main(["plot", str(FIXTURES / "TiO2_monomer.xyz"), "-o", str(out)])
    assert code == 0
    assert out.exists()
    assert "plotly" in out.read_text(encoding="utf-8").lower()


def test_cli_plot_default_output_uses_current_directory(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    monkeypatch.chdir(tmp_path)
    code = main(["plot", str(FIXTURES / "TiO2_monomer.xyz")])
    assert code == 0
    assert (tmp_path / "TiO2_monomer_bonds.html").exists()


def test_cli_plot_output_paths_default_to_cwd():
    class StructureStub:
        structure_id = "structure"

    path = _plot_output_path(
        "data/example_structure.xyz",
        StructureStub(),
        0,
        1,
        "ecn",
        None,
        False,
    )
    assert path == Path("example_structure_ecn.html")


def test_cli_plot_batch_output_uses_directory():
    class StructureStub:
        structure_id = "sample"

    path = _plot_output_path(
        "data/example_structure.xyz",
        StructureStub(),
        0,
        1,
        "distances",
        "plots",
        True,
    )
    assert path == Path("plots") / "example_structure_distances.html"


def test_cli_ecn_table_plot_output_defaults_to_requested_names():
    assert _ecn_table_plot_output_path("ecn-trend", None) == Path("ecn_trend.html")
    assert _ecn_table_plot_output_path("ecn-heatmap", None) == Path("ecn_heatmap.html")
    assert _ecn_table_plot_output_path("ecn-trend", "plots") == Path("plots") / "ecn_trend.html"


def test_cli_root_help_lists_workflows(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "Main workflows:" in text
    assert "plot" in text
    assert "Supported structure inputs:" in text


def test_cli_structure_help_describes_analysis(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["structure", "--help"])

    assert exc.value.code == 0
    text = capsys.readouterr().out
    normalized = " ".join(text.split())
    assert "Hoppe effective coordination numbers" in text
    assert "d_av" in text
    assert "ECN_act" in text
    assert "ECN_1d" in text
    assert "DirB" in text
    assert "MeanAux" in text
    assert "--details" in text
    assert "coordination polyhedra" in normalized
    assert "Examples:" in text


def test_cli_plot_help_lists_kinds(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["plot", "--help"])

    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "Plot kinds:" in text
    assert "bonds" in text and "molecule" not in text
    assert "distances" in text
    assert "ecn-trend" in text
    assert "ecn-heatmap" in text
    assert "--ecn-label" in text
    assert "--ecn-decimals" in text


def test_cadmia_doctor_passes_in_dev_environment():
    from click.testing import CliRunner
    from cadmia.cli import main as cadmia_main

    result = CliRunner().invoke(cadmia_main, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "All required checks passed." in result.output
    for pkg in ("cadmia.core", "cadmia.io", "cadmia.cdft",
                "cadmia.electronic", "cadmia.structure"):
        assert f"[✓] {pkg}" in result.output
    assert "atomic_data.yaml" in result.output
