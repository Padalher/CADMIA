"""Tests for cadmia.structure.visualization."""
from pathlib import Path

import pytest

from cadmia.io.xyz import parse_xyz

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_plot_bond_histogram_runs(config):
    pytest.importorskip("plotly")
    from cadmia.io.xyz import parse_xyz
    from cadmia.structure.structural.analyzer import analyze_structure
    from cadmia.structure.visualization.plots import plot_bond_histogram

    s = parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz")
    result = analyze_structure(s, config)
    fig = plot_bond_histogram(result)
    assert fig is not None


def test_plot_distance_heatmap_runs():
    pytest.importorskip("plotly")
    from cadmia.structure.visualization.plots import plot_distance_heatmap

    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    fig = plot_distance_heatmap(s)
    assert fig is not None


def _png_size(path):
    import struct
    return struct.unpack(">II", path.read_bytes()[16:24])


def _ecn_report(tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main
    d = FIXTURES / "1tial_d4"
    report = tmp_path / "serie.txt"
    res = CliRunner().invoke(main, ["structure", "report", str(d / "1tial_N.out"),
                                    str(d / "TiO2.out"), "-o", str(report)])
    assert res.exit_code == 0
    return report


@pytest.mark.parametrize("kind", ["ecn-trend", "ecn-heatmap"])
@pytest.mark.parametrize("width,inches", [("single", 3.4), ("double", 6.5)])
def test_static_ecn_series_figures_have_the_column_width(tmp_path, kind, width, inches):
    pytest.importorskip("matplotlib")
    from cadmia.structure.cli import main
    out = tmp_path / f"{kind}.png"
    assert main(["plot", str(_ecn_report(tmp_path)), "--kind", kind, "-o", str(out),
                 "--width", width]) == 0
    assert _png_size(out)[0] == round(inches * 300)


def test_static_ecn_trend_keeps_its_legend_inside_the_figure(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from cadmia.structure.visualization import figures
    built = []
    monkeypatch.setattr(figures, "save_figure", lambda fig, path: built.append(fig))
    figures.ecn_trend_figure(_ecn_report(tmp_path), tmp_path / "t.png")
    fig = built[0]
    fig.draw_without_rendering()
    legend_box = fig.legends[0].get_window_extent().transformed(fig.transFigure.inverted())
    assert 0 <= legend_box.x0 and legend_box.x1 <= 1 and legend_box.y1 <= 1
    assert [t.get_text() for t in fig.legends[0].get_texts()][0] == "Ti-O"


def test_static_figures_are_vector_in_svg_and_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    from cadmia.structure.cli import main
    report = _ecn_report(tmp_path)
    assert main(["plot", str(report), "--kind", "ecn-trend", "-o", str(tmp_path / "t.svg")]) == 0
    assert "Average ECN" in (tmp_path / "t.svg").read_text()
    assert main(["plot", str(report), "--kind", "ecn-heatmap", "-o", str(tmp_path / "h.pdf")]) == 0
    assert (tmp_path / "h.pdf").read_bytes().startswith(b"%PDF")


def test_static_bonds_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from cadmia.structure.cli import main
    n = str(FIXTURES / "1tial_d4" / "1tial_N.out")
    assert main(["plot", n, "--kind", "bonds", "-o", str(tmp_path / "b.pdf")]) == 0
    assert (tmp_path / "b.pdf").read_bytes().startswith(b"%PDF")


def test_plot_format_flag_beats_suffix_and_suffix_beats_default(tmp_path):
    pytest.importorskip("matplotlib")
    from cadmia.structure.cli import main
    xyz = str(FIXTURES / "TiO2_monomer.xyz")
    assert main(["plot", xyz, "--kind", "bonds", "--format", "svg", "-o", str(tmp_path / "a")]) == 0
    assert (tmp_path / "a.svg").exists()
    assert main(["plot", xyz, "--kind", "bonds", "-o", str(tmp_path / "b.png")]) == 0
    assert _png_size(tmp_path / "b.png")[0] == 1020


def test_plot_batch_writes_static_figures_with_format_and_rejects_a_file_name(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from cadmia.structure.cli import main
    pair = [str(FIXTURES / "TiO2_monomer.xyz"), str(FIXTURES / "Al2TiO5_cluster.xyz")]
    assert main(["plot", *pair, "--kind", "bonds", "--format", "png", "-o", str(tmp_path / "figs")]) == 0
    assert sorted(f.name for f in (tmp_path / "figs").iterdir()) == [
        "Al2TiO5_cluster_bonds.png", "TiO2_monomer_bonds.png"]
    assert main(["plot", *pair, "--kind", "bonds", "-o", str(tmp_path / "figs2.png")]) == 1
    assert "must be a directory" in capsys.readouterr().out
    assert not (tmp_path / "figs2.png").exists()
