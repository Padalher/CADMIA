"""Tests for cadmia.structure.visualization."""
from pathlib import Path

import pytest

from cadmia.io.xyz import parse_xyz

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_plot_molecule_3d_returns_figure():
    pytest.importorskip("plotly.graph_objects")
    from cadmia.structure.visualization.molecular import plot_molecule_3d

    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    fig = plot_molecule_3d(s, color_by="element")
    assert fig is not None
    assert hasattr(fig, "data")


def test_plot_molecule_3d_saves_html(tmp_path):
    pytest.importorskip("plotly")
    from cadmia.structure.visualization.molecular import plot_molecule_3d

    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    out = str(tmp_path / "mol.html")
    plot_molecule_3d(s, output_html=out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


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
