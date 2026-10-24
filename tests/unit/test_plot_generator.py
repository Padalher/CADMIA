"""Tests for ECN evolution plot data preparation."""
from pathlib import Path
import math

import pytest

from cadmia.io.xyz import parse_xyz
from cadmia.structure.reports import write_batch_report, write_report
from cadmia.structure.structural.analyzer import analyze_structure
from cadmia.structure.visualization.plot_generator import (
    extract_ecn_table_from_structural_report,
    format_ecn_label,
    generate_ecn_heatmap,
    generate_ecn_trend_plot,
    prepare_ecn_table,
    round_ecn_value,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_prepare_ecn_table_preserves_missing_values_as_nan():
    rows = prepare_ecn_table(
        [
            {"n": 5, "Ti-O_ECN": "2.5"},
            {"n": 3, "Al-O_ECN": "1.5"},
        ]
    )

    assert [row["n"] for row in rows] == [3, 5]
    assert rows[0]["Al-O_ECN"] == pytest.approx(1.5)
    assert math.isnan(rows[0]["Ti-O_ECN"])
    assert rows[1]["Ti-O_ECN"] == pytest.approx(2.5)
    assert math.isnan(rows[1]["Al-O_ECN"])


def test_prepare_ecn_table_supports_arbitrary_pairs():
    """Non-Ti-Al-O systems: pairs are discovered dynamically (bimetallic, mixed, etc.)."""
    rows = prepare_ecn_table(
        [
            {"n": 1, "Au-Au_ECN": 6.0, "Cu-S_ECN": 2.0},
            {"n": 2, "Au-Au_ECN": 8.0},
        ]
    )
    ecn_cols = [k for k in rows[0] if k.endswith("_ECN")]
    assert set(ecn_cols) == {"Au-Au_ECN", "Cu-S_ECN"}
    assert rows[0]["Au-Au_ECN"] == pytest.approx(6.0)
    assert math.isnan(rows[1]["Cu-S_ECN"])


def test_prepare_ecn_table_prefers_tial_order_then_appends_others():
    """Preferred Ti-Al-O pairs come first; any extra pair is appended after them."""
    rows = prepare_ecn_table([{"n": 1, "Fe-O_ECN": 4.0, "Ti-O_ECN": 2.0}])
    ecn_cols = [k for k in rows[0] if k.endswith("_ECN")]
    assert ecn_cols == ["Ti-O_ECN", "Fe-O_ECN"]


def test_prepare_ecn_table_preserves_explicit_n_values():
    rows = prepare_ecn_table(
        [
            {"n": 8, "Ti-O_ECN": 2.5},
            {"n": 32, "Ti-O_ECN": 3.5},
        ]
    )

    assert [row["n"] for row in rows] == [8, 32]
    assert "atom_count" not in rows[0]


def test_prepare_ecn_table_requires_n_column():
    with pytest.raises(ValueError, match="column 'n'"):
        prepare_ecn_table([{"Ti-O_ECN": 2.0}])


def test_round_ecn_value_uses_decimal_half_up():
    assert round_ecn_value(2.985, decimals=1) == pytest.approx(3.0)
    assert round_ecn_value(2.423, decimals=1) == pytest.approx(2.4)
    assert round_ecn_value("2.95", decimals=1) == pytest.approx(3.0)


def test_format_ecn_label_styles():
    assert format_ecn_label(2.985, style="raw") == "2.985"
    assert format_ecn_label(2.985, style="rounded") == "3.0"
    assert format_ecn_label(2.985, style="both") == "2.985 (3.0)"
    assert format_ecn_label(None, style="both") == ""


def test_extract_ecn_table_from_batch_structural_report(config, tmp_path):
    results = [
        analyze_structure(parse_xyz(FIXTURES / "TiO2_monomer.xyz"), config),
        analyze_structure(parse_xyz(FIXTURES / "Al2TiO5_cluster.xyz"), config),
    ]
    report = write_batch_report(results, tmp_path / "structural_report.txt", config=config)

    rows = prepare_ecn_table(extract_ecn_table_from_structural_report(report))

    assert [row["n"] for row in rows] == [1, 2]
    assert [row["atom_count"] for row in rows] == [3, 8]
    assert rows[0]["Ti-O_ECN"] > 0
    assert math.isnan(rows[0]["Al-O_ECN"])
    assert rows[1]["Al-O_ECN"] > 0


def test_extract_ecn_table_from_single_structural_report(config, tmp_path):
    result = analyze_structure(parse_xyz(FIXTURES / "TiO2_monomer.xyz"), config)
    report = write_report(result, tmp_path / "single_report.txt", config=config)

    rows = prepare_ecn_table(extract_ecn_table_from_structural_report(report))

    assert rows[0]["n"] == 1
    assert rows[0]["atom_count"] == 3
    assert rows[0]["Ti-O_ECN"] > 0


def test_generate_ecn_plots_write_html(tmp_path):
    pytest.importorskip("plotly")
    data = [
        {"n": 3, "Ti-O_ECN": 2.0, "Al-O_ECN": None},
        {"n": 8, "Ti-O_ECN": 3.0, "Al-O_ECN": 4.0},
    ]

    trend_path = tmp_path / "ecn_trend.html"
    heatmap_path = tmp_path / "ecn_heatmap.html"

    generate_ecn_trend_plot(data, trend_path, label_style="rounded")
    generate_ecn_heatmap(data, heatmap_path, label_style="rounded")

    assert trend_path.exists()
    assert heatmap_path.exists()
