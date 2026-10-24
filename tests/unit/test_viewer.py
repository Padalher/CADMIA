"""Tests for the interactive 3Dmol.js structure viewer and `cadmia view`."""
from __future__ import annotations

import numpy as np

from cadmia.core import Structure
from cadmia.structure.visualization.viewer import structure_viewer_html


def _structure():
    return Structure(
        atoms=["Ti", "O", "O", "O", "Al", "O", "Al", "O"],
        positions=np.array([
            [1.19, 0.0, -0.12], [2.74, 0.0, 0.31], [0.27, 1.64, -0.63], [0.27, -1.64, -0.63],
            [-1.16, -1.17, 0.13], [-0.43, 0.0, 1.26], [-1.16, 1.17, 0.13], [-2.34, 0.0, -0.39],
        ]),
        labels=[f"a{i}" for i in range(8)], structure_id="N", source_file="N",
    )


def test_viewer_html_is_self_contained_and_has_export():
    html = structure_viewer_html(_structure(), title="N")
    assert "3Dmol-min.js" in html            # loads the viewer
    assert "addModel" in html and "xyz" in html
    assert "getView()" in html and "exportView" in html   # orientation export
    assert "Ti" in html                       # geometry embedded


def test_viewer_highlights_sites():
    html = structure_viewer_html(_structure(), highlight=[5], highlight_labels=["O6 (#1)"])
    assert "index: 5" in html                 # highlighted atom (0-based)
    assert "O6 (#1)" in html                  # label drawn


def test_viewer_overlays_cube():
    html = structure_viewer_html(_structure(), cube_text="FAKE CUBE\n...\n", isovalue=0.01)
    assert "addVolumetricData" in html
    assert "0.01" in html and "-0.01" in html  # ±isosurface


def test_cadmia_view_cli(tmp_path):
    import json
    from click.testing import CliRunner
    from cadmia.cli import main
    xyz = tmp_path / "n.xyz"
    xyz.write_text("2\nN\nTi 0 0 0\nO 1.6 0 0\n")
    cdft = tmp_path / "cdft.json"
    cdft.write_text(json.dumps({"sites": [
        {"atom_index": 1, "atom_symbol": "Ti", "rank": 1},
        {"atom_index": 2, "atom_symbol": "O", "rank": 2},
    ]}))
    out = tmp_path / "v.html"
    res = CliRunner().invoke(main, ["view", str(xyz), "--cdft", str(cdft),
                                    "--n-sites", "2", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()
    html = out.read_text()
    assert "index: 0" in html and "index: 1" in html   # both sites (1-based → 0-based)
