"""Tests for native geometry extraction: io.structure_from_file, `cadmia xyz`, and
geometry accepting an output file for --xyz (no more python one-liner)."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from cadmia.cli import main
from cadmia.io import final_energy_from_file, structure_from_file

_ORCA_OUT = Path(__file__).resolve().parents[1] / "fixtures" / "1tial_orca" / "1tial_N_orca.out"


def test_structure_from_file_orca_output():
    s = structure_from_file(_ORCA_OUT)
    assert s.n_atoms == 8
    assert s.formula == "Al2O5Ti"


def test_structure_from_file_xyz(tmp_path):
    xyz = tmp_path / "m.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    s = structure_from_file(xyz)
    assert s.n_atoms == 2


def test_cli_xyz_extracts_geometry(tmp_path):
    out = tmp_path / "cluster.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_ORCA_OUT), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0].strip()
    assert header == "8"
    assert "Al2O5Ti" in res.output


def test_final_energy_from_file_orca():
    e = final_energy_from_file(_ORCA_OUT)
    assert e == -1710.170304663949  # last FINAL SINGLE POINT ENERGY (converged)


def test_final_energy_from_file_xyz_is_none(tmp_path):
    xyz = tmp_path / "m.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    assert final_energy_from_file(xyz) is None


def test_cli_xyz_writes_energy_in_comment(tmp_path):
    out = tmp_path / "cluster.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_ORCA_OUT), "-o", str(out)])
    assert res.exit_code == 0, res.output
    comment = out.read_text(encoding="utf-8").splitlines()[1]
    assert "-1710.17030466" in comment
    assert "-1710.17030466" in res.output  # shown at extraction, too


def test_cli_xyz_explicit_comment_wins(tmp_path):
    out = tmp_path / "cluster.xyz"
    res = CliRunner().invoke(main, ["xyz", str(_ORCA_OUT), "-o", str(out), "--comment", "mine"])
    assert res.exit_code == 0, res.output
    assert out.read_text(encoding="utf-8").splitlines()[1] == "mine"


def test_geometry_accepts_output_file(tmp_path):
    # a minimal CDFT ranking pointing at atom 1 of the 8-atom cluster
    cdft = tmp_path / "cdft.json"
    cdft.write_text(json.dumps({
        "cluster": "t", "sites": [
            # atom 1 of this cluster is Ti — the symbol must match or the site is skipped
            {"atom_index": 1, "atom_symbol": "Ti", "delta_f": 0.2, "rank": 1, "character": "electrophilic"},
        ],
    }), encoding="utf-8")
    out = tmp_path / "configs"
    res = CliRunner().invoke(main, [
        "geometry", "--cdft", str(cdft), "--xyz", str(_ORCA_OUT),
        "--probe", "H2", "--n-sites", "1", "--n-configs", "1", "-o", str(out), "--no-table",
    ])
    assert res.exit_code == 0, res.output
    assert list(out.glob("*.xyz"))  # at least one adsorption config written
