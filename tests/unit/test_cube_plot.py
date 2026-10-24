"""Tests for cadmia.electronic.cube_plot (self-contained HTML cube viewer)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from cadmia.core import Structure, VolumetricGrid
from cadmia.electronic.cube_plot import cube_html, write_cube_html

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _grid(data: np.ndarray) -> VolumetricGrid:
    s = Structure(
        atoms=["O"], positions=np.zeros((1, 3)), labels=["O1"],
        structure_id="t", source_file="t.cube",
    )
    return VolumetricGrid(
        structure=s, origin=np.zeros(3),
        vx=np.array([0.2, 0.0, 0.0]), vy=np.array([0.0, 0.2, 0.0]),
        vz=np.array([0.0, 0.0, 0.2]), data=data, field_type="orbital",
    )


def _traces_from_html(html: str) -> list:
    m = re.search(r"var data = (\[.*?\]);\nvar layout", html, re.S)
    assert m, "embedded data JSON not found"
    return json.loads(m.group(1))


def test_html_is_self_contained():
    html, warnings = cube_html(_grid(np.linspace(-1, 1, 27).reshape(3, 3, 3)), title="orb")
    assert "cdn.plot.ly" in html
    assert "<!DOCTYPE html>" in html
    assert "isosurface" in html


def test_two_surfaces_for_signed_field():
    # Field spans negative and positive → two isosurface traces (+ atoms).
    data = np.linspace(-1, 1, 27).reshape(3, 3, 3)
    html, _ = cube_html(_grid(data))
    traces = _traces_from_html(html)
    iso = [t for t in traces if t["type"] == "isosurface"]
    assert len(iso) == 2
    assert any(t["type"] == "scatter3d" for t in traces)  # atom overlay


def test_single_surface_for_positive_field():
    data = np.linspace(0.01, 1, 27).reshape(3, 3, 3)
    html, _ = cube_html(_grid(data))
    iso = [t for t in _traces_from_html(html) if t["type"] == "isosurface"]
    assert len(iso) == 1


def test_default_isovalue_is_fraction_of_maxabs():
    data = np.zeros((3, 3, 3))
    data[1, 1, 1] = 2.0          # max abs = 2.0 → default iso = 0.15*2 = 0.30
    html, _ = cube_html(_grid(data))
    iso = [t for t in _traces_from_html(html) if t["type"] == "isosurface"][0]
    assert iso["isomin"] == pytest.approx(0.30, abs=1e-6)


def test_explicit_isovalue_overrides():
    html, _ = cube_html(_grid(np.linspace(-1, 1, 27).reshape(3, 3, 3)), isovalue=0.5)
    iso = [t for t in _traces_from_html(html) if t["type"] == "isosurface" and t["isomin"] > 0][0]
    assert iso["isomin"] == pytest.approx(0.5, abs=1e-9)


def test_downsampling_warns_and_shrinks():
    data = np.random.default_rng(0).normal(size=(20, 20, 20))  # 8000 pts
    html, warnings = cube_html(_grid(data), max_points=500)
    assert any("downsampled" in w for w in warnings)
    iso = [t for t in _traces_from_html(html) if t["type"] == "isosurface"][0]
    assert len(iso["value"]) <= 500


def test_uniform_zero_field_warns():
    html, warnings = cube_html(_grid(np.zeros((3, 3, 3))))
    assert any("uniformly zero" in w for w in warnings)


def test_interactive_slider_present_and_wired():
    html, _ = cube_html(_grid(np.linspace(-1, 1, 27).reshape(3, 3, 3)))
    assert 'id="isoSlider"' in html
    assert "Plotly.restyle" in html
    assert "isomin" in html and "isomax" in html   # both signs wired


def test_interactive_can_be_disabled():
    html, _ = cube_html(_grid(np.linspace(0.01, 1, 27).reshape(3, 3, 3)), interactive=False)
    assert "isoSlider" not in html


def test_write_from_cube_file(tmp_path):
    out = tmp_path / "view.html"
    warnings = write_cube_html(FIXTURES / "test_small.cube", out)
    assert out.exists()
    assert "isosurface" in out.read_text(encoding="utf-8")
    assert isinstance(warnings, list)
