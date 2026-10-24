"""Tests for the orientation bridge (viewer → POV-Ray/VMD)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from cadmia.structure.rendering.orientation import (
    Orientation,
    apply_orientation,
    load_orientation,
    orientation_from_3dmol_view,
    quat_to_rotation_matrix,
    vmd_rotate_matrix,
)


def test_identity_quaternion_is_identity_matrix():
    np.testing.assert_allclose(quat_to_rotation_matrix(0, 0, 0, 1), np.eye(3), atol=1e-12)


def test_quaternion_90deg_about_z():
    # q for +90° about z = (0, 0, sin45, cos45)
    s = math.sin(math.pi / 4)
    R = quat_to_rotation_matrix(0, 0, s, s)
    # x-axis → y-axis
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(R @ np.array([0, 1.0, 0]), [-1, 0, 0], atol=1e-9)


def test_quaternion_is_normalized():
    # non-unit quaternion still yields a proper rotation (det = 1, orthonormal)
    R = quat_to_rotation_matrix(0, 0, 2.0, 2.0)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_orientation_from_3dmol_view():
    o = orientation_from_3dmol_view([1.0, 2.0, 3.0, 0.8, 0.0, 0.0, 0.7071, 0.7071])
    assert o.zoom == pytest.approx(0.8)
    assert (o.qz, o.qw) == pytest.approx((0.7071, 0.7071))
    with pytest.raises(ValueError):
        orientation_from_3dmol_view([1, 2, 3])


def test_apply_orientation_rotates_about_centroid():
    pos = np.array([[1.0, 0, 0], [-1.0, 0, 0]])   # centroid at origin
    s = math.sin(math.pi / 4)
    out = apply_orientation(pos, Orientation(0, 0, s, s))   # +90° about z
    np.testing.assert_allclose(out, [[0, 1, 0], [0, -1, 0]], atol=1e-9)


def test_apply_identity_is_noop():
    pos = np.array([[1.0, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(apply_orientation(pos, Orientation()), pos, atol=1e-12)


def test_load_orientation_both_forms(tmp_path):
    p1 = tmp_path / "view.json"
    p1.write_text(json.dumps({"view": [0, 0, 0, 1.0, 0, 0, 0.7071, 0.7071]}))
    assert load_orientation(p1).qz == pytest.approx(0.7071)
    p2 = tmp_path / "q.json"
    p2.write_text(json.dumps({"qx": 0, "qy": 0, "qz": 0, "qw": 1, "zoom": 2.0}))
    assert load_orientation(p2).zoom == pytest.approx(2.0)


def test_vmd_rotate_matrix_identity():
    s = vmd_rotate_matrix(Orientation())
    assert s.startswith("{{") and s.endswith("}}")
    assert "1.000000" in s and "0.000000" in s
