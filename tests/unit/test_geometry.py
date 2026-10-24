"""Tests for cadmia/utils/geometry.py — superposition helpers."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.utils.geometry import kabsch_deviations, kabsch_max_deviation


def test_kabsch_deviations_is_what_the_max_reduces():
    P = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Q = P.copy()
    Q[2] = Q[2] + [0.0, 0.3, 0.0]

    per_atom = kabsch_deviations(P, Q)
    assert len(per_atom) == 4
    assert float(per_atom.max()) == pytest.approx(kabsch_max_deviation(P, Q))


def test_a_rigid_motion_displaces_no_atom():
    P = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = P @ R.T + np.array([5.0, 2.0, 1.0])

    assert float(kabsch_deviations(P, moved).max()) == pytest.approx(0.0, abs=1e-9)
