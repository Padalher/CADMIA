"""CDFT warns when the N±1 states are spin-contaminated."""
from __future__ import annotations

import pytest

from cadmia.cdft.workflow import CDFTInputState, _spin_contamination_warnings
from cadmia.io.electronic import _parse_orca_spin_sq


def _state(spin_sq):
    st = CDFTInputState.__new__(CDFTInputState)
    st.spin_sq = spin_sq
    return st


def test_a_clean_doublet_says_nothing():
    assert _spin_contamination_warnings({"N+1": _state(0.7501)}) == []


def test_mild_contamination_stays_quiet():
    # 0.777 for a doublet is 3.6 % — real ORCA output, not worth interrupting for.
    assert _spin_contamination_warnings({"N-1": _state(0.7768)}) == []


def test_a_badly_contaminated_doublet_is_flagged():
    msgs = _spin_contamination_warnings({"N-1": _state(1.02)})
    assert len(msgs) == 1
    assert "N-1" in msgs[0] and "0.75" in msgs[0]


def test_the_expected_value_is_not_assumed_to_be_a_doublet():
    # A contaminated triplet is measured against 2.00, not against 0.75.
    msgs = _spin_contamination_warnings({"N+1": _state(2.40)})
    assert len(msgs) == 1 and "2.00" in msgs[0]


def test_a_closed_shell_state_has_nothing_to_check():
    assert _spin_contamination_warnings({"N": _state(None)}) == []


def test_both_charged_states_are_checked():
    msgs = _spin_contamination_warnings({"N+1": _state(1.10), "N-1": _state(1.20)})
    assert len(msgs) == 2


# ── parser ──

def test_the_orca_spin_expectation_is_read():
    lines = ["irrelevant", "Expectation value of <S**2>     :     0.776794", "more"]
    assert _parse_orca_spin_sq(lines) == pytest.approx(0.776794)


def test_a_closed_shell_output_has_no_spin_expectation():
    assert _parse_orca_spin_sq(["Total Energy  -1.0", "no spin block here"]) is None
