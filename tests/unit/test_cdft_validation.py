"""Validation: CADMIA CDFT workflow vs Multiwfn reference (fixtures/1tial_cdft/CDFT.txt).

Reference values are taken verbatim from that file,
which was produced by Multiwfn for the Al2TiO5 system at
PBE1PBE/Def2TZVP level.  Hirshfeld charges were computed by Multiwfn from the
.wfn files using Stockholder partitioning.
"""
import pytest
from pathlib import Path

from cadmia.cdft.workflow import CDFTInputState, run_cdft_workflow

BASE = Path(__file__).parents[1] / "fixtures"

# Hirshfeld charges from Multiwfn (CDFT.txt, 8 atoms: Ti O O O Al O Al O)
Q_N   = [ 0.7399, -0.4274, -0.4476, -0.4475,  0.7065, -0.3911,  0.7065, -0.4393]
Q_NP1 = [ 0.6351, -0.4939, -0.4855, -0.4854,  0.3542, -0.4085,  0.3543, -0.4701]
Q_NM1 = [ 0.9157, -0.1673, -0.3338, -0.3337,  0.8261, -0.3527,  0.8261, -0.3803]


@pytest.fixture(scope="module")
def result():
    sn   = CDFTInputState.from_wfn(BASE / "N.wfn",   charges=Q_N,   charge_scheme="hirshfeld")
    snp1 = CDFTInputState.from_wfn(BASE / "N+1.wfn", charges=Q_NP1, charge_scheme="hirshfeld")
    snm1 = CDFTInputState.from_wfn(BASE / "N-1.wfn", charges=Q_NM1, charge_scheme="hirshfeld")
    return run_cdft_workflow(sn, snp1, snm1)


# --- Global finite-difference descriptors ---

def test_ip(result):
    assert result.global_fd.ip_ev == pytest.approx(10.3381, abs=0.002)


def test_ea(result):
    assert result.global_fd.ea_ev == pytest.approx(2.3392, abs=0.002)


def test_hardness(result):
    assert result.global_fd.hardness_ev == pytest.approx(7.9988, abs=0.002)


def test_softness(result):
    # S = 1 / η = 1 / 7.9988 ≈ 0.1250 eV^-1
    assert result.global_fd.softness == pytest.approx(1.0 / 7.9988, abs=0.0001)


def test_electrophilicity(result):
    assert result.global_fd.omega_ev == pytest.approx(2.5115, abs=0.002)


def test_nucleophilicity(result):
    assert result.global_fd.nucleophilicity_ev == pytest.approx(0.8274, abs=0.005)


def test_homo_n(result):
    assert result.global_fd.homo_n_ev == pytest.approx(-8.2938, abs=0.01)


def test_homo_np1(result):
    assert result.global_fd.homo_np1_ev == pytest.approx(-0.7808, abs=0.01)


def test_homo_nm1(result):
    assert result.global_fd.homo_nm1_ev == pytest.approx(-13.8271, abs=0.01)


# --- Fukui functions (condensed) ---

def test_fukui_minus_ti(result):
    # f-(Ti) = q(N-1) - q(N) = 0.9157 - 0.7399 = 0.1758
    assert result.fukui.f_minus[0] == pytest.approx(0.1758, abs=0.0002)


def test_fukui_plus_ti(result):
    # f+(Ti) = q(N) - q(N+1) = 0.7399 - 0.6351 = 0.1048
    assert result.fukui.f_plus[0] == pytest.approx(0.1048, abs=0.0002)


def test_fukui_plus_al(result):
    # f+(Al5) = q(N) - q(N+1) = 0.7065 - 0.3542 = 0.3523
    assert result.fukui.f_plus[4] == pytest.approx(0.3523, abs=0.0002)


def test_fukui_minus_o2(result):
    # f-(O2) = 0.2601
    assert result.fukui.f_minus[1] == pytest.approx(0.2601, abs=0.0002)


def test_dual_ti(result):
    # CDD(Ti) = f+ - f- = 0.1048 - 0.1758 = -0.0710
    assert result.fukui.dual[0] == pytest.approx(-0.0710, abs=0.0002)


def test_dual_al(result):
    # CDD(Al5) = f+ - f- = 0.3523 - 0.1196 = 0.2327 ≈ 0.2328
    assert result.fukui.dual[4] == pytest.approx(0.2328, abs=0.0003)


# --- Local softnesses (must use FD softness S = 3.401919 Ha^-1) ---

def test_local_softness_minus_ti(result):
    # s-(Ti) = S * f-(Ti) = 3.401919 * 0.1758 = 0.5981
    assert result.fukui.local_softness_minus[0] == pytest.approx(0.5981, abs=0.003)


def test_local_softness_plus_al(result):
    # s+(Al5) = S * f+(Al5) = 3.401919 * 0.3523 = 1.1986
    assert result.fukui.local_softness_plus[4] == pytest.approx(1.1986, abs=0.003)


def test_local_softness_minus_o2(result):
    # s-(O2) = 3.401919 * 0.2601 = 0.8848
    assert result.fukui.local_softness_minus[1] == pytest.approx(0.8848, abs=0.003)


# --- Local electrophilicity ω * f+ ---

def test_local_electrophilicity_ti(result):
    # ω * f+(Ti) = 2.5115 * 0.1048 = 0.2632 ≈ 0.26315
    assert result.fukui.local_electrophilicity[0] == pytest.approx(0.26315, abs=0.002)


def test_local_electrophilicity_al(result):
    # ω * f+(Al5) = 2.5115 * 0.3523 = 0.8847 ≈ 0.88490
    assert result.fukui.local_electrophilicity[4] == pytest.approx(0.88490, abs=0.003)


# --- Local nucleophilicity N * f- ---

def test_local_nucleophilicity_o2(result):
    # N * f-(O2) = 0.8274 * 0.2601 = 0.21520
    assert result.fukui.local_nucleophilicity[1] == pytest.approx(0.21520, abs=0.003)


def test_local_nucleophilicity_ti(result):
    # N * f-(Ti) = 0.8274 * 0.1758 = 0.14546 ≈ 0.14547
    assert result.fukui.local_nucleophilicity[0] == pytest.approx(0.14547, abs=0.003)
