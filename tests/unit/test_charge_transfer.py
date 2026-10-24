"""Tests for electrodonating/accepting powers (ω⁻/ω⁺) and two-species charge transfer (ΔN)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.cdft.calculator import compute_finite_difference_descriptors
from cadmia.core.charge_transfer import compute_charge_transfer
from cadmia.core.electronic_descriptors import (
    compute_global_descriptors,
    electrodonating_accepting_powers,
)

_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Ti": 22}


# ── Point 1: electrodonating (ω⁻) / electroaccepting (ω⁺) powers ──

def test_electrodonating_accepting_hand_values():
    # I = 6, A = 4 (eV): η = 2, μ = -5.
    omega_minus, omega_plus = electrodonating_accepting_powers(6.0, 4.0)
    assert omega_minus == pytest.approx(484.0 / 32.0, abs=1e-9)   # (3I+A)²/16η = 22²/32
    assert omega_plus == pytest.approx(324.0 / 32.0, abs=1e-9)    # (I+3A)²/16η = 18²/32


def test_omega_plus_minus_identity_equals_mu():
    # Exact CDFT identity: ω⁺ − ω⁻ = μ = −(I+A)/2.
    for ip, ea in [(6.0, 4.0), (8.5, -1.5), (7.0, 0.3)]:
        omega_minus, omega_plus = electrodonating_accepting_powers(ip, ea)
        mu = -(ip + ea) / 2.0
        assert (omega_plus - omega_minus) == pytest.approx(mu, abs=1e-9)


def test_electrodonating_requires_positive_hardness():
    with pytest.raises(ValueError):
        electrodonating_accepting_powers(4.0, 4.0)   # η = 0
    with pytest.raises(ValueError):
        electrodonating_accepting_powers(3.0, 5.0)   # η < 0


def test_global_descriptors_carry_omega_plus_minus():
    d = compute_global_descriptors(homo_ev=-6.0, lumo_ev=-4.0)
    om, op = electrodonating_accepting_powers(d.ip_ev, d.ea_ev)
    assert d.omega_minus_ev == pytest.approx(om, abs=1e-9)
    assert d.omega_plus_ev == pytest.approx(op, abs=1e-9)
    assert (d.omega_plus_ev - d.omega_minus_ev) == pytest.approx(d.mu_ev, abs=1e-9)


def test_finite_difference_carries_omega_plus_minus():
    d = compute_finite_difference_descriptors(
        energy_n_hartree=-100.0,
        energy_np1_hartree=-100.05,   # anion lower → EA > 0
        energy_nm1_hartree=-99.70,    # cation higher → IP > 0
    )
    om, op = electrodonating_accepting_powers(d.ip_ev, d.ea_ev)
    assert d.omega_minus_ev == pytest.approx(om, abs=1e-9)
    assert d.omega_plus_ev == pytest.approx(op, abs=1e-9)
    assert (d.omega_plus_ev - d.omega_minus_ev) == pytest.approx(d.mu_ev, abs=1e-6)


# ── Point 2: two-species charge transfer (Parr–Pearson) ──

def test_charge_transfer_hand_value_and_direction():
    # A: μ=-3, η=2 ; B: μ=-5, η=3.  ΔN = (μ_A−μ_B)/(η_A+η_B) = 2/5 = 0.4 (A donates).
    ct = compute_charge_transfer(-3.0, 2.0, -5.0, 3.0, label_a="cluster", label_b="probe")
    assert ct.delta_n == pytest.approx(0.4, abs=1e-9)
    assert ct.delta_e_ev == pytest.approx(-4.0 / 10.0, abs=1e-9)   # −(μ_A−μ_B)²/(2(η_A+η_B))
    assert ct.donor == "cluster" and ct.acceptor == "probe"


def test_charge_transfer_direction_flips_with_potential():
    # Lower-μ species donates nothing; the higher-μ one does. Swap → probe donates.
    ct = compute_charge_transfer(-5.0, 3.0, -3.0, 2.0, label_a="cluster", label_b="probe")
    assert ct.delta_n < 0.0                       # negative ⇒ flows B→A
    assert ct.donor == "probe" and ct.acceptor == "cluster"


def test_charge_transfer_zero_for_identical_species():
    ct = compute_charge_transfer(-4.0, 2.0, -4.0, 2.0)
    assert ct.delta_n == pytest.approx(0.0, abs=1e-12)
    assert ct.delta_e_ev == pytest.approx(0.0, abs=1e-12)


def test_charge_transfer_requires_positive_hardness():
    with pytest.raises(ValueError):
        compute_charge_transfer(-3.0, 0.0, -5.0, 3.0)
    with pytest.raises(ValueError):
        compute_charge_transfer(-3.0, 2.0, -5.0, -1.0)


# ── Integration: analyze() surfaces the CDFT charge transfer ──

def _gauss(path: Path, atoms, energy, homo_ha, lumo_ha):
    lines = [
        "", " Standard orientation:",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
    ]
    for i, (sym, x, y, z) in enumerate(atoms, 1):
        lines.append(f"   {i:>4}    {_Z[sym]:>6}           0     {x:11.6f} {y:11.6f} {z:11.6f}")
    lines.append(" ---------------------------------------------------------------------")
    lines.append(f" SCF Done:  E(RB3LYP) = {energy:.8f}     A.U. after   10 cycles")
    lines.append(f" Alpha  occ. eigenvalues --  {homo_ha:.5f}")
    lines.append(f" Alpha virt. eigenvalues --  {lumo_ha:.5f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyze_reports_cdft_charge_transfer(tmp_path):
    from cadmia.agents.analysis_agent import analyze
    cluster = tmp_path / "cluster.out"
    probe = tmp_path / "probe.out"
    adduct = tmp_path / "adduct.out"
    # Cluster has the higher chemical potential (less bound HOMO) → it should donate.
    _gauss(cluster, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0)], -100.0, homo_ha=-0.18, lumo_ha=-0.05)
    _gauss(probe, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17, homo_ha=-0.40, lumo_ha=-0.02)
    _gauss(adduct, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                    ("H", 0, 0, 1.9), ("H", 0, 0, 2.72)], -101.20, homo_ha=-0.20, lumo_ha=-0.05)
    r = analyze(adduct, cluster, probe)
    assert r.charge_transfer is not None
    ct = r.charge_transfer
    assert ct["donor"] == "cluster" and ct["acceptor"] == "probe"
    assert ct["delta_n"] > 0.0
    assert ct["e_stab_ev"] <= 0.0
