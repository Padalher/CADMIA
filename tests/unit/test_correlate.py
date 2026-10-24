"""Tests for core.stats, the Correlate Agent, and `cadmia correlate`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cadmia.agents import correlate_agent
from cadmia.cli import main
from cadmia.core.stats import linregress, _student_t_two_tailed


# ── core.stats ──

def test_linregress_perfect_line():
    f = linregress([1, 2, 3, 4], [2, 4, 6, 8])
    assert f.slope == pytest.approx(2.0) and f.intercept == pytest.approx(0.0)
    assert f.r2 == pytest.approx(1.0) and f.p_value == pytest.approx(0.0)


def test_linregress_known_intercept():
    f = linregress([1, 2, 3, 4], [3, 5, 7, 9])   # y = 2x + 1
    assert f.slope == pytest.approx(2.0) and f.intercept == pytest.approx(1.0)


def test_linregress_constant_x_is_none():
    assert linregress([2, 2, 2], [1, 2, 3]) is None


def test_linregress_two_points_has_no_pvalue():
    f = linregress([1, 2], [1, 2])
    assert f.n == 2 and f.p_value is None      # df = 0


def test_student_t_matches_table():
    assert _student_t_two_tailed(2.306, 8) == pytest.approx(0.05, abs=1e-3)


# ── Correlate Agent ──

def _cdft_json(path: Path, omega_minus, duals):
    atoms = [{"index": i + 1, "element": "X", "dual": d} for i, d in enumerate(duals)]
    path.write_text(json.dumps({
        "format": "cadmia-cdft-v1",
        "structure": {"formula": "X2"},
        "global_fd": {"mu_ev": -5.0, "hardness_ev": 2.0, "omega_ev": 6.25,
                      "omega_minus_ev": omega_minus, "omega_plus_ev": omega_minus - 5.0,
                      "nucleophilicity_ev": 3.0, "ip_ev": 6.0, "ea_ev": 4.0},
        "atoms": atoms,
    }), encoding="utf-8")


def _analysis_json(path: Path, e_ads, delta_n, site_index):
    path.write_text(json.dumps({
        "cluster": "c", "probe": "CO2",
        "e_ads_kcal": e_ads,
        "site_atom_index": site_index,
        "charge_transfer": {"delta_n": delta_n, "e_stab_ev": -0.1},
    }), encoding="utf-8")


@pytest.fixture
def study(tmp_path):
    """4 systems, omega_minus = 1..4, with a perfect experimental line y = 2·ω⁻ + 1."""
    d = tmp_path / "study"
    d.mkdir()
    om = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0}
    dn = {"s1": 0.10, "s2": 0.05, "s3": 0.30, "s4": 0.20}   # weakly related
    for sid, o in om.items():
        _cdft_json(d / f"{sid}.cdft.json", o, duals=[o * 0.1, o * 0.2])
        _analysis_json(d / f"{sid}.analysis.json", e_ads=-10.0 - o, delta_n=dn[sid], site_index=2)
    exp = tmp_path / "exp.csv"
    exp.write_text("system,logk\n" + "".join(f"{s},{2*o+1}\n" for s, o in om.items()),
                   encoding="utf-8")
    return d, exp


def test_correlate_merges_cdft_and_analysis_by_system(study):
    d, exp = study
    systems, _ = correlate_agent.load_descriptor_systems([str(d)])
    assert set(systems) == {"s1", "s2", "s3", "s4"}
    # both artifacts merged under one id:
    assert correlate_agent.resolve_descriptor("omega_minus", systems["s2"]) == pytest.approx(2.0)
    assert correlate_agent.resolve_descriptor("delta_n", systems["s2"]) == pytest.approx(0.05)


def test_correlate_perfect_fit(study):
    d, exp = study
    res = correlate_agent.correlate(str(exp), [str(d)], x="omega_minus", y="logk")
    assert res.n_systems_matched == 4
    p = res.pairs[0]
    assert p.n == 4 and p.slope == pytest.approx(2.0) and p.intercept == pytest.approx(1.0)
    assert p.r2 == pytest.approx(1.0)


def test_correlate_atom_resolved_site(study):
    d, exp = study
    res = correlate_agent.correlate(str(exp), [str(d)], x="dual@site", y="logk")
    # dual@site uses site_atom_index=2 → dual = 0.2·ω⁻, still perfectly linear with logk.
    assert res.pairs[0].r2 == pytest.approx(1.0)


def test_correlate_scan_ranks_by_r(study):
    d, exp = study
    res = correlate_agent.correlate(str(exp), [str(d)], x="all", y="logk")
    # omega_minus is perfectly linear with logk → r²≈1 and it sits in the top group.
    assert res.ranked[0].r2 == pytest.approx(1.0)
    om = next(p for p in res.pairs if p.descriptor == "omega_minus")
    assert om.r2 == pytest.approx(1.0)
    # a constant descriptor (μ is the same for every system) yields an undefined fit, ranked last.
    mu = next(p for p in res.pairs if p.descriptor == "mu")
    assert mu.r is None and mu.note


def test_correlate_requires_y_when_multiple_observables(tmp_path, study):
    d, _ = study
    exp = tmp_path / "multi.csv"
    exp.write_text("system,logk,tof\ns1,3,1\ns2,5,2\ns3,7,3\ns4,9,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="several observables"):
        correlate_agent.correlate(str(exp), [str(d)], x="omega_minus")


def test_correlate_abs_x_transform(study):
    d, exp = study
    res = correlate_agent.correlate(str(exp), [str(d)], x="omega_minus", y="logk", x_transform="abs")
    assert res.pairs[0].x_transform == "abs" and res.pairs[0].r2 == pytest.approx(1.0)


def test_correlate_warns_on_unmatched(tmp_path, study):
    d, _ = study
    exp = tmp_path / "extra.csv"
    exp.write_text("system,logk\ns1,3\ns2,5\nGHOST,99\n", encoding="utf-8")
    res = correlate_agent.correlate(str(exp), [str(d)], x="omega_minus", y="logk")
    assert res.n_systems_matched == 2
    assert any("GHOST" in w for w in res.warnings)


# ── CLI ──

def test_cli_correlate_table(study):
    d, exp = study
    r = CliRunner().invoke(main, ["correlate", "-e", str(exp), "-d", str(d),
                                  "--x", "omega_minus", "--y", "logk"])
    assert r.exit_code == 0
    assert "omega_minus" in r.output and "Best fit" in r.output


def test_cli_correlate_json_output(tmp_path, study):
    d, exp = study
    out = tmp_path / "corr.json"
    r = CliRunner().invoke(main, ["correlate", "-e", str(exp), "-d", str(d),
                                  "--x", "all", "--y", "logk", "--format", "json", "-o", str(out)])
    assert r.exit_code == 0 and out.exists()
    payload = json.loads(out.read_text())
    assert payload["n_systems_matched"] == 4 and payload["pairs"]


# ── --cdft broadcast + observable-from-JSON ──

@pytest.fixture
def adducts(tmp_path):
    """One cluster CDFT (atoms dual = 0.1/0.2/0.3) + 3 configs binding at atoms 1/2/3.

    The configs carry NO cdft of their own — dual@site only resolves if the cluster CDFT is
    broadcast. e_ads is a perfect line in dual@site: e_ads = -100·dual@site.
    """
    d = tmp_path / "adducts"
    d.mkdir()
    for i, site in enumerate((1, 2, 3), start=1):
        _analysis_json(d / f"cfg{i}.analysis.json",
                       e_ads=-100.0 * (site * 0.1), delta_n=0.1 * i, site_index=site)
    cdft = tmp_path / "cluster_cdft.json"
    _cdft_json(cdft, omega_minus=1.0, duals=[0.1, 0.2, 0.3])
    return d, cdft


def test_broadcast_enables_dual_at_site(adducts):
    d, cdft = adducts
    # Without broadcast the cluster CDFT is a separate system → dual@site has no atoms → no fit.
    bare = correlate_agent.correlate(None, [str(d)], x="dual@site", y="e_ads")
    assert bare.pairs[0].n == 0
    # With --cdft the atoms[] spread to every config → perfect line, no CSV involved.
    res = correlate_agent.correlate(None, [str(d)], x="dual@site", y="e_ads", cdft=str(cdft))
    assert res.pairs[0].n == 3 and res.pairs[0].r2 == pytest.approx(1.0)


def test_broadcast_is_non_destructive(adducts):
    d, cdft = adducts
    systems, _ = correlate_agent.load_descriptor_systems([str(d)])
    warns = correlate_agent._broadcast_cdft(systems, str(cdft))
    assert not warns
    # per-config site_atom_index survives; the shared atoms[] were added.
    assert systems["cfg2"]["site_atom_index"] == 2
    assert len(systems["cfg2"]["atoms"]) == 3


def test_observable_from_json_without_csv(adducts):
    d, cdft = adducts
    res = correlate_agent.correlate(None, [str(d)], x="delta_n", y="e_ads")
    assert res.n_systems_matched == 3 and res.pairs[0].n == 3


def test_scan_skips_the_self_observable(adducts):
    d, _ = adducts
    res = correlate_agent.correlate(None, [str(d)], x="all", y="e_ads")
    assert all(p.descriptor != "e_ads" for p in res.pairs)   # no trivial e_ads↔e_ads


def test_missing_observable_and_table_errors(adducts):
    d, _ = adducts
    with pytest.raises(ValueError, match="--y"):
        correlate_agent.correlate(None, [str(d)])


def test_cli_correlate_no_csv_with_cdft(adducts):
    d, cdft = adducts
    r = CliRunner().invoke(main, ["correlate", "-d", str(d), "--cdft", str(cdft),
                                  "--x", "dual@site", "--y", "e_ads"])
    assert r.exit_code == 0
    assert "dual@site" in r.output and "3 system(s) matched" in r.output
