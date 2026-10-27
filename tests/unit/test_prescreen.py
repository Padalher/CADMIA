"""Tests for the xTB pre-screen agent and runner (xtb is mocked — no binary needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadmia.agents import prescreen_agent
from cadmia.calculators.xtb_runner import (
    XTBError, XTBResult, _parse_energy, run_xtb_opt, xtb_available as _xtb_available,
)


def _write_xyz(path: Path, coords: list[tuple[str, float, float, float]], comment: str = "") -> Path:
    lines = [str(len(coords)), comment]
    for s, x, y, z in coords:
        lines.append(f"{s} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── xtb_runner ──

def test_parse_energy_from_stdout(tmp_path):
    stdout = "blah\n          | TOTAL ENERGY              -10.123456 Eh   |\nmore\n"
    assert _parse_energy(stdout, tmp_path / "nope.xyz") == pytest.approx(-10.123456)


def test_parse_energy_falls_back_to_xyz_comment(tmp_path):
    opt = _write_xyz(tmp_path / "xtbopt.xyz", [("H", 0, 0, 0)], comment="energy: -5.5 gnorm: 0.001")
    assert _parse_energy("no energy here", opt) == pytest.approx(-5.5)


def test_run_xtb_opt_raises_without_binary(tmp_path, monkeypatch):
    xyz = _write_xyz(tmp_path / "m.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(XTBError, match="not found on PATH"):
        run_xtb_opt(xyz, xtb_path="xtb")


# ── convergence verdict ──
# Fragments below are literal xtb 6.x output captured from real runs.

_CONVERGED_TAIL = """
   *** convergence criteria satisfied after 10 iterations ***
   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS ***
          | TOTAL ENERGY               -5.070544442574 Eh   |
"""

# A run that gives up on the geometry still reports SCF convergence for every
# cycle — the reason the verdict must be anchored on "GEOMETRY OPTIMIZATION".
_FAILED_TAIL = """
   *** convergence criteria satisfied after 10 iterations ***
   *** convergence criteria satisfied after 7 iterations ***
   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 1 ITERATIONS ***
          | TOTAL ENERGY               -4.987139781408 Eh   |
"""


def test_parse_convergence_reads_the_geometry_verdict():
    from cadmia.calculators.xtb_runner import _parse_convergence
    assert _parse_convergence(_CONVERGED_TAIL) is True
    assert _parse_convergence(_FAILED_TAIL) is False
    assert _parse_convergence("no verdict here") is None


def test_scf_convergence_is_not_mistaken_for_geometry_convergence():
    """The failed run says "convergence criteria satisfied" twice — SCF, not geometry."""
    from cadmia.calculators.xtb_runner import _parse_convergence
    assert "convergence criteria satisfied" in _FAILED_TAIL
    assert _parse_convergence(_FAILED_TAIL) is False


def _fake_xtb_run(monkeypatch, stdout: str, *, write_optxyz: bool = True, rc: int = 0):
    """Patch subprocess.run so run_xtb_opt sees *stdout* and an xtbopt.xyz."""
    import subprocess as sp
    from cadmia.calculators import xtb_runner

    def _run(cmd, cwd=None, **kwargs):
        if write_optxyz:
            _write_xyz(Path(cwd) / "xtbopt.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
        return sp.CompletedProcess(cmd, rc, stdout=stdout, stderr="")

    monkeypatch.setattr(xtb_runner.subprocess, "run", _run)
    monkeypatch.setattr(xtb_runner.shutil, "which", lambda _: "/usr/bin/xtb")


def test_run_xtb_opt_flags_an_unconverged_optimization(tmp_path, monkeypatch):
    """xtb writes xtbopt.xyz and exits 0 even when it runs out of cycles."""
    xyz = _write_xyz(tmp_path / "m.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    _fake_xtb_run(monkeypatch, _FAILED_TAIL)
    res = run_xtb_opt(xyz, workdir=tmp_path / "scratch")
    assert res.converged is False
    assert res.energy_hartree == pytest.approx(-4.987139781408)


def test_freeze_holds_the_cluster_with_constrain_not_fix(tmp_path, monkeypatch):
    # $fix let the cluster move 1.6 A under xtb's default optimizer; $constrain holds it
    xyz = _write_xyz(tmp_path / "m.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    _fake_xtb_run(monkeypatch, _CONVERGED_TAIL)
    run_xtb_opt(xyz, workdir=tmp_path / "scratch", fix_atoms=8)
    control = (tmp_path / "scratch" / "xcontrol").read_text()
    assert control.startswith("$constrain") and "atoms: 1-8" in control


def test_run_xtb_opt_accepts_a_converged_optimization(tmp_path, monkeypatch):
    xyz = _write_xyz(tmp_path / "m.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    _fake_xtb_run(monkeypatch, _CONVERGED_TAIL)
    res = run_xtb_opt(xyz, workdir=tmp_path / "scratch")
    assert res.converged is True


def test_run_xtb_opt_without_a_verdict_is_not_called_converged(tmp_path, monkeypatch):
    """No banner ⇒ unverified; the safe reading is "not converged"."""
    xyz = _write_xyz(tmp_path / "m.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    _fake_xtb_run(monkeypatch, "          | TOTAL ENERGY   -1.0 Eh   |\n")
    assert run_xtb_opt(xyz, workdir=tmp_path / "scratch").converged is False


def test_nonzero_exit_is_never_reported_as_converged(tmp_path, monkeypatch):
    """xtb can print the converged banner and still fail; the failure wins."""
    xyz = _write_xyz(tmp_path / "m.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    _fake_xtb_run(monkeypatch, _CONVERGED_TAIL, rc=2)
    res = run_xtb_opt(xyz, workdir=tmp_path / "scratch")
    assert res.converged is False
    assert res.energy_hartree is not None      # the energy is still usable for reporting


def test_dedup_does_not_merge_different_species(tmp_path, monkeypatch):
    """kabsch_rmsd pairs atoms by list position: same points, different elements ⇒ RMSD 0."""
    d = tmp_path / "cfg"
    d.mkdir()
    a = _write_xyz(d / "a.xyz", [("H", 0, 0, 0), ("O", 1.0, 0, 0)])
    b = _write_xyz(d / "b.xyz", [("O", 0, 0, 0), ("H", 1.0, 0, 0)])

    def _fake(xyz_path, **kwargs):
        return XTBResult(energy_hartree=-10.0, optimized_xyz=Path(xyz_path), converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    res = prescreen_agent.prescreen(d, output_dir=tmp_path / "out")
    # Identical coordinates and identical energies, but they are not the same species.
    assert len(res.survivors) == 2, [it.reason for it in res.items]


def test_prescreen_warns_when_xtb_did_not_converge(tmp_path, monkeypatch):
    """The unconverged verdict must reach the user through the prescreen warnings."""
    d = tmp_path / "cfg"
    d.mkdir()
    _write_xyz(d / "c1.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])

    def _fake(xyz_path, **kwargs):
        return XTBResult(energy_hartree=-10.0, optimized_xyz=Path(xyz_path), converged=False)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    res = prescreen_agent.prescreen(d, output_dir=tmp_path / "out")
    assert any("did not converge" in w for w in res.warnings), res.warnings


def test_prescreen_drops_unconverged_configs(tmp_path, monkeypatch):
    """An unconverged geometry is not a minimum: it must not reach DFT."""
    d = tmp_path / "cfg"
    d.mkdir()
    good = _write_xyz(d / "good.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    bad = _write_xyz(d / "bad.xyz", [("H", 0, 0, 0), ("H", 3.0, 0, 0.74)])
    # The unconverged one is the LOWEST in energy: if it were kept it would become the
    # reference and shift every relative energy in the window. The gap is deliberately
    # small (~3 kcal/mol) so the physical-sanity filter has no reason to drop it —
    # convergence must be the only thing that excludes it.
    energies = {good.name: -10.0, bad.name: -10.005}

    def _fake(xyz_path, **kwargs):
        p = Path(xyz_path)
        return XTBResult(energy_hartree=energies[p.name], optimized_xyz=p,
                         converged=(p.name == good.name))

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    res = prescreen_agent.prescreen(d, output_dir=tmp_path / "out")

    kept = {Path(it.source_xyz).name for it in res.survivors}
    assert kept == {good.name}
    dropped = next(it for it in res.items if Path(it.source_xyz).name == bad.name)
    assert not dropped.kept and "did not converge" in dropped.reason
    # The survivor is the reference, so its ΔE is 0 — the unconverged −12.0 was ignored.
    assert res.survivors[0].rel_kcal == pytest.approx(0.0)


def test_prescreen_reports_when_nothing_converged(tmp_path, monkeypatch):
    d = tmp_path / "cfg"
    d.mkdir()
    _write_xyz(d / "c1.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])

    def _fake(xyz_path, **kwargs):
        return XTBResult(energy_hartree=-10.0, optimized_xyz=Path(xyz_path), converged=False)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    res = prescreen_agent.prescreen(d)
    assert res.n_kept == 0
    assert any("nothing to rank" in w for w in res.warnings), res.warnings


@pytest.mark.skipif(not _xtb_available(), reason="xtb binary not installed")
def test_real_xtb_reports_convergence(tmp_path):
    """Guard against xtb changing its wording: run the real binary both ways."""
    xyz = _write_xyz(tmp_path / "h2o.xyz",
                     [("O", 0, 0, 0.5), ("H", 0, 1.25, -0.5), ("H", 0, -1.25, -0.5)])
    ok = run_xtb_opt(xyz, workdir=tmp_path / "ok")
    assert ok.converged is True
    # --cycles 1 cannot relax this stretched geometry, so xtb gives up.
    import cadmia.calculators.xtb_runner as xr
    real_run = xr.subprocess.run

    def _one_cycle(cmd, **kwargs):
        return real_run(list(cmd) + ["--cycles", "1"], **kwargs)

    xr.subprocess.run = _one_cycle
    try:
        bad = run_xtb_opt(xyz, workdir=tmp_path / "bad")
    finally:
        xr.subprocess.run = real_run
    assert bad.converged is False


# ── prescreen agent ──

@pytest.fixture
def configs(tmp_path):
    a = _write_xyz(tmp_path / "site1_a.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    b = _write_xyz(tmp_path / "site1_b.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])  # same geom as a
    c = _write_xyz(tmp_path / "site2_c.xyz", [("H", 0, 0, 0), ("H", 1.0, 0, 0.74)])
    d = _write_xyz(tmp_path / "site3_d.xyz", [("H", 0, 0, 0), ("H", 2.0, 0, 0.74)])
    return tmp_path, {a.name: -10.0, b.name: -10.0, c.name: -9.995, d.name: -9.970}


@pytest.fixture
def mock_xtb(monkeypatch, configs):
    _, energy_map = configs

    def _fake(xyz_path, **kwargs):
        p = Path(xyz_path)
        return XTBResult(energy_hartree=energy_map[p.name], optimized_xyz=p, converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)


def test_prescreen_dedup_and_window(configs, mock_xtb, tmp_path):
    src_dir, _ = configs
    out = tmp_path / "kept"
    result = prescreen_agent.prescreen(src_dir, output_dir=out, energy_window_kcal=10.0)

    assert result.n_input == 4
    kept = {Path(it.source_xyz).name for it in result.survivors}
    # a is the lowest; b is a duplicate of a (same geometry); c is within window; d is outside.
    assert kept == {"site1_a.xyz", "site2_c.xyz"}
    b = next(it for it in result.items if it.source_xyz.endswith("site1_b.xyz"))
    assert not b.kept and "duplicate" in b.reason
    d = next(it for it in result.items if it.source_xyz.endswith("site3_d.xyz"))
    assert not d.kept and "window" in d.reason
    # survivors' optimized geometries were written out
    assert (out / "site1_a_xtbopt.xyz").exists()


def test_prescreen_reports_progress(configs, mock_xtb):
    src_dir, _ = configs
    seen = []
    prescreen_agent.prescreen(
        src_dir, energy_window_kcal=100.0,
        on_progress=lambda done, total, name, energy, err: seen.append((done, total, name)),
    )
    assert len(seen) == 4                  # one callback per input config
    assert [s[0] for s in seen] == [1, 2, 3, 4]
    assert all(s[1] == 4 for s in seen)    # total reported each time


def test_prescreen_top_n_cap(configs, mock_xtb):
    src_dir, _ = configs
    result = prescreen_agent.prescreen(src_dir, top_n=1, energy_window_kcal=100.0)
    assert result.n_kept == 1
    assert result.survivors[0].source_xyz.endswith("site1_a.xyz")  # the lowest


def test_prescreen_handles_xtb_failure(configs, monkeypatch, tmp_path):
    src_dir, _ = configs

    def _boom(xyz_path, **kwargs):
        raise XTBError("xtb blew up")

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _boom)
    result = prescreen_agent.prescreen(src_dir)
    assert result.n_kept == 0
    assert all(it.error for it in result.items)
    assert any("xtb blew up" in w for w in result.warnings)


# ── physical-sanity filter: nonphysical minima must not poison the ranking ──

def test_min_pair_distance():
    assert prescreen_agent._min_pair_distance([[0, 0, 0], [0, 0, 1.5], [3, 0, 0]]) == pytest.approx(1.5)
    assert prescreen_agent._min_pair_distance([[0.0, 0.0, 0.0]]) == float("inf")


def test_nonphysical_reasons_energy_outlier():
    # far below the median → spurious minimum
    r = prescreen_agent._nonphysical_reasons(-20.0, None, median_energy=-9.99)
    assert r and "below the median" in r[0]
    # normal spread → physical
    assert prescreen_agent._nonphysical_reasons(-9.97, None, median_energy=-9.99) == []


def test_nonphysical_reasons_collapsed_atoms():
    r = prescreen_agent._nonphysical_reasons(-10.0, [[0, 0, 0], [0.2, 0, 0]], median_energy=-10.0)
    assert r and "collapsed" in r[0]


@pytest.fixture
def poisoned(tmp_path):
    r1 = _write_xyz(tmp_path / "real1.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    r2 = _write_xyz(tmp_path / "real2.xyz", [("H", 0, 0, 0), ("H", 0.5, 0, 0.74)])
    r3 = _write_xyz(tmp_path / "real3.xyz", [("H", 0, 0, 0), ("H", 1.0, 0, 0.74)])
    bad = _write_xyz(tmp_path / "bad.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    return tmp_path, {r1.name: -10.00, r2.name: -9.99, r3.name: -9.97, bad.name: -20.0}


@pytest.fixture
def mock_xtb_poisoned(monkeypatch, poisoned):
    _, energy_map = poisoned

    def _fake(xyz_path, **kwargs):
        p = Path(xyz_path)
        return XTBResult(energy_hartree=energy_map[p.name], optimized_xyz=p, converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)


def test_prescreen_drops_energy_poison(poisoned, mock_xtb_poisoned):
    src, _ = poisoned
    result = prescreen_agent.prescreen(src, energy_window_kcal=10.0)
    bad = next(it for it in result.items if it.source_xyz.endswith("bad.xyz"))
    assert not bad.kept and not bad.physical and "nonphysical" in bad.reason
    # the real lowest is the reference (rank 1, +0.00), NOT the −20 Eh garbage
    r1 = next(it for it in result.items if it.source_xyz.endswith("real1.xyz"))
    assert r1.kept and r1.rank == 1 and r1.rel_kcal == pytest.approx(0.0, abs=1e-6)
    assert any("nonphysical" in w for w in result.warnings)


def test_prescreen_no_sanity_keeps_poison(poisoned, mock_xtb_poisoned):
    src, _ = poisoned
    result = prescreen_agent.prescreen(src, energy_window_kcal=10.0, sanity=False, min_keep=0)
    bad = next(it for it in result.items if it.source_xyz.endswith("bad.xyz"))
    assert bad.physical and bad.rank == 1   # filter off → garbage becomes the reference


# ── min_keep guarantee: DFT must get ≥N distinct candidates even outside the window ──

@pytest.fixture
def spread(tmp_path):
    """Three distinct minima; only the lowest lies within a 10 kcal/mol window."""
    a = _write_xyz(tmp_path / "a.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    b = _write_xyz(tmp_path / "b.xyz", [("H", 0, 0, 0), ("H", 0.5, 0, 0.74)])
    c = _write_xyz(tmp_path / "c.xyz", [("H", 0, 0, 0), ("H", 1.0, 0, 0.74)])
    # a=0, b≈+31 kcal/mol, c≈+63 kcal/mol → with a 10 kcal/mol window only `a` qualifies.
    return tmp_path, {a.name: -10.00, b.name: -9.95, c.name: -9.90}


@pytest.fixture
def mock_xtb_spread(monkeypatch, spread):
    _, energy_map = spread

    def _fake(xyz_path, **kwargs):
        p = Path(xyz_path)
        return XTBResult(energy_hartree=energy_map[p.name], optimized_xyz=p, converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)


def test_prescreen_min_keep_forces_second_candidate(spread, mock_xtb_spread):
    # window keeps only `a`, but min_keep=2 must also carry the 2nd-lowest distinct minimum.
    result = prescreen_agent.prescreen(spread[0], energy_window_kcal=10.0, min_keep=2)
    assert result.n_kept == 2
    kept_ranks = sorted(it.rank for it in result.survivors)
    assert kept_ranks == [1, 2]
    forced = next(it for it in result.survivors if it.rank == 2)
    assert "beyond window" in forced.reason
    assert any("guarantee" in w for w in result.warnings)


def test_prescreen_min_keep_zero_is_window_only(spread, mock_xtb_spread):
    result = prescreen_agent.prescreen(spread[0], energy_window_kcal=10.0, min_keep=0)
    assert result.n_kept == 1   # only `a` within the window; no guarantee


def test_prescreen_min_keep_capped_by_top_n(spread, mock_xtb_spread):
    # top_n=1 must win over min_keep=2 (never keep more than the explicit cap).
    result = prescreen_agent.prescreen(spread[0], energy_window_kcal=10.0, min_keep=2, top_n=1)
    assert result.n_kept == 1


def test_prescreen_pool_has_all_good_minima_in_energy_order(spread, mock_xtb_spread, tmp_path):
    out = tmp_path / "kept"
    # wide window + no cap → all three distinct minima selected, all in the pool file.
    result = prescreen_agent.prescreen(
        spread[0], energy_window_kcal=100.0, top_n=0, output_dir=out,
    )
    assert result.n_kept == 3
    assert result.candidates_xyz is not None
    pool = Path(result.candidates_xyz)
    assert pool.name == "candidates_by_energy.xyz"
    assert pool.parent.name == "pool" and pool.exists()
    text = pool.read_text()
    # three 2-atom H2 blocks → 3 × (1 + 1 + 2) = 12 lines
    assert len(text.splitlines()) == 12
    comments = [ln for ln in text.splitlines() if ln.startswith("rank ")]
    assert [c.split()[1] for c in comments] == ["1", "2", "3"]   # ascending energy
    assert all("[to-DFT]" in c for c in comments)                    # all selected here


def test_prescreen_pool_keeps_reserves_out_of_glob(spread, mock_xtb_spread, tmp_path):
    """A narrow window selects only 1, but the other good minima are preserved as
    reserves under pool/reserve/ (NOT loose in the output dir), so prepare's *.xyz glob
    never ingests them and no xTB re-run is needed to promote one later."""
    out = tmp_path / "kept"
    result = prescreen_agent.prescreen(
        spread[0], energy_window_kcal=10.0, min_keep=1, output_dir=out,
    )
    assert result.n_kept == 1 and result.n_reserve == 2
    # the pool file still lists ALL three good minima, tagged.
    text = Path(result.candidates_xyz).read_text()
    comments = [ln for ln in text.splitlines() if ln.startswith("rank ")]
    assert sum("[to-DFT]" in c for c in comments) == 1
    assert sum("reserve" in c for c in comments) == 2
    # main dir has ONLY the one selected *_xtbopt.xyz (safe for prepare's glob).
    loose = sorted(p.name for p in out.glob("*.xyz"))
    assert len(loose) == 1
    # reserves live under pool/reserve/, ready to promote.
    reserves = sorted(p.name for p in (out / "pool" / "reserve").glob("*.xyz"))
    assert len(reserves) == 2


def test_prescreen_no_pool_without_output_dir(spread, mock_xtb_spread):
    result = prescreen_agent.prescreen(spread[0], energy_window_kcal=100.0)
    assert result.candidates_xyz is None and result.n_reserve == 0


def test_prescreen_min_keep_bounded_by_distinct_minima(configs, mock_xtb):
    # a and b are duplicates → only 3 distinct minima; asking for 5 keeps at most 3.
    src_dir, _ = configs
    result = prescreen_agent.prescreen(src_dir, energy_window_kcal=1.0, min_keep=5)
    assert result.n_kept == 3
    assert any("distinct xTB min" in w for w in result.warnings)


# ── CLI + sidecar: the JSON record is written automatically, with no flag ──

def test_prescreen_writes_json_sidecar_in_output_dir(configs, mock_xtb, tmp_path):
    """Why each config was kept or dropped survives only here: the survivors' .xyz cannot say."""
    src_dir, _ = configs
    out = tmp_path / "kept"
    result = prescreen_agent.prescreen(src_dir, output_dir=out, energy_window_kcal=10.0)

    sidecar = out / "prescreen.json"
    assert result.output_json == str(sidecar) and sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["n_input"] == 4 and data["n_kept"] == result.n_kept
    dropped = [it for it in data["items"] if not it["kept"]]
    assert any("duplicate" in it["reason"] for it in dropped)


def test_prescreen_no_sidecar_without_output_dir(configs, mock_xtb):
    result = prescreen_agent.prescreen(configs[0], energy_window_kcal=10.0)
    assert result.output_json is None


def test_prescreen_cli_announces_the_sidecar(configs, mock_xtb, monkeypatch, tmp_path):
    from click.testing import CliRunner

    from cadmia.calculators import xtb_runner
    from cadmia.cli import main

    monkeypatch.setattr(xtb_runner, "xtb_available", lambda *a, **k: True)
    src_dir, _ = configs
    out = tmp_path / "kept"
    res = CliRunner().invoke(main, [
        "prescreen", *[str(p) for p in sorted(src_dir.glob("*.xyz"))], "-o", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert (out / "prescreen.json").exists()
    assert "JSON → " in res.output
    assert "Survivors for DFT" in res.output       # the table still prints


# ── A config with an energy but no readable geometry must not rank ──

@pytest.fixture
def unreadable_relaxed_geometry(tmp_path, monkeypatch):
    """xTB returns an energy for every config, but one relaxed .xyz comes back truncated.

    That is the order the runner works in: the energy is parsed before the geometry, so a
    truncated xtbopt.xyz still leaves a perfectly plausible number behind.
    """
    src = tmp_path / "configs"
    src.mkdir()
    for name in ("a.xyz", "b.xyz", "c.xyz"):
        _write_xyz(src / name, [("H", 0, 0, 0), ("H", 0, 0, 0.74)])

    relaxed = tmp_path / "relaxed"
    relaxed.mkdir()
    ok = _write_xyz(relaxed / "ok.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    truncated = relaxed / "truncated.xyz"
    truncated.write_text("2\nsays two, has one\nH 0.0 0.0 0.0\n", encoding="utf-8")

    # c is the LOWEST energy: if it ranks, it sets the zero for everything else.
    energies = {"a.xyz": -5.000, "b.xyz": -5.001, "c.xyz": -5.002}

    def _fake(xyz_path, **kwargs):
        p = Path(xyz_path)
        return XTBResult(energy_hartree=energies[p.name],
                         optimized_xyz=truncated if p.name == "c.xyz" else ok,
                         converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    return src


def test_config_without_a_readable_geometry_is_dropped(unreadable_relaxed_geometry, tmp_path):
    out = tmp_path / "out"
    result = prescreen_agent.prescreen(unreadable_relaxed_geometry, output_dir=out)

    c = next(it for it in result.items if it.source_xyz.endswith("c.xyz"))
    assert not c.kept
    assert c.rank is None
    assert c.optimized_xyz is None
    assert "geometry could not be read" in c.reason
    assert any("geometry could not be read" in w for w in result.warnings)


def test_an_unreadable_geometry_does_not_set_the_energy_zero(unreadable_relaxed_geometry):
    """The dropped config was the lowest; the zero must fall on the lowest USABLE one."""
    result = prescreen_agent.prescreen(unreadable_relaxed_geometry)
    b = next(it for it in result.items if it.source_xyz.endswith("b.xyz"))
    assert b.rel_kcal == pytest.approx(0.0)


def test_n_kept_matches_the_files_actually_written(unreadable_relaxed_geometry, tmp_path):
    out = tmp_path / "out"
    result = prescreen_agent.prescreen(unreadable_relaxed_geometry, output_dir=out)
    assert result.n_kept == len(list(out.glob("*.xyz")))
    assert all(it.optimized_xyz for it in result.survivors)


def test_the_scratch_of_a_failed_run_is_kept(tmp_path, monkeypatch):
    """A config whose xTB run failed keeps its scratch dir, and the result says where.

    item.error holds the exception message but not xTB's stdout, so wiping the scratch
    unconditionally made a failure impossible to diagnose after the fact.
    """
    a = _write_xyz(tmp_path / "a.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
    b = _write_xyz(tmp_path / "b.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.75)])

    def _fake(xyz_path, **kw):
        wd = Path(kw["workdir"])
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "xtb.out").write_text("xtb said something", encoding="utf-8")
        if Path(xyz_path).name == "b.xyz":
            raise XTBError("boom")
        opt = _write_xyz(wd / "xtbopt.xyz", [("H", 0, 0, 0), ("H", 0, 0, 0.74)])
        return XTBResult(energy_hartree=-1.0, optimized_xyz=opt, converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    out = tmp_path / "out"
    res = prescreen_agent.prescreen([a, b], output_dir=out)

    scratch = out / ".xtb_scratch"
    assert (scratch / "b" / "xtb.out").exists()
    assert not (scratch / "a").exists()
    assert any("kept for diagnosis" in w for w in res.warnings)


def test_the_dedup_tolerance_does_not_loosen_as_the_cluster_grows(tmp_path, monkeypatch):
    """Two configs whose probe sits 0.6 Å apart are distinct minima, cluster size aside.

    With an RMSD averaged over every atom, that probe displacement still falls under the
    0.20 Å tolerance once the cluster is big enough (16 atoms + a 4-atom probe already
    does it), so the two would collapse into one and only one would reach DFT.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=2.0, size=(16, 3))
    probe_a = np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 7.1], [0.9, 0.0, 7.4], [-0.9, 0.0, 7.4]])
    probe_b = probe_a + np.array([0.6, 0.0, 0.0])
    syms = ["Ti"] * 4 + ["O"] * 12 + ["N", "H", "H", "H"]

    def _xyz(path, coords):
        lines = [str(len(syms)), ""]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(syms, coords)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    a = _xyz(tmp_path / "a.xyz", np.vstack([cluster, probe_a]))
    b = _xyz(tmp_path / "b.xyz", np.vstack([cluster, probe_b]))

    from cadmia.io.xyz import parse_xyz
    from cadmia.utils.geometry import kabsch_rmsd, kabsch_max_deviation
    pa = np.asarray(parse_xyz(a).positions, float)
    pb = np.asarray(parse_xyz(b).positions, float)
    # the premise: the mean does not see it, the worst atom does
    assert kabsch_rmsd(pa, pb) < 0.20
    assert kabsch_max_deviation(pa, pb) > 0.20

    energies = {"a.xyz": -1.0000, "b.xyz": -1.0001}   # 0.06 kcal/mol apart → dedup pairs them

    def _fake(xyz_path, **kw):
        src = Path(xyz_path)
        return XTBResult(energy_hartree=energies[src.name], optimized_xyz=src, converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    res = prescreen_agent.prescreen([a, b], output_dir=tmp_path / "out")
    assert res.n_kept == 2
    assert all("duplicate" not in (it.reason or "") for it in res.items)


def test_the_same_minimum_written_in_two_atom_orders_is_one_candidate(tmp_path, monkeypatch):
    """Two configs that relaxed to the same structure count once, whatever the atom order.

    CO2 modes write their probe as C,O,O or as O,C,O, and the same relaxed carbonate
    reached from two of them used to be skipped as "a different species" — so both went
    to DFT, twice the cost for one structure.
    """
    coords = [("Ti", 0, 0, 0), ("O", 1.6, 0, 0), ("O", 0, 1.6, 0),
              ("C", 4.5, 0, 0), ("O", 5.66, 0, 0), ("O", 3.34, 0, 0)]
    swapped = coords[:3] + [coords[4], coords[3], coords[5]]   # C and one O swap places
    a = _write_xyz(tmp_path / "a.xyz", coords)
    b = _write_xyz(tmp_path / "b.xyz", swapped)
    c = _write_xyz(tmp_path / "c.xyz", coords[:3] + [("C", 4.5, 0, 2.0),
                                                     ("O", 5.66, 0, 2.0), ("O", 3.34, 0, 2.0)])
    energies = {"a.xyz": -10.0, "b.xyz": -10.0, "c.xyz": -9.99}

    def _fake(xyz_path, **kw):
        p = Path(xyz_path)
        return XTBResult(energy_hartree=energies[p.name], optimized_xyz=p, converged=True)

    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)
    res = prescreen_agent.prescreen([a, b, c], output_dir=tmp_path / "out")
    kept = {Path(it.source_xyz).name for it in res.survivors}
    assert kept == {"a.xyz", "c.xyz"}
    assert any("duplicate" in (it.reason or "") for it in res.items
               if Path(it.source_xyz).name == "b.xyz")


# ── The energy window must not mix cluster isomers ───────────────────

def _isomer_batch(tmp_path):
    """Four configs: two keep the reference cluster skeleton, two isomerize it (and are
    ~56 kcal/mol lower, so a global window would discard the intact ones)."""
    d = tmp_path / "cfg"
    d.mkdir()
    ref = _write_xyz(tmp_path / "cluster.xyz",
                     [("Al", 0, 0, 0), ("O", 1.7, 0, 0), ("Al", 3.4, 0, 0)])
    intact = [("Al", 0, 0, 0), ("O", 1.7, 0, 0), ("Al", 3.4, 0, 0)]
    bent   = [("Al", 0, 0, 0), ("O", 1.7, 0, 0), ("Al", 1.0, 1.5, 0)]   # Al–Al bond appears
    _write_xyz(d / "intact_a.xyz", intact + [("H", 1.7, 2.5, 0)])
    _write_xyz(d / "intact_b.xyz", intact + [("H", 1.7, -2.5, 0)])
    _write_xyz(d / "isomer_a.xyz", bent + [("H", 1.7, 2.5, 0)])
    _write_xyz(d / "isomer_b.xyz", bent + [("H", 1.7, -2.5, 0)])
    energies = {"intact_a": -10.00, "intact_b": -10.01,
                "isomer_a": -10.10, "isomer_b": -10.11}
    return d, ref, energies


def _fake_by_name(monkeypatch, energies):
    def _fake(xyz_path, **kwargs):
        return XTBResult(energy_hartree=energies[Path(xyz_path).stem],
                         optimized_xyz=Path(xyz_path), converged=True)
    monkeypatch.setattr(prescreen_agent, "run_xtb_opt", _fake)


def test_the_window_is_measured_within_each_skeleton_class(tmp_path, monkeypatch):
    d, ref, energies = _isomer_batch(tmp_path)
    _fake_by_name(monkeypatch, energies)
    res = prescreen_agent.prescreen(d, output_dir=tmp_path / "out", min_keep=0,
                                    cluster_ref=ref)
    classes = {Path(it.source_xyz).stem: it.cluster_skeleton for it in res.survivors}
    assert classes.get("intact_a") == "reference"
    assert classes.get("isomer_b") == "isomer-1"
    # Each class is ranked on its own zero, so the lowest of each sits at 0 kcal/mol.
    zeros = {it.cluster_skeleton for it in res.survivors if it.rel_kcal == 0.0}
    assert zeros == {"reference", "isomer-1"}


def test_without_a_cluster_reference_the_window_is_global(tmp_path, monkeypatch):
    d, _ref, energies = _isomer_batch(tmp_path)
    _fake_by_name(monkeypatch, energies)
    res = prescreen_agent.prescreen(d, output_dir=tmp_path / "out", min_keep=0)
    kept = {Path(it.source_xyz).stem for it in res.survivors}
    assert kept == {"isomer_a", "isomer_b"}          # the intact ones fall outside
    assert all(it.cluster_skeleton is None for it in res.items)


def test_a_cluster_reference_with_a_different_element_order_warns(tmp_path, monkeypatch):
    d, _ref, energies = _isomer_batch(tmp_path)
    bad = _write_xyz(tmp_path / "bad.xyz",
                     [("O", 0, 0, 0), ("Al", 1.7, 0, 0), ("Al", 3.4, 0, 0)])
    _fake_by_name(monkeypatch, energies)
    res = prescreen_agent.prescreen(d, output_dir=tmp_path / "out", min_keep=0, cluster_ref=bad)
    assert any("element order" in w for w in res.warnings), res.warnings
    assert {Path(it.source_xyz).stem for it in res.survivors} == {"isomer_a", "isomer_b"}


def test_the_skeleton_class_reaches_the_json(tmp_path, monkeypatch):
    d, ref, energies = _isomer_batch(tmp_path)
    _fake_by_name(monkeypatch, energies)
    out = tmp_path / "out"
    prescreen_agent.prescreen(d, output_dir=out, min_keep=0, cluster_ref=ref)
    items = json.loads((out / "prescreen.json").read_text())["items"]
    assert {it["cluster_skeleton"] for it in items if it["kept"]} == {"reference", "isomer-1"}
