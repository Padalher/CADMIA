"""Tests for the Rank Agent and `cadmia rank` (order DFT outputs by energy)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cadmia.agents import rank_agent
from cadmia.cli import main

FIX = Path(__file__).resolve().parents[1] / "fixtures"
G = FIX / "1tial_cdft"          # Gaussian N / N+1 / N-1
O = FIX / "1tial_orca"          # ORCA neutral / anion / cation
GAUSS = [G / "N.out", G / "N+1.out", G / "N-1.out"]


def test_rank_orders_by_energy_and_writes_one_file(tmp_path):
    out = tmp_path / "ranked.xyz"
    result = rank_agent.rank([str(p) for p in GAUSS], output=out)
    assert result.n_input == 3 and result.n_ranked == 3
    assert result.output_xyz == str(out) and out.exists()
    # ranks are contiguous 1..3 in ascending energy; lowest has ΔE = 0.
    ranked = result.ranked
    assert [it.rank for it in ranked] == [1, 2, 3]
    energies = [it.energy_hartree for it in ranked]
    assert energies == sorted(energies)
    assert ranked[0].rel_kcal == pytest.approx(0.0, abs=1e-6)
    assert all(it.engine == "gaussian" for it in ranked)


def test_rank_relative_energy_matches_hartree_gap(tmp_path):
    result = rank_agent.rank([str(p) for p in GAUSS], output=tmp_path / "r.xyz")
    lo = result.ranked[0].energy_hartree
    for it in result.ranked:
        assert it.rel_kcal == pytest.approx((it.energy_hartree - lo) * 627.509, abs=1e-3)


def test_rank_combined_file_blocks_in_energy_order(tmp_path):
    out = tmp_path / "ranked.xyz"
    rank_agent.rank([str(p) for p in GAUSS], output=out)
    comments = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("rank ")]
    assert [c.split()[1] for c in comments] == ["1", "2", "3"]
    assert "kcal/mol" in comments[0] and "Eh" in comments[0]


def test_rank_mixed_engines(tmp_path):
    result = rank_agent.rank(
        [str(G / "N.out"), str(O / "1tial_N_orca.out")], output=tmp_path / "m.xyz"
    )
    assert result.n_ranked == 2
    assert {it.engine for it in result.ranked} == {"gaussian", "orca"}


def test_rank_skips_unparseable_but_ranks_the_rest(tmp_path):
    bad = tmp_path / "garbage.out"
    bad.write_text("not a quantum chemistry output\n", encoding="utf-8")
    result = rank_agent.rank([str(G / "N.out"), str(bad), str(G / "N+1.out")],
                             output=tmp_path / "r.xyz")
    assert result.n_input == 3 and result.n_ranked == 2
    bad_item = next(it for it in result.items if it.source.endswith("garbage.out"))
    assert bad_item.rank is None and bad_item.error
    assert any("garbage.out" in w for w in result.warnings)


def test_rank_no_output_when_output_none():
    result = rank_agent.rank([str(p) for p in GAUSS])
    assert result.output_xyz is None and result.n_ranked == 3   # still ranked in memory


# ── window / top selection ──

def test_rank_default_keeps_all(tmp_path):
    result = rank_agent.rank([str(p) for p in GAUSS], output=tmp_path / "r.xyz")
    assert result.n_selected == result.n_ranked == 3
    assert all(it.selected for it in result.ranked)


def test_rank_top_n_trims_but_keeps_full_ranking(tmp_path):
    out = tmp_path / "r.xyz"
    result = rank_agent.rank([str(p) for p in GAUSS], output=out, top_n=1)
    assert result.n_ranked == 3 and result.n_selected == 1        # all ranked, one written
    assert [it.selected for it in result.ranked] == [True, False, False]
    comments = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("rank ")]
    assert [c.split()[1] for c in comments] == ["1"]              # only rank 1 in the .xyz


def test_rank_window_trims_high_energy_tail(tmp_path):
    full = rank_agent.rank([str(p) for p in GAUSS])               # read the real ΔE first
    dgap = full.ranked[1].rel_kcal
    assert dgap > 0
    result = rank_agent.rank([str(p) for p in GAUSS], output=tmp_path / "r.xyz",
                             window_kcal=dgap / 2)
    assert result.n_selected == 1
    assert result.ranked[0].selected and not result.ranked[1].selected


def test_rank_window_never_empties_the_minimum(tmp_path):
    # an absurdly tight window still keeps rank 1 (ΔE 0) — the .xyz is never empty.
    result = rank_agent.rank([str(p) for p in GAUSS], output=tmp_path / "r.xyz",
                             window_kcal=1e-4)
    assert result.n_selected >= 1 and result.ranked[0].selected


def test_rank_sidecar_records_all_with_selected_flag(tmp_path):
    out = tmp_path / "r.xyz"
    rank_agent.rank([str(p) for p in GAUSS], output=out, top_n=1)
    import json as _json
    data = _json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["n_selected"] == 1 and data["n_ranked"] == 3
    flags = [it["selected"] for it in sorted(data["items"], key=lambda x: x["rank"])]
    assert flags == [True, False, False]                         # full ordering + the cut


def test_cli_rank_top_marks_selection(tmp_path):
    out = tmp_path / "r.xyz"
    res = CliRunner().invoke(main, ["rank", *[str(p) for p in GAUSS], str(out), "--top", "1"])
    assert res.exit_code == 0
    assert "kept 1/3" in res.output
    assert "✓" in res.output and "·" in res.output


def test_cli_rank_default_window_trims_outliers(tmp_path):
    # The CLI default is generous but non-zero (--window 25): the charged N±1 states sit at
    # 54 and 292 kcal/mol here, so only rank 1 reaches the .xyz. (The agent default is still 0.)
    out = tmp_path / "r.xyz"
    res = CliRunner().invoke(main, ["rank", *[str(p) for p in GAUSS], str(out)])
    assert res.exit_code == 0 and "kept" in res.output
    blocks = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("rank ")]
    assert len(blocks) == 1


def test_cli_rank_window_zero_disables_trim(tmp_path):
    out = tmp_path / "r.xyz"
    res = CliRunner().invoke(
        main, ["rank", *[str(p) for p in GAUSS], str(out), "--window", "0", "--top", "0"])
    assert res.exit_code == 0
    blocks = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("rank ")]
    assert len(blocks) == 3          # nothing trimmed when both are off


# ── CLI ──

def test_cli_rank_trailing_xyz_positional(tmp_path):
    out = tmp_path / "ranked.xyz"
    r = CliRunner().invoke(main, ["rank", str(G / "N.out"), str(G / "N+1.out"), str(out)])
    assert r.exit_code == 0 and out.exists()
    assert "Ranked 2/2" in r.output


def test_cli_rank_output_dir(tmp_path):
    r = CliRunner().invoke(main, ["rank", str(G / "N.out"), str(G / "N+1.out"), "-o", str(tmp_path) + "/"])
    assert r.exit_code == 0
    assert (tmp_path / "ranked_by_energy.xyz").exists()


def test_cli_rank_rejects_two_xyz(tmp_path):
    r = CliRunner().invoke(main, ["rank", str(G / "N.out"), "a.xyz", "b.xyz"])
    assert r.exit_code != 0 and "exactly one output" in r.output


def test_cli_rank_rejects_o_and_trailing_xyz(tmp_path):
    r = CliRunner().invoke(main, ["rank", str(G / "N.out"), "a.xyz", "-o", str(tmp_path / "o.xyz")])
    assert r.exit_code != 0 and "not both" in r.output


def test_cli_rank_missing_input(tmp_path):
    r = CliRunner().invoke(main, ["rank", str(tmp_path / "nope.out"), str(tmp_path / "out.xyz")])
    assert r.exit_code != 0 and "Not found" in r.output


# ─────────────────────────────────────────────
# Neg Freq column: imaginary-frequency count
# ─────────────────────────────────────────────

def _orca_out_with_freqs(path: Path, freqs: list[float]) -> Path:
    """Minimal ORCA output carrying an energy, a geometry and a VIBRATIONAL FREQUENCIES block."""
    body = [
        "FINAL SINGLE POINT ENERGY      -76.123456789",
        "",
        "CARTESIAN COORDINATES (ANGSTROEM)",
        "---------------------------------",
        "  O      0.000000    0.000000    0.000000",
        "  H      0.000000    0.000000    0.960000",
        "",
        "-----------------------",
        "VIBRATIONAL FREQUENCIES",
        "-----------------------",
        "",
    ]
    for i, f in enumerate(freqs):
        body.append(f"  {i:>3}: {f:>12.2f} cm**-1")
    body += ["", "NORMAL MODES", ""]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def test_rank_reports_no_freq_as_none_not_zero(tmp_path):
    """A bare Opt has no frequencies → n_imaginary is None ('—'), never 0."""
    result = rank_agent.rank([str(p) for p in GAUSS], output=tmp_path / "r.xyz")
    assert all(it.n_imaginary is None for it in result.ranked)


def test_rank_counts_imaginary_frequencies_orca(tmp_path):
    saddle = _orca_out_with_freqs(tmp_path / "saddle.out", [0.0, 0.0, -180.5, 400.0, 1600.0])
    minimum = _orca_out_with_freqs(tmp_path / "min.out", [0.0, 0.0, 300.0, 400.0, 1600.0])
    result = rank_agent.rank([str(saddle), str(minimum)], output=tmp_path / "r.xyz")
    by_name = {Path(it.source).name: it for it in result.ranked}
    assert by_name["saddle.out"].n_imaginary == 1
    assert by_name["min.out"].n_imaginary == 0          # 0 = frequencies computed, true minimum


def test_rank_gaussian_minimum_reports_zero_imaginary(tmp_path):
    """The Gaussian fixture with frequencies is a true minimum → 0, not None."""
    result = rank_agent.rank([str(FIX / "TiO2_monomer.out")], output=tmp_path / "r.xyz")
    assert result.ranked[0].n_imaginary == 0


def test_rank_cli_shows_neg_freq_column(tmp_path):
    out = tmp_path / "ranked.xyz"
    res = CliRunner().invoke(main, ["rank", *[str(p) for p in GAUSS], str(out)])
    assert res.exit_code == 0
    assert "Neg Freq" in res.output
    assert "—" in res.output                            # no frequencies in these outputs


def test_rank_cli_warns_on_saddle_point(tmp_path):
    saddle = _orca_out_with_freqs(tmp_path / "saddle.out", [0.0, -250.0, 900.0])
    res = CliRunner().invoke(main, ["rank", str(saddle), str(tmp_path / "r.xyz")])
    assert res.exit_code == 0
    assert "saddle point" in res.output


def test_rank_xyz_comment_stays_minimal(tmp_path):
    """The .xyz carries only ΔE + absolute energy + provenance — no frequency metadata."""
    minimum = _orca_out_with_freqs(tmp_path / "min.out", [0.0, 300.0, 1600.0])
    out = tmp_path / "ranked.xyz"
    rank_agent.rank([str(minimum)], output=out)
    text = out.read_text(encoding="utf-8")
    assert "kcal/mol" in text and "E=-76.123457 Eh" in text
    assert "neg_freq" not in text


def test_rank_writes_json_sidecar_next_to_the_xyz(tmp_path):
    """No flag: the JSON record lands beside the .xyz, named after it."""
    out = tmp_path / "ranked.xyz"
    result = rank_agent.rank([str(p) for p in GAUSS], output=out)

    sidecar = tmp_path / "ranked.json"
    assert result.output_json == str(sidecar) and sidecar.exists()
    import json as _json
    data = _json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["n_ranked"] == 3
    assert data["output_xyz"] == str(out)
    assert "n_imaginary" in data["items"][0]      # the screening decision travels in the sidecar


def test_rank_no_sidecar_without_output(tmp_path):
    """Nothing is written when no output was asked for — the sidecar shadows the artifact."""
    result = rank_agent.rank([str(p) for p in GAUSS])
    assert result.output_json is None
    assert not list(tmp_path.iterdir())


def test_rank_cli_announces_the_sidecar(tmp_path):
    out = tmp_path / "ranked.xyz"
    res = CliRunner().invoke(main, ["rank", *[str(p) for p in GAUSS], str(out)])
    assert res.exit_code == 0
    assert (tmp_path / "ranked.json").exists()
    assert "JSON → " in res.output
    assert "Neg Freq" in res.output               # the table still prints


# ── Dedup post-optimization ────────────────────────────────────────
# The potential energy surface funnels distinct starting configurations into the same
# minimum, so a rank over N files can hide far fewer than N structures.

_ORCA_TEMPLATE = """
                       * O   R   C   A *

CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
{coords}

FINAL SINGLE POINT ENERGY      {energy:.8f}
"""


def _fake_out(path, energy, coords):
    """Minimal ORCA output carrying one energy and one geometry."""
    body = "\n".join(f"  {s}  {x:11.6f} {y:11.6f} {z:11.6f}" for s, x, y, z in coords)
    path.write_text(_ORCA_TEMPLATE.format(coords=body, energy=energy))
    return str(path)


_WATER = [("O", 0.0, 0.0, 0.0), ("H", 0.96, 0.0, 0.0), ("H", -0.24, 0.93, 0.0)]


def _nudged(d):
    """Move ONE atom. A rigid translation is invisible to Kabsch, so it would test nothing."""
    return _WATER[:-1] + [(s, x, y, z + d) for s, x, y, z in _WATER[-1:]]


def _three_groups(tmp_path):
    """Six outputs that are really three minima: A(3), B(2), C(1)."""
    return [
        _fake_out(tmp_path / "a1.out", -76.400000, _WATER),          # A representative
        _fake_out(tmp_path / "a2.out", -76.399999, _nudged(0.10)),    # A: noise-level RMSD
        _fake_out(tmp_path / "a3.out", -76.399998, _nudged(0.30)),    # A: still within 0.20 Å
        _fake_out(tmp_path / "b1.out", -76.390000, _WATER),          # B: same geometry, far in E
        _fake_out(tmp_path / "b2.out", -76.389999, _nudged(0.05)),    # B representative's twin
        _fake_out(tmp_path / "c1.out", -76.380000, _nudged(2.50)),    # C: alone, by geometry
    ]


def test_rank_dedup_groups_outputs_that_share_a_minimum(tmp_path):
    out = tmp_path / "ranked.xyz"
    result = rank_agent.rank(_three_groups(tmp_path), output=out)
    assert result.n_ranked == 6
    assert result.n_unique == 3                      # six files, three structures
    assert out.read_text(encoding="utf-8").count("FINAL") == 0
    assert len([b for b in out.read_text(encoding="utf-8").split("\n") if b.strip() == "3"]) == 3


def test_rank_dedup_representative_is_the_lowest_energy_of_its_group(tmp_path):
    result = rank_agent.rank(_three_groups(tmp_path))
    reps = [it for it in result.ranked if it.duplicate_of is None]
    assert [Path(it.source).name for it in reps] == ["a1.out", "b1.out", "c1.out"]
    dups = {Path(it.source).name: Path(it.duplicate_of).name
            for it in result.ranked if it.duplicate_of}
    # every duplicate points at a representative, never at another duplicate
    assert dups == {"a2.out": "a1.out", "a3.out": "a1.out", "b2.out": "b1.out"}


def test_rank_dedup_zero_disables_and_restores_the_old_behaviour(tmp_path):
    result = rank_agent.rank(_three_groups(tmp_path), dedup_rmsd=0.0)
    assert result.n_unique == result.n_ranked == 6
    assert all(it.duplicate_of is None for it in result.ranked)


def test_rank_top_counts_unique_minima_not_files(tmp_path):
    # Without the dedup, --top 2 would return a1 and a2 — the same structure twice.
    result = rank_agent.rank(_three_groups(tmp_path), top_n=2)
    kept = [Path(it.source).name for it in result.ranked if it.selected]
    assert kept == ["a1.out", "b1.out"]


def test_rank_dedup_never_pairs_different_compositions(tmp_path):
    # Same atom count, same energy, same coordinates — but not the same molecule.
    sources = [
        _fake_out(tmp_path / "w.out", -76.400000, _WATER),
        _fake_out(tmp_path / "x.out", -76.400000,
                  [("N", 0.0, 0.0, 0.0), ("H", 0.96, 0.0, 0.0), ("H", -0.24, 0.93, 0.0)]),
    ]
    result = rank_agent.rank(sources)
    assert result.n_unique == 2


def test_rank_dedup_keeps_distinct_minima_that_happen_to_be_isoenergetic(tmp_path):
    sources = [
        _fake_out(tmp_path / "p.out", -76.400000, _WATER),
        _fake_out(tmp_path / "q.out", -76.400000, _nudged(2.50)),    # ΔE ≈ 0, RMSD large
    ]
    result = rank_agent.rank(sources)
    assert result.n_unique == 2


def test_rank_sidecar_keeps_the_duplicates_with_their_representative(tmp_path):
    out = tmp_path / "ranked.xyz"
    rank_agent.rank(_three_groups(tmp_path), output=out)
    data = json.loads((tmp_path / "ranked.json").read_text(encoding="utf-8"))
    assert data["n_unique"] == 3 and data["n_ranked"] == 6
    assert len(data["items"]) == 6                   # nothing is dropped from the trail
    assert sum(1 for it in data["items"] if it["duplicate_of"]) == 3


def test_cli_rank_reports_the_unique_count(tmp_path):
    res = CliRunner().invoke(main, ["rank", *_three_groups(tmp_path),
                                    "-o", str(tmp_path / "r.xyz")])
    assert res.exit_code == 0, res.output
    assert "3 distinct minima" in res.output
    assert "≡" in res.output


def test_rank_warns_when_compositions_are_mixed(tmp_path):
    sources = [
        _fake_out(tmp_path / "w.out", -76.400000, _WATER),
        _fake_out(tmp_path / "a.out", -56.500000,
                  [("N", 0.0, 0.0, 0.0), ("H", 1.01, 0.0, 0.0), ("H", -0.34, 0.95, 0.0)]),
    ]
    result = rank_agent.rank(sources)
    assert any("not comparable" in w for w in result.warnings)


def test_rank_warns_when_the_atom_ordering_differs(tmp_path):
    # Same molecule, atoms written in a different order: kabsch_rmsd cannot pair them.
    reordered = [_WATER[1], _WATER[0], _WATER[2]]
    sources = [
        _fake_out(tmp_path / "u.out", -76.400000, _WATER),
        _fake_out(tmp_path / "v.out", -76.400001, reordered),
    ]
    result = rank_agent.rank(sources)
    assert result.n_unique == 2                      # blind, but conservatively so
    assert any("atom ordering" in w for w in result.warnings)


# ── Geometric convergence: an Opt that gave up is not a minimum ────────────────────
# ORCA exhausts its cycle budget, writes a final energy and terminates "normally".
# Without this flag that geometry enters the ranking as if it were a stationary point.

_CONV = "\n          ***        THE OPTIMIZATION HAS CONVERGED     ***\n"
_GAVE_UP = ("\n     The optimization did not converge but reached the maximum number of\n"
            "     optimization cycles.\n")


def _opt_out(path, energy, coords, tail):
    """ORCA output that contains an optimization, converged or not."""
    body = "\n".join(f"  {s}  {x:11.6f} {y:11.6f} {z:11.6f}" for s, x, y, z in coords)
    path.write_text("\nGEOMETRY OPTIMIZATION CYCLE   1\n" + tail
                    + _ORCA_TEMPLATE.format(coords=body, energy=energy))
    return str(path)


def test_rank_flags_an_optimization_that_ran_out_of_cycles(tmp_path):
    files = [
        _opt_out(tmp_path / "good.out", -76.40, _WATER, _CONV),
        _opt_out(tmp_path / "bad.out", -76.39, _nudged(0.4), _GAVE_UP),
    ]
    result = rank_agent.rank(files, output=tmp_path / "r.xyz")
    by_name = {Path(it.source).name: it for it in result.ranked}
    assert by_name["good.out"].geom_converged is True
    assert by_name["bad.out"].geom_converged is False
    assert result.n_unconverged == 1
    # ranked, not dropped: it is still a usable starting guess for the next stage.
    assert result.n_ranked == 2


def test_rank_warns_but_still_ranks_unconverged_outputs(tmp_path):
    files = [_opt_out(tmp_path / "a.out", -76.40, _WATER, _GAVE_UP)]
    result = rank_agent.rank(files, output=tmp_path / "r.xyz")
    assert any("did not converge geometrically" in w for w in result.warnings)
    assert result.ranked[0].rank == 1


def test_rank_single_point_has_no_convergence_verdict(tmp_path):
    """A single point never had an optimization, so it is neither converged nor failed."""
    result = rank_agent.rank([_fake_out(tmp_path / "sp.out", -76.40, _WATER)],
                             output=tmp_path / "r.xyz")
    assert result.ranked[0].geom_converged is None
    assert result.n_unconverged == 0


def test_rank_unconverged_verdict_survives_the_two_line_orca_message(tmp_path):
    """The ORCA failure text wraps mid-sentence; a naive one-line match misses it."""
    assert "maximum number of optimization cycles" not in _GAVE_UP   # wrapped, as ORCA writes it
    files = [_opt_out(tmp_path / "a.out", -76.40, _WATER, _GAVE_UP)]
    assert rank_agent.rank(files).ranked[0].geom_converged is False


def test_rank_cli_prints_the_opt_column_and_its_legend(tmp_path):
    files = [
        _opt_out(tmp_path / "good.out", -76.40, _WATER, _CONV),
        _opt_out(tmp_path / "bad.out", -76.39, _nudged(0.4), _GAVE_UP),
    ]
    res = CliRunner().invoke(main, ["rank", *files, str(tmp_path / "r.xyz")])
    assert res.exit_code == 0
    assert "Opt" in res.output and "NO-CONV" in res.output and "ok" in res.output
    assert "never reached a stationary point" in res.output


# ── The window is measured within each skeleton class ──────────────────
#
# A three-atom cluster Al-O-Al plus one probe atom. Linear, the two Al are too far to bond;
# bent, they are close enough, and the skeleton changes. That is the whole difference: the
# composition, the atom order and the element list are identical, so nothing but the
# connectivity can tell the two classes apart.

_REF_CLUSTER = [("Al", 0.0, 0.0, 0.0), ("O", 1.8, 0.0, 0.0), ("Al", 3.6, 0.0, 0.0)]
_ISO_CLUSTER = [("Al", 0.0, 0.0, 0.0), ("O", 1.8, 0.0, 0.0), ("Al", 1.8, 1.8, 0.0)]
_KCAL = 627.5094740631


def _cluster_ref_xyz(tmp_path) -> str:
    p = tmp_path / "cluster_ref.xyz"
    body = "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in _REF_CLUSTER)
    p.write_text(f"{len(_REF_CLUSTER)}\nbare cluster, reference skeleton\n{body}\n")
    return str(p)


def _adduct(tmp_path, name, energy, cluster, probe_z):
    """Cluster first, probe appended — the convention the adsorption agents share."""
    return _fake_out(tmp_path / name, energy, cluster + [("H", 8.0, 0.0, probe_z)])


def _two_classes(tmp_path):
    """Four adducts: the isomer class is 50 kcal/mol below the reference class."""
    iso0 = -100.0
    ref0 = iso0 + 50.0 / _KCAL
    return [
        _adduct(tmp_path, "iso_lo.out", iso0, _ISO_CLUSTER, 0.0),
        _adduct(tmp_path, "iso_hi.out", iso0 + 2.0 / _KCAL, _ISO_CLUSTER, 1.0),
        _adduct(tmp_path, "ref_lo.out", ref0, _REF_CLUSTER, 0.0),
        _adduct(tmp_path, "ref_hi.out", ref0 + 2.0 / _KCAL, _REF_CLUSTER, 1.0),
    ]


def _selected(result):
    return {Path(it.source).name for it in result.ranked if it.selected}


def test_rank_measures_the_window_within_each_skeleton_class(tmp_path):
    srcs = _two_classes(tmp_path)
    # Without a reference the isomer takes the zero and a 10 kcal/mol window buries the
    # reference class 50 kcal/mol up — this is the defect.
    blind = rank_agent.rank(srcs, window_kcal=10.0, output=tmp_path / "blind.xyz")
    assert _selected(blind) == {"iso_lo.out", "iso_hi.out"}

    aware = rank_agent.rank(srcs, window_kcal=10.0, cluster_ref=_cluster_ref_xyz(tmp_path),
                            output=tmp_path / "aware.xyz")
    assert _selected(aware) == {"iso_lo.out", "iso_hi.out", "ref_lo.out", "ref_hi.out"}


def test_rank_relative_energy_is_measured_against_its_own_class(tmp_path):
    result = rank_agent.rank(_two_classes(tmp_path), window_kcal=0.0,
                             cluster_ref=_cluster_ref_xyz(tmp_path))
    by_name = {Path(it.source).name: it for it in result.ranked}
    # each class has its own zero, and the global rank is preserved for traceability
    assert by_name["iso_lo.out"].rel_kcal == pytest.approx(0.0, abs=1e-3)
    assert by_name["ref_lo.out"].rel_kcal == pytest.approx(0.0, abs=1e-3)
    assert by_name["ref_hi.out"].rel_kcal == pytest.approx(2.0, abs=1e-2)
    assert [by_name[n].rank for n in ("iso_lo.out", "iso_hi.out", "ref_lo.out", "ref_hi.out")] \
        == [1, 2, 3, 4]


def test_rank_top_n_applies_within_each_class(tmp_path):
    """A global cap would let the first class eat the budget — the defect, not a fix."""
    result = rank_agent.rank(_two_classes(tmp_path), window_kcal=0.0, top_n=1,
                             cluster_ref=_cluster_ref_xyz(tmp_path))
    assert _selected(result) == {"iso_lo.out", "ref_lo.out"}


def test_rank_without_a_cluster_reference_is_unchanged(tmp_path):
    """Regression: no reference, no behaviour change."""
    srcs = _two_classes(tmp_path)
    before = rank_agent.rank(srcs, window_kcal=10.0, output=tmp_path / "a.xyz")
    assert all(it.cluster_skeleton is None for it in before.ranked)
    assert before.n_unique == 4 and before.n_selected == 2


def test_the_skeleton_class_reaches_the_json_and_the_xyz_comment(tmp_path):
    out = tmp_path / "ranked.xyz"
    result = rank_agent.rank(_two_classes(tmp_path), window_kcal=0.0,
                             cluster_ref=_cluster_ref_xyz(tmp_path), output=out)
    classes = {Path(it.source).name: it.cluster_skeleton for it in result.ranked}
    assert classes["ref_lo.out"] == "reference"
    assert classes["iso_lo.out"] == classes["iso_hi.out"] == "isomer-1"
    payload = json.loads(Path(result.output_json).read_text(encoding="utf-8"))
    assert {it["cluster_skeleton"] for it in payload["items"]} == {"reference", "isomer-1"}
    comments = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("rank ")]
    assert all("[reference]" in c or "[isomer-1]" in c for c in comments)


def test_rank_warns_on_a_reference_with_a_different_element_order(tmp_path):
    bad = tmp_path / "bad_order.xyz"
    bad.write_text("3\nprobe first, cluster after\nH 0.0 0.0 0.0\nAl 1.8 0.0 0.0\nO 3.6 0.0 0.0\n")
    result = rank_agent.rank(_two_classes(tmp_path), window_kcal=10.0, cluster_ref=str(bad))
    assert any("element order" in w for w in result.warnings)
    assert _selected(result) == {"iso_lo.out", "iso_hi.out"}     # falls back to the global cut


def test_rank_warns_when_the_cluster_reference_cannot_be_read(tmp_path):
    missing = tmp_path / "nope.xyz"
    result = rank_agent.rank(_two_classes(tmp_path), window_kcal=10.0, cluster_ref=str(missing))
    assert any("cluster reference" in w for w in result.warnings)
    assert _selected(result) == {"iso_lo.out", "iso_hi.out"}


def test_cli_rank_accepts_cluster(tmp_path):
    srcs = _two_classes(tmp_path)
    out = tmp_path / "cli.xyz"
    res = CliRunner().invoke(main, ["rank", *srcs, "--window", "10",
                                    "--cluster", _cluster_ref_xyz(tmp_path), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_text(encoding="utf-8").count("[reference]") == 2


def test_rank_cli_fails_when_nothing_could_be_ranked(tmp_path):
    bad = tmp_path / "broken.out"
    bad.write_text("not an engine output\n")
    res = CliRunner().invoke(main, ["rank", str(bad), "--no-db"])
    assert res.exit_code != 0 and "could be ranked" in res.output


def test_an_unreadable_output_is_not_kept(tmp_path):
    broken = tmp_path / "broken.out"
    broken.write_text("not an engine output\n")
    result = rank_agent.rank([str(p) for p in GAUSS] + [str(broken)], output=tmp_path / "r.xyz")
    item = next(i for i in result.items if i.source.endswith("broken.out"))
    assert item.error and not item.selected
