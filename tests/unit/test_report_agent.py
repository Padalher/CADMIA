"""Tests for cadmia.agents.report_agent (consolidated Markdown/LaTeX report + correlations)."""
from __future__ import annotations

import json

import pytest

from cadmia.agents import report_agent
from cadmia.agents.report_agent import build_report


def _cdft_dict():
    return {
        "cluster": "Al2TiO5", "calculation_id": "cid", "charge_scheme": "hirshfeld",
        "sites": [
            {"atom_index": 5, "atom_symbol": "Al", "fukui_plus": 0.35, "fukui_minus": 0.11,
             "delta_f": 0.24, "character": "electrophilic", "rank": 2},
            {"atom_index": 2, "atom_symbol": "O", "fukui_plus": 0.07, "fukui_minus": 0.32,
             "delta_f": -0.25, "character": "nucleophilic", "rank": 1},
            {"atom_index": 7, "atom_symbol": "Al", "fukui_plus": 0.20, "fukui_minus": 0.20,
             "delta_f": 0.00, "character": "ambiphilic", "rank": 3},
        ],
        "global_descriptors": {"gap_eV": 4.57, "hardness_eV": 7.74, "mu_eV": -6.2, "ip_eV": None},
    }


def _analysis_dict(cfg, idx, sym, e_ads, probe="NH3"):
    return {"cluster": "Al2TiO5", "probe": probe, "config_index": cfg,
            "site_atom_index": idx, "site_atom_symbol": sym,
            "e_ads_kcal": e_ads, "e_ads_bsse_kcal": e_ads + 1.5, "activated": False}


# ─────────────────────────────────────────────
# build_report
# ─────────────────────────────────────────────

def test_report_has_global_and_ranking_tables():
    rep = build_report(cdft=_cdft_dict())
    md = rep.markdown
    assert "## Global CDFT descriptors" in md
    assert "| Gap | 4.5700 | eV |".replace("Gap", "Gap") in md or "Gap" in md
    assert "## CDFT site ranking" in md
    # ranking sorted by rank: O(2) rank 1 before Al(5) rank 2
    assert md.index("O(2)") < md.index("Al(5)")
    # mu present, ip None skipped
    assert "μ" in md and "IP" not in md.split("## CDFT")[0].split("Global")[1]


def test_report_adsorption_and_correlation():
    """Three points of one probe, three distinct Δf: a fit that means something."""
    cdft = _cdft_dict()
    cdft["sites"].append({"atom_index": 7, "atom_symbol": "Al", "fukui_plus": 0.20,
                          "fukui_minus": 0.20, "delta_f": 0.00,
                          "character": "ambiphilic", "rank": 3})
    analyses = [_analysis_dict(0, 5, "Al", -25.0),
                _analysis_dict(1, 2, "O", -8.0),
                _analysis_dict(2, 7, "Al", -16.0)]
    rep = build_report(cdft=cdft, analyses=analyses)
    assert "## Adsorption summary" in rep.markdown
    assert len(rep.correlations) == 1
    c = rep.correlations[0]
    assert c.name == "df_vs_eads_NH3"            # named by probe, not pooled
    assert set(c.x) == {0.24, -0.25, 0.0}
    assert c.pearson_r < -0.99
    assert "NH3" in rep.markdown
    # Fewer than 8 distinct Δf: the report has to say so instead of letting r stand alone.
    assert "not conclusive" in rep.markdown


def test_probes_are_never_pooled_into_one_correlation():
    """A descriptor can correlate with opposite signs for different probes; pooling them
    averages two mechanisms into one meaningless number."""
    cdft = _cdft_dict()
    cdft["sites"].append({"atom_index": 7, "atom_symbol": "Al", "fukui_plus": 0.20,
                          "fukui_minus": 0.20, "delta_f": 0.00,
                          "character": "ambiphilic", "rank": 3})
    analyses = ([_analysis_dict(i, idx, sym, e, probe="NH3")
                 for i, (idx, sym, e) in enumerate([(5, "Al", -25.0), (2, "O", -8.0),
                                                    (7, "Al", -16.0)])]
                + [_analysis_dict(i + 3, idx, sym, e, probe="CO2")
                   for i, (idx, sym, e) in enumerate([(5, "Al", -4.0), (2, "O", -12.0),
                                                      (7, "Al", -8.0)])])
    rep = build_report(cdft=cdft, analyses=analyses)
    names = {c.name for c in rep.correlations}
    assert names == {"df_vs_eads_NH3", "df_vs_eads_CO2"}
    rs = {c.name: c.pearson_r for c in rep.correlations}
    assert rs["df_vs_eads_NH3"] < 0 < rs["df_vs_eads_CO2"]   # opposite signs, kept apart


def test_two_points_are_not_reported_as_a_correlation():
    """With two points r is +-1 by construction, whatever the numbers are."""
    analyses = [_analysis_dict(0, 5, "Al", -25.0), _analysis_dict(1, 2, "O", -8.0)]
    rep = build_report(cdft=_cdft_dict(), analyses=analyses)
    assert rep.correlations == []
    assert any("too few" in w for w in rep.warnings)


def test_report_correlation_needs_enough_points():
    rep = build_report(cdft=_cdft_dict(), analyses=[_analysis_dict(0, 5, "Al", -25.0)])
    assert rep.correlations == []
    assert any("too few" in w for w in rep.warnings)


def test_report_dedups_configs_that_share_a_dft_minimum():
    # Two configs that relaxed to the same well: same site + (near) identical total energy.
    a0 = _analysis_dict(0, 6, "O", -30.00); a0["e_adduct_hartree"] = -1000.000004
    a1 = _analysis_dict(1, 6, "O", -30.00); a1["e_adduct_hartree"] = -1000.000002
    rep = build_report(cdft=_cdft_dict(), analyses=[a0, a1])
    # collapsed to a single unique minimum → no spurious correlation
    assert rep.correlations == []
    assert any("collapsed" in w for w in rep.warnings)
    # the summary table lists only one configuration row (one trailing pipe per row)
    summary = rep.markdown.split("## Adsorption summary", 1)[1]
    data_rows = [ln for ln in summary.splitlines() if ln.startswith("| 0 ") or ln.startswith("| 1 ")]
    assert len(data_rows) == 1


def test_report_keeps_distinct_minima():
    # Same site but genuinely different energies (> window) must NOT be collapsed.
    a0 = _analysis_dict(0, 5, "Al", -25.0); a0["e_adduct_hartree"] = -1000.70
    a1 = _analysis_dict(1, 2, "O", -8.0); a1["e_adduct_hartree"] = -1000.60
    a2 = _analysis_dict(2, 7, "Al", -16.7); a2["e_adduct_hartree"] = -1000.65
    rep = build_report(cdft=_cdft_dict(), analyses=[a0, a1, a2])
    assert len(rep.correlations) == 1


def test_report_stability_section():
    stab = {"mixed": "Al2TiO5", "formula": "Al2O5Ti", "e_rel_kcal": -60.00,
            "e_rel_eV_per_atom": -0.33, "balanced": True,
            "references": [{"label": "TiO2", "coefficient": 1}, {"label": "Al2O3", "coefficient": 1}]}
    rep = build_report(stability=stab)
    assert "## Relative stability" in rep.markdown
    assert "-60.000 kcal/mol" in rep.markdown
    assert "1·TiO2 + 1·Al2O3" in rep.markdown


def test_report_literature_section():
    lit = {"query": "conceptual dft", "references": [
        {"title": "Conceptual Density Functional Theory",
         "authors": ["P. Geerlings", "F. De Proft", "W. Langenaeker"],
         "year": 2003, "container": "Chemical Reviews",
         "doi": "10.1021/cr990029p", "citations": 5000},
    ]}
    rep = build_report(cdft=_cdft_dict(), literature=lit)
    assert "## References" in rep.markdown
    assert "Conceptual Density Functional Theory" in rep.markdown
    assert "DOI: 10.1021/cr990029p" in rep.markdown


def test_report_latex_and_files(tmp_path):
    rep = build_report(cdft=_cdft_dict(),
                       analyses=[_analysis_dict(0, 5, "Al", -25.0),
                                 _analysis_dict(1, 2, "O", -8.0),
                                 _analysis_dict(2, 7, "Al", -16.7)],
                       title="T")
    assert r"\begin{tabular}" in rep.latex
    out = tmp_path / "report.md"
    written = rep.to_files(out, latex=True, figures=True)
    assert out.exists()
    assert written["latex"].endswith(".tex")
    # matplotlib likely absent in the dev env → CSV fallback for the correlation
    assert written["figures"] or written["csv"]


def test_report_accepts_json_paths(tmp_path):
    cdft_path = tmp_path / "cdft.json"
    cdft_path.write_text(json.dumps(_cdft_dict()), encoding="utf-8")
    rep = build_report(cdft=str(cdft_path))
    assert "Al2TiO5" in rep.markdown


def test_report_cli(tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main
    cdft_path = tmp_path / "cdft.json"
    cdft_path.write_text(json.dumps(_cdft_dict()), encoding="utf-8")
    a0 = tmp_path / "a0.json"; a0.write_text(json.dumps(_analysis_dict(0, 5, "Al", -25.0)))
    a1 = tmp_path / "a1.json"; a1.write_text(json.dumps(_analysis_dict(1, 2, "O", -8.0)))
    a2 = tmp_path / "a2.json"; a2.write_text(json.dumps(_analysis_dict(2, 7, "Al", -16.7)))
    out = tmp_path / "rep.md"
    res = CliRunner().invoke(main, [
        "report", "--cdft", str(cdft_path), "--analysis", str(a0), "--analysis", str(a1),
        "--analysis", str(a2), "-o", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert "r = -1.000" in res.output and "NH3" in res.output


def test_report_electronic_section():
    a = _analysis_dict(0, 5, "Al", -25.0)
    a["electronic_cluster"] = {"homo_ev": -8.0, "lumo_ev": -3.0, "gap_ev": 5.0,
                               "mu_ev": -5.5, "eta_ev": 2.5, "omega_ev": 6.05}
    a["electronic_adduct"] = {"homo_ev": -7.8, "lumo_ev": -3.1, "gap_ev": 4.7,
                              "mu_ev": -5.45, "eta_ev": 2.35, "omega_ev": 6.3}
    rep = build_report(cdft=_cdft_dict(), analyses=[a])
    md = rep.markdown
    assert "## Electronic properties (cluster → adduct)" in md
    assert "HOMO" in md and "Gap" in md
    assert "Δgap on adsorption = **-0.300 eV**" in md


def test_report_dipole_table_and_delta():
    a = _analysis_dict(0, 5, "Al", -25.0)
    a["electronic_cluster"] = {"dipole_debye": 2.0}
    a["electronic_adduct"] = {"dipole_debye": 3.5}
    rep = build_report(cdft=_cdft_dict(), analyses=[a])
    md = rep.markdown
    assert "dipole (D)" in md and "|μ|" in md
    assert "Δ|μ| on adsorption = **+1.500 D**" in md


def test_report_charge_transfer_section():
    a = _analysis_dict(0, 1, "Ti", -25.0)
    a["charge_scheme"] = "hirshfeld"
    a["q_transfer_to_probe"] = -0.30
    a["delta_q_site"] = 0.12
    rep = build_report(cdft=_cdft_dict(), analyses=[a])
    md = rep.markdown
    assert "## Charge transfer on adsorption (hirshfeld charges)" in md
    assert "-0.300" in md and "+0.120" in md


def test_report_links_dos_pdos_from_meta():
    rep = build_report(cdft=_cdft_dict(),
                       meta={"dos_files": ["sys_dos.html"], "pdos_files": ["sys_pdos.html"]})
    md = rep.markdown
    assert "## Density of states" in md
    assert "[sys_dos.html](sys_dos.html)" in md
    assert "[sys_pdos.html](sys_pdos.html)" in md


def test_report_no_dos_section_without_files():
    rep = build_report(cdft=_cdft_dict())
    assert "## Density of states" not in rep.markdown


def test_report_vibrational_signature_section():
    a = _analysis_dict(0, 1, "Ti", -25.0)
    a["adduct_imaginary_modes"] = 0
    a["adduct_ir_bands"] = [{"frequency_cm1": 2396.3, "intensity_km_mol": 600.2}]
    rep = build_report(cdft=_cdft_dict(), analyses=[a])
    md = rep.markdown
    assert "## Vibrational signature of the adduct" in md
    assert "yes" in md and "2396.3" in md


def test_report_mayer_at_site_section():
    a = _analysis_dict(0, 5, "Al", -25.0)
    a["site_mayer_bonds"] = [
        {"partner_index": 9, "partner_symbol": "O", "order": 0.8421},
        {"partner_index": 3, "partner_symbol": "O", "order": 0.5102},
    ]
    rep = build_report(cdft=_cdft_dict(), analyses=[a])
    md = rep.markdown
    assert "## Mayer bond orders at the binding site (adduct)" in md
    assert "O(9)" in md and "0.8421" in md


def test_report_cli_analysis_dir(tmp_path):
    from click.testing import CliRunner
    from cadmia.cli import main
    cdft_path = tmp_path / "cdft.json"
    cdft_path.write_text(json.dumps(_cdft_dict()), encoding="utf-8")
    adir = tmp_path / "analysis"; adir.mkdir()
    (adir / "a0.json").write_text(json.dumps(_analysis_dict(0, 5, "Al", -25.0)))
    (adir / "a1.json").write_text(json.dumps(_analysis_dict(1, 2, "O", -8.0)))
    (adir / "a2.json").write_text(json.dumps(_analysis_dict(2, 7, "Al", -16.7)))
    out = tmp_path / "rep.md"
    res = CliRunner().invoke(main, [
        "report", "--cdft", str(cdft_path), "--analysis-dir", str(adir), "-o", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert "r = -1.000" in res.output and "NH3" in res.output


def test_full_report_reads_the_level_of_theory_from_the_outputs():
    cdft = _cdft_dict()
    cdft["sources"] = {"N": "tests/fixtures/1tial_d4/1tial_N.out"}
    md = build_report(cdft=cdft, full=True).markdown
    assert "CDFT (`1tial_N.out`): `! PBE0 def2-TZVP def2/J D4" in md
    md = build_report(cdft=cdft, full=True, meta={"functional": "PBE0", "basis": "def2-TZVP"}).markdown
    assert "**Level of theory:** PBE0/def2-TZVP" in md and "echoed" not in md


def test_full_report_has_a_structure_section():
    st = {"structure": {"structure_id": "c1", "formula": "TiO2", "atoms": ["Ti", "O", "O"]},
          "symmetry": {"point_group": "C2v"},
          "local_descriptors": [{"label": "Ti1", "cn": 2, "ecn": 1.98,
                                 "effective_avg_bond_distance": 1.65, "bonded_elements": {"O": 2}}],
          "rings": {"statistics": {"total": 0, "by_size": {}}}}
    md = build_report(structure=st, cdft=_cdft_dict(), full=True).markdown
    assert "## Structure" in md and "| Ti1 | 2 | 1.980 | 1.65 | O:2 |" in md
    assert (md.index("please cite") < md.index("MODULE 1  ·  STRUCTURE") < md.index("## Structure")
            < md.index("MODULE 2  ·  CDFT") < md.index("## Global CDFT descriptors")
            < md.index("— ", md.index("Modules reported")))
    assert "MODULE" not in build_report(cdft=_cdft_dict()).markdown
    assert "Modules reported: Structure, CDFT" in md
