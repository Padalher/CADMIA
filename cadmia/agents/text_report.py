"""Full text reports of the adsorption and stability stages, built from their JSON dicts.

They follow the layout of the CDFT text report (a title, then sections under a rule, one value
per line), so the stages read as one document once `cadmia report` assembles them. Every field of
the result appears; a field the run did not produce is left out rather than shown as zero.
"""
from __future__ import annotations

import textwrap
from typing import Iterable, List, Optional

from cadmia.core.units import HARTREE_TO_EV, HARTREE_TO_KCAL
from cadmia.io.results import shown_path

KCAL_TO_EV = HARTREE_TO_EV / HARTREE_TO_KCAL
_WIDTH = 72


def _hdr(title: str) -> str:
    return f"\n {title}\n {'=' * (_WIDTH - 1)}"


def _line(label: str, value: str) -> str:
    return f" {label:<38}{value}"


def _kcal(label: str, kcal: Optional[float]) -> Optional[str]:
    if kcal is None:
        return None
    return _line(label, f"{kcal:>12.3f} kcal/mol,{kcal * KCAL_TO_EV:>10.4f} eV")


def _hartree(label: str, e: Optional[float]) -> Optional[str]:
    return None if e is None else _line(label, f"{e:>16.6f} Hartree")


def _table(headers: List[str], rows: Iterable[List[str]], widths: List[int]) -> List[str]:
    def fmt(cells):
        return " " + "".join(f"{c:<{w}}" if i == 0 else f"{c:>{w}}"
                             for i, (c, w) in enumerate(zip(cells, widths)))
    rows = list(rows)
    return [fmt(headers), " " + "-" * (sum(widths))] + [fmt(r) for r in rows]


def _warnings(warnings: List[str]) -> List[str]:
    if not warnings:
        return []
    lines = [_hdr("Warnings")]
    for w in warnings:
        lines += textwrap.wrap(w, _WIDTH - 4, initial_indent=" ! ", subsequent_indent="   ")
    return lines


def _num(v: Optional[float], spec: str) -> str:
    return "—" if v is None else format(v, spec)


def analysis_report_text(d: dict) -> str:
    from cadmia import __version__
    site = (f"{d.get('site_atom_symbol')}({d.get('site_atom_index')})"
            if d.get("site_atom_index") is not None else "—")
    out: List[Optional[str]] = [
        f" CADMIA Adsorption Report — CADMIA {__version__}",
        f" System  : {d.get('cluster')} + {d.get('probe')}",
        f" Adduct  : {shown_path(d.get('adduct_path')) or '(not specified)'}",
        f" Engine  : {d.get('engine') or '—'}    Binding site: {site}"
        + (f"    Configuration: {d['config_index']}" if d.get("config_index") is not None else ""),
    ]

    out.append(_hdr("Adsorption energetics"))
    out += [_hartree("E(adduct):", d.get("e_adduct_hartree")),
            _hartree("E(cluster):", d.get("e_cluster_hartree")),
            _hartree("E(probe):", d.get("e_probe_hartree")),
            _kcal("E_ads = E(adduct)-E(cluster)-E(probe):", d.get("e_ads_kcal")),
            _kcal("BSSE correction:", d.get("bsse_kcal")),
            _kcal("E_ads (BSSE-corrected):", d.get("e_ads_bsse_kcal")),
            _kcal("E_def(cluster):", d.get("e_def_cluster_kcal")),
            _kcal("E_def(probe):", d.get("e_def_probe_kcal")),
            _kcal("E_int = E_ads-E_def(cluster)-E_def(probe):", d.get("e_int_kcal"))]
    if d.get("bsse_kcal") is None:
        out.append(" (no BSSE correction)")

    if d.get("delta_g_ads_kcal") is not None or d.get("delta_h_ads_kcal") is not None:
        t = d.get("temperature_kelvin")
        out.append(_hdr(f"Thermochemistry of adsorption (T = {t:.2f} K)" if t else
                        "Thermochemistry of adsorption"))
        out += [_kcal("ΔH_ads:", d.get("delta_h_ads_kcal")),
                _kcal("ΔG_ads:", d.get("delta_g_ads_kcal")),
                _kcal("ΔG_ads (quasi-RRHO entropy):", d.get("delta_gq_ads_kcal")),
                _kcal("-TΔS_ads:", d.get("minus_t_delta_s_kcal"))]

    before, after = d.get("electronic_cluster") or {}, d.get("electronic_adduct") or {}
    if before or after:
        out.append(_hdr("Electronic properties: cluster → adduct (Koopmans)"))
        names = [("homo_ev", "HOMO (eV)"), ("lumo_ev", "LUMO (eV)"), ("gap_ev", "Gap (eV)"),
                 ("mu_ev", "μ (eV)"), ("eta_ev", "η (eV)"), ("omega_ev", "ω (eV)"),
                 ("omega_minus_ev", "ω- (eV)"), ("omega_plus_ev", "ω+ (eV)"),
                 ("dipole_debye", "|dipole| (Debye)")]
        rows = []
        for key, label in names:
            b, a = before.get(key), after.get(key)
            if b is None and a is None:
                continue
            delta = a - b if a is not None and b is not None else None
            rows.append([label, _num(b, ".4f"), _num(a, ".4f"), _num(delta, "+.4f")])
        out += _table(["quantity", "cluster", "adduct", "Δ"], rows, [22, 14, 14, 14])

    ct = d.get("charge_transfer")
    if ct:
        out.append(_hdr("Charge transfer predicted by conceptual DFT (cluster ↔ probe)"))
        out += [_line("ΔN (electrons):", f"{ct.get('delta_n'):>12.4f}"),
                _line("ΔE_stab:", f"{ct.get('e_stab_ev'):>12.4f} eV"),
                _line("Direction:", f"{ct.get('donor')} → {ct.get('acceptor')}"),
                _line("μ, η of the cluster:",
                      f"{ct.get('mu_cluster_ev'):.4f}, {ct.get('eta_cluster_ev'):.4f} eV"),
                _line("μ, η of the probe:",
                      f"{ct.get('mu_probe_ev'):.4f}, {ct.get('eta_probe_ev'):.4f} eV"),
                " (Parr–Pearson, one parabola, from the Koopmans μ and η above)"]

    if d.get("q_transfer_to_probe") is not None or d.get("delta_q_site") is not None:
        out.append(_hdr(f"Charge transfer on adsorption ({d.get('charge_scheme') or '—'} charges)"))
        out += [_line("Net charge on the probe in the adduct:", _num(d.get("q_transfer_to_probe"), ">12.4f")),
                _line("Δq(binding site):", _num(d.get("delta_q_site"), ">12.4f")),
                " (negative probe charge = electrons donated by the cluster)"]

    if d.get("site_cn_cluster") is not None or d.get("site_cn_adduct") is not None:
        out.append(_hdr(f"Binding-site coordination: cluster → adduct ({site})"))
        out += _table(["", "cluster", "adduct", "Δ"], [
            ["CN", _num(d.get("site_cn_cluster"), "d"), _num(d.get("site_cn_adduct"), "d"),
             _num(d.get("delta_cn_site"), "+d")],
            ["ECN", _num(d.get("site_ecn_cluster"), ".3f"), _num(d.get("site_ecn_adduct"), ".3f"),
             _num(d.get("delta_ecn_site"), "+.3f")]], [22, 14, 14, 14])

    if d.get("cluster_rearranged") is not None:
        out.append(_hdr("Cluster reorganization on adsorption"))
        changes = d.get("cluster_bond_changes") or {}
        out.append(_line("Bonding rearranged:", "yes" if d["cluster_rearranged"] else "no"))
        if changes:
            out.append(_line("Bond changes vs the free cluster:",
                             ", ".join(f"{k} {v:+d}" for k, v in changes.items())))
        out.append(_line("Displacement of the binding site:", _num(d.get("delta_r_site_ang"), ".4f") + " Å"))
        disp = d.get("cluster_displacement_ang") or []
        atoms = d.get("cluster_atoms") or []
        if disp:
            out.append(" Displacement per cluster atom (Å, after Kabsch alignment):")
            out += _table(["atom", "Δr (Å)"],
                          [[f"{atoms[i] if i < len(atoms) else '?'}{i + 1}", f"{r:.4f}"]
                           for i, r in enumerate(disp)], [10, 10])

    mayer = d.get("site_mayer_bonds") or []
    if mayer:
        out.append(_hdr(f"Mayer bond orders at the binding site {site} (adduct)"))
        out += _table(["partner", "bond order"],
                      [[f"{b['partner_symbol']}({b['partner_index']})", f"{b['order']:.4f}"]
                       for b in mayer], [14, 12])

    bonds = d.get("probe_bonds") or []
    if bonds:
        out.append(_hdr("Probe activation"))
        out += _table(["bond", "free (Å)", "adduct (Å)", "Δ (Å)", "Δ (%)"],
                      [[b["label"], f"{b['free_A']:.4f}", f"{b['adduct_A']:.4f}",
                        f"{b['delta_A']:+.4f}", f"{b['percent']:+.2f}"] for b in bonds],
                      [10, 12, 12, 12, 10])
        angles = d.get("probe_angles") or []
        if angles:
            out += [""] + _table(["angle", "free (°)", "adduct (°)", "Δ (°)"],
                                 [[a["label"], f"{a['free_deg']:.2f}", f"{a['adduct_deg']:.2f}",
                                   f"{a['delta_deg']:+.2f}"] for a in angles], [12, 12, 12, 12])
        out.append(_line("Maximum elongation:", f"{d.get('max_elongation_pct'):+.2f} %"))
        if d.get("max_bend_deg") is not None:
            out.append(_line("Bend of the linear probe:", f"{d['max_bend_deg']:.2f}°"))
        out.append(_line("Activation (5 % stretch, 10° bend):",
                         "ACTIVATED" if d.get("activated") else "no significant activation"))

    if d.get("adduct_imaginary_modes") is not None or d.get("adduct_ir_bands"):
        out.append(_hdr("Vibrational signature of the adduct"))
        n_imag = d.get("adduct_imaginary_modes")
        if n_imag is not None:
            out.append(_line("Imaginary modes:", f"{n_imag}  "
                             + ("(true minimum)" if n_imag == 0 else "(saddle point)")))
        ir = d.get("adduct_ir_bands") or []
        if ir:
            out.append(" Strongest IR-active modes:")
            out += _table(["ν (cm-1)", "I (km/mol)"],
                          [[f"{b['frequency_cm1']:.1f}", f"{b['intensity_km_mol']:.1f}"] for b in ir],
                          [12, 14])

    if d.get("cdft_agreement") is not None:
        out.append(_hdr("CDFT cross-reference"))
        out.append(f" The binding site is the #{d.get('cdft_site_rank')} "
                   f"{d.get('cdft_site_character')} site of the CDFT ranking → "
                   + ("agrees" if d["cdft_agreement"] else "discrepancy"))

    out += _warnings(d.get("warnings") or [])
    return "\n".join(line for line in out if line is not None)


def stability_report_text(d: dict) -> str:
    from cadmia import __version__
    basis = d.get("energy_basis") or "E"
    out: List[Optional[str]] = [
        f" CADMIA Stability Report — CADMIA {__version__}",
        f" System  : {d.get('mixed')}  ({d.get('formula')}, {d.get('n_atoms')} atoms)",
        f" Engine  : {d.get('engine') or '—'}    Energy basis: {basis}"
        + (f"    T = {d['temperature_k']:.2f} K" if d.get("temperature_k") else ""),
    ]

    out.append(_hdr("Reference reaction"))
    refs = d.get("references") or []
    out.append(" " + f"{d.get('formula')} → " + " + ".join(
        f"{_coeff(r.get('coefficient'))} {r.get('label')}" for r in refs))
    out.append(_line("Stoichiometry:", "balanced" if d.get("balanced") else "UNBALANCED"))
    out.append(_line("Reference size:", str(d.get("reference_size"))))
    out += [""] + _table(["term", "formula", "coefficient", f"{basis} (Hartree)"],
                         [[d.get("mixed") or "mixed", d.get("formula") or "", "1",
                           _num(d.get("e_mixed_hartree"), ".6f")]] +
                         [[r.get("label"), r.get("formula"), _coeff(r.get("coefficient")),
                           _num(r.get("e_hartree"), ".6f")] for r in refs],
                         [16, 12, 14, 20])

    mult, imag, sym = d.get("multiplicities") or {}, d.get("n_imaginary") or {}, d.get("symmetry") or {}
    if mult or sym:
        out.append(_hdr("Per-term checks"))
        rows = []
        for term in dict.fromkeys([*mult, *sym]):
            s = sym.get(term) or {}
            rows.append([d.get("mixed") if term == "mixed" else term, _num(mult.get(term), "d"),
                         "—" if imag.get(term) is None else str(imag[term]),
                         str(s.get("point_group") or "—"), _num(s.get("sigma"), "d"),
                         "—" if s.get("chiral") is None else ("yes" if s["chiral"] else "no")])
        out += _table(["term", "multiplicity", "imag. modes", "point group", "σ", "chiral"],
                      rows, [16, 13, 12, 13, 6, 8])
        out.append(" (point group and σ detected from the geometry; '—' imag. = no frequencies)")

    out.append(_hdr(f"Relative stability against the references ({basis})"))
    out += [_hartree("ΔE_rel:", d.get("e_rel_hartree")),
            _kcal("ΔE_rel:", d.get("e_rel_kcal")),
            _line("ΔE_rel per atom:", _num(d.get("e_rel_eV_per_atom"), ">12.4f") + " eV/atom")]
    if (d.get("formula_units") or 1) > 1 or d.get("reference_size") == "equal-size":
        out.append(_kcal(f"ΔE_rel per formula unit (n = {d.get('formula_units')}):",
                         d.get("e_rel_kcal_per_fu")))
    if d.get("e_rel_kcal") is not None:
        out.append(" " + ("More stable than the references." if d["e_rel_kcal"] < 0
                          else "Less stable than the references."))
    by_basis = d.get("e_rel_by_basis_kcal") or {}
    if len(by_basis) > 1:
        out.append(_hdr("Relative stability in every energy basis"))
        out += [_kcal(f"ΔE_rel({b}):", v) for b, v in by_basis.items()]

    out += _warnings(d.get("warnings") or [])
    return "\n".join(line for line in out if line is not None)


def _coeff(c) -> str:
    if c is None:
        return "—"
    return str(int(c)) if float(c).is_integer() else f"{c:g}"
