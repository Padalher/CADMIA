"""CDFT report writer — produces Multiwfn-compatible tables plus CADMIA extras."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from cadmia.cdft.calculator import HARTREE_TO_EV, HOMO_TCE_EV, FiniteDifferenceCDFT, FukuiResult
from cadmia.cdft.workflow import CDFTWorkflowResult
from cadmia.core.electronic_descriptors import GlobalElectronic
from cadmia.io.results import fit_width, shown_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hdr(title: str, width: int = 72) -> str:
    return f"\n {title}\n {'=' * (width - 1)}"


def _two_units(label: str, hartree: float, ev: Optional[float] = None) -> str:
    """Format a 'Label: X.XXXXXX Hartree,  Y.XXXX eV' line."""
    if ev is None:
        ev = hartree * HARTREE_TO_EV
    return f" {label:<38}{hartree:>14.6f} Hartree,{ev:>10.4f} eV"


def _ev_only(label: str, ev: float) -> str:
    return f" {label:<38}{ev:>14.4f} eV"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _fukui_title(result: CDFTWorkflowResult) -> str:
    title = (f"{result.charge_scheme.capitalize()} charges, condensed Fukui functions and "
             "condensed dual descriptors")
    if result.charge_scheme == "mulliken":
        # Hirshfeld is read whenever the output has it; landing here means none of the three
        # files carried a Hirshfeld block (Gaussian Pop=Hirshfeld, ORCA Print[P_Hirshfeld]).
        title += ("\n NOTE: no Hirshfeld block in the outputs, so Mulliken charges were used."
                  "\n Mulliken populations depend strongly on the basis set; rerun the three"
                  "\n single points with Pop=Hirshfeld (Gaussian) for basis-stable Fukui values.")
    return title


def _n_state(result: CDFTWorkflowResult):
    state = (result.states or {}).get("N")
    return state.electronic if state is not None else None


def _is_open_shell(electronic) -> bool:
    return (electronic is not None and electronic.homo_beta_ev is not None
            and (electronic.structure.multiplicity or 1) > 1)


def _section_provenance(result: CDFTWorkflowResult) -> str:
    """Who computed what from which files: what a reader needs to redo the numbers."""
    import numpy as np
    from cadmia.io.electronic import describe_level_mismatch, parse_level_of_theory

    states = result.states or {}
    lines = [_hdr("Input states")]
    level = parse_level_of_theory(result.source_n) if result.source_n else {}
    lines.append(f" Level of theory read from N: basis {level.get('basis') or '?'}, "
                 f"dispersion {level.get('dispersion') or '?'}")
    mismatch = describe_level_mismatch(
        {label: s.source_file for label, s in states.items() if s.source_file})
    if mismatch:
        lines.append(f" WARNING: {mismatch}")
    lines.append(f" Energy conversion: 1 Hartree = {HARTREE_TO_EV} eV")
    lines.append("")
    lines.append(f" {'State':<6}{'Charge':>7}{'Mult':>6}{'<S**2>':>10}{'ideal':>8}"
                 f"{'E (Hartree)':>20}{'max|dr| vs N (A)':>19}")
    reference = states.get("N")
    for label in ("N", "N+1", "N-1"):
        state = states.get(label)
        if state is None:
            continue
        mult = state.structure.multiplicity
        ideal = ((mult - 1) / 2) * ((mult - 1) / 2 + 1) if mult else None
        s2 = f"{state.spin_sq:>10.4f}" if state.spin_sq is not None else f"{'-':>10}"
        ideal_txt = f"{ideal:>8.4f}" if ideal is not None else f"{'-':>8}"
        if label == "N" or reference is None:
            shift = f"{'-':>19}"
        else:
            dr = np.abs(np.asarray(state.structure.positions, float)
                        - np.asarray(reference.structure.positions, float)).max()
            shift = f"{dr:>19.6f}"
        lines.append(f" {label:<6}{state.structure.charge if state.structure.charge is not None else '-':>7}"
                     f"{mult if mult is not None else '-':>6}{s2}{ideal_txt}"
                     f"{state.energy_hartree:>20.8f}{shift}")
    lines.append(" Vertical IP/EA require max|dr| = 0: N+1 and N-1 must be single points on the "
                 "N geometry.")
    return "\n".join(lines)


def _section_frontier_by_spin(result: CDFTWorkflowResult) -> str:
    electronic = _n_state(result)
    if not _is_open_shell(electronic):
        return ""
    lines = [
        _hdr("Frontier orbitals of N by spin channel (open shell)"),
        f" {'Channel':<12}{'HOMO (eV)':>12}{'LUMO (eV)':>12}{'Gap (eV)':>12}",
    ]
    for name, homo, lumo, gap in (
        ("alpha", electronic.homo_alpha_ev, electronic.lumo_alpha_ev, electronic.gap_alpha_ev),
        ("beta", electronic.homo_beta_ev, electronic.lumo_beta_ev, electronic.gap_beta_ev),
        ("cross", electronic.homo_ev, electronic.lumo_ev, electronic.homo_lumo_gap_ev),
    ):
        cells = "".join(f"{v:>12.4f}" if v is not None else f"{'-':>12}" for v in (homo, lumo, gap))
        lines.append(f" {name:<12}{cells}")
    lines.append(" cross = max(HOMO alpha, HOMO beta) to min(LUMO alpha, LUMO beta). It joins two")
    lines.append(" spin channels, so it is not a spin-conserving excitation; report the alpha and")
    lines.append(" beta gaps, and compare open-shell species with closed-shell ones channel by channel.")
    return "\n".join(lines)


def _section_fukui(result: CDFTWorkflowResult) -> str:
    """Fukui section without raw charge columns (fallback when q(N/N±1) unavailable)."""
    f = result.fukui
    if f.f_plus is None and f.f_minus is None:
        return ""

    lines = [
        _hdr(_fukui_title(result)),
        f" Charge scheme: {result.charge_scheme}",
        " Units: e (elementary charge)",
        f" {'Atom':>8}  {'q(N)':>8}  {'q(N+1)':>8}  {'q(N-1)':>8}"
        f"  {'f-':>8}  {'f+':>8}  {'f0':>8}  {'CDD':>9}",
    ]
    for i, atom in enumerate(result.structure.atoms):
        tag = f"{i + 1}({atom})"
        fp = f"{f.f_plus[i]:>8.4f}" if f.f_plus else "     ---"
        fm = f"{f.f_minus[i]:>8.4f}" if f.f_minus else "     ---"
        fz = f"{f.f_zero[i]:>8.4f}" if f.f_zero else "     ---"
        dd = f"{f.dual[i]:>9.4f}" if f.dual else "      ---"
        lines.append(f"  {tag:>8}  {'---':>8}  {'---':>8}  {'---':>8}  {fm}  {fp}  {fz}  {dd}")

    return "\n".join(lines)


def _section_fukui_full(
    result: CDFTWorkflowResult,
    charges_n: list,
    charges_np1: Optional[list],
    charges_nm1: Optional[list],
) -> str:
    """Fukui section with actual charge columns."""
    f = result.fukui
    lines = [
        _hdr(_fukui_title(result)),
        f" Charge scheme: {result.charge_scheme}",
        " Units: e (elementary charge)",
        f" {'Atom':>8}  {'q(N)':>8}  {'q(N+1)':>8}  {'q(N-1)':>8}"
        f"  {'f-':>8}  {'f+':>8}  {'f0':>8}  {'CDD':>9}",
    ]
    for i, atom in enumerate(result.structure.atoms):
        tag = f"{i + 1}({atom})"
        qn  = f"{charges_n[i]:>8.4f}"
        qnp = f"{charges_np1[i]:>8.4f}" if charges_np1 else "     ---"
        qnm = f"{charges_nm1[i]:>8.4f}" if charges_nm1 else "     ---"
        fp  = f"{f.f_plus[i]:>8.4f}"  if f.f_plus  else "     ---"
        fm  = f"{f.f_minus[i]:>8.4f}" if f.f_minus else "     ---"
        fz  = f"{f.f_zero[i]:>8.4f}"  if f.f_zero  else "     ---"
        dd  = f"{f.dual[i]:>9.4f}"    if f.dual    else "      ---"
        lines.append(f"  {tag:>8}  {qn}  {qnp}  {qnm}  {fm}  {fp}  {fz}  {dd}")
    return "\n".join(lines)


def _section_local_elec_nucl(result: CDFTWorkflowResult) -> str:
    f = result.fukui
    if f.local_electrophilicity is None and f.local_nucleophilicity is None:
        return ""
    lines = [
        _hdr("Condensed local electrophilicity / nucleophilicity index (e·eV)"),
        f" {'Atom':>8}  {'Electrophilicity':>18}  {'Nucleophilicity':>17}",
    ]
    for i, atom in enumerate(result.structure.atoms):
        tag = f"{i + 1}({atom})"
        le = f"{f.local_electrophilicity[i]:>18.5f}" if f.local_electrophilicity else "               ---"
        ln = f"{f.local_nucleophilicity[i]:>17.5f}" if f.local_nucleophilicity else "              ---"
        lines.append(f"  {tag:>8}  {le}  {ln}")
    return "\n".join(lines)


def _section_softness(result: CDFTWorkflowResult) -> str:
    f = result.fukui
    if f.local_softness_plus is None and f.local_softness_minus is None:
        return ""
    lines = [
        _hdr("Condensed local softnesses (Hartree·e), relative electro-/nucleophilicity"),
        f" {'Atom':>8}  {'s-':>9}  {'s+':>9}  {'s0':>9}  {'s+/s-':>9}  {'s-/s+':>9}",
    ]
    for i, atom in enumerate(result.structure.atoms):
        tag = f"{i + 1}({atom})"
        sm  = f"{f.local_softness_minus[i]:>9.4f}" if f.local_softness_minus else "      ---"
        sp  = f"{f.local_softness_plus[i]:>9.4f}"  if f.local_softness_plus  else "      ---"
        s0  = f"{f.softness_zero[i]:>9.4f}"         if f.softness_zero        else "      ---"
        re_ = f"{f.relative_electrophilicity[i]:>9.4f}" if f.relative_electrophilicity else "      ---"
        rn  = f"{f.relative_nucleophilicity[i]:>9.4f}"  if f.relative_nucleophilicity  else "      ---"
        lines.append(f"  {tag:>8}  {sm}  {sp}  {s0}  {re_}  {rn}")
    return "\n".join(lines)


def _section_extra_local(result: CDFTWorkflowResult) -> str:
    """CADMIA extra local descriptors not in Multiwfn."""
    f = result.fukui
    has_lh = f.f_zero_scaled_by_hardness is not None
    has_don = f.donicity is not None
    has_acc = f.accepting_capacity is not None
    if not (has_lh or has_don or has_acc):
        return ""
    lines = [
        _hdr("CADMIA — Additional local descriptors"),
        " Hardness-weighted f0 η·f0 (eV·e), donicity IP·f- (eV·e), accepting capacity EA·f+ (eV·e)",
        " (η·f0 is a heuristic weighting, NOT the canonical local hardness of CDFT theory)",
        f" {'Atom':>8}  {'η·f0':>12}  {'IP·f-':>12}  {'EA·f+':>12}",
    ]
    for i, atom in enumerate(result.structure.atoms):
        tag = f"{i + 1}({atom})"
        lh  = f"{f.f_zero_scaled_by_hardness[i]:>12.5f}" if has_lh else "           ---"
        don = f"{f.donicity[i]:>12.5f}"             if has_don else "           ---"
        acc = f"{f.accepting_capacity[i]:>12.5f}"   if has_acc else "           ---"
        lines.append(f"  {tag:>8}  {lh}  {don}  {acc}")
    return "\n".join(lines)


def _section_global_fd(fd: FiniteDifferenceCDFT, open_shell: bool = False) -> str:
    lines = [_hdr("Global CDFT descriptors — SCF finite difference (vertical)")]
    lines.append(_two_units("E(N):", fd.energy_n_hartree))
    lines.append(_two_units("E(N+1):", fd.energy_np1_hartree))
    lines.append(_two_units("E(N-1):", fd.energy_nm1_hartree))
    if fd.homo_n_ev is not None:
        lines.append(_two_units("E_HOMO(N):", fd.homo_n_ev / HARTREE_TO_EV, fd.homo_n_ev))
    if fd.homo_np1_ev is not None:
        lines.append(_two_units("E_HOMO(N+1):", fd.homo_np1_ev / HARTREE_TO_EV, fd.homo_np1_ev))
    if fd.homo_nm1_ev is not None:
        lines.append(_two_units("E_HOMO(N-1):", fd.homo_nm1_ev / HARTREE_TO_EV, fd.homo_nm1_ev))
    if fd.lumo_n_ev is not None:
        lines.append(_two_units("E_LUMO(N):", fd.lumo_n_ev / HARTREE_TO_EV, fd.lumo_n_ev))
        if fd.homo_n_ev is not None:
            gap_ev = fd.lumo_n_ev - fd.homo_n_ev
            label = "HOMO-LUMO gap (cross-spin):" if open_shell else "HOMO-LUMO gap:"
            lines.append(_two_units(label, gap_ev / HARTREE_TO_EV, gap_ev))
    lines.append(_two_units("Vertical IP (E(N-1)-E(N)):", fd.ip_ev / HARTREE_TO_EV, fd.ip_ev))
    lines.append(_two_units("Vertical EA (E(N)-E(N+1)):", fd.ea_ev / HARTREE_TO_EV, fd.ea_ev))
    lines.append(_two_units("Mulliken electronegativity χ:", fd.chi_ev / HARTREE_TO_EV, fd.chi_ev))
    lines.append(_two_units("Chemical potential μ:", fd.mu_ev / HARTREE_TO_EV, fd.mu_ev))
    lines.append(_two_units("Hardness η (IP-EA):", fd.hardness_ev / HARTREE_TO_EV, fd.hardness_ev))
    s_h = fd.softness * HARTREE_TO_EV  # Hartree^-1
    lines.append(f" {'Softness S:':<38}{fd.softness:>14.6f} eV^-1,  {s_h:>8.4f} Hartree^-1")
    lines.append(_two_units("Electrophilicity index ω:", fd.omega_ev / HARTREE_TO_EV, fd.omega_ev))
    lines.append(_two_units("Electrodonating power ω-:", fd.omega_minus_ev / HARTREE_TO_EV, fd.omega_minus_ev))
    lines.append(_two_units("Electroaccepting power ω+:", fd.omega_plus_ev / HARTREE_TO_EV, fd.omega_plus_ev))
    if fd.nucleophilicity_ev is not None:
        lines.append(
            _two_units(
                "Nucleophilicity index N:",
                fd.nucleophilicity_hartree,
                fd.nucleophilicity_ev,
            )
        )
        tce = fd.homo_tce_ev if fd.homo_tce_ev is not None else HOMO_TCE_EV
        lines.append(f" (N = HOMO(molecule) - HOMO(TCE);  HOMO_TCE = {tce:.4f} eV)")
    return "\n".join(lines)


def _section_global_koopmans(gk: GlobalElectronic, open_shell: bool = False) -> str:
    gap_label = "HOMO-LUMO gap (cross-spin):" if open_shell else "HOMO-LUMO gap:"
    lines = [
        _hdr("CADMIA — Koopmans descriptors (orbital energies of N state)"),
        f" {'HOMO(N):':<38}{gk.homo_ev:>14.4f} eV",
        f" {'LUMO(N):':<38}{gk.lumo_ev:>14.4f} eV",
        _two_units(gap_label, gk.gap_ev / HARTREE_TO_EV, gk.gap_ev),
        _two_units("IP_K = -HOMO:", gk.ip_ev / HARTREE_TO_EV, gk.ip_ev),
        _two_units("EA_K = -LUMO:", gk.ea_ev / HARTREE_TO_EV, gk.ea_ev),
        _two_units("Hardness η_K = LUMO-HOMO:", gk.eta_ev / HARTREE_TO_EV, gk.eta_ev),
        _two_units("Chemical potential μ_K:", gk.mu_ev / HARTREE_TO_EV, gk.mu_ev),
        _two_units("Electronegativity χ_K:", gk.chi_ev / HARTREE_TO_EV, gk.chi_ev),
        _two_units("Electrophilicity ω_K:", gk.omega_ev / HARTREE_TO_EV, gk.omega_ev),
        _two_units("Electrodonating power ω-_K:", gk.omega_minus_ev / HARTREE_TO_EV, gk.omega_minus_ev),
        _two_units("Electroaccepting power ω+_K:", gk.omega_plus_ev / HARTREE_TO_EV, gk.omega_plus_ev),
    ]
    if open_shell:
        lines.append(_KOOPMANS_OPEN_SHELL_NOTE)
    return "\n".join(lines)


_KOOPMANS_OPEN_SHELL_NOTE = (
    " NOTE: open shell. HOMO and LUMO may belong to different spins, so the gap and every\n"
    " Koopmans quantity built on it (η, S, μ, χ, ω, ω-, ω+) are cross-spin. Use the SCF\n"
    " (total-energy) descriptors; the per-spin gaps are in the frontier section.")


def _section_koopmans_vs_scf(fd: FiniteDifferenceCDFT, gk: GlobalElectronic,
                             open_shell: bool = False) -> str:
    """Side-by-side comparison table of Koopmans vs SCF descriptors."""
    col = 18
    lines = [
        _hdr("Comparison: Koopmans theorem vs SCF finite difference"),
        f" {'Descriptor':<22}  {'Koopmans':>{col}}  {'SCF (vertical)':>{col}}  Unit",
        f" {'-'*22}  {'-'*col}  {'-'*col}  ----",
    ]

    def row(label, k_val, s_val, unit="eV"):
        kv = f"{k_val:>{col}.4f}" if k_val is not None else f"{'---':>{col}}"
        sv = f"{s_val:>{col}.4f}" if s_val is not None else f"{'---':>{col}}"
        return f" {label:<22}  {kv}  {sv}  {unit}"

    gap_scf = (fd.lumo_n_ev - fd.homo_n_ev) if (fd.lumo_n_ev and fd.homo_n_ev) else None

    lines.append(row("HOMO",          gk.homo_ev,   fd.homo_n_ev))
    lines.append(row("LUMO",          gk.lumo_ev,   fd.lumo_n_ev))
    lines.append(row("Gap (cross-spin)" if open_shell else "Gap", gk.gap_ev, gap_scf))
    lines.append(row("IP",            gk.ip_ev,     fd.ip_ev))
    lines.append(row("EA",            gk.ea_ev,     fd.ea_ev))
    lines.append(row("Hardness η",    gk.eta_ev,    fd.hardness_ev))
    lines.append(row("Softness S",    gk.softness,  fd.softness,   "eV^-1"))
    lines.append(row("Chem. pot. μ",  gk.mu_ev,     fd.mu_ev))
    lines.append(row("Electroneg. χ", gk.chi_ev,    fd.chi_ev))
    lines.append(row("Electrophil. ω",gk.omega_ev,  fd.omega_ev))
    lines.append(row("Electrodon. ω-", gk.omega_minus_ev, fd.omega_minus_ev))
    lines.append(row("Electroacc. ω+", gk.omega_plus_ev,  fd.omega_plus_ev))
    lines.append(f" {'':22}  {'(orbital energies)':>{col}}  {'(total energy diffs)':>{col}}")
    if open_shell:
        lines.append(_KOOPMANS_OPEN_SHELL_NOTE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_cdft_report(
    result: CDFTWorkflowResult,
    filepath: str | Path,
    charges_n: Optional[list] = None,
    charges_np1: Optional[list] = None,
    charges_nm1: Optional[list] = None,
    fmt: str = "txt",
    nuclear: Optional[list] = None,
) -> None:
    """Write a CDFT report to *filepath*.

    Parameters
    ----------
    result:
        Output of :func:`run_cdft_workflow`.
    filepath:
        Destination path.
    charges_n / charges_np1 / charges_nm1:
        Raw per-atom charges for each state. When provided, the Fukui section
        includes the q(N), q(N+1), q(N-1) columns (matching Multiwfn format).
    fmt:
        ``"txt"`` (default) or ``"json"``.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        payload = cdft_report_payload(result, charges_n, charges_np1, charges_nm1, nuclear)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    path.write_text(cdft_report_text(result, charges_n, charges_np1, charges_nm1) + "\n",
                    encoding="utf-8")


def cdft_report_text(
    result: CDFTWorkflowResult,
    charges_n=None,
    charges_np1=None,
    charges_nm1=None,
) -> str:
    """The full text report."""
    import cadmia
    from datetime import datetime

    fd = result.global_fd
    formula = result.structure.formula

    header = (
        f" CADMIA CDFT Report — CADMIA {cadmia.__version__}, "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f" System  : {formula}  ({result.structure.n_atoms} atoms)\n"
        f" Source N: {shown_path(result.source_n) or '(not specified)'}\n"
    )
    if result.source_np1:
        header += f" Source N+1: {shown_path(result.source_np1)}\n"
    if result.source_nm1:
        header += f" Source N-1: {shown_path(result.source_nm1)}\n"
    header += (
        f"\n Note: the nucleophilicity index N is referenced to tetracyanoethylene\n"
        f"       (TCE) — the standard reference of Domingo's nucleophilicity scale,\n"
        f"       evaluated at B3LYP/6-31G*. Use `cadmia cdft --tce-homo` or the\n"
        f"       homo_tce_ev parameter to set a reference at your level of theory.\n"
    )

    sections = [header]
    if result.states:
        sections.append(_section_provenance(result))

    # Fukui / charges section
    if charges_n is not None:
        sections.append(
            _section_fukui_full(result, charges_n, charges_np1, charges_nm1)
        )
    elif result.fukui.f_plus is not None or result.fukui.f_minus is not None:
        sections.append(_section_fukui(result))

    sections.append(_section_local_elec_nucl(result))
    sections.append(_section_softness(result))
    sections.append(_section_extra_local(result))
    open_shell = _is_open_shell(_n_state(result))
    sections.append(_section_global_fd(fd, open_shell=open_shell))
    sections.append(_section_frontier_by_spin(result))
    if result.global_koopmans is not None:
        sections.append(_section_global_koopmans(result.global_koopmans, open_shell=open_shell))
        sections.append(_section_koopmans_vs_scf(fd, result.global_koopmans, open_shell=open_shell))

    return fit_width("\n".join(s for s in sections if s))


def cdft_report_payload(
    result: CDFTWorkflowResult,
    charges_n=None,
    charges_np1=None,
    charges_nm1=None,
    nuclear=None,
) -> dict:
    """Every value of the full report, as data (format ``cadmia-cdft-v1``)."""
    fd = result.global_fd
    f = result.fukui

    atoms = [
        {
            "index": i + 1,
            "element": result.structure.atoms[i],
            "label": result.structure.labels[i],
            "q_n":   charges_n[i]  if charges_n  else None,
            "q_np1": charges_np1[i] if charges_np1 else None,
            "q_nm1": charges_nm1[i] if charges_nm1 else None,
            "f_minus": f.f_minus[i] if f.f_minus else None,
            "f_plus":  f.f_plus[i]  if f.f_plus  else None,
            "f_zero":  f.f_zero[i]  if f.f_zero  else None,
            "dual":    f.dual[i]    if f.dual     else None,
            "s_minus": f.local_softness_minus[i]    if f.local_softness_minus    else None,
            "s_plus":  f.local_softness_plus[i]     if f.local_softness_plus     else None,
            "s_zero":  f.softness_zero[i]            if f.softness_zero           else None,
            "rel_electrophilicity": f.relative_electrophilicity[i] if f.relative_electrophilicity else None,
            "rel_nucleophilicity":  f.relative_nucleophilicity[i]  if f.relative_nucleophilicity  else None,
            "local_electrophilicity": f.local_electrophilicity[i] if f.local_electrophilicity else None,
            "local_nucleophilicity":  f.local_nucleophilicity[i]  if f.local_nucleophilicity  else None,
            # η·f0 — f0 rescaled by a global constant, not a local hardness (see calculator.py)
            "f_zero_scaled_by_hardness": (f.f_zero_scaled_by_hardness[i]
                                          if f.f_zero_scaled_by_hardness else None),
            "donicity":             f.donicity[i]             if f.donicity             else None,
            "accepting_capacity":   f.accepting_capacity[i]   if f.accepting_capacity   else None,
        }
        for i in range(result.structure.n_atoms)
    ]

    import cadmia

    gk = result.global_koopmans
    electronic = _n_state(result)
    states = [
        {
            "state": label,
            "source_file": state.source_file,
            "charge": state.structure.charge,
            "multiplicity": state.structure.multiplicity,
            "spin_sq": state.spin_sq,
            "energy_hartree": state.energy_hartree,
        }
        for label, state in (result.states or {}).items()
    ]
    payload = {
        "format": "cadmia-cdft-v1",
        "cadmia_version": cadmia.__version__,
        "hartree_to_ev": HARTREE_TO_EV,
        "states": states,
        "structure": {
            "formula": result.structure.formula,
            "n_atoms": result.structure.n_atoms,
            "charge_scheme": result.charge_scheme,
            "source_n": result.source_n,
            "source_np1": result.source_np1,
            "source_nm1": result.source_nm1,
        },
        "global_fd": {
            "energy_n_hartree": fd.energy_n_hartree,
            "energy_np1_hartree": fd.energy_np1_hartree,
            "energy_nm1_hartree": fd.energy_nm1_hartree,
            "homo_n_ev": fd.homo_n_ev,
            "homo_np1_ev": fd.homo_np1_ev,
            "homo_nm1_ev": fd.homo_nm1_ev,
            "lumo_n_ev": fd.lumo_n_ev,
            "open_shell": _is_open_shell(electronic),
            "homo_alpha_ev": electronic.homo_alpha_ev if electronic else None,
            "lumo_alpha_ev": electronic.lumo_alpha_ev if electronic else None,
            "homo_beta_ev": electronic.homo_beta_ev if electronic else None,
            "lumo_beta_ev": electronic.lumo_beta_ev if electronic else None,
            "gap_alpha_ev": electronic.gap_alpha_ev if electronic else None,
            "gap_beta_ev": electronic.gap_beta_ev if electronic else None,
            "ip_ev": fd.ip_ev,
            "ea_ev": fd.ea_ev,
            "mu_ev": fd.mu_ev,
            "chi_ev": fd.chi_ev,
            "hardness_ev": fd.hardness_ev,
            "softness_ev_inv": fd.softness,
            "omega_ev": fd.omega_ev,
            "omega_minus_ev": fd.omega_minus_ev,
            "omega_plus_ev": fd.omega_plus_ev,
            "nucleophilicity_ev": fd.nucleophilicity_ev,
            "nucleophilicity_hartree": fd.nucleophilicity_hartree,
            "homo_tce_ev": fd.homo_tce_ev,
        },
        "global_koopmans": {
            "homo_ev": gk.homo_ev,
            "lumo_ev": gk.lumo_ev,
            "gap_ev": gk.gap_ev,
            "ip_ev": gk.ip_ev,
            "ea_ev": gk.ea_ev,
            "eta_ev": gk.eta_ev,
            "mu_ev": gk.mu_ev,
            "chi_ev": gk.chi_ev,
            "softness": gk.softness,
            "omega_ev": gk.omega_ev,
            "omega_minus_ev": gk.omega_minus_ev,
            "omega_plus_ev": gk.omega_plus_ev,
        } if gk is not None else None,
        "atoms": atoms,
    }
    if nuclear:
        # The nuclear response is computed by a separate path and joined here BY ATOM INDEX,
        # never by list position: `atoms` is in atom order but the caller's rows need not be.
        by_index = {r["atom_index"]: r for r in nuclear}
        for entry in atoms:
            row = by_index.get(entry["index"])
            if row:
                for key in ("phi_norm", "phi_along", "phi_dual_norm"):
                    if key in row:
                        entry[key] = row[key]
    return payload
