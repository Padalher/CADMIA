"""Reproducible comparison: CADMIA's native CDFT vs a Multiwfn reference report.

CADMIA computes the conceptual-DFT descriptors natively (no Multiwfn at runtime). To document
that independence, this module lines up CADMIA's results against a Multiwfn CDFT text report
(`parse_multiwfn_cdft`) — global descriptors and condensed Fukui per atom — and reports the
absolute differences. It produces a Markdown table for a report and a machine-readable summary
with the maximum deviation, so a test can assert agreement within tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from cadmia.cdft.multiwfn import parse_multiwfn_cdft

# CADMIA global_fd attribute  ->  (Multiwfn global key, label)
_GLOBAL_MAP = [
    ("ip_ev",              "vertical_ip",               "IP (eV)"),
    ("ea_ev",              "vertical_ea",               "EA (eV)"),
    ("hardness_ev",        "hardness_fundamental_gap",  "η (eV)"),
    ("omega_ev",           "electrophilicity_index",    "ω (eV)"),
    ("nucleophilicity_ev", "nucleophilicity_index",     "N (eV)"),
]
# CADMIA fukui list attribute -> (Multiwfn atom attribute, label)
_FUKUI_MAP = [
    ("f_plus",  "f_plus",          "f+"),
    ("f_minus", "f_minus",         "f-"),
    ("dual",    "dual_descriptor", "Δf"),
]


@dataclass
class ComparisonRow:
    name: str
    cadmia: Optional[float]
    multiwfn: Optional[float]

    @property
    def abs_diff(self) -> Optional[float]:
        if self.cadmia is None or self.multiwfn is None:
            return None
        return abs(self.cadmia - self.multiwfn)


@dataclass
class CDFTComparison:
    global_rows: List[ComparisonRow] = field(default_factory=list)
    fukui_rows: List[ComparisonRow] = field(default_factory=list)

    @property
    def max_abs_diff(self) -> float:
        diffs = [r.abs_diff for r in self.global_rows + self.fukui_rows if r.abs_diff is not None]
        return max(diffs) if diffs else 0.0

    def to_dict(self) -> dict:
        def rows(rs):
            return [{"name": r.name, "cadmia": r.cadmia, "multiwfn": r.multiwfn,
                     "abs_diff": r.abs_diff} for r in rs]
        return {"global": rows(self.global_rows), "fukui": rows(self.fukui_rows),
                "max_abs_diff": self.max_abs_diff}


def compare_to_multiwfn(cadmia_result, multiwfn_report: "str | Path") -> CDFTComparison:
    """Build a CADMIA-vs-Multiwfn comparison from a CADMIA workflow result + a Multiwfn report.

    ``cadmia_result`` is a CDFTWorkflowResult (has ``global_fd`` and ``fukui``).
    ``multiwfn_report`` is a path to a Multiwfn CDFT text file (e.g. ``CDFT.txt``).
    """
    mw = parse_multiwfn_cdft(multiwfn_report)
    cmp = CDFTComparison()

    gfd = cadmia_result.global_fd
    for attr, mw_key, label in _GLOBAL_MAP:
        cad = getattr(gfd, attr, None)
        mw_val = mw.global_descriptors.get(mw_key, {}).get("ev")
        cmp.global_rows.append(ComparisonRow(label, cad, mw_val))

    fukui = cadmia_result.fukui
    for i, atom in enumerate(mw.atoms):
        for cad_attr, mw_attr, sym in _FUKUI_MAP:
            cad_list = getattr(fukui, cad_attr, None)
            cad = cad_list[i] if cad_list is not None and i < len(cad_list) else None
            cmp.fukui_rows.append(
                ComparisonRow(f"{atom.label} {sym}", cad, getattr(atom, mw_attr, None))
            )
    return cmp


def comparison_markdown(cmp: CDFTComparison, *, title: str = "CADMIA vs Multiwfn") -> str:
    """Render the comparison as a Markdown report (global + Fukui tables + max deviation)."""
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    def table(rows):
        out = ["| descriptor | CADMIA | Multiwfn | |Δ| |", "| --- | --- | --- | --- |"]
        for r in rows:
            out.append(f"| {r.name} | {fmt(r.cadmia)} | {fmt(r.multiwfn)} | {fmt(r.abs_diff)} |")
        return "\n".join(out)

    return "\n".join([
        f"# {title}", "",
        "CADMIA computes these descriptors natively; Multiwfn is used only as an external "
        "reference for validation (not at runtime).", "",
        "## Global descriptors", "", table(cmp.global_rows), "",
        "## Condensed Fukui functions", "", table(cmp.fukui_rows), "",
        f"**Maximum absolute deviation:** {cmp.max_abs_diff:.4f}", "",
    ])
