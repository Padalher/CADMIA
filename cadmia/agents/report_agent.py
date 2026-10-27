"""Report Agent — consolidate agent results into Markdown / LaTeX tables + correlations.

Consumes the JSON (or in-memory dicts/objects) produced by the CDFT, Analysis, Stability and
Benchmark agents and produces a single report: comparative tables, an adsorption-energy summary
(with BSSE), and correlations such as Δf vs E_ads. Matplotlib figures are written
only when matplotlib is installed; otherwise the (x, y) data is still emitted so the user can
plot elsewhere.
"""
from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cadmia.core.stats import linregress
from cadmia.core.units import HARTREE_TO_KCAL

# Two adducts whose total energies agree within this (and that bind the same site) are the
# same DFT minimum: the prescreen dedups at the xTB level, but different starts can still
# relax to one well at DFT, so we collapse them here before summarizing/correlating.
_DEDUP_E_KCAL = 0.3


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _load(obj: Any) -> Optional[dict]:
    """Normalize an input (dict / dataclass with to_dict / JSON path / JSON string) to a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    text = obj
    p = Path(str(obj))
    if p.exists():
        text = p.read_text(encoding="utf-8")
    return json.loads(text)


def _fmt(v: Any, prec: int = 4, signed: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:+.{prec}f}" if signed else f"{v:.{prec}f}"
    return str(v)


def _cplx_label(i: Any) -> str:
    """Column header for an adduct: 'adduct c{i}', or just 'adduct' for a single unindexed run."""
    return "adduct" if i is None else f"adduct c{i}"


def _cfg_cell(i: Any) -> str:
    """Config-index table cell: '—' when there is no configuration index (single run)."""
    return "—" if i is None else str(i)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def _latex_table(headers: List[str], rows: List[List[str]], caption: str, label: str) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    out = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{cols}}}", r"\hline",
        " & ".join(headers) + r" \\", r"\hline",
    ]
    for r in rows:
        out.append(" & ".join(r) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def _dedup_analyses(analyses: List[dict]) -> Tuple[List[dict], Dict[Any, List[Any]]]:
    """Collapse analyses that relaxed to the same physical adsorption minimum.

    Two configs share a minimum when their total adduct energies agree within
    ``_DEDUP_E_KCAL`` AND they bind the same site atom. Returns the unique analyses
    (the lowest-energy representative of each group) and a map kept id → [absorbed
    ids] so the report can state how many duplicates were removed.
    """
    unique: List[dict] = []
    duplicates: Dict[Any, List[Any]] = {}
    def key(a):
        # config_index restarts at 0 on every `analyze` run; the saved result's id does not
        return a.get("config_id") or a.get("adduct_path") or id(a)

    for a in sorted(analyses, key=lambda d: (d.get("e_adduct_hartree") is None,
                                             d.get("e_adduct_hartree") or 0.0)):
        e, site = a.get("e_adduct_hartree"), a.get("site_atom_index")
        match = None
        if e is not None:
            for u in unique:
                ue = u.get("e_adduct_hartree")
                if (ue is not None and site == u.get("site_atom_index")
                        and abs(e - ue) * HARTREE_TO_KCAL <= _DEDUP_E_KCAL):
                    match = u
                    break
        if match is None:
            unique.append(a)
            duplicates.setdefault(key(a), [])
        else:
            duplicates[key(match)].append(key(a))
    unique.sort(key=lambda d: (d.get("config_index") is None, d.get("config_index")))
    return unique, duplicates


# ─────────────────────────────────────────────
# Result objects
# ─────────────────────────────────────────────

@dataclass
class Correlation:
    name:      str
    x_label:   str
    y_label:   str
    x:         List[float] = field(default_factory=list)
    y:         List[float] = field(default_factory=list)
    labels:    List[str] = field(default_factory=list)
    pearson_r: Optional[float] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "x_label": self.x_label, "y_label": self.y_label,
                "x": self.x, "y": self.y, "labels": self.labels, "pearson_r": self.pearson_r}


@dataclass
class Report:
    title:         str
    markdown:      str
    latex:         str
    correlations:  List[Correlation] = field(default_factory=list)
    warnings:      List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"title": self.title, "markdown": self.markdown, "latex": self.latex,
                "correlations": [c.to_dict() for c in self.correlations],
                "warnings": self.warnings}

    def to_files(self, output: str | Path, *, latex: bool = False,
                 figures: bool = False) -> Dict[str, Any]:
        """Write the Markdown report (always), optional LaTeX tables, and optional figures.

        Figures need matplotlib; when it is missing they are skipped and each correlation is
        written as a CSV instead. Returns a dict of what was written.
        """
        out = Path(output)
        out.write_text(self.markdown, encoding="utf-8")
        written: Dict[str, Any] = {"markdown": str(out), "latex": None, "figures": [], "csv": []}

        if latex:
            tex = out.with_suffix(".tex")
            tex.write_text(self.latex, encoding="utf-8")
            written["latex"] = str(tex)

        if figures:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                have_mpl = True
            except ImportError:
                have_mpl = False
                self.warnings.append("matplotlib not installed — figures skipped, CSV written instead.")

            for corr in self.correlations:
                if len(corr.x) < 2:
                    continue
                stem = out.with_name(f"{out.stem}_{corr.name}")
                if have_mpl:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(corr.x, corr.y)
                    for xi, yi, lab in zip(corr.x, corr.y, corr.labels):
                        ax.annotate(lab, (xi, yi), fontsize=7)
                    ax.set_xlabel(corr.x_label)
                    ax.set_ylabel(corr.y_label)
                    r = f"r={corr.pearson_r:.3f}" if corr.pearson_r is not None else ""
                    ax.set_title(f"{corr.name}  {r}")
                    fig.tight_layout()
                    png = stem.with_suffix(".png")
                    fig.savefig(png, dpi=150)
                    plt.close(fig)
                    written["figures"].append(str(png))
                else:
                    csv = stem.with_suffix(".csv")
                    lines = [f"{corr.x_label},{corr.y_label},label"]
                    lines += [f"{xi},{yi},{lab}" for xi, yi, lab in zip(corr.x, corr.y, corr.labels)]
                    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    written["csv"].append(str(csv))
        return written


# ─────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────

_GLOBAL_ORDER = [
    ("homo_eV", "HOMO", "eV"), ("lumo_eV", "LUMO", "eV"), ("gap_eV", "Gap", "eV"),
    ("ip_eV", "IP", "eV"), ("ea_eV", "EA", "eV"), ("hardness_eV", "η(FD)", "eV"),
    ("mu_eV", "μ", "eV"), ("omega_eV", "ω", "eV"), ("nucleophilicity_eV", "N", "eV"),
]

_BANNER = """\
 ██████╗ █████╗ ██████╗ ███╗   ███╗██╗ █████╗
██╔════╝██╔══██╗██╔══██╗████╗ ████║██║██╔══██╗
██║     ███████║██║  ██║██╔████╔██║██║███████║
██║     ██╔══██║██║  ██║██║╚██╔╝██║██║██╔══██║
╚██████╗██║  ██║██████╔╝██║ ╚═╝ ██║██║██║  ██║
 ╚═════╝╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝"""


_STAGE_NAMES = {"structure": "Structure", "cdft": "CDFT", "analysis": "Adsorption",
                "stability": "Stability", "benchmark": "Benchmark"}
_STAGE_SCOPE = {
    "structure": "Structural analysis: coordination, polyhedra and rings",
    "cdft": "Conceptual DFT: global descriptors and reactive sites",
    "analysis": "Adsorption: energetics, electronic and structural changes",
    "stability": "Relative stability against the reference systems",
    "benchmark": "Engine benchmark: Gaussian vs ORCA",
}
_BOX_WIDTH = 72


# Fixed, so a module keeps its number whatever else the report holds. Reserved: 0 calculation and
# preparation (from an .xyz and an input), 5 characterization (DOS, PDOS, IR, MEP).
MODULE_NUMBER = {"structure": 1, "cdft": 2, "analysis": 3, "stability": 4}


def _stage_open(number: int, stage: str) -> List[str]:
    """A framed title, so a reader paging through the report sees where each stage starts."""
    inner = _BOX_WIDTH - 2
    title = f"  MODULE {number}  ·  {_STAGE_NAMES[stage].upper()}"
    return ["```", "╔" + "═" * inner + "╗", f"║{title:<{inner}}║",
            f"║{'  ' + _STAGE_SCOPE[stage]:<{inner}}║", "╚" + "═" * inner + "╝", "```", ""]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _level_of_theory(meta: dict) -> Optional[str]:
    """Build a 'PBE0/def2-TZVP + D3BJ' string from meta, or use a verbatim override."""
    if meta.get("level_of_theory"):
        return meta["level_of_theory"]
    func, basis = meta.get("functional"), meta.get("basis")
    if func and basis:
        lot = f"{func}/{basis}"
        disp = (meta.get("dispersion") or "").strip()
        if disp and disp.lower() != "none":
            lot += f" + {disp}"
        return lot
    return None


def _header_block(title: str, meta: dict) -> List[str]:
    from cadmia import __version__
    from cadmia.citation import cite_block
    return [
        "```", _BANNER, "", cite_block(), "```", "",
        f"# {title}", "",
        "**CADMIA** — Computational Agent-Driven Molecular Intelligence Architecture  ",
        f"Version {__version__} · Pablo Daniel Álvarez Hernández (UJAT/CICTAT) · pablo.alvarez@padalher.com  ",
        f"Report generated: {meta.get('generated') or _now_iso()}",
        "",
    ]


def _engine_keywords(cdft_d: Optional[dict], analysis_ds: List[dict],
                     stab_d: Optional[dict] = None, structures: List[dict] = ()) -> list:
    """(stage, output, keywords) for each stage whose output can still be read."""
    from cadmia.io.electronic import engine_keywords
    sources = [("CDFT", ((cdft_d or {}).get("sources") or {}).get("N"))]
    sources += [("Structure", (st.get("structure") or {}).get("source_file")) for st in structures]
    sources += [("Adsorption", d.get("adduct_path")) for d in analysis_ds]
    stab_src = (stab_d or {}).get("sources") or {}
    sources += [("Stability", f) for f in [stab_src.get("mixed"), *(stab_src.get("references") or [])]]
    found, seen = [], set()
    for stage, src in sources:
        if src and src not in seen and Path(src).is_file():
            seen.add(src)
            kw = engine_keywords(src)
            if kw:
                found.append((stage, src, kw))
    return found


def _run_info_block(meta: dict, contributions: List[str]) -> List[str]:
    lines = ["## Run information", ""]
    engine = meta.get("engine")
    lot = _level_of_theory(meta)
    if engine:
        lines.append(f"- **Engine:** {engine}")
    if lot:
        lines.append(f"- **Level of theory:** {lot}")
    elif meta.get("engine_keywords"):
        lines.append("- **Level of theory** (keywords echoed by the engine):")
        lines += [f"  - {stage} (`{Path(src).name}`): `{kw}`"
                  for stage, src, kw in meta["engine_keywords"]]
    if meta.get("command"):
        lines.append(f"- **Command:** `{meta['command']}`")
    files = meta.get("files") or []
    if files:
        lines.append("- **Inputs:** " + ", ".join(f"`{Path(f).name}`" for f in files))
    lines += ["", "## Workflow modules", ""]
    all_stages = [
        ("structure", "Structure: coordination, polyhedra, rings"),
        ("cdft",      "Reactivity (CDFT): global descriptors + site ranking"),
        ("analysis",  "Adsorption energetics (probe–cluster configurations)"),
        ("stability", "Relative (formation) stability"),
        # Only shown when benchmark data is given.
        ("benchmark", "Engine benchmark (Gaussian vs ORCA)"),
    ]
    for key, label in all_stages:
        if key == "benchmark" and key not in contributions:
            continue
        mark = "x" if key in contributions else " "
        lines.append(f"- [{mark}] {label}")
    lines.append("")
    return lines


def _footer_block(contributions: List[str], n_warnings: int) -> List[str]:
    import platform
    from cadmia import __version__
    from cadmia.core.provenance import software_version
    stages = ", ".join(_STAGE_NAMES[c] for c in contributions) or "none"
    from cadmia.citation import quote_block
    return ["---", "",
            f"Modules reported: {stages} · warnings: {n_warnings}  ",
            f"{software_version()} · Python {platform.python_version()} · {platform.system()}",
            "", "```", quote_block(), "```", "",
            f"Normal termination of CADMIA {__version__} — {_now_iso()}", ""]


def _structure_md(st: dict) -> List[str]:
    """Summary of a structural report JSON: the per-atom and polyhedron tables plus the
    connectivity and ring counts. The full analysis stays in the structural report itself."""
    s = st.get("structure") or {}
    sym = st.get("symmetry") or {}
    md = ["## Structure", "",
          f"- **Structure:** {s.get('structure_id')} · {s.get('formula')} · "
          f"{len(s.get('atoms') or [])} atoms · point group {sym.get('point_group') or '—'}", ""]
    rows = [[a.get("label"), str(a.get("cn")), _fmt(a.get("ecn"), 3),
             _fmt(a.get("effective_avg_bond_distance"), 2),
             ", ".join(f"{e}:{n}" for e, n in (a.get("bonded_elements") or {}).items())]
            for a in st.get("local_descriptors") or []]
    if rows:
        md += [_md_table(["atom", "CN", "ECN", "d_av (Å)", "neighbors"], rows), ""]
    prows = [[p.get("center_label"), str(p.get("cn")), str(p.get("polyhedron_type")),
              str(p.get("shape")), _fmt(p.get("shape_measure"), 2)]
             for p in st.get("polyhedra") or []]
    if prows:
        md += [_md_table(["center", "CN_L", "polyhedron", "shape", "CShM"], prows), ""]
    conn = (st.get("polyhedra_connectivity") or {}).get("summary") or {}
    if conn:
        k = conn.get("by_kind") or {}
        md.append(f"- **Polyhedra links:** {conn.get('links')} ({k.get('corner', 0)} corner, "
                  f"{k.get('edge', 0)} edge, {k.get('face', 0)} face)")
    rings = (st.get("rings") or {}).get("statistics") or {}
    if rings:
        sizes = ", ".join(f"{n} of size {size}" for size, n in (rings.get("by_size") or {}).items())
        md.append(f"- **Rings:** {rings.get('total')} ({sizes or 'none'})")
    return md + [""]


def _references_md(lit_d: dict) -> List[str]:
    """Render a 'References' section from a LiteratureResult dict."""
    refs = lit_d.get("references") or []
    if not refs:
        return []
    lines = ["## References", ""]
    for i, r in enumerate(refs, 1):
        authors = ", ".join((r.get("authors") or [])[:3])
        if len(r.get("authors") or []) > 3:
            authors += " et al."
        venue = f" *{r['container']}*" if r.get("container") else ""
        year = f" {r['year']}" if r.get("year") else ""
        doi = f" DOI: {r['doi']}" if r.get("doi") else ""
        cite = f" (cited by {r['citations']})" if r.get("citations") is not None else ""
        lines.append(f"{i}. {authors}. \"{r.get('title', '')}.\"{venue}{year}.{doi}{cite}")
    lines.append("")
    return lines


def build_report(
    cdft: Any = None,
    analyses: Any = (),
    stability: Any = None,
    benchmark: Any = None,
    *,
    literature: Any = None,
    structure: Any = None,
    title: str = "CADMIA report",
    meta: Optional[dict] = None,
    full: bool = False,
) -> Report:
    """Assemble a consolidated report from agent results (dicts, objects, or JSON paths).

    With ``full=True`` the document is framed as a complete CADMIA process report:
    a header (program, version, author, timestamp), a run-information / workflow-stages
    section (driven by ``meta``), the result sections, and a normal-termination footer.
    """
    struct_d = _load(structure)
    cdft_d = _load(cdft)
    analysis_ds = [d for d in (_load(a) for a in (analyses or [])) if d]
    stab_d = _load(stability)
    bench_d = _load(benchmark)
    lit_d = _load(literature)

    contributions = [name for name, present in (
        ("structure", struct_d), ("cdft", cdft_d), ("analysis", analysis_ds),
        ("stability", stab_d), ("benchmark", bench_d and bench_d.get("comparison")),
    ) if present]

    if full:
        meta = dict(meta or {})
        meta.setdefault("engine_keywords", _engine_keywords(cdft_d, analysis_ds))
        md: List[str] = _header_block(title, meta) + _run_info_block(meta, contributions)
    else:
        md = [f"# {title}", ""]
    tex: List[str] = []
    correlations: List[Correlation] = []
    warnings: List[str] = []

    cluster = (cdft_d or {}).get("cluster") or (stab_d or {}).get("mixed") or "system"
    md.append(f"**System:** {cluster}")
    md.append("")

    stage_no = MODULE_NUMBER

    def opens(stage: str) -> List[str]:
        return _stage_open(stage_no[stage], stage) if full else []

    if struct_d:
        md += opens("structure") + _structure_md(struct_d)

    # ── Global descriptors ──
    if cdft_d:
        md += opens("cdft")
        gd = cdft_d.get("global_descriptors", {})
        rows = [[label, _fmt(gd.get(key)), unit] for key, label, unit in _GLOBAL_ORDER
                if gd.get(key) is not None]
        if rows:
            md += ["## Global CDFT descriptors", "",
                   _md_table(["descriptor", "value", "unit"], rows), ""]
            tex.append(_latex_table(["Descriptor", "Value", "Unit"], rows,
                                    f"Global CDFT descriptors for {cluster}", "global"))

        # ── Site ranking ──
        sites = sorted(cdft_d.get("sites", []), key=lambda s: s.get("rank", 1e9))
        srows = [[str(s.get("rank")), f"{s.get('atom_symbol')}({s.get('atom_index')})",
                  _fmt(s.get("fukui_plus")), _fmt(s.get("fukui_minus")),
                  _fmt(s.get("delta_f"), signed=True), str(s.get("character")),
                  str(s.get("acid_base") or "—")]
                 for s in sites]
        if srows:
            md += ["## CDFT site ranking", "",
                   _md_table(["rank", "atom", "f+", "f-", "Δf", "character", "acid/base"], srows), ""]
            tex.append(_latex_table(["Rank", "Atom", "$f^+$", "$f^-$", "$\\Delta f$", "Character",
                                     "Acid/base"],
                                    srows, f"CDFT site ranking for {cluster}", "sites"))

        # ── Lewis acid/base sites (HSAB) ──
        ab = cdft_d.get("acid_base_summary") or {}
        acid, base = ab.get("strongest_acid_site"), ab.get("strongest_base_site")
        if acid or base:
            abrows = []
            if acid:
                abrows.append(["Lewis acid (e⁻ acceptor)",
                               f"{acid.get('atom_symbol')}({acid.get('atom_index')})",
                               _fmt(acid.get("strength")), _fmt(acid.get("delta_f"), signed=True)])
            if base:
                abrows.append(["Lewis base (e⁻ donor)",
                               f"{base.get('atom_symbol')}({base.get('atom_index')})",
                               _fmt(base.get("strength")), _fmt(base.get("delta_f"), signed=True)])
            md += ["## Lewis acid/base sites (HSAB)", "",
                   _md_table(["role", "strongest site", "strength (f±)", "Δf"], abrows), ""]
            if ab.get("acid_coordination_note"):
                md += [f"_{ab['acid_coordination_note']}_", ""]
            if ab.get("hsab_note"):
                md += [f"_{ab['hsab_note']}_", ""]

    # ── Adsorption summary + Δf vs E_ads correlation ──
    if analysis_ds:
        md += opens("analysis")
        analysis_ds, dup_map = _dedup_analyses(analysis_ds)
        n_dups = sum(len(v) for v in dup_map.values())
        if n_dups:
            warnings.append(
                f"{n_dups} adsorption configuration(s) relaxed to a minimum already present "
                f"and were collapsed; {len(analysis_ds)} unique adsorption minima remain."
            )
        df_by_idx = {s.get("atom_index"): s.get("delta_f")
                     for s in (cdft_d or {}).get("sites", [])}
        arows = []
        by_probe: dict = {}
        for a in analysis_ds:
            idx = a.get("site_atom_index")
            df = df_by_idx.get(idx)
            site = f"{a.get('site_atom_symbol')}({idx})" if idx is not None else "—"
            arows.append([
                _cfg_cell(a.get("config_index")), a.get("probe", "—"), site,
                _fmt(a.get("e_ads_kcal"), 2, signed=True),
                _fmt(a.get("e_ads_bsse_kcal"), 2, signed=True),
                _fmt(a.get("delta_h_ads_kcal"), 2, signed=True),
                _fmt(a.get("delta_g_ads_kcal"), 2, signed=True),
                _fmt(df, signed=True) if df is not None else "—",
                "yes" if a.get("activated") else "no",
            ])
            if df is not None and a.get("e_ads_kcal") is not None:
                # Kept PER PROBE: a CDFT ranking can correlate with opposite signs for
                # different probes, and a single r over the pool would average them away.
                by_probe.setdefault(a.get("probe") or "probe", ([], [], []))
                px, py, plab = by_probe[a.get("probe") or "probe"]
                px.append(df); py.append(a["e_ads_kcal"])
                plab.append(f"{site} c{a.get('config_index')}")
        md += ["## Adsorption summary", ""]
        if n_dups:
            md += [f"_{n_dups} configuration(s) collapsed to an existing minimum; "
                   f"{len(analysis_ds)} unique minima shown._", ""]
        md += [_md_table(["cfg", "probe", "site", "E_ads", "E_ads(BSSE)", "ΔH_ads", "ΔG_ads",
                          "Δf(site)", "activated"], arows), ""]
        tex.append(_latex_table(["Cfg", "Probe", "Site", "$E_{ads}$", "$E_{ads}^{BSSE}$",
                                 "$\\Delta H_{ads}$", "$\\Delta G_{ads}$", "$\\Delta f$",
                                 "Activated"], arows,
                                f"Adsorption summary for {cluster}", "ads"))
        reported_any = False
        for probe_name, (px, py, plab) in sorted(by_probe.items()):
            n_distinct = len({round(v, 6) for v in px})
            if len(px) < 3 or n_distinct < 2:
                # Two points give r = ±1 by construction, whatever they are. Printing that as a
                # correlation is the trap, not the small sample itself.
                warnings.append(
                    f"Δf vs E_ads for {probe_name}: {len(px)} configuration(s), "
                    f"{n_distinct} distinct Δf — too few for a correlation; not reported."
                )
                continue
            fit = linregress(px, py)
            corr = Correlation(name=f"df_vs_eads_{probe_name}", x_label="delta_f(site)",
                               y_label="E_ads(kcal/mol)", x=px, y=py, labels=plab,
                               pearson_r=(fit.r if fit else None))
            correlations.append(corr)
            if not reported_any:
                md += ["### Correlation: Δf(site) vs E_ads", "",
                       "One fit per probe: a descriptor can correlate with opposite signs for "
                       "different probes, and a pooled r would cancel them.", ""]
                reported_any = True
            p_text = f", p = {_fmt(fit.p_value, 3)}" if fit and fit.p_value is not None else ""
            md += [f"- **{probe_name}**: r = {_fmt(corr.pearson_r, 3)}{p_text} "
                   f"over {len(px)} configurations ({n_distinct} distinct Δf)."]
            if n_distinct < 8:
                md += ["  Fewer than 8 distinct Δf values: not conclusive, whatever r says."]
        if reported_any:
            md += [""]
        if not reported_any and not by_probe:
            warnings.append(
                "Δf vs E_ads correlation needs ≥2 unique adsorption minima at distinct CDFT "
                f"sites; only {len(analysis_ds)} unique minimum(a) available."
            )

        # ── Electronic properties: cluster (before) → complex (after) ──
        ecl = next((a.get("electronic_cluster") for a in analysis_ds
                    if a.get("electronic_cluster")), None)
        complexes = [(a.get("config_index"), a.get("electronic_adduct"))
                     for a in analysis_ds if a.get("electronic_adduct")]
        if ecl and complexes:
            props = [("homo_ev", "HOMO"), ("lumo_ev", "LUMO"), ("gap_ev", "Gap"),
                     ("mu_ev", "μ"), ("eta_ev", "η(K)"), ("omega_ev", "ω"),
                     ("omega_minus_ev", "ω⁻"), ("omega_plus_ev", "ω⁺")]
            headers = ["property (eV)", "cluster"] + [_cplx_label(i) for i, _ in complexes]
            erows = []
            for key, label in props:
                vals = [ecl.get(key)] + [ed.get(key) for _, ed in complexes]
                if all(v is None for v in vals):
                    continue
                erows.append([label, _fmt(ecl.get(key))]
                             + [_fmt(ed.get(key)) for _, ed in complexes])
            if erows:
                md += ["## Electronic properties (cluster → adduct)", "",
                       _md_table(headers, erows), "",
                       "η(K) = LUMO − HOMO is the Koopmans estimate of the same hardness as the "
                       "finite-difference η(FD) = IP − EA in the CDFT global descriptors (both "
                       "I − A, no ½); they differ only in method (frontier orbitals vs ΔSCF).", ""]
                if len(complexes) == 1 and ecl.get("gap_ev") is not None \
                        and complexes[0][1].get("gap_ev") is not None:
                    dgap = complexes[0][1]["gap_ev"] - ecl["gap_ev"]
                    md += [f"Δgap on adsorption = **{dgap:+.3f} eV**.", ""]
                tex.append(_latex_table(headers, erows,
                                        f"Electronic properties (cluster vs adduct) for {cluster}", "elec"))

            # Dipole magnitude (Debye) — separate table to keep units unambiguous.
            dcl = ecl.get("dipole_debye")
            dpls = [(i, ed.get("dipole_debye")) for i, ed in complexes]
            if dcl is not None and any(d is not None for _, d in dpls):
                dhdr = ["dipole (D)", "cluster"] + [_cplx_label(i) for i, _ in dpls]
                drow = ["|μ|", _fmt(dcl, 3)] + [_fmt(d, 3) for _, d in dpls]
                md += [_md_table(dhdr, [drow]), ""]
                if len(dpls) == 1 and dpls[0][1] is not None:
                    md += [f"Δ|μ| on adsorption = **{dpls[0][1] - dcl:+.3f} D**.", ""]

        # ── Strongest Mayer bond orders at the binding site (complex) ──
        mrows = []
        for a in analysis_ds:
            bonds = a.get("site_mayer_bonds") or []
            if not bonds:
                continue
            idx = a.get("site_atom_index")
            site = f"{a.get('site_atom_symbol')}({idx})" if idx is not None else "—"
            for b in bonds:
                mrows.append([_cfg_cell(a.get("config_index")), site,
                              f"{b.get('partner_symbol')}({b.get('partner_index')})",
                              _fmt(b.get("order"))])
        if mrows:
            md += ["## Mayer bond orders at the binding site (adduct)", "",
                   _md_table(["cfg", "site", "partner", "Mayer BO"], mrows), ""]
            tex.append(_latex_table(["Cfg", "Site", "Partner", "Mayer BO"], mrows,
                                    f"Mayer bond orders at the binding site for {cluster}", "mayer"))

        # ── Structural coordination of the binding site: cluster → complex ──
        srows = []
        for a in analysis_ds:
            if a.get("site_cn_cluster") is None and a.get("site_ecn_cluster") is None:
                continue
            idx = a.get("site_atom_index")
            site = f"{a.get('site_atom_symbol')}({idx})" if idx is not None else "—"
            srows.append([
                _cfg_cell(a.get("config_index")), site,
                _fmt(a.get("site_cn_cluster"), 0), _fmt(a.get("site_cn_adduct"), 0),
                _fmt(a.get("delta_cn_site"), 0, signed=True),
                _fmt(a.get("site_ecn_cluster"), 2), _fmt(a.get("site_ecn_adduct"), 2),
                _fmt(a.get("delta_ecn_site"), 2, signed=True),
            ])
        if srows:
            hdr = ["cfg", "site", "CN(clu)", "CN(ad)", "ΔCN", "ECN(clu)", "ECN(ad)", "ΔECN"]
            md += ["## Binding-site coordination (cluster → adduct)", "",
                   _md_table(hdr, srows), ""]
            tex.append(_latex_table(["Cfg", "Site", "CN$_{clu}$", "CN$_{ad}$", "$\\Delta$CN",
                                     "ECN$_{clu}$", "ECN$_{ad}$", "$\\Delta$ECN"], srows,
                                    f"Binding-site coordination for {cluster}", "coord"))

        # ── Charge transfer predicted by conceptual DFT (cluster ↔ probe) ──
        # One row per PROBE. Taking the first analysis published the ΔN of whichever probe
        # happened to come first under a heading that does not name it, and ΔN is a property of
        # the pair: the same cluster donates to one probe and accepts from another.
        ct_by_probe: dict = {}
        for a in analysis_ds:
            item = a.get("charge_transfer")
            if item and item.get("delta_n") is not None:
                ct_by_probe.setdefault(a.get("probe") or "probe", item)
        if ct_by_probe:
            ctrows = [[
                probe_name,
                _fmt(item.get("mu_cluster_ev"), 3), _fmt(item.get("eta_cluster_ev"), 3),
                _fmt(item.get("mu_probe_ev"), 3), _fmt(item.get("eta_probe_ev"), 3),
                _fmt(item["delta_n"], 4, signed=True), _fmt(item.get("e_stab_ev"), 4),
                f"{item.get('donor')} → {item.get('acceptor')}",
            ] for probe_name, item in ct_by_probe.items()]
            ctrow = ctrows[0]
            cthdr = ["probe", "μ(clu)", "η(clu)", "μ(pr)", "η(pr)", "ΔN", "ΔE (eV)", "flow"]
            md += ["## Charge transfer predicted by conceptual DFT (cluster ↔ probe)", "",
                   "Parr–Pearson (one parabola) from the Koopmans μ/η of the isolated cluster and the "
                   "free probe. ΔN > 0 means the **cluster donates** electrons to the probe; ΔE is the "
                   "second-order stabilization (all values in eV, η in the I − A convention). This is the "
                   "*predicted* flow before contact — compare with the *observed* Δq(probe) below.", "",
                   _md_table(cthdr, ctrows), ""]
            tex.append(_latex_table(
                ["Probe", "$\\mu_{clu}$", "$\\eta_{clu}$", "$\\mu_{pr}$", "$\\eta_{pr}$",
                 "$\\Delta N$", "$\\Delta E$ (eV)", "Flow"],
                ctrows, f"Conceptual-DFT charge transfer (cluster $\\leftrightarrow$ probe) for {cluster}",
                "ct"))

        # ── Charge transfer on adsorption ──
        qrows = []
        qscheme = None
        for a in analysis_ds:
            if a.get("q_transfer_to_probe") is None and a.get("delta_q_site") is None:
                continue
            qscheme = qscheme or a.get("charge_scheme")
            idx = a.get("site_atom_index")
            site = f"{a.get('site_atom_symbol')}({idx})" if idx is not None else "—"
            qrows.append([_cfg_cell(a.get("config_index")), site,
                          _fmt(a.get("q_transfer_to_probe"), 3, signed=True),
                          _fmt(a.get("delta_q_site"), 3, signed=True)])
        if qrows:
            scheme_note = f" ({qscheme} charges)" if qscheme else ""
            md += [f"## Charge transfer on adsorption{scheme_note}", "",
                   "Δq(probe) < 0 ⇒ the cluster donates electrons to the probe.", "",
                   _md_table(["cfg", "site", "Δq(probe)", "Δq(site)"], qrows), ""]
            tex.append(_latex_table(["Cfg", "Site", "$\\Delta q_{probe}$", "$\\Delta q_{site}$"],
                                    qrows, f"Charge transfer on adsorption for {cluster}", "charge"))

        # ── Vibrational signature of the complex ──
        vrows = []
        for a in analysis_ds:
            bands = a.get("adduct_ir_bands") or []
            if a.get("adduct_imaginary_modes") is None and not bands:
                continue
            imag = a.get("adduct_imaginary_modes")
            minimum = "—" if imag is None else ("yes" if imag == 0 else f"no ({imag} imag.)")
            top = "; ".join(f"{b.get('frequency_cm1')} ({b.get('intensity_km_mol')})"
                            for b in bands[:3]) or "—"
            vrows.append([_cfg_cell(a.get("config_index")), minimum, top])
        if vrows:
            md += ["## Vibrational signature of the adduct", "",
                   "Top IR bands as cm⁻¹ (km/mol).", "",
                   _md_table(["cfg", "minimum?", "top IR bands"], vrows), ""]

    # ── Density of states (linked artifacts) ──
    dos_files = (meta or {}).get("dos_files") or []
    pdos_files = (meta or {}).get("pdos_files") or []
    if dos_files or pdos_files:
        md += ["## Density of states", ""]
        for p in dos_files:
            md.append(f"- DOS: [{Path(p).name}]({p})")
        for p in pdos_files:
            md.append(f"- PDOS: [{Path(p).name}]({p})")
        md.append("")

    # ── Stability ──
    if stab_d:
        md += opens("stability")
        refs = " + ".join(f"{r.get('coefficient')}·{r.get('label')}" for r in stab_d.get("references", []))
        bal = "balanced" if stab_d.get("balanced") else "UNBALANCED"
        md += ["## Relative stability", "",
               f"- E_rel({stab_d.get('energy_basis') or 'E'}) = "
               f"**{_fmt(stab_d.get('e_rel_kcal'), 3, signed=True)} kcal/mol** "
               f"({_fmt(stab_d.get('e_rel_eV_per_atom'), 4, signed=True)} eV/atom)",
               f"- Reference: {stab_d.get('formula')} = {refs} ({bal})", ""]


    # ── Benchmark ──
    if bench_d and bench_d.get("comparison"):
        md += opens("benchmark")
        c = bench_d["comparison"]
        md += ["## Engine benchmark (Gaussian vs ORCA)", "",
               _md_table(["ΔE (kcal/mol)", "RMSD (Å)", "timing ORCA/Gaussian", "recommendation"],
                         [[_fmt(c.get("delta_e_kcal"), 3), _fmt(c.get("rmsd_ang"), 4),
                           _fmt(c.get("timing_ratio"), 2), str(c.get("recommendation"))]]), ""]

    # ── References (Literature Agent) ──
    if lit_d:
        md += _references_md(lit_d)

    # The Markdown is the deliverable; the CLI print is not. A correlation that was withheld or
    # configurations collapsed as duplicates belong in the document, next to the numbers they
    # qualify.
    if warnings:
        md += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    if full:
        md += [""] + _footer_block(contributions, len(warnings))

    return Report(title=title, markdown="\n".join(md).rstrip() + "\n",
                  latex="\n\n".join(tex) + "\n" if tex else "",
                  correlations=correlations, warnings=warnings)


# ─────────────────────────────────────────────
# The output: every stage's full report, in one fixed-width text file
# ─────────────────────────────────────────────

_NOT_STORED = (" This module's JSON holds no full report: it was written outside a CADMIA 0.3.1 run of\n"
               " the module (an older version, or by hand). Run the module again to include it.")


def _stage_text(stage: str, d: dict) -> str:
    if d.get("report_text"):
        return d["report_text"].rstrip()
    from cadmia.agents.text_report import analysis_report_text, stability_report_text
    if stage == "analysis":
        return analysis_report_text(d)
    if stage == "stability":
        return stability_report_text(d)
    return _NOT_STORED


def _box(lines: List[str]) -> List[str]:
    inner = _BOX_WIDTH - 2
    return (["╔" + "═" * inner + "╗"] + [f"║{line:<{inner}}║" for line in lines]
            + ["╚" + "═" * inner + "╝"])


def cluster_mismatches(stages: dict) -> Tuple[List[str], List[str]]:
    """Saved results that are not about the same cluster, by chemical formula.

    Returns ``(conflicts, others)``: CDFT, adsorption and stability results whose cluster
    differs from the first one's (they cannot share one output), and structure results of
    other formulas (a series or the references, worth a mention but not wrong). Two isomers
    with the same formula are not told apart.
    """
    from cadmia.core.elements import hill_formula
    found = []
    for stage, items in stages.items():
        for d in items:
            if stage == "cdft":
                formula = (d.get("structure") or {}).get("formula")
            elif stage == "analysis":
                formula = hill_formula(d["cluster_atoms"]) if d.get("cluster_atoms") else None
            elif stage == "stability":
                formula = d.get("formula")
            else:
                continue
            if formula:
                found.append((formula, d.get("_file") or stage))
    cluster = found[0][0] if found else None
    conflicts = [f"{name}: {formula}, not {cluster}" for formula, name in found if formula != cluster]
    others = [f"{d.get('_file') or 'structure'}: {(d.get('structure') or {}).get('formula')}"
              for d in stages.get("structure", [])
              if cluster and (d.get("structure") or {}).get("formula") not in (None, cluster)]
    return conflicts, others


def build_output(
    stages: Dict[str, List[dict]],
    *,
    meta: Optional[dict] = None,
    summary: Optional[Report] = None,
) -> str:
    """The CADMIA output: header, every stage's full text report under its own frame, the
    cross-stage results (correlations, warnings) and the closing block.

    *stages* maps a stage name to its result dicts, as `cadmia.io.results.load_stage_results`
    returns them. *summary* is the Markdown report of the same data, whose correlations and
    warnings are the cross-stage part.
    """
    from cadmia import __version__
    from cadmia.citation import cite_block, quote_block
    meta = dict(meta or {})
    present = [st for st in ("structure", "cdft", "analysis", "stability") if stages.get(st)]

    out = [_BANNER, "", cite_block(), "",
           f" CADMIA output — {meta.get('title') or 'CADMIA report'}",
           " Computational Agent-Driven Molecular Intelligence Architecture",
           f" Version {__version__} · Pablo Daniel Álvarez Hernández (UJAT/CICTAT)",
           f" Generated: {meta.get('generated') or _now_iso()}", ""]

    out += [" Run information", " " + "=" * (_BOX_WIDTH - 1)]
    lot = _level_of_theory(meta)
    keywords = _engine_keywords((stages.get("cdft") or [None])[0], stages.get("analysis") or [],
                                (stages.get("stability") or [None])[0], stages.get("structure") or [])
    if lot:
        out.append(f" Level of theory : {lot}")
    elif keywords:
        out.append(" Level of theory (keywords echoed by the engine):")
        for stage, src, kw in keywords:
            out += [f"   {stage:<11}{Path(src).name}", f"              {kw}"]
    if meta.get("command"):
        out.append(f" Command         : {meta['command']}")
    from cadmia.io.results import PATHS_NOTE, report_paths_mode
    mode = report_paths_mode()
    if meta.get("root"):
        root = meta["root"] if mode == "absolute" else Path(meta["root"]).name
        out.append(f" Project         : {root}")
    out += textwrap.wrap(f"Paths           : {PATHS_NOTE[mode]}", _BOX_WIDTH,
                         initial_indent=" ", subsequent_indent=" " * 19)
    out += ["", " Modules reported:"]
    out += [f"   [x] {i}. {_STAGE_SCOPE[st]}" for i, st in enumerate(present, 1)]
    out.append("")

    for stage in present:
        items = stages[stage]
        out += _box([f"  MODULE {MODULE_NUMBER[stage]}  ·  {_STAGE_NAMES[stage].upper()}", f"  {_STAGE_SCOPE[stage]}"])
        for j, d in enumerate(items, 1):
            if len(items) > 1:
                label = Path(d.get("_file") or d.get("adduct_path") or f"item {j}").stem
                out += ["", f" ── {_STAGE_NAMES[stage]} {j} of {len(items)}: {label} ".ljust(_BOX_WIDTH, "─")]
            out += [_stage_text(stage, d), ""]
        out.append("")

    warnings = list(summary.warnings) if summary else []
    correlations = summary.correlations if summary else []
    if correlations or warnings:
        out += _box(["  ACROSS MODULES", "  Correlations and checks that combine several modules"])
        for c in correlations:
            r = f"{c.pearson_r:.3f}" if c.pearson_r is not None else "n/a"
            out.append(f" Correlation {c.name}: Pearson r = {r} ({len(c.x)} points)")
        for w in warnings:
            out += textwrap.wrap(w, _BOX_WIDTH - 4, initial_indent=" ! ", subsequent_indent="   ")
        out.append("")
    if meta.get("narration"):
        out += _box(["  DISCUSSION", "  AI-drafted, experimental: verify every statement"]) \
            + [meta["narration"].strip(), ""]

    n_warn = len(warnings) + sum(len(d.get("warnings") or []) for st in present for d in stages[st])
    import platform
    from cadmia.core.provenance import software_version
    out += [" " + "-" * (_BOX_WIDTH - 1),
            f" Modules reported: {', '.join(_STAGE_NAMES[s] for s in present) or 'none'}"
            f" · warnings: {n_warn}",
            f" {software_version()} · Python {platform.python_version()} · {platform.system()}",
            "", quote_block(), "",
            f" Normal termination of CADMIA {__version__} — {_now_iso()}"]
    from cadmia.io.results import fit_width
    return fit_width("\n".join(out)) + "\n"
