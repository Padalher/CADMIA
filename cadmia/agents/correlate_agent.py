"""Correlate Agent — regress computed CDFT/adsorption descriptors against experimental data.

Terminal step of a study: given the JSON artifacts already produced by ``cadmia cdft`` and
``cadmia analyze`` for several systems, and a user-supplied table of experimental observables
(log k, Mayr N/E, TOF, ΔH_ads…), it fits a least-squares line descriptor → observable and reports
slope, intercept, r, r² and a two-tailed p-value. In *scan* mode it ranks every available
descriptor against one observable by |r|.

Descriptors are addressed by short aliases (``omega_minus``, ``delta_n``, ``dual@site``…) so the
user never has to know the JSON layout.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cadmia.core.stats import LinFit, linregress

# Alias → candidate field names, searched (in order) inside the CDFT global blocks and, as a
# fallback, the adsorption cluster block. These are scalar, one value per system.
_CDFT_SCALARS: Dict[str, Tuple[str, ...]] = {
    "mu": ("mu_ev",),
    "eta": ("hardness_ev", "eta_ev"),
    "omega": ("omega_ev",),
    "omega_minus": ("omega_minus_ev",),
    "omega_plus": ("omega_plus_ev",),
    "N": ("nucleophilicity_ev",),
    "ip": ("ip_ev",),
    "ea": ("ea_ev",),
    "gap": ("gap_ev",),
}
# Alias → field at the top level of the analysis JSON.
_ANALYSIS_SCALARS: Dict[str, str] = {
    "e_ads": "e_ads_kcal",
    "dG_ads": "delta_g_ads_kcal",
    "dH_ads": "delta_h_ads_kcal",
    "delta_gap": "delta_gap_ev",
    "dq_probe": "q_transfer_to_probe",
    "dq_site": "delta_q_site",
    "dr_site": "delta_r_site_ang",
}
# Alias → field inside the analysis "charge_transfer" block.
_CT_SCALARS: Dict[str, str] = {"delta_n": "delta_n", "e_stab": "e_stab_ev"}
# Atom-resolved Fukui aliases (need "name@atom", atom = 1-based index or the token "site").
_FUKUI_BASES = {"f_plus", "f_minus", "f_zero", "dual",
                # nuclear response (present only when the CDFT report was built from .engrad)
                "phi_norm", "phi_along", "phi_dual_norm"}

_KIND_SUFFIXES = (".analysis", ".cdft", ".report", ".fd", ".koopmans")


@dataclass
class CorrelatePair:
    """One descriptor → observable regression."""

    descriptor: str
    observable: str
    n: int
    slope: Optional[float]
    intercept: Optional[float]
    r: Optional[float]
    r2: Optional[float]
    p_value: Optional[float]
    x_transform: str
    y_transform: str
    systems: List[str] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    note: str = ""

    def equation(self) -> str:
        if self.slope is None:
            return f"{self.observable} = (undetermined)  (n={self.n})"
        xs = _wrap_transform(self.descriptor, self.x_transform)
        ys = _wrap_transform(self.observable, self.y_transform)
        r2 = f"{self.r2:.4f}" if self.r2 is not None else "n/a"
        return (f"{ys} = {self.slope:.4g}·{xs} {'+' if self.intercept >= 0 else '−'} "
                f"{abs(self.intercept):.4g}   (R² = {r2}, n = {self.n})")


@dataclass
class CorrelateResult:
    n_systems_matched: int
    pairs: List[CorrelatePair] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ranked(self) -> List[CorrelatePair]:
        """Pairs sorted by |r| descending (undefined fits last)."""
        return sorted(self.pairs, key=lambda p: (abs(p.r) if p.r is not None else -1.0), reverse=True)

    def to_dict(self) -> dict:
        return {"n_systems_matched": self.n_systems_matched,
                "pairs": [asdict(p) for p in self.ranked],
                "warnings": self.warnings}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ── loading ──

def _system_id(path: Path, data: dict) -> str:
    if isinstance(data, dict) and data.get("system"):
        return str(data["system"])
    stem = path.stem  # "cluster_co2.analysis.json" → "cluster_co2.analysis"
    for suf in _KIND_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _expand_json(sources: List[str]) -> List[Path]:
    out: List[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.exists():
            out.append(p)
        else:
            out.extend(Path(m) for m in sorted(glob.glob(src)) if m.lower().endswith(".json"))
    # de-dup preserving order
    seen, uniq = set(), []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def load_descriptor_systems(sources: List[str]) -> Tuple[Dict[str, dict], List[str]]:
    """Merge every JSON artifact into one descriptor namespace per system id."""
    systems: Dict[str, dict] = {}
    warnings: List[str] = []
    for path in _expand_json(sources):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"Skipped {path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            warnings.append(f"Skipped {path.name}: not a JSON object.")
            continue
        sid = _system_id(path, data)
        systems.setdefault(sid, {}).update(data)   # shallow merge; cdft/analysis keys don't collide
    return systems, warnings


def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def load_experimental_table(path: str | Path, system_key: str = "system") -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Read a CSV table → {system: {observable: value}}, plus the ordered observable columns."""
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError("Experimental table is empty.")
    fields = list(rows[0].keys())
    key = next((f for f in fields if f.strip().lower() == system_key.lower()), None)
    if key is None:
        raise ValueError(f"Experimental table has no '{system_key}' column (found: {', '.join(fields)}).")
    observables = [f for f in fields if f != key]
    table: Dict[str, Dict[str, float]] = {}
    for row in rows:
        sid = (row.get(key) or "").strip()
        if not sid:
            continue
        table[sid] = {obs: _to_float(row.get(obs)) for obs in observables}
    return table, observables


# ── descriptor resolution ──

def _search(blocks: List[Optional[dict]], names: Tuple[str, ...]) -> Optional[float]:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for name in names:
            if name in block:
                v = _to_float(block[name])
                if v is not None:
                    return v
    return None


def resolve_descriptor(alias: str, merged: dict) -> Optional[float]:
    """Resolve a descriptor alias to a scalar for one system, or None if unavailable."""
    base, _, atom_tok = alias.partition("@")
    if atom_tok:                       # atom-resolved Fukui: name@site / name@N
        if base not in _FUKUI_BASES:
            return None
        atoms = merged.get("atoms")
        if not isinstance(atoms, list):
            return None
        if atom_tok == "site":
            idx = merged.get("site_atom_index")
        else:
            try:
                idx = int(atom_tok)
            except ValueError:
                return None
        if idx is None:
            return None
        entry = next((a for a in atoms if a.get("index") == idx), None)
        return _to_float(entry.get(base)) if entry else None

    if base in _CDFT_SCALARS:
        return _search([merged.get("global_fd"), merged.get("global_koopmans"),
                        merged.get("electronic_cluster")], _CDFT_SCALARS[base])
    if base in _ANALYSIS_SCALARS:
        return _to_float(merged.get(_ANALYSIS_SCALARS[base]))
    if base in _CT_SCALARS:
        ct = merged.get("charge_transfer")
        return _to_float(ct.get(_CT_SCALARS[base])) if isinstance(ct, dict) else None
    return None


def available_descriptors(merged: dict) -> List[str]:
    """Scalar aliases (not atom-resolved) that resolve to a value for this system."""
    aliases = list(_CDFT_SCALARS) + list(_ANALYSIS_SCALARS) + list(_CT_SCALARS)
    return [a for a in aliases if resolve_descriptor(a, merged) is not None]


# ── transforms + core ──

def _apply(value: float, transform: str) -> Optional[float]:
    if transform == "abs":
        return abs(value)
    if transform == "log":
        return math.log(value) if value > 0 else None
    return value


def _wrap_transform(name: str, transform: str) -> str:
    if transform == "abs":
        return f"|{name}|"
    if transform == "log":
        return f"ln({name})"
    return name


def _regress_pair(descriptor: str, observable: str,
                  systems: Dict[str, dict], table: Dict[str, Dict[str, float]],
                  x_transform: str, y_transform: str) -> CorrelatePair:
    labels, xs, ys = [], [], []
    for sid, merged in systems.items():
        if sid not in table:
            continue
        raw_x = resolve_descriptor(descriptor, merged)
        raw_y = table[sid].get(observable)
        if raw_x is None or raw_y is None:
            continue
        tx = _apply(raw_x, x_transform)
        ty = _apply(raw_y, y_transform)
        if tx is None or ty is None:
            continue
        labels.append(sid)
        xs.append(tx)
        ys.append(ty)

    fit: Optional[LinFit] = linregress(xs, ys) if len(xs) >= 2 else None
    note = ""
    if len(xs) < 3:
        note = "n<3: not conclusive" if len(xs) >= 2 else "n<2: no fit"
    elif fit is None:
        note = "constant descriptor: fit undefined"
    return CorrelatePair(
        descriptor=descriptor, observable=observable, n=len(xs),
        slope=fit.slope if fit else None, intercept=fit.intercept if fit else None,
        r=fit.r if fit else None, r2=fit.r2 if fit else None,
        p_value=fit.p_value if fit else None,
        x_transform=x_transform, y_transform=y_transform,
        systems=labels, x=xs, y=ys, note=note,
    )


# Cluster-level CDFT blocks that apply to every adsorption config of that cluster.
_CDFT_BROADCAST_KEYS = ("atoms", "global_fd", "global_koopmans", "electronic_cluster")


def _broadcast_cdft(systems: Dict[str, dict], cdft_path: str) -> List[str]:
    """Merge a cluster-level CDFT report into every system.

    The per-atom Fukui (``atoms``) and global blocks describe the bare cluster, so they apply
    to all of its adsorption configs. Non-destructive: never overwrites a key a system already
    carries (per-config fields like ``site_atom_index`` win).
    """
    name = Path(cdft_path).name
    try:
        data = json.loads(Path(cdft_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Skipped --cdft {name}: {exc}"]
    if not isinstance(data, dict):
        return [f"Skipped --cdft {name}: not a JSON object."]
    shared = {k: data[k] for k in _CDFT_BROADCAST_KEYS if k in data}
    if not shared:
        return [f"--cdft {name}: no CDFT descriptor blocks "
                f"({', '.join(_CDFT_BROADCAST_KEYS)}) found — is it a `--report-format json` CDFT?"]
    for merged in systems.values():
        for key, value in shared.items():
            merged.setdefault(key, value)
    return []


def correlate(
    experimental: Optional[str | Path] = None,
    descriptor_sources: Optional[List[str]] = None,
    *,
    x: str = "all",
    y: Optional[str] = None,
    x_transform: str = "none",
    y_transform: str = "none",
    system_key: str = "system",
    cdft: Optional[str] = None,
) -> CorrelateResult:
    """Regress descriptor(s) against an observable.

    ``x`` is a descriptor alias or ``"all"`` (scan every available scalar descriptor).
    ``y`` is the observable. With ``experimental`` it is a column of that CSV table; without it,
    ``y`` must be a descriptor alias (e.g. ``dG_ads``) read straight from the JSON artifacts — so a
    theory↔theory correlation needs no hand-written table. ``cdft`` broadcasts one cluster-level
    CDFT report over every config, enabling atom-resolved descriptors like ``dual@site``.
    """
    systems, warnings = load_descriptor_systems(descriptor_sources or [])
    if cdft:
        warnings.extend(_broadcast_cdft(systems, cdft))

    if experimental is not None:
        table, observables = load_experimental_table(experimental, system_key=system_key)
    else:
        if not y:
            raise ValueError(
                "Without -e/--experimental you must set --y to a descriptor alias "
                "(e.g. --y dG_ads) so the observable is read from the JSON artifacts.")
        table = {}
        for sid, merged in systems.items():
            val = resolve_descriptor(y, merged)
            if val is not None:
                table[sid] = {y: val}
        observables = [y]
        if not table:
            warnings.append(f"Observable descriptor '{y}' did not resolve for any system.")

    matched = sorted(set(systems) & set(table))
    if not matched:
        warnings.append("No systems matched between the experimental table and the descriptor files.")
    unmatched_exp = sorted(set(table) - set(systems))
    unmatched_desc = sorted(set(systems) - set(table))
    if unmatched_exp:
        warnings.append(f"In table but no descriptors: {', '.join(unmatched_exp)}.")
    if unmatched_desc:
        warnings.append(f"Have descriptors but not in table: {', '.join(unmatched_desc)}.")

    # Which observables?
    if y is not None:
        if y not in observables:
            raise ValueError(f"Observable '{y}' not in table (available: {', '.join(observables)}).")
        y_cols = [y]
    elif len(observables) == 1:
        y_cols = observables
    else:
        raise ValueError(
            f"Table has several observables ({', '.join(observables)}); choose one with --y.")

    # Which descriptors?
    if x == "all":
        avail: List[str] = []
        for sid in matched:
            for a in available_descriptors(systems[sid]):
                if a not in avail:
                    avail.append(a)
        x_aliases = [a for a in avail if a != y]   # skip the trivial y↔y self-fit
        if not x_aliases:
            warnings.append("No scalar descriptors available for the matched systems.")
    else:
        x_aliases = [x]

    pairs = [
        _regress_pair(descr, obs, systems, table, x_transform, y_transform)
        for obs in y_cols for descr in x_aliases
    ]
    return CorrelateResult(n_systems_matched=len(matched), pairs=pairs, warnings=warnings)
