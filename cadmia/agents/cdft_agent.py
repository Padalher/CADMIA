"""CDFT Agent — reactive site ranking and JSON contract from N/N±1 SCF outputs."""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cadmia.cdft.acid_base import classify_acid_base, scaled_threshold, summarize as acid_base_summary
from cadmia.cdft.workflow import CDFTInputState, CDFTWorkflowResult, run_cdft_workflow

# |Δf| below this threshold → ambiphilic classification
_AMPHIPHILIC_THRESHOLD = 0.02


@dataclass
class SiteDescriptor:
    atom_index: int
    atom_symbol: str
    coordinates: Tuple[float, float, float]
    fukui_plus: Optional[float]
    fukui_minus: Optional[float]
    delta_f: Optional[float]
    character: str   # "electrophilic" | "nucleophilic" | "ambiphilic" | "unknown"
    rank: int
    acid_base: str = ""   # "lewis_acid" | "lewis_base" | "amphoteric" | "unknown"
    cn: Optional[int] = None      # coordination number (None if structure analysis unavailable)
    ecn: Optional[float] = None   # Hoppe effective coordination number


@dataclass
class CDFTResult:
    cluster: str
    calculation_id: str
    charge_scheme: str
    sites: List[SiteDescriptor]
    global_descriptors: Dict
    sources: Dict[str, str]
    acid_base: Dict = field(default_factory=dict)
    nuclear: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_dict(self) -> dict:
        payload = {
            "cluster": self.cluster,
            "calculation_id": self.calculation_id,
            "charge_scheme": self.charge_scheme,
            "sites": [
                {
                    "atom_index": s.atom_index,
                    "atom_symbol": s.atom_symbol,
                    "coordinates_ang": [round(c, 6) for c in s.coordinates],
                    "fukui_plus": _r(s.fukui_plus),
                    "fukui_minus": _r(s.fukui_minus),
                    "delta_f": _r(s.delta_f),
                    "character": s.character,
                    "acid_base": s.acid_base,
                    "cn": s.cn,
                    "ecn": _r(s.ecn, 4),
                    "rank": s.rank,
                }
                for s in self.sites
            ],
            "global_descriptors": self.global_descriptors,
            "acid_base_summary": self.acid_base,
            "sources": self.sources,
            "warnings": self.warnings,
        }
        if self.nuclear:
            payload["nuclear"] = self.nuclear
        return payload


def _r(v: Optional[float], n: int = 6) -> Optional[float]:
    return round(v, n) if v is not None else None


def _detect_engine(filepath: Path) -> str:
    """Detect 'gaussian' or 'orca' from file content.

    Only the first 16 KB are read, so the ASCII banner 'O   R   C   A' is what usually
    identifies ORCA; 'FINAL SINGLE POINT ENERGY' and 'ORBITAL ENERGIES' cover short outputs.
    Gaussian: 'Entering Gaussian System', 'Gaussian, Inc.' or 'This is part of the Gaussian'.
    """
    _ORCA_MARKERS = (
        "FINAL SINGLE POINT ENERGY",
        "ORBITAL ENERGIES",
        "O   R   C   A",
    )
    _GAUSSIAN_MARKERS = (
        "Entering Gaussian System",
        "Gaussian, Inc.",
        "This is part of the Gaussian",
    )
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(16384)
    except OSError:
        return "gaussian"

    upper = content.upper()
    for marker in _ORCA_MARKERS:
        if marker.upper() in upper:
            return "orca"
    for marker in _GAUSSIAN_MARKERS:
        if marker in content:
            return "gaussian"
    # No ORCA marker found: Gaussian.
    return "gaussian"


def _file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.resolve()).encode())
        try:
            h.update(p.read_bytes()[:4096])
        except OSError:
            pass
    return h.hexdigest()[:16]


def _classify(delta_f: Optional[float], threshold: float = _AMPHIPHILIC_THRESHOLD) -> str:
    if delta_f is None:
        return "unknown"
    if abs(delta_f) < threshold:
        return "ambiphilic"
    return "electrophilic" if delta_f > 0 else "nucleophilic"


def _coordination_lookup(structure) -> Dict[int, Dict]:
    """Per-atom CN/ECN keyed by 0-based index; empty if structure analysis fails."""
    try:
        from cadmia.structure.structural.summary import structural_summary
        summary = structural_summary(structure)
    except Exception:
        return {}
    return {a["index"]: a for a in summary.get("atoms", [])}


def _build_sites(result: CDFTWorkflowResult) -> List[SiteDescriptor]:
    fukui = result.fukui
    structure = result.structure
    coord = _coordination_lookup(structure)
    # Scaled to the system's mean Fukui per atom: f sums to 1 over N atoms, so a fixed cut is
    # 16 % of the mean in an 8-atom cluster and 64 % in a 32-atom one, and the fraction of sites
    # called "ambiphilic" would grow with size for no chemical reason.
    threshold = scaled_threshold(len(fukui.atoms))

    sites = []
    for i, (atom, pos) in enumerate(zip(fukui.atoms, structure.positions)):
        f_plus  = fukui.f_plus[i]  if fukui.f_plus  else None
        f_minus = fukui.f_minus[i] if fukui.f_minus else None
        delta_f = fukui.dual[i]    if fukui.dual    else None
        c = coord.get(i, {})
        sites.append(SiteDescriptor(
            atom_index=i + 1,
            atom_symbol=atom,
            coordinates=tuple(pos.tolist()),
            fukui_plus=f_plus,
            fukui_minus=f_minus,
            delta_f=delta_f,
            character=_classify(delta_f, threshold),
            acid_base=classify_acid_base(delta_f, threshold,
                                         fukui_plus=f_plus, fukui_minus=f_minus),
            cn=c.get("cn"),
            ecn=c.get("ecn"),
            rank=0,
        ))

    # Rank by |Δf| (dual descriptor magnitude), not filtered by electrophilic/nucleophilic
    # character: this is a screening pass that surfaces the most *reactive* sites regardless of
    # sign, and the geometry agent then places the probe on the top n for a DFT relaxation that
    # decides direction. Character is carried per site (`character`/`acid_base`) for the caller
    # that wants to match a probe's Lewis role. This is deliberate.
    sites.sort(key=lambda s: abs(s.delta_f) if s.delta_f is not None else 0.0, reverse=True)
    for rank, site in enumerate(sites, start=1):
        site.rank = rank
    return sites


def _build_global(result: CDFTWorkflowResult) -> dict:
    fd = result.global_fd
    homo = fd.homo_n_ev
    lumo = fd.lumo_n_ev
    gap = _r(lumo - homo) if (homo is not None and lumo is not None) else None
    state_n = (result.states or {}).get("N")
    electronic = state_n.electronic if state_n is not None else None
    spin = {}
    if (electronic is not None and electronic.homo_beta_ev is not None
            and (electronic.structure.multiplicity or 1) > 1):
        # Open shell: gap_eV joins two spin channels (max HOMO, min LUMO); these keep them apart.
        spin = {"gap_alpha_eV": _r(electronic.gap_alpha_ev),
                "gap_beta_eV":  _r(electronic.gap_beta_ev)}
    return {
        "homo_eV":            _r(homo),
        "lumo_eV":            _r(lumo),
        "gap_eV":             gap,
        **spin,
        "ip_eV":              _r(fd.ip_ev),
        "ea_eV":              _r(fd.ea_ev),
        "hardness_eV":        _r(fd.hardness_ev),
        "mu_eV":              _r(fd.mu_ev),
        "chi_eV":             _r(fd.chi_ev),
        "omega_eV":           _r(fd.omega_ev),
        "omega_minus_eV":     _r(fd.omega_minus_ev),   # electrodonating power ω⁻ (GCV)
        "omega_plus_eV":      _r(fd.omega_plus_ev),    # electroaccepting power ω⁺ (GCV)
        "softness_eV_inv":    _r(fd.softness),   # S = 1/η, in eV^-1 (the whole dict is in eV)
        "nucleophilicity_eV": _r(fd.nucleophilicity_ev),
    }


def run(
    path_n: str | Path,
    path_np1: str | Path,
    path_nm1: str | Path,
    cluster: str = "",
    engine: Optional[str] = None,
) -> CDFTResult:
    """Run the CDFT Agent on three SCF output files.

    Parameters
    ----------
    path_n, path_np1, path_nm1:
        Paths to the N, N+1, and N-1 electron output files.
    cluster:
        Cluster label (e.g. 'Al2TiO5'). Defaults to the parent directory name.
    engine:
        'gaussian' or 'orca'. Auto-detected from file content when omitted.
    """
    p_n   = Path(path_n)
    p_np1 = Path(path_np1)
    p_nm1 = Path(path_nm1)

    if engine is None:
        # CDFT needs a native parser: _detect_engine() never returns a third value and silently
        # falls back to "gaussian" when the file is neither ORCA nor Gaussian. That is fine for
        # structure_from_file, but here it would misparse silently, so refuse instead.
        from cadmia.io.electronic import sniff_native_engine
        if not p_n.exists():
            raise FileNotFoundError(f"State-N file for CDFT not found: {p_n}")
        is_orca, is_gaussian = sniff_native_engine(p_n)
        if not is_orca and not is_gaussian:
            raise ValueError(
                f"CDFT needs a native parser (ORCA/Gaussian); {p_n} matches neither."
            )

    detected = engine or _detect_engine(p_n)
    loader = CDFTInputState.from_gaussian if detected == "gaussian" else CDFTInputState.from_orca

    state_n   = loader(p_n)
    state_np1 = loader(p_np1)
    state_nm1 = loader(p_nm1)

    # The workflow warns (unbound anion, spin contamination, Fukui sums); kept on the result so
    # they reach the table, the JSON and every stage that reads it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        workflow_result = run_cdft_workflow(state_n, state_np1, state_nm1)
    workflow_warnings = list(dict.fromkeys(
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)))

    sites = _build_sites(workflow_result)
    global_desc = _build_global(workflow_result)
    acid_base = acid_base_summary(sites, hardness_ev=global_desc.get("hardness_eV"))

    return CDFTResult(
        cluster=cluster or p_n.parent.name,
        calculation_id=_file_hash(p_n, p_np1, p_nm1),
        charge_scheme=workflow_result.charge_scheme,
        sites=sites,
        global_descriptors=global_desc,
        acid_base=acid_base,
        sources={
            "N":   str(p_n.resolve()),
            "N+1": str(p_np1.resolve()),
            "N-1": str(p_nm1.resolve()),
        },
        warnings=workflow_warnings,
    )
