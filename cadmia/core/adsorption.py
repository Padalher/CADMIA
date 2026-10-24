"""AdsorptionSystem — common bridge object for a (cluster, probe, config) adsorption result.

Connects the geometry agent (configs), the analysis agent (energetics/activation), the
database, and future visualization. Structure objects are referenced by label/id for now;
they will hold core.Structure instances once the Structure consolidation lands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from cadmia.core.provenance import Provenance


@dataclass
class AdsorptionSystem:
    cluster:            str                       # cluster label / id
    probe:              str                       # probe label (H2 / CO2 / NH3)
    site_atom_index:    Optional[int] = None      # 1-based binding-site atom
    site_atom_symbol:   Optional[str] = None
    orientation:        Optional[str] = None
    config_index:       Optional[int] = None
    e_ads_kcal:         Optional[float] = None
    e_ads_bsse_kcal:    Optional[float] = None
    e_def_cluster_kcal: Optional[float] = None
    e_def_probe_kcal:   Optional[float] = None
    e_int_kcal:         Optional[float] = None
    activation:         dict = field(default_factory=dict)   # {activated, max_elongation_pct, bonds}
    provenance:         Optional[Provenance] = None
    metadata:           dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict() if self.provenance else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AdsorptionSystem":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if isinstance(kw.get("provenance"), dict):
            kw["provenance"] = Provenance.from_dict(kw["provenance"])
        return cls(**kw)
