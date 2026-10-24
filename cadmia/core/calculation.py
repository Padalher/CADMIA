"""Calculation — common container for a quantum-chemistry calculation (any engine).

Unifies the calculation concept scattered across ElectronicState (energy/eigenvalues),
EngineResult (benchmark), CalcParams (method/basis/dispersion) and the DB calculations table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

from cadmia.core.provenance import Provenance
from cadmia.core.structure import Structure


@dataclass
class Calculation:
    structure:        Optional[Structure] = None
    engine:           Optional[str] = None       # gaussian | orca
    method:           Optional[str] = None       # functional / level of theory
    basis:            Optional[str] = None
    dispersion:       Optional[str] = None        # d3bj | d3zero | d4 | none
    task:             Optional[str] = None        # sp | opt | freq | opt+freq
    charge:           Optional[int] = None
    multiplicity:     Optional[int] = None
    energy_hartree:   Optional[float] = None
    frequencies_cm1:  List[float] = field(default_factory=list)
    status:           str = "unknown"             # done | failed | running | unknown
    wall_time_s:      Optional[float] = None
    source_file:      Optional[str] = None
    metadata:         dict = field(default_factory=dict)
    provenance:       Optional[Provenance] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["structure"] = self.structure.to_dict() if self.structure else None
        d["provenance"] = self.provenance.to_dict() if self.provenance else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Calculation":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in data.items() if k in names}
        if isinstance(kw.get("structure"), dict):
            kw["structure"] = Structure.from_dict(kw["structure"])
        if isinstance(kw.get("provenance"), dict):
            kw["provenance"] = Provenance.from_dict(kw["provenance"])
        return cls(**kw)
