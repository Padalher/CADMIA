"""Bond — connectivity/bond record (distances, bond order, type). Used by structural analysis, ECN, adsorption."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class Bond:
    atom_i:   int
    atom_j:   int
    distance: Optional[float] = None
    order:    Optional[float] = None        # Wiberg / Mayer / NLMO bond order
    kind:     Optional[str] = None          # covalent | ionic | dative | weak
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Bond":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
