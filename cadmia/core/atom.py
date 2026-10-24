"""Atom — a single-atom view/record. A rich complement to Structure's flat arrays."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Tuple

from cadmia.core.elements import atomic_number as _z_of


@dataclass
class Atom:
    index:         int
    symbol:        str
    position:      Tuple[float, float, float]
    atomic_number: Optional[int] = None
    charge:        Optional[float] = None
    metadata:      dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.symbol:
            raise ValueError("Atom.symbol must be a non-empty element symbol.")
        if len(tuple(self.position)) != 3:
            raise ValueError(f"Atom.position must have 3 components, got {self.position!r}.")
        self.position = tuple(float(c) for c in self.position)
        if self.atomic_number is None:
            self.atomic_number = _z_of(self.symbol)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Atom":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if isinstance(kw.get("position"), list):
            kw["position"] = tuple(kw["position"])
        return cls(**kw)
