"""Descriptor / DescriptorSet — the unified container for every computed property.

A single ``Descriptor`` is a named value with a scope (global / atom / bond / structure /
adsorption / grid). A ``DescriptorSet`` groups the descriptors produced by one source
(e.g. the CDFT agent) so that structural, electronic and adsorption properties can be stored,
queried and correlated through one model instead of a different dataclass per agent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, List, Optional, Tuple

_SCOPES = {"global", "atom", "bond", "structure", "adsorption", "grid"}


@dataclass
class Descriptor:
    name:           str
    value:          Optional[Any] = None      # float | str | bool | None
    unit:           Optional[str] = None
    scope:          str = "global"
    atom_index:     Optional[int] = None
    bond:           Optional[Tuple[int, int]] = None
    structure_id:   Optional[str] = None
    calculation_id: Optional[str] = None
    method:         Optional[str] = None
    metadata:       dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError(f"scope must be one of {sorted(_SCOPES)}, got {self.scope!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Descriptor":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if isinstance(kw.get("bond"), list):
            kw["bond"] = tuple(kw["bond"])
        return cls(**kw)


@dataclass
class DescriptorSet:
    descriptors:    List[Descriptor] = field(default_factory=list)
    source:         str = ""
    calculation_id: Optional[str] = None
    structure_id:   Optional[str] = None
    metadata:       dict = field(default_factory=dict)

    def add(self, name: str, value: Any, **kwargs) -> Descriptor:
        """Create a Descriptor, append it, and return it. Inherits set ids when not given."""
        kwargs.setdefault("calculation_id", self.calculation_id)
        kwargs.setdefault("structure_id", self.structure_id)
        d = Descriptor(name=name, value=value, **kwargs)
        self.descriptors.append(d)
        return d

    def get(self, name: str, *, scope: Optional[str] = None,
            atom_index: Optional[int] = None) -> Optional[Descriptor]:
        for d in self.descriptors:
            if d.name == name and (scope is None or d.scope == scope) \
                    and (atom_index is None or d.atom_index == atom_index):
                return d
        return None

    def filter(self, *, scope: Optional[str] = None) -> List[Descriptor]:
        return [d for d in self.descriptors if scope is None or d.scope == scope]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "calculation_id": self.calculation_id,
            "structure_id": self.structure_id,
            "metadata": self.metadata,
            "descriptors": [d.to_dict() for d in self.descriptors],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DescriptorSet":
        return cls(
            descriptors=[Descriptor.from_dict(x) for x in d.get("descriptors", [])],
            source=d.get("source", ""),
            calculation_id=d.get("calculation_id"),
            structure_id=d.get("structure_id"),
            metadata=d.get("metadata", {}),
        )
