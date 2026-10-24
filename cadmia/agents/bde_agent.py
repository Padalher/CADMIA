"""BDE Agent — bond dissociation energy from fragment total energies.

Given the parent molecule and the fragments it dissociates into (each computed at the
same level of theory), the bond dissociation energy is

    BDE = Σ_i E(fragment_i) − E(molecule)

A positive BDE is the energy required to break the bond(s) and separate the fragments;
larger values mean a stronger bond. Fragments are taken as provided by the user (usually
the two radicals of a homolytic cleavage, at their relaxed geometries). Energies come from
the user's own output files; CADMIA only parses and combines them.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from cadmia.core.elements import hill_formula
from cadmia.core.units import HARTREE_TO_KCAL as _HARTREE_TO_KCAL, HARTREE_TO_EV as _HARTREE_TO_EV
from cadmia.io.electronic import load_electronic as _parse_output
from cadmia.io.electronic import describe_level_mismatch


@dataclass
class BDEFragment:
    label:     str
    formula:   str
    e_hartree: Optional[float] = None


@dataclass
class BDEResult:
    molecule:           str
    formula:            str
    engine:             str = ""
    e_molecule_hartree: Optional[float] = None
    fragments:          List[BDEFragment] = field(default_factory=list)
    bde_hartree:        Optional[float] = None
    bde_kcal:           Optional[float] = None
    bde_eV:             Optional[float] = None
    atoms_conserved:    bool = False
    warnings:           List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fragments"] = [asdict(f) for f in self.fragments]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compute_bde(
    molecule_out: "str | Path",
    fragment_outs: List["str | Path"],
    *,
    molecule_label: str = "",
    fragment_labels: Optional[List[str]] = None,
) -> BDEResult:
    """Bond dissociation energy: Σ E(fragment) − E(molecule).

    ``fragment_outs`` are the output files of the dissociation fragments (typically two).
    Atom conservation (Σ fragment atoms == molecule atoms) is checked and reported; a
    mismatch does not abort but raises a warning, since the number is only physically
    meaningful when the fragments together account for the molecule's atoms.
    """
    mol_state, engine = _parse_output(molecule_out)
    mol_atoms = mol_state.structure.atoms

    result = BDEResult(
        molecule=molecule_label or mol_state.structure.structure_id or _formula(mol_atoms),
        formula=_formula(mol_atoms),
        engine=engine,
        e_molecule_hartree=mol_state.energy_hartree,
    )

    frag_states = [_parse_output(p)[0] for p in fragment_outs]
    frag_counts: "Counter[str]" = Counter()
    for idx, st in enumerate(frag_states):
        atoms = st.structure.atoms
        frag_counts.update(atoms)
        label = (fragment_labels[idx] if fragment_labels and idx < len(fragment_labels)
                 else st.structure.structure_id or _formula(atoms))
        result.fragments.append(
            BDEFragment(label=label, formula=_formula(atoms), e_hartree=st.energy_hartree)
        )

    result.atoms_conserved = (dict(frag_counts) == dict(Counter(mol_atoms)))
    if not result.atoms_conserved:
        result.warnings.append(
            f"Atom count mismatch: fragments {hill_formula(list(frag_counts.elements()))} "
            f"vs molecule {result.formula}. BDE may not be physically meaningful."
        )

    mismatch = describe_level_mismatch(
        {"molecule": molecule_out, **{f"fragment[{i}]": p for i, p in enumerate(fragment_outs)}})
    if mismatch:
        result.warnings.append(mismatch)

    energies = [f.e_hartree for f in result.fragments]
    if mol_state.energy_hartree is None or any(e is None for e in energies):
        result.warnings.append("Missing total energy in the molecule or a fragment; BDE not computed.")
        return result

    bde = sum(energies) - mol_state.energy_hartree
    result.bde_hartree = bde
    result.bde_kcal = bde * _HARTREE_TO_KCAL
    result.bde_eV = bde * _HARTREE_TO_EV
    return result


def _formula(atoms: List[str]) -> str:
    return hill_formula(atoms)
