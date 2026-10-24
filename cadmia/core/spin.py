"""Electron counting and automatic spin multiplicity.

The total electron count fixes the *parity* of the spin: an even number of electrons
admits a closed-shell singlet, an odd number forces at least one unpaired electron.
``multiplicity_from_electrons`` returns the lowest multiplicity compatible with that
parity (1 for even, 2 for odd) — the standard automatic default for input generation.

This is correct for the vast majority of closed-shell main-group species and for the
charged states of a finite-difference CDFT triplet (the N±1 ions flip parity and come
out as doublets). It does NOT predict high-spin ground states of transition metals;
those need an explicit multiplicity. ``transition_metals_present`` flags that case so
callers can warn the user.
"""
from __future__ import annotations

from typing import List, Sequence

from cadmia.core.elements import atomic_number

# Atomic numbers of the d-block (Sc–Zn, Y–Cd, Hf–Hg) plus the lanthanides/actinides,
# where the lowest-parity multiplicity is often NOT the ground state.
_TRANSITION_AND_F_BLOCK = (
    set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))
    | set(range(57, 72)) | set(range(89, 104))
)


def electron_count(atoms: Sequence[str], charge: int = 0) -> int:
    """Total number of electrons: Σ Z(atom) − charge.

    Raises ValueError if any symbol is not a known element.
    """
    total = 0
    for sym in atoms:
        z = atomic_number(sym)
        if z is None:
            raise ValueError(f"unknown element symbol: {sym!r}")
        total += z
    return total - charge


def multiplicity_from_electrons(atoms: Sequence[str], charge: int = 0) -> int:
    """Lowest spin multiplicity (2S+1) consistent with the electron-count parity.

    Even electron count → 1 (singlet); odd → 2 (doublet).
    """
    return 1 if electron_count(atoms, charge) % 2 == 0 else 2


def transition_metals_present(atoms: Sequence[str]) -> List[str]:
    """Distinct d-/f-block element symbols in *atoms* (order of first appearance).

    Used to warn that an automatically chosen multiplicity may miss a high-spin
    ground state.
    """
    seen: List[str] = []
    for sym in atoms:
        z = atomic_number(sym)
        if z in _TRANSITION_AND_F_BLOCK and sym not in seen:
            seen.append(sym)
    return seen
