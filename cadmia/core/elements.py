"""Element data — canonical periodic table for the CADMIA core (foundational, no deps)."""
from __future__ import annotations

# Index = atomic number (Z); index 0 is a placeholder so PERIODIC_TABLE[Z] == symbol.
PERIODIC_TABLE = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F",
    "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

ATOMIC_NUMBER_TO_SYMBOL = {z: s for z, s in enumerate(PERIODIC_TABLE) if s}
SYMBOL_TO_ATOMIC_NUMBER = {s: z for z, s in ATOMIC_NUMBER_TO_SYMBOL.items()}
VALID_SYMBOLS = set(ATOMIC_NUMBER_TO_SYMBOL.values())


def atomic_number(symbol: str) -> int | None:
    """Atomic number for an element symbol (case-insensitive), or None if unknown."""
    if not symbol:
        return None
    s = symbol[0].upper() + symbol[1:].lower()
    return SYMBOL_TO_ATOMIC_NUMBER.get(s)


def hill_formula(atoms: "list[str]") -> str:
    """Chemical formula in Hill notation: C first, H second, then alphabetical.

    Single counts are omitted (e.g. C2H4O). Canonical formula helper for the whole
    CADMIA core — every other module should call this instead of re-implementing it.
    """
    from collections import Counter

    counter = Counter(atoms)
    order = sorted(counter, key=lambda s: (s not in ("C", "H"), s != "H" if s != "C" else False, s))
    return "".join(s if counter[s] == 1 else f"{s}{counter[s]}" for s in order)
