"""Tests for the hybrid Structure additions: Atom, Bond, Cell + Structure views."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.core import Atom, Bond, Cell, Structure
from cadmia.core.elements import atomic_number


# ─────────────────────────────────────────────
# elements
# ─────────────────────────────────────────────

def test_atomic_number_lookup_case_insensitive():
    assert atomic_number("Ti") == 22
    assert atomic_number("al") == 13
    assert atomic_number("O") == 8
    assert atomic_number("Xx") is None
    assert atomic_number("") is None


# ─────────────────────────────────────────────
# Atom
# ─────────────────────────────────────────────

def test_atom_autofills_atomic_number_and_validates():
    a = Atom(index=0, symbol="Ti", position=(0.0, 0.0, 0.0))
    assert a.atomic_number == 22
    assert a.position == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        Atom(index=0, symbol="", position=(0, 0, 0))
    with pytest.raises(ValueError):
        Atom(index=0, symbol="O", position=(0, 0))


def test_atom_roundtrip():
    a = Atom(index=2, symbol="O", position=(1.0, 2.0, 3.0), charge=-0.5)
    back = Atom.from_dict(a.to_dict())
    assert back == a
    assert back.atomic_number == 8


# ─────────────────────────────────────────────
# Bond
# ─────────────────────────────────────────────

def test_bond_roundtrip():
    b = Bond(atom_i=0, atom_j=1, distance=1.8, order=1.2, kind="covalent")
    back = Bond.from_dict(b.to_dict())
    assert back == b


# ─────────────────────────────────────────────
# Cell
# ─────────────────────────────────────────────

def test_cell_volume_and_fractional_roundtrip():
    cell = Cell(lattice_vectors=np.diag([2.0, 3.0, 4.0]), periodic=True)
    assert cell.volume() == pytest.approx(24.0)
    cart = np.array([1.0, 1.5, 2.0])
    frac = cell.cart_to_frac(cart)
    assert np.allclose(frac, [0.5, 0.5, 0.5])
    assert np.allclose(cell.frac_to_cart(frac), cart)
    back = Cell.from_dict(cell.to_dict())
    assert back.periodic is True
    assert np.allclose(back.lattice_vectors, cell.lattice_vectors)


def test_cell_empty():
    c = Cell()
    assert c.volume() is None
    with pytest.raises(ValueError):
        c.cart_to_frac(np.array([0.0, 0.0, 0.0]))


# ─────────────────────────────────────────────
# Structure hybrid views (flat storage unchanged)
# ─────────────────────────────────────────────

def _water() -> Structure:
    return Structure(atoms=["O", "H", "H"],
                     positions=np.array([[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]], float),
                     labels=["O", "H", "H"], structure_id="water", source_file="")


def test_structure_atom_view():
    s = _water()
    a0 = s.atom(0)
    assert isinstance(a0, Atom)
    assert a0.symbol == "O" and a0.atomic_number == 8
    assert a0.position == (0.0, 0.0, 0.0)
    atoms = s.to_atoms()
    assert len(atoms) == 3
    assert [a.symbol for a in atoms] == ["O", "H", "H"]


def test_structure_from_atoms_roundtrip():
    s = _water()
    rebuilt = Structure.from_atoms(s.to_atoms(), structure_id="water")
    assert rebuilt.atoms == s.atoms
    assert np.allclose(rebuilt.positions, s.positions)
    assert rebuilt.formula == "H2O"


def test_structure_bonds_cell_dict_roundtrip():
    s = _water()
    s.bonds = [Bond(atom_i=0, atom_j=1, distance=0.96, kind="covalent")]
    s.cell = Cell(lattice_vectors=np.eye(3) * 10.0, periodic=True)
    back = Structure.from_dict(s.to_dict())
    assert len(back.bonds) == 1 and back.bonds[0].kind == "covalent"
    assert back.cell is not None and back.cell.periodic is True
    # flat storage preserved
    assert back.atoms == s.atoms
    assert np.allclose(back.positions, s.positions)


def test_structure_without_bonds_cell_still_roundtrips():
    """Existing flat structures (no bonds/cell) round-trip unchanged."""
    s = _water()
    back = Structure.from_dict(s.to_dict())
    assert back.bonds == []
    assert back.cell is None
