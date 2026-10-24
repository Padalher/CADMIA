"""Integration test: TiO2 -> structural analysis -> polyhedra."""
from pathlib import Path

from cadmia.io.xyz import parse_xyz
from cadmia.structure.structural.analyzer import analyze_structure
from cadmia.structure.structural.polyhedra import analyze_polyhedra

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_tio2_full_pipeline(config, tmp_path):
    s = parse_xyz(FIXTURES / "TiO2_monomer.xyz")
    assert s.n_atoms == 3
    assert s.formula == "O2Ti"

    result = analyze_structure(s, config)
    ti_o = [b for b in result.bonds if "Ti" in b.bond_type and "O" in b.bond_type]
    assert len(ti_o) == 2
    assert not any("Al" in b.bond_type for b in result.bonds)

    ti_desc = next(d for d in result.local_descriptors if d.element == "Ti")
    assert ti_desc.cn == 2
    assert ti_desc.ecn > 0

    polyhedra = analyze_polyhedra(s, result)
    assert isinstance(polyhedra, list)
