"""Tests for cadmia.core (Provenance, Descriptor/DescriptorSet, AdsorptionSystem, adapters)."""
from __future__ import annotations

import json

import pytest

from cadmia.core import AdsorptionSystem, Descriptor, DescriptorSet, Provenance
from cadmia.core import adapters


# ─────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────

def test_provenance_from_file_hashes(tmp_path):
    f = tmp_path / "calc.out"
    f.write_text("hello", encoding="utf-8")
    prov = Provenance.from_file(f, parser="gaussian", command="cadmia analyze")
    assert prov.file_hash is not None and len(prov.file_hash) == 64
    assert prov.parser == "gaussian"
    assert prov.timestamp is not None
    assert Provenance.from_dict(prov.to_dict()) == prov


# ─────────────────────────────────────────────
# Descriptor / DescriptorSet
# ─────────────────────────────────────────────

def test_descriptor_scope_validation():
    Descriptor(name="gap", value=4.5, scope="global")  # ok
    with pytest.raises(ValueError):
        Descriptor(name="x", value=1.0, scope="not_a_scope")


def test_descriptorset_add_inherits_ids_and_query():
    ds = DescriptorSet(source="cdft", calculation_id="cid", structure_id="Al2TiO5")
    ds.add(name="gap_eV", value=4.5, unit="eV", scope="global")
    ds.add(name="fukui_plus", value=0.35, scope="atom", atom_index=1)
    ds.add(name="fukui_plus", value=0.10, scope="atom", atom_index=2)
    assert ds.get("gap_eV").calculation_id == "cid"
    assert ds.get("gap_eV").structure_id == "Al2TiO5"
    assert len(ds.filter(scope="atom")) == 2
    assert ds.get("fukui_plus", atom_index=2).value == 0.10


def test_descriptorset_json_roundtrip():
    ds = DescriptorSet(source="cdft")
    ds.add(name="gap_eV", value=4.5, unit="eV", scope="global")
    ds.add(name="delta_f", value=-0.2, scope="atom", atom_index=3)
    back = DescriptorSet.from_dict(json.loads(ds.to_json()))
    assert back.source == "cdft"
    assert len(back.descriptors) == 2
    assert back.get("delta_f", atom_index=3).value == -0.2


# ─────────────────────────────────────────────
# AdsorptionSystem
# ─────────────────────────────────────────────

def test_adsorption_system_roundtrip():
    a = AdsorptionSystem(cluster="Al2TiO5", probe="H2", site_atom_index=1,
                         site_atom_symbol="Ti", e_ads_kcal=-18.8,
                         activation={"activated": True, "max_elongation_pct": 10.8},
                         provenance=Provenance(warnings=["w1"]))
    back = AdsorptionSystem.from_dict(a.to_dict())
    assert back.cluster == "Al2TiO5"
    assert back.activation["activated"] is True
    assert back.provenance.warnings == ["w1"]


# ─────────────────────────────────────────────
# Adapters
# ─────────────────────────────────────────────

def test_cdft_to_descriptorset():
    from cadmia.agents.cdft_agent import CDFTResult, SiteDescriptor
    res = CDFTResult(
        cluster="Al2TiO5", calculation_id="cid", charge_scheme="hirshfeld",
        sites=[
            SiteDescriptor(atom_index=1, atom_symbol="Ti", coordinates=(0.0, 0.0, 0.0),
                           fukui_plus=0.35, fukui_minus=0.11, delta_f=0.24,
                           character="electrophilic", rank=1),
            SiteDescriptor(atom_index=2, atom_symbol="O", coordinates=(1.8, 0.0, 0.0),
                           fukui_plus=0.07, fukui_minus=0.32, delta_f=-0.25,
                           character="nucleophilic", rank=2),
        ],
        global_descriptors={"gap_eV": 4.57, "hardness_eV": 7.74, "mu_eV": None},
        sources=[], warnings=[],
    )
    ds = adapters.cdft_to_descriptorset(res)
    assert ds.source == "cdft" and ds.structure_id == "Al2TiO5"
    g = ds.filter(scope="global")
    assert {d.name for d in g} == {"gap_eV", "hardness_eV"}   # mu None skipped
    assert ds.get("gap_eV", scope="global").unit == "eV"
    assert len(ds.filter(scope="atom")) == 6                  # 3 fukui × 2 sites
    f1 = ds.get("delta_f", atom_index=1)
    assert f1.value == 0.24 and f1.metadata["rank"] == 1


def test_analysis_to_adsorption():
    from cadmia.agents.analysis_agent import AnalysisResult, ProbeBond
    res = AnalysisResult(
        cluster="Al2TiO5", probe="H2", config_index=0,
        site_atom_index=1, site_atom_symbol="Ti", engine="orca",
        e_ads_kcal=-18.8, bsse_kcal=1.25, e_ads_bsse_kcal=-17.6,
        probe_bonds=[ProbeBond(label="H-H", free_A=0.74, adduct_A=0.82, delta_A=0.08, percent=10.8)],
        max_elongation_pct=10.8, activated=True,
        cdft_site_rank=1, cdft_site_character="electrophilic", cdft_agreement=True,
        warnings=["w"],
    )
    a = adapters.analysis_to_adsorption(res)
    assert a.cluster == "Al2TiO5" and a.probe == "H2"
    assert a.site_atom_index == 1 and a.e_ads_bsse_kcal == -17.6
    assert a.activation["activated"] is True
    assert a.activation["bonds"][0]["label"] == "H-H"
    assert a.metadata["cdft_agreement"] is True
    assert a.provenance.warnings == ["w"]


def test_stability_to_descriptorset():
    from cadmia.agents.stability_agent import StabilityResult
    res = StabilityResult(
        mixed="Al2TiO5", formula="Al2O5Ti", n_atoms=8, engine="orca",
        e_mixed_hartree=-150.0, e_rel_hartree=-0.1, e_rel_kcal=-62.75,
        e_rel_eV=-2.72, e_rel_eV_per_atom=-0.34, balanced=True,
    )
    ds = adapters.stability_to_descriptorset(res)
    assert ds.source == "stability" and ds.structure_id == "Al2TiO5"
    assert ds.get("e_rel", scope="structure").value == -62.75
    assert ds.get("e_rel", scope="structure").unit == "kcal/mol"
    assert ds.get("stoichiometry_balanced", scope="structure").value is True


# ─────────────────────────────────────────────
# Structure (consolidated in cadmia.core) — methods
# ─────────────────────────────────────────────

def test_structure_methods_and_xyz_roundtrip():
    import numpy as np
    from cadmia.core import Structure
    s = Structure(atoms=["O", "H", "H"],
                  positions=np.array([[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]], float),
                  labels=["O", "H", "H"], structure_id="water", source_file="")
    assert s.symbols() == ["O", "H", "H"]
    assert s.distance(0, 1) == pytest.approx(0.96, abs=1e-6)
    assert s.formula == "H2O"
    xyz = s.to_xyz(comment="water")
    back = Structure.from_xyz(xyz, structure_id="water")
    assert back.atoms == s.atoms
    assert back.distance(0, 1) == pytest.approx(0.96, abs=1e-6)


# ─────────────────────────────────────────────
# Calculation + adapters
# ─────────────────────────────────────────────

def test_calculation_roundtrip():
    import numpy as np
    from cadmia.core import Calculation, Structure
    s = Structure(atoms=["Ti", "O"], positions=np.array([[0, 0, 0], [1.8, 0, 0]], float),
                  labels=["Ti", "O"], structure_id="TiO", source_file="", charge=0, multiplicity=1)
    c = Calculation(structure=s, engine="orca", method="PBE0", basis="def2-TZVP",
                    dispersion="d3bj", task="sp", energy_hartree=-100.0, status="done")
    back = Calculation.from_dict(c.to_dict())
    assert back.engine == "orca" and back.energy_hartree == -100.0
    assert back.structure.formula == "OTi"


def test_electronic_state_to_calculation():
    import numpy as np
    from cadmia.core import ElectronicState, Structure
    s = Structure(atoms=["Ti"], positions=np.array([[0, 0, 0]], float),
                  labels=["Ti"], structure_id="x", source_file="x.out", charge=1, multiplicity=2)
    state = ElectronicState(structure=s, energy_hartree=-50.0, homo_ev=-8.0, lumo_ev=-3.0,
                            source_file="x.out")
    calc = adapters.electronic_state_to_calculation(state, engine="orca", method="PBE0",
                                                    basis="def2-TZVP", dispersion="d3bj", task="sp")
    assert calc.engine == "orca" and calc.energy_hartree == -50.0
    assert calc.charge == 1 and calc.multiplicity == 2
    assert calc.provenance.source_file == "x.out"
