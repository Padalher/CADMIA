"""Adapters from the existing agent result dataclasses to the unified core model.

These let current agents (CDFT, analysis, stability) feed the shared Descriptor / DescriptorSet
/ AdsorptionSystem model without changing the agents themselves — the migration path of the
restructuring (add core + adapters first, refactor agents later).
"""
from __future__ import annotations

from typing import Any, Optional

from cadmia.core.adsorption import AdsorptionSystem
from cadmia.core.calculation import Calculation
from cadmia.core.descriptor import DescriptorSet
from cadmia.core.provenance import Provenance


def electronic_state_to_calculation(
    state: Any,   # cadmia.core.ElectronicState
    *,
    engine: Optional[str] = None,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    dispersion: Optional[str] = None,
    task: Optional[str] = None,
    status: str = "done",
) -> Calculation:
    """ElectronicState → Calculation (energy + frequencies + structure + level of theory)."""
    return Calculation(
        structure=state.structure,
        engine=engine,
        method=method,
        basis=basis,
        dispersion=dispersion,
        task=task,
        charge=getattr(state.structure, "charge", None),
        multiplicity=getattr(state.structure, "multiplicity", None),
        energy_hartree=state.energy_hartree,
        frequencies_cm1=list(state.frequencies_cm1 or []),
        status=status,
        source_file=state.source_file or None,
        provenance=Provenance(source_file=state.source_file or None, parser=engine),
    )


def cdft_to_descriptorset(result: Any) -> DescriptorSet:
    """CDFTResult → DescriptorSet (global descriptors + per-atom Fukui as scope='atom')."""
    ds = DescriptorSet(source="cdft", calculation_id=result.calculation_id,
                       structure_id=result.cluster)
    for key, val in result.global_descriptors.items():
        if val is None:
            continue
        unit = "eV" if key.endswith("_eV") else ("Ha" if key.endswith("_Ha") else None)
        ds.add(name=key, value=val, unit=unit, scope="global")
    for site in result.sites:
        for name, value in (("fukui_plus", site.fukui_plus),
                            ("fukui_minus", site.fukui_minus),
                            ("delta_f", site.delta_f)):
            if value is not None:
                ds.add(name=name, value=value, scope="atom", atom_index=site.atom_index,
                       metadata={"symbol": site.atom_symbol, "rank": site.rank,
                                 "character": site.character})
    return ds


def stability_to_descriptorset(result: Any) -> DescriptorSet:
    """StabilityResult → DescriptorSet (relative-stability descriptors, scope='structure')."""
    ds = DescriptorSet(source="stability", structure_id=result.mixed)
    for name, value, unit in (("e_rel", result.e_rel_kcal, "kcal/mol"),
                              ("e_rel_eV", result.e_rel_eV, "eV"),
                              ("e_rel_per_atom", result.e_rel_eV_per_atom, "eV/atom")):
        if value is not None:
            ds.add(name=name, value=value, unit=unit, scope="structure")
    ds.add(name="stoichiometry_balanced", value=result.balanced, scope="structure")
    return ds


def analysis_to_adsorption(result: Any) -> AdsorptionSystem:
    """AnalysisResult → AdsorptionSystem (energetics + activation + CDFT cross-ref)."""
    return AdsorptionSystem(
        cluster=result.cluster,
        probe=result.probe,
        site_atom_index=result.site_atom_index,
        site_atom_symbol=result.site_atom_symbol,
        config_index=result.config_index,
        e_ads_kcal=result.e_ads_kcal,
        e_ads_bsse_kcal=result.e_ads_bsse_kcal,
        e_def_cluster_kcal=result.e_def_cluster_kcal,
        e_def_probe_kcal=result.e_def_probe_kcal,
        e_int_kcal=result.e_int_kcal,
        activation={
            "activated": result.activated,
            "max_elongation_pct": result.max_elongation_pct,
            "bonds": [{"label": b.label, "delta_A": b.delta_A, "percent": b.percent}
                      for b in result.probe_bonds],
        },
        provenance=Provenance(warnings=list(result.warnings)),
        metadata={
            "engine": result.engine,
            "cdft_site_rank": result.cdft_site_rank,
            "cdft_site_character": result.cdft_site_character,
            "cdft_agreement": result.cdft_agreement,
        },
    )
