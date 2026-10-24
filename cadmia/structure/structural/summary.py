"""Combined structural summary: CN, ECN, average bond lengths, and polyhedral angles.

`analyze_structure` gives bonds, Hoppe ECN and per-bond-type distance statistics; `analyze_polyhedra`
adds coordination-polyhedron geometry (mean angle, angle/length distortion). This helper merges both
into one JSON-serializable dict so the adsorption study can report the full structural picture for a
bare cluster and for an adduct, and look up a single atom's coordination (to see how it changes on
adsorption).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from cadmia.core import Structure
from cadmia.structure.structural.analyzer import analyze_structure
from cadmia.structure.structural.polyhedra import analyze_polyhedra


def structural_summary(structure: Structure, config=None) -> Dict:
    """Return CN/ECN/average-bond-length/angle descriptors for a whole structure.

    Keys: ``n_atoms``, ``bond_statistics`` (mean/std/min/max per bond type), ``ecn_statistics``,
    ``atoms`` (per-atom element/CN/ECN), and ``polyhedra`` (per metal center: CN, mean bond length,
    mean angle, distortions). All values are plain floats/ints for JSON.
    """
    result = analyze_structure(structure, config=config)
    polyhedra = analyze_polyhedra(structure, result)

    atoms = [
        {
            "index": d.atom_index,
            "label": d.label,
            "element": d.element,
            "cn": d.cn,
            "ecn": d.ecn,
            "avg_bond_distance": d.avg_bond_distance,
        }
        for d in result.local_descriptors
    ]
    polys = [
        {
            "center_index": p.center_index,
            "center_label": p.center_label,
            "center_element": p.center_element,
            "cn": p.cn,
            "type": p.polyhedron_type,
            "mean_bond_length": p.mean_bond_length,
            "bond_length_distortion": p.bond_length_distortion,
            "mean_angle": p.mean_angle,
            "angle_distortion": p.angle_distortion,
            "volume": p.volume,
            "quadratic_elongation": p.quadratic_elongation,
        }
        for p in polyhedra
    ]
    return {
        "n_atoms": structure.n_atoms,
        "bond_statistics": result.bond_statistics,
        "ecn_statistics": result.ecn_statistics,
        "atoms": atoms,
        "polyhedra": polys,
    }


def site_coordination(summary: Dict, atom_index: int) -> Optional[Dict]:
    """Look up one atom's CN/ECN in a structural_summary by its 0-based index."""
    for a in summary.get("atoms", []):
        if a["index"] == atom_index:
            return {"cn": a["cn"], "ecn": a["ecn"], "element": a["element"]}
    return None
