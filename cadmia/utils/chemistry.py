"""Covalent radii and bond distance utilities.

Values from Cordero et al., Dalton Trans. 2008, 2832.
DOI: 10.1039/b801115j
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

# Covalent radii in Å (Cordero 2008)
COVALENT_RADII: dict[str, float] = {
    "H":  0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B":  0.84, "C":  0.76, "N":  0.71, "O":  0.66, "F":  0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P":  1.07, "S":  1.05, "Cl": 1.02, "Ar": 1.06,
    "K":  2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V":  1.53, "Cr": 1.39, "Mn": 1.50, "Fe": 1.42, "Co": 1.38,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95,
    "Y":  1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I":  1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89,
    "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Hf": 1.75, "Ta": 1.70, "W":  1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
}

# Tolerance factors for acceptable interatomic distance ranges
_MIN_FACTOR = 0.65   # < 65% of summed radii → clash
_MAX_FACTOR = 2.00   # > 200% → non-interacting (not reported as error)


def covalent_radius(element: str) -> Optional[float]:
    """Return the covalent radius in Å, or None if the element is not in the table."""
    return COVALENT_RADII.get(element.capitalize())


def bond_distance_range(elem_a: str, elem_b: str) -> Tuple[float, float]:
    """Return (d_min, d_max) in Å for a pair of elements.

    d_min = MIN_FACTOR × (r_A + r_B)
    d_max = MAX_FACTOR × (r_A + r_B)
    Falls back to 1.5 Å per atom if the radius is unknown.
    """
    r_a = covalent_radius(elem_a) or 1.5
    r_b = covalent_radius(elem_b) or 1.5
    r_sum = r_a + r_b
    return _MIN_FACTOR * r_sum, _MAX_FACTOR * r_sum


def expected_spin_sq(multiplicity: int) -> float:
    """Return the expected ⟨S²⟩ = S(S+1) for a given spin multiplicity."""
    s = (multiplicity - 1) / 2.0
    return s * (s + 1)

# The skeleton of an oxide cluster needs a tighter cutoff than a probe molecule's own bonds:
# in a metal-oxide cluster, real bonds reach about d/(r_i+r_j) = 1.0 and non-bonded contacts
# start well above, so 1.20 falls in an empty gap. At 1.3 a metal···metal contact starts
# counting as a bond, which flips the verdict for no chemical reason.
_SKELETON_CUTOFF = 1.20
_SKELETON_MARGIN = 0.08   # ratios this close to the cutoff make the verdict unsafe


def skeleton(atoms: List[str], positions: np.ndarray) -> Tuple[tuple, Counter, bool]:
    """(connectivity signature, bond counts per element pair, any pair near the cutoff?).

    The signature is a Weisfeiler-Lehman refinement: every atom is relabelled with its own
    element plus its neighbours', three times over. It is invariant to rotation, translation
    and the order the atoms happen to be written in, and it changes only when a bond is made
    or broken. That is what separates a substrate that rearranged from one merely deformed by
    the adsorbate — a distance comparison cannot tell those apart, because an adsorbate routinely
    deforms a cluster as much as a genuine rearrangement moves it.
    """
    n = len(atoms)
    adj: List[List[int]] = [[] for _ in range(n)]
    borderline = False
    for i in range(n):
        ri = covalent_radius(atoms[i]) or 0.77
        for j in range(i + 1, n):
            rj = covalent_radius(atoms[j]) or 0.77
            ratio = float(np.linalg.norm(positions[i] - positions[j])) / (ri + rj)
            if ratio < _SKELETON_CUTOFF:
                adj[i].append(j)
                adj[j].append(i)
            if abs(ratio - _SKELETON_CUTOFF) < _SKELETON_MARGIN:
                borderline = True
    labels = list(atoms)
    for _ in range(3):
        labels = [f"{labels[i]}({','.join(sorted(labels[j] for j in adj[i]))})" for i in range(n)]
    counts = Counter("-".join(sorted((atoms[i], atoms[j])))
                     for i in range(n) for j in adj[i] if i < j)
    return tuple(sorted(labels)), counts, borderline


def classify_skeletons(
    ref_atoms: List[str],
    ref_signature: tuple,
    entries,
) -> Tuple[dict, List[str]]:
    """Label each adduct by the skeleton its cluster block relaxed to.

    `entries` are ``(key, atoms, coordinates)`` of complete adducts written cluster first
    and probe appended, the convention the adsorption agents share. Returns
    ``{key: "reference" | "isomer-N"}`` and any warnings; a key is missing when its
    geometry could not be judged, and an empty mapping means the caller should fall back
    to a single global comparison.

    Two adducts are only comparable by total energy when their substrate is the same
    substance. When the relaxation rearranges the cluster, the adduct's energy also carries
    the isomerization energy, and ranking them together lets a cluster isomer that happens
    to be more stable take over the zero and discard every adduct that kept the reference
    skeleton.

    The reference must come from the SAME level of theory as the geometries being judged:
    the skeleton of a bare cluster can itself differ between DFT and GFN2-xTB.
    """
    n_cluster = len(ref_atoms)
    classes: dict = {}
    isomers: dict = {}
    warnings: List[str] = []
    for key, atoms, coords in entries:
        atoms = list(atoms)
        if len(atoms) < n_cluster:
            continue
        if atoms[:n_cluster] != ref_atoms:
            warnings.append(
                "the adducts' first atoms do not match the cluster reference's element order "
                "(expected cluster first, probe appended); ranking against a single global "
                "minimum instead")
            return {}, warnings
        sig = skeleton(ref_atoms, np.asarray(coords, dtype=float)[:n_cluster])[0]
        classes[key] = ("reference" if sig == ref_signature
                        else isomers.setdefault(sig, f"isomer-{len(isomers) + 1}"))
    return classes, warnings
