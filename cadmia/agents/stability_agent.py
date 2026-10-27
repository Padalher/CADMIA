"""Stability Agent — relative (formation) stability of a mixed oxide vs. pure oxides.

Intrinsic thermodynamic descriptor of the cluster, independent of adsorption:

    E_rel = E(mixed) − Σ_i c_i · E(pure_i)

where the stoichiometric coefficients c_i balance the mixed formula against the pure-oxide
references (e.g. Al2TiO5 = 1·Al2O3 + 1·TiO2). A negative E_rel means the mixed oxide is more
stable than the sum of its pure constituents. Reference energies come from the user's own
calculations at the same level of theory (parsed from their .out files).

Which energy E stands for is a choice: E0 = E + ZPE by default, or the electronic
energy E, the enthalpy H, the Gibbs energy G as the engine printed it, or Gq, G with
Grimme's quasi-RRHO vibrational entropy for every term. Every difference uses ONE basis for all its
terms; when the requested one is missing from any input (no frequency job), all of them fall
back to E and the result says so, rather than subtracting an E from an E0.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from math import exp, gcd, log
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cadmia.core.elements import hill_formula
from cadmia.core.units import HARTREE_TO_KCAL as _HARTREE_TO_KCAL, HARTREE_TO_EV as _HARTREE_TO_EV
from cadmia.io.electronic import load_electronic as _parse_output
from cadmia.io.electronic import describe_level_mismatch


ENERGY_BASES = ("E0", "E", "H", "G", "Gq")
DEFAULT_ENERGY_BASIS = "E0"


def energy_in_basis(state, basis: str) -> Optional[float]:
    """Total energy of a parsed state in the given basis (Hartree), or None if not printed."""
    if basis == "E":
        return state.energy_hartree
    if basis == "E0":
        if state.electronic_zpe_energy_hartree is not None:      # Gaussian prints the sum
            return state.electronic_zpe_energy_hartree
        if state.energy_hartree is not None and state.zpe_correction_hartree is not None:
            return state.energy_hartree + state.zpe_correction_hartree   # ORCA prints the ZPE
        return None
    if basis == "H":
        return state.enthalpy_hartree
    if basis in ("G", "Gq"):
        if basis == "G":
            g = state.gibbs_free_energy_hartree
        else:
            from cadmia.electronic.thermo import gibbs_quasi_rrho
            g = gibbs_quasi_rrho(state)
        return None if g is None else g + _sigma_correction(state)
    raise ValueError(f"Unknown energy basis {basis!r}; expected one of {ENERGY_BASES}.")


_KB_HARTREE = 3.166811563e-6      # Boltzmann constant, Hartree/K


def _sigma_correction(state) -> float:
    """RT·ln(σ_geometry/σ_engine), in Hartree: G with the symmetry the geometry actually has.

    −T·S_rot carries +RT·ln σ, so an engine that took a C2 minimum for C1 (σ 1 instead of 2)
    printed a G too low by RT·ln 2. Zero when either σ or the temperature is unknown.
    """
    sym = state.geometry_symmetry
    sigma_engine = state.rotational_symmetry_number
    if sym is None or not sigma_engine or not state.temperature_kelvin or sym.sigma == sigma_engine:
        return 0.0
    return _KB_HARTREE * state.temperature_kelvin * log(sym.sigma / sigma_engine)


def detect_symmetries(states) -> None:
    """Attach the symmetry detected from each final geometry."""
    from cadmia.structure.structural.symmetry import detect_symmetry
    for state in states:
        if state is not None and state.geometry_symmetry is None:
            state.geometry_symmetry = detect_symmetry(state.structure.atoms, state.structure.positions)


def _resolve_basis(requested: str, named_states) -> tuple[str, Optional[str]]:
    """The basis every term will use, and the warning to raise if it is not the requested one."""
    missing = [label for label, state in named_states
               if state is not None and energy_in_basis(state, requested) is None]
    if not missing or requested == "E":
        return requested, None
    return "E", (
        f"{requested} is not available in: {', '.join(missing)} (no frequency calculation"
        + ("; or, for Gq, an ORCA output that does not say which entropy it used" if requested == "Gq" else "")
        + "). "
        f"Every energy falls back to the electronic energy E, so that no difference mixes E "
        f"with {requested}.")


def _check_terms(result, terms, basis: str, frequencies_missing_reported: bool) -> None:
    """Record multiplicity and imaginary modes of every term, and warn where they matter.

    ``terms`` are (label, path, state, engine). A term with an imaginary mode is a saddle point,
    and the engine leaves that mode out of its ZPE, so E0/H/G would enter the difference silently.
    """
    from cadmia.agents.rank_agent import _count_imaginary

    imaginary, unverified, temps, engines, resym = [], [], set(), set(), []
    for label, path, state, engine in terms:
        if state is None:
            continue
        sym = state.geometry_symmetry
        if sym is not None:
            result.symmetry[label] = {
                "point_group": sym.point_group, "sigma": sym.sigma, "chiral": sym.chiral,
                "tolerance_angstrom": sym.tolerance,
                "engine_point_group": state.structure.point_group,
                "engine_sigma": state.rotational_symmetry_number,
            }
            if state.rotational_symmetry_number and sym.sigma != state.rotational_symmetry_number:
                resym.append(f"{label} {sym.point_group} (σ {sym.sigma}) vs engine "
                             f"{state.structure.point_group} (σ {state.rotational_symmetry_number}), "
                             f"{_sigma_correction(state) * _HARTREE_TO_KCAL:+.2f} kcal/mol")
        n_imag = _count_imaginary(Path(path), state, engine)
        result.n_imaginary[label] = n_imag
        result.multiplicities[label] = state.structure.multiplicity
        if n_imag is None:
            unverified.append(label)
        elif n_imag:
            imaginary.append(f"{label} ({n_imag})")
        temps.add(state.temperature_kelvin)
        engines.add(engine)

    if resym:
        result.warnings.append(
            "Symmetry of the geometry differs from the one the engine used: " + "; ".join(resym)
            + ". G and Gq are corrected by RT·ln(σ/σ_engine); E, E0 and H do not "
              "depend on σ.")
    if imaginary:
        result.warnings.append(
            "Imaginary frequencies in: " + ", ".join(imaginary) + ". Those geometries are not "
            "minima, and the engine leaves the imaginary mode out of E0/H/G.")
    if unverified and not frequencies_missing_reported:
        result.warnings.append(
            "No frequency calculation in: " + ", ".join(unverified)
            + "; they are not verified as minima.")
    mults = set(result.multiplicities.values())
    if len(mults) > 1:
        result.warnings.append(
            "Spin multiplicity differs across terms ("
            + ", ".join(f"{k} {v}" for k, v in result.multiplicities.items())
            + "): the difference includes the change of spin state; declare it.")
    if basis in ("H", "G", "Gq") and len(temps) > 1:
        result.warnings.append(
            f"{basis} was computed at different temperatures across terms "
            f"({', '.join(str(t) for t in sorted(temps, key=str))} K).")
    if basis == "G" and len(engines) > 1:
        result.warnings.append(
            "G mixes engines (" + ", ".join(sorted(engines)) + "): ORCA applies Grimme's "
            "quasi-RRHO entropy by default and Gaussian a purely harmonic one.")


def _formula(atoms: List[str]) -> str:
    return hill_formula(atoms)


def _formula_units(atoms: List[str]) -> int:
    """Number of formula units n in a stoichiometric cluster (gcd of the atom counts).

    For a (TiAl2O5)_n cluster the composition is {Ti:n, Al:2n, O:5n}; since the unit
    counts (1,2,5) are coprime, gcd = n. General for any series whose repeating unit
    has coprime element counts.
    """
    counts = list(Counter(atoms).values())
    if not counts:
        return 0
    result = 0
    for c in counts:
        result = gcd(result, int(c))
    return result or 1


_MAX_DENOMINATOR = 1000


def _solve_coefficients(
    mixed_counts: Dict[str, int], ref_counts: List[Dict[str, int]],
) -> tuple[List[Fraction], bool]:
    """Solve c_i so that Σ c_i · ref_i = mixed (per element). Returns (coeffs, balanced).

    The coefficients are small rationals, not integers: references of twice the size of the
    mixed oxide's pure contributions give ½, ½. Balance is checked exactly.
    """
    elements = sorted(set().union(mixed_counts, *ref_counts))
    A = np.array([[rc.get(e, 0) for rc in ref_counts] for e in elements], dtype=float)
    b = np.array([mixed_counts.get(e, 0) for e in elements], dtype=float)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    coeffs = [Fraction(float(c)).limit_denominator(_MAX_DENOMINATOR) for c in sol]
    return coeffs, all(c >= 0 for c in coeffs) and _balances(mixed_counts, ref_counts, coeffs)


def _balances(mixed_counts, ref_counts, coeffs) -> bool:
    combined = Counter()
    for c, rc in zip(coeffs, ref_counts):
        for el, k in rc.items():
            combined[el] += c * k
    return {el: v for el, v in combined.items() if v} == mixed_counts


def _reference_size(n: int, coeffs, units: List[int]) -> str:
    """'same-n' when every reference used has the mixed oxide's n formula units; 'equal-size'
    when each has as many units as all the pure contributions together (2n for
    Al2TiO5 = Al2O3 + TiO2); 'other' otherwise."""
    used = [u for c, u in zip(coeffs, units) if c]
    if not used:
        return "other"
    if all(u == n for u in used):
        return "same-n"
    total = sum(c * u for c, u in zip(coeffs, units))
    if all(u == total for u in used):
        return "equal-size"
    return "other"


def _coefficient(c: Fraction):
    return int(c) if c.denominator == 1 else float(c)


@dataclass
class StabilityReference:
    label:       str
    formula:     str
    coefficient: float                     # int when whole; 0.5 for ½
    e_hartree:   Optional[float] = None


@dataclass
class StabilityResult:
    mixed:             str
    formula:           str
    n_atoms:           int
    engine:            str = ""
    e_mixed_hartree:   Optional[float] = None
    references:        List[StabilityReference] = field(default_factory=list)
    e_rel_hartree:     Optional[float] = None
    e_rel_kcal:        Optional[float] = None
    e_rel_eV:          Optional[float] = None
    e_rel_eV_per_atom: Optional[float] = None
    formula_units:     int = 1
    e_rel_kcal_per_fu: Optional[float] = None
    reference_size:    str = "other"    # same-n | equal-size | other
    balanced:          bool = False
    # Basis of every energy above (E0, E, H or G), and E_rel in each basis all inputs carry.
    energy_basis:      str = "E"
    e_rel_by_basis_kcal: Dict[str, float] = field(default_factory=dict)
    temperature_k:     Optional[float] = None
    # Per term: spin multiplicity and imaginary modes (None = no frequency calculation).
    multiplicities:    Dict[str, int] = field(default_factory=dict)
    n_imaginary:       Dict[str, Optional[int]] = field(default_factory=dict)
    symmetry:          Dict[str, dict] = field(default_factory=dict)   # detected
    mixed_atoms:       List[str] = field(default_factory=list)
    mixed_positions:   List[List[float]] = field(default_factory=list)
    warnings:          List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["references"] = [asdict(r) for r in self.references]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _warn_unconverged(result, named_paths) -> None:
    """Flag any input whose optimization did not converge.

    Same reason as in analysis_agent: a run that exhausted its cycles still ends normally and
    still prints a last energy, and that energy belongs to a step rather than to a minimum.
    A relative stability built on it is a difference between two things that are not comparable.
    Reported rather than suppressed here, because a series is read as a whole and dropping one
    member silently would distort the trend more than flagging it.
    """
    from cadmia.agents.cdft_agent import _detect_engine
    from cadmia.agents.validation_agent import _check_termination, _read_lines, geometry_converged

    failed, unfinished = [], []
    for label, path in named_paths:
        if path is None:
            continue
        try:
            engine = _detect_engine(Path(path))      # each term with its own engine's markers
            if geometry_converged(path, engine) is False:
                failed.append(str(label))
            term = _check_termination(_read_lines(Path(path)), engine)
            if term.status == "fail":
                unfinished.append(f"{label} ({term.message.split(':')[0].split(' —')[0]})")
        except (OSError, ValueError):
            continue          # unreadable here means the parser already complained
    if unfinished:
        result.warnings.append(
            "Did not terminate normally: " + ", ".join(unfinished)
            + ". The energy taken is the last one printed by an unfinished calculation.")
    if failed:
        result.warnings.append(
            "Geometry NOT converged in: " + ", ".join(failed)
            + ". Their energies are not those of a minimum, so the relative stability that "
              "uses them is not comparable."
        )


def compute_stability(
    mixed_out: "str | Path",
    reference_outs: List["str | Path"],
    *,
    coefficients: Optional[list] = None,
    mixed_label: str = "",
    reference_labels: Optional[List[str]] = None,
    energy: str = DEFAULT_ENERGY_BASIS,
) -> StabilityResult:
    """Relative (formation) stability of a mixed oxide vs. its pure-oxide references.

    ``coefficients`` overrides the auto-solved stoichiometry (one per reference, same order as
    ``reference_outs``; int, float or a fraction such as "1/2"). When omitted, coefficients are solved to balance the
    mixed formula against the references. ``energy`` picks the basis (module docstring).
    """
    if coefficients is not None and len(coefficients) != len(reference_outs):
        raise ValueError(f"{len(coefficients)} coefficients for {len(reference_outs)} references; "
                         "pass one per reference, in the same order.")
    mixed_state, engine = _parse_output(mixed_out)
    mixed_atoms = mixed_state.structure.atoms
    mixed_counts = dict(Counter(mixed_atoms))

    ref_parsed = [_parse_output(p) for p in reference_outs]
    ref_states = [s for s, _ in ref_parsed]
    ref_counts = [dict(Counter(s.structure.atoms)) for s in ref_states]
    ref_labels = reference_labels or [_formula(s.structure.atoms) for s in ref_states]

    mismatch = describe_level_mismatch(
        {"mixed": mixed_out, **{f"ref[{i}]": p for i, p in enumerate(reference_outs)}})

    named_states = [("mixed", mixed_state)] + list(zip(ref_labels, ref_states))
    detect_symmetries([mixed_state, *ref_states])
    basis, fallback = _resolve_basis(energy, named_states)

    result = StabilityResult(
        mixed=mixed_label or mixed_state.structure.structure_id or _formula(mixed_atoms),
        formula=_formula(mixed_atoms),
        n_atoms=len(mixed_atoms),
        engine=engine,
        e_mixed_hartree=energy_in_basis(mixed_state, basis),
        energy_basis=basis,
        temperature_k=mixed_state.temperature_kelvin if basis in ("H", "G", "Gq") else None,
        mixed_atoms=list(mixed_atoms),
        mixed_positions=np.asarray(mixed_state.structure.positions, dtype=float).tolist(),
    )
    if fallback:
        result.warnings.append(fallback)
    _warn_unconverged(result, [("mixed", mixed_out)]
                      + [(lbl, pth) for lbl, pth in zip(ref_labels, reference_outs)])
    _check_terms(result, [("mixed", mixed_out, mixed_state, engine)]
                 + [(lbl, pth, s, eng) for lbl, pth, (s, eng)
                    in zip(ref_labels, reference_outs, ref_parsed)],
                 basis, frequencies_missing_reported=fallback is not None)

    if mismatch:
        result.warnings.append(mismatch)

    if coefficients is not None:
        try:
            coeffs = [Fraction(c).limit_denominator(_MAX_DENOMINATOR) for c in coefficients]
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid coefficient: {exc}") from None
        balanced = _balances(mixed_counts, ref_counts, coeffs)
    else:
        coeffs, balanced = _solve_coefficients(mixed_counts, ref_counts)

    result.balanced = balanced
    result.formula_units = _formula_units(mixed_atoms)
    result.reference_size = _reference_size(
        result.formula_units, coeffs, [_formula_units(s.structure.atoms) for s in ref_states])
    if not balanced:
        result.warnings.append(
            f"Stoichiometry does not balance: {result.formula} vs "
            f"{' + '.join(f'{c}·{lbl}' for c, lbl in zip(coeffs, ref_labels))}. "
            "Pass explicit --coeff values."
        )

    result.references = [
        StabilityReference(label=lbl, formula=_formula(s.structure.atoms),
                           coefficient=_coefficient(c), e_hartree=energy_in_basis(s, basis))
        for lbl, c, s in zip(ref_labels, coeffs, ref_states)
    ]

    for other in ENERGY_BASES:
        values = [energy_in_basis(s, other) for s in [mixed_state, *ref_states]]
        if all(v is not None for v in values):
            result.e_rel_by_basis_kcal[other] = (
                values[0] - sum(c * e for c, e in zip(coeffs, values[1:]))) * _HARTREE_TO_KCAL

    energies = [ref.e_hartree for ref in result.references]
    if result.e_mixed_hartree is None or any(e is None for e in energies):
        result.warnings.append("Missing total energy in the mixed or a reference output; E_rel not computed.")
        return result

    e_rel = result.e_mixed_hartree - sum(c * e for c, e in zip(coeffs, energies))
    result.e_rel_hartree = e_rel
    result.e_rel_kcal = e_rel * _HARTREE_TO_KCAL
    result.e_rel_eV = e_rel * _HARTREE_TO_EV
    result.e_rel_eV_per_atom = result.e_rel_eV / result.n_atoms if result.n_atoms else None
    result.e_rel_kcal_per_fu = result.e_rel_kcal / result.formula_units
    return result


# ── Size-dependent stability descriptors along a cluster series ──────────────
# Definitions for the series S_n = (TiAl2O5)_n:
#   E_b   = E[S_n]/n − (E[TiO2] + E[Al2O3])          (binding energy per formula unit)
#   E_f   = E[S_(n-1)] + E[S_1] − E[S_n]             (fragmentation energy)
#   Δ²E   = E[S_(n-1)] + E[S_(n+1)] − 2·E[S_n]       (second-order energy difference)
# and, against the series' OWN monomer instead of the pure oxides (any family, pure or mixed):
#   E_b,fam = (n·E[S_1] − E[S_n])/n                  (positive when the cluster is bound)
# E_b and E_b,fam answer different questions and are kept under different names.
# With pure references of any size (--ref), E_b is E_rel/n against the size-1
# references, ΔE_mix(n) is E_rel against those of the same n, and ΔE_mix,eq is E_rel/n against
# those of equal size (½·(Al2O3)_2n + ½·(TiO2)_2n).
# Several clusters with the same n are isomers: the lowest in the chosen basis
# represents n in the descriptors, and the E0 and G minima are both reported.


@dataclass
class SeriesMember:
    n:            int
    label:        str
    formula:      str
    e_hartree:    Optional[float] = None
    e_b_kcal:     Optional[float] = None   # binding energy per formula unit
    e_b_eV:       Optional[float] = None
    e_b_family_kcal: Optional[float] = None   # per f.u., against the series' own monomer
    e_b_family_eV:   Optional[float] = None
    e_f_kcal:     Optional[float] = None   # fragmentation energy
    e_f_eV:       Optional[float] = None
    delta2_kcal:  Optional[float] = None   # second-order energy difference
    delta2_eV:    Optional[float] = None
    de_mix_kcal:  Optional[float] = None   # vs. pure references of the same n
    de_mix_eV:    Optional[float] = None
    de_mix_eq_kcal: Optional[float] = None   # per f.u., vs. pure references of equal size
    de_mix_eq_eV:   Optional[float] = None


@dataclass
class SeriesIsomer:
    n:              int
    label:          str
    formula:        str
    point_group:    Optional[str] = None
    chiral:         Optional[bool] = None
    representative: bool = False
    # Δ against the lowest isomer of the same n, one basis at a time (E0, G and the chosen one).
    delta_kcal:     Dict[str, float] = field(default_factory=dict)
    degeneracy:     Optional[int] = None     # g = 2 if chiral; set with the population
    population:     Optional[float] = None   # Boltzmann over ΔG (σ-corrected)
    population_e0:  Optional[float] = None   # Boltzmann over ΔE0, same g; σ is not in E0


@dataclass
class SeriesStabilityResult:
    members:          List[SeriesMember] = field(default_factory=list)
    engine:           str = ""
    energy_basis:     str = "E"
    references:       List[dict] = field(default_factory=list)
    isomers:          List[SeriesIsomer] = field(default_factory=list)
    population_basis: Optional[str] = None
    population_temperature_k: Optional[float] = None
    # Descriptors from Σ P_i·E_i per n: a weighted average, not the free energy of the ensemble.
    weighted_members: List[SeriesMember] = field(default_factory=list)
    multiplicities:   Dict[str, int] = field(default_factory=dict)
    n_imaginary:      Dict[str, Optional[int]] = field(default_factory=dict)
    symmetry:         Dict[str, dict] = field(default_factory=dict)    # detected
    notes:            List[str] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _family(atoms: List[str]) -> str:
    u = _formula_units(atoms)
    return hill_formula([el for el, k in Counter(atoms).items() for _ in range(k // u)])


def _set(member: SeriesMember, name: str, hartree: float) -> None:
    setattr(member, f"{name}_kcal", hartree * _HARTREE_TO_KCAL)
    setattr(member, f"{name}_eV", hartree * _HARTREE_TO_EV)


def _descriptors(members: Dict[int, SeriesMember], counts: Dict[int, dict], ref_sets, basis) -> None:
    """Fill every descriptor of each member from its own e_hartree and its neighbours'."""
    e_by_n = {n: m.e_hartree for n, m in members.items()}
    e1 = e_by_n.get(1)
    for n, m in members.items():
        en = m.e_hartree
        if en is None:
            continue
        for size, refs in sorted(ref_sets.items()):
            states = list(refs.values())
            coeffs, ok = _solve_coefficients(counts[n], [dict(Counter(s.structure.atoms)) for s in states])
            if not ok or any(energy_in_basis(s, basis) is None for s in states):
                continue
            e_rel = en - sum(c * energy_in_basis(s, basis) for c, s in zip(coeffs, states) if c)
            if size == 1:
                _set(m, "e_b", e_rel / n)
            kind = _reference_size(n, coeffs, [size] * len(states))
            if kind == "same-n":
                _set(m, "de_mix", e_rel)
            elif kind == "equal-size":
                _set(m, "de_mix_eq", e_rel / n)
        if e1 is not None and n > 1:
            _set(m, "e_b_family", (n * e1 - en) / n)
        e_nm1, e_np1 = e_by_n.get(n - 1), e_by_n.get(n + 1)
        if e_nm1 is not None and e1 is not None:
            _set(m, "e_f", e_nm1 + e1 - en)
        if e_nm1 is not None and e_np1 is not None:
            _set(m, "delta2", e_nm1 + e_np1 - 2 * en)


def _isomers(result, n: int, isos, basis: str, representative: str, populations: bool) -> None:
    """Rows of the isomers of one n: Δ per basis, the E0 and G minima, and the population."""
    rows = [SeriesIsomer(n=n, label=label, formula=_formula(s.structure.atoms),
                         point_group=getattr(s.geometry_symmetry, "point_group", None),
                         chiral=getattr(s.geometry_symmetry, "chiral", None),
                         representative=label == representative)
            for label, s in isos]
    lowest = {}
    for b in dict.fromkeys([basis, "E0", "G"]):
        values = [energy_in_basis(s, b) for _, s in isos]
        if any(v is None for v in values):
            continue
        low = min(values)
        lowest[b] = rows[values.index(low)].label
        for row, v in zip(rows, values):
            row.delta_kcal[b] = (v - low) * _HARTREE_TO_KCAL
    if "E0" in lowest and "G" in lowest and lowest["E0"] != lowest["G"]:
        result.notes.append(
            f"n={n}: the E0 minimum ({lowest['E0']}) and the G minimum ({lowest['G']}) are "
            f"different isomers; both are kept, and {representative} represents n in the "
            f"{basis} descriptors.")
    result.isomers.extend(rows)
    if not populations:
        return

    pb = result.population_basis
    states = [s for _, s in isos]
    if _boltzmann(rows, states, pb, "population", result, n):
        _boltzmann(rows, states, "E0", "population_e0", result, n, quiet=True)


def _boltzmann(rows, states, basis: str, attr: str, result, n: int, quiet: bool = False) -> bool:
    """P_i = g_i·exp(−ΔX_i/RT)/Σ over the isomers that have *basis*, at the outputs' T."""
    included = [(row, s, energy_in_basis(s, basis)) for row, s in zip(rows, states)]
    excluded = [row.label for row, _, x in included if x is None]
    included = [t for t in included if t[2] is not None]
    if excluded and not quiet:
        result.warnings.append(f"n={n}: no {basis} (no frequency calculation) for "
                               f"{', '.join(excluded)}; left out of the population.")
    temps = {s.temperature_kelvin for _, s, _ in included}
    if not included or len(temps) != 1 or None in temps:
        if not quiet:
            result.warnings.append(f"n={n}: no population ({basis} missing, or computed at "
                                   "different temperatures).")
        return False
    t = temps.pop()
    result.population_temperature_k = t
    low = min(x for _, _, x in included)
    weights = []
    for row, s, x in included:
        row.degeneracy = 2 if s.geometry_symmetry.chiral else 1
        weights.append(row.degeneracy * exp(-(x - low) / (_KB_HARTREE * t)))
    for (row, _, _), w in zip(included, weights):
        setattr(row, attr, w / sum(weights))
    return True


def _unique_labels(terms: list) -> list:
    """Terms whose labels collide (``opt.out`` in n1/ and in n2/) get their folder, then a
    number, appended: every per-term table is keyed by label, so two equal ones would merge."""
    from collections import Counter
    count = Counter(t[0] for t in terms)
    out, seen = [], set()
    for label, path, state, eng in terms:
        new = label if count[label] == 1 else f"{label} ({Path(path).parent.name or '.'})"
        k = 2
        base = new
        while new in seen:
            new, k = f"{base} #{k}", k + 1
        seen.add(new)
        out.append((new, path, state, eng))
    return out


def compute_series_stability(
    cluster_outs: List["str | Path"],
    *,
    reference_outs: List["str | Path"] = (),
    labels: Optional[List[str]] = None,
    energy: str = DEFAULT_ENERGY_BASIS,
    populations: bool = False,
) -> SeriesStabilityResult:
    """Size-dependent stability descriptors along a cluster series.

    ``cluster_outs`` are the output files of the clusters of ONE family (mixed or pure); the
    number of formula units n is inferred per cluster from its stoichiometry (gcd of atom
    counts), and clusters sharing n are isomers. ``reference_outs`` are pure-oxide
    outputs of any size, matched by stoichiometry; they give E_b, ΔE_mix(n)
    and ΔE_mix,eq. ``populations`` adds the Boltzmann population of the isomers and the
    descriptors averaged with it. ``energy`` picks the basis (module docstring), shared by all.
    """
    result = SeriesStabilityResult()

    def parse(path, label=None):
        state, eng = _parse_output(path)
        result.engine = result.engine or eng
        return (label or state.structure.structure_id or _formula(state.structure.atoms),
                path, state, eng)

    clusters = [parse(p, labels[i] if labels and i < len(labels) else None)
                for i, p in enumerate(cluster_outs)]
    refs = [parse(p) for p in reference_outs]
    terms = _unique_labels(refs + clusters)
    refs, clusters = terms[:len(refs)], terms[len(refs):]

    _warn_unconverged(result, [(t[0], t[1]) for t in terms])
    detect_symmetries([t[2] for t in terms])
    basis, fallback = _resolve_basis(energy, [(t[0], t[2]) for t in terms])
    result.energy_basis = basis
    result.population_basis = ("Gq" if basis == "Gq" else "G") if populations else None
    if fallback:
        result.warnings.append(fallback)
    _check_terms(result, terms, basis, frequencies_missing_reported=fallback is not None)

    ref_sets: Dict[int, Dict[str, object]] = {}
    for label, _, state, _ in refs:
        atoms = state.structure.atoms
        size, family = _formula_units(atoms), _family(atoms)
        result.references.append({"label": label, "formula": _formula(atoms), "n": size,
                                  "e_hartree": energy_in_basis(state, basis)})
        sized = ref_sets.setdefault(size, {})
        if family in sized:
            result.warnings.append(f"Two references {family} of size {size}; keeping "
                                   f"{sized[family].structure.structure_id or 'the first'}.")
            continue
        sized[family] = state

    groups: Dict[int, list] = {}
    for label, _, state, _ in clusters:
        n = _formula_units(state.structure.atoms)
        group = groups.setdefault(n, [])
        if group and _formula(state.structure.atoms) != _formula(group[0][1].structure.atoms):
            result.warnings.append(f"n={n} ({label}) has another composition than "
                                   f"{group[0][0]}: not an isomer; skipped.")
            continue
        group.append((label, state))

    members: Dict[int, SeriesMember] = {}
    counts: Dict[int, dict] = {}
    for n, isos in sorted(groups.items()):
        label, state = min(isos, key=lambda t: (energy_in_basis(t[1], basis) is None,
                                                energy_in_basis(t[1], basis) or 0.0, t[0]))
        members[n] = SeriesMember(n=n, label=label, formula=_formula(state.structure.atoms),
                                  e_hartree=energy_in_basis(state, basis))
        counts[n] = dict(Counter(state.structure.atoms))
        if members[n].e_hartree is None:
            result.warnings.append(f"Missing total energy for n={n} ({label}); descriptors skipped.")
        if len(isos) > 1:
            _isomers(result, n, sorted(isos, key=lambda t: t[0]), basis, label, populations)

    _descriptors(members, counts, ref_sets, basis)
    result.members = [members[n] for n in sorted(members)]

    if not ref_sets:
        result.warnings.append("E_b and ΔE_mix not computed: pass the pure-oxide outputs with --ref.")
    else:
        for name, attr, needed in (("E_b", "e_b_kcal", "size 1"),
                                   ("ΔE_mix(n)", "de_mix_kcal", "the same n"),
                                   ("ΔE_mix,eq", "de_mix_eq_kcal", "equal size")):
            missing = [str(m.n) for m in result.members if getattr(m, attr) is None]
            if missing:
                result.warnings.append(f"{name} not computed for n={', '.join(missing)}: no "
                                       f"balanced set of pure references of {needed}.")

    if populations:
        weighted = {}
        for n, m in members.items():
            rows = [r for r in result.isomers if r.n == n]
            if not rows:
                e = m.e_hartree
            elif any(r.population is not None for r in rows):
                by_label = dict(groups[n])
                e = sum(r.population * energy_in_basis(by_label[r.label], basis)
                        for r in rows if r.population is not None)
            else:
                e = None
            weighted[n] = SeriesMember(n=n, label="weighted", formula=m.formula, e_hartree=e)
        _descriptors(weighted, counts, ref_sets, basis)
        result.weighted_members = [weighted[n] for n in sorted(weighted)]
    return result
