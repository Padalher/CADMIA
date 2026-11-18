"""Analysis Agent — post-calculation adsorption energetics and probe activation.

Activated after the Validation Agent confirms the calculations are valid.
Given the three validated outputs — adduct (cluster+probe), isolated cluster, and free
probe — it computes the adsorption energy, optional BSSE (counterpoise) correction,
optional deformation/interaction energies, the probe-bond elongation and activation
flag, and cross-references the CDFT site ranking. The result maps to one row of the
``adsorption`` table per (cluster, probe, config) triplet.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from cadmia.core.elements import hill_formula
from cadmia.core.units import HARTREE_TO_KCAL as _HARTREE_TO_KCAL
from cadmia.io.electronic import load_electronic as _parse_output
from cadmia.io.electronic import describe_level_mismatch
from cadmia.utils.chemistry import covalent_radius, skeleton as _skeleton
from cadmia.utils.geometry import kabsch_deviations

_BOND_CUTOFF = 1.3
_ACTIVATION_PCT = 5.0
_LINEAR_DEG = 175.0
_ACTIVATION_BEND_DEG = 10.0


def _electronic_descriptors(state) -> Optional[dict]:
    """HOMO/LUMO/gap + global Koopmans descriptors (μ, η, ω) + |dipole| for one state.

    Returns a compact dict so the report can compare the cluster (before) and the
    adduct (after) adsorption. Falls back gracefully when HOMO/LUMO are missing or the
    gap is non-positive; the dipole magnitude (Debye) is included when present and is
    independent of the frontier orbitals. Returns None only when nothing is available.
    """
    homo, lumo = getattr(state, "homo_ev", None), getattr(state, "lumo_ev", None)
    out: dict = {}
    if homo is not None and lumo is not None:
        out.update(homo_ev=float(homo), lumo_ev=float(lumo), gap_ev=float(lumo - homo))
        try:
            from cadmia.core.electronic_descriptors import compute_global_descriptors
            gd = compute_global_descriptors(homo, lumo)
            out.update(mu_ev=gd.mu_ev, eta_ev=gd.eta_ev, omega_ev=gd.omega_ev,
                       omega_minus_ev=gd.omega_minus_ev, omega_plus_ev=gd.omega_plus_ev)
        except ValueError:
            pass
    dipole = getattr(state, "dipole_debye", None)
    if dipole is not None:
        out["dipole_debye"] = float(np.linalg.norm(np.asarray(dipole, dtype=float)))
    return out or None


def _angle(pos, i: int, c: int, j: int) -> float:
    """Angle i–c–j in degrees."""
    u, v = np.asarray(pos[i]) - np.asarray(pos[c]), np.asarray(pos[j]) - np.asarray(pos[c])
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _counterpoise_bsse_hartree(adduct: str | Path) -> Optional[float]:
    """BSSE from the four ghost-atom outputs that `cadmia prepare bsse` sets up (ORCA), looked
    for next to the adduct and in the working directory; None unless all four are there.

    BSSE = [E_cl(own basis) − E_cl(full)] + [E_pr(own) − E_pr(full)], all at the adduct
    geometry: positive, and added to E_ads.
    """
    from cadmia.agents.calculation_agent import COUNTERPOISE_TERMS
    stem = Path(adduct).stem
    for folder in (Path(adduct).parent, Path.cwd()):
        files = [folder / f"{stem}_cp_{t}.out" for t in COUNTERPOISE_TERMS]
        if all(f.is_file() for f in files):
            e = [_parse_output(f)[0].energy_hartree for f in files]
            if None in e:
                return None
            return (e[0] - e[1]) + (e[2] - e[3])
    return None


def _parse_bsse_hartree(path: str | Path) -> Optional[float]:
    """Extract the counterpoise BSSE energy (Hartree) from a Gaussian/ORCA output.

    Gaussian (Counterpoise=N) prints ``Counterpoise: BSSE energy = <Ha>``; ORCA prints
    a ``BSSE correction`` / ``counterpoise`` line. Returns None when absent.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"Counterpoise:\s*BSSE energy\s*=\s*([-+]?\d*\.\d+)", text)
    if hits:
        return float(hits[-1])
    hits = re.findall(r"BSSE\s*(?:correction|energy)[^\n=:]*[=:]\s*([-+]?\d*\.\d+)",
                      text, re.IGNORECASE)
    if hits:
        return float(hits[-1])
    return None


def _intramolecular_bonds(
    atoms: List[str], positions: np.ndarray, cutoff: float = _BOND_CUTOFF,
) -> List[Tuple[int, int, float]]:
    """Return (i, j, distance) for atom pairs within ``cutoff`` × sum of covalent radii."""
    bonds: List[Tuple[int, int, float]] = []
    n = len(atoms)
    for i in range(n):
        ri = covalent_radius(atoms[i]) or 0.77
        for j in range(i + 1, n):
            rj = covalent_radius(atoms[j]) or 0.77
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d <= cutoff * (ri + rj):
                bonds.append((i, j, d))
    return bonds


def _formula(atoms: List[str]) -> str:
    return hill_formula(atoms)


@dataclass
class ProbeBond:
    label:      str
    free_A:     float
    adduct_A:   float
    delta_A:    float
    percent:    float


@dataclass
class AnalysisResult:
    cluster:            str
    probe:              str
    config_index:       Optional[int] = None
    adduct_path:        Optional[str] = None
    site_atom_index:    Optional[int] = None
    site_atom_symbol:   Optional[str] = None
    engine:             str = ""

    cluster_atoms:      List[str] = field(default_factory=list)
    cluster_positions:  List[List[float]] = field(default_factory=list)

    e_adduct_hartree:   Optional[float] = None
    e_cluster_hartree:  Optional[float] = None
    e_probe_hartree:    Optional[float] = None
    e_ads_kcal:         Optional[float] = None

    bsse_kcal:          Optional[float] = None
    e_ads_bsse_kcal:    Optional[float] = None

    e_def_cluster_kcal: Optional[float] = None
    e_def_probe_kcal:   Optional[float] = None
    e_int_kcal:         Optional[float] = None

    delta_h_ads_kcal:   Optional[float] = None
    delta_g_ads_kcal:   Optional[float] = None
    delta_gq_ads_kcal:  Optional[float] = None
    minus_t_delta_s_kcal: Optional[float] = None
    temperature_kelvin: Optional[float] = None

    electronic_cluster: Optional[dict] = None
    electronic_adduct:  Optional[dict] = None
    delta_gap_ev:       Optional[float] = None
    delta_dipole_debye: Optional[float] = None

    charge_transfer:    Optional[dict] = None

    site_mayer_bonds:   List[dict] = field(default_factory=list)

    charge_scheme:      Optional[str] = None
    q_transfer_to_probe: Optional[float] = None
    delta_q_site:       Optional[float] = None

    adduct_imaginary_modes: Optional[int] = None
    adduct_ir_bands:    List[dict] = field(default_factory=list)

    site_cn_cluster:    Optional[int] = None
    site_ecn_cluster:   Optional[float] = None
    site_cn_adduct:     Optional[int] = None
    site_ecn_adduct:    Optional[float] = None
    delta_cn_site:      Optional[int] = None
    delta_ecn_site:     Optional[float] = None

    cluster_rearranged: Optional[bool] = None
    cluster_bond_changes: dict = field(default_factory=dict)
    cluster_displacement_ang: List[float] = field(default_factory=list)
    delta_r_site_ang:   Optional[float] = None

    probe_bonds:        List[ProbeBond] = field(default_factory=list)
    max_elongation_pct: Optional[float] = None
    probe_angles:       List[dict] = field(default_factory=list)
    max_bend_deg:       Optional[float] = None
    activated:          bool = False

    cdft_site_rank:     Optional[int] = None
    cdft_site_character: Optional[str] = None
    cdft_agreement:     Optional[bool] = None

    warnings:           List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["probe_bonds"] = [asdict(b) for b in self.probe_bonds]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _load_cdft(cdft_json: "str | Path | dict") -> dict:
    if isinstance(cdft_json, dict):
        return cdft_json
    text = Path(cdft_json).read_text(encoding="utf-8") if Path(str(cdft_json)).exists() else str(cdft_json)
    return json.loads(text)


def _adduct_energy_is_cp_corrected(path) -> bool:
    """True when the adduct log reports a counterpoise-corrected energy (Gaussian only)."""
    try:
        from cadmia.io.electronic import gaussian_energy_is_counterpoise_corrected
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return gaussian_energy_is_counterpoise_corrected(handle.read().splitlines())
    except Exception:
        return False


def _unconverged_roles(named_paths) -> List[str]:
    """Which of the given (role, path, engine) triples hold an optimization that did not
    converge. Each output is checked with its own engine's markers: a Gaussian cluster next to
    an ORCA adduct is not read for ORCA's.

    A single point has no optimization to converge, so it is never listed: `geometry_converged`
    returns None there and that is not a failure.
    """
    from cadmia.agents.validation_agent import geometry_converged

    failed: List[str] = []
    for role, path, engine in named_paths:
        if path is None:
            continue
        try:
            if geometry_converged(path, engine) is False:
                failed.append(role)
        except Exception:
            continue
    return failed


_PROBE_GROUND_MULTIPLICITY = {("O", "O"): 3}


def spin_and_charge_warnings(adduct, cluster, probe) -> List[str]:
    """Whether the adduct is the cluster plus the probe in charge and in spin.

    E_ads subtracts three energies; if the charges do not add up the adduct is a different
    system, and a multiplicity the two spins cannot couple to is one too. A probe computed in an
    excited spin state (singlet O2) shifts E_ads by that excitation without any other sign.
    """
    out = []
    q_sum = int(cluster.charge or 0) + int(probe.charge or 0)
    if int(adduct.charge or 0) != q_sum:
        out.append(f"Charge does not add up: the adduct has charge {adduct.charge}, the cluster "
                   f"plus the probe {q_sum}. E_ads then compares different systems.")
    s_cl, s_pr = (int(cluster.multiplicity or 1) - 1) / 2, (int(probe.multiplicity or 1) - 1) / 2
    allowed = [int(2 * s + 1) for s in _spin_range(abs(s_cl - s_pr), s_cl + s_pr)]
    if int(adduct.multiplicity or 1) not in allowed:
        out.append(f"Multiplicity does not couple: the adduct is {adduct.multiplicity}, but a "
                   f"cluster of multiplicity {cluster.multiplicity} and a probe of "
                   f"{probe.multiplicity} can only give {', '.join(map(str, allowed))}.")
    ground = _PROBE_GROUND_MULTIPLICITY.get(tuple(sorted(probe.atoms)))
    if ground and int(probe.multiplicity or 1) != ground:
        out.append(f"The free probe was computed with multiplicity {probe.multiplicity}; its "
                   f"ground state is {ground}. E_ads is then referenced to an excited probe.")
    return out


def _spin_range(lo: float, hi: float) -> List[float]:
    n = int(round(hi - lo))
    return [lo + k for k in range(n + 1)]


def analyze(
    adduct: "str | Path",
    cluster: "str | Path",
    probe_free: "str | Path",
    *,
    probe: Optional[str] = None,
    bsse: bool = False,
    cluster_deformed: "str | Path | None" = None,
    probe_deformed: "str | Path | None" = None,
    cdft_json: "str | Path | dict | None" = None,
    config_index: Optional[int] = None,
    cluster_label: str = "",
    bend_threshold_deg: float = _ACTIVATION_BEND_DEG,
) -> AnalysisResult:
    """Compute adsorption energetics and activation for one (cluster, probe, config) triplet."""
    adduct_state, engine = _parse_output(adduct)
    adduct_source = str(Path(adduct).resolve())
    cluster_state, cluster_engine = _parse_output(cluster)
    probe_state, probe_engine = _parse_output(probe_free)

    ad_atoms = adduct_state.structure.atoms
    ad_pos = np.asarray(adduct_state.structure.positions, dtype=float)
    cl_atoms = cluster_state.structure.atoms
    pr_atoms = probe_state.structure.atoms
    pr_pos = np.asarray(probe_state.structure.positions, dtype=float)

    result = AnalysisResult(
        cluster=cluster_label or cluster_state.structure.structure_id or "cluster",
        probe=probe or _formula(pr_atoms),
        config_index=config_index,
        adduct_path=adduct_source,
        engine=engine,
        e_adduct_hartree=adduct_state.energy_hartree,
        e_cluster_hartree=cluster_state.energy_hartree,
        e_probe_hartree=probe_state.energy_hartree,
        cluster_atoms=list(cl_atoms),
        cluster_positions=np.asarray(cluster_state.structure.positions, dtype=float).tolist(),
    )
    mismatch = describe_level_mismatch(
        {"adduct": adduct, "cluster": cluster, "probe": probe_free})
    if mismatch:
        result.warnings.append(mismatch)
    result.warnings += spin_and_charge_warnings(
        adduct_state.structure, cluster_state.structure, probe_state.structure)

    unconverged = _unconverged_roles((("adduct", adduct, engine), ("cluster", cluster, cluster_engine),
                                      ("probe", probe_free, probe_engine)))

    if unconverged:
        result.warnings.append(
            "Geometry NOT converged in: " + ", ".join(unconverged)
            + ". E_ads is not computed: an energy from a step that is not a minimum is not "
              "comparable with one that is."
        )
    elif None not in (adduct_state.energy_hartree, cluster_state.energy_hartree, probe_state.energy_hartree):
        e_ads_ha = adduct_state.energy_hartree - cluster_state.energy_hartree - probe_state.energy_hartree
        result.e_ads_kcal = e_ads_ha * _HARTREE_TO_KCAL
    else:
        result.warnings.append("Missing total energy in one of the outputs; E_ads not computed.")

    from cadmia.agents.stability_agent import detect_symmetries, energy_in_basis
    states = (adduct_state, cluster_state, probe_state)
    detect_symmetries(states)
    g = [None] * 3 if unconverged else [energy_in_basis(s, "G") for s in states]
    gq = [None] * 3 if unconverged else [energy_in_basis(s, "Gq") for s in states]
    h = [None] * 3 if unconverged else [s.enthalpy_hartree for s in states]
    if None not in gq:
        result.delta_gq_ads_kcal = (gq[0] - gq[1] - gq[2]) * _HARTREE_TO_KCAL
    if None not in g:
        resym = [f"{name} {(gi - s.gibbs_free_energy_hartree) * _HARTREE_TO_KCAL:+.2f}"
                 for name, s, gi in zip(("adduct", "cluster", "probe"), states, g)
                 if abs(gi - s.gibbs_free_energy_hartree) > 1e-12]
        if resym:
            result.warnings.append(
                "G corrected to the symmetry number of each geometry, RT·ln(σ/σ_engine), in "
                f"kcal/mol: {', '.join(resym)}.")
        result.delta_g_ads_kcal = (g[0] - g[1] - g[2]) * _HARTREE_TO_KCAL
        result.temperature_kelvin = adduct_state.temperature_kelvin
        temps = [s.temperature_kelvin for s in (adduct_state, cluster_state, probe_state)]
        if all(t is not None for t in temps) and max(temps) - min(temps) > 0.1:
            result.warnings.append(
                "ΔG_ads combines Gibbs energies computed at different temperatures "
                f"({', '.join(f'{t:.2f}' for t in temps)} K); the value assumes a common T.")
    if None not in h:
        result.delta_h_ads_kcal = (h[0] - h[1] - h[2]) * _HARTREE_TO_KCAL
    if result.delta_g_ads_kcal is not None and result.delta_h_ads_kcal is not None:
        result.minus_t_delta_s_kcal = result.delta_g_ads_kcal - result.delta_h_ads_kcal
    elif result.e_ads_kcal is not None and not unconverged and (None in g or None in h):
        missing = [name for name, st in (("adduct", adduct_state), ("cluster", cluster_state),
                                         ("probe", probe_state))
                   if st.gibbs_free_energy_hartree is None]
        result.warnings.append(
            "ΔG/ΔH_ads not computed: no thermochemistry (run opt+freq) in "
            f"{', '.join(missing)}."
        )

    result.electronic_cluster = _electronic_descriptors(cluster_state)
    result.electronic_adduct = _electronic_descriptors(adduct_state)
    if (result.electronic_cluster and result.electronic_adduct
            and result.electronic_cluster.get("gap_ev") is not None
            and result.electronic_adduct.get("gap_ev") is not None):
        result.delta_gap_ev = result.electronic_adduct["gap_ev"] - result.electronic_cluster["gap_ev"]
    if (result.electronic_cluster and result.electronic_adduct
            and result.electronic_cluster.get("dipole_debye") is not None
            and result.electronic_adduct.get("dipole_debye") is not None):
        result.delta_dipole_debye = (result.electronic_adduct["dipole_debye"]
                                     - result.electronic_cluster["dipole_debye"])

    ecp = _electronic_descriptors(probe_state)
    ecl = result.electronic_cluster
    if (ecl and ecp
            and ecl.get("mu_ev") is not None and ecl.get("eta_ev") is not None
            and ecp.get("mu_ev") is not None and ecp.get("eta_ev") is not None):
        try:
            from cadmia.core.charge_transfer import compute_charge_transfer
            ct = compute_charge_transfer(
                ecl["mu_ev"], ecl["eta_ev"], ecp["mu_ev"], ecp["eta_ev"],
                label_a="cluster", label_b="probe",
            )
            result.charge_transfer = {
                "delta_n": round(ct.delta_n, 4),
                "e_stab_ev": round(ct.delta_e_ev, 4),
                "donor": ct.donor,
                "acceptor": ct.acceptor,
                "mu_cluster_ev": round(ct.mu_a_ev, 4),
                "eta_cluster_ev": round(ct.eta_a_ev, 4),
                "mu_probe_ev": round(ct.mu_b_ev, 4),
                "eta_probe_ev": round(ct.eta_b_ev, 4),
            }
        except ValueError as exc:
            result.warnings.append(f"CDFT charge transfer not computed: {exc}")

    n_cluster = len(cl_atoms)
    n_probe = len(ad_atoms) - n_cluster
    if n_probe != len(pr_atoms):
        result.warnings.append(
            f"Adduct has {len(ad_atoms)} atoms, cluster {n_cluster}, free probe {len(pr_atoms)}: "
            f"inferred {n_probe} probe atoms in the adduct — verify atom ordering "
            "(cluster first, probe appended)."
        )
    if 0 < n_probe <= len(ad_atoms) and n_probe == len(pr_atoms):
        probe_block = ad_pos[n_cluster:]
        cluster_block = ad_pos[:n_cluster]
        if list(ad_atoms[:n_cluster]) != list(cl_atoms):
            result.warnings.append(
                "The adduct's first atoms do not match the isolated cluster's element order; "
                "the binding-site symbol and per-site Δq may be unreliable."
            )
        else:
            cl_pos = np.asarray(cluster_state.structure.positions, dtype=float)
            result.cluster_displacement_ang = [
                round(float(v), 4) for v in kabsch_deviations(cluster_block, cl_pos)]
            sig_ad, bonds_ad, edge_ad = _skeleton(list(cl_atoms), cluster_block)
            sig_cl, bonds_cl, edge_cl = _skeleton(list(cl_atoms), cl_pos)
            result.cluster_rearranged = sig_ad != sig_cl
            if result.cluster_rearranged:
                result.cluster_bond_changes = {
                    b: bonds_ad[b] - bonds_cl[b]
                    for b in set(bonds_ad) | set(bonds_cl) if bonds_ad[b] != bonds_cl[b]}
                result.warnings.append(
                    "The cluster's own bonding changes on adsorption ("
                    + ", ".join(f"{b} {d:+d}"
                                for b, d in sorted(result.cluster_bond_changes.items()))
                    + "). That is substrate reorganization and it belongs in E_ads — unless the "
                      "rearranged cluster is a separate minimum of the free cluster, in which "
                      "case E_ads also carries the isomerization energy and should be referenced "
                      "against that minimum instead. Optimize the adduct's cluster on its own to "
                      "tell the two apart: if it relaxes back, E_ads stands as it is."
                )
            if edge_ad or edge_cl:
                result.warnings.append(
                    "Some cluster distances sit close to the bond/no-bond cutoff, so the verdict "
                    "above is not safe on geometry alone; check this one by hand."
                )
        dmat = np.linalg.norm(cluster_block[:, None, :] - probe_block[None, :, :], axis=2)
        site_local = int(np.argmin(dmat.min(axis=1)))
        result.site_atom_index = site_local + 1
        if 0 <= site_local < len(result.cluster_displacement_ang):
            result.delta_r_site_ang = result.cluster_displacement_ang[site_local]
        result.site_atom_symbol = cl_atoms[site_local]

        free_bonds = _intramolecular_bonds(pr_atoms, pr_pos)
        for i, j, l_free in free_bonds:
            l_ad = float(np.linalg.norm(probe_block[i] - probe_block[j]))
            delta = l_ad - l_free
            pct = 100.0 * delta / l_free if l_free else 0.0
            result.probe_bonds.append(
                ProbeBond(label=f"{pr_atoms[i]}-{pr_atoms[j]}",
                          free_A=round(l_free, 4), adduct_A=round(l_ad, 4),
                          delta_A=round(delta, 4), percent=round(pct, 2))
            )
        if result.probe_bonds:
            worst = max(result.probe_bonds, key=lambda b: b.percent)
            result.max_elongation_pct = worst.percent
            result.activated = worst.percent >= _ACTIVATION_PCT

        neighbours: dict = {}
        for i, j, _ in free_bonds:
            neighbours.setdefault(i, []).append(j)
            neighbours.setdefault(j, []).append(i)
        for c, ends in sorted(neighbours.items()):
            for k, i in enumerate(ends):
                for j in ends[k + 1:]:
                    free_deg, ad_deg = _angle(pr_pos, i, c, j), _angle(probe_block, i, c, j)
                    result.probe_angles.append({
                        "label": f"{pr_atoms[i]}-{pr_atoms[c]}-{pr_atoms[j]}",
                        "free_deg": round(free_deg, 2), "adduct_deg": round(ad_deg, 2),
                        "delta_deg": round(ad_deg - free_deg, 2)})
                    if free_deg >= _LINEAR_DEG:
                        bend = round(free_deg - ad_deg, 2)
                        result.max_bend_deg = max(bend, result.max_bend_deg or bend)
        if result.max_bend_deg is not None and result.max_bend_deg >= bend_threshold_deg:
            result.activated = True

        q_ad = adduct_state.charges
        q_cl = cluster_state.charges
        schemes = {s.charge_scheme for s in (adduct_state, cluster_state, probe_state)
                   if s.charges is not None}
        if (q_ad is not None and len(q_ad) == len(ad_atoms)
                and q_cl is not None and len(q_cl) == n_cluster):
            if len(schemes) > 1:
                result.warnings.append(
                    "Charge transfer skipped: adduct and cluster use different charge "
                    f"schemes ({', '.join(sorted(schemes))})."
                )
            else:
                result.charge_scheme = adduct_state.charge_scheme
                q_probe_free = (sum(probe_state.charges)
                                if probe_state.charges is not None
                                and len(probe_state.charges) == n_probe else 0.0)
                result.q_transfer_to_probe = round(sum(q_ad[n_cluster:]) - q_probe_free, 4)
                result.delta_q_site = round(q_ad[site_local] - q_cl[site_local], 4)
    else:
        result.warnings.append("Could not isolate the probe block in the adduct; skipped bond/site analysis.")

    if bsse or _adduct_energy_is_cp_corrected(adduct):
        bsse_ha = _parse_bsse_hartree(adduct)
        if bsse_ha is None and bsse:
            bsse_ha = _counterpoise_bsse_hartree(adduct)
        if bsse_ha is not None:
            result.bsse_kcal = bsse_ha * _HARTREE_TO_KCAL
            if result.e_ads_kcal is not None:
                already_corrected = _adduct_energy_is_cp_corrected(adduct)
                result.e_ads_bsse_kcal = (result.e_ads_kcal if already_corrected
                                          else result.e_ads_kcal + result.bsse_kcal)
                if already_corrected:
                    result.warnings.append(
                        "The adduct energy is the counterpoise-corrected one, so E_ads already "
                        "includes the BSSE; it was not added a second time."
                    )
        elif bsse:
            result.warnings.append(
                "--bsse set but no counterpoise BSSE was found: not in the adduct output, and not "
                "the four `cadmia prepare bsse` outputs (<adduct>_cp_*.out) next to it.")

    if cluster_deformed is not None and cluster_state.energy_hartree is not None:
        defc, _ = _parse_output(cluster_deformed)
        if defc.energy_hartree is not None:
            result.e_def_cluster_kcal = (defc.energy_hartree - cluster_state.energy_hartree) * _HARTREE_TO_KCAL
    if probe_deformed is not None and probe_state.energy_hartree is not None:
        defp, _ = _parse_output(probe_deformed)
        if defp.energy_hartree is not None:
            result.e_def_probe_kcal = (defp.energy_hartree - probe_state.energy_hartree) * _HARTREE_TO_KCAL
    if (result.e_ads_kcal is not None
            and result.e_def_cluster_kcal is not None
            and result.e_def_probe_kcal is not None):
        result.e_int_kcal = result.e_ads_kcal - result.e_def_cluster_kcal - result.e_def_probe_kcal

    if cdft_json is not None and result.site_atom_index is not None:
        try:
            cdft = _load_cdft(cdft_json)
            sites = cdft.get("sites", [])
            match = next((s for s in sites if s.get("atom_index") == result.site_atom_index), None)
            if match is not None:
                character = match.get("character")
                result.cdft_site_character = character
                if character in ("nucleophilic", "electrophilic"):
                    peers = sorted(
                        (s for s in sites if s.get("character") == character
                         and s.get("delta_f") is not None),
                        key=lambda s: abs(s["delta_f"]), reverse=True,
                    )
                    char_rank = next((r for r, s in enumerate(peers, start=1)
                                      if s.get("atom_index") == result.site_atom_index), None)
                else:
                    char_rank = None
                role = {"nucleophilic": "donor", "electrophilic": "acceptor"}.get(character, character)
                if char_rank is None:
                    result.cdft_site_rank = match.get("rank")
                    result.cdft_agreement = None
                    if character == "ambiphilic":
                        result.warnings.append(
                            f"Binding site (atom {result.site_atom_index}) is ambiphilic in the "
                            f"CDFT (|Δf| below the threshold), #{match.get('rank')} of "
                            f"{len(sites)} sites by |Δf|: no donor/acceptor preference, so "
                            "agreement is not evaluated."
                        )
                    else:
                        result.warnings.append(
                            f"Binding site (atom {result.site_atom_index}) has no Δf in the CDFT "
                            "ranking; donor/acceptor agreement could not be evaluated."
                        )
                else:
                    result.cdft_site_rank = char_rank
                    result.cdft_agreement = (char_rank == 1)
                    if not result.cdft_agreement:
                        result.warnings.append(
                            f"Binding site (atom {result.site_atom_index}) is the #{char_rank} "
                            f"{character} ({role}) site, not the strongest of its character."
                        )
            else:
                result.warnings.append(
                    f"Binding site (atom {result.site_atom_index}) not found in the CDFT ranking."
                )
        except (ValueError, OSError, KeyError) as exc:
            result.warnings.append(f"Could not cross-reference CDFT ranking: {exc}")

    if result.site_atom_index is not None:
        idx0 = result.site_atom_index - 1
        try:
            from cadmia.structure.structural.summary import (site_coordination,
                                                              structural_summary)
            cl_site = site_coordination(structural_summary(cluster_state.structure), idx0)
            ad_site = site_coordination(structural_summary(adduct_state.structure), idx0)
            if cl_site:
                result.site_cn_cluster = cl_site["cn"]
                result.site_ecn_cluster = cl_site["ecn"]
            if ad_site:
                result.site_cn_adduct = ad_site["cn"]
                result.site_ecn_adduct = ad_site["ecn"]
            if cl_site and ad_site:
                result.delta_cn_site = ad_site["cn"] - cl_site["cn"]
                result.delta_ecn_site = ad_site["ecn"] - cl_site["ecn"]
        except (ValueError, KeyError, OSError, ImportError) as exc:
            result.warnings.append(f"Structural coordination not computed: {exc}")

    if result.site_atom_index is not None:
        try:
            from cadmia.io.mayer import parse_mayer
            mayer = parse_mayer(adduct)
        except (OSError, ValueError):
            mayer = None
        if mayer and mayer.bonds:
            site0 = result.site_atom_index - 1
            at_site: List[Tuple[int, str, float]] = []
            for b in mayer.bonds:
                if b.atom_i == site0:
                    at_site.append((b.atom_j, b.symbol_j, b.order))
                elif b.atom_j == site0:
                    at_site.append((b.atom_i, b.symbol_i, b.order))
            at_site.sort(key=lambda t: t[2], reverse=True)
            result.site_mayer_bonds = [
                {"partner_index": p + 1, "partner_symbol": s, "order": round(o, 4)}
                for p, s, o in at_site[:6]
            ]

    try:
        from cadmia.io.vibrational import parse_orca_vibrational
        vib = parse_orca_vibrational(adduct)
    except (OSError, ValueError):
        vib = None
    if vib is not None and vib.modes:
        result.adduct_imaginary_modes = len(vib.imaginary_modes)
        bands = sorted(vib.ir_active_modes,
                       key=lambda m: m.ir_intensity_km_mol, reverse=True)
        result.adduct_ir_bands = [
            {"frequency_cm1": round(m.frequency_cm1, 1),
             "intensity_km_mol": round(m.ir_intensity_km_mol, 1)}
            for m in bands[:6]
        ]
        if result.adduct_imaginary_modes:
            result.warnings.append(
                f"Adduct has {result.adduct_imaginary_modes} imaginary frequency(ies) — "
                "it is a saddle point, not a minimum."
            )

    return result
