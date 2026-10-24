"""CDFT workflow: takes N, N+1, N-1 electronic states → full descriptor set."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from cadmia.cdft.calculator import (
    HOMO_TCE_EV,
    FiniteDifferenceCDFT,
    FukuiResult,
    compute_condensed_fukui,
    compute_finite_difference_descriptors,
)
from cadmia.core import ElectronicState, Structure
from cadmia.core.electronic_descriptors import GlobalElectronic, compute_global_descriptors
from cadmia.io.electronic import parse_gaussian_electronic, parse_orca_electronic
from cadmia.io.gaussian import parse_gaussian_geometry
from cadmia.io.orca import parse_orca_geometry


@dataclass
class CDFTInputState:
    """Electronic data for one charge state (N, N+1, or N-1).

    All fields required for the full CDFT workflow:
    - ``energy_hartree``: total SCF energy
    - ``homo_ev``:        HOMO orbital energy (eV)
    - ``lumo_ev``:        LUMO orbital energy (eV) — only needed for state N
    - ``charges``:        per-atom charges (same scheme for all three states)
    """

    structure: Structure
    energy_hartree: float
    homo_ev: Optional[float]
    lumo_ev: Optional[float]
    charges: List[float]
    charge_scheme: str = "hirshfeld"
    source_file: str = ""
    spin_sq: Optional[float] = None
    electronic: Optional[ElectronicState] = None   # the parsed state, kept for the report

    @classmethod
    def from_electronic_state(cls, state: ElectronicState) -> CDFTInputState:
        """Wrap an existing :class:`ElectronicState` into a CDFTInputState."""
        if state.energy_hartree is None:
            raise ValueError(
                f"ElectronicState from '{state.source_file}' is missing energy_hartree."
            )
        if state.charges is None:
            raise ValueError(
                f"ElectronicState from '{state.source_file}' has no charges — "
                "run with pop=hirshfeld (Gaussian) or Print[P_Hirshfeld] (ORCA)."
            )
        return cls(
            structure=state.structure,
            energy_hartree=state.energy_hartree,
            homo_ev=state.homo_ev,
            lumo_ev=state.lumo_ev,
            charges=state.charges,
            charge_scheme=state.charge_scheme,
            source_file=state.source_file,
            spin_sq=state.spin_sq,
            electronic=state,
        )

    @classmethod
    def from_gaussian(
        cls,
        filepath: str | Path,
        prefer_hirshfeld: bool = True,
    ) -> CDFTInputState:
        """Parse a Gaussian output file and return a CDFTInputState.

        Geometry is read from the last Standard/Input Orientation block.
        Charges prefer Hirshfeld (pop=hirshfeld) and fall back to Mulliken.
        """
        structure = parse_gaussian_geometry(filepath)
        state = parse_gaussian_electronic(filepath, structure, prefer_hirshfeld=prefer_hirshfeld)
        return cls.from_electronic_state(state)

    @classmethod
    def from_orca(
        cls,
        filepath: str | Path,
        prefer_hirshfeld: bool = True,
    ) -> CDFTInputState:
        """Parse an ORCA output file and return a CDFTInputState."""
        structure = parse_orca_geometry(filepath)
        state = parse_orca_electronic(filepath, structure, prefer_hirshfeld=prefer_hirshfeld)
        return cls.from_electronic_state(state)

    @classmethod
    def from_wfn(
        cls,
        filepath: str | Path,
        charges: Optional[List[float]] = None,
        charge_scheme: str = "hirshfeld",
    ) -> CDFTInputState:
        """Parse a Gaussian .wfn file and return a CDFTInputState.

        WFN files provide total energy and orbital energies but not atomic
        charges.  Supply charges separately (Gaussian pop=hirshfeld or
        Multiwfn Hirshfeld partitioning).  When *charges* is omitted, the
        state can only contribute global FD descriptors; Fukui functions
        require charges for all three states (N, N+1, N-1).
        """
        from cadmia.io.wfn import parse_wfn
        import numpy as np
        from cadmia.core import Structure

        data = parse_wfn(filepath)
        if data.total_energy is None:
            raise ValueError(f"No TOTAL ENERGY found in WFN file: {filepath}")

        from cadmia.core.units import HARTREE_TO_EV as HARTREE_TO_EV_LOCAL
        coords = np.array(data.coords_ang)
        labels = [f"{sym}{i + 1}" for i, sym in enumerate(data.atoms)]
        structure = Structure(
            atoms=data.atoms,
            positions=coords,
            labels=labels,
            structure_id=Path(filepath).stem,
            source_file=str(Path(filepath).resolve()),
        )
        return cls(
            structure=structure,
            energy_hartree=data.total_energy,
            homo_ev=data.homo_hartree * HARTREE_TO_EV_LOCAL if data.homo_hartree is not None else None,
            lumo_ev=data.lumo_hartree * HARTREE_TO_EV_LOCAL if data.lumo_hartree is not None else None,
            charges=list(charges) if charges is not None else [],
            charge_scheme=charge_scheme,
            source_file=str(Path(filepath).resolve()),
        )


@dataclass
class CDFTWorkflowResult:
    """Complete output of the CDFT workflow for a system."""

    structure: Structure
    charge_scheme: str

    # --- global descriptors from finite differences ---
    global_fd: FiniteDifferenceCDFT

    # --- global Koopmans descriptors (from N HOMO/LUMO) ---
    global_koopmans: Optional[GlobalElectronic]

    # --- per-atom condensed descriptors ---
    fukui: FukuiResult

    # --- source files ---
    source_n: str
    source_np1: Optional[str]
    source_nm1: Optional[str]

    # --- the three input states, keyed "N", "N+1", "N-1" (absent ones omitted) ---
    states: Optional[dict] = None

    # --- convenience accessors ---
    @property
    def homo_n_ev(self) -> Optional[float]:
        return self.global_fd.homo_n_ev

    @property
    def homo_np1_ev(self) -> Optional[float]:
        return self.global_fd.homo_np1_ev

    @property
    def homo_nm1_ev(self) -> Optional[float]:
        return self.global_fd.homo_nm1_ev

    @property
    def nucleophilicity_ev(self) -> Optional[float]:
        return self.global_fd.nucleophilicity_ev

    @property
    def nucleophilicity_hartree(self) -> Optional[float]:
        return self.global_fd.nucleophilicity_hartree


# Spin contamination beyond this fraction of the ideal ⟨S²⟩ makes the state a poor
# representation of the intended spin eigenstate. 10 % is the usual working threshold.
_SPIN_CONTAMINATION_TOL = 0.10


def _spin_contamination_warnings(states: "dict[str, CDFTInputState | None]") -> List[str]:
    """Flag N±1 states whose ⟨S²⟩ is far from the nearest spin eigenvalue.

    The Fukui functions are differences of charges taken from the charged states, which are
    open shell. If one of them converged to a spin-contaminated solution, every condensed
    descriptor built on it is degraded — and nothing downstream can tell, because the
    numbers still normalize and still look reasonable.

    The expected ⟨S²⟩ is not assumed: the closest S(S+1) for half-integer S is used, so the
    check holds whether the ions are doublets, triplets or anything else.
    """
    out: List[str] = []
    for label, st in states.items():
        if st is None or st.spin_sq is None:
            continue
        # nearest ideal S(S+1) over S = 0, 0.5, 1, 1.5, ...
        candidates = [(abs(st.spin_sq - s * (s + 1)), s * (s + 1))
                      for s in (i / 2 for i in range(0, 13))]
        _, ideal = min(candidates)
        if ideal == 0:
            # A closed-shell SCF has no ⟨S²⟩ at all, so reaching here with ideal = 0 means an
            # UNRESTRICTED calculation that landed near a singlet — and a broken-symmetry singlet
            # is a real possibility in a Ti oxide. Dividing by zero is impossible, so the check
            # is absolute: any appreciable ⟨S²⟩ on a nominal singlet is worth saying out loud.
            if st.spin_sq > 0.1:
                out.append(
                    f"{label}: <S**2> = {st.spin_sq:.4f} on a nominally closed-shell state — "
                    "an unrestricted SCF that broke spin symmetry. The Fukui functions built "
                    "from it describe a state that is not the one intended.")
            continue
        excess = (st.spin_sq - ideal) / ideal
        if abs(excess) > _SPIN_CONTAMINATION_TOL:
            out.append(
                f"{label}: <S**2> = {st.spin_sq:.4f} against an ideal {ideal:.2f} "
                f"({excess:+.0%}). The Fukui functions are built from this state, so its "
                "condensed descriptors carry that contamination."
            )
    return out



def run_cdft_workflow(
    state_n: CDFTInputState,
    state_np1: Optional[CDFTInputState] = None,
    state_nm1: Optional[CDFTInputState] = None,
    homo_tce_ev: Optional[float] = None,
) -> CDFTWorkflowResult:
    """Run the full CDFT workflow from three electronic states.

    Parameters
    ----------
    state_n:
        Ground-state (N electrons) data.
    state_np1:
        Anion (N+1 electrons) data.  Required for f+, s+, electrophilicity.
    state_nm1:
        Cation (N-1 electrons) data.  Required for f-, s-, nucleophilicity.
    homo_tce_ev:
        HOMO energy of the TCE reference (eV).  Defaults to the Multiwfn
        B3LYP/6-31G* value (:data:`HOMO_TCE_EV`).
    """
    # --- validate charge scheme consistency ---
    # The condensed Fukui functions subtract charges POSITION BY POSITION, so the three states
    # must list their atoms in the same order. Same length is not enough: a file regenerated
    # from a reordered .xyz gives f = q_i(N) - q_j(N±1) with i ≠ j, still sums to ~1, and raises
    # nothing.
    for label, state in (("N+1", state_np1), ("N-1", state_nm1)):
        if state is None:
            continue
        if state.structure.atoms != state_n.structure.atoms:
            raise ValueError(
                f"{label}: the atom list differs from N. The three charge states must share the "
                "atom order — the condensed Fukui functions subtract charges by position.")

    # Swapped N+1/N-1 files exchange f+ and f- with no other symptom. Checked only for states
    # read from an output, where the total charge is the engine's own.
    q = state_n.structure.charge
    for label, state, expected in (("N+1", state_np1, q - 1), ("N-1", state_nm1, q + 1)):
        if (state is not None and state.electronic is not None and state_n.electronic is not None
                and state.structure.charge != expected):
            raise ValueError(
                f"{label}: total charge {state.structure.charge:+d}, expected {expected:+d} for "
                f"N with charge {q:+d}. --np1 is the anion (one more electron) and --nm1 the "
                "cation (one fewer).")

    schemes = {state_n.charge_scheme}
    if state_np1 is not None:
        schemes.add(state_np1.charge_scheme)
    if state_nm1 is not None:
        schemes.add(state_nm1.charge_scheme)
    if len(schemes) > 1:
        # Not a warning: a truncated Hirshfeld block makes the parser fall back to Mulliken
        # silently, and f = q_Mulliken - q_Hirshfeld is a plausible-looking, wrong number that
        # reaches the JSON either way.
        raise ValueError(
            f"Charge schemes differ across states: {sorted(schemes)}. Fukui functions are only "
            "meaningful when the same scheme is used for all three — check that every output "
            "carries a complete population block.")

    # --- the charged states are open shell: say so if either is contaminated ---
    # The neutral is checked too: it is usually closed shell, but nothing forces it to be, and
    # a broken-symmetry singlet there poisons every descriptor, not just the ones from the ions.
    for msg in _spin_contamination_warnings({"N": state_n, "N+1": state_np1, "N-1": state_nm1}):
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # --- collect charges per state (empty list → treat as absent) ---
    charges_n   = state_n.charges   if state_n.charges   else None
    charges_np1 = state_np1.charges if (state_np1 is not None and state_np1.charges) else None
    charges_nm1 = state_nm1.charges if (state_nm1 is not None and state_nm1.charges) else None

    # --- energies ---
    en = state_n.energy_hartree
    enp1 = state_np1.energy_hartree if state_np1 is not None else None
    enm1 = state_nm1.energy_hartree if state_nm1 is not None else None

    if enp1 is None or enm1 is None:
        raise ValueError(
            "run_cdft_workflow requires state_np1 and state_nm1 to have energy_hartree."
        )

    # --- global finite-difference descriptors ---
    global_fd = compute_finite_difference_descriptors(
        energy_n_hartree=en,
        energy_np1_hartree=enp1,
        energy_nm1_hartree=enm1,
        homo_n_ev=state_n.homo_ev,
        homo_np1_ev=state_np1.homo_ev if state_np1 else None,
        homo_nm1_ev=state_nm1.homo_ev if state_nm1 else None,
        lumo_n_ev=state_n.lumo_ev,
        homo_tce_ev=homo_tce_ev,
    )

    # --- Koopmans descriptors from N state HOMO/LUMO ---
    global_koopmans: Optional[GlobalElectronic] = None
    if state_n.homo_ev is not None and state_n.lumo_ev is not None:
        try:
            global_koopmans = compute_global_descriptors(
                homo_ev=state_n.homo_ev,
                lumo_ev=state_n.lumo_ev,
            )
        except ValueError:
            pass

    # --- condensed Fukui and local descriptors ---
    structure = state_n.structure
    atoms = structure.atoms
    labels = structure.labels

    if charges_n is not None:
        # Local softnesses follow Multiwfn convention: s = S[Ha^-1] × f[e]
        # global_fd.softness is in eV^-1; multiply by HARTREE_TO_EV to get Ha^-1
        from cadmia.core.units import HARTREE_TO_EV as _HtoEV
        fukui = compute_condensed_fukui(
            atoms=atoms,
            labels=labels,
            charges_N=charges_n,
            charges_Np1=charges_np1,
            charges_Nm1=charges_nm1,
            global_descriptors=global_koopmans,
            nucleophilicity_ev=global_fd.nucleophilicity_ev,
            hardness_ev=global_fd.hardness_ev,
            ip_ev=global_fd.ip_ev,
            ea_ev=global_fd.ea_ev,
            softness_override=global_fd.softness * _HtoEV,  # eV^-1 → Ha^-1
            omega_ev_override=global_fd.omega_ev,           # FD omega (eV)
        )
    else:
        fukui = FukuiResult(
            atoms=list(atoms),
            labels=list(labels),
            f_plus=None,
            f_minus=None,
            f_zero=None,
            dual=None,
            local_softness_plus=None,
            local_softness_minus=None,
            local_electrophilicity=None,
        )

    return CDFTWorkflowResult(
        structure=structure,
        charge_scheme=state_n.charge_scheme,
        global_fd=global_fd,
        global_koopmans=global_koopmans,
        fukui=fukui,
        source_n=state_n.source_file,
        source_np1=state_np1.source_file if state_np1 else None,
        source_nm1=state_nm1.source_file if state_nm1 else None,
        states={label: state for label, state in
                (("N", state_n), ("N+1", state_np1), ("N-1", state_nm1)) if state is not None},
    )


def run_workflow_from_files(
    path_n: str | Path,
    path_np1: str | Path,
    path_nm1: str | Path,
    *,
    prefer_hirshfeld: bool = True,
    homo_tce_ev: Optional[float] = None,
):
    """Load three SCF outputs (Gaussian or ORCA, auto-detected) and run the CDFT workflow.

    Returns ``(result, charges_n, charges_np1, charges_nm1)``. Used by ``cadmia cdft --report``.
    Engine-agnostic and
    element-agnostic: it works for any cluster the parsers can read.

    Raises ``ValueError`` if a file cannot be parsed as Gaussian or ORCA.
    """
    def _load(path: str | Path, label: str) -> CDFTInputState:
        p = Path(path)
        try:
            return CDFTInputState.from_gaussian(p, prefer_hirshfeld=prefer_hirshfeld)
        except (ValueError, FileNotFoundError):
            pass
        try:
            return CDFTInputState.from_orca(p, prefer_hirshfeld=prefer_hirshfeld)
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(f"could not parse {label} file '{path}': {exc}") from exc

    state_n   = _load(path_n,   "N")
    state_np1 = _load(path_np1, "N+1")
    state_nm1 = _load(path_nm1, "N-1")
    result = run_cdft_workflow(
        state_n=state_n, state_np1=state_np1, state_nm1=state_nm1, homo_tce_ev=homo_tce_ev,
    )
    return result, state_n.charges, state_np1.charges, state_nm1.charges
