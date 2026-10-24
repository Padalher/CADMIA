"""Tests for cdft."""
import pytest

from cadmia.cdft.calculator import (
    HOMO_TCE_EV,
    compute_condensed_fukui,
    compute_finite_difference_descriptors,
)
from cadmia.cdft.multiwfn import parse_multiwfn_cdft
from cadmia.cdft.workflow import CDFTWorkflowResult, run_cdft_workflow, CDFTInputState
from cadmia.core.electronic_descriptors import compute_global_descriptors
from cadmia.core import Structure
import numpy as np

Q_N = [+0.3, -0.3]
Q_Np1 = [-0.1, -0.9]
Q_Nm1 = [+0.7, +0.3]
ATOMS = ["Ti", "O"]
LABELS = ["Ti1", "O2"]


def test_fukui_plus_values():
    r = compute_condensed_fukui(ATOMS, LABELS, Q_N, charges_Np1=Q_Np1)
    assert r.f_plus is not None
    assert r.f_plus[0] == pytest.approx(0.3 - (-0.1), abs=1e-9)
    assert r.f_plus[1] == pytest.approx(-0.3 - (-0.9), abs=1e-9)


def test_fukui_minus_values():
    r = compute_condensed_fukui(ATOMS, LABELS, Q_N, charges_Nm1=Q_Nm1)
    assert r.f_minus is not None
    assert r.f_minus[0] == pytest.approx(0.7 - 0.3, abs=1e-9)
    assert r.f_minus[1] == pytest.approx(0.3 - (-0.3), abs=1e-9)


def test_dual_descriptor():
    r = compute_condensed_fukui(
        ATOMS,
        LABELS,
        Q_N,
        charges_Np1=Q_Np1,
        charges_Nm1=Q_Nm1,
    )
    assert r.dual is not None
    for i in range(len(ATOMS)):
        assert r.dual[i] == pytest.approx(r.f_plus[i] - r.f_minus[i], abs=1e-9)


def test_f_zero():
    r = compute_condensed_fukui(
        ATOMS,
        LABELS,
        Q_N,
        charges_Np1=Q_Np1,
        charges_Nm1=Q_Nm1,
    )
    assert r.f_zero is not None
    for i in range(len(ATOMS)):
        expected = (r.f_plus[i] + r.f_minus[i]) / 2
        assert r.f_zero[i] == pytest.approx(expected, abs=1e-9)


def test_none_when_missing_charges():
    r = compute_condensed_fukui(ATOMS, LABELS, Q_N)
    assert r.f_plus is None
    assert r.f_minus is None
    assert r.dual is None


def test_local_softness_with_global():
    gd = compute_global_descriptors(homo_ev=-6.0, lumo_ev=-4.0)
    r = compute_condensed_fukui(
        ATOMS,
        LABELS,
        Q_N,
        charges_Np1=Q_Np1,
        global_descriptors=gd,
    )
    assert r.local_softness_plus is not None
    for i in range(len(ATOMS)):
        expected = gd.softness * r.f_plus[i]
        assert r.local_softness_plus[i] == pytest.approx(expected, abs=1e-9)


def test_finite_difference_descriptors_from_charged_state_energies():
    d = compute_finite_difference_descriptors(
        energy_n_hartree=-100.0,
        energy_np1_hartree=-100.1,
        energy_nm1_hartree=-99.8,
    )

    assert d.ip_ev == pytest.approx(0.2 * 27.2114)
    assert d.ea_ev == pytest.approx(0.1 * 27.2114)
    assert d.hardness_ev == pytest.approx(d.ip_ev - d.ea_ev)
    assert d.softness == pytest.approx(1.0 / d.hardness_ev)


def test_parse_multiwfn_cdft_report(tmp_path):
    report = tmp_path / "multiwfn_cdft.txt"
    report.write_text(
        """
 Hirshfeld charges, condensed Fukui functions and condensed dual descriptors
     Atom     q(N)    q(N+1)   q(N-1)     f-       f+       f0      CDD
     1(Ti)   0.7399   0.6351   0.9157   0.1758   0.1048   0.1403  -0.0710

 Condensed local electrophilicity/nucleophilicity index (e*eV)
     Atom              Electrophilicity          Nucleophilicity
     1(Ti)                  0.26315                  0.14547

 Condensed local softnesses (Hartree*e) and relative electrophilicity/nucleophilicity (dimensionless)
     Atom         s-          s+          s0        s+/s-       s-/s+
     1(Ti)      0.5981      0.3564      0.4773      0.5960      1.6779

 E(N):    -1710.163302 Hartree
 E(N+1):  -1710.249267 Hartree
 E(N-1):  -1709.783385 Hartree
 E_HOMO(N):     -0.304791 Hartree,   -8.2938 eV
 Vertical IP:    0.379917 Hartree,   10.3381 eV
 Vertical EA:    0.085965 Hartree,    2.3392 eV
 Softness:    3.401919 Hartree^-1,    0.1250 eV^-1
""",
        encoding="utf-8",
    )

    parsed = parse_multiwfn_cdft(report)

    assert parsed.global_descriptors["homo_n"]["ev"] == pytest.approx(-8.2938)
    assert parsed.global_descriptors["vertical_ip"]["ev"] == pytest.approx(10.3381)
    assert parsed.atoms[0].element == "Ti"
    assert parsed.atoms[0].f_plus == pytest.approx(0.1048)
    assert parsed.atoms[0].electrophilicity == pytest.approx(0.26315)


# ---------------------------------------------------------------------------
# Extended FukuiResult descriptors
# ---------------------------------------------------------------------------

def test_softness_zero_computed():
    gd = compute_global_descriptors(homo_ev=-6.0, lumo_ev=-4.0)
    r = compute_condensed_fukui(
        ATOMS, LABELS, Q_N,
        charges_Np1=Q_Np1,
        charges_Nm1=Q_Nm1,
        global_descriptors=gd,
    )
    assert r.softness_zero is not None
    for i in range(len(ATOMS)):
        assert r.softness_zero[i] == pytest.approx(gd.softness * r.f_zero[i], abs=1e-9)


def test_relative_electrophilicity_and_nucleophilicity():
    gd = compute_global_descriptors(homo_ev=-6.0, lumo_ev=-4.0)
    r = compute_condensed_fukui(
        ATOMS, LABELS, Q_N,
        charges_Np1=Q_Np1,
        charges_Nm1=Q_Nm1,
        global_descriptors=gd,
    )
    assert r.relative_electrophilicity is not None
    assert r.relative_nucleophilicity is not None
    for i in range(len(ATOMS)):
        if r.local_softness_minus[i] != 0.0:
            assert r.relative_electrophilicity[i] == pytest.approx(
                r.local_softness_plus[i] / r.local_softness_minus[i], abs=1e-9
            )


def test_local_nucleophilicity():
    r = compute_condensed_fukui(
        ATOMS, LABELS, Q_N,
        charges_Nm1=Q_Nm1,
        nucleophilicity_ev=1.5,
    )
    assert r.local_nucleophilicity is not None
    for i in range(len(ATOMS)):
        assert r.local_nucleophilicity[i] == pytest.approx(1.5 * r.f_minus[i], abs=1e-9)


def test_f_zero_scaled_by_hardness_is_exactly_that():
    """η is a global constant, so η·f0 ranks atoms identically to f0.
    The name used to be `local_hardness`, which invited reading it as a property."""
    r = compute_condensed_fukui(
        ATOMS, LABELS, Q_N,
        charges_Np1=Q_Np1,
        charges_Nm1=Q_Nm1,
        hardness_ev=3.0,
    )
    assert r.f_zero_scaled_by_hardness is not None
    for i in range(len(ATOMS)):
        assert r.f_zero_scaled_by_hardness[i] == pytest.approx(3.0 * r.f_zero[i], abs=1e-9)


def test_donicity_and_accepting_capacity():
    r = compute_condensed_fukui(
        ATOMS, LABELS, Q_N,
        charges_Np1=Q_Np1,
        charges_Nm1=Q_Nm1,
        ip_ev=9.0,
        ea_ev=2.5,
    )
    assert r.donicity is not None
    assert r.accepting_capacity is not None
    for i in range(len(ATOMS)):
        assert r.donicity[i] == pytest.approx(9.0 * r.f_minus[i], abs=1e-9)
        assert r.accepting_capacity[i] == pytest.approx(2.5 * r.f_plus[i], abs=1e-9)


# ---------------------------------------------------------------------------
# FiniteDifferenceCDFT with HOMO and nucleophilicity
# ---------------------------------------------------------------------------

def test_fd_nucleophilicity_index():
    homo_n = -7.8851
    d = compute_finite_difference_descriptors(
        energy_n_hartree=-100.0,
        energy_np1_hartree=-100.1,
        energy_nm1_hartree=-99.8,
        homo_n_ev=homo_n,
    )
    assert d.homo_n_ev == pytest.approx(homo_n)
    expected_nucl = homo_n - HOMO_TCE_EV
    assert d.nucleophilicity_ev == pytest.approx(expected_nucl, abs=1e-6)
    assert d.nucleophilicity_hartree == pytest.approx(expected_nucl / 27.2114, abs=1e-8)


def test_fd_custom_tce_reference():
    homo_n = -7.0
    custom_tce = -9.0
    d = compute_finite_difference_descriptors(
        energy_n_hartree=-100.0,
        energy_np1_hartree=-100.1,
        energy_nm1_hartree=-99.8,
        homo_n_ev=homo_n,
        homo_tce_ev=custom_tce,
    )
    assert d.nucleophilicity_ev == pytest.approx(homo_n - custom_tce, abs=1e-9)
    assert d.homo_tce_ev == pytest.approx(custom_tce)


# ---------------------------------------------------------------------------
# CDFTWorkflow end-to-end (in-memory states, no real output files)
# ---------------------------------------------------------------------------

def _make_state(energy, homo, lumo, charges, atoms, positions, labels):
    structure = Structure(
        atoms=atoms,
        positions=np.array(positions),
        labels=labels,
        structure_id="test",
        source_file="",
    )
    return CDFTInputState(
        structure=structure,
        energy_hartree=energy,
        homo_ev=homo,
        lumo_ev=lumo,
        charges=charges,
        charge_scheme="hirshfeld",
    )


_POSITIONS = [[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]]

def test_run_cdft_workflow_complete():
    # Charges are the module fixtures: they conserve charge across the triplet (0 / −1 / +1), so
    # the condensed Fukui functions normalize to 1 as the physics requires.
    sn   = _make_state(-100.0, -7.89, -4.5, Q_N, ATOMS, _POSITIONS, LABELS)
    snp1 = _make_state(-100.1, -0.94, None, Q_Np1, ATOMS, _POSITIONS, LABELS)
    snm1 = _make_state(-99.7,  -11.4, None, Q_Nm1, ATOMS, _POSITIONS, LABELS)

    result = run_cdft_workflow(sn, snp1, snm1)

    # Global FD
    assert result.global_fd.ip_ev == pytest.approx(0.3 * 27.2114, abs=1e-3)
    assert result.global_fd.ea_ev == pytest.approx(0.1 * 27.2114, abs=1e-3)
    assert result.nucleophilicity_ev == pytest.approx(-7.89 - HOMO_TCE_EV, abs=1e-4)

    # Koopmans
    assert result.global_koopmans is not None
    assert result.global_koopmans.homo_ev == pytest.approx(-7.89)

    # Local Fukui
    f = result.fukui
    assert f.f_plus is not None
    assert f.f_minus is not None
    assert f.softness_zero is not None
    assert f.local_nucleophilicity is not None
    assert f.relative_electrophilicity is not None
    assert f.f_zero_scaled_by_hardness is not None
    assert f.donicity is not None
    assert f.accepting_capacity is not None


# --- the three charge states must be comparable atom for atom ---

def _audit_state(atoms, charges, scheme="hirshfeld", energy=-100.0):
    from cadmia.cdft.workflow import CDFTInputState

    return CDFTInputState(
        structure=Structure(atoms=list(atoms),
                            positions=np.array([(0.0, 0.0, float(i)) for i in range(len(atoms))]),
                            labels=[f"{a}{i + 1}" for i, a in enumerate(atoms)],
                            structure_id="toy", source_file=""),
        energy_hartree=energy, homo_ev=-8.0, lumo_ev=-3.0,
        charges=list(charges), charge_scheme=scheme)


def test_states_with_a_different_atom_order_are_refused():
    """The condensed Fukui functions subtract charges BY POSITION; same length is not enough."""
    import pytest

    from cadmia.cdft.workflow import run_cdft_workflow

    n = _audit_state(["Ti", "O"], [0.5, -0.5], energy=-100.0)
    np1 = _audit_state(["O", "Ti"], [-0.6, 0.4], energy=-100.1)      # swapped
    nm1 = _audit_state(["Ti", "O"], [0.7, -0.3], energy=-99.6)

    with pytest.raises(ValueError, match="atom order"):
        run_cdft_workflow(n, np1, nm1)


def test_mixing_charge_schemes_is_an_error_not_a_warning():
    """A truncated Hirshfeld block makes the parser fall back to Mulliken silently."""
    import pytest

    from cadmia.cdft.workflow import run_cdft_workflow

    n = _audit_state(["Ti", "O"], [0.5, -0.5], scheme="hirshfeld", energy=-100.0)
    np1 = _audit_state(["Ti", "O"], [0.4, -0.6], scheme="mulliken", energy=-100.1)
    nm1 = _audit_state(["Ti", "O"], [0.7, -0.3], scheme="hirshfeld", energy=-99.6)

    with pytest.raises(ValueError, match="Charge schemes differ"):
        run_cdft_workflow(n, np1, nm1)


# ─────────────────────────────────────────────
# Regression tests: CDFT edge cases
# ─────────────────────────────────────────────

def test_the_ambiphilic_threshold_scales_with_the_system():
    """f sums to 1 over N atoms, so an absolute cut means a different thing at each size."""
    from cadmia.cdft.acid_base import scaled_threshold

    assert scaled_threshold(8) == pytest.approx(0.02)     # the reference size
    assert scaled_threshold(32) < scaled_threshold(8)     # a 32-atom cluster needs a smaller cut
    assert scaled_threshold(32) == pytest.approx(0.005)
    assert scaled_threshold(None) == pytest.approx(0.02)  # unknown size → the base value


def test_a_negative_fukui_cannot_make_an_acid_or_a_base():
    """f+ = −0.03 with f− = −0.08 gives Δf = +0.05 and was labelled a Lewis acid."""
    from cadmia.cdft.acid_base import LEWIS_ACID, UNKNOWN, classify_acid_base

    assert classify_acid_base(0.05, fukui_plus=-0.03, fukui_minus=-0.08) == UNKNOWN
    assert classify_acid_base(0.05, fukui_plus=0.30, fukui_minus=0.25) == LEWIS_ACID


def test_exchanging_a_lot_in_both_directions_is_not_the_same_as_exchanging_nothing():
    """An oxygen with f+ ≈ f− ≈ 0.10 is a strong donor; one with f+ ≈ f− ≈ 0 is inert.
    Both used to be called amphoteric."""
    from cadmia.cdft.acid_base import AMPHOTERIC, DUAL_ROLE, classify_acid_base

    assert classify_acid_base(0.001, fukui_plus=0.10, fukui_minus=0.099) == DUAL_ROLE
    assert classify_acid_base(0.001, fukui_plus=0.002, fukui_minus=0.001) == AMPHOTERIC


def test_an_unbound_anion_is_reported():
    """η > 0 holds trivially with EA < 0, so it checks nothing."""
    from cadmia.cdft.calculator import compute_finite_difference_descriptors

    with pytest.warns(RuntimeWarning, match="NOT bound"):
        r = compute_finite_difference_descriptors(
            energy_n_hartree=-100.0, energy_np1_hartree=-99.9, energy_nm1_hartree=-99.6)
    assert r.ea_ev < 0 and r.hardness_ev > 0      # the old guard would have let this through


def test_a_weakly_bound_anion_is_flagged_as_basis_sensitive():
    from cadmia.cdft.calculator import compute_finite_difference_descriptors

    with pytest.warns(RuntimeWarning, match="weakly bound"):
        compute_finite_difference_descriptors(
            energy_n_hartree=-100.0, energy_np1_hartree=-100.01, energy_nm1_hartree=-99.6)


def test_a_broken_symmetry_singlet_is_no_longer_skipped():
    """`if ideal == 0: continue` discarded exactly the contaminated-singlet range."""
    from cadmia.cdft.workflow import _spin_contamination_warnings

    class _S:
        spin_sq = 0.30      # nearest ideal is 0 (singlet), not 0.75

    msgs = _spin_contamination_warnings({"N": _S()})
    assert any("closed-shell" in m for m in msgs)

    class _Clean:
        spin_sq = 0.02      # a genuine closed shell stays quiet

    assert _spin_contamination_warnings({"N": _Clean()}) == []
