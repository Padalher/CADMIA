"""Tests for cadmia.core.electronic_descriptors."""
import pytest

from cadmia.core.electronic_descriptors import compute_global_descriptors


def test_global_descriptors_known_values():
    d = compute_global_descriptors(homo_ev=-6.0, lumo_ev=-4.0)

    assert d.gap_ev == pytest.approx(2.0, abs=1e-9)
    assert d.ip_ev == pytest.approx(6.0, abs=1e-9)
    assert d.ea_ev == pytest.approx(4.0, abs=1e-9)
    assert d.mu_ev == pytest.approx(-5.0, abs=1e-9)
    assert d.chi_ev == pytest.approx(5.0, abs=1e-9)
    assert d.eta_ev == pytest.approx(2.0, abs=1e-9)      # η = LUMO − HOMO = gap (no ½)
    assert d.softness == pytest.approx(0.5, abs=1e-9)    # S = 1/η
    assert d.omega_ev == pytest.approx(6.25, abs=1e-9)   # ω = μ²/(2η)


def test_global_descriptors_positive_gap():
    d = compute_global_descriptors(homo_ev=-8.0, lumo_ev=-3.0)
    assert d.gap_ev > 0


def test_global_descriptors_positive_eta():
    d = compute_global_descriptors(homo_ev=-7.0, lumo_ev=-2.0)
    assert d.eta_ev > 0


def test_global_descriptors_positive_omega():
    d = compute_global_descriptors(homo_ev=-6.0, lumo_ev=-4.0)
    assert d.omega_ev > 0


# ─────────────────────────────────────────────
# parse_all_charges — all ORCA schemes
# ─────────────────────────────────────────────

_ORCA_CHARGES = """\
MULLIKEN ATOMIC CHARGES
-----------------------
   0 Ti:    1.320575
   1 O :   -0.563734
   2 O :   -0.756841
Sum of atomic charges:    0.0000000

LOEWDIN ATOMIC CHARGES
----------------------
   0 Ti:    0.515936
   1 O :   -0.213270
   2 O :   -0.302666

CHELPG Charges            
--------------------------------
  0   Ti  :       1.510721
  1   O   :      -0.640365
  2   O   :      -0.870356
--------------------------------
Total charge:     0.000000
"""

_ORCA_CHARGES_OPEN = """\
MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS
--------------------------------------------
   0 Al:    0.683596    0.512345
   1 O :   -0.341798    0.743827
   2 O :   -0.341798    0.743828
Sum of atomic charges         :   -0.0000000
Sum of atomic spin populations:    2.0000000

LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS
-------------------------------------------
   0 Al:   -0.065066    0.412345
   1 O :    0.032533    0.793827
   2 O :    0.032533    0.793828
"""


def test_parse_all_charges_orca_closed_shell(tmp_path):
    from cadmia.io.electronic import parse_all_charges
    p = tmp_path / "x.out"
    p.write_text(_ORCA_CHARGES)
    charges = parse_all_charges(p, 3)
    assert set(charges) == {"mulliken", "loewdin", "chelpg"}
    assert charges["mulliken"][0] == pytest.approx(1.320575)
    assert charges["loewdin"][1] == pytest.approx(-0.213270)
    assert charges["chelpg"][2] == pytest.approx(-0.870356)
    for vals in charges.values():
        assert sum(vals) == pytest.approx(0.0, abs=1e-5)


def test_parse_all_charges_orca_open_shell_takes_charge_column(tmp_path):
    from cadmia.io.electronic import parse_all_charges
    p = tmp_path / "x.out"
    p.write_text(_ORCA_CHARGES_OPEN)
    charges = parse_all_charges(p, 3)
    assert charges["mulliken"] == pytest.approx([0.683596, -0.341798, -0.341798])
    assert charges["loewdin"] == pytest.approx([-0.065066, 0.032533, 0.032533])
    for vals in charges.values():
        assert sum(vals) == pytest.approx(0.0, abs=1e-5)


def test_parse_all_charges_wrong_atom_count_excluded(tmp_path):
    from cadmia.io.electronic import parse_all_charges
    p = tmp_path / "x.out"
    p.write_text(_ORCA_CHARGES)
    assert parse_all_charges(p, 5) == {}


# ─────────────────────────────────────────────
# ORCA HOMO/LUMO — open-shell cross-spin rule (regression)
# ─────────────────────────────────────────────

def test_frontier_from_eigenvalues_cross_spin():
    from cadmia.io.electronic import _frontier_from_eigenvalues
    # α: HOMO -10, LUMO -3 ; β: HOMO -9, LUMO -5  → HOMO=max(-10,-9)=-9, LUMO=min(-3,-5)=-5
    alpha = [-12.0, -10.0, -3.0, 1.0]
    beta = [-11.0, -9.0, -5.0, 2.0]
    homo, lumo = _frontier_from_eigenvalues(alpha, beta, na=2, nb=2)
    assert homo == pytest.approx(-9.0)
    assert lumo == pytest.approx(-5.0)

def test_frontier_from_eigenvalues_closed_shell():
    from cadmia.io.electronic import _frontier_from_eigenvalues
    homo, lumo = _frontier_from_eigenvalues([-8.0, -6.0, -2.0], None, na=2, nb=0)
    assert homo == pytest.approx(-6.0)
    assert lumo == pytest.approx(-2.0)

def test_frontier_handles_missing_virtuals():
    from cadmia.io.electronic import _frontier_from_eigenvalues
    homo, lumo = _frontier_from_eigenvalues([-8.0, -6.0], None, na=2, nb=0)
    assert homo == pytest.approx(-6.0)
    assert lumo is None
    assert _frontier_from_eigenvalues(None, None, 0, 0) == (None, None)


_ORCA_OPEN_SHELL = """\
                                  ORBITAL ENERGIES
                                  ----------------
                                SPIN UP ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000      -1.00000        -27.2114
   1   1.0000      -0.50000        -13.6057
   2   0.0000      -0.10000         -2.7211
   3   0.0000       0.10000          2.7211

                                SPIN DOWN ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000      -0.95000        -25.8508
   1   1.0000      -0.48000        -13.0615
   2   0.0000      -0.30000         -8.1634
   3   0.0000       0.20000          5.4423
"""


def test_parse_orca_electronic_open_shell_uses_beta(tmp_path):
    """Regression: the β LUMO (-8.16 eV) is lower than the α LUMO (-2.72 eV);
    the alpha-only inline scan used to miss it entirely."""
    import numpy as np
    from cadmia.core import Structure
    from cadmia.io.electronic import parse_orca_electronic
    p = tmp_path / "open.out"
    p.write_text(_ORCA_OPEN_SHELL)
    s = Structure(atoms=["H"], positions=np.zeros((1, 3)),
                  labels=["H1"], structure_id="t", source_file=str(p))
    es = parse_orca_electronic(p, s)
    assert es.homo_ev == pytest.approx(-13.0615, abs=1e-3)   # β HOMO > α HOMO
    assert es.lumo_ev == pytest.approx(-8.1634, abs=1e-3)    # β LUMO < α LUMO
    assert es.homo_beta_ev == pytest.approx(-13.0615, abs=1e-3)
    assert es.lumo_beta_ev == pytest.approx(-8.1634, abs=1e-3)
    assert es.gap_beta_ev == pytest.approx(4.8981, abs=1e-3)


def test_gaussian_open_shell_keeps_both_spin_gaps_and_spin_sq():
    """The combined gap (max HOMO, min LUMO) crosses spin channels on an open shell; the
    per-channel gaps and <S**2> are read from the real Gaussian anion of the 1tial fixture."""
    from pathlib import Path
    from cadmia.io.electronic import parse_gaussian_electronic
    from cadmia.io.gaussian import parse_gaussian_geometry
    path = Path(__file__).parent.parent / "fixtures" / "1tial_cdft" / "N+1.out"
    es = parse_gaussian_electronic(path, parse_gaussian_geometry(path))
    assert es.spin_sq == pytest.approx(0.7591)     # before annihilation, not the 0.7500 after
    assert es.gap_alpha_ev == pytest.approx(1.7628, abs=1e-3)
    assert es.gap_beta_ev == pytest.approx(5.6499, abs=1e-3)
    assert es.homo_lumo_gap_ev == pytest.approx(es.gap_alpha_ev)   # here both frontiers are α


# ─────────────────────────────────────────────
# Gaussian Opt with pop=Hirshfeld prints one block per step (regression)
# ─────────────────────────────────────────────

# Trimmed from a real Gaussian 16 run (PBE1PBE/Def2SVP Opt pop=Hirshfeld on a distorted
# water): the first block belongs to the starting geometry (E = -76.2285) and the last to
# the converged one (E = -76.2763). The O charge moves by 0.012 e between them.
_GAUSSIAN_HIRSHFELD_TWO_STEPS = """\
 SCF Done:  E(RPBE1PBE) =  -76.2285231881     A.U. after   10 cycles
 Hirshfeld charges, spin densities, dipoles, and CM5 charges using IRadAn=      5:
              Q-H        S-H        Dx         Dy         Dz        Q-CM5
     1  O   -0.307213   0.000000  -0.000007  -0.106296  -0.000000  -0.516083
     2  H    0.153608   0.000000   0.130593  -0.160360  -0.000000   0.258040
     3  H    0.153605   0.000000  -0.130600  -0.160356  -0.000000   0.258044
 Hirshfeld charges with hydrogens summed into heavy atoms:
              Q-H       Q-CM5
     1  O    0.000000   0.000000
 SCF Done:  E(RPBE1PBE) =  -76.2763272714     A.U. after    8 cycles
 Hirshfeld charges, spin densities, dipoles, and CM5 charges using IRadAn=      5:
              Q-H        S-H        Dx         Dy         Dz        Q-CM5
     1  O   -0.319426   0.000000   0.000001  -0.162831  -0.000000  -0.651628
     2  H    0.159713   0.000000   0.168105  -0.137545   0.000000   0.325814
     3  H    0.159713   0.000000  -0.168104  -0.137546  -0.000000   0.325813
 Hirshfeld charges with hydrogens summed into heavy atoms:
              Q-H       Q-CM5
     1  O   -0.000000  -0.000000
"""


def test_gaussian_hirshfeld_keeps_the_converged_step():
    """Regression, confirmed against a real run: `Opt pop=Hirshfeld` writes one block per
    optimization step, and the FIRST is the starting geometry. The parser used to return it,
    so the charges described a geometry the reported energy did not."""
    from cadmia.io.electronic import _parse_gaussian_hirshfeld_charges
    charges = _parse_gaussian_hirshfeld_charges(_GAUSSIAN_HIRSHFELD_TWO_STEPS.splitlines(), 3)
    assert charges == pytest.approx([-0.319426, 0.159713, 0.159713])   # converged step
    assert sum(charges) == pytest.approx(0.0, abs=1e-6)                # neutral molecule


def test_gaussian_hirshfeld_ignores_the_heavy_atom_summary_block():
    """The `summed into heavy atoms` block has one row per heavy atom, not per atom: it must
    not be mistaken for the charges of a 3-atom molecule."""
    from cadmia.io.electronic import _parse_gaussian_hirshfeld_charges
    assert _parse_gaussian_hirshfeld_charges(_GAUSSIAN_HIRSHFELD_TWO_STEPS.splitlines(), 3)[0] \
        == pytest.approx(-0.319426)


# ─────────────────────────────────────────────
# ORCA Opt/Freq prints two blocks — the converged one wins (regression)
# ─────────────────────────────────────────────

# An Opt/Freq output as ORCA writes it: one set of blocks on the starting geometry of
# cycle 1, another once the optimization has converged. Values differ in the last decimals
# exactly as in a real run, so a parser reading the wrong block still looks plausible.
_ORCA_OPT_FREQ_TWO_BLOCKS = """\
                    *      GEOMETRY OPTIMIZATION CYCLE   1      *

ORBITAL ENERGIES
----------------

  NO   OCC          E(Eh)            E(eV)
   0   2.0000      -1.000000        -27.2114
   1   2.0000      -0.304762         -8.2928
   2   0.0000      -0.136906         -3.7253

------------------
HIRSHFELD ANALYSIS
------------------

Total integrated alpha density =     43.999995065

  ATOM     CHARGE      SPIN
   0 Ti   0.751341    0.000000
   1 O   -0.430910    0.000000

                    ***        THE OPTIMIZATION HAS CONVERGED     ***

ORBITAL ENERGIES
----------------

  NO   OCC          E(Eh)            E(eV)
   0   2.0000      -1.000100        -27.2141
   1   2.0000      -0.304571         -8.2876
   2   0.0000      -0.136910         -3.7254

------------------
HIRSHFELD ANALYSIS
------------------

Total integrated alpha density =     43.999995065

  ATOM     CHARGE      SPIN
   0 Ti   0.750954    0.000000
   1 O   -0.431219    0.000000

FINAL SINGLE POINT ENERGY     -1710.170750050696
"""


def test_orca_orbital_energies_keep_converged_block():
    """Regression: the first ORBITAL ENERGIES block of an Opt/Freq belongs to cycle 1 —
    the STARTING geometry. The parser used to return it, so the frontier orbitals
    described a geometry the reported energy did not."""
    from cadmia.io.electronic import _parse_orca_orbital_energies
    alpha, beta, na, nb = _parse_orca_orbital_energies(_ORCA_OPT_FREQ_TWO_BLOCKS.splitlines())
    assert alpha[na - 1] == pytest.approx(-8.2876, abs=1e-4)   # converged, not -8.2928
    assert alpha[na] == pytest.approx(-3.7254, abs=1e-4)       # converged, not -3.7253
    assert len(alpha) == 3                                     # not 6 (no concatenation)


def test_orca_hirshfeld_keeps_converged_block():
    """Regression: same as above for the Hirshfeld charges, which fed the site charges."""
    from cadmia.io.electronic import _parse_orca_hirshfeld_charges
    charges = _parse_orca_hirshfeld_charges(_ORCA_OPT_FREQ_TWO_BLOCKS.splitlines())
    assert charges == pytest.approx([0.750954, -0.431219])     # not [0.751341, -0.430910]


def test_orca_electronic_energy_and_orbitals_share_the_final_geometry(tmp_path):
    """The energy already came from the final SCF; the orbitals must come from the same
    geometry, or a single ElectronicState mixes two different structures."""
    import numpy as np
    from cadmia.core import Structure
    from cadmia.io.electronic import parse_orca_electronic
    p = tmp_path / "opt_freq.out"
    p.write_text(_ORCA_OPT_FREQ_TWO_BLOCKS)
    s = Structure(atoms=["Ti", "O"], positions=np.zeros((2, 3)),
                  labels=["Ti1", "O2"], structure_id="t", source_file=str(p))
    es = parse_orca_electronic(p, s)
    assert es.energy_hartree == pytest.approx(-1710.170750050696)
    assert es.homo_ev == pytest.approx(-8.2876, abs=1e-4)
    assert es.lumo_ev == pytest.approx(-3.7254, abs=1e-4)


def test_orca_single_point_still_reads_its_only_block():
    """A single-block SP (the CDFT triplet) must be unaffected by the last-block rule."""
    from cadmia.io.electronic import _parse_orca_orbital_energies
    single = _ORCA_OPT_FREQ_TWO_BLOCKS.split("THE OPTIMIZATION HAS CONVERGED")[0]
    alpha, _beta, na, _nb = _parse_orca_orbital_energies(single.splitlines())
    assert alpha[na - 1] == pytest.approx(-8.2928, abs=1e-4)


# ─────────────────────────────────────────────
# Gaussian eigenvalues — multiple dumps must not concatenate (regression)
# ─────────────────────────────────────────────

_GAUSSIAN_TWO_DUMPS = """\
 Alpha  occ. eigenvalues --  -0.80000  -0.50000
 Alpha virt. eigenvalues --   0.20000   0.60000
 SCF Done:  E(RPBE1PBE) =  -100.0  A.U.
 Alpha  occ. eigenvalues --  -0.85000  -0.55000
 Alpha virt. eigenvalues --   0.10000   0.50000
 Mulliken charges:
     1  Ti   0.10
 Sum of Mulliken charges
"""


def test_gaussian_eigenvalues_keep_last_dump(tmp_path):
    """Regression: a second eigenvalue dump (opt+freq, pop=full) must REPLACE the
    first, not extend onto it. The old parser kept LUMO from the first dump (0.20)
    and a doubled eigenvalue list."""
    import numpy as np
    from cadmia.core import Structure
    from cadmia.core.units import HARTREE_TO_EV
    from cadmia.io.electronic import parse_gaussian_electronic
    p = tmp_path / "two.log"
    p.write_text(_GAUSSIAN_TWO_DUMPS)
    s = Structure(atoms=["Ti"], positions=np.zeros((1, 3)),
                  labels=["Ti1"], structure_id="t", source_file=str(p))
    es = parse_gaussian_electronic(p, s)
    assert es.homo_ev / HARTREE_TO_EV == pytest.approx(-0.55, abs=1e-6)   # last dump
    assert es.lumo_ev / HARTREE_TO_EV == pytest.approx(0.10, abs=1e-6)    # last dump, not 0.20
    assert len(es.alpha_eigenvalues_ev) == 4                              # not 8 (no concat)


# ─────────────────────────────────────────────
# ORCA thermochemistry (Freq run) — G/H/S parsing
# ─────────────────────────────────────────────

_ORCA_THERMO = """\
                              THERMOCHEMISTRY AT 298.15K
-------------------------------------------------------
Temperature         ...   298.15 K
Pressure            ...     1.00 atm
Zero point energy                ...      0.03706219 Eh      23.26 kcal/mol
Total Enthalpy                    ...  -1898.64464673 Eh
Final entropy term                ...      0.04608514 Eh     28.92 kcal/mol
Final Gibbs free energy         ...  -1898.69073187 Eh
G-E(el)                           ...      0.00176214 Eh       1.11 kcal/mol
"""


def test_parse_orca_thermochemistry():
    from cadmia.io.electronic import _parse_orca_thermochemistry
    t = _parse_orca_thermochemistry(_ORCA_THERMO.splitlines())
    assert t["enthalpy_hartree"] == pytest.approx(-1898.64464673, abs=1e-8)
    assert t["gibbs_free_energy_hartree"] == pytest.approx(-1898.69073187, abs=1e-8)
    assert t["entropy_term_hartree"] == pytest.approx(0.04608514, abs=1e-8)
    assert t["zpe_correction_hartree"] == pytest.approx(0.03706219, abs=1e-8)
    assert t["temperature_kelvin"] == pytest.approx(298.15, abs=1e-6)


def test_parse_orca_thermochemistry_absent_when_no_freq():
    from cadmia.io.electronic import _parse_orca_thermochemistry
    assert _parse_orca_thermochemistry(["FINAL SINGLE POINT ENERGY  -100.0"]) == {}


# A second THERMOCHEMISTRY block, as a re-run or a concatenated output carries. Values are
# deliberately far apart: reading the wrong one is not a rounding difference, it is another
# molecule.
_ORCA_THERMO_SECOND = """\
                              THERMOCHEMISTRY AT 298.15K
-------------------------------------------------------
Temperature         ...   350.00 K
Pressure            ...     1.00 atm
Zero point energy                ...      0.02177826 Eh      13.67 kcal/mol
Total Enthalpy                    ...  -1710.14031232 Eh
Final entropy term                ...      0.04150484 Eh     26.04 kcal/mol
Final Gibbs free energy         ...  -1710.18181716 Eh
"""


def test_parse_orca_thermochemistry_keeps_the_last_block():
    """Regression: the energy already came from the last SCF, so G/H/S must come from the
    last thermochemistry block too — otherwise ΔG_ads mixes two geometries in silence."""
    from cadmia.io.electronic import _parse_orca_thermochemistry
    t = _parse_orca_thermochemistry(
        (_ORCA_THERMO + _ORCA_THERMO_SECOND).splitlines())
    assert t["gibbs_free_energy_hartree"] == pytest.approx(-1710.18181716, abs=1e-8)
    assert t["enthalpy_hartree"] == pytest.approx(-1710.14031232, abs=1e-8)
    assert t["zpe_correction_hartree"] == pytest.approx(0.02177826, abs=1e-8)
    assert t["temperature_kelvin"] == pytest.approx(350.00, abs=1e-6)


def test_parse_orca_thermochemistry_truncated_last_block_does_not_borrow_fields():
    """A block cut short (killed job) must yield None for what it lacks, not the value of
    the previous block: reading a whole block is what keeps the fields consistent."""
    from cadmia.io.electronic import _parse_orca_thermochemistry
    truncated = "THERMOCHEMISTRY AT 298.15K\nTemperature         ...   298.15 K\n"
    t = _parse_orca_thermochemistry((_ORCA_THERMO + truncated).splitlines())
    assert t["temperature_kelvin"] == pytest.approx(298.15)
    assert t.get("gibbs_free_energy_hartree") is None    # not -1898.69 from the block above
    assert t.get("enthalpy_hartree") is None


_REAL_ORCA = __import__("pathlib").Path(__file__).resolve().parent.parent / "fixtures" / "1tial_orca"

@pytest.mark.skipif(not (_REAL_ORCA / "1tial_cation_orca.out").exists(),
                    reason="1tial_orca fixtures not present")
def test_parse_orca_electronic_real_cation_cross_spin():
    from cadmia.io.orca import parse_orca_geometry
    from cadmia.io.electronic import parse_orca_electronic
    f = _REAL_ORCA / "1tial_cation_orca.out"
    es = parse_orca_electronic(f, parse_orca_geometry(f))
    # The β LUMO (-11.8784) is well below the α LUMO (-8.0776); the old parser
    # returned the alpha-only -8.0776 (a 3.8 eV error).
    assert es.lumo_ev == pytest.approx(-11.8784, abs=1e-3)
    assert es.homo_ev == pytest.approx(-13.6053, abs=1e-3)


@pytest.mark.skipif(not (_REAL_ORCA / "1tial_N_orca.out").exists(),
                    reason="1tial_orca fixtures not present")
def test_parse_orca_dipole_matches_printed_debye_magnitude():
    import numpy as np
    from cadmia.io.orca import parse_orca_geometry
    from cadmia.io.electronic import parse_orca_electronic
    f = _REAL_ORCA / "1tial_N_orca.out"
    es = parse_orca_electronic(f, parse_orca_geometry(f))
    assert es.dipole_debye is not None
    # ORCA prints the vector in a.u.; the parser converts to Debye, so its
    # magnitude must match the file's "Magnitude (Debye)" line.
    assert float(np.linalg.norm(es.dipole_debye)) == pytest.approx(6.693495273, abs=1e-3)


# ─────────────────────────────────────────────
# Level-of-theory signature (parse_level_of_theory / describe_level_mismatch)
# ─────────────────────────────────────────────

from cadmia.io.electronic import parse_level_of_theory, describe_level_mismatch

_ORCA_HEAD = """\
                                       INPUT FILE
================================================================================
|  1> ! {kw}
|  2> * xyz 0 1
Your calculation utilizes the basis: {basis}
"""

_GAUSS_HEAD = " # {route}\n Standard basis: {basis} (5D, 7F)\n"


def _orca(tmp_path, name, kw, basis):
    p = tmp_path / name
    p.write_text(_ORCA_HEAD.format(kw=kw, basis=basis))
    return p


def _gauss_lvl(tmp_path, name, route, basis):
    p = tmp_path / name
    p.write_text(_GAUSS_HEAD.format(route=route, basis=basis))
    return p


def test_level_orca_basis_and_dispersion(tmp_path):
    f = _orca(tmp_path, "n.out", "PBE0 def2-TZVP def2/J D3BJ RIJCOSX", "def2-TZVP")
    assert parse_level_of_theory(f) == {"basis": "def2tzvp", "dispersion": "d3bj"}


def test_level_orca_no_dispersion_is_none(tmp_path):
    f = _orca(tmp_path, "n.out", "PBE0 def2-SVP def2/J RIJCOSX", "def2-SVP")
    assert parse_level_of_theory(f) == {"basis": "def2svp", "dispersion": "none"}


def test_level_gaussian_normalizes_gd3_and_basis(tmp_path):
    f = _gauss_lvl(tmp_path, "n.out", "PBE1PBE Def2TZVP EmpiricalDispersion=GD3", "def2TZVP")
    # ORCA's def2-TZVP/D3BJ and Gaussian's def2TZVP/GD3 must normalize comparably
    assert parse_level_of_theory(f) == {"basis": "def2tzvp", "dispersion": "d3"}


def test_level_orca_ignores_dispersion_in_citation_banner(tmp_path):
    # D3/D4 appear in ORCA's author credits; only the echoed `!` line counts.
    p = tmp_path / "n.out"
    p.write_text("Sebastian Ehlert : r2SCAN-3c, D4, dhf basis sets\n" + _ORCA_HEAD.format(
        kw="PBE0 def2-SVP def2/J RIJCOSX", basis="def2-SVP"))
    assert parse_level_of_theory(p)["dispersion"] == "none"


def test_describe_level_mismatch_flags_basis_difference(tmp_path):
    a = _orca(tmp_path, "a.out", "PBE0 def2-TZVP def2/J D3BJ", "def2-TZVP")
    b = _orca(tmp_path, "b.out", "PBE0 def2-SVP def2/J D3BJ", "def2-SVP")
    msg = describe_level_mismatch({"adduct": a, "probe": b})
    assert msg is not None and "basis" in msg and "def2tzvp" in msg and "def2svp" in msg


def test_describe_level_mismatch_silent_when_equal(tmp_path):
    a = _orca(tmp_path, "a.out", "PBE0 def2-TZVP def2/J D3BJ", "def2-TZVP")
    b = _orca(tmp_path, "b.out", "PBE0 def2-TZVP def2/J D3BJ", "def2-TZVP")
    assert describe_level_mismatch({"adduct": a, "probe": b}) is None


def test_describe_level_mismatch_silent_when_component_missing(tmp_path):
    # No basis line for b → component unknown → no false alarm.
    a = _orca(tmp_path, "a.out", "PBE0 def2-TZVP def2/J D3BJ", "def2-TZVP")
    b = tmp_path / "b.out"
    b.write_text("| 1> ! PBE0 def2-SVP D3BJ\n")     # dispersion present, basis absent
    assert describe_level_mismatch({"adduct": a, "probe": b}) is None


def test_koopmans_section_flags_cross_spin_quantities_only_in_open_shell():
    from cadmia.cdft.report import _section_global_koopmans
    gk = compute_global_descriptors(-7.010, -4.804)
    closed = _section_global_koopmans(gk)
    opened = _section_global_koopmans(gk, open_shell=True)
    assert "cross-spin" not in closed
    assert "HOMO-LUMO gap (cross-spin):" in opened
    assert "(η, S, μ, χ, ω, ω-, ω+) are cross-spin" in opened
