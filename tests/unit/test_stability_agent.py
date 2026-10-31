"""Tests for cadmia.agents.stability_agent (relative/formation stability)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadmia.agents.stability_agent import compute_stability, compute_series_stability

_HARTREE_TO_KCAL = 627.509
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Ti": 22}


def _gauss(path: Path, atoms, energy, thermo=None, freqs=(120.0, 250.0, 400.0), mult=1,
           temperature=298.15, sigma=None):
    lines = [
        f" Charge =  0 Multiplicity = {mult}",
        "", " Standard orientation:",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
    ]
    for i, (sym, x, y, z) in enumerate(atoms, 1):
        lines.append(f"   {i:>4}    {_Z[sym]:>6}           0     {x:11.6f} {y:11.6f} {z:11.6f}")
    lines.append(" ---------------------------------------------------------------------")
    lines.append(f" SCF Done:  E(RB3LYP) = {energy:.8f}     A.U. after   10 cycles")
    lines.append(" Alpha  occ. eigenvalues --  -0.30000")
    lines.append(" Alpha virt. eigenvalues --  -0.10000")
    if thermo:   # (zpe, h_corr, g_corr) added to E, as Gaussian prints the sums
        zpe, h, g = thermo
        lines += [
            " Frequencies --  " + "  ".join(f"{f:.4f}" for f in freqs),
            f" Temperature   {temperature:.3f} Kelvin.  Pressure   1.00000 Atm.",
            *([f" Rotational symmetry number {sigma:>2}."] if sigma else []),
            f" Sum of electronic and zero-point Energies=          {energy + zpe:.6f}",
            f" Sum of electronic and thermal Enthalpies=           {energy + h:.6f}",
            f" Sum of electronic and thermal Free Energies=        {energy + g:.6f}",
        ]
    lines.append(" Normal termination of Gaussian 16.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _al2tio5(path, e):  # 1 Ti, 2 Al, 5 O = 8 atoms
    _gauss(path, [("Ti", 0, 0, 0), ("Al", 1, 0, 0), ("Al", -1, 0, 0),
                  ("O", 2, 0, 0), ("O", -2, 0, 0), ("O", 0, 2, 0),
                  ("O", 0, -2, 0), ("O", 0, 0, 2)], e)


def _tio2(path, e):
    _gauss(path, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0), ("O", -1.8, 0, 0)], e)


def _al2o3(path, e):
    _gauss(path, [("Al", 0, 0, 0), ("Al", 2, 0, 0),
                  ("O", 1, 0, 0), ("O", 1, 1, 0), ("O", 1, -1, 0)], e)


@pytest.fixture
def oxides(tmp_path):
    mixed = tmp_path / "al2tio5.out"
    tio2 = tmp_path / "tio2.out"
    al2o3 = tmp_path / "al2o3.out"
    _al2tio5(mixed, -150.0)
    _tio2(tio2, -100.0)
    _al2o3(al2o3, -49.9)
    return mixed, tio2, al2o3


def test_auto_stoichiometry_balances(oxides):
    mixed, tio2, al2o3 = oxides
    r = compute_stability(mixed, [tio2, al2o3])
    assert r.balanced is True
    coeffs = {ref.formula: ref.coefficient for ref in r.references}
    # Al2TiO5 = 1·TiO2 + 1·Al2O3
    assert coeffs.get("O2Ti") == 1 or coeffs.get("TiO2") == 1
    assert all(c == 1 for c in (ref.coefficient for ref in r.references))


def test_relative_stability_value(oxides):
    mixed, tio2, al2o3 = oxides
    r = compute_stability(mixed, [tio2, al2o3])
    expected = (-150.0 - (-100.0) - (-49.9)) * _HARTREE_TO_KCAL  # -0.1 Ha → negative
    assert r.e_rel_kcal == pytest.approx(expected, abs=1e-3)
    assert r.e_rel_kcal < 0           # mixed more stable than pure oxides
    assert r.e_rel_eV_per_atom == pytest.approx(r.e_rel_eV / 8, abs=1e-9)


def test_explicit_coefficients(oxides):
    mixed, tio2, al2o3 = oxides
    r = compute_stability(mixed, [tio2, al2o3], coefficients=[1, 1])
    assert r.balanced is True
    assert r.e_rel_kcal == pytest.approx((-150.0 + 100.0 + 49.9) * _HARTREE_TO_KCAL, abs=1e-3)


def test_unbalanced_stoichiometry_warns(oxides):
    mixed, tio2, al2o3 = oxides
    r = compute_stability(mixed, [tio2, al2o3], coefficients=[1, 2])  # wrong
    assert r.balanced is False
    assert any("balance" in w.lower() or "stoich" in w.lower() for w in r.warnings)


def _al2tio5_n(path, n, e):
    """A (TiAl2O5)_n cluster: n repeats of the 8-atom motif, so gcd of counts = n."""
    atoms = []
    for k in range(n):
        dx = 10.0 * k
        atoms += [("Ti", dx, 0, 0), ("Al", dx + 1, 0, 0), ("Al", dx - 1, 0, 0),
                  ("O", dx + 2, 0, 0), ("O", dx - 2, 0, 0), ("O", dx, 2, 0),
                  ("O", dx, -2, 0), ("O", dx, 0, 2)]
    _gauss(path, atoms, e)


@pytest.fixture
def series(tmp_path):
    n1, n2, n3 = tmp_path / "n1.out", tmp_path / "n2.out", tmp_path / "n3.out"
    tio2, al2o3 = tmp_path / "tio2.out", tmp_path / "al2o3.out"
    _al2tio5_n(n1, 1, -150.0)
    _al2tio5_n(n2, 2, -305.0)
    _al2tio5_n(n3, 3, -460.0)
    _tio2(tio2, -100.0)
    _al2o3(al2o3, -49.9)
    return n1, n2, n3, tio2, al2o3


def test_series_infers_n_from_stoichiometry(series):
    n1, n2, n3, tio2, al2o3 = series
    r = compute_series_stability([n3, n1, n2])  # unordered on purpose
    assert [m.n for m in r.members] == [1, 2, 3]  # sorted by n
    assert r.members[2].formula == "Al6O15Ti3"


def test_series_fragmentation_and_second_order(series):
    n1, n2, n3, tio2, al2o3 = series
    r = compute_series_stability([n1, n2, n3])
    by_n = {m.n: m for m in r.members}
    # E_f(2) = e1 + e1 - e2 = +5 Ha ; E_f(3) = e2 + e1 - e3 = +5 Ha
    assert by_n[2].e_f_kcal == pytest.approx(5.0 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[3].e_f_kcal == pytest.approx(5.0 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[1].e_f_kcal is None                # n=1 has no n-1
    # Δ²E(2) = e1 + e3 - 2 e2 = 0 ; edges undefined
    assert by_n[2].delta2_kcal == pytest.approx(0.0, abs=1e-3)
    assert by_n[1].delta2_kcal is None and by_n[3].delta2_kcal is None


def test_series_binding_energy_needs_monomers(series):
    n1, n2, n3, tio2, al2o3 = series
    r = compute_series_stability([n1, n2, n3], reference_outs=[tio2, al2o3])
    by_n = {m.n: m for m in r.members}
    # E_b(1) = e1/1 - (e_tio2 + e_al2o3) = -150 - (-149.9) = -0.1 Ha
    assert by_n[1].e_b_kcal == pytest.approx(-0.1 * _HARTREE_TO_KCAL, abs=1e-3)
    # E_b(2) = e2/2 - (-149.9) = -152.5 + 149.9 = -2.6 Ha
    assert by_n[2].e_b_kcal == pytest.approx(-2.6 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[1].e_b_kcal < 0   # bound relative to pure oxides


def test_series_binding_energy_skipped_without_monomers(series):
    n1, n2, n3, tio2, al2o3 = series
    r = compute_series_stability([n1, n2, n3])
    assert all(m.e_b_kcal is None for m in r.members)
    assert any("E_b" in w for w in r.warnings)


def test_persist_stability_result(tmp_path, oxides):
    from cadmia.db.session import init_db, get_connection
    from cadmia.db import repository
    mixed, tio2, al2o3 = oxides
    r = compute_stability(mixed, [tio2, al2o3], mixed_label="Al2TiO5")

    db = tmp_path / "stab.db"
    conn = init_db(db)
    try:
        ids = repository.persist_stability_result(conn, r)
        conn.commit()
    finally:
        conn.close()

    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1
        row = conn.execute(
            "SELECT e_rel_kcal, balanced, n_atoms, references_json FROM stability"
        ).fetchone()
        assert row["balanced"] == 1
        assert row["n_atoms"] == 8
        assert row["e_rel_kcal"] == pytest.approx(r.e_rel_kcal, abs=1e-3)
        import json
        refs = json.loads(row["references_json"])
        assert len(refs) == 2
        assert conn.execute("SELECT COUNT(*) FROM run_log WHERE agent='stability'").fetchone()[0] == 1


# ── The energy basis ────────────────────────────────────────────────

_TIO2_ATOMS = [("Ti", 0, 0, 0), ("O", 1.8, 0, 0), ("O", -1.8, 0, 0)]
_AL2O3_ATOMS = [("Al", 0, 0, 0), ("Al", 2, 0, 0), ("O", 1, 0, 0), ("O", 1, 1, 0), ("O", 1, -1, 0)]
_AL2TIO5_ATOMS = [("Ti", 0, 0, 0), ("Al", 1, 0, 0), ("Al", -1, 0, 0), ("O", 2, 0, 0),
                  ("O", -2, 0, 0), ("O", 0, 2, 0), ("O", 0, -2, 0), ("O", 0, 0, 2)]


@pytest.fixture
def oxides_with_freq(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0, thermo=(0.030, 0.040, 0.010))
    _gauss(tio2, _TIO2_ATOMS, -100.0, thermo=(0.010, 0.015, 0.000))
    _gauss(al2o3, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002))
    return mixed, tio2, al2o3


def test_stability_defaults_to_zero_point_corrected_energy(oxides_with_freq):
    r = compute_stability(*oxides_with_freq[:1], list(oxides_with_freq[1:]))
    assert r.energy_basis == "E0"
    # E0: -0.1 Ha electronic + (0.030 - 0.010 - 0.015) ZPE = -0.095 Ha
    assert r.e_rel_kcal == pytest.approx(-0.095 * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.e_mixed_hartree == pytest.approx(-149.970)
    by_basis = r.e_rel_by_basis_kcal
    assert set(by_basis) == {"E0", "E", "H", "G", "Gq"}
    assert by_basis["E"] == pytest.approx(-0.100 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_basis["H"] == pytest.approx(-0.095 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_basis["G"] == pytest.approx(-0.092 * _HARTREE_TO_KCAL, abs=1e-3)
    assert not r.warnings


def test_stability_never_mixes_bases_when_one_output_lacks_frequencies(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0, thermo=(0.030, 0.040, 0.010))
    _gauss(tio2, _TIO2_ATOMS, -100.0)                                  # single point only
    _gauss(al2o3, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002))
    r = compute_stability(mixed, [tio2, al2o3])
    assert r.energy_basis == "E"
    assert r.e_rel_kcal == pytest.approx(-0.100 * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.e_mixed_hartree == pytest.approx(-150.0)                 # E, not E0, for all terms
    assert set(r.e_rel_by_basis_kcal) == {"E"}
    assert any("falls back to the electronic energy E" in w for w in r.warnings)


def test_explicit_energy_basis_is_honoured(oxides_with_freq):
    mixed, tio2, al2o3 = oxides_with_freq
    r = compute_stability(mixed, [tio2, al2o3], energy="G")
    assert r.energy_basis == "G"
    assert r.e_rel_kcal == pytest.approx(-0.092 * _HARTREE_TO_KCAL, abs=1e-3)


def test_series_family_binding_energy_uses_the_series_own_monomer(tmp_path):
    paths = []
    for n, e in ((1, -150.0), (2, -305.0), (3, -460.0)):
        p = tmp_path / f"n{n}.out"
        _al2tio5_n(p, n, e)
        paths.append(p)
    r = compute_series_stability(paths)
    by_n = {m.n: m for m in r.members}
    assert by_n[1].e_b_family_kcal is None
    # (2·(-150) - (-305))/2 = +2.5 Ha ; (3·(-150) - (-460))/3 = +10/3 Ha
    assert by_n[2].e_b_family_kcal == pytest.approx(2.5 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[3].e_b_family_kcal == pytest.approx(10 / 3 * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.energy_basis == "E"     # synthetic outputs carry no frequencies


def test_persisted_stability_records_its_energy_basis(tmp_path, oxides_with_freq):
    from cadmia.db.session import init_db, get_connection
    from cadmia.db import repository
    mixed, tio2, al2o3 = oxides_with_freq
    r = compute_stability(mixed, [tio2, al2o3])
    db = tmp_path / "stab.db"
    conn = init_db(db)
    try:
        repository.persist_stability_result(conn, r)
        conn.commit()
    finally:
        conn.close()
    with get_connection(db) as conn:
        assert conn.execute("SELECT energy_basis FROM stability").fetchone()[0] == "E0"


# ── Imaginary modes, multiplicity, temperature, coefficients ───────

def test_stability_flags_imaginary_modes_and_records_them_per_term(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0, thermo=(0.030, 0.040, 0.010), freqs=(-85.3, 120.0, 300.0))
    _gauss(tio2, _TIO2_ATOMS, -100.0, thermo=(0.010, 0.015, 0.000))
    _gauss(al2o3, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002))
    r = compute_stability(mixed, [tio2, al2o3], reference_labels=["TiO2", "Al2O3"])
    assert r.n_imaginary == {"mixed": 1, "TiO2": 0, "Al2O3": 0}
    assert any("Imaginary frequencies in: mixed (1)" in w for w in r.warnings)


def test_stability_without_frequencies_is_not_verified_as_minimum(oxides):
    r = compute_stability(oxides[0], list(oxides[1:]), energy="E")
    assert set(r.n_imaginary.values()) == {None}
    assert any("not verified as minima" in w for w in r.warnings)


def test_stability_declares_a_change_of_spin_multiplicity(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0, thermo=(0.030, 0.040, 0.010))
    _gauss(tio2, _TIO2_ATOMS, -100.0, thermo=(0.010, 0.015, 0.000))
    _gauss(al2o3, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002), mult=3)
    r = compute_stability(mixed, [tio2, al2o3], reference_labels=["TiO2", "Al2O3"])
    assert r.multiplicities == {"mixed": 1, "TiO2": 1, "Al2O3": 3}
    assert any("Spin multiplicity differs" in w for w in r.warnings)


def test_stability_warns_when_g_comes_from_different_temperatures(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0, thermo=(0.030, 0.040, 0.010))
    _gauss(tio2, _TIO2_ATOMS, -100.0, thermo=(0.010, 0.015, 0.000), temperature=400.0)
    _gauss(al2o3, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002))
    assert any("different temperatures" in w
               for w in compute_stability(mixed, [tio2, al2o3], energy="G").warnings)
    assert not any("different temperatures" in w
                   for w in compute_stability(mixed, [tio2, al2o3]).warnings)   # E0: no T


def test_wrong_number_of_coefficients_is_an_error_not_a_number(oxides):
    with pytest.raises(ValueError, match="2 references"):
        compute_stability(oxides[0], list(oxides[1:]), coefficients=[1])


def test_series_records_multiplicity_and_imaginary_modes_of_every_member(tmp_path):
    n1, n2 = tmp_path / "n1.out", tmp_path / "n2.out"
    _gauss(n1, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002), mult=3)
    _gauss(n2, _AL2O3_ATOMS + [(s, x + 5, y, z) for s, x, y, z in _AL2O3_ATOMS], -100.0,
           thermo=(0.031, 0.041, 0.005), freqs=(-40.0, 90.0))
    r = compute_series_stability([n1, n2], labels=["n1", "n2"])
    assert r.multiplicities == {"n1": 3, "n2": 1}
    assert r.n_imaginary == {"n1": 0, "n2": 1}
    assert any("Imaginary frequencies in: n2 (1)" in w for w in r.warnings)
    assert any("Spin multiplicity differs" in w for w in r.warnings)


def test_series_never_mixes_bases_when_one_cluster_lacks_frequencies(tmp_path):
    n1, n2 = tmp_path / "n1.out", tmp_path / "n2.out"
    _gauss(n1, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002))
    _gauss(n2, _AL2O3_ATOMS + [(s, x + 5, y, z) for s, x, y, z in _AL2O3_ATOMS], -100.0)
    r = compute_series_stability([n1, n2])
    assert r.energy_basis == "E"
    by_n = {m.n: m for m in r.members}
    assert by_n[1].e_hartree == pytest.approx(-49.9)          # E, not E0, also for n = 1
    assert any("falls back to the electronic energy E" in w for w in r.warnings)


# ── Symmetry from the geometry ─────────────────────────────────────

def test_g_is_corrected_when_the_engine_missed_the_symmetry(tmp_path):
    """The toy Al2TiO5 is C2v (σ 2); an engine that printed σ 1 left G too low by RT·ln 2."""
    import math
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0, thermo=(0.030, 0.040, 0.010), sigma=1)
    _gauss(tio2, _TIO2_ATOMS, -100.0, thermo=(0.010, 0.015, 0.000), sigma=2)     # D∞h, right
    _gauss(al2o3, _AL2O3_ATOMS, -49.9, thermo=(0.015, 0.020, 0.002), sigma=4)    # D2h, right
    r = compute_stability(mixed, [tio2, al2o3], energy="G", reference_labels=["TiO2", "Al2O3"])
    rt_ln2 = 0.0019872043 * 298.15 * math.log(2)
    assert r.e_rel_kcal == pytest.approx(-0.092 * _HARTREE_TO_KCAL + rt_ln2, abs=2e-3)
    assert r.symmetry["mixed"]["point_group"] == "C2v" and r.symmetry["mixed"]["engine_sigma"] == 1
    assert any("differs from the one the engine used" in w for w in r.warnings)
    assert r.e_rel_by_basis_kcal["E0"] == pytest.approx(-0.095 * _HARTREE_TO_KCAL, abs=1e-3)  # σ-free


# ── Rational coefficients and references of any size ────────

def _copies(atoms, k):
    return [(s, x + 10.0 * i, y, z) for i in range(k) for s, x, y, z in atoms]


def test_references_of_twice_the_size_give_one_half_each(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t2.out", tmp_path / "a2.out"
    _gauss(mixed, _AL2TIO5_ATOMS, -150.0)
    _gauss(tio2, _copies(_TIO2_ATOMS, 2), -200.3)
    _gauss(al2o3, _copies(_AL2O3_ATOMS, 2), -99.9)
    r = compute_stability(mixed, [tio2, al2o3], energy="E")
    assert r.balanced and [ref.coefficient for ref in r.references] == [0.5, 0.5]
    assert r.reference_size == "equal-size"
    assert r.e_rel_kcal == pytest.approx(0.1 * _HARTREE_TO_KCAL, abs=1e-3)
    explicit = compute_stability(mixed, [tio2, al2o3], energy="E", coefficients=["1/2", "0.5"])
    assert explicit.balanced and explicit.e_rel_kcal == pytest.approx(r.e_rel_kcal)


def test_same_n_references_keep_integer_coefficients_and_give_per_formula_unit(tmp_path):
    mixed, tio2, al2o3 = tmp_path / "m.out", tmp_path / "t.out", tmp_path / "a.out"
    _al2tio5_n(mixed, 2, -305.0)
    _gauss(tio2, _copies(_TIO2_ATOMS, 2), -200.3)
    _gauss(al2o3, _copies(_AL2O3_ATOMS, 2), -99.9)
    r = compute_stability(mixed, [tio2, al2o3])
    assert [ref.coefficient for ref in r.references] == [1, 1] and r.reference_size == "same-n"
    assert r.formula_units == 2
    assert r.e_rel_kcal_per_fu == pytest.approx(-4.8 / 2 * _HARTREE_TO_KCAL, abs=1e-3)


def test_series_matches_references_of_every_size_by_stoichiometry(tmp_path):
    n1, n2 = tmp_path / "n1.out", tmp_path / "n2.out"
    _al2tio5_n(n1, 1, -150.0)
    _al2tio5_n(n2, 2, -305.0)
    refs = []
    for k, e_t, e_a in ((1, -100.0, -49.9), (2, -200.3, -99.9), (4, -400.8, -199.9)):
        refs += [tmp_path / f"t{k}.out", tmp_path / f"a{k}.out"]
        _gauss(refs[-2], _copies(_TIO2_ATOMS, k), e_t)
        _gauss(refs[-1], _copies(_AL2O3_ATOMS, k), e_a)
    by_n = {m.n: m for m in compute_series_stability([n2, n1], reference_outs=refs).members}
    assert by_n[1].e_b_kcal == pytest.approx(-0.1 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[1].de_mix_kcal == pytest.approx(-0.1 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[1].de_mix_eq_kcal == pytest.approx(0.1 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[2].e_b_kcal == pytest.approx(-2.6 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[2].de_mix_kcal == pytest.approx(-4.8 * _HARTREE_TO_KCAL, abs=1e-3)
    assert by_n[2].de_mix_eq_kcal == pytest.approx(-2.325 * _HARTREE_TO_KCAL, abs=1e-3)

    r = compute_series_stability([n1, n2], reference_outs=refs[:4])     # no size 4
    assert r.members[1].de_mix_eq_kcal is None
    assert any("ΔE_mix,eq not computed for n=2" in w for w in r.warnings)


# ── Isomers of the same n ─────────────────────────

_CHIRAL_AL2TIO5 = [("Ti", 0.0, 0.0, 0.0), ("Al", 1.7, 0.3, 0.2), ("Al", -0.4, 2.1, 0.5),
                   ("O", 0.3, -0.6, 1.9), ("O", 2.6, 1.4, -0.7), ("O", -1.8, 0.4, -0.9),
                   ("O", 1.1, 2.9, 1.3), ("O", -0.9, -1.7, 0.6)]


@pytest.fixture
def isomers(tmp_path):
    """a: achiral, lowest in E0. b: chiral, lowest in G by 0.003 Ha."""
    a, b = tmp_path / "iso_a.out", tmp_path / "iso_b.out"
    _gauss(a, _AL2TIO5_ATOMS, -150.000, thermo=(0.030, 0.040, 0.010))   # E0 -149.970, G -149.990
    _gauss(b, _CHIRAL_AL2TIO5, -150.001, thermo=(0.032, 0.042, 0.008))  # E0 -149.969, G -149.993
    return a, b


def test_isomers_are_kept_and_the_order_of_the_files_does_not_matter(isomers):
    a, b = isomers
    one, two = compute_series_stability([a, b]), compute_series_stability([b, a])
    assert one.to_dict() == two.to_dict()
    rows = {i.label: i for i in one.isomers}
    assert set(rows) == {"iso_a", "iso_b"} and rows["iso_b"].chiral and not rows["iso_a"].chiral
    assert rows["iso_b"].delta_kcal["E0"] == pytest.approx(0.001 * _HARTREE_TO_KCAL, abs=1e-3)
    assert rows["iso_a"].delta_kcal["G"] == pytest.approx(0.003 * _HARTREE_TO_KCAL, abs=1e-3)
    assert any("different isomers" in note for note in one.notes)


def test_the_representative_follows_the_chosen_basis(isomers):
    assert compute_series_stability(list(isomers)).members[0].label == "iso_a"            # E0
    assert compute_series_stability(list(isomers), energy="G").members[0].label == "iso_b"


def test_populations_are_off_by_default_and_count_chirality_when_asked(isomers):
    import math
    assert all(i.population is None for i in compute_series_stability(list(isomers)).isomers)
    r = compute_series_stability(list(isomers), populations=True)
    rows = {i.label: i for i in r.isomers}
    w_a = math.exp(-0.003 / (3.166811563e-6 * 298.15))
    assert (rows["iso_a"].degeneracy, rows["iso_b"].degeneracy) == (1, 2)
    assert rows["iso_a"].population == pytest.approx(w_a / (w_a + 2))
    weighted = r.weighted_members[0].e_hartree
    assert weighted == pytest.approx(rows["iso_a"].population * -149.970
                                     + rows["iso_b"].population * -149.969)
    assert r.population_basis == "G" and r.population_temperature_k == 298.15
    # E0 basis: same g, ΔE0 instead of ΔG; a is lower in E0 by 0.001 Ha
    w_b = 2 * math.exp(-0.001 / (3.166811563e-6 * 298.15))
    assert rows["iso_a"].population_e0 == pytest.approx(1 / (1 + w_b))
    assert rows["iso_b"].population_e0 == pytest.approx(w_b / (1 + w_b))


def test_cli_pop_flag_reports_both_population_bases(isomers):
    from click.testing import CliRunner
    from cadmia.cli import main
    import json
    out = CliRunner().invoke(main, ["stability-series", "--cluster", str(isomers[0]),
                                    "--cluster", str(isomers[1]), "--pop", "--no-table"])
    assert out.exit_code == 0, out.output
    rows = json.loads(out.output)["isomers"]
    assert all(r["population"] is not None and r["population_e0"] is not None for r in rows)


def test_one_cluster_per_n_has_no_isomer_rows(series):
    r = compute_series_stability(list(series[:3]), populations=True)
    assert r.isomers == [] and [m.e_hartree for m in r.weighted_members] == [m.e_hartree for m in r.members]


def test_cli_reference_options_are_ref_only(series):
    from click.testing import CliRunner
    from cadmia.cli import main
    n1, n2, n3, tio2, al2o3 = series
    out = CliRunner().invoke(main, ["stability-series", "--cluster", str(n1), "--cluster", str(n2),
                                    "--tio2", str(tio2), "--no-table"])
    assert out.exit_code != 0 and "No such option" in out.output


def test_stability_warns_when_a_calculation_did_not_terminate_normally(oxides):
    mixed, tio2, al2o3 = oxides
    tio2.write_text(tio2.read_text(encoding="utf-8").replace(" Normal termination of Gaussian 16.",
                                             " Error termination via Lnk1e in l508.exe."))
    al2o3.write_text(al2o3.read_text(encoding="utf-8").replace(" Normal termination of Gaussian 16.\n", ""))
    r = compute_stability(mixed, [tio2, al2o3], reference_labels=["TiO2", "Al2O3"], energy="E")
    w = next(w for w in r.warnings if w.startswith("Did not terminate normally"))
    assert "TiO2 (Error termination)" in w and "Al2O3 (No termination marker found)" in w
    assert "mixed" not in w


def test_series_labels_that_collide_are_made_unique():
    from cadmia.agents.stability_agent import _unique_labels
    terms = [("opt", "n1/opt.out", None, "orca"), ("opt", "n2/opt.out", None, "orca"),
             ("opt", "n2/opt.out", None, "orca"), ("TiO2", "ref/TiO2.out", None, "orca")]
    assert [t[0] for t in _unique_labels(terms)] == ["opt (n1)", "opt (n2)", "opt (n2) #2", "TiO2"]
