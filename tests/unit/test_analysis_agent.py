"""Tests for cadmia.agents.analysis_agent (Analysis Agent — adsorption energetics)."""
from __future__ import annotations

from pathlib import Path

import hashlib

import pytest

from cadmia.agents import analysis_agent
from cadmia.agents.analysis_agent import analyze

_HARTREE_TO_KCAL = 627.509
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Ti": 22}


def _gauss(path: Path, atoms, energy, bsse_ha=None):
    """Write a minimal Gaussian-style output with geometry + SCF energy (+ optional BSSE)."""
    lines = [
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
    if bsse_ha is not None:
        lines.append(f" Counterpoise: corrected energy =    {energy + bsse_ha:.10f}")
        lines.append(f" Counterpoise: BSSE energy =        {bsse_ha:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def triplet(tmp_path):
    """Ti–O cluster + free H2 + adduct (H2 near Ti, H–H stretched 0.74 → 0.82 Å)."""
    cluster = tmp_path / "cluster.out"
    h2 = tmp_path / "h2.out"
    adduct = tmp_path / "adduct.out"
    _gauss(cluster, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0)], -100.0)
    _gauss(h2, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17)
    _gauss(adduct, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                    ("H", 0, 0, 1.9), ("H", 0, 0, 2.72)], -101.20)
    return cluster, h2, adduct


@pytest.fixture
def triplet_cp(tmp_path):
    """The same triplet with a counterpoise adduct, in Gaussian's real format."""
    cluster = tmp_path / "cluster.out"
    h2 = tmp_path / "h2.out"
    adduct = tmp_path / "adduct_cp.out"
    _gauss(cluster, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0)], -100.0)
    _gauss(h2, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17)
    _gauss(adduct, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                    ("H", 0, 0, 1.9), ("H", 0, 0, 2.72)], -101.20, bsse_ha=0.0020)
    return cluster, h2, adduct


_CDFT = {
    "sites": [
        {"atom_index": 1, "atom_symbol": "Ti", "rank": 1, "character": "electrophilic", "delta_f": 0.20},
        {"atom_index": 2, "atom_symbol": "O", "rank": 2, "character": "nucleophilic", "delta_f": -0.19},
    ]
}


# ─────────────────────────────────────────────
# Core energetics
# ─────────────────────────────────────────────

def test_adsorption_energy(triplet):
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2)
    expected = (-101.20 - (-100.0) - (-1.17)) * _HARTREE_TO_KCAL
    assert r.e_ads_kcal == pytest.approx(expected, abs=1e-3)
    assert r.probe == "H2"
    assert r.engine == "gaussian"


def test_level_mismatch_warns_on_e_ads(triplet):
    """A probe at a different basis than cluster/adduct must raise a warning."""
    cluster, h2, adduct = triplet
    for f in (cluster, adduct):
        f.write_text(f.read_text() + "\n Standard basis: def2TZVP (5D, 7F)\n")
    h2.write_text(h2.read_text() + "\n Standard basis: def2SVP (5D, 7F)\n")   # smaller basis
    r = analyze(adduct, cluster, h2)
    assert any("different levels of theory" in w and "basis" in w for w in r.warnings)


def test_level_match_is_silent(triplet):
    """The common path — all three at the same basis — must not warn (no false alarms)."""
    cluster, h2, adduct = triplet
    for f in (cluster, adduct, h2):
        f.write_text(f.read_text() + "\n Standard basis: def2TZVP (5D, 7F)\n")
    r = analyze(adduct, cluster, h2)
    assert not any("different levels of theory" in w for w in r.warnings)


def test_binding_site_is_nearest_cluster_atom(triplet):
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2)
    # H2 sits above Ti (z), closer to Ti(0,0,0) than O(1.8,0,0)
    assert r.site_atom_index == 1
    assert r.site_atom_symbol == "Ti"


def test_missing_energy_warns(tmp_path, triplet):
    cluster, h2, _ = triplet
    bad = tmp_path / "bad.out"
    bad.write_text(" Standard orientation:\n"
                   " ---------------------------------------------------------------------\n"
                   " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
                   " Number     Number       Type             X           Y           Z\n"
                   " ---------------------------------------------------------------------\n"
                   "      1         22           0        0.0  0.0  0.0\n"
                   "      2          8           0        1.8  0.0  0.0\n"
                   " ---------------------------------------------------------------------\n",
                   encoding="utf-8")
    r = analyze(bad, cluster, h2)
    assert r.e_ads_kcal is None
    assert any("E_ads" in w for w in r.warnings)


# ─────────────────────────────────────────────
# Bond elongation / activation
# ─────────────────────────────────────────────

def test_probe_bond_elongation_and_activation(triplet):
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2)
    assert len(r.probe_bonds) == 1
    b = r.probe_bonds[0]
    assert b.label == "H-H"
    assert b.free_A == pytest.approx(0.74, abs=1e-3)
    assert b.adduct_A == pytest.approx(0.82, abs=1e-3)
    assert r.max_elongation_pct == pytest.approx(10.81, abs=0.1)
    assert r.activated is True


@pytest.mark.filterwarnings(       # synthetic geometry: atoms 3 Å apart, no real polyhedra
    "ignore:Found .* but no metal-ligand bonds:UserWarning")
def test_no_activation_when_bond_unchanged(tmp_path):
    cluster = tmp_path / "c.out"
    h2 = tmp_path / "h2.out"
    adduct = tmp_path / "a.out"
    _gauss(cluster, [("Ti", 0, 0, 0)], -50.0)
    _gauss(h2, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17)
    # H2 physisorbed far away, bond unchanged (0.74)
    _gauss(adduct, [("Ti", 0, 0, 0), ("H", 0, 0, 3.0), ("H", 0.74, 0, 3.0)], -51.18)
    r = analyze(adduct, cluster, h2)
    assert r.activated is False
    assert r.max_elongation_pct == pytest.approx(0.0, abs=0.5)


# ─────────────────────────────────────────────
# BSSE + deformation
# ─────────────────────────────────────────────

def test_bsse_is_not_added_twice_when_the_energy_is_already_corrected(triplet_cp):
    """Gaussian prints the corrected energy, so E_ads built on it already carries the BSSE.

    The parser takes that corrected value as the adduct energy, which is the only way to avoid
    depending on where the complex falls among the counterpoise sub-calculations. Adding the
    BSSE on top would then count it twice.
    """
    cluster, h2, adduct = triplet_cp
    r = analyze(adduct, cluster, h2, bsse=True)
    assert r.bsse_kcal == pytest.approx(0.0020 * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.e_ads_bsse_kcal == pytest.approx(r.e_ads_kcal, abs=1e-9)
    assert any("already includes the BSSE" in w for w in r.warnings)
    # And E_ads itself is built on the corrected energy, not on the raw one.
    raw = (-101.20 - (-100.0) - (-1.17)) * _HARTREE_TO_KCAL
    assert r.e_ads_kcal == pytest.approx(raw + 0.0020 * _HARTREE_TO_KCAL, abs=1e-3)


def test_bsse_flag_without_counterpoise_warns(tmp_path):
    cluster = tmp_path / "c.out"
    h2 = tmp_path / "h2.out"
    adduct = tmp_path / "a.out"
    _gauss(cluster, [("Ti", 0, 0, 0)], -50.0)
    _gauss(h2, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17)
    _gauss(adduct, [("Ti", 0, 0, 0), ("H", 0, 0, 1.9), ("H", 0, 0, 2.64)], -51.20)
    r = analyze(adduct, cluster, h2, bsse=True)
    assert r.bsse_kcal is None
    assert any("counterpoise" in w.lower() for w in r.warnings)


def test_deformation_and_interaction(tmp_path, triplet):
    cluster, h2, adduct = triplet
    cdef = tmp_path / "cdef.out"
    pdef = tmp_path / "pdef.out"
    _gauss(cdef, [("Ti", 0, 0, 0), ("O", 1.8, 0, 0)], -99.99)   # cluster strained: +0.01 Ha
    _gauss(pdef, [("H", 0, 0, 0), ("H", 0.82, 0, 0)], -1.165)   # probe strained: +0.005 Ha
    r = analyze(adduct, cluster, h2, cluster_deformed=cdef, probe_deformed=pdef)
    assert r.e_def_cluster_kcal == pytest.approx(0.01 * _HARTREE_TO_KCAL, abs=1e-2)
    assert r.e_def_probe_kcal == pytest.approx(0.005 * _HARTREE_TO_KCAL, abs=1e-2)
    assert r.e_int_kcal == pytest.approx(
        r.e_ads_kcal - r.e_def_cluster_kcal - r.e_def_probe_kcal, abs=1e-6)


# ─────────────────────────────────────────────
# CDFT cross-reference
# ─────────────────────────────────────────────

def test_cdft_agreement(triplet):
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2, cdft_json=_CDFT)
    assert r.cdft_site_rank == 1
    assert r.cdft_site_character == "electrophilic"
    assert r.cdft_agreement is True


@pytest.mark.filterwarnings(       # synthetic geometry: atoms 3 Å apart, no real polyhedra
    "ignore:Found .* but no metal-ligand bonds:UserWarning")
def test_cdft_nucleophilic_site_is_coherent(tmp_path):
    # H2 binds near the (only) nucleophilic O (atom 2): even though Ti is the global rank-1 site,
    # the strongest donor binding a probe is coherent, not a discrepancy (within-character rank 1).
    cluster = tmp_path / "c.out"
    h2 = tmp_path / "h2.out"
    adduct = tmp_path / "a.out"
    _gauss(cluster, [("Ti", 0, 0, 0), ("O", 3.0, 0, 0)], -100.0)
    _gauss(h2, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17)
    _gauss(adduct, [("Ti", 0, 0, 0), ("O", 3.0, 0, 0),
                    ("H", 3.0, 0, 1.0), ("H", 3.0, 0, 1.74)], -101.10)
    r = analyze(adduct, cluster, h2, cdft_json=_CDFT)
    assert r.site_atom_index == 2
    assert r.cdft_site_character == "nucleophilic"
    assert r.cdft_site_rank == 1
    assert r.cdft_agreement is True


@pytest.mark.filterwarnings(       # synthetic geometry: atoms 3 Å apart, no real polyhedra
    "ignore:Found .* but no metal-ligand bonds:UserWarning")
def test_cdft_discrepancy(tmp_path):
    cluster = tmp_path / "c.out"
    h2 = tmp_path / "h2.out"
    adduct = tmp_path / "a.out"
    # Two nucleophilic O sites; H2 binds the WEAKER one (atom 3) → genuine within-character discrepancy.
    cdft = {"sites": [
        {"atom_index": 1, "atom_symbol": "Ti", "rank": 1, "character": "electrophilic", "delta_f": 0.30},
        {"atom_index": 2, "atom_symbol": "O", "rank": 2, "character": "nucleophilic", "delta_f": -0.20},
        {"atom_index": 3, "atom_symbol": "O", "rank": 3, "character": "nucleophilic", "delta_f": -0.05},
    ]}
    _gauss(cluster, [("Ti", 0, 0, 0), ("O", 3.0, 0, 0), ("O", 6.0, 0, 0)], -100.0)
    _gauss(h2, [("H", 0, 0, 0), ("H", 0.74, 0, 0)], -1.17)
    _gauss(adduct, [("Ti", 0, 0, 0), ("O", 3.0, 0, 0), ("O", 6.0, 0, 0),
                    ("H", 6.0, 0, 1.0), ("H", 6.0, 0, 1.74)], -101.10)
    r = analyze(adduct, cluster, h2, cdft_json=cdft)
    assert r.site_atom_index == 3
    assert r.cdft_site_character == "nucleophilic"
    assert r.cdft_site_rank == 2
    assert r.cdft_agreement is False
    assert any("character" in w.lower() for w in r.warnings)


def test_to_json_roundtrip(triplet):
    import json
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2, bsse=True, cdft_json=_CDFT)
    d = json.loads(r.to_json())
    assert d["probe"] == "H2"
    assert d["activated"] is True
    assert d["probe_bonds"][0]["label"] == "H-H"


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def test_persist_analysis_result(tmp_path, triplet):
    from cadmia.db.session import init_db, get_connection
    from cadmia.db import repository
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2, bsse=True, cdft_json=_CDFT, config_index=0,
                cluster_label="TiO")

    db = tmp_path / "an.db"
    conn = init_db(db)
    try:
        ids = repository.persist_analysis_result(conn, r)
        conn.commit()
    finally:
        conn.close()

    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 1
        row = conn.execute(
            "SELECT probe, site_atom_idx, config_index, e_ads_kcal, e_bsse_kcal, delta_bond_A "
            "FROM adsorption"
        ).fetchone()
        assert row["probe"] == "H2"
        assert row["site_atom_idx"] == 1
        assert row["config_index"] == 0
        assert row["e_ads_kcal"] == pytest.approx(r.e_ads_kcal, abs=1e-3)
        assert row["e_bsse_kcal"] == pytest.approx(r.e_ads_bsse_kcal, abs=1e-3)
        assert row["delta_bond_A"] == pytest.approx(0.08, abs=1e-3)
        assert conn.execute("SELECT COUNT(*) FROM run_log WHERE agent='analysis'").fetchone()[0] == 1


def test_persist_links_the_result_to_the_configuration_geometry_created(tmp_path, triplet):
    """The v2 result row hangs off the configuration row, found by the shared filename stem."""
    from cadmia.db.session import init_db, get_connection
    from cadmia.db import repository
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2, bsse=True, cdft_json=_CDFT, config_index=0,
                cluster_label="TiO")

    db = tmp_path / "an2.db"
    conn = init_db(db)
    try:
        # What `geometry --db` would have written: the configuration, with its real orientation
        # and the path of the .xyz it produced. The adduct output shares that stem.
        stem = Path(r.adduct_path).stem
        atom_lines = [f"{s} {x:.8f} {y:.8f} {z:.8f}"
                      for s, (x, y, z) in zip(r.cluster_atoms, r.cluster_positions)]
        xyz_text = f"{len(atom_lines)}\nTiO\n" + "\n".join(atom_lines) + "\n"
        structure_id = hashlib.sha256(xyz_text.encode("utf-8")).hexdigest()
        repository.upsert_structure(conn, structure_id=structure_id, xyz_text=xyz_text,
                                    name="TiO", n_atoms=len(atom_lines))
        planned = repository.insert_adsorption_config(
            conn, structure_id=structure_id, probe="H2",
            site_atom_idx=r.site_atom_index, site_atom_symbol=r.site_atom_symbol,
            orientation="end_on_p0_a0",
            xyz_path=str(tmp_path / f"{stem}.xyz"),
        )
        ids = repository.persist_analysis_result(conn, r)
        conn.commit()
    finally:
        conn.close()

    assert ids["config_id"] == planned          # attached, not re-created
    with get_connection(db) as conn:
        cfg = conn.execute("SELECT * FROM adsorption_configs WHERE id = ?",
                           (ids["config_id"],)).fetchone()
        assert cfg["orientation"] == "end_on_p0_a0"      # the real identity survived
        res = conn.execute("SELECT * FROM adsorption_results WHERE id = ?",
                           (ids["result_id"],)).fetchone()
        assert res["config_id"] == ids["config_id"]
        assert res["e_ads_kcal"] == pytest.approx(r.e_ads_kcal, abs=1e-3)
        assert res["e_bsse_kcal"] == pytest.approx(r.e_ads_bsse_kcal, abs=1e-3)
        assert res["activated"] == 1


def test_persist_does_not_invent_a_configuration_when_there_is_none(tmp_path, triplet):
    """With no configuration row to attach to, the v2 result is skipped, not faked.

    Creating one here would mean calling insert_adsorption_config without an orientation, and
    its idempotent lookup then matches ANY adduct on the same (cluster, probe, site): the second
    configuration of a site would overwrite the energy of the first. That is the intrinsic-identity
    collision. The energetics still reach the `adsorption` row, so nothing is lost.
    """
    from cadmia.db.session import init_db, get_connection
    from cadmia.db import repository
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2, cdft_json=_CDFT, config_index=0, cluster_label="TiO")

    db = tmp_path / "an3.db"
    conn = init_db(db)
    try:
        ids = repository.persist_analysis_result(conn, r)
        conn.commit()
    finally:
        conn.close()

    assert ids["config_id"] is None
    assert ids["result_id"] is None
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM adsorption_results").fetchone()[0] == 0
        # The energetics are still recorded, and the log says why the v2 row is missing.
        assert conn.execute("SELECT COUNT(*) FROM adsorption WHERE id = ?",
                            (ids["adsorption_id"],)).fetchone()[0] == 1
        note = conn.execute(
            "SELECT message FROM run_log WHERE agent = 'analysis' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        assert "no configuration row matches" in note


# ─────────────────────────────────────────────
# CLI batch mode (--adduct-dir)
# ─────────────────────────────────────────────

def test_analyze_cli_batch(tmp_path, triplet):
    from click.testing import CliRunner
    from cadmia.cli import main
    cluster, h2, adduct = triplet
    addir = tmp_path / "adducts"; addir.mkdir()
    # two adduct outputs at different separations → two distinct analyses
    _gauss(addir / "c0.out", [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                              ("H", 0, 0, 1.9), ("H", 0, 0, 2.72)], -101.20)
    _gauss(addir / "c1.out", [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                              ("H", 0, 0, 2.1), ("H", 0, 0, 2.90)], -101.05)
    outdir = tmp_path / "analysis"
    res = CliRunner().invoke(main, [
        "analyze", "--adduct-dir", str(addir), "--cluster", str(cluster),
        "--probe-free", str(h2), "-o", str(outdir),
    ])
    assert res.exit_code == 0, res.output
    written = sorted(outdir.glob("analysis_*.json"))
    assert len(written) == 2
    assert "Analyzed 2 adduct(s)" in res.output


# ─────────────────────────────────────────────
# Adsorption thermochemistry (ΔG/ΔH from frequencies)
# ─────────────────────────────────────────────

def _state(energy, *, h=None, g=None, t=None, homo=None, lumo=None, dipole=None, provenance=None):
    import numpy as np
    from cadmia.core import ElectronicState, Structure
    s = Structure(atoms=["O"], positions=np.zeros((1, 3)), labels=["O1"],
                  structure_id="x", source_file="x")
    dip = np.asarray(dipole, dtype=float) if dipole is not None else None
    return ElectronicState(structure=s, energy_hartree=energy, enthalpy_hartree=h,
                           gibbs_free_energy_hartree=g, temperature_kelvin=t,
                           homo_ev=homo, lumo_ev=lumo, dipole_debye=dip, provenance=provenance)


def test_electronic_properties_cluster_and_adduct(monkeypatch):
    states = {"ad": _state(-101.20, homo=-7.8, lumo=-3.1),
              "cl": _state(-100.00, homo=-8.0, lumo=-3.0),
              "pr": _state(-1.17,   homo=-9.0, lumo=-1.0)}
    order = iter(["ad", "cl", "pr"])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (states[next(order)], "orca"))
    r = analysis_agent.analyze("ad", "cl", "pr", probe="O2")
    assert r.electronic_cluster["gap_ev"] == pytest.approx(5.0)
    assert r.electronic_adduct["gap_ev"] == pytest.approx(4.7)
    assert r.delta_gap_ev == pytest.approx(-0.3, abs=1e-6)
    assert "mu_ev" in r.electronic_cluster   # global descriptors filled


def test_dipole_captured_and_delta(monkeypatch):
    states = {"ad": _state(-101.20, dipole=[3.0, 4.0, 0.0]),   # |μ| = 5.0
              "cl": _state(-100.00, dipole=[0.0, 0.0, 2.0]),   # |μ| = 2.0
              "pr": _state(-1.17)}
    order = iter(["ad", "cl", "pr"])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (states[next(order)], "orca"))
    r = analysis_agent.analyze("ad", "cl", "pr", probe="O2")
    assert r.electronic_cluster["dipole_debye"] == pytest.approx(2.0)
    assert r.electronic_adduct["dipole_debye"] == pytest.approx(5.0)
    assert r.delta_dipole_debye == pytest.approx(3.0, abs=1e-6)


def _state_full(atoms, positions, energy, charges=None, scheme="hirshfeld"):
    import numpy as np
    from cadmia.core import ElectronicState, Structure
    s = Structure(atoms=list(atoms), positions=np.asarray(positions, dtype=float),
                  labels=[f"{a}{i}" for i, a in enumerate(atoms)],
                  structure_id="x", source_file="x")
    return ElectronicState(structure=s, energy_hartree=energy,
                           charges=charges, charge_scheme=(scheme if charges else None))


def test_charge_transfer_on_adsorption(monkeypatch):
    ad = _state_full(["Ti", "O", "H", "H"],
                     [[0, 0, 0], [1.8, 0, 0], [0, 0, 1.9], [0, 0, 2.72]],
                     -101.20, charges=[0.5, -0.6, -0.1, -0.2])
    cl = _state_full(["Ti", "O"], [[0, 0, 0], [1.8, 0, 0]], -100.0, charges=[0.4, -0.4])
    pr = _state_full(["H", "H"], [[0, 0, 0], [0.74, 0, 0]], -1.17, charges=[0.0, 0.0])
    order = iter([ad, cl, pr])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (next(order), "orca"))
    r = analyze("ad", "cl", "pr", probe="H2")
    assert r.charge_scheme == "hirshfeld"
    assert r.site_atom_symbol == "Ti"
    # probe block sum = -0.1 + -0.2 = -0.3 (cluster donates 0.3 e to the probe)
    assert r.q_transfer_to_probe == pytest.approx(-0.3, abs=1e-6)
    # site Ti: 0.5 (adduct) − 0.4 (cluster) = +0.1
    assert r.delta_q_site == pytest.approx(0.1, abs=1e-6)


def test_charge_transfer_skipped_when_schemes_differ(monkeypatch):
    ad = _state_full(["Ti", "O", "H", "H"],
                     [[0, 0, 0], [1.8, 0, 0], [0, 0, 1.9], [0, 0, 2.72]],
                     -101.20, charges=[0.5, -0.6, -0.1, -0.2], scheme="hirshfeld")
    cl = _state_full(["Ti", "O"], [[0, 0, 0], [1.8, 0, 0]], -100.0,
                     charges=[0.4, -0.4], scheme="mulliken")
    pr = _state_full(["H", "H"], [[0, 0, 0], [0.74, 0, 0]], -1.17, charges=[0.0, 0.0])
    order = iter([ad, cl, pr])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (next(order), "orca"))
    r = analyze("ad", "cl", "pr", probe="H2")
    assert r.q_transfer_to_probe is None
    assert any("different charge scheme" in w for w in r.warnings)


def _freq_ir_output() -> str:
    """A minimal ORCA freq output (3-atom, no imaginary modes, one strong IR band)."""
    return "\n".join([
        "-----------------------", "VIBRATIONAL FREQUENCIES", "-----------------------", "",
        "   0:         0.00 cm**-1", "   1:         0.00 cm**-1", "   2:         0.00 cm**-1",
        "   3:         0.00 cm**-1", "   4:         0.00 cm**-1", "   5:         0.00 cm**-1",
        "   6:       667.45 cm**-1", "   7:       667.45 cm**-1", "   8:      2396.32 cm**-1", "",
        "-----------", "IR SPECTRUM", "-----------", "",
        " Mode   freq       eps      Int      T**2         TX        TY        TZ",
        "        cm**-1   L/(mol*cm) km/mol    a.u.",
        "----------------------------------------------------------------------------",
        "   6:    667.45   0.001234     30.50  0.000123  ( 0.001  0.002  0.003)",
        "   7:    667.45   0.001234     30.50  0.000123  ( 0.001  0.002  0.003)",
        "   8:   2396.32   0.025678    600.20  0.002456  ( 0.010  0.000  0.000)", "",
        "The first frequency considered to be a vibration is 6",
    ]) + "\n"


def test_adduct_vibrational_signature(tmp_path, monkeypatch):
    adduct = tmp_path / "adduct.out"
    adduct.write_text(_freq_ir_output(), encoding="utf-8")
    ad = _state_full(["C", "O", "O"], [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]], -188.0)
    cl = _state_full(["C"], [[0, 0, 0]], -37.0)
    pr = _state_full(["O", "O"], [[0, 0, 0], [1.2, 0, 0]], -150.0)
    order = iter([ad, cl, pr])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (next(order), "orca"))
    r = analyze(str(adduct), "cl", "pr", probe="O2")
    assert r.adduct_imaginary_modes == 0
    assert r.adduct_ir_bands
    assert r.adduct_ir_bands[0]["frequency_cm1"] == pytest.approx(2396.3, abs=0.1)
    assert r.adduct_ir_bands[0]["intensity_km_mol"] == pytest.approx(600.2, abs=0.1)


def test_delta_g_h_ads_when_all_have_frequencies(monkeypatch):
    states = {"ad": _state(-101.20, h=-101.10, g=-101.18, t=298.15),
              "cl": _state(-100.00, h=-99.95, g=-100.01),
              "pr": _state(-1.17,   h=-1.10,  g=-1.15)}
    order = iter(["ad", "cl", "pr"])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (states[next(order)], "orca"))
    r = analysis_agent.analyze("ad", "cl", "pr", probe="O2")
    assert r.delta_h_ads_kcal == pytest.approx((-101.10 + 99.95 + 1.10) * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.delta_g_ads_kcal == pytest.approx((-101.18 + 100.01 + 1.15) * _HARTREE_TO_KCAL, abs=1e-3)
    assert r.minus_t_delta_s_kcal == pytest.approx(r.delta_g_ads_kcal - r.delta_h_ads_kcal, abs=1e-6)
    assert r.temperature_kelvin == pytest.approx(298.15)


def test_delta_g_warns_on_mixed_temperature(monkeypatch):
    """ΔG_ads combining Gibbs energies at different T must warn."""
    states = {"ad": _state(-101.20, g=-101.18, t=298.15),
              "cl": _state(-100.00, g=-100.01, t=298.15),
              "pr": _state(-1.17,   g=-1.15,   t=350.00)}   # probe thermo at a different T
    order = iter(["ad", "cl", "pr"])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (states[next(order)], "orca"))
    r = analysis_agent.analyze("ad", "cl", "pr", probe="O2")
    assert r.delta_g_ads_kcal is not None
    assert any("different temperatures" in w for w in r.warnings)


def test_delta_g_warns_when_probe_lacks_frequencies(monkeypatch):
    states = {"ad": _state(-101.20, h=-101.10, g=-101.18, t=298.15),
              "cl": _state(-100.00, h=-99.95, g=-100.01),
              "pr": _state(-1.17)}   # opt only → no thermo
    order = iter(["ad", "cl", "pr"])
    monkeypatch.setattr(analysis_agent, "_parse_output",
                        lambda p: (states[next(order)], "orca"))
    r = analysis_agent.analyze("ad", "cl", "pr", probe="O2")
    assert r.delta_g_ads_kcal is None and r.delta_h_ads_kcal is None
    assert r.e_ads_kcal is not None
    assert any("opt+freq" in w and "probe" in w for w in r.warnings)


def test_structural_coordination_change(triplet):
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2, cdft_json=_CDFT)
    # binding site coordination is computed for cluster and adduct
    assert r.site_cn_cluster is not None
    assert r.site_ecn_cluster is not None
    assert r.site_cn_adduct is not None
    # deltas are consistent
    assert r.delta_cn_site == r.site_cn_adduct - r.site_cn_cluster
    assert r.delta_ecn_site == pytest.approx(r.site_ecn_adduct - r.site_ecn_cluster, abs=1e-9)


def test_analyze_cli_requires_exactly_one_source(tmp_path, triplet):
    from click.testing import CliRunner
    from cadmia.cli import main
    cluster, h2, adduct = triplet
    # neither --adduct nor --adduct-dir
    res = CliRunner().invoke(main, [
        "analyze", "--cluster", str(cluster), "--probe-free", str(h2),
    ])
    assert res.exit_code != 0
    assert "exactly one" in res.output.lower()


# ─────────────────────────────────────────────
# CLI positional adducts (files, directory, glob)
# ─────────────────────────────────────────────

def _two_adducts(tmp_path):
    addir = tmp_path / "adducts"; addir.mkdir()
    _gauss(addir / "c0.out", [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                              ("H", 0, 0, 1.9), ("H", 0, 0, 2.72)], -101.20)
    _gauss(addir / "c1.out", [("Ti", 0, 0, 0), ("O", 1.8, 0, 0),
                              ("H", 0, 0, 2.1), ("H", 0, 0, 2.90)], -101.05)
    return addir


def test_analyze_cli_positional_files_and_directory_agree(tmp_path, triplet):
    """A shell glob (two files) and the directory holding them must analyze the same set."""
    from click.testing import CliRunner
    from cadmia.cli import main
    cluster, h2, _ = triplet
    addir = _two_adducts(tmp_path)
    common = ["--cluster", str(cluster), "--probe-free", str(h2)]

    by_glob = CliRunner().invoke(main, ["analyze", str(addir / "c0.out"), str(addir / "c1.out"),
                                        *common, "-o", str(tmp_path / "a")])
    by_dir = CliRunner().invoke(main, ["analyze", str(addir), *common, "-o", str(tmp_path / "b")])

    assert by_glob.exit_code == 0, by_glob.output
    assert by_dir.exit_code == 0, by_dir.output
    assert "Analyzed 2 adduct(s)" in by_glob.output and "Analyzed 2 adduct(s)" in by_dir.output
    assert (sorted(p.name for p in (tmp_path / "a").glob("*.json"))
            == sorted(p.name for p in (tmp_path / "b").glob("*.json")))


def test_analyze_cli_one_positional_file_is_single_mode(tmp_path, triplet):
    from click.testing import CliRunner
    from cadmia.cli import main
    cluster, h2, adduct = triplet
    res = CliRunner().invoke(main, ["analyze", str(adduct), "--cluster", str(cluster),
                                    "--probe-free", str(h2)])
    assert res.exit_code == 0, res.output
    assert "Analyzed" not in res.output          # the batch summary must not appear


def test_analyze_cli_rejects_positional_mixed_with_the_flags(tmp_path, triplet):
    from click.testing import CliRunner
    from cadmia.cli import main
    cluster, h2, adduct = triplet
    res = CliRunner().invoke(main, ["analyze", str(adduct), "--adduct", str(adduct),
                                    "--cluster", str(cluster), "--probe-free", str(h2)])
    assert res.exit_code != 0


# ── Cluster isomerism: is the substrate in the adduct the same isomer as on its own? ──

def _adduct_with_cluster(tmp_path, cluster_geom, name="adduct_iso.out"):
    """Adduct whose cluster block is *cluster_geom*, with H2 parked well away from it."""
    path = tmp_path / name
    _gauss(path, list(cluster_geom) + [("H", 0, 0, 1.9), ("H", 0, 0, 2.72)], -101.20)
    return path


def test_cluster_keeping_its_bonding_is_not_flagged(triplet):
    cluster, h2, adduct = triplet
    r = analyze(adduct, cluster, h2)
    assert r.cluster_rearranged is False
    assert r.cluster_bond_changes == {}
    assert not [w for w in r.warnings if "bonding changes" in w]


def test_broken_cluster_bond_is_flagged_with_the_bond_that_changed(tmp_path, triplet):
    cluster, h2, _ = triplet
    # Ti–O 1.8 → 3.0 Å: past 1.20 × (1.60 + 0.66) = 2.71 Å, so the bond is gone.
    r = analyze(_adduct_with_cluster(tmp_path, [("Ti", 0, 0, 0), ("O", 3.0, 0, 0)]), cluster, h2)
    assert r.cluster_rearranged is True
    assert r.cluster_bond_changes == {"O-Ti": -1}
    assert any("bonding changes on adsorption" in w and "O-Ti -1" in w for w in r.warnings)


def test_the_cluster_displacement_reaches_the_result(tmp_path, triplet):
    """The deformation is the observable, not only the yes/no verdict."""
    cluster, h2, _ = triplet
    # Ti–O stretched 1.8 → 2.2 Å: bonding unchanged, but the atoms did move.
    r = analyze(_adduct_with_cluster(tmp_path, [("Ti", 0, 0, 0), ("O", 2.2, 0, 0)]), cluster, h2)

    assert len(r.cluster_displacement_ang) == 2
    assert max(r.cluster_displacement_ang) == pytest.approx(0.2, abs=0.01)
    assert r.cluster_rearranged is False          # it deformed without rearranging


def test_a_rigid_motion_of_the_cluster_displaces_nothing(tmp_path, triplet):
    """Kabsch first: a translated and rotated cluster has not deformed."""
    cluster, h2, _ = triplet
    rotated = [("Ti", 5.0, -2.0, 1.0), ("O", 5.0, -2.0, 2.8)]      # same 1.8 Å Ti–O
    r = analyze(_adduct_with_cluster(tmp_path, rotated), cluster, h2)

    assert max(r.cluster_displacement_ang) == pytest.approx(0.0, abs=1e-3)


def test_a_cluster_merely_stretched_by_the_probe_is_not_flagged(tmp_path, triplet):
    """The whole point of using connectivity: 0.4 Å of deformation must not read as a change."""
    cluster, h2, _ = triplet
    r = analyze(_adduct_with_cluster(tmp_path, [("Ti", 0, 0, 0), ("O", 2.2, 0, 0)]), cluster, h2)
    assert r.cluster_rearranged is False
    assert not [w for w in r.warnings if "bonding changes" in w]


def test_verdict_survives_rotation_and_translation(tmp_path, triplet):
    cluster, h2, _ = triplet
    rotated = [("Ti", 5.0, -2.0, 1.0), ("O", 5.0, -2.0, 2.8)]      # same Ti–O of 1.8 Å
    r = analyze(_adduct_with_cluster(tmp_path, rotated), cluster, h2)
    assert r.cluster_rearranged is False


def test_a_distance_near_the_cutoff_makes_the_verdict_unsafe(tmp_path, triplet):
    """A verdict that would flip with a slightly different cutoff has to say so."""
    cluster, h2, _ = triplet
    # 2.75 Å sits within _SKELETON_MARGIN of the 2.71 Å cutoff.
    r = analyze(_adduct_with_cluster(tmp_path, [("Ti", 0, 0, 0), ("O", 2.75, 0, 0)]), cluster, h2)
    assert any("close to the bond/no-bond cutoff" in w for w in r.warnings)


# ─────────────────────────────────────────────
# Counterpoise
# ─────────────────────────────────────────────

def _cp_log(path: Path, complex_e: float, fragment_e: float, corrected: float | None) -> Path:
    """A counterpoise-shaped Gaussian log: complex first, fragments after."""
    lines = [
        " # B3LYP/6-31G* Counterpoise=2",
        "",
        " Counterpoise: doing DCBS calculation for fragment   1 NewBq=T",
        f" SCF Done:  E(RB3LYP) =  {complex_e:.9f}     A.U. after   10 cycles",
        f" SCF Done:  E(RB3LYP) =  {fragment_e:.9f}     A.U. after    8 cycles",
    ]
    if corrected is not None:
        lines.append(f" Counterpoise: corrected energy = {corrected:.9f}")
    lines.append(" Normal termination of Gaussian 16")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_counterpoise_energy_is_not_taken_from_the_last_fragment(tmp_path):
    """The last SCF Done in a counterpoise job belongs to a fragment, not to the complex."""
    from cadmia.io.electronic import _parse_gaussian_energy

    log = _cp_log(tmp_path / "cp.log", complex_e=-100.5, fragment_e=-50.1, corrected=-100.4)
    lines = log.read_text().splitlines()
    assert _parse_gaussian_energy(lines) == pytest.approx(-100.4)   # not -50.1


def test_counterpoise_without_a_corrected_line_warns(tmp_path):
    """No corrected line means the value cannot be trusted, and that is said out loud."""
    from cadmia.io.electronic import _parse_gaussian_energy

    log = _cp_log(tmp_path / "cp2.log", complex_e=-100.5, fragment_e=-50.1, corrected=None)
    lines = log.read_text().splitlines()
    with pytest.warns(RuntimeWarning, match="counterpoise"):
        _parse_gaussian_energy(lines)


def test_a_plain_log_is_untouched_by_the_counterpoise_path(tmp_path):
    from cadmia.io.electronic import (_parse_gaussian_energy,
                                      gaussian_energy_is_counterpoise_corrected)

    log = tmp_path / "plain.log"
    log.write_text(
        " # B3LYP/6-31G*\n\n"
        " SCF Done:  E(RB3LYP) =  -76.400000000     A.U. after   10 cycles\n"
        " SCF Done:  E(RB3LYP) =  -76.410000000     A.U. after   12 cycles\n",
        encoding="utf-8",
    )
    lines = log.read_text().splitlines()
    assert _parse_gaussian_energy(lines) == pytest.approx(-76.41)   # last one still wins
    assert gaussian_energy_is_counterpoise_corrected(lines) is False


def test_gaussian_counterpoise_log():
    """Gaussian runs the complex FIRST and then each fragment, so the last SCF Done belongs to a
    fragment. That is why the corrected-energy line is what gets read.
    """
    from cadmia.agents.analysis_agent import _parse_bsse_hartree
    from cadmia.io.electronic import (_parse_gaussian_energy, _has_counterpoise,
                                      gaussian_energy_is_counterpoise_corrected)

    log = Path(__file__).parents[1] / "fixtures" / "gaussian_cp" / "counterpoise_synthetic.log"
    lines = log.read_text(errors="replace").splitlines()

    assert _has_counterpoise(lines)
    assert gaussian_energy_is_counterpoise_corrected(lines)
    assert _parse_gaussian_energy(lines) == pytest.approx(-1001.408680229182, abs=1e-9)
    assert _parse_bsse_hartree(log) == pytest.approx(0.001604618594, abs=1e-12)
    # The trap: the last SCF Done in the file is a fragment, an order of magnitude away.
    assert _parse_gaussian_energy(lines) < -1000.0
