"""Tests for cadmia.agents.geometry_agent."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cadmia.agents import geometry_agent
from cadmia.agents.geometry_agent import (
    AdsorptionConfig,
    GeometryResult,
    _BUILTIN_PROBES,
    _co_c_down,
    _h2o_o_down,
    _h2o_h_down,
    _h2_eta1,
    _h2_eta2,
    _co2_eta1_c,
    _co2_o_down,
    _co2_side_on,
    _co2_bidentate,
    _n2_eta1,
    _nh3_n_down,
    _nh3_tripod,
    _o2_eta1,
    _clash_ratio,
    _place_probe,
    _read_xyz,
    _write_xyz,
    available_probes,
    run,
)

# ─────────────────────────────────────────────
# Probe geometry correctness
# ─────────────────────────────────────────────

def test_h2_eta1_bond_length():
    pg = _h2_eta1()
    coords = pg.coords
    h1 = np.array(coords[0][1:])
    h2 = np.array(coords[1][1:])
    assert abs(np.linalg.norm(h2 - h1) - 0.741) < 1e-6


def test_h2_eta2_bond_length():
    pg = _h2_eta2()
    coords = pg.coords
    h1 = np.array(coords[0][1:])
    h2 = np.array(coords[1][1:])
    assert abs(np.linalg.norm(h2 - h1) - 0.741) < 1e-6


def test_h2_eta2_midpoint_at_origin():
    pg = _h2_eta2()
    h1 = np.array(pg.coords[0][1:])
    h2 = np.array(pg.coords[1][1:])
    midpoint = (h1 + h2) / 2
    assert np.allclose(midpoint, 0.0, atol=1e-10)


def test_co2_linear():
    pg = _co2_side_on()
    c  = np.array(pg.coords[0][1:])
    o1 = np.array(pg.coords[1][1:])
    o2 = np.array(pg.coords[2][1:])
    # C at origin
    assert np.allclose(c, 0.0)
    # O–C–O angle = 180°
    v1 = (o1 - c) / np.linalg.norm(o1 - c)
    v2 = (o2 - c) / np.linalg.norm(o2 - c)
    angle = math.degrees(math.acos(np.dot(v1, v2)))
    assert abs(angle - 180.0) < 1e-4


def test_co2_bond_lengths():
    pg = _co2_side_on()
    c  = np.array(pg.coords[0][1:])
    for atom in pg.coords[1:]:
        o = np.array(atom[1:])
        assert abs(np.linalg.norm(o - c) - 1.160) < 1e-6


@pytest.mark.parametrize("builder", [_nh3_n_down, _nh3_tripod])
def test_nh3_bond_lengths(builder):
    pg = builder()
    n = np.array(pg.coords[0][1:])
    for atom in pg.coords[1:]:
        h = np.array(atom[1:])
        assert abs(np.linalg.norm(h - n) - 1.012) < 1e-5


@pytest.mark.parametrize("builder", [_nh3_n_down, _nh3_tripod])
def test_nh3_hnh_angle(builder):
    pg = builder()
    n  = np.array(pg.coords[0][1:])
    hs = [np.array(a[1:]) for a in pg.coords[1:]]
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            v1 = (hs[i] - n) / np.linalg.norm(hs[i] - n)
            v2 = (hs[j] - n) / np.linalg.norm(hs[j] - n)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2))))))
            assert abs(angle - 106.7) < 0.5, f"H–N–H angle = {angle:.2f}°"


@pytest.mark.parametrize("builder", [_nh3_n_down, _nh3_tripod])
def test_nh3_n_at_origin(builder):
    pg = builder()
    n = np.array(pg.coords[0][1:])
    assert np.allclose(n, 0.0)


def test_nh3_n_down_h_point_away():
    """n_down: N lone pair toward the surface, so H atoms point away (+z)."""
    pg = _nh3_n_down()
    h_z = [pg.coords[i][3] for i in range(1, 4)]
    assert all(z > 0 for z in h_z)


def test_nh3_tripod_h_point_toward_surface():
    """tripod: the three H point toward the surface (−z)."""
    pg = _nh3_tripod()
    h_z = [pg.coords[i][3] for i in range(1, 4)]
    assert all(z < 0 for z in h_z)


def test_co2_all_modes_bond_lengths():
    """Every built-in CO2 mode keeps both C–O bonds at 1.160 Å."""
    for builder in (_co2_eta1_c, _co2_o_down, _co2_side_on, _co2_bidentate):
        pg = builder()
        c = np.array(next(a[1:] for a in pg.coords if a[0] == "C"))
        for atom in (a for a in pg.coords if a[0] == "O"):
            o = np.array(atom[1:])
            # in o_down the far O is 2 bonds from C; check nearest-neighbour topology instead
            d = np.linalg.norm(o - c)
            assert abs(d - 1.160) < 1e-6 or abs(d - 2 * 1.160) < 1e-6


def test_co2_bidentate_is_bent():
    """bidentate CO2 is bent (O–C–O = 127°, not linear) with both O toward −z."""
    pg = _co2_bidentate()
    c  = np.array(pg.coords[0][1:])
    o1 = np.array(pg.coords[1][1:])
    o2 = np.array(pg.coords[2][1:])
    v1 = (o1 - c) / np.linalg.norm(o1 - c)
    v2 = (o2 - c) / np.linalg.norm(o2 - c)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2))))))
    assert abs(angle - 127.0) < 0.5
    assert o1[2] < 0 and o2[2] < 0


def test_co2_eta1_c_binds_through_carbon_with_both_oxygens_away():
    """eta1-C CO2 is bent, C nearest the surface, both O on the far side.

    The mode this replaced was linear along the approach axis, so one O ended up between
    the carbon and the site — closer to the cluster than the atom that was supposed to
    bind. Nothing in the probe may sit on the -z side of C.
    """
    pg = _co2_eta1_c()
    c  = np.array(pg.coords[0][1:])
    o1 = np.array(pg.coords[1][1:])
    o2 = np.array(pg.coords[2][1:])
    assert np.allclose(c, 0.0)
    assert o1[2] > 0 and o2[2] > 0
    v1 = (o1 - c) / np.linalg.norm(o1 - c)
    v2 = (o2 - c) / np.linalg.norm(o2 - c)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2))))))
    assert abs(angle - 127.0) < 0.5


def test_no_co2_mode_puts_a_non_binding_atom_ahead_of_the_binding_atom():
    """In every CO2 mode, no atom sits deeper along the approach direction than the binder.

    -z is what points at the cluster, so an atom below the binding atom reaches the site
    first and the placement distance no longer means what it says.
    """
    for pg in _BUILTIN_PROBES["CO2"]:
        binder_z = next(a[3] for a in pg.coords if a[0] == pg.binding_atom)
        if pg.orientation == "bidentate":
            continue   # by design the two O lead: it is the O,O-bound carbonate precursor
        assert all(a[3] >= binder_z - 1e-9 for a in pg.coords), pg.orientation


def test_co2_side_on_is_horizontal():
    """side_on CO2 lies in the xy-plane (all z = 0)."""
    pg = _co2_side_on()
    assert all(abs(a[3]) < 1e-12 for a in pg.coords)


def test_builtin_probe_mode_counts():
    assert len(_BUILTIN_PROBES["CO2"]) == 4
    assert len(_BUILTIN_PROBES["NH3"]) == 2
    assert {pg.orientation for pg in _BUILTIN_PROBES["NH3"]} == {"n_down", "tripod"}
    assert {pg.orientation for pg in _BUILTIN_PROBES["CO2"]} == {
        "eta1_c", "o_down", "side_on", "bidentate",
    }


# ─────────────────────────────────────────────
# _place_probe
# ─────────────────────────────────────────────

# Angles are measured in the site's local frame. These unit tests pin the placement geometry
# itself, so they feed the cartesian frame explicitly: with it, "polar=0" is +z again and the
# expected values below stay the ones you can work out by hand.
_Z_FRAME = (np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]))

def test_place_probe_top_binding_distance():
    """Binding atom is at exactly *dist* from site when polar=0."""
    pg = _h2_eta1()
    site = np.array([0.0, 0.0, 0.0])
    dist = 2.0
    placed = _place_probe(site, pg, dist, polar_deg=0.0, azimuthal_deg=0.0, frame=_Z_FRAME)
    binding_atom_pos = placed[0][1]
    assert abs(float(np.linalg.norm(binding_atom_pos - site)) - dist) < 1e-10


def test_place_probe_top_approach_direction():
    """Binding atom is directly above (+z) when polar=0."""
    pg = _h2_eta1()
    site = np.array([1.0, 2.0, 3.0])
    dist = 2.0
    placed = _place_probe(site, pg, dist, polar_deg=0.0, azimuthal_deg=0.0, frame=_Z_FRAME)
    binding_pos = placed[0][1]
    delta = binding_pos - site
    expected = np.array([0.0, 0.0, dist])
    assert np.allclose(delta, expected, atol=1e-10)


def test_place_probe_preserves_internal_bond():
    """The H–H bond length in the placed probe matches the template."""
    pg = _h2_eta1()
    site = np.array([0.0, 0.0, 0.0])
    placed = _place_probe(site, pg, 2.0, polar_deg=45.0, azimuthal_deg=90.0, frame=_Z_FRAME)
    h1 = placed[0][1]
    h2 = placed[1][1]
    assert abs(float(np.linalg.norm(h2 - h1)) - 0.741) < 1e-8


def test_place_probe_eta2_bond_perpendicular_to_approach():
    """For η² H2 at polar=0, H–H bond should be in the xy-plane (perpendicular to approach)."""
    pg = _h2_eta2()
    site = np.array([0.0, 0.0, 0.0])
    placed = _place_probe(site, pg, 2.0, polar_deg=0.0, azimuthal_deg=0.0, frame=_Z_FRAME)
    h1 = placed[0][1]
    h2 = placed[1][1]
    bond = h2 - h1
    # Bond vector should be perpendicular to approach (+z), so z-component ≈ 0
    assert abs(float(bond[2])) < 1e-8


# ─────────────────────────────────────────────
# XYZ I/O
# ─────────────────────────────────────────────

def test_write_read_xyz_roundtrip(tmp_path):
    symbols = ["Ti", "O", "O"]
    coords = [np.array([0.0, 0.0, 0.0]), np.array([1.95, 0.0, 0.0]), np.array([-1.95, 0.0, 0.0])]
    p = tmp_path / "test.xyz"
    _write_xyz(p, symbols, coords, comment="test molecule")
    s2, c2, comment = _read_xyz(p)
    assert s2 == symbols
    for orig, read in zip(coords, c2):
        assert np.allclose(orig, read, atol=1e-7)
    assert "test molecule" in comment


# ─────────────────────────────────────────────
# Full run()
# ─────────────────────────────────────────────

@pytest.fixture
def simple_cluster_xyz(tmp_path) -> Path:
    """Minimal 3-atom TiO2 cluster in a temp .xyz file."""
    content = "3\nTi2O cluster\nTi  0.00000000  0.00000000  0.00000000\nO   1.95000000  0.00000000  0.00000000\nO  -1.95000000  0.00000000  0.00000000\n"
    p = tmp_path / "ti2o.xyz"
    p.write_text(content)
    return p


@pytest.fixture
def simple_cdft_json(tmp_path) -> Path:
    """Minimal CDFTResult-like JSON with two sites."""
    # atom_index is 1-based (as emitted by the CDFT Agent): 1=Ti, 2=first O.
    data = {
        "cluster": "ti2o",
        "sites": [
            {"atom_index": 1, "atom_symbol": "Ti", "fukui_plus": 0.45, "fukui_minus": 0.10, "delta_f": 0.35, "rank": 1, "character": "electrophilic"},
            {"atom_index": 2, "atom_symbol": "O",  "fukui_plus": 0.10, "fukui_minus": 0.40, "delta_f": -0.30, "rank": 2, "character": "nucleophilic"},
        ],
        "global_descriptors": {},
        "warnings": [],
    }
    p = tmp_path / "cdft.json"
    p.write_text(json.dumps(data))
    return p


def test_run_returns_geometry_result(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=2, n_configs=2)
    assert isinstance(result, GeometryResult)
    assert result.probe == "H2"
    assert result.n_sites == 2


def test_run_generates_configs_for_all_sites(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "NH3", n_sites=2, n_configs=4, polar_angles=[0.0, 30.0])
    # n_sites=2, NH3 has 2 modes (n_down, tripod), polar=[0,30]. Both NH3 modes are pyramidal,
    # so polar=0 no longer collapses: there the azimuth is the molecule's own roll about the
    # normal, and whether the three H face the lattice oxygens or sit between them is a real
    # degree of freedom. 4 azimuths × 2 polars × 2 modes × 2 sites = 32.
    assert len(result.configs) == 32


def test_run_h2_config_count(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=1, n_configs=3, polar_angles=[0.0, 45.0])
    # H2 has 2 modes. eta1 is end-on, so its roll changes nothing and polar=0 still collapses to
    # one: 1 + 3 = 4. eta2 is side-on, so polar=0 samples the roll too: 3 + 3 = 6. Total 10.
    assert len(result.configs) == 10


def test_run_writes_xyz_files(tmp_path, simple_cluster_xyz, simple_cdft_json):
    out = tmp_path / "configs"
    result = run(
        simple_cdft_json, simple_cluster_xyz, "NH3",
        n_sites=1, n_configs=2, polar_angles=[0.0],
        output_dir=out,
    )
    assert out.exists()
    # polar=0 → 1 config × 1 mode × 1 site = 1 file
    written = list(out.glob("*.xyz"))
    assert len(written) == len([c for c in result.configs if c.xyz_path is not None])


def test_run_xyz_has_correct_atom_count(tmp_path, simple_cluster_xyz, simple_cdft_json):
    out = tmp_path / "configs"
    run(simple_cdft_json, simple_cluster_xyz, "NH3", n_sites=1, n_configs=1, polar_angles=[0.0], output_dir=out)
    xyz_files = list(out.glob("*.xyz"))
    assert xyz_files
    syms, _, _ = _read_xyz(xyz_files[0])
    # cluster (3) + NH3 (4) = 7
    assert len(syms) == 7


def test_run_placement_distance_in_range(simple_cluster_xyz, simple_cdft_json):
    """Placement distance for Ti site + H binding must be in chemical range."""
    result = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=1, n_configs=1, polar_angles=[0.0])
    dists = {c.placement_dist for c in result.configs}
    for d in dists:
        # Ti cov radius ~1.60, H cov radius ~0.31, +0.5 → ~2.41; plausible
        assert 1.5 < d < 4.0, f"Unexpected distance {d:.3f} Å"


def test_run_places_probe_on_correct_atom_1based(tmp_path, simple_cluster_xyz, simple_cdft_json):
    """Regression: atom_index=1 (Ti at origin) must place the probe over the origin,
    not over cl_coords[1] (the first O at x=1.95). Guards the 1-based/0-based contract.

    CO2 has several placement modes; in every one the atom at the probe's local origin
    (written as atom 4, index 3) sits directly above the site at polar=0.
    """
    out = tmp_path / "configs"
    run(simple_cdft_json, simple_cluster_xyz, "CO2",
        n_sites=1, n_configs=1, polar_angles=[0.0], output_dir=out)
    xyz_files = list(out.glob("*.xyz"))
    assert len(xyz_files) == len(_BUILTIN_PROBES["CO2"])
    for f in xyz_files:
        syms, coords, _ = _read_xyz(f)
        # First 3 atoms are the cluster; atom 4 (index 3) is the probe's binding atom.
        binding = coords[3]
        ti_origin = coords[0]
        first_o   = coords[1]
        # Binding atom must be over Ti (x≈y≈0, z>0), and far closer to Ti than to the O.
        assert abs(binding[0]) < 1e-6 and abs(binding[1]) < 1e-6, (
            f"probe placed at {binding} — expected above Ti at the origin (off-by-one regression)"
        )
        assert binding[2] > 0
        assert np.linalg.norm(binding - ti_origin) < np.linalg.norm(binding - first_o)


def test_run_warns_on_out_of_range_site(tmp_path, simple_cluster_xyz):
    cdft_data = {
        "cluster": "test",
        "sites": [{"atom_index": 99, "atom_symbol": "Ti", "delta_f": 0.5, "rank": 1, "character": "electrophilic"}],
        "global_descriptors": {},
        "warnings": [],
    }
    p = tmp_path / "bad_cdft.json"
    p.write_text(json.dumps(cdft_data))
    result = run(p, simple_cluster_xyz, "H2", n_sites=1)
    assert result.configs == []
    assert any("out of range" in w for w in result.warnings)


def test_run_invalid_probe_raises(simple_cluster_xyz, simple_cdft_json):
    with pytest.raises(ValueError, match="Unknown probe"):
        run(simple_cdft_json, simple_cluster_xyz, "CH4")


def test_run_no_output_dir_no_files(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "CO2", n_sites=1, n_configs=1, polar_angles=[0.0])
    for c in result.configs:
        assert c.xyz_path is None


def test_run_from_dict(simple_cluster_xyz, simple_cdft_json):
    data = json.loads(simple_cdft_json.read_text())
    result = run(data, simple_cluster_xyz, "CO2", n_sites=1, n_configs=1, polar_angles=[0.0])
    # CO2 has 4 built-in modes → 1 config per mode at polar=0
    assert len(result.configs) == len(_BUILTIN_PROBES["CO2"])


def test_to_json_is_valid(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "NH3", n_sites=1, n_configs=1, polar_angles=[0.0])
    parsed = json.loads(result.to_json())
    assert parsed["probe"] == "NH3"
    assert "configs" in parsed
    assert len(parsed["configs"]) == len(result.configs)


def test_run_cluster_label_override(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "H2", cluster_label="my_cluster",
                 n_sites=1, n_configs=1, polar_angles=[0.0])
    assert result.cluster == "my_cluster"


# ─────────────────────────────────────────────
# New built-in probes (CO, N2, O2, H2O)
# ─────────────────────────────────────────────

def test_builtin_probe_catalog():
    # At least the project's three plus the new ones must be available.
    for name in ("H2", "CO2", "NH3", "CO", "N2", "O2", "H2O"):
        assert name in _BUILTIN_PROBES


def test_co_bond_length_and_binding():
    pg = _co_c_down()
    assert pg.binding_atom == "C"
    syms = [a[0] for a in pg.coords]
    assert syms == ["C", "O"]
    d = math.dist(pg.coords[0][1:], pg.coords[1][1:])
    assert d == pytest.approx(1.128, abs=1e-6)


def test_n2_and_o2_diatomic_lengths():
    assert math.dist(_n2_eta1().coords[0][1:], _n2_eta1().coords[1][1:]) == pytest.approx(1.098, abs=1e-6)
    assert math.dist(_o2_eta1().coords[0][1:], _o2_eta1().coords[1][1:]) == pytest.approx(1.208, abs=1e-6)


@pytest.mark.parametrize("builder", [_h2o_o_down, _h2o_h_down])
def test_h2o_geometry(builder):
    pg = builder()
    assert pg.binding_atom == "O"
    o, h1, h2 = pg.coords
    assert math.dist(o[1:], h1[1:]) == pytest.approx(0.9572, abs=1e-6)
    assert math.dist(o[1:], h2[1:]) == pytest.approx(0.9572, abs=1e-6)
    # H–O–H angle from the two O–H vectors.
    v1 = np.array(h1[1:]) - np.array(o[1:])
    v2 = np.array(h2[1:]) - np.array(o[1:])
    ang = math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    assert ang == pytest.approx(104.5, abs=1e-3)


def test_h2o_mode_orientations():
    # o_down: O lone pair toward the surface → H point away (+z).
    o_down = _h2o_o_down()
    assert all(o_down.coords[i][3] > 0 for i in (1, 2))
    # h_down: H toward the surface (−z).
    h_down = _h2o_h_down()
    assert all(h_down.coords[i][3] < 0 for i in (1, 2))
    assert len(_BUILTIN_PROBES["H2O"]) == 2


def test_run_with_new_probe(simple_cluster_xyz, simple_cdft_json):
    result = run(simple_cdft_json, simple_cluster_xyz, "CO", n_sites=1, n_configs=1, polar_angles=[0.0])
    # CO has two modes (c_down, o_down) → two configs at polar=0.
    assert len(result.configs) == 2
    assert result.probe == "CO"


# ─────────────────────────────────────────────
# User-defined probes via probes.yaml
# ─────────────────────────────────────────────

def test_available_probes_includes_user_yaml(tmp_path):
    yaml_path = tmp_path / "probes.yaml"
    yaml_path.write_text(
        "probes:\n"
        "  CH4:\n"
        "    binding_atom: C\n"
        "    modes:\n"
        "      - orientation: c_down\n"
        "        atoms:\n"
        "          - [C, 0.0, 0.0, 0.0]\n"
        "          - [H, 0.0, 0.0, 1.09]\n"
    )
    names = available_probes(yaml_path)
    assert "CH4" in names
    assert "NH3" in names  # built-ins still present


def test_run_with_user_probe(simple_cluster_xyz, simple_cdft_json, tmp_path):
    yaml_path = tmp_path / "probes.yaml"
    yaml_path.write_text(
        "probes:\n"
        "  XE:\n"
        "    modes:\n"
        "      - orientation: atop\n"
        "        atoms:\n"
        "          - [Xe, 0.0, 0.0, 0.0]\n"
    )
    result = run(simple_cdft_json, simple_cluster_xyz, "XE",
                 n_sites=1, n_configs=1, polar_angles=[0.0], probes_yaml=yaml_path)
    assert result.probe == "XE"
    assert len(result.configs) == 1


def test_user_probe_overrides_builtin(simple_cluster_xyz, simple_cdft_json, tmp_path):
    yaml_path = tmp_path / "probes.yaml"
    yaml_path.write_text(
        "probes:\n"
        "  H2:\n"
        "    modes:\n"
        "      - orientation: custom\n"
        "        atoms:\n"
        "          - [H, 0.0, 0.0, 0.0]\n"
        "          - [H, 0.0, 0.0, 0.80]\n"
    )
    result = run(simple_cdft_json, simple_cluster_xyz, "H2",
                 n_sites=1, n_configs=1, polar_angles=[0.0], probes_yaml=yaml_path)
    # Override has a single mode named "custom" (built-in has two: eta1/eta2).
    assert len(result.configs) == 1
    assert "custom" in result.configs[0].orientation


def test_unknown_probe_lists_available(simple_cluster_xyz, simple_cdft_json):
    with pytest.raises(ValueError, match="Unknown probe"):
        run(simple_cdft_json, simple_cluster_xyz, "ZZZ", n_sites=1)


def test_malformed_probe_yaml_raises(simple_cluster_xyz, simple_cdft_json, tmp_path):
    yaml_path = tmp_path / "probes.yaml"
    yaml_path.write_text("probes:\n  BAD:\n    modes: []\n")
    with pytest.raises(ValueError, match="non-empty 'modes'"):
        run(simple_cdft_json, simple_cluster_xyz, "BAD", n_sites=1, probes_yaml=yaml_path)


# ─────────────────────────────────────────────
# Local frame
# ─────────────────────────────────────────────

from cadmia.agents.geometry_agent import (_has_clash, _local_frame, _rodrigues,
                                          _site_frame, _site_normal)


def _tetra():
    """A central atom with four neighbours: symmetric inside, exposed outside."""
    d = 1.9
    symbols = ["Ti", "O", "O", "O", "O"]
    coords = [np.array([0.0, 0.0, 0.0]),
              np.array([ d,  d,  d]) / np.sqrt(3),
              np.array([ d, -d, -d]) / np.sqrt(3),
              np.array([-d,  d, -d]) / np.sqrt(3),
              np.array([-d, -d,  d]) / np.sqrt(3)]
    return symbols, coords


def test_normal_points_away_from_the_neighbours():
    # Two neighbours on -x: the normal must lean toward +x, whatever +z says.
    symbols = ["Ti", "O", "O"]
    coords = [np.array([0.0, 0.0, 0.0]),
              np.array([-1.9, 0.0, 0.5]),
              np.array([-1.9, 0.0, -0.5])]
    n, source = _site_normal(symbols, coords, 0)
    assert source == "neighbours"
    assert n[0] > 0.9
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-10


def test_centrosymmetric_site_falls_back_to_the_centroid():
    symbols, coords = _tetra()
    n, source = _site_normal(symbols, coords, 0)
    # The four neighbour vectors cancel; the centroid is the site itself, so nothing is left.
    assert source in ("centroid", "fallback_z")
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-10


def test_local_frame_falls_back_when_the_reference_is_parallel_to_the_normal():
    n = np.array([0.0, 0.0, 1.0])
    t1, t2 = _local_frame(n, reference=np.array([0.0, 0.0, 5.0]))
    assert abs(float(np.dot(t1, n))) < 1e-10
    assert abs(float(np.linalg.norm(t1)) - 1.0) < 1e-10


def test_local_frame_is_orthonormal_and_never_degenerate():
    for v in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
              np.array([1.0, 1.0, 1.0]) / np.sqrt(3)):
        t1, t2 = _local_frame(v)
        assert abs(float(np.dot(t1, v))) < 1e-10
        assert abs(float(np.dot(t2, v))) < 1e-10
        assert abs(float(np.dot(t1, t2))) < 1e-10
        assert abs(float(np.linalg.norm(t1)) - 1.0) < 1e-10
        assert abs(float(np.linalg.norm(t2)) - 1.0) < 1e-10


def test_normal_rotates_with_the_cluster():
    """The whole point: the normal is a property of the geometry, not of the file's axes."""
    symbols = ["Ti", "O", "O"]
    coords = [np.array([0.0, 0.0, 0.0]),
              np.array([-1.9, 0.0, 0.5]),
              np.array([-1.9, 0.0, -0.5])]
    n0, _ = _site_normal(symbols, coords, 0)

    R = _rodrigues(np.array([0.3, -0.7, 0.5]), 1.1)
    rotated = [R @ c for c in coords]
    n1, _ = _site_normal(symbols, rotated, 0)

    assert np.allclose(R @ n0, n1, atol=1e-10)


@pytest.mark.parametrize("probe_geom", [
    _h2_eta1,        # axial: was already invariant
    _h2_eta2,        # the rest have internal structure to orient, and were NOT
    _co2_side_on,
    _co2_eta1_c,
    _nh3_n_down,
    _nh3_tripod,
])
def test_placement_is_invariant_under_rigid_rotation_of_the_input(probe_geom):
    """Rotate the cluster, regenerate, and the configurations must follow the rotation.

    This is what the file's +z could never give: with a global frame the probe lands on a
    different face of the cluster depending on how the .xyz happened to be oriented — and
    so did its own orientation about the approach axis. Parametrized over
    non-axial probes on purpose: an end-on diatomic is the one case that cannot detect it.
    """
    symbols = ["Ti", "O", "O"]
    coords = [np.array([0.0, 0.0, 0.0]),
              np.array([-1.9, 0.0, 0.5]),
              np.array([-1.9, 0.0, -0.5])]
    pg = probe_geom()

    R = _rodrigues(np.array([0.2, 0.9, -0.4]), 0.8)
    rotated = [R @ c for c in coords]

    f0, _ = _site_frame(symbols, coords, 0)
    f1, _ = _site_frame(symbols, rotated, 0)
    for polar, azimuth in ((0.0, 0.0), (30.0, 90.0), (60.0, 270.0)):
        a = _place_probe(coords[0], pg, 2.0, polar, azimuth, frame=f0)
        b = _place_probe(rotated[0], pg, 2.0, polar, azimuth, frame=f1)
        for (_, pa), (_, pb) in zip(a, b):
            assert np.allclose(R @ pa, pb, atol=1e-8)


def test_polar_zero_places_the_probe_outside_the_cluster():
    """polar=0 means 'off the surface', so it can never overlap the cluster."""
    symbols = ["Ti", "O", "O"]
    coords = [np.array([0.0, 0.0, 0.0]),
              np.array([-1.9, 0.0, 0.5]),
              np.array([-1.9, 0.0, -0.5])]
    frame, _ = _site_frame(symbols, coords, 0)
    placed = _place_probe(coords[0], _h2_eta1(), 2.0, 0.0, 0.0, frame=frame)
    assert not _has_clash(symbols, coords, placed)


def test_clash_filter_catches_an_overlapping_probe():
    symbols = ["Ti"]
    coords = [np.array([0.0, 0.0, 0.0])]
    assert _has_clash(symbols, coords, [("H", np.array([0.1, 0.0, 0.0]))])
    assert not _has_clash(symbols, coords, [("H", np.array([2.5, 0.0, 0.0]))])


# ─────────────────────────────────────────────
# Starting-geometry clash floor and push-out (A1)
# ─────────────────────────────────────────────

def _worst_contact(path: Path, n_cluster: int) -> float:
    """Closest probe–cluster contact in a written config, in summed-covalent-radii units."""
    symbols, coords, _ = _read_xyz(path)
    placed = list(zip(symbols[n_cluster:], coords[n_cluster:]))
    return _clash_ratio(symbols[:n_cluster], coords[:n_cluster], placed)


def test_clash_floor_is_stricter_than_the_optimized_geometry_criterion():
    """0.65 x (r_i + r_j) let an H sit 0.69 A from an oxygen; the starting floor is 0.80."""
    symbols, coords = ["O"], [np.array([0.0, 0.0, 0.0])]
    r_sum = 0.31 + 0.66     # Cordero H + O
    assert _has_clash(symbols, coords, [("H", np.array([0.70 * r_sum, 0.0, 0.0]))])
    assert _has_clash(symbols, coords, [("H", np.array([0.79 * r_sum, 0.0, 0.0]))])
    assert not _has_clash(symbols, coords, [("H", np.array([0.85 * r_sum, 0.0, 0.0]))])


@pytest.mark.parametrize("probe", ["H2", "CO2", "NH3", "H2O", "O2", "CO", "N2"])
def test_no_written_config_sits_below_the_clash_floor(
    tmp_path, simple_cluster_xyz, simple_cdft_json, probe
):
    """Whatever survives to disk must be a geometry worth an xTB relaxation."""
    out = tmp_path / probe
    result = run(simple_cdft_json, simple_cluster_xyz, probe, n_sites=2, output_dir=out)
    assert result.configs
    for c in result.configs:
        assert _worst_contact(c.xyz_path, 3) >= 0.80, f"{c.xyz_path.name} overlaps the cluster"


def test_an_overlapping_orientation_is_pushed_out_instead_of_dropped(
    tmp_path, simple_cluster_xyz, simple_cdft_json
):
    """CO2's bent and linear modes lean a non-binding O toward the site.

    That is a placement artefact, not a dead direction: the sampling angle is fine, only the
    distance was measured to the binding atom alone. Pushing the probe out keeps it.
    """
    out = tmp_path / "co2"
    result = run(simple_cdft_json, simple_cluster_xyz, "CO2", n_sites=2, output_dir=out)
    pushed = [c for c in result.configs if c.pushed_out > 0]
    assert pushed, "expected at least one CO2 orientation to need pushing out"
    assert result.n_pushed_out == len(pushed)
    assert result.n_rejected_clash == 0
    for c in pushed:
        assert c.placement_dist > 0
        assert _worst_contact(c.xyz_path, 3) >= 0.80
    assert any("moved outward" in w for w in result.warnings)


def test_a_placement_with_room_is_not_pushed(simple_cluster_xyz, simple_cdft_json):
    """The fix must not perturb the configurations that were already fine."""
    result = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=2)
    assert result.n_pushed_out == 0
    assert all(c.pushed_out == 0.0 for c in result.configs)


def test_an_orientation_with_no_room_is_dropped_and_reported(
    monkeypatch, simple_cluster_xyz, simple_cdft_json
):
    """With nowhere to push to, an overlapping orientation is still dropped — and said so.

    Forcing the budget to zero is the cheapest way to reach that branch: a site genuinely
    walled in on every side needs a cluster far larger than this fixture.
    """
    monkeypatch.setattr(geometry_agent, "_PUSHOUT_MAX", 0.0)
    result = run(simple_cdft_json, simple_cluster_xyz, "CO2", n_sites=2)
    assert result.n_rejected_clash > 0
    assert result.n_pushed_out == 0
    assert any("no room" in w for w in result.warnings)


def test_pushed_out_distance_is_recorded_in_the_sidecar(
    tmp_path, simple_cluster_xyz, simple_cdft_json
):
    """How far a probe was moved is provenance: it has to survive into the record."""
    result = run(simple_cdft_json, simple_cluster_xyz, "CO2", n_sites=2,
                 output_dir=tmp_path / "co2")
    data = json.loads(result.to_json())
    assert data["n_pushed_out"] == result.n_pushed_out
    assert [c for c in data["configs"] if c["pushed_out_A"] > 0]

    # Subtracting the push-out has to give back the distance the site/mode started from,
    # the same for every orientation of that pair — otherwise the record does not say
    # what was actually added.
    nominal = {}
    for c in data["configs"]:
        key = (c["site_atom_idx"], c["orientation"].split("_p")[0])
        base = c["placement_dist_A"] - c["pushed_out_A"]
        nominal.setdefault(key, base)
        assert base == pytest.approx(nominal[key], abs=1e-9)


# ─────────────────────────────────────────────
# The cluster reader validates its input (A3)
# ─────────────────────────────────────────────

def test_read_xyz_rejects_a_header_that_lies_about_the_atom_count(tmp_path):
    """A truncated cluster used to seed the whole sampling, silently."""
    p = tmp_path / "truncated.xyz"
    p.write_text(
        "8\ntruncated\nTi 0.0 0.0 0.0\nO 1.6 0.0 0.0\nO 0.0 1.6 0.0\n"
    )
    with pytest.raises(ValueError, match="Expected 8 atoms"):
        _read_xyz(p)


def test_read_xyz_rejects_an_element_that_does_not_exist(tmp_path):
    """An unknown symbol reached covalent_radius() as None and fell back to 1.5 A."""
    p = tmp_path / "bad.xyz"
    p.write_text("2\nx\nTi 0.0 0.0 0.0\nXx 1.6 0.0 0.0\n")
    with pytest.raises(ValueError, match="invalid element symbol"):
        _read_xyz(p)


def test_a_cdft_json_without_sites_says_so(tmp_path):
    """No 'sites' key: zero configurations, and a warning that says why.

    The silent version of this returned n_sites=0, configs=[] and warnings=[] — a result
    that looks like a normal run of a cluster with nothing reactive on it.
    """
    xyz = tmp_path / "c.xyz"
    xyz.write_text("2\ntest\nTi 0.0 0.0 0.0\nO 1.8 0.0 0.0\n", encoding="utf-8")
    res = run({"descriptors": {}}, xyz, probe="H2", output_dir=tmp_path / "out")
    assert res.configs == []
    assert any("no 'sites' list" in w for w in res.warnings)


def test_the_azimuth_origin_survives_a_tie_between_equidistant_neighbours():
    """A symmetric site has two neighbours at the same distance; comparing raw floats lets a
    1e-16 difference — which a rigid rotation is enough to flip — pick the azimuth origin."""
    import numpy as np

    from cadmia.agents.geometry_agent import _site_frame

    symbols = ["Ti", "O", "O", "Al"]
    coords = [np.array([0.0, 0.0, 0.0]), np.array([1.95, 0.0, 0.0]),
              np.array([-1.95, 0.0, 0.0]), np.array([0.0, 1.90, 0.0])]
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])

    (n0, t1_0, t2_0), _ = _site_frame(symbols, coords, 0)
    (n1, t1_1, t2_1), _ = _site_frame(symbols, [R @ c for c in coords], 0)

    # the whole frame must be carried along by the rotation, tie or no tie
    assert np.allclose(R @ n0, n1, atol=1e-9)
    assert np.allclose(R @ t1_0, t1_1, atol=1e-9)
    assert np.allclose(R @ t2_0, t2_1, atol=1e-9)


def test_a_site_whose_symbol_disagrees_with_the_xyz_is_skipped(tmp_path):
    """generating anyway puts every configuration on the wrong atom while the
    filename and the JSON still claim the right one — and the orchestrator drops warnings."""
    cluster = tmp_path / "c.xyz"
    cluster.write_text("2\n\nTi 0.0 0.0 0.0\nO 1.8 0.0 0.0\n", encoding="utf-8")
    cdft = tmp_path / "cdft.json"
    cdft.write_text(json.dumps({"sites": [
        {"atom_index": 1, "atom_symbol": "O", "rank": 1},   # the .xyz says Ti
    ]}), encoding="utf-8")

    res = geometry_agent.run(cdft_json=str(cdft), cluster_xyz=str(cluster), probe="H2",
                             n_sites=1, n_configs=1, output_dir=str(tmp_path / "out"))

    assert res.configs == []
    assert any("does not match the SCF output" in w for w in res.warnings)


def test_the_top_sites_are_taken_by_rank_not_by_list_position(tmp_path):
    """`repository.py` reorders sites by atom_index when persisting, so a JSON that
    came back from the database would otherwise sample the first ATOMS, not the top sites."""
    cluster = tmp_path / "c.xyz"
    cluster.write_text("3\n\nTi 0.0 0.0 0.0\nO 1.8 0.0 0.0\nAl 0.0 1.9 0.0\n", encoding="utf-8")
    cdft = tmp_path / "cdft.json"
    # stored in atom order; the most reactive site is atom 3
    cdft.write_text(json.dumps({"sites": [
        {"atom_index": 1, "atom_symbol": "Ti", "rank": 3},
        {"atom_index": 2, "atom_symbol": "O", "rank": 2},
        {"atom_index": 3, "atom_symbol": "Al", "rank": 1},
    ]}), encoding="utf-8")

    res = geometry_agent.run(cdft_json=str(cdft), cluster_xyz=str(cluster), probe="H2",
                             n_sites=1, n_configs=1, output_dir=str(tmp_path / "out"))

    assert {c.site_atom_idx for c in res.configs} == {3}


def test_polar_zero_collapses_only_for_an_axial_probe(simple_cluster_xyz, simple_cdft_json):
    """At polar=0 the azimuth is the probe's own roll, not an approach direction."""
    import numpy as np

    axial = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=1, n_configs=4,
                polar_angles=[0.0])
    end_on = [c for c in axial.configs if c.orientation.startswith("eta1")]
    side_on = [c for c in axial.configs if c.orientation.startswith("eta2")]

    assert len(end_on) == 1        # rolling a diatomic about its own axis changes nothing
    assert len(side_on) == 4       # rolling a side-on one does


def test_no_sites_limit_means_every_site(simple_cluster_xyz, simple_cdft_json):
    """The default no longer narrows to the extremes of the dual descriptor."""
    everything = run(simple_cdft_json, simple_cluster_xyz, "H2", n_configs=1,
                     polar_angles=[45.0])
    narrowed = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=1, n_configs=1,
                   polar_angles=[45.0])

    n_available = len(json.loads(Path(simple_cdft_json).read_text())["sites"])
    assert len({c.site_atom_idx for c in everything.configs}) == n_available
    assert len({c.site_atom_idx for c in narrowed.configs}) == 1


# ── Regression tests: geometry edge cases ──────────────────────────────

def test_metal_metal_pairs_do_not_tilt_the_normal_of_a_metal_site():
    """With Cordero radii and the 1.30 factor, Ti···Ti counts as a neighbour out to 4.16 Å,
    while non-bonded M···M in an oxide sit at 2.8–3.5 Å."""
    import numpy as np

    from cadmia.agents.geometry_agent import _site_normal

    symbols = ["Ti", "O", "Ti"]
    coords = [np.array([0.0, 0.0, 0.0]),        # the site
              np.array([0.0, 1.9, 0.0]),        # a real Ti–O bond
              np.array([3.2, 0.0, 0.0])]        # a non-bonded Ti at 3.2 Å

    normal, source = _site_normal(symbols, coords, 0)
    assert source == "neighbours"
    # only the oxygen votes, so the normal is exactly opposite to it
    assert np.allclose(normal, [0.0, -1.0, 0.0], atol=1e-9)


def test_hydrogen_starts_further_from_an_oxygen_than_from_a_metal():
    """O···H at r_O + r_H + 0.5 is 1.47 Å — a proton-abstraction geometry, not a contact."""
    from cadmia.utils.chemistry import covalent_radius

    r_o, r_h = covalent_radius("O"), covalent_radius("H")
    assert r_o + r_h + 0.5 < 1.6            # the old distance, inside transfer range
    assert r_o + r_h + 1.0 > 1.9            # the new one, a hydrogen-bond distance


def test_two_polar_angles_that_round_the_same_get_different_names(tmp_path):
    """`int()` truncation made --polar 20 and --polar 20.5 write to the same file, and since
    the intrinsic identity, the orientation string is part of the database identity as well."""
    cluster = tmp_path / "c.xyz"
    cluster.write_text("2\n\nTi 0.0 0.0 0.0\nO 1.8 0.0 0.0\n", encoding="utf-8")
    cdft = tmp_path / "cdft.json"
    cdft.write_text(json.dumps({"sites": [{"atom_index": 1, "atom_symbol": "Ti", "rank": 1}]}),
                    encoding="utf-8")

    res = geometry_agent.run(cdft_json=str(cdft), cluster_xyz=str(cluster), probe="H2",
                             n_sites=1, n_configs=1, polar_angles=[20.0, 20.5],
                             output_dir=str(tmp_path / "out"))

    names = {Path(c.xyz_path).name for c in res.configs}
    assert len(names) == len(res.configs), names
    assert len({c.orientation for c in res.configs}) == len(res.configs)


def test_a_user_probe_with_an_unknown_element_is_refused(tmp_path):
    """`covalent_radius` returns None and the clash filter silently falls back to 1.5 Å."""
    import pytest
    import yaml

    from cadmia.agents.geometry_agent import _load_user_probes

    bad = tmp_path / "probes.yaml"
    bad.write_text(yaml.safe_dump({
        "XX": {"binding_atom": "Xx",
               "modes": [{"orientation": "a", "atoms": [["Xx", 0.0, 0.0, 0.0]]}]},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="not a known element"):
        _load_user_probes(bad)


def test_a_second_run_into_the_same_directory_reports_the_leftovers(
        tmp_path, simple_cluster_xyz, simple_cdft_json):
    """The concrete way a sidecar stops describing its directory.

    `geometry` does not clean the output directory, so a second run with fewer sites overwrites
    the names it produces and leaves the rest of the first run in place. geometry.json then
    lists what this run wrote while `prescreen` globs the directory and reads more than that,
    under a record carrying this run's command and commit.
    """
    import json as _json

    out = tmp_path / "configs"
    out.mkdir()
    (out / "leftover_from_an_earlier_run.xyz").write_text(
        "1\nold\nH 0.0 0.0 0.0\n", encoding="utf-8")

    result = run(simple_cdft_json, simple_cluster_xyz, "H2", n_sites=1, n_configs=1,
                 output_dir=out, polar_angles=[0.0])

    assert any("left over from an earlier run" in w for w in result.warnings)
    assert any("leftover_from_an_earlier_run.xyz" in w for w in result.warnings)

    # And the sidecar lists only what this run wrote, with a fingerprint over exactly those.
    data = _json.loads(Path(result.output_json).read_text(encoding="utf-8"))
    listed = {Path(p).name for p in data["artifact_paths"]}
    assert "leftover_from_an_earlier_run.xyz" not in listed
    assert data["provenance"]["artifacts"]["n"] == len(listed)
