"""Geometry Agent — adsorption starting configurations guided by CDFT site ranking."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cadmia.core.sidecar import write_json_sidecar
from cadmia.io.xyz import parse_xyz
from cadmia.utils.chemistry import covalent_radius

# ─────────────────────────────────────────────
# Probe geometry definitions (binding atom at origin, molecule along +z)
# ─────────────────────────────────────────────

_H2_BOND = 0.741    # Å, experimental
_CO2_CO  = 1.160    # Å, C=O in free CO2
_CO2_BENT_OCO = 127.0  # degrees, O–C–O in the activated (bent, ~CO2^δ−) carbonate precursor
_NH3_NH  = 1.012    # Å, N–H in free NH3
_NH3_HNH = 106.7    # degrees, H–N–H angle
_CO_BOND = 1.128    # Å, C≡O in free CO
_N2_BOND = 1.098    # Å, N≡N in free N2
_O2_BOND = 1.208    # Å, O=O in free O2 (triplet ground state)
_H2O_OH  = 0.9572   # Å, O–H in free H2O
_H2O_HOH = 104.5    # degrees, H–O–H angle


@dataclass(frozen=True)
class _ProbeGeom:
    """Internal geometry of a probe mode.

    coords: (symbol, x, y, z) tuples with the binding point at origin.
    binding_atom: element symbol whose covalent radius sets the M–X distance.
    orientation: human-readable label (eta1_c, eta2, o_down, upright, …).
    """
    name:         str
    binding_atom: str
    coords:       Tuple[Tuple[str, float, float, float], ...]
    orientation:  str


def _h2_eta1() -> _ProbeGeom:
    """H2 end-on (η¹): binding H at origin, second H at +z."""
    return _ProbeGeom(
        "H2", "H",
        (("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, _H2_BOND)),
        "eta1",
    )


def _h2_eta2() -> _ProbeGeom:
    """H2 side-on (η²): H–H midpoint at origin, bond along x."""
    half = _H2_BOND / 2
    return _ProbeGeom(
        "H2", "H",
        (("H", -half, 0.0, 0.0), ("H", half, 0.0, 0.0)),
        "eta2",
    )


def _co2_bent(z_sign: float, orientation: str) -> _ProbeGeom:
    """Bent CO2 (activated, O–C–O = 127°): C at origin, both O toward *z_sign*·z (±)."""
    half = math.radians(_CO2_BENT_OCO / 2)
    x = _CO2_CO * math.sin(half)
    z = _CO2_CO * math.cos(half) * z_sign
    return _ProbeGeom(
        "CO2", "C",
        (("C", 0.0, 0.0, 0.0), ("O", x, 0.0, z), ("O", -x, 0.0, z)),
        orientation,
    )


def _co2_eta1_c() -> _ProbeGeom:
    """CO2 bound through carbon (eta1-C): C toward the surface, both O bent away (+z).

    A *linear* CO2 cannot approach end-on through C — the carbon sits in the middle, so
    one O always reaches the site first. The C-bound mode is the bent, activated CO2
    anion, and that is what this builds. A linear CO2 along the approach axis would leave
    one O about 1.7 A from the site — a Ti-O bond formed before the calculation decided
    anything.
    """
    return _co2_bent(+1.0, "eta1_c")


def _co2_o_down() -> _ProbeGeom:
    """CO2 end-on through one O (O-down): O at origin, linear O–C–O along +z (away)."""
    return _ProbeGeom(
        "CO2", "O",
        (("O", 0.0, 0.0, 0.0), ("C", 0.0, 0.0, _CO2_CO), ("O", 0.0, 0.0, 2 * _CO2_CO)),
        "o_down",
    )


def _co2_side_on() -> _ProbeGeom:
    """CO2 lying parallel to the surface (η²): C at origin, linear O–C–O along x."""
    return _ProbeGeom(
        "CO2", "C",
        (("C", 0.0, 0.0, 0.0), ("O", _CO2_CO, 0.0, 0.0), ("O", -_CO2_CO, 0.0, 0.0)),
        "side_on",
    )


def _co2_bidentate() -> _ProbeGeom:
    """CO2 bent toward the surface (carbonate precursor): C up at origin, both O toward -z."""
    return _co2_bent(-1.0, "bidentate")


def _nh3_geom(z_sign: float, orientation: str) -> _ProbeGeom:
    """NH3 pyramid: N at origin, the three N–H bonds along *z_sign*·z (±)."""
    r = _NH3_NH
    alpha = math.radians(_NH3_HNH)
    # Angle β each N–H bond makes with the pyramid axis: cos(α) = (3cos²β − 1)/2
    cos_beta = math.sqrt((2 * math.cos(alpha) + 1) / 3)
    sin_beta = math.sqrt(max(0.0, 1.0 - cos_beta ** 2))
    sq3 = math.sqrt(3)
    zc = z_sign * r * cos_beta
    h1 = ("H",  r * sin_beta,           0.0,                     zc)
    h2 = ("H", -r * sin_beta * 0.5,  r * sin_beta * sq3 / 2,  zc)
    h3 = ("H", -r * sin_beta * 0.5, -r * sin_beta * sq3 / 2,  zc)
    return _ProbeGeom("NH3", "N", (("N", 0.0, 0.0, 0.0), h1, h2, h3), orientation)


def _nh3_n_down() -> _ProbeGeom:
    """NH3 molecular adsorption: N lone pair toward the surface, H atoms pointing away (+z).

    This is the primary Lewis base → Lewis acid (e.g. undercoordinated metal) mode.
    """
    return _nh3_geom(+1.0, "n_down")


def _nh3_tripod() -> _ProbeGeom:
    """NH3 tripod: the three H point toward the surface (−z), N above.

    Relevant for H-bonding to basic surface oxygens rather than N-donation.
    """
    return _nh3_geom(-1.0, "tripod")


def _co_c_down() -> _ProbeGeom:
    """CO bound through carbon (η¹ C-down): C at origin, O at +z."""
    return _ProbeGeom("CO", "C", (("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, _CO_BOND)), "c_down")


def _co_o_down() -> _ProbeGeom:
    """CO bound through oxygen (η¹ O-down): O at origin, C at +z."""
    return _ProbeGeom("CO", "O", (("O", 0.0, 0.0, 0.0), ("C", 0.0, 0.0, _CO_BOND)), "o_down")


def _n2_eta1() -> _ProbeGeom:
    """N2 end-on (η¹): binding N at origin, second N at +z."""
    return _ProbeGeom("N2", "N", (("N", 0.0, 0.0, 0.0), ("N", 0.0, 0.0, _N2_BOND)), "eta1")


def _n2_eta2() -> _ProbeGeom:
    """N2 side-on (η²): N–N midpoint at origin, bond along x."""
    half = _N2_BOND / 2
    return _ProbeGeom("N2", "N", (("N", -half, 0.0, 0.0), ("N", half, 0.0, 0.0)), "eta2")


def _o2_eta1() -> _ProbeGeom:
    """O2 end-on (η¹): binding O at origin, second O at +z (triplet)."""
    return _ProbeGeom("O2", "O", (("O", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, _O2_BOND)), "eta1")


def _o2_eta2() -> _ProbeGeom:
    """O2 side-on (η²): O–O midpoint at origin, bond along x (triplet)."""
    half = _O2_BOND / 2
    return _ProbeGeom("O2", "O", (("O", -half, 0.0, 0.0), ("O", half, 0.0, 0.0)), "eta2")


def _h2o_geom(z_sign: float, orientation: str) -> _ProbeGeom:
    """H2O with the O at origin and the two O–H bonds along *z_sign*·z (±)."""
    r = _H2O_OH
    theta = math.radians(_H2O_HOH / 2)
    zc = z_sign * r * math.cos(theta)
    h1 = ("H",  r * math.sin(theta), 0.0, zc)
    h2 = ("H", -r * math.sin(theta), 0.0, zc)
    return _ProbeGeom("H2O", "O", (("O", 0.0, 0.0, 0.0), h1, h2), orientation)


def _h2o_o_down() -> _ProbeGeom:
    """H2O molecular adsorption: O lone pair toward the surface, both H pointing away (+z)."""
    return _h2o_geom(+1.0, "o_down")


def _h2o_h_down() -> _ProbeGeom:
    """H2O H-bond donor: both H toward the surface (−z), O above (H-bonds to basic oxygen)."""
    return _h2o_geom(-1.0, "h_down")


_BUILTIN_PROBES: Dict[str, List[_ProbeGeom]] = {
    "H2":  [_h2_eta1(), _h2_eta2()],
    "CO2": [_co2_eta1_c(), _co2_o_down(), _co2_side_on(), _co2_bidentate()],
    "NH3": [_nh3_n_down(), _nh3_tripod()],
    "CO":  [_co_c_down(), _co_o_down()],
    "N2":  [_n2_eta1(), _n2_eta2()],
    "O2":  [_o2_eta1(), _o2_eta2()],
    "H2O": [_h2o_o_down(), _h2o_h_down()],
}

DEFAULT_PROBES_YAML = Path.home() / ".cadmia" / "probes.yaml"


def _probe_from_spec(name: str, spec: dict) -> List[_ProbeGeom]:
    """Build probe modes from a user YAML spec.

    spec = {binding_atom?: str, modes: [{orientation?, binding_atom?, atoms: [[sym,x,y,z], …]}]}
    The binding point (origin) is wherever the atoms place it; binding_atom only
    selects the covalent radius used for the initial M–X distance.
    """
    modes = spec.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"Probe '{name}' in probes.yaml must define a non-empty 'modes' list.")
    out: List[_ProbeGeom] = []
    for i, mode in enumerate(modes):
        atoms = mode.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            raise ValueError(f"Probe '{name}' mode {i} must define a non-empty 'atoms' list.")
        coords: List[Tuple[str, float, float, float]] = []
        for a in atoms:
            if len(a) != 4:
                raise ValueError(
                    f"Probe '{name}' mode {i}: each atom must be [symbol, x, y, z], got {a!r}."
                )
            # A misspelled symbol is not harmless: `covalent_radius` returns None for it and the
            # clash filter silently falls back to 1.5 A, so the probe is placed by a radius that
            # belongs to no element. The cluster reader already refuses unknown symbols; user
            # probes went through unchecked.
            if covalent_radius(str(a[0])) is None:
                raise ValueError(
                    f"Probe '{name}' mode {i}: '{a[0]}' is not a known element symbol.")
        coords = tuple((str(a[0]), float(a[1]), float(a[2]), float(a[3])) for a in atoms)
        binding = mode.get("binding_atom") or spec.get("binding_atom") or coords[0][0]
        orientation = str(mode.get("orientation", f"mode{i}"))
        out.append(_ProbeGeom(name, str(binding), coords, orientation))
    return out


def _load_user_probes(probes_yaml: "str | Path | None" = None) -> Dict[str, List[_ProbeGeom]]:
    """Read user-defined probes from probes.yaml (top-level mapping or under 'probes')."""
    path = Path(probes_yaml) if probes_yaml else DEFAULT_PROBES_YAML
    if not path.exists():
        return {}
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid probes file (expected a mapping): {path}")
    probes = data.get("probes", data)
    result: Dict[str, List[_ProbeGeom]] = {}
    for name, spec in probes.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Probe '{name}' in {path} must be a mapping.")
        result[str(name).upper()] = _probe_from_spec(str(name).upper(), spec)
    return result


def _resolve_registry(probes_yaml: "str | Path | None" = None) -> Dict[str, List[_ProbeGeom]]:
    """Built-in probes overlaid with user-defined ones (user overrides on name clash)."""
    registry = dict(_BUILTIN_PROBES)
    registry.update(_load_user_probes(probes_yaml))
    return registry


def available_probes(probes_yaml: "str | Path | None" = None) -> List[str]:
    """Sorted names of all usable probes (built-in + user). Best-effort: never raises."""
    registry = dict(_BUILTIN_PROBES)
    try:
        registry.update(_load_user_probes(probes_yaml))
    except Exception:
        pass
    return sorted(registry)


# ─────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────

@dataclass
class AdsorptionConfig:
    """Single adsorption starting geometry."""
    site_atom_idx:    int
    site_atom_symbol: str
    config_index:     int
    orientation:      str
    probe:            str
    approach_angle:   float   # polar angle from +z, degrees
    azimuthal_angle:  float   # rotation around z, degrees
    placement_dist:   float   # M–X distance used, Å (after any push-out)
    pushed_out:       float = 0.0   # Å added to clear an overlap (0 = placed as computed)
    xyz_path:         Optional[Path] = None


@dataclass
class GeometryResult:
    """Output of the Geometry Agent."""
    cluster:    str
    probe:      str
    n_sites:    int
    configs:    List[AdsorptionConfig] = field(default_factory=list)
    warnings:   List[str]             = field(default_factory=list)
    n_rejected_clash: int             = 0   # dropped: no room for the probe at all
    n_pushed_out:     int             = 0   # moved outward to clear an overlap
    output_dir: Optional[Path]        = None
    output_json: Optional[str]        = None   # sidecar written inside output_dir

    def to_json(self) -> str:
        return json.dumps(
            {
                "cluster":    self.cluster,
                "probe":      self.probe,
                "n_sites":    self.n_sites,
                "n_configs":  len(self.configs),
                "n_rejected_clash": self.n_rejected_clash,
                "n_pushed_out": self.n_pushed_out,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "output_json": self.output_json,
                "warnings":   self.warnings,
                "configs": [
                    {
                        "site_atom_idx":       c.site_atom_idx,
                        "site_atom_symbol":    c.site_atom_symbol,
                        "config_index":        c.config_index,
                        "orientation":         c.orientation,
                        "probe":               c.probe,
                        "approach_angle_deg":  c.approach_angle,
                        "azimuthal_angle_deg": c.azimuthal_angle,
                        "placement_dist_A":    c.placement_dist,
                        "pushed_out_A":        c.pushed_out,
                        "xyz_path":            str(c.xyz_path) if c.xyz_path else None,
                    }
                    for c in self.configs
                ],
            },
            indent=2,
        )


# ─────────────────────────────────────────────
# XYZ I/O
# ─────────────────────────────────────────────

def _read_xyz(path: Path) -> Tuple[List[str], List[np.ndarray], str]:
    """Cluster geometry from an .xyz, through the project's validating parser.

    Reading it with a private loop here is what let a file whose header lies about its atom
    count seed the whole sampling on a truncated cluster without a word: a header of 8 over
    5 atom lines simply produced a 5-atom cluster. parse_xyz refuses that, and refuses an
    element symbol the periodic table does not have — which otherwise reached
    covalent_radius() as None and silently fell back to a 1.5 A radius.
    """
    struct = parse_xyz(path)
    return (
        list(struct.atoms),
        [np.asarray(row, dtype=float) for row in struct.positions],
        str(struct.metadata.get("comment", "")),
    )


def _write_xyz(
    path: Path,
    symbols: List[str],
    coords: List[np.ndarray],
    comment: str = "",
) -> None:
    lines = [str(len(symbols)), comment]
    for sym, c in zip(symbols, coords):
        lines.append(f"{sym:<4} {float(c[0]):14.8f} {float(c[1]):14.8f} {float(c[2]):14.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────
# Coordinate geometry
# ─────────────────────────────────────────────

def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix around *axis* by *angle* radians (Rodrigues formula)."""
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c + ux * ux * (1 - c),       ux * uy * (1 - c) - uz * s,  ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s,  c + uy * uy * (1 - c),       uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s,  uz * uy * (1 - c) + ux * s,  c + uz * uz * (1 - c)      ],
    ])


def _z_to_approach(approach: np.ndarray) -> np.ndarray:
    """Rotation matrix that aligns +z with *approach* unit vector."""
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, approach)
    norm = float(np.linalg.norm(cross))
    if norm < 1e-10:
        return np.eye(3) if approach[2] > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / norm
    angle = math.acos(max(-1.0, min(1.0, float(np.dot(z, approach)))))
    return _rodrigues(axis, angle)


# A neighbour is anything within this multiple of the summed covalent radii. Wider than a
# strict bond so a stretched or under-coordinated contact still counts toward the normal.
_NEIGHBOR_FACTOR = 1.30


# Elements that bridge a cluster: a bond between two of the others is not what defines the
# coordination gap of a metal site.
_NONMETALS = frozenset({"O", "N", "C", "S", "F", "Cl", "Br", "I", "H", "P", "Se"})


def _site_normal(
    symbols: List[str],
    coords: List[np.ndarray],
    idx: int,
) -> Tuple[np.ndarray, str]:
    """Outward normal at atom *idx*: the direction with the least coordination.

    Summing the unit vectors that point *from each neighbour to the site* gives a resultant
    that leans toward whichever side of the atom has no neighbours — that is, toward the
    vacuum. It is the cheapest local normal that needs no surface reconstruction, and it is
    invariant under rigid rotations of the input, which the +z of the .xyz file is not.

    Returns (unit normal, source) where *source* names which rule produced it, so the caller
    can warn when the geometry forced a fallback.
    """
    site = coords[idx]
    resultant = np.zeros(3)
    for j, other in enumerate(coords):
        if j == idx:
            continue
        # Metal–metal pairs are excluded from the sum. With Cordero radii and the 1.30 factor,
        # Ti···Ti counts as a neighbour out to 4.16 A, and in an oxide the non-bonded M···M
        # distances are 2.8–3.5 A: every one of them would enter the resultant and tilt the
        # normal of a metal site away from its actual coordination gap. Oxygens are unaffected
        # (O+O reaches only 1.72 A).
        if symbols[idx] not in _NONMETALS and symbols[j] not in _NONMETALS:
            continue
        d = float(np.linalg.norm(site - other))
        r_sum = (covalent_radius(symbols[idx]) or 1.5) + (covalent_radius(symbols[j]) or 1.5)
        if d < _NEIGHBOR_FACTOR * r_sum and d > 1e-8:
            resultant += (site - other) / d

    norm = float(np.linalg.norm(resultant))
    if norm > 1e-6:
        return resultant / norm, "neighbours"

    # Centrosymmetric environment (an interior atom): the neighbour vectors cancel and carry
    # no directional information. Point away from the cluster instead.
    centroid = np.mean(np.asarray(coords, dtype=float), axis=0)
    outward = site - centroid
    norm = float(np.linalg.norm(outward))
    if norm > 1e-6:
        return outward / norm, "centroid"

    return np.array([0.0, 0.0, 1.0]), "fallback_z"


def _local_frame(
    normal: np.ndarray,
    reference: "Optional[np.ndarray]" = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to *normal*.

    The azimuth needs an origin, and that origin must come from the geometry: seeding it
    from a cartesian axis would tie the azimuth back to the file's orientation, which is the
    very coupling this frame exists to break. *reference* is therefore a vector from the
    site's own surroundings — it is projected onto the plane and normalized. Only when it is
    absent or parallel to the normal does the cartesian seed take over, as a last resort for
    a frame that would otherwise be undefined.
    """
    t1 = None
    if reference is not None:
        projected = reference - np.dot(reference, normal) * normal
        if float(np.linalg.norm(projected)) > 1e-6:
            t1 = projected / np.linalg.norm(projected)
    if t1 is None:
        seed = np.zeros(3)
        seed[int(np.argmin(np.abs(normal)))] = 1.0
        t1 = seed - np.dot(seed, normal) * normal
        t1 /= np.linalg.norm(t1)
    return t1, np.cross(normal, t1)


def _site_frame(
    symbols: List[str],
    coords: List[np.ndarray],
    idx: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], str]:
    """The local frame ``(normal, t1, t2)`` at atom *idx*, plus how the normal was obtained.

    The azimuth origin is the site's nearest neighbour, so the whole frame is carried along
    by any rigid rotation of the input geometry.
    """
    normal, source = _site_normal(symbols, coords, idx)
    site = coords[idx]
    # Distances are rounded before the tie-break: a symmetric site has two neighbours at the
    # SAME distance, and comparing raw floats lets a 1e-16 difference — which a rigid rotation
    # of the input is enough to flip — pick the azimuth origin. Rounding makes the tie fall to
    # the lower atom index, which no rotation can change.
    others = sorted((round(float(np.linalg.norm(site - c)), 6), i)
                    for i, c in enumerate(coords) if i != idx)
    # The NEAREST neighbour is not always usable: when it lies along the normal its projection
    # onto the azimuth plane vanishes and `_local_frame` falls back to a cartesian seed, which
    # ties the azimuth back to the file's orientation — exactly the coupling this frame exists
    # to break. Walk outwards to the first neighbour that spans the plane.
    reference = None
    for _, j in others:
        candidate = coords[j] - site
        projected = candidate - np.dot(candidate, normal) * normal
        if float(np.linalg.norm(projected)) > 1e-6:
            reference = candidate
            break
    return (normal, *_local_frame(normal, reference)), source


# Closest admissible probe–cluster contact in a STARTING geometry, as a multiple of the
# summed covalent radii. Deliberately stricter than the 0.65 of `bond_distance_range`,
# which exists to spot fused atoms in a geometry that is already *optimized*: 0.65 admits an
# H about 0.7 A from a cluster oxygen — shorter than the real O–H bond, 0.96 — and that is not
# sampling adsorption, it is seeding a reaction.
_CLASH_FACTOR = 0.80

# The placement distance is measured to the binding atom alone, so the rest of the probe
# can dip toward the cluster on its own — worst in the bent and side-on CO2 modes, where
# an oxygen ends up closer to the site than the carbon it binds through. Sliding the whole
# probe back out along its own approach recovers those configurations instead of throwing
# them away, which is what simply raising the rejection floor to 0.80 would do.
_PUSHOUT_STEP = 0.05    # A per step
_PUSHOUT_MAX  = 2.00    # A; past this the site has no room for this orientation at all


def _clash_ratio(
    cluster_symbols: List[str],
    cluster_coords: List[np.ndarray],
    placed: List[Tuple[str, np.ndarray]],
) -> float:
    """Closest probe–cluster contact, as a multiple of the summed covalent radii."""
    return min(
        (
            float(np.linalg.norm(p_pos - c_pos))
            / ((covalent_radius(p_sym) or 1.5) + (covalent_radius(c_sym) or 1.5))
            for p_sym, p_pos in placed
            for c_sym, c_pos in zip(cluster_symbols, cluster_coords)
        ),
        default=float("inf"),
    )


def _has_clash(
    cluster_symbols: List[str],
    cluster_coords: List[np.ndarray],
    placed: List[Tuple[str, np.ndarray]],
) -> bool:
    """True if any probe–cluster pair sits below the starting-geometry clash floor."""
    return _clash_ratio(cluster_symbols, cluster_coords, placed) < _CLASH_FACTOR


def _is_axial(pg: "_ProbeGeom") -> bool:
    """True when every probe atom sits on the approach axis, so its own roll changes nothing.

    Only for such a probe do all azimuths collapse at polar=0. For a bent or pyramidal one the
    azimuth there is a pure roll about the normal, and whether the O=C=O is eclipsed or staggered
    with respect to the lattice oxygens is a real degree of freedom.
    """
    return all(abs(px) < 1e-9 and abs(py) < 1e-9 for _, px, py, _ in pg.coords)


def _place_probe(
    site_pos: np.ndarray,
    pg: _ProbeGeom,
    dist: float,
    polar_deg: float,
    azimuthal_deg: float,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> List[Tuple[str, np.ndarray]]:
    """Place probe atoms near *site_pos* along the specified approach direction.

    The angles are measured in the site's *local* frame ``(normal, t1, t2)``: polar_deg=0 →
    along the outward normal, i.e. perpendicular to the surface at the site; azimuthal_deg
    rotates around that normal. Measuring them against the file's +z instead would make the
    sampling depend on how the input .xyz happens to be oriented, and would aim the probe
    into the cluster for any site facing away from +z.

    Returns list of (symbol, global_position) for each probe atom.
    """
    polar = math.radians(polar_deg)
    azimuth = math.radians(azimuthal_deg)
    normal, t1, t2 = frame
    # Everything is built INSIDE the site frame and carried out to the global one by F, which
    # rotates with the cluster. Composing `_z_to_approach` on the local direction instead of the
    # global one is what makes the probe's own orientation covariant: the minimal rotation from
    # the file's +z is not (R(G·a) ≠ G·R(a) for a rigid rotation G), and that left the roll of
    # every non-axial probe tied to how the input .xyz happened to be oriented.
    F = np.column_stack((t1, t2, normal))
    local = np.array([math.sin(polar) * math.cos(azimuth),
                      math.sin(polar) * math.sin(azimuth),
                      math.cos(polar)])
    binding_pos = site_pos + (F @ local) * dist
    R = F @ _z_to_approach(local)
    return [(sym, binding_pos + R @ np.array([px, py, pz])) for sym, px, py, pz in pg.coords]


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run(
    cdft_json: "str | Path | dict",
    cluster_xyz: "str | Path",
    probe: str,
    *,
    n_sites: int = 0,
    n_configs: int = 6,
    polar_angles: Optional[List[float]] = None,
    output_dir: "str | Path | None" = None,
    cluster_label: str = "",
    probes_yaml: "str | Path | None" = None,
) -> GeometryResult:
    """Generate adsorption starting geometries guided by CDFT site ranking.

    Parameters
    ----------
    cdft_json    : CDFTResult JSON — path to a file, a dict, or a raw JSON string.
    cluster_xyz  : cluster geometry — a .xyz file or a Gaussian/ORCA output
                   (.out/.log); for outputs the geometry is extracted preserving
                   the SCF atom order.
    probe        : probe molecule — any built-in (H2, CO2, NH3, CO, N2, O2, H2O) or a
                   user-defined probe from ~/.cadmia/probes.yaml.
    n_sites      : number of top-ranked CDFT sites to place the probe on; 0 or less
                   means every site in the ranking.
    n_configs    : number of azimuthal orientations per (site × mode × polar).
    polar_angles : polar approach angles in degrees (default: [0, 20, 40, 60]).
                   At polar=0 all azimuths are equivalent; only one config is written.
    output_dir   : directory for .xyz output files; created if absent.
    cluster_label: label override (default: stem of cluster_xyz).
    probes_yaml  : path to a user probes file (default: ~/.cadmia/probes.yaml).
    """
    registry = _resolve_registry(probes_yaml)
    probe = probe.upper()
    if probe not in registry:
        raise ValueError(f"Unknown probe '{probe}'. Available: {sorted(registry)}")

    # ── Parse CDFTResult ─────────────────────────────────────
    if isinstance(cdft_json, dict):
        data = cdft_json
    else:
        p = Path(str(cdft_json))
        data = json.loads(p.read_text(encoding="utf-8") if p.exists() else str(cdft_json))

    # By rank, never by list position: the CDFT Agent emits its sites ordered by |Δf|,
    # but `repository.py` reorders them by atom_index when persisting, so a JSON that
    # came back from the database would otherwise sample the first n ATOMS of the
    # cluster instead of the n most reactive sites. The CLI already sorts by rank.
    ranked_sites = sorted(data.get("sites", []), key=lambda s: s.get("rank", 1 << 30))
    # n_sites <= 0 means every site: the generator's job is to cover the configuration space,
    # and the screening downstream is cheap.
    sites_data = ranked_sites if n_sites <= 0 else ranked_sites[:n_sites]

    # ── Parse cluster geometry (.xyz or a Gaussian/ORCA output) ──
    # Either way the atom order is preserved, so it keeps matching the SCF the CDFT
    # ranking came from.
    from cadmia.io import structure_from_file
    cluster_path = Path(cluster_xyz)
    struct = structure_from_file(cluster_path)
    cl_symbols = list(struct.atoms)
    cl_coords = [np.asarray(row, dtype=float) for row in struct.positions]
    label = cluster_label or cluster_path.stem

    out_dir: Optional[Path] = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if polar_angles is None:
        polar_angles = [0.0, 20.0, 40.0, 60.0]

    azimuthal_angles = [360.0 * i / n_configs for i in range(n_configs)]

    result = GeometryResult(cluster=label, probe=probe, n_sites=len(sites_data), output_dir=out_dir)
    config_idx = 0

    if not sites_data:
        # Without this the agent walks zero sites, writes zero files and returns a
        # normal-looking result: no configuration and no reason given.
        result.warnings.append(
            "The CDFT input has no 'sites' list, so no adsorption site was sampled and no "
            "configuration was generated — check that this is the CDFT Agent's JSON and "
            "that its run succeeded."
        )

    for site in sites_data:
        site_idx = site["atom_index"]   # 1-based, as emitted by the CDFT Agent
        site_sym = site["atom_symbol"]

        if site_idx < 1 or site_idx > len(cl_coords):
            result.warnings.append(
                f"Site atom_index {site_idx} out of range "
                f"(cluster has {len(cl_coords)} atoms); skipped."
            )
            continue

        zero_based = site_idx - 1
        if cl_symbols[zero_based] != site_sym:
            # Skipped, not merely flagged: generating anyway puts every configuration of this
            # site on the wrong atom while the filename and the JSON still claim the right one,
            # and the orchestrator discards warnings, so through `pipeline` nothing is seen.
            result.warnings.append(
                f"Atom {site_idx}: cluster .xyz has {cl_symbols[zero_based]} but "
                f"CDFT site is {site_sym} — the .xyz atom order does not match the SCF output; "
                f"no configuration generated for this site."
            )
            continue

        site_pos = cl_coords[zero_based]
        r_site   = covalent_radius(site_sym)

        frame, source = _site_frame(cl_symbols, cl_coords, zero_based)
        if source != "neighbours":
            result.warnings.append(
                f"Atom {site_idx} ({site_sym}): no outward direction from its neighbours "
                f"(centrosymmetric environment); fell back to '{source}'."
            )

        for pg in registry[probe]:
            r_probe = covalent_radius(pg.binding_atom)

            if r_site is None or r_probe is None:
                dist = 2.5
                result.warnings.append(
                    f"Missing covalent radius for {site_sym} or {pg.binding_atom}; "
                    f"using fallback {dist:.1f} Å."
                )
            else:
                # A hydrogen reaching an oxygen at r_O + r_H + 0.5 starts at 1.47 A, which is
                # already a proton-abstraction transition state and not a physisorption. Given an
                # atop site with no metal to receive the hydride, xTB resolves it into O–H plus a
                # stranded H, and that "dissociated H2" is an artefact of the seed. The extra
                # clearance puts it at a hydrogen-bond distance instead.
                clearance = 1.0 if (pg.binding_atom == "H" and site_sym == "O") else 0.5
                dist = r_site + r_probe + clearance

            axial = _is_axial(pg)
            for polar in polar_angles:
                # At polar=0 the azimuth stops being an approach direction and becomes the
                # probe's own roll about the normal. That collapses to one geometry only for an
                # axial probe; for a bent or pyramidal one it is a degree of freedom worth sampling
                # at exactly the most important approach.
                azims = [0.0] if (polar == 0.0 and axial) else azimuthal_angles
                for azimuth in azims:
                    placed = _place_probe(site_pos, pg, dist, polar, azimuth, frame)
                    # Slide the probe outward until nothing overlaps, rather than discard a
                    # sampling direction because one non-binding atom of the probe leans in.
                    pushed = 0.0
                    while (_has_clash(cl_symbols, cl_coords, placed)
                           and pushed < _PUSHOUT_MAX):
                        pushed += _PUSHOUT_STEP
                        placed = _place_probe(
                            site_pos, pg, dist + pushed, polar, azimuth, frame)
                    if _has_clash(cl_symbols, cl_coords, placed):
                        result.n_rejected_clash += 1
                        continue
                    if pushed > 0:
                        result.n_pushed_out += 1
                    used_dist = dist + pushed

                    all_symbols = list(cl_symbols) + [s for s, _ in placed]
                    all_coords  = list(cl_coords)  + [p for _, p in placed]

                    comment = (
                        f"cluster={label} probe={probe} "
                        f"site={site_sym}({site_idx}) orient={pg.orientation} "
                        f"polar={polar:.0f} azimuth={azimuth:.0f} dist={used_dist:.3f}A"
                        + (f" pushed=+{pushed:.2f}A" if pushed else "")
                    )

                    xyz_path: Optional[Path] = None
                    if out_dir is not None:
                        fname = (
                            f"{label}_{probe}_site{site_idx}"
                            f"_{pg.orientation}_p{polar:g}_a{azimuth:g}.xyz"
                        )
                        xyz_path = out_dir / fname
                        _write_xyz(xyz_path, all_symbols, all_coords, comment)

                    result.configs.append(
                        AdsorptionConfig(
                            site_atom_idx=site_idx,
                            site_atom_symbol=site_sym,
                            config_index=config_idx,
                            orientation=f"{pg.orientation}_p{polar:g}_a{azimuth:g}",
                            probe=probe,
                            approach_angle=polar,
                            azimuthal_angle=azimuth,
                            placement_dist=used_dist,
                            pushed_out=round(pushed, 3),
                            xyz_path=xyz_path,
                        )
                    )
                    config_idx += 1

    if result.n_pushed_out:
        result.warnings.append(
            f"{result.n_pushed_out} configuration(s) had a non-binding probe atom too close "
            f"to the cluster and were moved outward along their approach (up to "
            f"{_PUSHOUT_MAX:.1f} Å) to clear it; see pushed_out_A in the record."
        )
    if result.n_rejected_clash:
        result.warnings.append(
            f"{result.n_rejected_clash} configuration(s) dropped: the probe still overlapped "
            f"the cluster after being pushed {_PUSHOUT_MAX:.1f} Å out — that site has no room "
            f"for those orientations."
        )

    if out_dir is not None:
        # The site/orientation each geometry came from lives in this record, not in the .xyz.
        written_xyz = [c.xyz_path for c in result.configs if c.xyz_path]

        # A second run into the same directory overwrites the names it produces and leaves the
        # rest of the previous run in place. The sidecar then lists N2 files while the next
        # stage globs N1 > N2 of them, and the ones it did not write carry this run's command
        # and commit in a record that never mentions them.
        orphans = sorted({f.name for f in out_dir.glob("*.xyz")}
                         - {Path(x).name for x in written_xyz})
        if orphans:
            result.warnings.append(
                f"{len(orphans)} .xyz in {out_dir} are left over from an earlier run and are "
                f"not part of this one: " + ", ".join(orphans[:5])
                + (" …" if len(orphans) > 5 else "")
                + ". The next stage would read them too."
            )

        result.output_json = str(out_dir / "geometry.json")
        _, err = write_json_sidecar(result.output_json, result.to_json(), artifacts=written_xyz)
        if err:
            result.output_json = None
            result.warnings.append(err)

    return result
