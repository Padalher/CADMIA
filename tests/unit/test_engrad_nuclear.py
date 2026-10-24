"""Tests for the ORCA .engrad reader and the nuclear Fukui function."""
from __future__ import annotations

import numpy as np
import pytest

from cadmia.cdft.nuclear import nuclear_fukui
from cadmia.io.engrad import EngradResult, parse_engrad


def _write_engrad(path, gradient, energy=-10.0, numbers=(8, 1), coords=None):
    n = len(gradient)
    coords = coords if coords is not None else [[0.0, 0.0, float(i)] for i in range(n)]
    lines = ["#", "# Number of atoms", "#", f" {n}",
             "#", "# The current total energy in Eh", "#", f"  {energy:.12f}",
             "#", "# The current gradient in Eh/bohr", "#"]
    for row in gradient:
        lines += [f"  {v:.12f}" for v in row]
    lines += ["#", "# The atomic numbers and current coordinates in Bohr", "#"]
    for z, c in zip(numbers, coords):
        lines.append(f"  {z}   {c[0]:.7f}  {c[1]:.7f}  {c[2]:.7f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── reader ──

def test_parse_engrad_reads_energy_gradient_and_geometry(tmp_path):
    f = _write_engrad(tmp_path / "a.engrad", [[0.1, 0.2, 0.3], [-0.1, 0.0, 0.0]], energy=-5.5)
    r = parse_engrad(f)
    assert r.n_atoms == 2
    assert r.energy_hartree == pytest.approx(-5.5)
    assert r.gradient.shape == (2, 3)
    assert r.gradient[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert r.atomic_numbers == [8, 1]


def test_the_force_is_minus_the_gradient(tmp_path):
    # ORCA writes ∂E/∂R; every consumer wants F = -∂E/∂R, converted in one place.
    f = _write_engrad(tmp_path / "a.engrad", [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert parse_engrad(f).force()[0].tolist() == pytest.approx([-0.1, 0.0, 0.0])


def test_a_truncated_gradient_is_refused(tmp_path):
    # Read short, the remaining components land on the wrong atoms and nothing says so.
    f = tmp_path / "bad.engrad"
    f.write_text("#\n# Number of atoms\n#\n 3\n#\n# energy\n#\n -1.0\n"
                 "#\n# grad\n#\n 0.1\n 0.2\n 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares 3 atoms"):
        parse_engrad(f)


def test_a_file_that_is_not_an_engrad_is_refused(tmp_path):
    f = tmp_path / "nope.txt"
    f.write_text("# just a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an ORCA .engrad"):
        parse_engrad(f)


# ── nuclear Fukui ──

def _res(gradient, coords=None, numbers=(8, 1)):
    g = np.asarray(gradient, dtype=float)
    c = np.asarray(coords if coords is not None else [[0.0, 0.0, float(i)] for i in range(len(g))],
                   dtype=float)
    return EngradResult(len(g), 0.0, g, list(numbers), c, "mem")


def test_phi_is_the_central_difference_of_the_forces(tmp_path):
    anion = _res([[0.0, 0.0, -0.1], [0.0, 0.0, 0.0]])     # force +0.1 z
    cation = _res([[0.0, 0.0, +0.1], [0.0, 0.0, 0.0]])    # force -0.1 z
    phi = nuclear_fukui(anion, cation).phi
    assert phi[0].tolist() == pytest.approx([0.0, 0.0, 0.1])
    assert phi[1].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_charged_states_on_a_different_geometry_are_refused():
    with pytest.raises(ValueError, match="SAME geometry"):
        nuclear_fukui(_res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                      _res([[0.0, 0.0, 0.0]], numbers=(8,)))


def test_a_drifted_geometry_warns_because_phi_is_defined_at_fixed_nuclei():
    a = _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], coords=[[0, 0, 0], [0, 0, 1.0]])
    c = _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], coords=[[0, 0, 0], [0, 0, 1.5]])
    assert any("fixed nuclei" in w for w in nuclear_fukui(a, c).warnings)


def test_a_neutral_that_is_not_stationary_warns():
    a = _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    c = _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    n = _res([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])      # big residual force
    assert any("stationary point" in w for w in nuclear_fukui(a, c, neutral=n).warnings)


def test_the_one_sided_phis_are_reported_separately_not_averaged():
    # ∂F/∂N is discontinuous at integer N, as it is for the electronic Fukui function.
    a = _res([[0.0, 0.0, -0.3], [0.0, 0.0, 0.0]])
    c = _res([[0.0, 0.0, +0.1], [0.0, 0.0, 0.0]])
    n = _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    r = nuclear_fukui(a, c, neutral=n)
    assert r.phi_plus[0][2] == pytest.approx(0.3)
    assert r.phi_minus[0][2] == pytest.approx(0.1)
    assert r.phi_plus[0][2] != pytest.approx(r.phi_minus[0][2])


def test_the_nuclear_dual_is_the_second_difference_of_the_forces():
    # ∂²F/∂N² = F(N+1) - 2F(N) + F(N-1): a third derivative of E.
    a = _res([[0.0, 0.0, -0.3], [0.0, 0.0, 0.0]])
    c = _res([[0.0, 0.0, +0.1], [0.0, 0.0, 0.0]])
    n = _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    r = nuclear_fukui(a, c, neutral=n)
    assert r.phi_dual[0][2] == pytest.approx(0.3 - 0.1)
    assert r.phi_dual[0].tolist() == pytest.approx((r.phi_plus[0] - r.phi_minus[0]).tolist())


def test_without_the_neutral_there_is_no_nuclear_dual():
    r = nuclear_fukui(_res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                      _res([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert r.phi_dual is None


# --- CLI wiring (cadmia cdft --engrad-*) ------------------------------------------------

def _stub_result(n_atoms=2, ranked=True):
    """A CDFT result whose sites come RANK-ordered, as cdft_agent really emits them."""
    from cadmia.agents.cdft_agent import CDFTResult, SiteDescriptor
    # symbols must match `_write_engrad`'s default numbers=(8, 1) — O, H — because the nuclear
    # block now checks the .engrad against the sites element by element.
    elements = ["O", "H"]
    sites = [SiteDescriptor(atom_index=i + 1, atom_symbol=elements[i % len(elements)],
                            coordinates=(0.0, 0.0, 0.0),
                            fukui_plus=None, fukui_minus=None, delta_f=0.5 * (i + 1),
                            character="electrophilic", rank=i + 1)
             for i in range(n_atoms)]
    if ranked:   # cdft_agent sorts by |Δf| descending, so atom N comes first
        sites.sort(key=lambda s: abs(s.delta_f), reverse=True)
    return CDFTResult(cluster="stub", calculation_id="x", charge_scheme="hirshfeld",
                      sites=sites, global_descriptors={}, sources={})


def test_the_nuclear_block_needs_both_charged_gradients(tmp_path):
    import click

    from cadmia.cli import _attach_nuclear_fukui

    f = _write_engrad(tmp_path / "a.engrad", [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(click.UsageError):
        _attach_nuclear_fukui(_stub_result(), str(f), None, None, 5)


def test_the_nuclear_block_reaches_the_json(tmp_path):
    import json

    from cadmia.cli import _attach_nuclear_fukui

    anion = _write_engrad(tmp_path / "np1.engrad", [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cation = _write_engrad(tmp_path / "nm1.engrad", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    neutral = _write_engrad(tmp_path / "n.engrad", [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = _stub_result()
    _attach_nuclear_fukui(result, str(anion), str(cation), str(neutral), 5)
    payload = json.loads(result.to_json())

    assert payload["nuclear"]["has_neutral"] is True
    first = payload["nuclear"]["sites"][0]
    # forces are -gradient, so Φ = ½(F₊ − F₋) = ½(−0.2 − 0) = −0.1 on x
    assert first["phi"][0] == pytest.approx(-0.1)
    assert "theta" not in first
    assert "phi_dual" in first


def test_without_the_engrads_the_json_has_no_nuclear_block():
    import json

    assert "nuclear" not in json.loads(_stub_result().to_json())


def test_phi_rows_follow_the_atom_not_the_rank(tmp_path):
    """Regression: the sites arrive ranked by |Δf|, the gradients in atom order."""
    import json

    from cadmia.cli import _attach_nuclear_fukui

    # atom 1 carries all the force response; atom 2 carries the larger |Δf| and so ranks first
    anion = _write_engrad(tmp_path / "np1.engrad", [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cation = _write_engrad(tmp_path / "nm1.engrad", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = _stub_result()
    assert result.sites[0].atom_index == 2, "the fixture must arrive ranked, not in atom order"

    _attach_nuclear_fukui(result, str(anion), str(cation), None, 5)
    rows = {r["atom_index"]: r for r in json.loads(result.to_json())["nuclear"]["sites"]}

    assert rows[1]["phi_norm"] == pytest.approx(0.1)   # the moving nucleus
    assert rows[2]["phi_norm"] == pytest.approx(0.0)


# --- Φ projected on a direction ---------------------------------------------------

def test_directional_phi_keeps_the_sign_the_magnitude_discards():
    from cadmia.cdft.nuclear import directional_phi

    phi = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]])
    axes = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    along = directional_phi(phi, axes)

    assert along[0] == pytest.approx(0.3)
    assert along[1] == pytest.approx(-0.3)
    # the norm cannot tell them apart, which is the whole point
    assert np.linalg.norm(phi[0]) == pytest.approx(np.linalg.norm(phi[1]))


def test_directional_phi_accepts_one_axis_per_atom_or_one_shared():
    from cadmia.cdft.nuclear import directional_phi

    phi = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    shared = directional_phi(phi, np.array([1.0, 0.0, 0.0]))
    per_atom = directional_phi(phi, np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

    assert shared == pytest.approx(per_atom)
    assert shared[0] == pytest.approx(1.0)
    assert shared[1] == pytest.approx(0.0)


def test_the_projection_only_appears_when_asked(tmp_path):
    import json

    from cadmia.cli import _attach_nuclear_fukui

    anion = _write_engrad(tmp_path / "np1.engrad", [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cation = _write_engrad(tmp_path / "nm1.engrad", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    plain = _stub_result()
    _attach_nuclear_fukui(plain, str(anion), str(cation), None, 5)
    assert "phi_along" not in json.loads(plain.to_json())["nuclear"]["sites"][0]

    projected = _stub_result()
    _attach_nuclear_fukui(projected, str(anion), str(cation), None, 5, project_normal=True)
    assert "phi_along" in json.loads(projected.to_json())["nuclear"]["sites"][0]


def test_engrad_atoms_must_match_the_sites_element_by_element(tmp_path):
    """Counting atoms is not enough: two runs of the same cluster can differ in atom order."""
    import click

    from cadmia.cli import _attach_nuclear_fukui

    # sites say O,H (via _stub_result); the gradients say H,O — same count, swapped
    anion = _write_engrad(tmp_path / "np1.engrad", [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
                          numbers=(1, 8))
    cation = _write_engrad(tmp_path / "nm1.engrad", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                           numbers=(1, 8))

    with pytest.raises(click.UsageError, match="element by element"):
        _attach_nuclear_fukui(_stub_result(), str(anion), str(cation), None, 5)
