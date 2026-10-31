"""Tests for total DOS from orbital eigenvalues (electronic.dos + cadmia.io.electronic eigenvalues)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadmia.electronic.dos import compute_dos, dos_from_output, write_dos

_FIX = Path(__file__).resolve().parent.parent / "fixtures" / "1tial_orca"
_ORCA_N = _FIX / "1tial_N_orca.out"
_HAS_ORCA = _ORCA_N.exists()
_requires_orca = pytest.mark.skipif(not _HAS_ORCA, reason="ORCA fixture not present")


# ─────────────────────────────────────────────
# compute_dos (pure)
# ─────────────────────────────────────────────

def test_compute_dos_closed_shell():
    dos = compute_dos([-10.0, -8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0, sigma=0.2)
    assert dos.dos_beta is None
    assert not dos.is_spin_polarized
    assert dos.gap_ev == pytest.approx(6.0)
    # peak near each eigenvalue
    assert dos.energy_ev[np.argmax(dos.dos_alpha)] < 0


def test_compute_dos_spin_polarized():
    dos = compute_dos([-9.0, -3.0], beta_eigenvalues_ev=[-9.1, -2.5], sigma=0.2)
    assert dos.is_spin_polarized
    assert np.allclose(dos.dos_total, dos.dos_alpha + dos.dos_beta)


# ─────────────────────────────────────────────
# cadmia.io.electronic ORCA eigenvalue extraction
# ─────────────────────────────────────────────

@_requires_orca
def test_orca_eigenvalues_parsed():
    from cadmia.io.orca import parse_orca_geometry
    from cadmia.io.electronic import parse_orca_electronic
    state = parse_orca_electronic(_ORCA_N, parse_orca_geometry(_ORCA_N))
    assert state.alpha_eigenvalues_ev is not None
    assert len(state.alpha_eigenvalues_ev) > 8          # core + valence + virtual
    assert state.beta_eigenvalues_ev is None             # neutral = closed shell
    # HOMO lies within the parsed spectrum
    assert min(state.alpha_eigenvalues_ev) < state.homo_ev < max(state.alpha_eigenvalues_ev)


# ─────────────────────────────────────────────
# dos_from_output (real ORCA)
# ─────────────────────────────────────────────

@_requires_orca
def test_dos_from_orca_output():
    dos = dos_from_output(_ORCA_N, sigma=0.15)
    assert dos.gap_ev == pytest.approx(4.5675, abs=1e-3)
    assert dos.homo_ev < dos.lumo_ev
    assert dos.dos_total.max() > 0
    assert len(dos.energy_ev) >= 2000


@_requires_orca
def test_write_dos_dat(tmp_path):
    dos = dos_from_output(_ORCA_N)
    out = tmp_path / "dos.dat"
    write_dos(dos, out)
    text = out.read_text(encoding="utf-8")
    assert "# HOMO" in text and "Energy(eV)" in text


def test_dos_from_output_without_eigenvalues_raises(tmp_path):
    bad = tmp_path / "bad.out"
    bad.write_text(
        " Standard orientation:\n"
        " ---------------------------------------------------------------------\n"
        " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
        " Number     Number       Type             X           Y           Z\n"
        " ---------------------------------------------------------------------\n"
        "      1         22           0        0.0  0.0  0.0\n"
        " ---------------------------------------------------------------------\n"
        " SCF Done:  E(RB3LYP) = -100.0     A.U. after 10 cycles\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        dos_from_output(bad)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@_requires_orca
def test_cli_dos_summary(capsys):
    from cadmia.electronic.cli import main
    code = main(["dos", str(_ORCA_N)])
    assert code == 0
    out = capsys.readouterr().out
    assert "DOS:" in out and "gap=" in out


@_requires_orca
def test_cli_dos_json(tmp_path, capsys):
    import json
    from cadmia.electronic.cli import main
    out = tmp_path / "dos.json"
    code = main(["dos", str(_ORCA_N), "--format", "json", "-o", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "dos_total" in payload and payload["gap_ev"] == pytest.approx(4.5675, abs=1e-3)
    assert payload["spin_polarized"] is False


# ─────────────────────────────────────────────
# PDOS (per-MO Loewdin populations)
# ─────────────────────────────────────────────

_PDOS_BLOCK = """\
                                 O   R   C   A

LOEWDIN ORBITAL POPULATIONS PER MO
----------------------------------
THRESHOLD FOR PRINTING IS 0.1%%
                      0         1         2   
                  -0.50000  -0.30000   0.10000
                   2.00000   2.00000   0.00000
                  --------  --------  --------
  0Ti  1s             90.0       5.0       0.0
  1O   2pz            10.0      95.0      20.0
  1O   2px             0.0       0.0      80.0
"""

_PDOS_OPEN_SHELL = """\
                                 O   R   C   A

LOEWDIN ORBITAL POPULATIONS PER MO
----------------------------------
THRESHOLD FOR PRINTING IS 0.1%%
SPIN UP
                      0         1   
                  -0.50000  -0.10000
                   1.00000   0.00000
                  --------  --------
  0Al  3s            100.0      50.0
SPIN DOWN
                      0         1   
                  -0.40000   0.20000
                   1.00000   0.00000
                  --------  --------
  1O   2pz           100.0      60.0
"""


def test_parse_mo_populations_closed_shell(tmp_path):
    from cadmia.io.electronic import parse_orca_mo_populations
    p = tmp_path / "x.out"
    p.write_text(_PDOS_BLOCK)
    pops = parse_orca_mo_populations(p)
    assert pops["beta"] is None
    alpha = pops["alpha"]
    assert len(alpha) == 3
    assert alpha[0]["energy_ev"] == pytest.approx(-0.5 * 27.2114)
    assert alpha[0]["occ"] == 2.0
    assert alpha[0]["contributions"][(0, "Ti", "1s")] == 90.0
    assert alpha[2]["contributions"][(1, "O", "2px")] == 80.0
    assert (0, "Ti", "1s") not in alpha[2]["contributions"]   # zeros dropped


def test_parse_mo_populations_open_shell(tmp_path):
    from cadmia.io.electronic import parse_orca_mo_populations
    p = tmp_path / "x.out"
    p.write_text(_PDOS_OPEN_SHELL)
    pops = parse_orca_mo_populations(p)
    assert len(pops["alpha"]) == 2
    assert len(pops["beta"]) == 2
    assert pops["alpha"][0]["contributions"] == {(0, "Al", "3s"): 100.0}
    assert pops["beta"][1]["contributions"] == {(1, "O", "2pz"): 60.0}


def test_parse_mo_populations_missing_block(tmp_path):
    from cadmia.io.electronic import parse_orca_mo_populations
    p = tmp_path / "x.out"
    p.write_text("FINAL SINGLE POINT ENERGY -1.0\n")
    assert parse_orca_mo_populations(p) == {"alpha": None, "beta": None}


def test_pdos_from_output_series(tmp_path):
    from cadmia.electronic.dos import pdos_from_output
    p = tmp_path / "x.out"
    p.write_text(_PDOS_BLOCK)
    pdos = pdos_from_output(p, sigma=0.2, n_points=500)
    assert set(pdos.series) == {"Ti", "O"}
    assert pdos.homo_ev == pytest.approx(-0.3 * 27.2114)
    assert pdos.lumo_ev == pytest.approx(0.1 * 27.2114)
    # at the first MO energy, Ti dominates (90% vs 10%)
    i = int(np.argmin(np.abs(pdos.energy_ev - (-0.5 * 27.2114))))
    assert pdos.series["Ti"][i] > pdos.series["O"][i]
    # projections sum to ~the total there (no threshold losses in this fixture)
    assert pdos.series["Ti"][i] + pdos.series["O"][i] == pytest.approx(pdos.total[i], rel=1e-6)


_PDOS_NEGATIVE = """\
                                 O   R   C   A

LOEWDIN ORBITAL POPULATIONS PER MO
----------------------------------
THRESHOLD FOR PRINTING IS 0.1%%
                      0         1
                  -0.50000  -0.30000
                   2.00000   2.00000
                  --------  --------
  0Ti  1s             90.0      -5.0
  1O   2pz            10.0     105.0
"""


def test_pdos_warns_on_negative_loewdin_populations(tmp_path):
    """A negative Loewdin projection must be flagged, not propagated silently."""
    from cadmia.electronic.dos import pdos_from_output
    p = tmp_path / "x.out"
    p.write_text(_PDOS_NEGATIVE)
    with pytest.warns(UserWarning, match="Negative Loewdin populations"):
        pdos_from_output(p, sigma=0.2, n_points=200)


def test_pdos_from_output_raises_without_block(tmp_path):
    from cadmia.electronic.dos import pdos_from_output
    p = tmp_path / "x.out"
    p.write_text("FINAL SINGLE POINT ENERGY -1.0\n")
    with pytest.raises(ValueError, match="P_OrbPopMO_L"):
        pdos_from_output(p)


def test_cli_pdos_writes_dat(tmp_path, capsys):
    from cadmia.electronic.cli import main
    src = tmp_path / "x.out"
    src.write_text(_PDOS_BLOCK)
    out = tmp_path / "pdos.dat"
    code = main(["pdos", str(src), "-o", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "PDOS_O" in text and "PDOS_Ti" in text and "DOS_total" in text


# ─────────────────────────────────────────────
# Interactive HTML plots
# ─────────────────────────────────────────────

def test_pdos_html_contains_traces_and_frontier(tmp_path):
    from cadmia.electronic.dos import pdos_from_output
    from cadmia.electronic.plot import pdos_html
    p = tmp_path / "x.out"
    p.write_text(_PDOS_BLOCK)
    pdos = pdos_from_output(p, sigma=0.2, n_points=200)
    html = pdos_html(pdos, title="PDOS test")
    assert "Plotly.newPlot" in html
    assert '"name": "Ti"' in html and '"name": "O"' in html
    assert "HOMO" in html and "LUMO" in html and "gap" in html
    assert "cdn.plot.ly" in html


def test_dos_html_spin_polarized(tmp_path):
    from cadmia.electronic.dos import DOSResult
    from cadmia.electronic.plot import dos_html
    e = np.linspace(-10, 5, 50)
    dos = DOSResult(energy_ev=e, dos_alpha=np.ones(50), dos_beta=np.ones(50) * 0.5,
                    homo_ev=-6.0, lumo_ev=-2.0)
    html = dos_html(dos)
    assert '"name": "alpha"' in html and '"name": "beta"' in html
    assert "<b>HOMO</b><br>-6.00 eV" in html
    assert "<b>gap</b> 4.00 eV" in html


def test_cli_pdos_html(tmp_path, capsys):
    from cadmia.electronic.cli import main
    src = tmp_path / "x.out"
    src.write_text(_PDOS_BLOCK)
    out = tmp_path / "pdos.html"
    code = main(["pdos", str(src), "--format", "html", "-o", str(out)])
    assert code == 0
    assert "Plotly.newPlot" in out.read_text(encoding="utf-8")


def test_cli_dos_html_default_name(tmp_path, capsys, monkeypatch):
    from cadmia.electronic.cli import main
    src = tmp_path / "y.out"
    src.write_text(_PDOS_BLOCK + "\nORBITAL ENERGIES\n")
    monkeypatch.chdir(tmp_path)
    if not _HAS_ORCA:
        pytest.skip("ORCA fixture not present")
    code = main(["dos", str(_ORCA_N), "--format", "html"])
    assert code == 0
    produced = tmp_path / (_ORCA_N.stem + "_dos.html")
    assert produced.exists()
    assert "Plotly.newPlot" in produced.read_text(encoding="utf-8")


def test_dos_grid_resolves_valence_peaks_despite_core_orbitals():
    import numpy as np
    dos = compute_dos([-4900.0, -8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0, sigma=0.1)
    step = dos.energy_ev[1] - dos.energy_ev[0]
    assert step <= 0.1 / 4
    peak = dos.dos_total[np.argmin(abs(dos.energy_ev + 8.0))]
    assert abs(peak - 2 / (0.1 * np.sqrt(2 * np.pi))) < 0.05 * peak
    assert abs(np.trapezoid(dos.dos_total, dos.energy_ev) - 6.0) < 1e-3     # 3 orbitals × 2 spins


def test_dos_html_opens_on_the_frontier_window():
    from cadmia.electronic.plot import dos_html
    dos = compute_dos([-4900.0, -8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0, sigma=0.1)
    assert '"range": [-28.0, 0.0]' in dos_html(dos)     # LUMO + 10 eV, cut where the curve ends


def test_closed_shell_dos_counts_both_spins():
    closed = compute_dos([-9.0, -3.0], sigma=0.2)
    assert np.allclose(closed.dos_total, 2 * closed.dos_alpha)
    assert abs(np.trapezoid(closed.dos_total, closed.energy_ev) - 4.0) < 1e-3
    opened = compute_dos([-9.0, -3.0], beta_eigenvalues_ev=[-9.0, -3.0], sigma=0.2)
    assert np.allclose(opened.dos_total, closed.dos_total)


def test_pdos_closed_shell_counts_both_spins(tmp_path):
    from cadmia.electronic.dos import pdos_from_output
    p = tmp_path / "x.out"
    p.write_text(_PDOS_BLOCK)
    pdos = pdos_from_output(p, sigma=0.2, n_points=500)
    assert abs(np.trapezoid(pdos.total, pdos.energy_ev) - 6.0) < 1e-3                  # 3 orbitals × 2 spins


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_nonpositive_sigma_is_rejected(tmp_path, sigma):
    from cadmia.electronic.dos import pdos_from_output
    p = tmp_path / "x.out"
    p.write_text(_PDOS_BLOCK)
    with pytest.raises(ValueError, match="sigma must be positive"):
        compute_dos([-9.0, -3.0], sigma=sigma)
    with pytest.raises(ValueError, match="sigma must be positive"):
        pdos_from_output(p, sigma=sigma)


def test_wide_broadening_is_not_cut_at_the_grid_edge():
    dos = compute_dos([-9.0, -3.0], sigma=1.5)
    assert abs(np.trapezoid(dos.dos_total, dos.energy_ev) - 4.0) < 1e-3


def test_dos_html_drops_the_empty_stretches_between_core_levels():
    import json
    import re
    from cadmia.electronic.plot import dos_html
    dos = compute_dos([-4900.0, -8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0, sigma=0.1)
    html = dos_html(dos)
    x = json.loads(re.search(r'"x": (\[.*?\])', html).group(1))
    assert len(x) < len(dos.energy_ev) / 20
    assert min(x) <= -4900.0 and max(x) >= -2.0


@_requires_orca
def test_cli_dos_format_follows_the_output_suffix(tmp_path):
    import json
    from cadmia.electronic.cli import main
    out = tmp_path / "dos.json"
    assert main(["dos", str(_ORCA_N), "-o", str(out)]) == 0
    assert "dos_total" in json.loads(out.read_text(encoding="utf-8"))
    assert main(["dos", str(_ORCA_N), "-o", str(tmp_path / "d.csv")]) == 0
    assert (tmp_path / "d.csv").read_text(encoding="utf-8").splitlines()[-1].count(",") == 1


def _png_width(path):
    import struct
    return struct.unpack(">I", path.read_bytes()[16:20])[0]


@pytest.mark.parametrize("width,inches", [("single", 3.4), ("double", 6.5)])
def test_static_dos_figure_width_in_pixels(tmp_path, width, inches):
    pytest.importorskip("matplotlib")
    from cadmia.electronic.figures import write_dos_figure
    dos = compute_dos([-4900.0, -8.0, -7.5, -2.0], homo_ev=-7.5, lumo_ev=-2.0, sigma=0.1)
    out = write_dos_figure(dos, tmp_path / "dos.png", width=width)
    assert _png_width(out) == round(inches * 300)


@pytest.mark.parametrize("fmt", ["svg", "pdf"])
def test_static_vector_figures(tmp_path, fmt):
    pytest.importorskip("matplotlib")
    from cadmia.electronic.figures import write_dos_figure
    dos = compute_dos([-8.0, -7.5, -2.0], homo_ev=-7.5, lumo_ev=-2.0, sigma=0.1)
    out = write_dos_figure(dos, tmp_path / f"dos.{fmt}")
    head = out.read_bytes()[:200]
    assert (b"<?xml" in head) if fmt == "svg" else head.startswith(b"%PDF")
    if fmt == "svg":
        assert "DOS (states/eV)" in out.read_text(encoding="utf-8")


def test_static_ir_figure_has_sticks_on_their_own_km_mol_axis(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from cadmia.electronic import figures
    from cadmia.electronic.ir_spectrum import IRSpectrum
    grid = np.linspace(500.0, 2500.0, 400)
    spectrum = IRSpectrum(wavenumber_cm1=grid, intensity=np.exp(-((grid - 1500) / 20) ** 2) * 3e-3,
                          peak_positions=np.array([1500.0]), peak_intensities=np.array([350.0]),
                          scaling_factor=1.0, fwhm_cm1=16.0, lineshape="lorentzian")
    built = []
    monkeypatch.setattr(figures, "save_figure", lambda fig, path: built.append(fig))
    figures.write_ir_figure(spectrum, tmp_path / "ir.png")
    curve_axis, stick_axis = built[0].axes
    assert stick_axis.get_ylabel() == "IR intensity (km/mol)"
    assert curve_axis.get_ylabel() == "Intensity (arb. units)"
    assert stick_axis.get_ylim()[1] >= 350.0 and curve_axis.get_ylim()[1] < 1.0
    assert curve_axis.xaxis_inverted()


def test_static_figures_leave_matplotlib_global_state_alone(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    from cadmia.electronic.figures import write_dos_figure
    before = (matplotlib.get_backend(), dict(matplotlib.rcParams))
    dos = compute_dos([-8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0)
    write_dos_figure(dos, tmp_path / "dos.png")
    assert (matplotlib.get_backend(), dict(matplotlib.rcParams)) == before


@_requires_orca
def test_cli_dos_reports_a_bad_sigma_instead_of_a_traceback(capsys):
    from cadmia.electronic.cli import main
    assert main(["dos", str(_ORCA_N), "--sigma", "0"]) == 1
    assert "sigma must be positive" in capsys.readouterr().out
