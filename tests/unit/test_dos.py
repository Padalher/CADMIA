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
    text = out.read_text()
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
    payload = json.loads(out.read_text())
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
    text = out.read_text()
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
    assert "HOMO = -6.00 eV" in html
    assert "gap = 4.00 eV" in html


def test_cli_pdos_html(tmp_path, capsys):
    from cadmia.electronic.cli import main
    src = tmp_path / "x.out"
    src.write_text(_PDOS_BLOCK)
    out = tmp_path / "pdos.html"
    code = main(["pdos", str(src), "--format", "html", "-o", str(out)])
    assert code == 0
    assert "Plotly.newPlot" in out.read_text()


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
    assert "Plotly.newPlot" in produced.read_text()


def test_dos_grid_resolves_valence_peaks_despite_core_orbitals():
    import numpy as np
    dos = compute_dos([-4900.0, -8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0, sigma=0.1)
    step = dos.energy_ev[1] - dos.energy_ev[0]
    assert step <= 0.1 / 4
    peak = dos.dos_total[np.argmin(abs(dos.energy_ev + 8.0))]
    assert abs(peak - 1 / (0.1 * np.sqrt(2 * np.pi))) < 0.05 * peak
    assert abs(np.trapezoid(dos.dos_total, dos.energy_ev) - 3.0) < 1e-3


def test_dos_html_opens_on_the_frontier_window():
    from cadmia.electronic.plot import dos_html
    dos = compute_dos([-4900.0, -8.0, -2.0], homo_ev=-8.0, lumo_ev=-2.0, sigma=0.1)
    assert '"range": [-28.0, 8.0]' in dos_html(dos)
