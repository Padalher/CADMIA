"""Tests for ORCA vibrational parsing and IR spectrum broadening."""
import numpy as np
import pytest

from cadmia.electronic.ir_spectrum import compute_ir_spectrum, ir_spectrum_from_output
from cadmia.io.vibrational import VibrationalAnalysis, VibrationalMode, parse_orca_vibrational


def test_imaginary_dead_zone_is_symmetric():
    """|ν| ≤ 10 cm⁻¹ is trans/rot noise, not imaginary; only ν < −10 is a saddle."""
    modes = [VibrationalMode(index=i, frequency_cm1=f)
             for i, f in enumerate([-50.0, -5.0, 0.0, 5.0, 250.0])]
    va = VibrationalAnalysis(modes=modes, n_atoms=2)
    assert [m.frequency_cm1 for m in va.imaginary_modes] == [-50.0]   # −5 is NOT imaginary
    assert [m.frequency_cm1 for m in va.real_modes] == [250.0]        # +5 is NOT a real mode


def _orca_freq_output(freq_hi: float = 2396.32, ir_hi: float = 600.20,
                      disp_offset: float = 0.0) -> str:
    """A minimal ORCA 6 frequency output for a 3-atom (CO2-like) molecule.

    The parameters let a caller emit a SECOND, distinguishable section (as a re-run or a
    concatenated output would hold) to check which one the parsers read.
    """
    freq_lines = [
        "-----------------------",
        "VIBRATIONAL FREQUENCIES",
        "-----------------------",
        "",
        "Scaling factor for frequencies =  1.000000000  (already applied!)",
        "",
        "   0:         0.00 cm**-1",
        "   1:         0.00 cm**-1",
        "   2:         0.00 cm**-1",
        "   3:         0.00 cm**-1",
        "   4:         0.00 cm**-1",
        "   5:         0.00 cm**-1",
        "   6:       667.45 cm**-1",
        "   7:       667.45 cm**-1",
        f"   8:      {freq_hi:.2f} cm**-1",
        "",
    ]
    # NORMAL MODES: 9 cartesian coords (rows) x 9 modes (cols), two column blocks.
    nm = ["------------", "NORMAL MODES", "------------",
          "", "These modes are the Cartesian displacements.", ""]
    block1_header = "      " + "".join(f"{c:>11d}" for c in range(6))
    nm.append(block1_header)
    rng = np.arange(9, dtype=float) / 100.0 + disp_offset
    for coord in range(9):
        vals = "".join(f"{(rng[coord] + 0.01 * c):>11.6f}" for c in range(6))
        nm.append(f"  {coord:>4d}{vals}")
    block2_header = "      " + "".join(f"{c:>11d}" for c in range(6, 9))
    nm.append(block2_header)
    for coord in range(9):
        vals = "".join(f"{(rng[coord] + 0.1 * c):>11.6f}" for c in range(3))
        nm.append(f"  {coord:>4d}{vals}")
    nm.append("")

    ir = [
        "-----------",
        "IR SPECTRUM",
        "-----------",
        "",
        " Mode   freq       eps      Int      T**2         TX        TY        TZ",
        "        cm**-1   L/(mol*cm) km/mol    a.u.",
        "----------------------------------------------------------------------------",
        "   6:    667.45   0.001234     30.50  0.000123  ( 0.001  0.002  0.003)",
        "   7:    667.45   0.001234     30.50  0.000123  ( 0.001  0.002  0.003)",
        f"   8:   {freq_hi:.2f}   0.025678    {ir_hi:.2f}  0.002456  ( 0.010  0.000  0.000)",
        "",
        "The first frequency considered to be a vibration is 6",
    ]
    return "\n".join(freq_lines + nm + ir) + "\n"


@pytest.fixture()
def orca_output(tmp_path):
    p = tmp_path / "co2_freq.out"
    p.write_text(_orca_freq_output(), encoding="utf-8")
    return p


def test_parses_the_last_vibrational_section(tmp_path):
    """Regression: a re-run or a concatenated output holds several vibrational sections and
    the converged one is the LAST. The parsers used to stop at the first, so an output whose
    energy came from the final geometry reported the frequencies of an earlier one."""
    p = tmp_path / "two_sections.out"
    p.write_text(_orca_freq_output()
                 + _orca_freq_output(freq_hi=2000.00, ir_hi=111.10, disp_offset=5.0),
                 encoding="utf-8")
    a = parse_orca_vibrational(p)
    assert len(a.modes) == 9                                       # not 18: no concatenation
    assert a.real_modes[-1].frequency_cm1 == pytest.approx(2000.00)  # last, not 2396.32
    assert a.modes[8].ir_intensity_km_mol == pytest.approx(111.10)   # last, not 600.20
    # Frequencies, intensities and displacements must all come from the SAME section:
    # taking them from different blocks would describe two different geometries.
    # displacements is (n_atoms, 3); the first coordinate carries the offset of the section.
    assert a.modes[0].displacements[0][0] == pytest.approx(5.0)


def test_parse_frequencies_and_atoms(orca_output):
    analysis = parse_orca_vibrational(orca_output)
    assert analysis.n_atoms == 3
    assert len(analysis.modes) == 9
    assert analysis.scaling_factor == pytest.approx(1.0)
    # Six near-zero translational/rotational modes are filtered out of real_modes.
    assert len(analysis.real_modes) == 3
    assert analysis.real_modes[-1].frequency_cm1 == pytest.approx(2396.32)


def test_ir_intensities_picked_from_correct_column(orca_output):
    analysis = parse_orca_vibrational(orca_output)
    by_index = {m.index: m for m in analysis.modes}
    assert by_index[6].ir_intensity_km_mol == pytest.approx(30.50)
    assert by_index[8].ir_intensity_km_mol == pytest.approx(600.20)
    # Translational modes have no IR data.
    assert by_index[0].ir_intensity_km_mol is None


def test_normal_mode_displacements_shape(orca_output):
    analysis = parse_orca_vibrational(orca_output)
    mode8 = next(m for m in analysis.modes if m.index == 8)
    assert mode8.displacements is not None
    assert mode8.displacements.shape == (3, 3)


def test_ir_spectrum_peaks_and_scaling(orca_output):
    analysis = parse_orca_vibrational(orca_output)
    spec = compute_ir_spectrum(analysis, scale=0.96, fwhm=16.0)
    assert spec.peak_positions.size == 3
    # Strongest band is the asymmetric stretch, scaled.
    assert spec.peak_positions[np.argmax(spec.peak_intensities)] == pytest.approx(2396.32 * 0.96)
    # The curve peaks near the most intense, scaled mode.
    apex = spec.wavenumber_cm1[np.argmax(spec.intensity)]
    assert apex == pytest.approx(2396.32 * 0.96, abs=spec.fwhm_cm1)


def test_ir_spectrum_from_output_lorentzian_vs_gaussian(orca_output):
    lor = ir_spectrum_from_output(orca_output, lineshape="lorentzian")
    gau = ir_spectrum_from_output(orca_output, lineshape="gaussian")
    assert lor.intensity.max() > 0
    assert gau.intensity.max() > 0
    assert lor.lineshape == "lorentzian"
    assert gau.lineshape == "gaussian"


def test_no_frequencies_returns_empty(tmp_path):
    p = tmp_path / "single_point.out"
    p.write_text("FINAL SINGLE POINT ENERGY  -123.456\n", encoding="utf-8")
    analysis = parse_orca_vibrational(p)
    assert analysis.modes == []
    assert analysis.n_atoms == 0
