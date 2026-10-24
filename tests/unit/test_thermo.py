"""Tests for cadmia.electronic.thermo: quasi-RRHO vibrational entropy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadmia.core.electronic_state import ElectronicState
from cadmia.core.structure import Structure
from cadmia.electronic.thermo import gibbs_quasi_rrho, vibrational_entropy
from cadmia.io.electronic import load_electronic

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_CAL = 4.184


def test_harmonic_entropy_reproduces_what_gaussian_prints():
    """TiO2_monomer.out prints S_vib = 1.335 cal/(mol·K) at 298.15 K (harmonic)."""
    state, engine = load_electronic(FIXTURES / "TiO2_monomer.out")
    assert engine == "gaussian" and state.entropy_scheme == "harmonic"
    s = vibrational_entropy(state.frequencies_cm1, state.temperature_kelvin) / _CAL
    assert s == pytest.approx(1.335, abs=0.002)


def test_grimme_entropy_reproduces_orca():
    """ORCA 6.1.1 (Quasi RRHO True, 100 cm⁻¹) printed T·S_vib = 0.00799945 Eh for these 18 modes
    of an Al2TiO5 opt+freq."""
    freqs = [116.45, 187.18, 202.81, 251.86, 309.34, 319.74, 376.13, 456.40, 541.36,
             549.56, 577.97, 585.67, 705.49, 734.59, 756.40, 879.02, 932.18, 1077.42]
    ts = 298.15 * vibrational_entropy(freqs, 298.15, "grimme") / 2625499.6394799
    assert ts == pytest.approx(0.00799945, abs=1e-6)


def test_soft_mode_diverges_harmonically_and_stays_bounded_with_grimme():
    nus = (1e-2, 1e-4, 1e-6)
    harmonic = [vibrational_entropy([nu], 298.15) for nu in nus]
    grimme = [vibrational_entropy([nu], 298.15, "grimme") for nu in nus]
    assert harmonic[2] - harmonic[1] == pytest.approx(8.314462618 * 4.60517, rel=1e-3)  # R·ln 100
    assert abs(grimme[2] - grimme[1]) < 0.1                        # free-rotor limit (μ' → B_av)
    assert all(g < h for g, h in zip(grimme, harmonic))
    assert vibrational_entropy([2000.0], 298.15, "grimme") == pytest.approx(
        vibrational_entropy([2000.0], 298.15), abs=1e-4)            # stiff modes untouched


def test_imaginary_modes_never_enter():
    assert vibrational_entropy([-50.0, 300.0], 298.15) == vibrational_entropy([300.0], 298.15)


def _state(**kw):
    s = Structure(atoms=["H", "H"], positions=np.zeros((2, 3)), labels=["H1", "H2"],
                  structure_id="h2", source_file="")
    return ElectronicState(structure=s, **kw)


def test_quasi_rrho_gibbs_corrects_harmonic_and_trusts_orca():
    soft = dict(gibbs_free_energy_hartree=-10.0, temperature_kelvin=298.15,
                frequencies_cm1=[30.0, 900.0])
    g_harm = gibbs_quasi_rrho(_state(entropy_scheme="harmonic", **soft))
    assert g_harm > -10.0                       # less entropy for the soft mode → higher G
    assert gibbs_quasi_rrho(_state(entropy_scheme="grimme", **soft)) == -10.0
    assert gibbs_quasi_rrho(_state(entropy_scheme=None, **soft)) is None   # unknown → no guess
