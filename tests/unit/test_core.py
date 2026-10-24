"""Tests for cadmia.structure.core."""
import numpy as np
import pytest

from cadmia.core import ElectronicState, Structure, VolumetricGrid


def test_structure_basic():
    s = Structure(
        atoms=["Ti", "O", "O"],
        positions=np.array([[0, 0, 0], [1.62, 0, 0], [-1.62, 0, 0]]),
        labels=["Ti1", "O2", "O3"],
        structure_id="TiO2_test",
        source_file="test.xyz",
    )
    assert s.n_atoms == 3
    assert s.formula == "O2Ti"


def test_structure_rejects_wrong_shape():
    with pytest.raises(ValueError):
        Structure(
            atoms=["Ti"],
            positions=np.array([1.0, 2.0]),
            labels=["Ti1"],
            structure_id="bad",
            source_file="",
        )


def test_structure_rejects_length_mismatch():
    with pytest.raises(ValueError):
        Structure(
            atoms=["Ti", "O"],
            positions=np.array([[0, 0, 0]]),
            labels=["Ti1", "O2"],
            structure_id="bad",
            source_file="",
        )


def test_structure_roundtrip():
    s = Structure(
        atoms=["Al", "O"],
        positions=np.array([[0, 0, 0], [1.75, 0, 0]]),
        labels=["Al1", "O2"],
        structure_id="AlO_test",
        source_file="",
    )
    s2 = Structure.from_dict(s.to_dict())
    assert s2.atoms == s.atoms
    np.testing.assert_allclose(s2.positions, s.positions)


def test_electronic_state_gap():
    s = Structure(
        atoms=["Ti"],
        positions=np.zeros((1, 3)),
        labels=["Ti1"],
        structure_id="t",
        source_file="",
    )
    es = ElectronicState(structure=s, homo_ev=-6.5, lumo_ev=-4.0)
    assert abs(es.homo_lumo_gap_ev - 2.5) < 1e-9


def test_electronic_state_gap_none_when_missing():
    s = Structure(
        atoms=["Ti"],
        positions=np.zeros((1, 3)),
        labels=["Ti1"],
        structure_id="t",
        source_file="",
    )
    es = ElectronicState(structure=s, homo_ev=-6.5)
    assert es.homo_lumo_gap_ev is None


def test_volumetric_grid_properties():
    s = Structure(
        atoms=["Ti"],
        positions=np.zeros((1, 3)),
        labels=["Ti1"],
        structure_id="t",
        source_file="",
    )
    data = np.random.uniform(-0.01, 0.01, (4, 4, 4))
    g = VolumetricGrid(
        structure=s,
        origin=np.zeros(3),
        vx=np.array([0.5292, 0, 0]),
        vy=np.array([0, 0.5292, 0]),
        vz=np.array([0, 0, 0.5292]),
        data=data,
        field_type="density",
    )
    assert g.shape == (4, 4, 4)
    assert g.vmax == pytest.approx(data.max())
    assert g.vmin == pytest.approx(data.min())


def test_volumetric_grid_rejects_2d_data():
    s = Structure(
        atoms=["Ti"],
        positions=np.zeros((1, 3)),
        labels=["Ti1"],
        structure_id="t",
        source_file="",
    )
    with pytest.raises(ValueError):
        VolumetricGrid(
            structure=s,
            origin=np.zeros(3),
            vx=np.eye(3)[0],
            vy=np.eye(3)[1],
            vz=np.eye(3)[2],
            data=np.zeros((4, 4)),
            field_type="density",
        )
