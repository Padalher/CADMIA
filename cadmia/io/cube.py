"""Gaussian cube parser."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cadmia.core import Structure, VolumetricGrid
from cadmia.core.elements import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER
from cadmia.core.units import BOHR_TO_ANG
from cadmia.io.xyz import get_atom_labels


def _parse_axis(line: str, name: str) -> tuple[int, np.ndarray]:
    """Return the signed voxel count and the raw (unconverted) step vector.

    The sign of the voxel count carries the grid units (positive → Bohr,
    negative → Angstrom); the caller applies the conversion accordingly.
    """
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid cube {name} axis line: {line!r}")
    try:
        count = int(parts[0])
        vector = np.array([float(value) for value in parts[1:4]], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Invalid cube {name} axis values: {line!r}") from exc
    return count, vector


def parse_cube(filepath: str | Path, field_type: str = "other") -> VolumetricGrid:
    """
    Read a Gaussian .cube file and return a VolumetricGrid.

    Molecular-orbital cubes with a negative atom count are supported by
    skipping the extra orbital-index line after the atom block.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cube file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError(f"Cube file too short: {path}")

    parts = lines[2].split()
    if len(parts) < 4:
        raise ValueError("Invalid cube atom-count/origin line.")
    try:
        nat_raw = int(parts[0])
        origin = np.array([float(value) for value in parts[1:4]], dtype=float)
    except ValueError as exc:
        raise ValueError("Invalid cube atom-count/origin values.") from exc

    n_atoms = abs(nat_raw)
    nx_s, vx = _parse_axis(lines[3], "x")
    ny_s, vy = _parse_axis(lines[4], "y")
    nz_s, vz = _parse_axis(lines[5], "z")
    # Positive voxel counts → coordinates in Bohr (convert to Å); negative → already Å.
    to_ang = BOHR_TO_ANG if nx_s >= 0 else 1.0
    nx, ny, nz = abs(nx_s), abs(ny_s), abs(nz_s)
    origin *= to_ang
    vx, vy, vz = vx * to_ang, vy * to_ang, vz * to_ang

    atom_start = 6
    atom_end = atom_start + n_atoms
    if len(lines) < atom_end:
        raise ValueError(f"Cube file expected {n_atoms} atom lines.")

    atoms = []
    positions = []
    for line_number, line in enumerate(lines[atom_start:atom_end], start=atom_start + 1):
        atom_parts = line.split()
        if len(atom_parts) < 5:
            raise ValueError(f"Invalid cube atom line {line_number}: {line!r}")
        try:
            atomic_number = int(float(atom_parts[0]))
            xyz_bohr = [float(value) for value in atom_parts[2:5]]
        except ValueError as exc:
            raise ValueError(f"Invalid cube atom values at line {line_number}") from exc
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)
        if symbol is None:
            raise ValueError(f"Unknown atomic number in cube: {atomic_number}")
        atoms.append(symbol)
        positions.append(np.array(xyz_bohr, dtype=float) * to_ang)

    data_start = atom_end
    if nat_raw < 0:
        if data_start >= len(lines):
            raise ValueError("Cube MO file missing orbital-index line.")
        # That line starts with how many orbitals the file holds. With more than one their
        # values are INTERLEAVED voxel by voxel, so reading the first grid-worth of numbers
        # returns a mixture of orbitals and nothing says so.
        try:
            n_orbitals = int(lines[data_start].split()[0])
        except (IndexError, ValueError):
            n_orbitals = 1
        if n_orbitals != 1:
            raise ValueError(
                f"This cube holds {n_orbitals} orbitals interleaved; CADMIA reads one per file. "
                "Regenerate it with a single orbital."
            )
        data_start += 1

    values = []
    for line in lines[data_start:]:
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(f"Invalid cube data value: {token}") from exc

    expected = nx * ny * nz
    if len(values) < expected:
        raise ValueError(
            f"Cube data has {len(values)} values, expected at least {expected}."
        )
    data = np.array(values[:expected], dtype=float).reshape((nx, ny, nz))

    structure = Structure(
        atoms=atoms,
        positions=np.array(positions, dtype=float),
        labels=get_atom_labels(atoms),
        structure_id=path.stem,
        source_file=str(path.resolve()),
    )
    return VolumetricGrid(
        structure=structure,
        origin=origin,
        vx=vx,
        vy=vy,
        vz=vz,
        data=data,
        field_type=field_type,
        source_file=str(path.resolve()),
    )


def _ang_to_bohr(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=float) / BOHR_TO_ANG


def write_cube(grid: VolumetricGrid, filepath: str | Path, comment: str = "") -> Path:
    """Write a VolumetricGrid to a Gaussian .cube file (Bohr, positive voxel counts).

    Inverse of `parse_cube`: positions/origin/step vectors are stored in Bohr (CADMIA keeps
    them in Angstrom internally), with positive voxel counts so other tools read Bohr units.
    Round-trips with `parse_cube`. Used for the density/Fukui/dual/Δρ maps written by
    `cadmia density`.
    """
    path = Path(filepath)
    nx, ny, nz = grid.shape
    origin_b = _ang_to_bohr(grid.origin)
    vx_b, vy_b, vz_b = _ang_to_bohr(grid.vx), _ang_to_bohr(grid.vy), _ang_to_bohr(grid.vz)
    atoms = grid.structure.atoms
    positions_b = _ang_to_bohr(np.asarray(grid.structure.positions, dtype=float))

    title = comment or grid.field_type or "cube"
    lines = [
        f"CADMIA cube: {title}",
        f"field_type={grid.field_type}",
        f"{len(atoms):5d} {origin_b[0]:12.6f} {origin_b[1]:12.6f} {origin_b[2]:12.6f}",
        f"{nx:5d} {vx_b[0]:12.6f} {vx_b[1]:12.6f} {vx_b[2]:12.6f}",
        f"{ny:5d} {vy_b[0]:12.6f} {vy_b[1]:12.6f} {vy_b[2]:12.6f}",
        f"{nz:5d} {vz_b[0]:12.6f} {vz_b[1]:12.6f} {vz_b[2]:12.6f}",
    ]
    for sym, pos in zip(atoms, positions_b):
        z = SYMBOL_TO_ATOMIC_NUMBER.get(sym)
        if z is None:
            raise ValueError(f"cannot write cube: unknown element symbol {sym!r}")
        lines.append(f"{z:5d} {float(z):12.6f} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

    # Gaussian cube data order: x outer, y middle, z inner; 6 values per line.
    flat = np.asarray(grid.data, dtype=float).reshape(-1)
    for start in range(0, flat.size, 6):
        chunk = flat[start:start + 6]
        lines.append("".join(f"{v:13.5e}" for v in chunk))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
