"""cadmia.io — readers/writers for chemical formats (geometry, electronic output, cubes)."""
from __future__ import annotations

from pathlib import Path


def structure_from_file(path: "str | Path", engine: "str | None" = None):
    """Parse a geometry from a Gaussian/ORCA output or an .xyz file → Structure.

    Detects the source: ``.xyz`` is read directly; anything else is treated as a
    quantum-chemistry output and the geometry is extracted with the engine parser
    (auto-detected from content unless *engine* is given). Atom order is preserved,
    which is what the CDFT → geometry pipeline needs.
    """
    p = Path(path)
    if p.suffix.lower() == ".xyz":
        from cadmia.io.xyz import parse_xyz
        return parse_xyz(p)
    from cadmia.agents.cdft_agent import _detect_engine
    eng = engine or _detect_engine(p)
    if eng == "orca":
        from cadmia.io.orca import parse_orca_geometry
        return parse_orca_geometry(p)
    from cadmia.io.gaussian import parse_gaussian_geometry
    return parse_gaussian_geometry(p)


def final_energy_from_file(path: "str | Path", engine: "str | None" = None):
    """Return the final electronic energy (Hartree) from a Gaussian/ORCA output.

    Reads the last ``FINAL SINGLE POINT ENERGY`` (ORCA) or ``SCF Done`` (Gaussian) line,
    which is the converged value. Returns ``None`` for ``.xyz`` inputs (no energy) or when
    no energy line is present. Engine auto-detected from content unless *engine* is given.
    """
    p = Path(path)
    if p.suffix.lower() == ".xyz":
        return None
    from cadmia.agents.cdft_agent import _detect_engine
    from cadmia.io.electronic import _parse_gaussian_energy, _parse_orca_energy
    eng = engine or _detect_engine(p)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return _parse_orca_energy(lines) if eng == "orca" else _parse_gaussian_energy(lines)
