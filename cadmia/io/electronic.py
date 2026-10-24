"""Electronic-property parsers for quantum chemistry outputs."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from cadmia.core import ElectronicState, Structure
from cadmia.core.units import AU_DIPOLE_TO_DEBYE, HARTREE_TO_EV


def _read_lines(filepath: str | Path) -> List[str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Electronic output file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _floats_from_line(line: str) -> List[float]:
    values = []
    for token in line.replace("D", "E").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _parse_gaussian_hirshfeld_charges(lines: List[str], n_atoms: int) -> Optional[List[float]]:
    """Parse Hirshfeld (Q-H) charges from Gaussian pop=hirshfeld output.

    The block header contains "Hirshfeld charges" but NOT "Fukui" (to avoid
    the Multiwfn re-analysis blocks).  Each atom line has the form:
        idx  Symbol  Q-H  S-H  Qx  Qy  Qz  Q-CM5

    Returns the last occurrence in the file (final SCF): an Opt with pop=Hirshfeld prints
    one block per optimization step, the first of them on the starting geometry.
    """
    found: Optional[List[float]] = None
    for i, line in enumerate(lines):
        if "Hirshfeld charges" not in line or "Fukui" in line:
            continue
        cursor = i + 1
        # skip the column-header line(s) until we hit actual atom data
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if stripped and stripped[0].isdigit():
                break
            cursor += 1
        parsed: List[float] = []
        while cursor < len(lines) and len(parsed) < n_atoms:
            parts = lines[cursor].split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    parsed.append(float(parts[2]))
                except ValueError:
                    break
            elif parsed:
                break
            cursor += 1
        if len(parsed) == n_atoms:
            found = parsed
    return found


def _orca_atom_charge_lines(lines: List[str], start: int, has_spin: bool) -> List[float]:
    """Read consecutive ORCA atom-charge rows starting at/after ``start``.

    Each atom row begins with the integer atom index.  The charge is the last
    numeric column for closed shell, or the second-to-last when a spin-population
    column is printed alongside it.  Parsing stops at the first non-atom line once
    at least one atom has been read — this drops the trailing ``Sum of atomic
    charges`` / ``Sum of atomic spin populations`` lines that would otherwise be
    misread as extra atoms.
    """
    charge_idx = -2 if has_spin else -1
    parsed: List[float] = []
    cursor = start
    while cursor < len(lines):
        parts = lines[cursor].split()
        if parts and parts[0].isdigit():
            try:
                parsed.append(float(parts[charge_idx]))
            except (ValueError, IndexError):
                break
        elif parsed:
            break
        cursor += 1
    return parsed


def _parse_orca_hirshfeld_charges(lines: List[str]) -> Optional[List[float]]:
    """Parse the ORCA Hirshfeld population block.

    Handles the modern ORCA ``HIRSHFELD ANALYSIS`` block (columns
    ``idx Sym charge spin``, no colon) as well as the older
    ``HIRSHFELD ATOMIC CHARGES`` layout (``idx Sym : charge``).

    Returns the last occurrence in the file (final SCF): an Opt/Freq output prints one
    block on the starting geometry of cycle 1 and another once converged.
    """
    found: Optional[List[float]] = None
    for i, line in enumerate(lines):
        up = line.upper()
        if "HIRSHFELD" not in up or ("ANALYSIS" not in up and "ATOMIC CHARGES" not in up):
            continue
        # Walk to the column header (the line naming CHARGE) to learn whether a
        # spin column is present; the modern block also has "Total integrated
        # ... density" lines in between that must be skipped.
        cursor = i + 1
        has_spin = False
        while cursor < len(lines):
            cu = lines[cursor].upper()
            if "CHARGE" in cu and "ATOM" in cu:
                has_spin = "SPIN" in cu
                cursor += 1
                break
            parts = lines[cursor].split()
            if len(parts) >= 4 and parts[2] == ":":  # old layout: rows start here
                break
            if parts and parts[0].isdigit():  # modern block without a column header
                break
            cursor += 1
        parsed = _orca_atom_charge_lines(lines, cursor, has_spin)
        if parsed:
            found = parsed
    return found


def _parse_orca_titled_charges(lines: List[str], title: str) -> Optional[List[float]]:
    """Parse an ORCA charge block headed by *title* (e.g. ``MULLIKEN ATOMIC CHARGES``).

    Handles the open-shell variant (``... AND SPIN POPULATIONS``), where the charge
    is the second-to-last column. Returns the last occurrence in the file (final SCF).
    """
    found: Optional[List[float]] = None
    for i, line in enumerate(lines):
        up = line.strip().upper()
        if not up.startswith(title):
            continue
        has_spin = "SPIN" in up
        parsed = _orca_atom_charge_lines(lines, i + 1, has_spin)
        if parsed:
            found = parsed
    return found


def _parse_gaussian_mulliken_charges(lines: List[str], n_atoms: int) -> Optional[List[float]]:
    found: Optional[List[float]] = None
    for i, line in enumerate(lines):
        if "Mulliken charges:" not in line and "Mulliken charges and spin densities:" not in line:
            continue
        parsed: List[float] = []
        cursor = i + 1
        while cursor < len(lines):
            parts = lines[cursor].split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    parsed.append(float(parts[2]))
                except ValueError:
                    break
            elif parsed:
                break
            cursor += 1
        if len(parsed) == n_atoms:
            found = parsed
    return found


def parse_all_charges(filepath: str | Path, n_atoms: int) -> Dict[str, List[float]]:
    """Return all charge schemes found in a Gaussian or ORCA output file.

    Keys: "hirshfeld", "mulliken", "loewdin", "chelpg". Only schemes whose atom count
    matches ``n_atoms`` are included. NPA is not parsed yet (requires an ORCA run with
    a working NBOEXE; pending real reference output).
    """
    lines = _read_lines(filepath)
    result: Dict[str, List[float]] = {}

    hirshfeld = _parse_gaussian_hirshfeld_charges(lines, n_atoms)
    if hirshfeld is None:
        hirshfeld = _parse_orca_hirshfeld_charges(lines)
    if hirshfeld is not None:
        result["hirshfeld"] = hirshfeld

    mulliken = _parse_orca_titled_charges(lines, "MULLIKEN ATOMIC CHARGES")
    if mulliken is None:
        mulliken = _parse_gaussian_mulliken_charges(lines, n_atoms)
    if mulliken is not None:
        result["mulliken"] = mulliken

    loewdin = _parse_orca_titled_charges(lines, "LOEWDIN ATOMIC CHARGES")
    if loewdin is not None:
        result["loewdin"] = loewdin

    chelpg = _parse_orca_titled_charges(lines, "CHELPG CHARGES")
    if chelpg is not None:
        result["chelpg"] = chelpg

    return {k: v for k, v in result.items() if len(v) == n_atoms}


# Gaussian 03 prints "Counterpoise: corrected energy =" and later versions are reported to drop
# the prefix. Matching the literal string without the colon, as this first did, found neither the
# real file on disk nor anything else: the parser fell back to the last SCF Done, which in a
# counterpoise job belongs to a fragment.
_CP_ENERGY_RE = re.compile(r"Counterpoise:?\s+corrected\s+energy\s*=", re.IGNORECASE)


def _parse_gaussian_energy(lines: List[str]) -> Optional[float]:
    """Total electronic energy from a Gaussian log, taking the last SCF Done.

    A counterpoise job breaks that rule: it runs the complex and then each fragment, with and
    without ghost atoms, printing an SCF Done for every one. The last of them belongs to a
    fragment, so "the last one wins" silently returns a piece of the system as if it were the
    whole. Gaussian prints the number that IS the whole, corrected, on its own line, and that
    one does not depend on knowing the order of the sub-calculations.
    """
    counterpoise = None
    energy = None
    for line in lines:
        if _CP_ENERGY_RE.search(line):
            match = re.search(r"=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)", line)
            if match:
                # The LAST one: an optimization prints a corrected energy per step.
                counterpoise = float(match.group(1).replace("D", "E").replace("d", "E"))
            continue
        if "SCF Done:" not in line:
            continue
        match = re.search(r"=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)", line)
        if match:
            energy = float(match.group(1).replace("D", "E").replace("d", "E"))

    if counterpoise is not None:
        return counterpoise
    if energy is not None and _has_counterpoise(lines):
        warnings.warn(
            "This Gaussian log runs a counterpoise calculation but prints no 'Counterpoise "
            "corrected energy' line, so the last SCF Done may belong to a fragment rather than "
            "to the whole system. Verify the energy before using it.",
            RuntimeWarning,
            stacklevel=2,
        )
    return energy


def _has_counterpoise(lines: List[str]) -> bool:
    """True when this log is a counterpoise calculation.

    Looks at the BODY, not the route. Gaussian wraps a long route over continuation lines that
    do not start with '#', so `counterpoise=2` at the end of one would be missed; the body
    prints "Counterpoise: doing ..." for every fragment of every such job. A false positive
    here costs one warning, a false negative costs a wrong energy reported in silence.
    """
    return any("counterpoise:" in line.lower() for line in lines)


def gaussian_energy_is_counterpoise_corrected(lines: List[str]) -> bool:
    """Whether the energy this module returns already carries the BSSE correction."""
    return any(_CP_ENERGY_RE.search(line) for line in lines)


def _parse_orca_energy(lines: List[str]) -> Optional[float]:
    energy = None
    for line in lines:
        if "FINAL SINGLE POINT ENERGY" not in line.upper():
            continue
        values = _floats_from_line(line)
        if values:
            energy = values[-1]
    return energy


# Content-signature markers for the native engines. Deliberately duplicated (not imported)
# from cadmia.agents.cdft_agent._detect_engine: `io` must not import from `agents`. Only used to tell "not this engine at all" from "this engine, but the native
# parser choked on it" — the two cases load_electronic() must not treat the same way.
# NOTE: "ORBITAL ENERGIES" is deliberately NOT included here even though _detect_engine uses
# it — it is a generic section header other programs print too (e.g. Q-Chem's "Orbital
# Energies (a.u.) and Symmetries"), which would false-positive as ORCA for outputs ORCA never
# produced. "_detect_engine" can afford that ambiguity
# because it only ever chooses between ORCA and Gaussian; this sniffer must not.
_ORCA_MARKERS = (
    "FINAL SINGLE POINT ENERGY",
    "O   R   C   A",
)
_GAUSSIAN_MARKERS = (
    "Entering Gaussian System",
    "Gaussian, Inc.",
    "This is part of the Gaussian",
)


def sniff_native_engine(path: Path) -> "tuple[bool, bool]":
    """Best-effort (is_orca, is_gaussian) from the file content, not its parsed structure.

    Read-only heuristic: a ``True`` here means the file carries that engine's banner, so a
    ``ValueError`` from its native parser is a genuinely broken/unsupported output, and the
    real failure is surfaced instead of being reported as "not an ORCA/Gaussian output".
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")[:16384]
    except OSError:
        return False, False
    upper = content.upper()
    is_orca = any(marker.upper() in upper for marker in _ORCA_MARKERS)
    is_gaussian = any(marker in content for marker in _GAUSSIAN_MARKERS)
    return is_orca, is_gaussian


def load_electronic(path: "str | Path") -> "tuple[ElectronicState, str]":
    """Parse geometry + electronic state from a Gaussian or ORCA output.

    Returns (ElectronicState, engine). Tries Gaussian first, then ORCA. This is the single
    dispatch point shared by the analysis, stability, rank and BDE agents.

    A file that carries an ORCA/Gaussian banner but whose native parser raises ``ValueError``
    is a broken/unsupported output: that error is re-raised. Any other file raises
    ``ValueError`` saying it is not an ORCA or Gaussian output.
    """
    from cadmia.io.gaussian import parse_gaussian_geometry
    from cadmia.io.orca import parse_orca_geometry

    file_path = Path(path)
    is_orca, is_gaussian = sniff_native_engine(file_path)

    try:
        structure = parse_gaussian_geometry(file_path)
        return parse_gaussian_electronic(file_path, structure), "gaussian"
    except ValueError:
        if is_gaussian:
            raise
    try:
        structure = parse_orca_geometry(file_path)
        return parse_orca_electronic(file_path, structure), "orca"
    except ValueError:
        if is_orca:
            raise

    raise ValueError(f"{file_path.name}: not an ORCA or Gaussian output")


# ── level of theory echoed by the engine (for cross-species consistency checks) ──
_BASIS_ORCA_RE  = re.compile(r"utilizes the basis:\s*(\S+)")
_BASIS_GAUSS_RE = re.compile(r"Standard basis:\s*(\S+)")
_GAUSS_DISP_RE  = re.compile(r"EmpiricalDispersion\s*=\s*(\w+)", re.IGNORECASE)
_ORCA_ECHO_RE   = re.compile(r"^\s*\|\s*\d+>\s*(!.*)$")            # echoed simple-input line
_DISP_TOKEN_RE  = re.compile(r"\b(D2|D3|D3BJ|D3ZERO|D4)\b", re.IGNORECASE)
_GAUSS_ROUTE_RE = re.compile(r"^\s*#", re.MULTILINE)


def _norm_level(token: Optional[str]) -> Optional[str]:
    """Normalize a basis/dispersion token for comparison (def2-TZVP == def2TZVP)."""
    return re.sub(r"[^a-z0-9]", "", token.lower()) if token else None


def parse_level_of_theory(path: "str | Path") -> "dict[str, Optional[str]]":
    """Best-effort ``{basis, dispersion}`` signature echoed by an ORCA/Gaussian output.

    Reads the basis set and dispersion the engine printed — the two components whose mismatch
    between species silently corrupts a subtracted energy (E_ads, stability). Values are
    normalized to lowercase alphanumerics so ``def2-TZVP`` and ``def2TZVP`` compare equal and
    ORCA's ``D3BJ`` matches Gaussian's ``GD3BJ``; a component that cannot be read is ``None``.
    The functional is deliberately not compared: its name is engine-specific (PBE0 vs
    PBE1PBE, with R/U prefixes) and a functional mismatch across a triplet is not a realistic
    failure mode, so including it would only invite false alarms. An unreadable file yields an
    all-``None`` signature rather than raising: a consistency check must never abort the
    analysis it is only advising.
    """
    try:
        lines = _read_lines(path)
    except OSError:
        return {"basis": None, "dispersion": None}
    text  = "\n".join(lines)

    m = _BASIS_ORCA_RE.search(text) or _BASIS_GAUSS_RE.search(text)
    basis = _norm_level(m.group(1)) if m else None

    dispersion: Optional[str] = None
    gd = _GAUSS_DISP_RE.search(text)
    if gd:                                             # Gaussian: EmpiricalDispersion=GD3
        tok = gd.group(1).lower()
        dispersion = tok[1:] if tok.startswith("g") else tok
    else:
        # ORCA: read the echoed `! …` simple-input line, not the whole text (D3/D4 also
        # appear in the citation banner).
        echoed = " ".join(mo.group(1) for line in lines
                          for mo in (_ORCA_ECHO_RE.match(line),) if mo)
        if echoed:
            dm = _DISP_TOKEN_RE.search(echoed)
            dispersion = dm.group(1).lower() if dm else "none"
        elif _GAUSS_ROUTE_RE.search(text):             # Gaussian route, no dispersion keyword
            dispersion = "none"

    return {"basis": basis, "dispersion": dispersion}


def engine_keywords(path: "str | Path") -> Optional[str]:
    """The keyword line the engine echoed: ORCA's `! …` lines, or the first line of the
    Gaussian route. It is the level of theory exactly as the calculation was asked for."""
    try:
        lines = _read_lines(path)
    except OSError:
        return None
    echoed = [mo.group(1).strip() for line in lines for mo in (_ORCA_ECHO_RE.match(line),) if mo]
    if echoed:
        return " ".join(echoed)
    # ponytail: first route line only; a route wrapped over several lines is cut there.
    return next((line.strip() for line in lines if line.lstrip().startswith("#")), None)


def describe_level_mismatch(named_paths: "dict[str, str | Path]") -> Optional[str]:
    """Warn if the species in *named_paths* were computed at different levels of theory.

    Compares the basis and dispersion echoed by each output. Returns a one-line warning
    naming the differing component(s) and species, or ``None`` when they agree — or when a
    component could not be read for *every* species (silence beats a false alarm). Used by
    the analysis/stability/BDE agents, which subtract energies that are only comparable at a
    common level.
    """
    levels = {name: parse_level_of_theory(p) for name, p in named_paths.items()}
    parts = []
    for comp in ("basis", "dispersion"):
        present = {n: lv[comp] for n, lv in levels.items() if lv.get(comp)}
        if len(present) == len(levels) and len(set(present.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in sorted(present.items()))
            parts.append(f"{comp} ({detail})")
    if not parts:
        return None
    return ("Species computed at different levels of theory — subtracted energies "
            "(E_ads/ΔG_ads/stability) may be off by several kcal/mol: " + "; ".join(parts) + ".")


def _parse_gaussian_spin_sq(lines: List[str]) -> Optional[float]:
    """⟨S²⟩ of the last unrestricted SCF, before annihilation; None for a restricted run.

    Gaussian prints it after every converged SCF of an open-shell calculation, so the last
    occurrence is the one that belongs to the final energy.
    """
    for line in reversed(lines):
        match = re.search(r"S\*\*2 before annihilation\s+(-?\d+\.\d+)", line)
        if match:
            return float(match.group(1))
    return None


def parse_gaussian_electronic(
    filepath: str | Path,
    structure: Structure,
    prefer_hirshfeld: bool = True,
) -> ElectronicState:
    """Read Gaussian HOMO/LUMO, charges, and dipole moment.

    When ``prefer_hirshfeld`` is True (default), Hirshfeld charges (Q-H) are
    used when the ``pop=hirshfeld`` keyword was present in the Gaussian run.
    Falls back to Mulliken charges when Hirshfeld data is absent.
    """
    path = Path(filepath)
    lines = _read_lines(path)

    occupied = []
    virtual = []
    beta_occupied = []
    beta_virtual = []
    in_eig_block = False   # an eigenvalue dump is a contiguous run of these lines
    mulliken_charges: Optional[List[float]] = None
    dipole = None
    energy_hartree = _parse_gaussian_energy(lines)

    zpe_correction: Optional[float] = None
    thermal_energy_corr: Optional[float] = None
    thermal_enthalpy_corr: Optional[float] = None
    thermal_gibbs_corr: Optional[float] = None
    electronic_zpe_energy: Optional[float] = None
    enthalpy_ha: Optional[float] = None
    gibbs_ha: Optional[float] = None
    temperature_k: Optional[float] = None
    frequencies: List[float] = []

    _thermo_re = re.compile(r'=\s*([-\d.]+)')

    for index, line in enumerate(lines):
        is_alpha_occ = "Alpha  occ. eigenvalues" in line or "Alpha occ. eigenvalues" in line
        is_alpha_virt = "Alpha virt. eigenvalues" in line
        is_beta_occ = "Beta  occ. eigenvalues" in line or "Beta occ. eigenvalues" in line
        is_beta_virt = "Beta virt. eigenvalues" in line
        is_eig = is_alpha_occ or is_alpha_virt or is_beta_occ or is_beta_virt
        # A new dump always starts with "Alpha occ"; reset so later dumps (opt+freq,
        # pop=full) replace earlier ones instead of concatenating onto them.
        if is_alpha_occ and not in_eig_block:
            occupied, virtual, beta_occupied, beta_virtual = [], [], [], []
        if is_alpha_occ:
            occupied.extend(_floats_from_line(line.split("--", 1)[-1]))
        elif is_alpha_virt:
            virtual.extend(_floats_from_line(line.split("--", 1)[-1]))
        elif is_beta_occ:
            beta_occupied.extend(_floats_from_line(line.split("--", 1)[-1]))
        elif is_beta_virt:
            beta_virtual.extend(_floats_from_line(line.split("--", 1)[-1]))
        in_eig_block = is_eig

        if "Mulliken charges:" in line or "Mulliken charges and spin densities:" in line:
            parsed = []
            cursor = index + 1
            while cursor < len(lines):
                parts = lines[cursor].split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        parsed.append(float(parts[2]))  # col 3 = charge (open-shell has 4 cols)
                    except ValueError:
                        pass
                elif parsed:
                    break
                cursor += 1
            if len(parsed) == structure.n_atoms:
                mulliken_charges = parsed

        if "Zero-point correction=" in line:
            m = _thermo_re.search(line)
            if m:
                zpe_correction = float(m.group(1))
        elif "Thermal correction to Gibbs Free Energy=" in line:
            m = _thermo_re.search(line)
            if m:
                thermal_gibbs_corr = float(m.group(1))
        elif "Thermal correction to Enthalpy=" in line:
            m = _thermo_re.search(line)
            if m:
                thermal_enthalpy_corr = float(m.group(1))
        elif "Thermal correction to Energy=" in line:
            m = _thermo_re.search(line)
            if m:
                thermal_energy_corr = float(m.group(1))
        elif "Sum of electronic and zero-point Energies=" in line:
            m = _thermo_re.search(line)
            if m:
                electronic_zpe_energy = float(m.group(1))
        elif "Sum of electronic and thermal Enthalpies=" in line:
            m = _thermo_re.search(line)
            if m:
                enthalpy_ha = float(m.group(1))
        elif "Sum of electronic and thermal Free Energies=" in line:
            m = _thermo_re.search(line)
            if m:
                gibbs_ha = float(m.group(1))
        elif "Temperature" in line and "Kelvin" in line and temperature_k is None:
            m = re.search(r'Temperature\s+([\d.]+)\s+Kelvin', line)
            if m:
                temperature_k = float(m.group(1))
        # "Frequencies --" is also a substring of hpmodes' "Frequencies ---", so matching it
        # as a substring counted every frequency twice in a job that used hpmodes.
        elif re.match(r"\s*Frequencies\s+--\s", line):
            frequencies.extend(_floats_from_line(line.split("--", 1)[-1]))

        if "Dipole moment" in line:
            for candidate in lines[index : min(index + 4, len(lines))]:
                match = re.search(
                    r"X=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)\s+"
                    r"Y=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)\s+"
                    r"Z=\s*([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)",
                    candidate,
                )
                if match:
                    dipole = np.array(
                        [float(value.replace("D", "E")) for value in match.groups()],
                        dtype=float,
                    )
                    break

    # For open-shell: HOMO is the highest energy occupied orbital across both spin channels.
    alpha_homo = occupied[-1] if occupied else None
    beta_homo = beta_occupied[-1] if beta_occupied else None
    if alpha_homo is not None and beta_homo is not None:
        homo_ha = max(alpha_homo, beta_homo)
    else:
        homo_ha = alpha_homo
    homo_ev = homo_ha * HARTREE_TO_EV if homo_ha is not None else None

    alpha_lumo = virtual[0] if virtual else None
    beta_lumo = beta_virtual[0] if beta_virtual else None
    if alpha_lumo is not None and beta_lumo is not None:
        lumo_ev = min(alpha_lumo, beta_lumo) * HARTREE_TO_EV
    else:
        lumo_ev = alpha_lumo * HARTREE_TO_EV if alpha_lumo is not None else None

    hirshfeld = _parse_gaussian_hirshfeld_charges(lines, structure.n_atoms) if prefer_hirshfeld else None
    if hirshfeld is not None:
        charges, scheme = hirshfeld, "hirshfeld"
    else:
        charges, scheme = mulliken_charges, "mulliken"

    all_alpha_ev = [e * HARTREE_TO_EV for e in occupied + virtual] if occupied or virtual else None
    all_beta_ev = [e * HARTREE_TO_EV for e in beta_occupied + beta_virtual] if beta_occupied or beta_virtual else None

    def _ev(value):
        return value * HARTREE_TO_EV if value is not None else None

    return ElectronicState(
        structure=structure,
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        energy_hartree=energy_hartree,
        charges=charges,
        charge_scheme=scheme,
        dipole_debye=dipole,
        source_file=str(path.resolve()),
        spin_sq=_parse_gaussian_spin_sq(lines),
        alpha_eigenvalues_ev=all_alpha_ev,
        beta_eigenvalues_ev=all_beta_ev,
        homo_alpha_ev=_ev(alpha_homo),
        lumo_alpha_ev=_ev(alpha_lumo),
        homo_beta_ev=_ev(beta_homo),
        lumo_beta_ev=_ev(beta_lumo),
        zpe_correction_hartree=zpe_correction,
        thermal_energy_correction_hartree=thermal_energy_corr,
        thermal_enthalpy_correction_hartree=thermal_enthalpy_corr,
        thermal_gibbs_correction_hartree=thermal_gibbs_corr,
        electronic_zpe_energy_hartree=electronic_zpe_energy,
        enthalpy_hartree=enthalpy_ha,
        gibbs_free_energy_hartree=gibbs_ha,
        temperature_kelvin=temperature_k,
        frequencies_cm1=frequencies if frequencies else None,
        entropy_scheme="harmonic" if gibbs_ha is not None else None,
        rotational_symmetry_number=_last_int(lines, r"Rotational symmetry number\s+(\d+)"),
    )


def _parse_orca_orbital_energies(lines: List[str]):
    """Parse the full ORCA ORBITAL ENERGIES block.

    Returns (alpha_ev, beta_ev, n_alpha_occ, n_beta_occ). For closed shell, beta_ev is
    None. Open-shell blocks are split by "SPIN UP"/"SPIN DOWN ORBITALS" sub-headers.
    Rows have the form ``NO OCC E(Eh) E(eV)``.

    Returns the last block in the file (final SCF): an Opt/Freq output prints one block
    on the starting geometry of cycle 1 and another once converged, and the frontier
    orbitals must match the geometry the energy refers to.
    """
    result: Tuple[Optional[List[float]], Optional[List[float]], int, int] = (None, None, 0, 0)
    for i, line in enumerate(lines):
        if "ORBITAL ENERGIES" not in line:
            continue
        alpha: List[float] = []
        beta: List[float] = []
        na = nb = 0
        channel = "alpha"
        started = False
        cursor = i + 1
        while cursor < len(lines):
            row = lines[cursor]
            up = row.upper()
            if "SPIN DOWN" in up:
                channel = "beta"
                started = False   # the SPIN DOWN block repeats the NO/OCC header
                cursor += 1
                continue
            if "SPIN UP" in up:
                channel = "alpha"
                cursor += 1
                continue
            parts = row.split()
            if not parts:         # blank lines separate the alpha and beta blocks
                cursor += 1
                continue
            if parts[0] == "NO" and "OCC" in parts:
                cursor += 1
                continue
            if len(parts) >= 4:
                try:
                    occ = float(parts[1]); e_ev = float(parts[3])
                except ValueError:
                    if started or alpha:
                        break
                    cursor += 1
                    continue
                started = True
                if channel == "beta":
                    beta.append(e_ev); nb += 1 if occ > 0.0 else 0
                else:
                    alpha.append(e_ev); na += 1 if occ > 0.0 else 0
            elif started or alpha:
                break
            else:
                cursor += 1
                continue
            cursor += 1
        if alpha:
            result = (alpha, (beta or None), na, nb)
    return result


def _frontier_from_eigenvalues(
    alpha: Optional[List[float]],
    beta: Optional[List[float]],
    na: int,
    nb: int,
) -> tuple[Optional[float], Optional[float]]:
    """HOMO/LUMO (eV) from the α/β eigenvalue lists of _parse_orca_orbital_energies.

    Open-shell rule: HOMO = max(α-HOMO, β-HOMO), LUMO = min(α-LUMO, β-LUMO).
    Returns (None, None) when no eigenvalues were parsed, and tolerates blocks with
    no printed virtuals (LUMO None) or no occupied orbitals (HOMO None).
    """
    if not alpha:
        return None, None
    homo = alpha[na - 1] if na > 0 else None
    lumo = alpha[na] if na < len(alpha) else None
    if beta:
        b_homo = beta[nb - 1] if nb > 0 else None
        b_lumo = beta[nb] if nb < len(beta) else None
        if b_homo is not None:
            homo = b_homo if homo is None else max(homo, b_homo)
        if b_lumo is not None:
            lumo = b_lumo if lumo is None else min(lumo, b_lumo)
    return homo, lumo


# ORCA thermochemistry block (printed by a Freq run): "<label> ... <value> Eh".
_ORCA_THERMO_PATTERNS = {
    "zpe_correction_hartree":        r"Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh",
    "enthalpy_hartree":              r"Total Enthalpy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh",
    "gibbs_free_energy_hartree":     r"Final Gibbs free energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh",
    "entropy_term_hartree":          r"Final entropy term\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh",
}


def _last_int(lines: List[str], pattern: str) -> Optional[int]:
    """Integer of the last line matching ``pattern`` (one capture group), or None."""
    rx = re.compile(pattern)
    for line in reversed(lines):
        m = rx.search(line)
        if m:
            return int(m.group(1))
    return None


def _parse_orca_thermochemistry(lines: List[str]) -> Dict[str, Union[float, str, None]]:
    """Parse the ORCA THERMOCHEMISTRY block (present only after a frequency calc).

    Returns enthalpy, Gibbs free energy, the entropy term (T·S), ZPE and temperature
    in Hartree/Kelvin. Missing values stay None so callers can degrade gracefully
    (e.g. report ΔE_ads only when frequencies were not computed).

    Only the LAST block is read, to match the energy and the orbitals: a re-run or a
    concatenated output holds several, and mixing G from one geometry with E from another
    is a silent error. Scanning inside a single block (instead of taking the last match of
    each field over the whole file) also keeps a truncated final block from borrowing
    fields off the previous one — a missing value stays None, which the callers handle.
    """
    starts = [i for i, line in enumerate(lines) if "THERMOCHEMISTRY" in line.upper()]
    if not starts:
        return {}
    text = "\n".join(lines[starts[-1]:])
    out: Dict[str, Union[float, str, None]] = {}
    for field_name, pattern in _ORCA_THERMO_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            out[field_name] = float(m.group(1))
    mt = re.search(r"Temperature\s+\.\.\.\s+([\d.]+)\s*K", text)
    if mt:
        out["temperature_kelvin"] = float(mt.group(1))
    # The scheme is read, never assumed: without either marker it stays unknown (None).
    mq = re.search(r"Quasi RRHO\s+\.\.\.\s+(True|False)", text)
    if mq:
        out["entropy_scheme"] = "grimme" if mq.group(1) == "True" else "harmonic"
    elif "QRRHO of S. Grimme" in text:
        out["entropy_scheme"] = "grimme"
    return out


def _parse_orca_dipole(lines: List[str]) -> Optional[np.ndarray]:
    """Return the converged total dipole moment (Debye) from an ORCA output.

    ORCA prints ``Total Dipole Moment`` as the X/Y/Z vector in atomic units
    (e·Bohr); the last block in the file corresponds to the converged density.
    The vector is converted to Debye so it matches the Gaussian parser, which
    already stores ``dipole_debye``.
    """
    vector: Optional[np.ndarray] = None
    for line in lines:
        if "Total Dipole Moment" in line:
            values = _floats_from_line(line.split(":", 1)[-1])
            if len(values) >= 3:
                vector = np.array(values[:3], dtype=float) * AU_DIPOLE_TO_DEBYE
    return vector


def _parse_orca_spin_sq(lines: List[str]) -> Optional[float]:
    """⟨S²⟩ of the converged SCF, or None for a closed-shell run that never prints it.

    The Fukui functions are built from the N±1 states, which are open shell; if those
    converge to a spin-contaminated solution, every condensed descriptor derived from them
    is degraded and nothing downstream can tell. Reading it here is what lets the CDFT step
    say so.
    """
    for line in reversed(lines):
        if "<S**2>" in line:
            vals = _floats_from_line(line.split("<S**2>")[-1])
            if vals:
                return vals[-1]
    return None


def parse_orca_electronic(
    filepath: str | Path,
    structure: Structure,
    prefer_hirshfeld: bool = True,
) -> ElectronicState:
    """Read ORCA HOMO/LUMO, charges, energy, and (if a Freq run) thermochemistry.

    When ``prefer_hirshfeld`` is True (default), Hirshfeld charges are used
    when the ORCA run included ``%output Print[P_Hirshfeld] 1 end``.
    Falls back to Mulliken charges when Hirshfeld data is absent.
    """
    path = Path(filepath)
    lines = _read_lines(path)

    mulliken_charges: Optional[List[float]] = None
    energy_hartree = _parse_orca_energy(lines)
    thermo = _parse_orca_thermochemistry(lines)

    for index, line in enumerate(lines):
        if "MULLIKEN ATOMIC CHARGES" in line.upper():
            # Open-shell blocks are titled "...AND SPIN POPULATIONS" and carry a
            # trailing spin column; the charge is then the second-to-last field.
            has_spin = "SPIN" in line.upper()
            parsed = _orca_atom_charge_lines(lines, index + 1, has_spin)
            if len(parsed) == structure.n_atoms:
                mulliken_charges = parsed

    # HOMO/LUMO come from the full α/β eigenvalue lists (open-shell rule
    # HOMO = max(α,β), LUMO = min(α,β)); never from an alpha-only inline scan.
    alpha_ev, beta_ev, n_alpha_occ, n_beta_occ = _parse_orca_orbital_energies(lines)
    homo_ev, lumo_ev = _frontier_from_eigenvalues(
        alpha_ev, beta_ev, n_alpha_occ, n_beta_occ,
    )
    homo_alpha, lumo_alpha = _frontier_from_eigenvalues(alpha_ev, None, n_alpha_occ, 0)
    homo_beta, lumo_beta = _frontier_from_eigenvalues(beta_ev, None, n_beta_occ, 0)

    if homo_ev is not None and lumo_ev is not None and homo_ev >= lumo_ev:
        warnings.warn(
            f"ORCA parser: HOMO ({homo_ev:.4f} eV) >= LUMO ({lumo_ev:.4f} eV). "
            "Orbital ordering may be incorrect — verify the output file.",
            RuntimeWarning,
            stacklevel=2,
        )

    hirshfeld = _parse_orca_hirshfeld_charges(lines) if prefer_hirshfeld else None
    if hirshfeld is not None and len(hirshfeld) == structure.n_atoms:
        charges, scheme = hirshfeld, "hirshfeld"
    else:
        charges, scheme = mulliken_charges, "mulliken"

    return ElectronicState(
        structure=structure,
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        energy_hartree=energy_hartree,
        charges=charges,
        charge_scheme=scheme,
        dipole_debye=_parse_orca_dipole(lines),
        source_file=str(path.resolve()),
        spin_sq=_parse_orca_spin_sq(lines),
        alpha_eigenvalues_ev=alpha_ev,
        beta_eigenvalues_ev=beta_ev,
        homo_alpha_ev=homo_alpha,
        lumo_alpha_ev=lumo_alpha,
        homo_beta_ev=homo_beta,
        lumo_beta_ev=lumo_beta,
        zpe_correction_hartree=thermo.get("zpe_correction_hartree"),
        enthalpy_hartree=thermo.get("enthalpy_hartree"),
        gibbs_free_energy_hartree=thermo.get("gibbs_free_energy_hartree"),
        entropy_term_hartree=thermo.get("entropy_term_hartree"),
        temperature_kelvin=thermo.get("temperature_kelvin"),
        entropy_scheme=thermo.get("entropy_scheme"),
        rotational_symmetry_number=_last_int(lines, r"Symmetry Number:\s+(\d+)"),
    )


# ─────────────────────────────────────────────
# ORCA per-MO orbital populations (PDOS source)
# ─────────────────────────────────────────────

_ATOM_LABEL_RE = re.compile(r"^(\d+)([A-Za-z]+)$")


def parse_orca_mo_populations(filepath: str | Path) -> Dict[str, Optional[List[Dict]]]:
    """Parse the ORCA ``LOEWDIN ORBITAL POPULATIONS PER MO`` block.

    Requires the run to include ``%output Print[P_OrbPopMO_L] 1 end``. Returns
    ``{"alpha": [...], "beta": [...] | None}`` where each MO is a dict with
    ``index``, ``energy_ev``, ``occ`` and ``contributions`` mapping
    ``(atom_index, element, orbital)`` to the printed percentage. Contributions
    below ORCA's print threshold (0.1%) are absent, so per-MO percentages may
    sum to slightly less than 100.
    """
    lines = _read_lines(filepath)
    alpha: List[Dict] = []
    beta: List[Dict] = []
    current = alpha
    header = None
    for i, line in enumerate(lines):
        if "LOEWDIN ORBITAL POPULATIONS PER MO" in line:
            header = i      # last occurrence wins (final SCF)
    if header is None:
        return {"alpha": None, "beta": None}

    i = header + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("THRESHOLD") or set(stripped) == {"-"}:
            i += 1
            continue
        if stripped.upper() == "SPIN UP":
            current = alpha
            i += 1
            continue
        if stripped.upper() == "SPIN DOWN":
            current = beta
            i += 1
            continue
        parts = stripped.split()
        if not all(p.isdigit() for p in parts):
            break               # next output section
        try:
            indices = [int(p) for p in parts]
            energies = [float(x) for x in lines[i + 1].split()]
            occs = [float(x) for x in lines[i + 2].split()]
        except (ValueError, IndexError):
            break
        chunk = [
            {"index": k, "energy_ev": e * HARTREE_TO_EV, "occ": o, "contributions": {}}
            for k, e, o in zip(indices, energies, occs)
        ]
        i += 4                  # indices, energies, occupations, dashed separator
        while i < len(lines):
            rparts = lines[i].split()
            if len(rparts) < 2:
                break
            m = _ATOM_LABEL_RE.match(rparts[0])
            if not m:
                break
            atom_idx, element = int(m.group(1)), m.group(2)
            orbital = rparts[1]
            try:
                values = [float(v) for v in rparts[2:2 + len(indices)]]
            except ValueError:
                break
            for mo, value in zip(chunk, values):
                if value:
                    mo["contributions"][(atom_idx, element, orbital)] = value
            i += 1
        current.extend(chunk)

    return {"alpha": alpha or None, "beta": beta or None}
