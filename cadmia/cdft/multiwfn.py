"""Parsers for Multiwfn Conceptual DFT text output."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


_ATOM_RE = re.compile(r"^\s*(\d+)\(([A-Za-z]+)\s*\)\s+(.+?)\s*$")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?")


@dataclass
class MultiwfnAtomCDFT:
    """Condensed CDFT descriptors for one atom from Multiwfn."""

    index: int
    element: str
    q_n: Optional[float] = None
    q_np1: Optional[float] = None
    q_nm1: Optional[float] = None
    f_minus: Optional[float] = None
    f_plus: Optional[float] = None
    f_zero: Optional[float] = None
    dual_descriptor: Optional[float] = None
    electrophilicity: Optional[float] = None
    nucleophilicity: Optional[float] = None
    softness_minus: Optional[float] = None
    softness_plus: Optional[float] = None
    softness_zero: Optional[float] = None
    relative_electrophilicity: Optional[float] = None
    relative_nucleophilicity: Optional[float] = None

    @property
    def label(self) -> str:
        return f"{self.element}{self.index}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["label"] = self.label
        return payload


@dataclass
class MultiwfnCDFTResult:
    """Parsed Multiwfn CDFT report."""

    source_file: str
    atoms: List[MultiwfnAtomCDFT]
    global_descriptors: Dict[str, Dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "format": "multiwfn-cdft-report-v1",
            "source_file": self.source_file,
            "global_descriptors": self.global_descriptors,
            "atoms": [atom.to_dict() for atom in self.atoms],
        }


def _numbers(text: str) -> List[float]:
    return [float(match.replace("D", "E").replace("d", "E")) for match in _NUMBER_RE.findall(text)]


def _atom_parts(line: str):
    match = _ATOM_RE.match(line)
    if not match:
        return None
    index = int(match.group(1))
    element = match.group(2).strip()
    values = _numbers(match.group(3))
    return index, element, values


def _get_atom(atoms_by_index: dict[int, MultiwfnAtomCDFT], index: int, element: str):
    atom = atoms_by_index.get(index)
    if atom is None:
        atom = MultiwfnAtomCDFT(index=index, element=element)
        atoms_by_index[index] = atom
    return atom


def _global_key(label: str) -> str:
    key = label.strip()
    compact = key.replace(" ", "")
    lower = compact.lower()
    special = {
        "e(n)": "energy_n",
        "e(n+1)": "energy_np1",
        "e(n-1)": "energy_nm1",
        "e_homo(n)": "homo_n",
        "e_homo(n+1)": "homo_np1",
        "e_homo(n-1)": "homo_nm1",
    }
    if lower in special:
        return special[lower]

    normalized = key.lower()
    normalized = (
        normalized.replace("=", "")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "p")
        .replace("-", "m")
        .replace("/", "_")
        .replace(" ", "_")
    )
    return re.sub(r"_+", "_", normalized).strip("_")


def _parse_global_line(line: str, values: dict[str, dict[str, float]]) -> None:
    label, sep, rest = line.partition(":")
    if not sep:
        return
    normalized = _global_key(label)
    if not normalized:
        return

    hartree = re.search(
        r"([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)\s*Hartree(?:\^-?1)?",
        rest,
        re.IGNORECASE,
    )
    ev = re.search(
        r"([-+]?\d*\.?\d+(?:[DEde][-+]?\d+)?)\s*eV(?:\^-?1)?",
        rest,
        re.IGNORECASE,
    )
    if hartree or ev:
        payload = {}
        if hartree:
            payload["hartree"] = float(hartree.group(1).replace("D", "E").replace("d", "E"))
        if ev:
            payload["ev"] = float(ev.group(1).replace("D", "E").replace("d", "E"))
        values[normalized] = payload


def parse_multiwfn_cdft(path: str | Path) -> MultiwfnCDFTResult:
    """Parse Multiwfn text output for global and condensed CDFT descriptors."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Multiwfn CDFT file not found: {file_path}")

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms_by_index: dict[int, MultiwfnAtomCDFT] = {}
    global_values: dict[str, dict[str, float]] = {}
    section = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if "Hirshfeld charges" in stripped and "Fukui" in stripped:
            section = "fukui"
            continue
        if "Condensed local electrophilicity/nucleophilicity" in stripped:
            section = "local_indices"
            continue
        if "Condensed local softnesses" in stripped:
            section = "softness"
            continue
        if stripped.startswith("Atom"):
            continue

        parsed = _atom_parts(line)
        if parsed:
            index, element, values = parsed
            atom = _get_atom(atoms_by_index, index, element)
            if section == "fukui" and len(values) >= 7:
                (
                    atom.q_n,
                    atom.q_np1,
                    atom.q_nm1,
                    atom.f_minus,
                    atom.f_plus,
                    atom.f_zero,
                    atom.dual_descriptor,
                ) = values[:7]
            elif section == "local_indices" and len(values) >= 2:
                atom.electrophilicity, atom.nucleophilicity = values[:2]
            elif section == "softness" and len(values) >= 5:
                (
                    atom.softness_minus,
                    atom.softness_plus,
                    atom.softness_zero,
                    atom.relative_electrophilicity,
                    atom.relative_nucleophilicity,
                ) = values[:5]
            continue

        _parse_global_line(line, global_values)

    if not atoms_by_index and not global_values:
        raise ValueError(f"No Multiwfn CDFT data found in {file_path}")

    atoms = [atoms_by_index[index] for index in sorted(atoms_by_index)]
    return MultiwfnCDFTResult(
        source_file=str(file_path.resolve()),
        atoms=atoms,
        global_descriptors=global_values,
    )
