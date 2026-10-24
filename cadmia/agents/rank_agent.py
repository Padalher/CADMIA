"""Rank Agent — order many DFT outputs by energy and emit their geometries in one file.

Universal, engine-agnostic discrimination of already-computed Gaussian/ORCA outputs:
reads each output's final electronic energy and geometry, sorts them low→high, and writes
a single concatenated .xyz in energy order with the energy gap (kcal/mol) in every block's
comment. No optimization and no xTB — just ordering results you already have (e.g. a cheap
PBE/def2-SVP screen) so you can pick the lowest few for a high-level calculation.
"""
from __future__ import annotations

import json

import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from cadmia.agents.validation_agent import geometry_converged
from cadmia.core.sidecar import atomic_write_text, write_json_sidecar
from cadmia.core.units import HARTREE_TO_KCAL
from cadmia.io.electronic import load_electronic
from cadmia.io.vibrational import parse_orca_vibrational
from cadmia.io.xyz import parse_xyz
from cadmia.utils.chemistry import classify_skeletons, skeleton
from cadmia.utils.geometry import kabsch_max_deviation

# Same thresholds prescreen uses: the potential energy surface funnels distinct starting
# configurations into the same minimum, and different tolerances between the two stages
# would make "unique" mean two different things in one campaign.
_DEFAULT_DEDUP_RMSD = 0.20    # Å
_DEDUP_ENERGY_KCAL = 0.5      # energy tolerance for the dedup pairing


@dataclass
class RankItem:
    """Energy/geometry summary for one output file."""
    source:         str
    energy_hartree: Optional[float] = None
    rel_kcal:       Optional[float] = None    # relative to the lowest-energy output
    rank:           Optional[int] = None
    selected:       bool = True               # written to the .xyz (passed the window/top cut)
    formula:        str = ""
    n_atoms:        int = 0
    engine:         str = ""
    n_imaginary:    Optional[int] = None      # negative frequencies; None = output has no Freq
    geom_converged: Optional[bool] = None     # did the Opt converge? None = no Opt in the output
    duplicate_of:   Optional[str] = None      # source of the representative it collapsed onto
    cluster_skeleton: Optional[str] = None    # "reference"/"isomer-N"; None = not classified
    error:          Optional[str] = None


def _count_imaginary(path: Path, state, engine: str) -> Optional[int]:
    """Number of imaginary (negative) frequencies in an output.

    Returns None when the output carries no frequencies at all — a bare Opt is NOT the same as
    a verified minimum, so "no Freq" must never be reported as "0 imaginary".
    """
    if engine == "orca":
        try:
            vib = parse_orca_vibrational(path)
        except (ValueError, OSError):
            return None
        return len(vib.imaginary_modes) if vib.modes else None
    freqs = getattr(state, "frequencies_cm1", None)
    if not freqs:
        return None
    return sum(1 for f in freqs if f < 0.0)


@dataclass
class RankResult:
    """Output of the Rank Agent."""
    n_input:     int = 0
    n_ranked:    int = 0
    n_unique:    int = 0     # distinct minima behind the ranked outputs (after the RMSD dedup)
    n_unconverged: int = 0   # ranked outputs whose optimization never reached a stationary point
    n_selected:  int = 0     # unique minima kept within the window/top (written to the .xyz)
    output_xyz:  Optional[str] = None
    output_json: Optional[str] = None
    items:       List[RankItem] = field(default_factory=list)
    warnings:    List[str] = field(default_factory=list)

    @property
    def ranked(self) -> List[RankItem]:
        return sorted((it for it in self.items if it.rank is not None), key=lambda x: x.rank)

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_input": self.n_input,
                "n_ranked": self.n_ranked,
                "n_unique": self.n_unique,
                "n_unconverged": self.n_unconverged,
                "n_selected": self.n_selected,
                "output_xyz": self.output_xyz,
                "output_json": self.output_json,
                "items": [asdict(it) for it in self.items],
                "warnings": self.warnings,
            },
            indent=2,
        )


def _skeleton_classes(cluster_ref, parsed, reps, warnings) -> dict:
    """Map each unique minimum's position to the skeleton its cluster block relaxed to."""
    try:
        ref = parse_xyz(Path(cluster_ref))
    except (OSError, ValueError) as exc:
        warnings.append(f"could not read the cluster reference {cluster_ref}: {exc}; "
                        "ranking against a single global minimum instead")
        return {}
    ref_atoms = list(ref.atoms)
    ref_sig = skeleton(ref_atoms, np.asarray(ref.positions, dtype=float))[0]
    entries = [(pos, parsed[pos][1].atoms, parsed[pos][1].positions) for pos in reps]
    classes, msgs = classify_skeletons(ref_atoms, ref_sig, entries)
    warnings.extend(msgs)
    return classes


def _default_output(output: "Path | str | None") -> Optional[Path]:
    """Resolve the output path: a file stays as-is; a directory (or trailing /) gets a name."""
    if output is None:
        return None
    out = Path(output)
    if str(output).endswith(("/", "\\")) or out.is_dir():
        return out / "ranked_by_energy.xyz"
    return out


def rank(
    sources: "List[Path | str]",
    *,
    output: "Path | str | None" = None,
    window_kcal: float = 0.0,
    top_n: int = 0,
    dedup_rmsd: float = _DEFAULT_DEDUP_RMSD,
    cluster_ref: "Path | str | None" = None,
) -> RankResult:
    """Order DFT outputs by energy and write their geometries into one .xyz.

    Parameters
    ----------
    sources     : Gaussian/ORCA output files (any extension; engine auto-detected by content).
    output      : single .xyz to write (a directory writes ranked_by_energy.xyz inside it).
                  When None, nothing is written and only the ranking is returned.
    window_kcal : keep only geometries within this ΔE of the lowest (0 = no window). Trims the
                  high-energy tail at the DFT level so its outliers never reach the expensive
                  production run; the full ordering is still recorded in the JSON sidecar.
    top_n       : keep at most this many lowest-energy geometries (0 = no cap). Combined with
                  the window (a geometry is written only if it is within BOTH). Counted over
                  *unique* minima, so --top 15 means fifteen structures, not fifteen files.
    dedup_rmsd  : group outputs that relaxed to the same minimum (Å; 0 disables). Distinct
                  starting configurations routinely funnel to one minimum, and only the
                  lowest-energy member of each group reaches the .xyz.
    cluster_ref : bare-cluster .xyz, optimized at the SAME level as these outputs. Groups the
                  minima by the skeleton their cluster relaxed to and gives each group its own
                  zero, window and cap, so a more stable cluster isomer cannot take over the
                  zero and discard every adduct that kept the reference skeleton.
                  Without it the window is global, exactly as before.
    """
    files = [Path(s) for s in sources]
    result = RankResult(n_input=len(files))
    parsed: List[Tuple[float, str, RankItem]] = []   # (energy, xyz_block_placeholder, item)

    for f in files:
        item = RankItem(source=str(f))
        try:
            state, engine = load_electronic(f)
            item.engine = engine
            item.energy_hartree = state.energy_hartree
            item.formula = state.structure.formula
            item.n_atoms = state.structure.n_atoms
            item.n_imaginary = _count_imaginary(f, state, engine)
            item.geom_converged = geometry_converged(f, engine)
            if state.energy_hartree is None:
                item.error = "no energy found"
                result.warnings.append(f"{f.name}: no energy found in the output; skipped.")
            else:
                # keep the structure to render later, once ranks/ΔE are known
                parsed.append((state.energy_hartree, state.structure, item))
        except (ValueError, OSError) as exc:
            item.error = f"{type(exc).__name__}: {exc}"
            result.warnings.append(f"{f.name}: {item.error}")
        result.items.append(item)

    if not parsed:
        result.warnings.append("No output with a parseable energy; nothing to rank.")
        return result

    # An optimization that gave up still reports a final energy, and the engine still terminates
    # normally. Ranking it silently puts a non-stationary point in the ordering — and, because
    # it is still descending, possibly above where it belongs, where the window can cut it.
    result.n_unconverged = sum(1 for _, _, it in parsed if it.geom_converged is False)
    if result.n_unconverged:
        result.warnings.append(
            f"{result.n_unconverged} of {len(parsed)} output(s) did not converge geometrically: "
            "their geometry is not a stationary point and their energy is provisional, so the "
            "order around them is not trustworthy. They are ranked and marked NO-CONV rather "
            "than dropped — reoptimize them before relying on the ranking.")

    # Energies of different compositions are not comparable. Ranking a directory that mixes
    # probes silently produces ΔE in the tens of thousands of kcal/mol and a meaningless order.
    formulas = sorted({it.formula for _, _, it in parsed if it.formula})
    if len(formulas) > 1:
        result.warnings.append(
            f"outputs span {len(formulas)} different compositions ({', '.join(formulas)}); "
            "their energies are not comparable — rank one species at a time.")

    # The dedup pairs atoms by list position, so outputs of the same molecule written in a
    # different atom order can never be compared. Staying blind is the safe failure, but it
    # is not a silent one: duplicates hiding behind a reordering would go undetected.
    orders: dict = {}
    for _, structure, item in parsed:
        orders.setdefault(item.formula, set()).add(tuple(structure.atoms))
    mixed = sorted(f for f, seqs in orders.items() if len(seqs) > 1)
    if dedup_rmsd > 0 and mixed:
        result.warnings.append(
            f"{', '.join(mixed)}: outputs use more than one atom ordering; the dedup cannot "
            "compare across orderings, so some duplicates may go undetected.")

    parsed.sort(key=lambda t: t[0])
    e_min = parsed[0][0]

    # Group outputs that relaxed to the same minimum. `parsed` is sorted by energy, so the
    # first member of a group is always its lowest-energy one: the representative falls out
    # of the ordering and never has to be chosen. Only representatives are compared, so a
    # duplicate is pinned to a real minimum, not to another duplicate.
    reps: List[int] = []
    dup_rep: dict = {}
    for pos, (energy, structure, item) in enumerate(parsed):
        if dedup_rmsd > 0:
            for r in reps:
                r_energy, r_structure, r_item = parsed[r]
                if abs(energy - r_energy) * HARTREE_TO_KCAL > _DEDUP_ENERGY_KCAL:
                    continue
                if len(structure.atoms) != len(r_structure.atoms):
                    continue
                # Largest atomic displacement rather than the RMSD: an RMSD averages over
                # the whole system, so its tolerance loosens as the cluster grows and the
                # adsorbate — the part that tells two minima apart — weighs ever less.
                # Atoms pair by ELEMENT, not by list position, so the same structure
                # written in a different atom order is still recognized as the same one.
                if kabsch_max_deviation(structure.positions, r_structure.positions,
                                        structure.atoms, r_structure.atoms) <= dedup_rmsd:
                    item.duplicate_of = r_item.source
                    dup_rep[pos] = r
                    break
        if item.duplicate_of is None:
            reps.append(pos)

    # Each skeleton class is its own comparison set: its own zero, its own window, its own
    # cap. A global cap would let the class that ranks first eat the budget, which is the
    # defect being fixed rather than a separate one. A duplicate inherits the class of the
    # minimum it collapsed onto. Without a reference every minimum falls in one nameless
    # class and the cut is the global one, unchanged.
    classes = _skeleton_classes(cluster_ref, parsed, reps, result.warnings) if cluster_ref else {}
    labels: dict = {pos: classes.get(pos) for pos in reps}
    for pos, r in dup_rep.items():
        labels[pos] = labels.get(r)

    groups: dict = {}
    for pos in reps:
        groups.setdefault(labels.get(pos), []).append(pos)
    if classes:
        for label, members in groups.items():
            result.warnings.append(
                f"skeleton class {label}: {len(members)} distinct minima, ranked among themselves")

    # `reps` walks `parsed` in energy order, so each group's first member is its lowest.
    class_min = {label: parsed[members[0]][0] for label, members in groups.items()}
    kept_in = {label: 0 for label in groups}

    blocks: List[str] = []
    for rk, (energy, structure, item) in enumerate(parsed, start=1):
        item.rank = rk
        label = labels.get(rk - 1)
        item.cluster_skeleton = label
        item.rel_kcal = round((energy - class_min.get(label, e_min)) * HARTREE_TO_KCAL, 3)
        if item.duplicate_of is not None:
            item.selected = False
            continue
        # Selection cut (window/top) over unique minima. The lowest of each class (ΔE 0)
        # always passes, so no class is emptied by the cut.
        within_window = window_kcal <= 0 or item.rel_kcal <= window_kcal
        within_top = top_n <= 0 or kept_in[label] < top_n
        item.selected = bool(within_window and within_top)
        if not item.selected:
            continue
        kept_in[label] += 1
        # The .xyz comment stays minimal: ΔE + absolute energy + provenance, plus the skeleton
        # class when there is one — the next stage must not have to recompute it. The
        # imaginary-frequency count is a screening decision shown in the table, not geometry
        # metadata.
        where = f"  [{label}]" if label else ""
        comment = (f"rank {rk}  +{item.rel_kcal:.3f} kcal/mol  E={energy:.6f} Eh  "
                   f"{item.engine}  {Path(item.source).stem}{where}")
        blocks.append(structure.to_xyz(comment=comment))
    result.n_ranked = len(parsed)
    result.n_unique = len(reps)
    result.n_selected = len(blocks)

    out = _default_output(output)
    if out is not None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            # Atomic, like every other persistent write here: a partial .xyz next to
            # a sidecar that describes the whole thing is worse than no file.
            atomic_write_text(out, "".join(blocks))
            result.output_xyz = str(out)
        except OSError as exc:
            result.warnings.append(f"could not write {out}: {exc}")
        # The .xyz feeds the next engine run; the sidecar keeps what the screen decided
        # (ΔE and the imaginary-frequency count per output), which the geometries cannot carry.
        result.output_json = str(out.with_suffix(".json"))
        written, err = write_json_sidecar(result.output_json, result.to_json(),
                                          artifacts=[out] if result.output_xyz else [])
        if err:
            result.output_json = None
            result.warnings.append(err)

    return result
