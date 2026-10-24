"""Prescreen Agent — GFN2-xTB pre-screening of adsorption starting geometries.

Sits between the Geometry Agent and the DFT stage (worth it when there are
many configurations per site). For each starting .xyz it runs a cheap xTB optimization,
deduplicates configurations that relax to the same minimum (Kabsch RMSD + energy), ranks
by xTB energy, and keeps the survivors within an energy window (capped at top_n) so only
the promising configurations are sent to expensive DFT. The xTB energies are used only to
RANK and DEDUP — never as final energetics; survivors must be re-optimized at DFT.

Two filters run before ranking, both because a bad reference poisons every relative
energy: configurations whose xTB optimization did not converge (their energy is not that
of a minimum) and nonphysical minima (collapsed atoms, absurd energies). Dropped
configurations stay in the result with their reason, so the sidecar records a verdict for
every input geometry.
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from cadmia.calculators.xtb_runner import XTBError, run_xtb_opt, xtb_available
from cadmia.core.sidecar import check_input_directory, write_json_sidecar
from cadmia.core.units import HARTREE_TO_KCAL
from cadmia.io.xyz import parse_xyz
from cadmia.utils.chemistry import classify_skeletons, skeleton
from cadmia.utils.geometry import kabsch_max_deviation

_DEFAULT_WINDOW_KCAL = 10.0   # keep configs within this of the lowest xTB energy
_DEFAULT_DEDUP_RMSD = 0.20    # Å; below this (and ~same energy) → same minimum
_DEDUP_ENERGY_KCAL = 0.5      # energy tolerance for the dedup pairing
_DEFAULT_MIN_KEEP = 2         # guarantee ≥ this many distinct survivors for DFT comparison

# Physical-sanity filter: drop nonphysical xTB minima before ranking, so a spurious
# collapsed/exploded structure cannot become the (lowest-energy) reference and poison
# the whole window. Same-composition adducts must lie within a chemically bounded range,
# so an energy hundreds of kcal/mol below the median (or fused atoms) signals garbage.
_MIN_ATOM_DIST = 0.5              # Å; no real bond is shorter → collapsed/fused atoms
_MAX_BELOW_MEDIAN_KCAL = 250.0    # energy this far BELOW the median → spurious minimum
# The reference is the batch's own median, so the filter catches isolated outliers, not a
# batch that is mostly garbage: if most configs collapsed, so did the median, and nothing
# stands out against it. The geometric half (_MIN_ATOM_DIST) is absolute and still applies.


def _min_pair_distance(coords: np.ndarray) -> float:
    """Smallest interatomic distance (Å) in a geometry; inf for < 2 atoms."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return float("inf")
    diffs = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _nonphysical_reasons(
    energy_hartree: float,
    coords: "Optional[np.ndarray]",
    median_energy: "Optional[float]",
    *,
    min_atom_dist: float = _MIN_ATOM_DIST,
    max_below_median_kcal: float = _MAX_BELOW_MEDIAN_KCAL,
) -> List[str]:
    """Reasons a relaxed config is nonphysical (empty list ⇒ looks physical)."""
    reasons: List[str] = []
    if coords is not None and np.asarray(coords).shape[0] >= 2:
        d = _min_pair_distance(coords)
        if d < min_atom_dist:
            reasons.append(f"atoms {d:.2f} Å apart (collapsed/fused)")
    if median_energy is not None:
        below_kcal = (median_energy - energy_hartree) * HARTREE_TO_KCAL
        if below_kcal > max_below_median_kcal:
            reasons.append(
                f"energy {below_kcal:.0f} kcal/mol below the median (spurious xTB minimum)"
            )
    return reasons


@dataclass
class PrescreenItem:
    """xTB result for one starting configuration."""
    source_xyz:      str
    energy_hartree:  Optional[float] = None
    rel_kcal:        Optional[float] = None      # relative to the lowest survivor
    rank:            Optional[int] = None
    kept:            bool = False
    reason:          str = ""                     # why kept/dropped
    optimized_xyz:   Optional[str] = None         # path written for survivors
    converged:       bool = True
    physical:        bool = True                   # False ⇒ dropped as a nonphysical minimum
    cluster_skeleton: Optional[str] = None        # "reference" / "isomer-N" (needs cluster_ref)
    error:           Optional[str] = None


@dataclass
class PrescreenResult:
    """Output of the Prescreen Agent."""
    n_input:    int = 0
    n_kept:     int = 0
    window_kcal: float = _DEFAULT_WINDOW_KCAL
    items:      List[PrescreenItem] = field(default_factory=list)
    warnings:   List[str] = field(default_factory=list)
    output_dir: Optional[str] = None
    output_json: Optional[str] = None       # sidecar written inside output_dir
    candidates_xyz: Optional[str] = None   # ALL good distinct minima, one file, energy order
    n_reserve:  int = 0                     # good minima kept as reserve (not selected for DFT)

    @property
    def survivors(self) -> List[PrescreenItem]:
        return [it for it in self.items if it.kept]

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_input": self.n_input,
                "n_kept": self.n_kept,
                "window_kcal": self.window_kcal,
                "output_dir": self.output_dir,
                "output_json": self.output_json,
                "candidates_xyz": self.candidates_xyz,
                "n_reserve": self.n_reserve,
                "items": [asdict(it) for it in self.items],
                "warnings": self.warnings,
            },
            indent=2,
        )


def _relabel_xyz(xyz_text: str, comment: str) -> str:
    """Return the .xyz text with its comment line (line 2) replaced."""
    lines = xyz_text.splitlines()
    if len(lines) >= 2:
        lines[1] = comment
    return "\n".join(lines).rstrip("\n") + "\n"


def _resolve_configs(configs: "Path | str | List[Path | str]") -> List[Path]:
    """Resolve a dir / single file / list of paths into a sorted list of .xyz files.

    A glob *pattern* is not expanded here: pass the expanded list (the shell already does
    this on the command line).
    """
    if isinstance(configs, (str, Path)):
        p = Path(configs)
        if p.is_dir():
            files = sorted(p.glob("*.xyz"))
            if not files:
                raise ValueError(f"No .xyz files found in {p}")
            return files
        if p.is_file():
            return [p]
        raise FileNotFoundError(f"configs not found: {p}")
    return [Path(c) for c in configs]


def _resolve_jobs(jobs: int, n_files: int) -> int:
    """How many xTB processes to run at once.

    One xTB optimization of a 30-atom cluster is far too small to scale over OpenMP
    threads, so the parallelism that pays is across *configurations*: N single-threaded
    processes, not one N-threaded process.

    jobs=0 means auto. On Linux the CPU budget is the affinity mask, not the machine's core
    count: under SLURM (or taskset) a job pinned to 8 cores on a 128-core node sees 128 from
    ``os.cpu_count()`` and would oversubscribe by 16x.
    """
    if jobs > 0:
        return max(1, min(jobs, n_files))
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                      # not Linux
        available = os.cpu_count() or 1
    return max(1, min(available, n_files))


def _classify_skeletons(
    reps: List[int],
    optimized_atoms: dict,
    optimized_coords: dict,
    cluster_ref: "Path | str",
    warnings: List[str],
) -> "Optional[dict]":
    """Map each representative to its cluster's skeleton class, or None if it cannot be done.

    Two adducts are only comparable by total energy when their substrate is the same
    substance. When the relaxation rearranges the cluster, the adduct's energy also carries
    the isomerization energy, and ranking them together lets a cluster isomer that happens
    to be more stable take over the zero and discard every adduct that kept the reference
    skeleton.

    The reference must come from the SAME level of theory as the screen: the skeleton of a
    bare cluster can itself differ between DFT and GFN2-xTB, and then an xTB screen judged
    against a DFT reference calls every adduct an isomer.
    """
    try:
        ref = parse_xyz(Path(cluster_ref))
    except (OSError, ValueError) as exc:
        warnings.append(f"could not read the cluster reference {cluster_ref}: {exc}; "
                        "ranking against a single global minimum instead")
        return None
    ref_atoms = list(ref.atoms)
    ref_sig = skeleton(ref_atoms, np.asarray(ref.positions, dtype=float))[0]

    entries = [(i, optimized_atoms[i], optimized_coords[i]) for i in reps
               if optimized_atoms.get(i) is not None and optimized_coords.get(i) is not None]
    classes, msgs = classify_skeletons(ref_atoms, ref_sig, entries)
    warnings.extend(msgs)
    return classes or None


def prescreen(
    configs: "Path | str | List[Path | str]",
    *,
    charge: int = 0,
    multiplicity: int = 1,
    gfn: int = 2,
    top_n: int = 0,
    energy_window_kcal: float = _DEFAULT_WINDOW_KCAL,
    dedup_rmsd: float = _DEFAULT_DEDUP_RMSD,
    min_keep: int = _DEFAULT_MIN_KEEP,
    output_dir: "Path | str | None" = None,
    xtb_path: str = "xtb",
    threads: int = 1,
    jobs: int = 0,
    sanity: bool = True,
    cluster_ref: "Path | str | None" = None,
    on_progress: "Optional[Callable[[int, int, str, Optional[float], Optional[str]], None]]" = None,
) -> PrescreenResult:
    """Pre-screen adsorption configs with GFN-xTB; return ranked/deduped survivors.

    Parameters
    ----------
    configs            : a dir of .xyz, a glob/file, or an explicit list.
    charge, multiplicity, gfn : passed to xTB (gfn=2 → GFN2-xTB with D4).
    top_n              : keep at most this many survivors (0 = no cap, window only).
    energy_window_kcal : keep configs within this of the lowest energy.
    dedup_rmsd         : Å threshold for treating two optimized geometries as the
                         same minimum (combined with a small energy tolerance).
    min_keep           : guarantee at least this many DISTINCT survivors, even if the
                         next-lowest minima fall outside the energy window — so DFT
                         always gets ≥min_keep candidates to compare (0 = window only).
                         Bounded by the number of distinct minima found and by top_n.
    cluster_ref        : bare-cluster geometry, from the SAME level of theory as the screen.
                         Given it, configs are grouped by the skeleton their cluster relaxed
                         to and the window, top_n and min_keep apply WITHIN each class, each
                         with its own zero — so a more stable cluster isomer cannot discard
                         the adducts that kept the reference skeleton. Omitted, a single
                         global zero is used, exactly as before.
    output_dir         : where to write the survivors' optimized .xyz (created if absent).
    jobs               : xTB optimizations to run concurrently (0 = one per available CPU).
                         Each stays single-threaded; see _resolve_jobs.
    """
    files = _resolve_configs(configs)
    result = PrescreenResult(n_input=len(files), window_kcal=energy_window_kcal)

    # If the input directory carries a sidecar from the stage that produced it, check that it
    # still describes what is there. Reading files its record does not account for is how a
    # second run with fewer sites quietly feeds the leftovers of the first one into the screen.
    if isinstance(configs, (str, Path)) and Path(configs).is_dir():
        result.warnings += check_input_directory(configs)

    out_dir: Optional[Path] = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = str(out_dir)

    # Scratch lives on the USER's disk (under the output dir, or the cwd) — never
    # in /tmp, which is small/shared — and each config's scratch is removed right
    # after we read what we need, so disk use stays bounded to ~one xTB run.
    scratch_root = (out_dir or Path.cwd()) / ".xtb_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    # ── 1. xTB optimize each configuration ──
    optimized_coords: dict[int, np.ndarray] = {}
    optimized_atoms: dict[int, tuple] = {}
    opt_xyz_text: dict[int, str] = {}
    def _optimize(f: Path):
        """Run one config. Returns (item, relaxed-or-None, warning-or-None).

        Runs on a worker thread, so it touches nothing shared: every result comes back
        through the return value and is filed by the main thread below. Each config already
        has its own scratch dir, so the runs cannot collide on disk.
        """
        item = PrescreenItem(source_xyz=str(f))
        workdir = scratch_root / Path(f).stem
        relaxed = warning = None
        try:
            res = run_xtb_opt(f, charge=charge, multiplicity=multiplicity,
                              gfn=gfn, xtb_path=xtb_path, workdir=workdir, threads=threads)
            item.energy_hartree = res.energy_hartree
            item.converged = res.converged
            opt_struct = parse_xyz(res.optimized_xyz)
            relaxed = (
                np.asarray(opt_struct.positions, dtype=float),
                tuple(opt_struct.atoms),
                # keep the optimized geometry in memory (tiny) so we can drop the scratch now
                opt_struct.to_xyz(comment=f"xtbopt {Path(f).stem}"),
            )
            if not res.converged:
                warning = f"{f.name}: xTB optimization did not converge; dropped (not a minimum)."
        except (XTBError, ValueError, OSError) as exc:
            item.error = f"{type(exc).__name__}: {exc}"
            item.reason = "xtb failed"
            warning = f"{f.name}: {item.error}"
        finally:
            if item.error is None:
                shutil.rmtree(workdir, ignore_errors=True)   # never leave xTB litter behind
            # A run that failed keeps its scratch: item.error holds the message but not
            # xTB's stdout, so deleting it makes the failure impossible to diagnose later.
        return item, relaxed, warning

    def _drop_scratch() -> None:
        """Remove the scratch root, unless a failed run left its directory to be read."""
        if any(it.error for it in result.items):
            result.warnings.append(
                f"scratch of the failed run(s) kept for diagnosis: {scratch_root}"
            )
            return
        shutil.rmtree(scratch_root, ignore_errors=True)

    collected: dict[int, PrescreenItem] = {}
    warnings_by_index: dict[int, str] = {}
    n_jobs = _resolve_jobs(jobs, len(files))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_optimize, f): i for i, f in enumerate(files)}
        for done, future in enumerate(as_completed(futures), start=1):
            i = futures[future]
            item, relaxed, warning = future.result()
            collected[i] = item
            if relaxed is not None:
                optimized_coords[i], optimized_atoms[i], opt_xyz_text[i] = relaxed
            if warning is not None:
                warnings_by_index[i] = warning
            if on_progress is not None:
                on_progress(done, len(files), files[i].name, item.energy_hartree, item.error)

    # Everything downstream indexes by position in *files*, so refile in input order —
    # completion order is whatever the scheduler happened to produce.
    result.items = [collected[i] for i in range(len(files))]
    result.warnings.extend(warnings_by_index[i] for i in sorted(warnings_by_index))

    # ── 1b. Physical-sanity filter: drop nonphysical minima BEFORE they can become the
    #        lowest-energy reference and poison the window (see _nonphysical_reasons). ──
    if sanity:
        finite = [it.energy_hartree for it in result.items if it.energy_hartree is not None]
        median_e = float(np.median(finite)) if finite else None
        for i, it in enumerate(result.items):
            if it.energy_hartree is None:
                continue
            reasons = _nonphysical_reasons(it.energy_hartree, optimized_coords.get(i), median_e)
            if reasons:
                it.physical = False
                it.reason = "nonphysical xTB minimum (" + "; ".join(reasons) + ")"
                result.warnings.append(
                    f"{Path(it.source_xyz).name}: dropped as nonphysical minimum ({'; '.join(reasons)})"
                )

    # ── 1c. Drop configurations whose xTB optimization did not converge: their energy
    #        is not that of a minimum, so it cannot be compared against the others and
    #        must not spend DFT time. They stay in `items` with the reason, and in the
    #        JSON sidecar, so the verdict on every input geometry is auditable. ──
    for it in result.items:
        if it.energy_hartree is not None and not it.converged:
            it.reason = "dropped (xTB optimization did not converge — not a minimum)"

    # ── 1d. Drop configurations that came back with an energy but no readable relaxed
    #        geometry: xTB reports the energy before the .xyz is parsed, so a truncated or
    #        unreadable xtbopt.xyz still leaves a number behind. Such a config cannot be
    #        deduplicated (there is nothing to superimpose), cannot be written out for the
    #        DFT stage, and — worst — would still set the zero of every relative energy if
    #        it happened to rank first, shifting the whole window. ──
    for i, it in enumerate(result.items):
        if it.energy_hartree is not None and i not in optimized_coords:
            it.reason = "dropped (xTB gave an energy but its relaxed geometry could not be read)"
            result.warnings.append(
                f"{Path(it.source_xyz).name}: dropped — xTB returned an energy but its "
                f"relaxed geometry could not be read, so it is not a usable candidate."
            )

    valid = [it for i, it in enumerate(result.items)
             if it.energy_hartree is not None and it.physical and it.converged
             and i in optimized_coords]
    if not valid:
        result.warnings.append(
            "No configuration produced a converged, physical xTB minimum; nothing to rank."
        )
        _drop_scratch()
        return result

    # ── 2. Dedup: group configs that relaxed to the same minimum (RMSD + energy) ──
    e_min = min(it.energy_hartree for it in valid)
    order = sorted(range(len(result.items)),
                   key=lambda i: (result.items[i].energy_hartree
                                  if result.items[i].energy_hartree is not None else float("inf")))
    representative_idx: List[int] = []
    for i in order:
        it = result.items[i]
        if (it.energy_hartree is None or not it.physical or not it.converged
                or i not in optimized_coords):
            continue
        dup_of = None
        for j in representative_idx:
            de_kcal = abs(it.energy_hartree - result.items[j].energy_hartree) * HARTREE_TO_KCAL
            if de_kcal <= _DEDUP_ENERGY_KCAL:
                # Largest atomic displacement, not the RMSD: an RMSD averages over the
                # whole system, so the probe would have to move dedup_rmsd·√(N/n_probe)
                # before the mean noticed, so the tolerance would silently loosen as the
                # cluster grows.
                # Atoms are paired by ELEMENT, not by list position: probe modes write
                # their atoms in different orders (o_down gives O,C,O where side_on gives
                # C,O,O), and pairing by position would take the same relaxed carbonate reached
                # from two modes for two species and send both to DFT.
                a_syms, b_syms = optimized_atoms.get(i), optimized_atoms.get(j)
                if a_syms is not None and b_syms is not None and len(a_syms) == len(b_syms):
                    dev = kabsch_max_deviation(optimized_coords[i], optimized_coords[j],
                                               a_syms, b_syms)
                    if dev <= dedup_rmsd:
                        dup_of = j
                        break
        if dup_of is not None:
            it.reason = f"duplicate of {Path(result.items[dup_of].source_xyz).name} (same minimum)"
        else:
            representative_idx.append(i)

    # ── 3. Rank unique reps by energy; keep within window OR to guarantee min_keep. ──
    #        A rep survives if it is within the energy window, or if it is among the
    #        lowest `guarantee` distinct minima (so DFT always gets ≥min_keep to compare),
    #        always bounded above by top_n when that cap is set.
    reps = sorted(representative_idx, key=lambda i: result.items[i].energy_hartree)
    classes = (_classify_skeletons(reps, optimized_atoms, optimized_coords,
                                   cluster_ref, result.warnings)
               if cluster_ref is not None else None)
    # Each skeleton class is its own comparison set: its own zero, its own window, its own
    # cap and guarantee. A global cap would let the class that ranks first eat the budget,
    # which is the defect being fixed rather than a separate one.
    if classes:
        for i in reps:
            result.items[i].cluster_skeleton = classes.get(i)
        groups: dict = {}
        for i in reps:
            groups.setdefault(classes.get(i), []).append(i)
        for label, members in groups.items():
            result.warnings.append(
                f"skeleton class {label}: {len(members)} distinct minima, ranked among themselves")
    else:
        groups = {None: reps}

    forced = 0
    kept_in: dict = {}
    for label, members in groups.items():
        cls_min = min(result.items[i].energy_hartree for i in members)
        guarantee = min(min_keep, top_n) if top_n > 0 else min_keep
        guarantee = max(0, min(guarantee, len(members)))
        kept_in[label] = 0
        for cls_rank, i in enumerate(members, start=1):
            it = result.items[i]
            it.rank = reps.index(i) + 1
            it.rel_kcal = round((it.energy_hartree - cls_min) * HARTREE_TO_KCAL, 3)
            within_window = it.rel_kcal <= energy_window_kcal
            within_cap = (top_n <= 0) or (kept_in[label] < top_n)
            by_guarantee = (not within_window) and (cls_rank <= guarantee)
            rank = it.rank
            where = f" in {label}" if label else ""
            if (within_window or by_guarantee) and within_cap:
                it.kept = True
                if by_guarantee:
                    it.reason = (
                        f"kept to reach ≥{guarantee} candidates{where} (rank {rank}, "
                        f"+{it.rel_kcal:.2f} kcal/mol, beyond window {energy_window_kcal})"
                    )
                    forced += 1
                else:
                    it.reason = f"kept (rank {rank}, +{it.rel_kcal:.2f} kcal/mol{where})"
                kept_in[label] += 1
                if out_dir is not None and i in opt_xyz_text:
                    dst = out_dir / (Path(it.source_xyz).stem + "_xtbopt.xyz")
                    try:
                        dst.write_text(opt_xyz_text[i], encoding="utf-8")
                        it.optimized_xyz = str(dst)
                    except OSError as exc:
                        result.warnings.append(f"could not write {dst}: {exc}")
            elif not within_window:
                it.reason = (f"dropped (+{it.rel_kcal:.2f} kcal/mol{where} "
                             f"> window {energy_window_kcal})")
            else:
                it.reason = f"dropped (beyond top {top_n}{where})"

    kept = sum(kept_in.values())
    if forced:
        result.warnings.append(
            f"kept {forced} candidate(s) beyond the {energy_window_kcal} kcal/mol window "
            f"to guarantee ≥{min_keep} distinct minima for DFT comparison (--min-keep)."
        )
    if kept < min_keep and len(reps) < min_keep:
        result.warnings.append(
            f"only {len(reps)} distinct xTB minimum/minima found (< min_keep={min_keep}); "
            f"widen sampling (more --n-sites / binding modes) to get more candidates."
        )

    # ── 4. Preserve the FULL pool of good minima so more can go to DFT later WITHOUT
    #        re-running xTB: one ranked file with every physical distinct minimum (tagged
    #        [DFT] vs reserve) + the reserves as individual .xyz. These live under pool/,
    #        NOT the main output dir, so prepare/run (which glob *.xyz there) never ingest
    #        the multi-structure file or re-prepare the reserves by accident. ──
    if out_dir is not None:
        pool_dir = out_dir / "pool"
        reserve_dir = pool_dir / "reserve"
        all_blocks: List[str] = []
        n_reserve = 0
        for rank, i in enumerate(reps, start=1):
            it = result.items[i]
            if i not in opt_xyz_text:
                continue
            tag = "to-DFT" if it.kept else f"reserve +{it.rel_kcal:.2f} kcal/mol beyond window"
            comment = (f"rank {rank}  +{it.rel_kcal:.3f} kcal/mol  "
                       f"E={it.energy_hartree:.6f} Eh  [{tag}]  {Path(it.source_xyz).stem}")
            all_blocks.append(_relabel_xyz(opt_xyz_text[i], comment))
            if not it.kept:
                reserve_dir.mkdir(parents=True, exist_ok=True)
                dst = reserve_dir / (Path(it.source_xyz).stem + "_xtbopt.xyz")
                try:
                    dst.write_text(opt_xyz_text[i], encoding="utf-8")
                    n_reserve += 1
                except OSError as exc:
                    result.warnings.append(f"could not write {dst}: {exc}")
        if all_blocks:
            pool_dir.mkdir(parents=True, exist_ok=True)
            candidates_path = pool_dir / "candidates_by_energy.xyz"
            try:
                candidates_path.write_text("".join(all_blocks), encoding="utf-8")
                result.candidates_xyz = str(candidates_path)
            except OSError as exc:
                result.warnings.append(f"could not write {candidates_path}: {exc}")
        result.n_reserve = n_reserve

    _drop_scratch()
    result.n_kept = kept

    if out_dir is not None:
        # Why every config was kept or dropped (and its xTB ΔE) survives only here: the
        # survivors' .xyz cannot carry the verdict on the ones that did not make it.
        result.output_json = str(out_dir / "prescreen.json")
        written_xyz = [i.optimized_xyz for i in result.items if i.optimized_xyz]
        if result.candidates_xyz:
            written_xyz.append(result.candidates_xyz)
        _, err = write_json_sidecar(result.output_json, result.to_json(), artifacts=written_xyz)
        if err:
            result.output_json = None
            result.warnings.append(err)

    return result
