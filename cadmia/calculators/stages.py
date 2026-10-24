"""Workflow stages — one standardized input per phase of the study.

Each phase of the workflow needs a *different* input: the pre-screening ranks many
xTB candidates by energy as cheaply as possible, the final optimization/frequencies
tighten convergence, and only the final electronic characterization requests the full
property set (Hirshfeld/CHELPG/Mayer/PDOS). Requesting analysis output on the early
stages just wastes compute and clutters the outputs.

A *stage* bundles the three axes a phase implies — the **task** (SP/Opt/Freq), the
**accuracy block** (RIJCOSX/def2-J/TightSCF/DefGrid3) and the **output** :mod:`preset`
— behind one short token so ``cadmia prepare <stage> …`` expands to the right input.

Stages only set sensible defaults; any explicit flag (``--task``, ``--preset``,
``--basis`` …) still overrides them. Apply one with :func:`apply_stage`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

from cadmia.calculators.params import CalcParams
from cadmia.calculators.presets import apply_preset, _merge_keywords


@dataclass(frozen=True)
class WorkflowStage:
    """A named phase of the workflow.

    task         : internal task key (see ``orca._TASK_KW`` / ``gaussian._TASK_KW``);
                   ``opt_tight`` maps to ORCA ``TightOpt`` / Gaussian ``Opt=Tight``.
    orca_accuracy: ORCA-only simple-input keywords appended for accuracy/efficiency.
    preset       : output preset name (``none``/``analysis``/``cdft``).
    basis        : basis-set override for the phase (None keeps the caller's default).
    cdft_triplet : True for the CDFT stage, which expands to N/N+1/N-1 single points.
    description  : one-line human description.
    """
    name:          str
    task:          str
    orca_accuracy: Tuple[str, ...]
    preset:        str
    basis:         "str | None" = None
    cdft_triplet:  bool = False
    description:   str = ""


# The recommended accuracy/efficiency block for hybrid DFT on these clusters.
# Frequencies are only meaningful on a geometry optimized at the SAME level, and the level
# includes the SCF threshold and the integration grid, not just the basis: running the _TIGHT
# `freq` on a geometry optimized with bare _RIJ yields spurious small imaginary modes.
# Hence the screening pair screen-opt/screen-freq shares _RIJ, and the
# production pair opt/freq shares _TIGHT.
_RIJ = ("RIJCOSX", "def2/J")
_TIGHT = _RIJ + ("TightSCF", "DefGrid3")

STAGES: Dict[str, WorkflowStage] = {
    "screen-opt": WorkflowStage(
        name="screen-opt", task="opt", orca_accuracy=_RIJ, preset="none", basis="def2-SVP",
        description="Pre-screening: quick Opt in def2-SVP to rank xTB candidates by energy.",
    ),
    "screen-freq": WorkflowStage(
        name="screen-freq", task="freq", orca_accuracy=_RIJ, preset="none", basis="def2-SVP",
        description="Pre-screening: Freq at the screen-opt level to drop saddle points before the "
                    "expensive stages.",
    ),
    "opt": WorkflowStage(
        name="opt", task="opt", orca_accuracy=_TIGHT, preset="none",
        description="Final geometry: Opt at the production level (no analysis).",
    ),
    "freq": WorkflowStage(
        name="freq", task="freq", orca_accuracy=_TIGHT, preset="none",
        description="Frequencies on the optimized geometry: confirm the minimum + thermochemistry.",
    ),
    "opt+freq": WorkflowStage(
        name="opt+freq", task="opt+freq", orca_accuracy=_TIGHT, preset="none",
        description="Combined final geometry + frequencies in one job (Opt Freq).",
    ),
    "char": WorkflowStage(
        name="char", task="sp", orca_accuracy=_TIGHT, preset="analysis",
        description="Electronic characterization: single point with Hirshfeld/CHELPG/Mayer/PDOS.",
    ),
    "cdft": WorkflowStage(
        name="cdft", task="sp", orca_accuracy=(), preset="cdft", cdft_triplet=True,
        description="CDFT triplet: N/N+1/N-1 single points for finite-difference descriptors.",
    ),
}


def get_stage(name: str) -> WorkflowStage:
    """Look up a stage by token; raise ValueError listing the choices if unknown."""
    try:
        return STAGES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown stage '{name}'; choose one of {sorted(STAGES)}")


def apply_stage(
    params: CalcParams,
    name: str,
    *,
    engine: str = "orca",
    set_preset: bool = True,
    set_basis: bool = True,
) -> Tuple[CalcParams, WorkflowStage]:
    """Return ``(params, stage)`` with the stage's task/accuracy/output/basis merged in.

    The stage always sets the task (there is no ``--task`` override). *set_preset* and
    *set_basis* let the caller keep an explicit ``--preset`` / ``--basis`` the user passed
    on the command line. The ORCA accuracy keywords are only added for ``engine == "orca"``.
    """
    stage = get_stage(name)
    updated = replace(params, task=stage.task)
    if set_basis and stage.basis:
        updated = replace(updated, basis=stage.basis)
    if engine == "orca" and stage.orca_accuracy:
        updated = replace(
            updated,
            extra_orca=_merge_keywords(updated.extra_orca, stage.orca_accuracy),
        )
    if set_preset:
        updated = apply_preset(updated, stage.preset, engine=engine)
    return updated, stage
