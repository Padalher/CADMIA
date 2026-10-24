"""orca_2mkl runner — export an ORCA wavefunction (.gbw) to a portable Molden file.

Shells out to ORCA's ``orca_2mkl`` (ships with ORCA; no Python dependency). The ``.gbw`` is
ORCA's *binary, version-specific* wavefunction: only ORCA and its own utilities read it. The
external population/topology tools CADMIA relies on for the ``char`` stage —

    NPA   → JANPA (default, free, Java) or NBO7 (optional)
    AIM   → Multiwfn (QTAIM: bond/ring/cage critical points, Laplacian, ellipticity)

do **not** read ``.gbw``; they read the open, text-based **Molden** format. So ``orca_2mkl name
-molden`` is the bridge that lets that post-processing run *without re-running ORCA* and without
being pinned to one ORCA build. (Cubes for ``cadmia density`` go the other way — ``orca_plot``
reads the ``.gbw`` directly — so the Molden export is only for the NPA/AIM branch.)

Caveat (why we generate it ourselves, controlled): ORCA's Molden has known conventions for the
normalization and ordering of f/g functions; JANPA and Multiwfn each have an "ORCA Molden" loader
that expects the file exactly as ``orca_2mkl`` writes it. Producing it by hand (a stray
``%output`` Molden block) risks a mismatch, so we always route through ``orca_2mkl``.

Gated on ``orca_2mkl`` being on PATH — validated with a real orca_2mkl installation, not in the dev env.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# orca_2mkl produces "<stem>.molden.input" for -molden; flags as in ORCA 6.
_FORMAT_FLAG = {"molden": "-molden", "mkl": "-mkl"}
_VALID_FORMATS = tuple(_FORMAT_FLAG)


class OrcaToMoldenError(RuntimeError):
    """orca_2mkl executable missing or the conversion failed."""


@dataclass
class MoldenExportResult:
    """One exported wavefunction file."""
    molden_path: Path
    gbw_path: Path


def orca_2mkl_available(orca_2mkl_path: str = "orca_2mkl") -> bool:
    """True when the orca_2mkl executable is on PATH."""
    return shutil.which(orca_2mkl_path) is not None


def _build_2mkl_cmd(gbw_stem: str, fmt: str = "molden", *,
                    orca_2mkl_path: str = "orca_2mkl") -> list[str]:
    """Build the orca_2mkl command line (pure function; testable offline).

    ``orca_2mkl`` takes the wavefunction *basename* (no extension — it appends ``.gbw``) and a
    format flag, and writes the output next to the ``.gbw``.
    """
    if fmt not in _FORMAT_FLAG:
        raise OrcaToMoldenError(f"unknown format {fmt!r}; expected one of {_VALID_FORMATS}.")
    return [orca_2mkl_path, gbw_stem, _FORMAT_FLAG[fmt]]


def gbw_to_molden(
    gbw_path: "str | Path",
    out_path: "str | Path | None" = None,
    *,
    orca_2mkl_path: str = "orca_2mkl",
    timeout_s: int = 600,
) -> MoldenExportResult:
    """Convert a ``.gbw`` to a Molden file with orca_2mkl. Gated on orca_2mkl being installed.

    ``orca_2mkl`` writes ``<stem>.molden.input`` next to the ``.gbw``; we move it to ``out_path``
    (default ``<stem>.molden`` alongside the input) so the extension is unambiguous for JANPA /
    Multiwfn. The file content is ORCA's Molden, unchanged.
    """
    gbw = Path(gbw_path).resolve()
    if not gbw.exists():
        raise OrcaToMoldenError(f"gbw file not found: {gbw}")
    if shutil.which(orca_2mkl_path) is None:
        raise OrcaToMoldenError(
            f"orca_2mkl executable '{orca_2mkl_path}' not found on PATH. It ships with ORCA; "
            "run this on the machine where ORCA is installed (or pass --orca-2mkl-path)."
        )

    cmd = _build_2mkl_cmd(gbw.stem, "molden", orca_2mkl_path=orca_2mkl_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=gbw.parent)
    except subprocess.TimeoutExpired as exc:
        raise OrcaToMoldenError(f"orca_2mkl timed out after {timeout_s}s on {gbw.name}") from exc
    if proc.returncode != 0:
        raise OrcaToMoldenError(f"orca_2mkl failed on {gbw.name} (rc={proc.returncode}): "
                                f"{proc.stderr.strip()[:200]}")

    produced = gbw.parent / f"{gbw.stem}.molden.input"
    if not produced.exists():
        raise OrcaToMoldenError(
            f"orca_2mkl produced no Molden file for {gbw.name} (expected {produced.name}); "
            "check the orca_2mkl version/flags."
        )
    dst = Path(out_path) if out_path else gbw.with_suffix(".molden")
    dst.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(dst)
    return MoldenExportResult(molden_path=dst, gbw_path=gbw)
