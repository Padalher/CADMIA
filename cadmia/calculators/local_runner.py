"""Local execution backend — run DFT jobs sequentially on this machine.

Workstation counterpart of hpc_runner: no SSH, no SLURM. Jobs run one at a
time (a serial queue), each using the cores/memory requested in its input file
(%pal/%maxcore for ORCA, %nprocshared/%mem for Gaussian), so a single job can
take the whole machine without oversubscribing it.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


class LocalRunnerError(RuntimeError):
    pass


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """SIGKILL the whole process group led by *proc*.

    ORCA with ``%pal`` forks MPI ranks; killing only ``proc.pid`` leaves them orphaned,
    consuming the workstation's CPU/RAM after a timeout. Because the child is started in a
    new session (`start_new_session=True`), it leads its own group, so one `killpg` reaches
    every descendant. Falls back to killing the direct child if the group is already gone.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


# Binaries smaller than this are not a QC engine (e.g. /usr/bin/orca is the
# GNOME screen reader on many Linux desktops).
_MIN_ENGINE_SIZE = 5_000_000


def find_engine(engine: str, explicit_path: Optional[str] = None) -> Path:
    """Locate the ORCA or Gaussian binary; raise LocalRunnerError with guidance."""
    if explicit_path:
        path = Path(explicit_path)
        if path.is_dir():
            path = path / ("orca" if engine == "orca" else "g16")
        if not path.exists():
            raise LocalRunnerError(f"{engine} binary not found at {path}.")
        return path.resolve()

    name = "orca" if engine == "orca" else "g16"
    found = shutil.which(name)
    if not found:
        raise LocalRunnerError(
            f"'{name}' not found in PATH. Pass --engine-path /path/to/{name} "
            f"or add it to PATH."
        )
    path = Path(found).resolve()
    if engine == "orca" and path.stat().st_size < _MIN_ENGINE_SIZE:
        raise LocalRunnerError(
            f"{path} looks like the GNOME screen reader, not ORCA QC. "
            "Pass --engine-path with the real ORCA installation directory."
        )
    return path


def run_job(
    input_file: Path,
    engine: str,
    output_file: Path,
    *,
    engine_path: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> Tuple[bool, float]:
    """Run one input to completion; return (succeeded, wall_time_s).

    ORCA is invoked with its absolute path (required for %pal parallel runs).
    stdout goes to *output_file*; the job runs in the input's directory so
    auxiliary files (.gbw, .densities, .chk) land next to it.
    """
    binary = find_engine(engine, engine_path)
    workdir = input_file.parent.resolve()

    if engine == "orca":
        cmd: list[str] = [str(binary), input_file.name]
        stdin = None
    else:
        cmd = [str(binary)]
        stdin = open(input_file, "rb")

    start = time.monotonic()
    try:
        with open(output_file, "wb") as out:
            # start_new_session=True puts the engine in its own process group so a timeout
            # can reap the whole tree (subprocess.run's own timeout kill would only hit the
            # direct child, orphaning ORCA's MPI ranks).
            proc = subprocess.Popen(
                cmd, cwd=workdir, stdin=stdin, stdout=out,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_s)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                proc.wait()
                ok = False
            except BaseException:
                # Its own process group is what lets a timeout reap the whole tree; it is also
                # what stops Ctrl-C from reaching the engine. Without this the queue dies and
                # ORCA carries on writing to the same .out, which a later --resume then reopens
                # in "wb" underneath it.
                _kill_process_tree(proc)
                proc.wait()
                raise
    finally:
        if stdin is not None:
            stdin.close()
    return ok, time.monotonic() - start
