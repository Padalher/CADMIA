"""Local SLURM executor — submits jobs to SLURM on THIS machine (no SSH).

Mirrors the public interface of :class:`cadmia.calculators.hpc_runner.HPCRunner`
(submit / check_status / cancel / list_jobs / wait_for_completion / ping, plus
upload / download as local copies) so the Calculation Agent can dispatch to either
backend by profile, with no other change. Used for the workstation that already
runs SLURM locally; the remote SSH backend stays available for clusters.
"""
from __future__ import annotations

import getpass
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

from cadmia.calculators.hpc_runner import HPCConnectionError, _SLURM_STATUS, marker_status

if TYPE_CHECKING:
    from cadmia.calculators.params import CalcParams
    from cadmia.config import HPCProfile

_CMD_TIMEOUT = 30   # seconds per SLURM command


class LocalSlurmRunner:
    """Submit and manage SLURM jobs on the local machine via subprocess."""

    def __init__(self, profile: "HPCProfile") -> None:
        self.profile = profile

    # ── command helper ─────────────────────────────────

    def _run(self, cmd: List[str], timeout: int = _CMD_TIMEOUT) -> Tuple[int, str, str]:
        """Run a local command; returns (returncode, stdout, stderr)."""
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def ping(self) -> bool:
        """Return True if SLURM (sbatch) is available on this machine."""
        return shutil.which("sbatch") is not None

    # ── File transfer (local: copies, not transfers) ───

    def upload(self, local_path: Path, remote_path: str, timeout: int = _CMD_TIMEOUT) -> None:
        """Copy *local_path* to *remote_path* (both local); no-op if identical."""
        dst = Path(remote_path)
        if Path(local_path).resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dst)

    def download(self, remote_path: str, local_path: Path, timeout: int = _CMD_TIMEOUT) -> None:
        """Copy *remote_path* to *local_path* (both local); no-op if identical."""
        src = Path(remote_path)
        dst = Path(local_path)
        if src.resolve() == dst.resolve():
            return
        if not src.exists():
            raise HPCConnectionError(f"output not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    def mkdir_remote(self, remote_dir: str) -> None:
        """Create a local directory (parents included)."""
        Path(remote_dir).mkdir(parents=True, exist_ok=True)

    # ── SLURM ──────────────────────────────────────────

    def submit(
        self,
        input_path: Path,
        engine: str,
        params: "CalcParams",
        label: str = "",
        remote_work_dir: str = "",
    ) -> Tuple[str, str]:
        """Stage the input in the work dir, write the SLURM script and sbatch it.

        Returns ``(job_id, output_path)`` — the SLURM job id and the local path of
        the expected output file.
        """
        from cadmia.calculators.slurm import write_slurm_script

        input_path = Path(input_path)
        label = label or input_path.stem
        work_dir = Path(remote_work_dir or f"{self.profile.work_dir}/{label}")
        work_dir.mkdir(parents=True, exist_ok=True)

        ext = ".log" if engine == "gaussian" else ".out"
        staged_inp = work_dir / input_path.name
        if input_path.resolve() != staged_inp.resolve():
            shutil.copy(input_path, staged_inp)
        out_path = work_dir / f"{label}{ext}"

        script_str = write_slurm_script(
            remote_input=str(staged_inp),
            remote_output=str(out_path),
            engine=engine,
            params=params,
            profile=self.profile,
            job_name=label,
            remote_work_dir=str(work_dir),
        )
        script_path = work_dir / f"{label}.sh"
        script_path.write_text(script_str, encoding="utf-8")

        rc, stdout, stderr = self._run(["sbatch", str(script_path)])
        if rc != 0:
            raise HPCConnectionError(f"sbatch failed: {stderr}")

        m = re.search(r"Submitted batch job (\d+)", stdout)
        if not m:
            raise HPCConnectionError(f"Could not parse job id from sbatch output: {stdout!r}")

        return m.group(1), str(out_path)

    def check_status(self, job_id: str, output_path: str = "") -> str:
        """Return CADMIA job status: queued | running | done | failed | unknown.

        Falls back to the ``CADMIA_DONE`` marker in *output_path* when neither
        squeue nor sacct can resolve the job (common on a workstation SLURM with
        accounting disabled, where a finished job just disappears from squeue).
        """
        rc, stdout, _ = self._run(["squeue", "-j", job_id, "-h", "-o", "%T"])
        slurm_state = stdout.strip().upper()
        if slurm_state:
            return _SLURM_STATUS.get(slurm_state, "unknown")

        # Job no longer queued — consult accounting.
        rc, stdout, _ = self._run(["sacct", "-j", job_id, "-n", "-o", "State%20"])
        if stdout.strip():
            state = stdout.strip().splitlines()[0].split()[0].upper()
            mapped = _SLURM_STATUS.get(state, "unknown")
            if mapped != "unknown":
                return mapped

        # No queue entry and no accounting: use the completion marker.
        status = self._status_from_marker(output_path)
        return status or "unknown"

    def _status_from_marker(self, output_path: str) -> Optional[str]:
        """Read the CADMIA_DONE marker from a local output file, if present."""
        if not output_path:
            return None
        p = Path(output_path)
        if not p.exists():
            return None
        return marker_status(p.read_text(errors="ignore"))

    def cancel(self, job_id: str) -> None:
        """Cancel a SLURM job."""
        self._run(["scancel", job_id])

    def list_jobs(self, user: Optional[str] = None) -> List[dict]:
        """Return dicts with job_id, name, state, wall_time for the user's jobs."""
        u = user or self.profile.user or getpass.getuser()
        rc, stdout, _ = self._run(["squeue", "-u", u, "-h", "-o", "%i %j %T %M"])
        jobs = []
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                jobs.append({
                    "job_id":    parts[0],
                    "name":      parts[1],
                    "state":     _SLURM_STATUS.get(parts[2].upper(), "unknown"),
                    "wall_time": parts[3],
                })
        return jobs

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int = 86400,
        output_path: Optional[str] = None,
    ) -> str:
        """Poll until the job is done or failed. Returns the final status."""
        elapsed = 0
        while elapsed < timeout:
            status = self.check_status(job_id, output_path)
            if status in ("done", "failed"):
                return status
            time.sleep(poll_interval)
            elapsed += poll_interval
        return "unknown"
