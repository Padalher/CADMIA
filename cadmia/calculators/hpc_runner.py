"""SSH/SLURM executor — uploads inputs, submits jobs, polls status, downloads outputs.

Uses the system ``ssh`` and ``scp`` binaries via subprocess.
No paramiko dependency required.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cadmia.calculators.params import CalcParams
    from cadmia.config import HPCProfile

# SLURM state → CADMIA status mapping
_SLURM_STATUS = {
    "PENDING":    "queued",
    "RUNNING":    "running",
    "COMPLETED":  "done",
    "FAILED":     "failed",
    "CANCELLED":  "failed",
    "TIMEOUT":    "failed",
    "NODE_FAIL":  "failed",
    "PREEMPTED":  "failed",
    "OUT_OF_MEMORY": "failed",
}

_SSH_TIMEOUT  = 30   # seconds per SSH command
_SCP_TIMEOUT  = 300  # seconds per file transfer


def marker_status(text: str) -> Optional[str]:
    """Map a ``CADMIA_DONE:<code>`` completion marker to a status, or None if absent.

    The SLURM script appends this marker to the output file so completion can be
    detected even on clusters without accounting (sacct), where a finished job
    simply vanishes from squeue.
    """
    m = re.search(r"CADMIA_DONE:(\d+)", text)
    if not m:
        return None
    return "done" if m.group(1) == "0" else "failed"


class HPCConnectionError(RuntimeError):
    """Raised when an SSH or SCP command fails."""


class HPCRunner:
    """Thin wrapper around system ssh/scp for SLURM job management."""

    def __init__(self, profile: "HPCProfile") -> None:
        self.profile = profile

    # ── SSH helpers ────────────────────────────────────

    def _ssh_base(self) -> List[str]:
        """Return base ssh command list (host, port, key as applicable)."""
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if self.profile.ssh_port != 22:
            cmd += ["-p", str(self.profile.ssh_port)]
        if self.profile.ssh_key:
            cmd += ["-i", str(Path(self.profile.ssh_key).expanduser())]
        cmd.append(f"{self.profile.user}@{self.profile.host}")
        return cmd

    def _scp_base(self) -> List[str]:
        """Return base scp command list."""
        cmd = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-q"]
        if self.profile.ssh_port != 22:
            cmd += ["-P", str(self.profile.ssh_port)]
        if self.profile.ssh_key:
            cmd += ["-i", str(Path(self.profile.ssh_key).expanduser())]
        return cmd

    def _run_ssh(self, remote_cmd: str, timeout: int = _SSH_TIMEOUT) -> Tuple[int, str, str]:
        """Execute *remote_cmd* on the HPC. Returns (returncode, stdout, stderr)."""
        full = self._ssh_base() + [remote_cmd]
        result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def ping(self) -> bool:
        """Return True if the HPC host is reachable via SSH."""
        try:
            rc, _, _ = self._run_ssh("echo cadmia_ping", timeout=10)
            return rc == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    # ── File transfer ──────────────────────────────────

    def upload(self, local_path: Path, remote_path: str, timeout: int = _SCP_TIMEOUT) -> None:
        """Copy *local_path* to *remote_path* on the HPC."""
        target = f"{self.profile.user}@{self.profile.host}:{remote_path}"
        cmd = self._scp_base() + [str(local_path), target]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise HPCConnectionError(
                f"scp upload failed ({result.returncode}): {result.stderr}"
            )

    def download(self, remote_path: str, local_path: Path, timeout: int = _SCP_TIMEOUT) -> None:
        """Copy *remote_path* from the HPC to *local_path*."""
        src = f"{self.profile.user}@{self.profile.host}:{remote_path}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._scp_base() + [src, str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise HPCConnectionError(
                f"scp download failed ({result.returncode}): {result.stderr}"
            )

    # ── SLURM ──────────────────────────────────────────

    def mkdir_remote(self, remote_dir: str) -> None:
        """Create directory on HPC (parents included)."""
        rc, _, err = self._run_ssh(f"mkdir -p {remote_dir}")
        if rc != 0:
            raise HPCConnectionError(f"mkdir -p {remote_dir} failed: {err}")

    def submit(
        self,
        input_path: Path,
        engine: str,
        params: "CalcParams",
        label: str = "",
        remote_work_dir: str = "",
    ) -> Tuple[str, str]:
        """Upload the input file and submit via sbatch.

        Returns
        -------
        (job_id, remote_output_path)
            SLURM job id (as string) and the expected remote output file path.
        """
        from cadmia.calculators.slurm import write_slurm_script

        label = label or input_path.stem
        work_dir = remote_work_dir or f"{self.profile.work_dir}/{label}"
        self.mkdir_remote(work_dir)

        # Remote paths
        ext = ".log" if engine == "gaussian" else ".out"
        remote_inp = f"{work_dir}/{input_path.name}"
        remote_out = f"{work_dir}/{label}{ext}"
        script_str = write_slurm_script(
            remote_input=remote_inp,
            remote_output=remote_out,
            engine=engine,
            params=params,
            profile=self.profile,
            job_name=label,
            remote_work_dir=work_dir,
        )
        remote_script = f"{work_dir}/{label}.sh"

        # Upload input file and SLURM script
        self.upload(input_path, remote_inp)

        # Write script file via ssh heredoc (avoids tmp file on local)
        safe = script_str.replace("'", "'\\''")
        rc, _, err = self._run_ssh(f"printf '%s' '{safe}' > {remote_script}")
        if rc != 0:
            raise HPCConnectionError(f"Could not write SLURM script to {remote_script}: {err}")

        # Submit
        rc, stdout, stderr = self._run_ssh(f"sbatch {remote_script}")
        if rc != 0:
            raise HPCConnectionError(f"sbatch failed: {stderr}")

        # Parse job id: "Submitted batch job 12345"
        m = re.search(r"Submitted batch job (\d+)", stdout)
        if not m:
            raise HPCConnectionError(f"Could not parse job id from sbatch output: {stdout!r}")

        return m.group(1), remote_out

    def check_status(self, job_id: str, output_path: str = "") -> str:
        """Return CADMIA job status: queued | running | done | failed | unknown.

        Falls back to the ``CADMIA_DONE`` marker in *output_path* when neither
        squeue nor sacct can resolve the job (e.g. accounting disabled).
        """
        rc, stdout, _ = self._run_ssh(
            f"squeue -j {job_id} -h -o %T 2>/dev/null"
        )
        slurm_state = stdout.strip().upper()
        if slurm_state:
            return _SLURM_STATUS.get(slurm_state, "unknown")

        # Job no longer in queue — check sacct
        rc, stdout, _ = self._run_ssh(
            f"sacct -j {job_id} -n -o State%20 2>/dev/null | head -1"
        )
        if stdout.strip():
            state = stdout.strip().split()[0].upper()
            mapped = _SLURM_STATUS.get(state, "unknown")
            if mapped != "unknown":
                return mapped

        # Neither squeue nor sacct know (too new, sacct down, no accounting).
        # Consult the completion marker the SLURM script appends to the output.
        if output_path:
            rc, stdout, _ = self._run_ssh(
                f"grep -a CADMIA_DONE {output_path} 2>/dev/null | tail -1"
            )
            status = marker_status(stdout)
            if status:
                return status
        return "unknown"

    def cancel(self, job_id: str) -> None:
        """Cancel a SLURM job."""
        self._run_ssh(f"scancel {job_id}")

    def list_jobs(self, user: Optional[str] = None) -> List[dict]:
        """Return list of dicts with job_id, name, state, time for the user's jobs."""
        u = user or self.profile.user
        rc, stdout, _ = self._run_ssh(
            f"squeue -u {u} -h -o '%i %j %T %M' 2>/dev/null"
        )
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
        """Poll until the job is done or failed. Returns final status.

        Parameters
        ----------
        poll_interval : seconds between squeue calls.
        timeout       : maximum wall time to wait (seconds).
        output_path   : the job's output file. Without it, a cluster with no accounting has no
                        way to answer once the job leaves the queue, because the CADMIA_DONE
                        marker is the only fallback and it lives in that file.
        """
        elapsed = 0
        while elapsed < timeout:
            status = self.check_status(job_id, output_path)
            if status in ("done", "failed"):
                return status
            time.sleep(poll_interval)
            elapsed += poll_interval
        return "unknown"
