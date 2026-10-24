"""SLURM batch script generation for Gaussian and ORCA jobs."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadmia.calculators.params import CalcParams
    from cadmia.config import HPCProfile


def write_slurm_script(
    remote_input:  str,    # remote absolute path to .gjf or .inp
    remote_output: str,    # remote absolute path for the output file
    engine:        str,    # "gaussian" | "orca"
    params:        "CalcParams",
    profile:       "HPCProfile",
    job_name:      str = "",
    remote_work_dir: str = "",
    tidy:          bool = True,
) -> str:
    """Return a SLURM batch script as a string.

    The script: loads the relevant module, changes to *remote_work_dir*,
    runs the calculation, and writes a ``CADMIA_DONE:<exit_code>`` marker
    to the output file so completion can be detected without squeue.

    Parameters
    ----------
    remote_input    : absolute path to the input file on the remote host.
    remote_output   : absolute path for the calculation output file.
    engine          : "gaussian" or "orca".
    params          : CalcParams (for nprocs, mem).
    profile         : HPCProfile (for partition, account, modules, etc.).
    job_name        : SLURM job name (default: input file stem).
    remote_work_dir : directory to cd into before running; defaults to dirname(remote_input).
    """
    if engine not in ("gaussian", "orca"):
        raise ValueError(f"engine must be 'gaussian' or 'orca', got '{engine}'")

    inp = Path(remote_input)
    job_name = job_name or inp.stem
    work_dir = remote_work_dir or str(inp.parent)
    mem_gb   = max(params.mem_gaussian_gb, params.mem_orca_mb_core * params.nprocs // 1024)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=cadmia_{job_name}",
        f"#SBATCH --partition={profile.partition}",
    ]

    if profile.account:
        lines.append(f"#SBATCH --account={profile.account}")

    lines += [
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node={params.nprocs}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={profile.max_time}",
        f"#SBATCH --output={work_dir}/cadmia_{job_name}_%j.slurm.out",
        f"#SBATCH --error={work_dir}/cadmia_{job_name}_%j.slurm.err",
    ]

    for extra in profile.extra_sbatch:
        lines.append(f"#SBATCH {extra}")

    lines += ["", f"cd {work_dir}", ""]

    if engine == "gaussian":
        if profile.gaussian_module:
            lines.append(f"module load {profile.gaussian_module}")
        lines += [
            "export GAUSS_SCRDIR=${TMPDIR:-/tmp}",
            "",
            f"g16 < {remote_input} > {remote_output}",
            "_rc=$?",
        ]
    else:
        orca_bin = f"{profile.orca_path}/orca" if profile.orca_path else "orca"
        if profile.orca_module:
            lines.append(f"module load {profile.orca_module}")
        if profile.orca_path:
            lines.append(f"export PATH={profile.orca_path}:$PATH")
        lines += [
            "",
            f"{orca_bin} {remote_input} > {remote_output}",
            "_rc=$?",
        ]

    # Completion marker — lets check_status fall back to file inspection
    lines += [
        "",
        'echo "CADMIA_DONE:$_rc" >> ' + remote_output,
    ]

    # Tuck accessory byproducts into aux/.
    if tidy:
        from cadmia.calculators.tidy import slurm_tidy_block
        lines += ["", slurm_tidy_block(remote_input, work_dir)]

    # The script's exit status must be the ENGINE's, and nothing else's. A bash script exits
    # with the status of its last command, and the last command here is the tidy loop, whose
    # `[ -f X ] && mv` returns 1 whenever the file is absent. ORCA only writes <stem>.xyz for
    # an optimization, so every single point and every frequency job ended with status 1 and
    # SLURM recorded it as FAILED — which check_status reads from sacct BEFORE it ever looks
    # at the CADMIA_DONE marker. Hence the explicit exit.
    lines += ["", "exit $_rc"]

    return "\n".join(lines) + "\n"
