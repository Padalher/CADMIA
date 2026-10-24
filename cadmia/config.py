"""HPC profile management — load, save, and list YAML connection profiles."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_PROFILES_DIR: Path = Path.home() / ".cadmia" / "profiles"

_VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class HPCProfile:
    """Connection and scheduler parameters for one HPC cluster.

    Attributes
    ----------
    name              : unique profile identifier (used in CLI --hpc).
    host              : SSH hostname or IP address (ignored when local=True).
    user              : SSH username (ignored when local=True).
    work_dir          : scratch directory where input/output files land
                        (remote path for SSH profiles, local path when local=True).
    partition         : SLURM partition name.
    account           : SLURM account (empty = not used).
    ssh_key           : path to private SSH key (empty = use SSH agent / default key).
    gaussian_module   : module to load before running Gaussian (empty = none, use PATH).
    orca_module       : module to load before running ORCA (empty = none, use PATH).
    orca_path         : absolute path to the ORCA binary directory (empty = use PATH).
    max_time          : SLURM wall-time limit in HH:MM:SS format.
    ssh_port          : SSH port (default: 22).
    extra_sbatch      : additional #SBATCH lines appended verbatim to all scripts.
    local             : if True, jobs are submitted to SLURM on THIS machine
                        (no SSH); host/user are not required.
    """

    name:            str
    host:            str
    user:            str
    work_dir:        str
    partition:       str  = "general"
    account:         str  = ""
    ssh_key:         str  = ""
    gaussian_module: str  = "gaussian/16"
    orca_module:     str  = "orca/6"
    orca_path:       str  = ""
    max_time:        str  = "24:00:00"
    ssh_port:        int  = 22
    extra_sbatch:    List[str] = field(default_factory=list)
    local:           bool = False

    def __post_init__(self) -> None:
        if not _VALID_NAME.match(self.name):
            raise ValueError(
                f"Profile name '{self.name}' contains invalid characters. "
                "Use only letters, digits, hyphens, and underscores."
            )
        if not self.work_dir:
            raise ValueError("HPCProfile.work_dir must not be empty.")
        if not self.local:
            if not self.host:
                raise ValueError("HPCProfile.host must not be empty.")
            if not self.user:
                raise ValueError("HPCProfile.user must not be empty.")

    # ── Persistence ────────────────────────────────────

    def save(self, directory: Path = DEFAULT_PROFILES_DIR) -> Path:
        """Serialize to YAML and write to *directory*/<name>.yaml."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.yaml"
        data = asdict(self)
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, name: str, directory: Path = DEFAULT_PROFILES_DIR) -> "HPCProfile":
        """Load a profile by name from *directory*/<name>.yaml."""
        directory = Path(directory)
        path = directory / f"{name}.yaml"
        if not path.exists():
            available = [p.stem for p in directory.glob("*.yaml")] if directory.exists() else []
            hint = f"  Available: {available}" if available else "  No profiles found."
            raise FileNotFoundError(
                f"Profile '{name}' not found at {path}.\n{hint}"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**data)

    @staticmethod
    def list_profiles(directory: Path = DEFAULT_PROFILES_DIR) -> List[str]:
        """Return sorted list of available profile names."""
        directory = Path(directory)
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.yaml"))
