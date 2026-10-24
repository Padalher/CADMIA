"""Tests for version consistency across package metadata files."""
from __future__ import annotations

from pathlib import Path
import tomllib
import yaml

import cadmia


def test_version_consistency_across_metadata():
    """Verify pyproject.toml, CITATION.cff, and cadmia.__version__ remain identical."""
    root = Path(__file__).resolve().parent.parent.parent
    pyproject_path = root / "pyproject.toml"
    citation_path = root / "CITATION.cff"

    assert pyproject_path.exists(), f"Missing pyproject.toml at {pyproject_path}"
    assert citation_path.exists(), f"Missing CITATION.cff at {citation_path}"

    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    pyproject_version = pyproject_data["project"]["version"]

    with open(citation_path, "r", encoding="utf-8") as f:
        citation_data = yaml.safe_load(f)
    citation_version = str(citation_data.get("version"))

    package_version = cadmia.__version__

    assert pyproject_version == package_version, (
        f"pyproject.toml version ({pyproject_version}) does not match "
        f"cadmia.__version__ ({package_version})"
    )
    assert citation_version == package_version, (
        f"CITATION.cff version ({citation_version}) does not match "
        f"cadmia.__version__ ({package_version})"
    )
