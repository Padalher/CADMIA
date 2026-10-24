<p align="center">
  <img src="https://raw.githubusercontent.com/Padalher/CADMIA/main/docs/logo.png" alt="CADMIA" width="560">
</p>

<h3 align="center">Computational Agent-Driven Molecular Intelligence Architecture</h3>

<p align="center">

**English** · [Español](https://github.com/Padalher/CADMIA/blob/main/README.es.md)

[![CI](https://github.com/Padalher/CADMIA/actions/workflows/ci.yml/badge.svg)](https://github.com/Padalher/CADMIA/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](https://github.com/Padalher/CADMIA/blob/main/LICENSE)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)](https://github.com/Padalher/CADMIA/blob/main/pyproject.toml)

</p>

Author: Pablo Daniel Alvarez Hernandez, Universidad Juárez Autónoma de Tabasco (UJAT)

Status: alpha. Commands and output formats may still change between minor versions.

## What is CADMIA?

CADMIA is a command-line suite for studying the reactivity and adsorption properties of
nanoclusters from quantum-chemistry outputs (ORCA and Gaussian). It reads the calculations,
derives the descriptors, prepares the next calculations and records where every number came from.
A single command, `cadmia`, gives access to every tool.

Main features:

- Conceptual DFT: global descriptors (I, A, η, μ, ω) by finite differences or Koopmans, and
  condensed Fukui functions and the dual descriptor, matching Multiwfn.
- Adsorption: starting configurations for small probe molecules on the most reactive sites,
  GFN2-xTB pre-screening, and adsorption energetics (ΔE, ΔH, ΔG, deformation and interaction).
- Stability: mixing and binding energies against the pure oxides, size series, isomers and
  Boltzmann populations.
- Structure: coordination numbers (CN, ECN), bond lengths, rings, coordination polyhedra and
  point-group symmetry.
- Electronic structure: DOS and PDOS, electrostatic potential on the molecular surface, IR spectra
  and volumetric maps (density, Fukui, Δρ).
- Calculations: input preparation, a local sequential queue and SLURM submission, with
  resumable pipelines.
- Provenance: results stored in a local SQLite database with file hash, exact command and
  program version.
- Bilingual help (`--help` in English, `--ayuda` in Spanish) and Markdown/LaTeX reports.

## What's new in 0.3.0

Highlights only; the full detail, with the earlier development versions, is in
[CHANGELOG.md](https://github.com/Padalher/CADMIA/blob/main/CHANGELOG.md).

- **Isomers in `stability-series`:** several `--cluster` outputs with the same n are treated as
  isomers, each with point group, chirality, ΔE₀ and ΔG to its own minimum. The lowest in the
  chosen basis (E₀ by default) represents n; if the E₀ and G minima differ, both are kept and
  flagged.
- **Boltzmann populations in both bases:** `--pop` (off by default) adds P(G), with σ from the
  geometry and g = 2 for chiral isomers, and P(E₀), with the same g and no σ. Population-weighted
  descriptors are labelled as such.
- **Mixing energy at equal size:** `stability` solves the coefficients as exact fractions
  (`--coeff` accepts `1/2` or `0.5`), and `stability-series --ref` pairs pure oxides of any size by
  stoichiometry, reporting ΔE_mix per formula unit.
- **Symmetry from the geometry:** point group, σ and chirality are detected (0.01 Å tolerance); G
  and Gq are corrected when the engine used a different σ.
- **Every term is checked:** `stability` and `stability-series` record the multiplicity and the
  imaginary frequencies of each output, and warn about missing frequencies, mismatched
  multiplicities or temperatures, mixed ORCA/Gaussian entropies and abnormal terminations.
- **`--energy Gq`:** free energy with Grimme's quasi-RRHO entropy (ν₀ = 100 cm⁻¹, as in ORCA 6).
- **Fixed:** `electronic mep --density` now samples the actual ρ = isovalue surface with area
  weights (it used a shell that reached into the nuclei); it matches Multiwfn within 0.52 %.
  A frequency-only Gaussian job is no longer reported as a non-converged optimization.

## Requirements

- Python 3.11 or newer.
- Outputs from ORCA 6 or Gaussian 16. ORCA is needed only to run calculations; analysis works on
  any machine.
- Optional: `xtb` (pre-screening), SLURM (job submission) and POV-Ray (structure rendering).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .            # core
pip install -e ".[full]"    # optional: figures, interactive plots, isosurfaces
cadmia doctor               # checks engines, dependencies and configuration
```

## Quick start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Padalher/CADMIA/blob/main/examples/quickstart.ipynb)
runs the examples in the browser, with nothing to install.

From a clone of the repository, with the sample outputs that ship with the tests:

```bash
cadmia doctor                     # engines, dependencies and configuration
cadmia cdft --n tests/fixtures/1tial_cdft/N.out \
            --np1 tests/fixtures/1tial_cdft/N+1.out \
            --nm1 tests/fixtures/1tial_cdft/N-1.out     # descriptors + ranked reactive sites
cadmia structure report tests/fixtures/1tial_orca/1tial_opt.xyz -o structure.txt   # coordination, bonds, rings
cadmia guide                      # the full user guide
```

## Documentation

- `cadmia --help` and `cadmia <command> --help` (English); `--ayuda` (Spanish).
- User guide: `docs/usage_guide_en.txt` (English) and `docs/usage_guide.txt` (Spanish).
- Dictionary of terms and descriptors: [docs/descriptor_dictionary.md](https://github.com/Padalher/CADMIA/blob/main/docs/descriptor_dictionary.md).
- Validation of the CDFT descriptors against Multiwfn: [docs/cdft_vs_multiwfn.md](https://github.com/Padalher/CADMIA/blob/main/docs/cdft_vs_multiwfn.md).
- Changes between versions: [CHANGELOG.md](https://github.com/Padalher/CADMIA/blob/main/CHANGELOG.md).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Citation and license

If CADMIA is used in published work, please cite it as described in
[CITATION.cff](https://github.com/Padalher/CADMIA/blob/main/CITATION.cff). CADMIA is free software, distributed under the GNU General Public License,
version 3 or (at your option) any later version ([LICENSE](https://github.com/Padalher/CADMIA/blob/main/LICENSE)).

## Acknowledgements

CADMIA was designed, directed and scientifically validated by Pablo Daniel Álvarez Hernández.
AI-based coding assistants were used to write and refactor parts of the code, the automated
tests and the documentation; every change was reviewed and validated by the author. Numerical
results always come from the quantum-chemistry outputs, never from a language model.

## Contact

Bugs and questions: [GitHub Issues](https://github.com/Padalher/CADMIA/issues).

Direct contact: Pablo Daniel Alvarez Hernandez — pablo.alvarez@padalher.com — https://padalher.com
