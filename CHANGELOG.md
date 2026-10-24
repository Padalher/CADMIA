# Changelog

All notable changes to CADMIA are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the version is 0.x, a minor version may change commands and output formats.

## [0.3.0] — 2026-10-23

First public release.

### Added
- **Quick-start notebook.** `examples/quickstart.ipynb` runs in Google Colab, with nothing to
  install: output checks, CDFT against Multiwfn, structure, adsorption geometries, GFN2-xTB
  pre-screening, adsorption thermochemistry, stability and the final report.

- **Citation notice.** `cadmia report` and a completed `cadmia pipeline` end with a short
  notice on how to cite CADMIA.

- **Isomers in `stability-series`.** Clusters with the same size are kept as isomers instead of
  being dropped; each reports its point group, chirality and energy against the lowest E₀ and G minima.

- **Boltzmann populations.** `--pop` adds populations on both bases (E₀ and G), with G corrected
  for symmetry.

- **Equal-size mixing energy.** Reference coefficients are exact fractions (`1/2` or `0.5`); `--ref`
  accepts pure oxides of any size and reports the value per formula unit.

- **Symmetry from geometry.** Point group, symmetry number σ and chirality are detected from the
  structure; `stability` corrects G and Gq when the program used a different σ, and says so.

- **Per-term checks.** `stability` and `stability-series` report multiplicities and imaginary
  frequencies, and warn about missing frequencies, mismatched temperatures and mixed entropy schemes.

- **`--energy Gq`.** Free energy with Grimme's quasi-RRHO entropy, for Gaussian and ORCA outputs.

- **Open-shell CDFT.** Koopmans descriptors (η, S, μ, χ, ω) and the Koopmans-vs-SCF gap row are
  labelled as cross-spin quantities.

### Changed

- Supported engines: ORCA 6 and Gaussian 16. The optional cclib backup reader for other
  engines (0.3.0a4) was removed.

- `--coeff` with a number of values different from the number of references is now an error,
  instead of printing meaningless energies.

### Fixed

- **DOS and PDOS sampling.** The energy grid spanned the core orbitals with a fixed 2000 points,
  a step of several eV for a Ti-containing system, so the valence peaks were aliased. The step is
  now at most σ/4 over the whole range (`--points` is a minimum), and the HTML plot opens on
  HOMO − 20 eV to LUMO + 10 eV instead of the full core-to-virtual range.

- **MEP on isosurfaces.** `electronic mep --density` measures on the isosurface itself, not on a
  voxel shell that reached toward the nuclei; it fails clearly when the isovalue misses the grid.

- **Frequency-only Gaussian jobs** were flagged as "optimization not converged". Routes without
  `Opt` no longer trigger that check.

- **Abnormal termination** (Error termination or a missing end marker) is reported by `stability`
  and `stability-series`, instead of silently using the last energy printed.

- **`pipeline` engine.** Without `--engine`, the calculation stage now uses the engine that wrote
  the N output instead of always ORCA. A stage whose outputs all fail validation is reported as
  failed, and settings the engine cannot take (D4 in Gaussian) stop the stage with the reason.

- **Metals from `--config`** now reach the neighbourhood, ring, polyhedra and angle sections of
  `structure`, which had reloaded the default list.

- **CDFT input checks.** `cdft` stops when the N+1 and N−1 files are swapped (their total charges
  are checked against N), and its warnings (unbound anion, spin contamination) are kept in the
  JSON and the database record.

- **`validate` on Gaussian** reads ⟨S²⟩ before annihilation, as the CDFT report does; the value
  after it hid the contamination.

- **`rank` → `prepare`.** Inputs prepared from a ranked `.xyz` keep the name of the output each
  geometry came from, also with skeleton classes or a single geometry.

- **Single-geometry `.xyz`** with extra lines (a trajectory, a wrong atom count) is an error instead
  of silently giving the first block.

- **`structure` on a directory** no longer reads its own earlier report from it, and Li, Na, K,
  Mg, Ga, In, Sn, La and Ce are in the default metal list.

- **Recorded commands** name the program (`cadmia …`) instead of its install path, which put the
  user's home directory into reports and database rows.
- **`cadmia report`** reads the level of theory from the outputs behind the CDFT and adsorption
  results (the keyword line the engine echoed) when `--functional`/`--basis` are not given.
  `--structure` adds the structural report (`cadmia structure report -o structure.json`); each
  stage opens with a framed, numbered title; the citation follows the title banner, and the
  footer lists the stages reported, the warning count and the software, Python and OS versions.
- **`cadmia init`** no longer writes a level of theory or probe into `cadmia_project.yaml`: no
  command read them, and they looked like defaults.
- **`cadmia cdft`** without `-o` no longer prints the JSON after the table; with `--no-table`
  the JSON still goes to stdout, for piping.

## Development history before the first public release

These versions predate the first public release and were never published.

- **0.3.0a10** (2026-10-18): without the optional cclib extra, unreadable input files raised an
  `ImportError` in some commands instead of a clear error.

- **0.3.0a9** (2026-10-17): Hoppe effective coordination has no fixed distance cutoff; `stability`
  uses E₀ = E + ZPE by default; structural reports gain analysis parameters and metal–metal neighbours.

- **0.3.0a8** (2026-09-12): Gaussian counterpoise corrections were silently ignored; successful
  SLURM jobs were marked as failed; several density and output-handling fixes.

- **0.3.0a7** (2026-09-12): full structural section (continuous shape measures, polyhedra
  connectivity, bond angles, rings, global shape, `cadmia structure compare`); `analyze` refuses
  adsorption energies from unconverged geometries.

- **0.3.0a6** (2026-09-12): provenance recorded end to end; database persistence on by default
  (`--db/--no-db`); structure generation samples every site and probe orientation.

- **0.3.0a5** (2026-09-10): nuclear Fukui function in `cadmia cdft`;
  duplicate minima grouped in `rank`; probes approach along the local surface normal.

- **0.3.0a4** (2026-09-03): cclib as a backup reader for other engines; `correlate` command;
  `rank --window/--top`; `cadmia density --step` replaces `--ngrid`; production accuracy by default in `pipeline`.

- **0.3.0a3**: Koopmans hardness uses η = LUMO − HOMO (ω halved, η doubled); the standalone `cdft` and
  `electronic` aliases are removed.

- **0.3.0a2**: workstation and SLURM integration (`prepare`, `submit`, `run`); GFN2-xTB pre-screening;
  D4 dispersion by default; interactive structure viewer; density cubes and Fukui maps; ΔG/ΔH from frequencies.

- **0.3.0a1**: single `cadmia` entry point with subcommands; Mayer bond orders; project scaffolding and
  output import; interactive cube and VMD exports; literature search; extensible adsorption probes.

- **0.1.0 – 0.2.0**: common data core; Gaussian and ORCA parsers; native conceptual DFT (CDFT) with
  condensed Fukui functions; staged pipeline with a resumable manifest; SQLite database; optional,
  provider-agnostic LLM layer for report narration and diagnostics.
