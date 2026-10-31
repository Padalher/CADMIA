# Changelog

All notable changes to CADMIA are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the version is 0.x, a minor version may change commands and output formats.

## [Unreleased]

### Changed
- **Author affiliation:** División Académica de Ciencias Básicas, UJAT (DACB-UJAT) in the citation,
  the guides and the headers of the outputs.

### Fixed
- The help of `structure plot`, `structure` and `doctor` no longer mentions removed features
  (3D plot kinds, POV-Ray rendering, scikit-image); the guides call the parts of `cadmia.out`
  modules and list every main command.
- `tools/guide_to_latex.py` keeps the structure of the text guide in the PDF: headings in
  capitals, bulleted and numbered lists, and column-aligned tables no longer run into one
  paragraph; paragraphs are separated in the LaTeX, and a CLI option keeps its two hyphens.
  The longest paragraphs of the text guides are split at a sentence end (no wording changed).
- The CDFT report labels the condensed local softness in e/Hartree (it said Hartree·e); the
  values were already correct.
- Every JSON and report file is written as UTF-8, and the xTB output is read as UTF-8: on a
  system whose default encoding is not UTF-8 (Windows), `analyze` failed on the `Δ` of its
  output.
- `analyze` links its result to the configuration `geometry` recorded when that path was stored
  with Windows separators.
- Report paths relative to the project use `/` on every system, and so do the stage files the
  report lists and the file names inside the combined digest of the inputs, which is now the
  same on every system.

## [0.3.1] — 2026-10-27

### Added
- **`prescreen --freeze`** holds the cluster rigid in xTB (an `$constrain` on its atoms, given
  by `--cluster`) so only the probe relaxes. Relaxing everything stays the default: on the Al₂TiO₅ + CO₂
  example the frozen screen ranked the configurations alike (Spearman 0.83) but sent none of
  the Al sites to DFT, which bind only by moving the cluster. The DFT stage is never constrained.
- **DOI.** CADMIA is archived on Zenodo (concept DOI 10.5281/zenodo.23608477); the citation
  notice, `CITATION.cff` and the README cite it.
- **Module results in the project.** Inside a `cadmia init` project, `structure report`, `cdft`,
  `analyze` and `stability` save their result to `results/` by themselves
  (`results/cdft.json`, `results/stability.json`, `results/structure/<id>.json`,
  `results/analysis/<adduct>.json`). Each file holds the data and the module's full text
  report, so the two cannot disagree; the output frames them as `MODULE n · NAME`.
- **Counterpoise BSSE for ORCA.** `cadmia prepare bsse ADDUCT.out --cluster C.out --probe-free P.out`
  writes the four single points of the Boys–Bernardi correction (each fragment at the adduct
  geometry, in its own and in the full basis, with ORCA ghost atoms); `analyze --bsse` reads
  their outputs and reports E_ads(BSSE).
- **Static figures.** `electronic dos|pdos|ir --format png|svg|pdf` and
  `structure plot --kind bonds|distances|ecn-trend|ecn-heatmap -o NAME.png|.svg|.pdf` write
  publication-ready figures with matplotlib (in `[full]`): 300-dpi PNG or vector SVG/PDF, one
  style for all of them; the DOS ones mark the frontier orbitals and open on the valence window.
  The distance matrix shows one half with the value in each cell and outlines the pairs within
  the bond cut-offs, so it no longer reads like the ECN heatmap.
  The interactive HTML stays for exploring.
- **`analyze --bend-threshold DEG`** (default 10°): the bending of a linear probe that counts
  as activation.
- **Figure width and format.** `--width single|double` (3.4 in / 6.5 in, single by default)
  for the static figures of `electronic dos|pdos|ir` and `structure plot`. Both commands take
  the format from `--format`, else from the `-o` suffix, else the default; `structure plot
  --format png|svg|pdf` writes static figures in a batch too.
- **Full text reports for adsorption and stability**, in the layout of the CDFT report: every
  field of the result, from the energetics to the cluster reorganization and the per-term checks.

### Fixed
- **No stray files or full paths from a project.** Inside a project, `structure report` without
  `-o` wrote a second copy of its report beside the input file, outside the project; it now
  writes only its module result. The adsorption report printed the adduct path in full even in
  `relative` mode.
- **Spin and charge through the adsorption workflow.** `geometry` writes `charge=` and `mult=`
  in each configuration's comment line, with the probe's ground state (O2 is a triplet) coupled
  high-spin to the cluster's; `prescreen` (whose `--mult` is now `auto`), `rank` and `prepare`
  carry them on, so a triplet adduct no longer turns into a singlet by electron-count parity.
  A spin read for a different charge than `--charge` is not used, and says so. `analyze` warns
  when the adduct's charge is not the cluster's plus the probe's, when its multiplicity cannot
  come from coupling theirs, and when the free probe is not in its ground spin state.

- **Free energies of adsorption.** `analyze` no longer reports ΔH and ΔG when it refuses E_ads
  because a geometry did not converge. ΔG uses G corrected to the symmetry number of each
  geometry, as `stability` does, and ΔG with Grimme's quasi-RRHO entropy is added
  (`delta_gq_ads_kcal`).

- **`pipeline --resume` keeps the level of theory.** The manifest saves the calculation
  parameters; a resume without level options uses them. A different level is accepted while no
  calculation has run (the inputs are written again) and refused once outputs exist.

- **Dictionary.** The BSSE entry explains the counterpoise correction with its equations, and
  the free-energy entry the σ correction and the quasi-RRHO ΔG.
- **Gaussian counterpoise jobs** are recognized without `--bsse`: their E_ads is already
  corrected and is no longer labelled "no BSSE".
- **Activation by bending.** A probe that is linear when free (CO2) counts as activated when it
  bends by 10° or more in the adduct, not only when a bond stretches by 5 %; the probe's angles
  are reported (`probe_angles`, `max_bend_deg`).
- **Colliding names.** `stability-series` gives terms with the same label (`opt.out` in two
  folders) distinct ones instead of merging their rows, and `prescreen` gives configs with the
  same name distinct scratch folders and outputs instead of overwriting one with the other.
- **Exit codes and partial runs.** `rank` fails when none of its outputs could be ranked; a
  `pipeline` stopped by `--until` says what is still to run instead of "complete"; `analyze`
  on several adducts keeps going past one that cannot be read and lists it.
- **Windows timeout.** A job that hits its time limit is ended with its whole process tree on
  Windows too (`taskkill /T`), where process groups do not exist. CI runs the suite on macOS and
  Windows as an experimental job; they are not declared supported until it passes.
- **`structure report -o`** with a relative path writes to the current folder, like every other
  command, not to the folder of the analyzed file.
- **Results that cannot be mixed.** A result in `results/` is never overwritten by one
  computed from other files (a second CDFT, an adduct with the same name in another folder):
  it goes to the next free copy, `cdft-2.json`, `site5-2.json`…, and the same inputs run again
  update their own file. That name is the result's persistent id (`config_id`), which `report`
  uses to collapse duplicate minima across `analyze` runs. `cadmia report` stops when `results/` holds CDFT,
  adsorption or stability results of different clusters (by formula) and names them. Paths are
  saved absolute, so the level of theory is found from any folder.
- **A new `pipeline` on a used folder** is refused when it already holds calculations (continue
  it with `--resume`); `--nprocs` and memory no longer count as a change of level on resume,
  and a manifest from 0.3.0 keeps no default level it never ran.
- **Spin, the remaining paths.** An XYZ with only `charge=` no longer counts as carrying the
  spin; `prescreen` uses the file's multiplicity only for the same charge; `prepare cdft` and
  `benchmark` take the multiplicity the geometry carries (the ions at m − 1); `pipeline` gains
  `--charge` and shows the calculation stage's warnings.
- **Each output checked with its own engine.** `analyze` and `stability` read the convergence
  and termination markers of every output with the engine that wrote it.
- **`rank`** no longer records unreadable outputs as kept; `cdft` saves the reason when the
  full report cannot be built; JSON outputs write NaN as null; `report -o DIR/NAME` creates the
  folder; `--simple` and `structure --report` with a subcommand say they do not apply.

- **Figures.** `structure plot` in a batch with an `-o` that names a file is an error instead
  of a folder called `figs.png`; `--sigma` ≤ 0 is refused and
  the grid padding grows with σ; the static figures no longer change matplotlib's backend or
  settings in a Jupyter session; the IR sticks have their own right axis in km/mol.

### Changed
- **80-column text reports.** The structural and CDFT reports and `cadmia.out` wrap their prose at
  80 columns, as Gaussian and ORCA do; tables keep their layout and stay within 100. The
  directed-pair statistics are now a table with one column per value (the ECN plots read both
  layouts), and the polyhedra table is narrower.
- **DOS units.** DOS and PDOS are in states/eV over both spins: in a closed shell each spatial
  orbital now counts 2, as α and β already did in an open shell. Every DOS/PDOS output changes
  accordingly. Multiwfn weighs each spatial orbital 1, so its closed-shell amplitudes are half
  of CADMIA's; level positions and curve shapes agree. The view stops where the curve ends, and the DOS/PDOS HTML pages are much smaller (the empty stretches between core levels
  are left out) and `.dat`/`.json` use shorter number formats.
- **Paths in text reports.** The structural report, the CDFT report and `cadmia.out` print paths
  relative to the project (or the file name), so a shared report does not carry the user's
  folders; `cadmia paths absolute` prints them in full, with a warning (`cadmia paths relative`
  goes back; the mode is kept in `~/.cadmia/config.yaml`).
  JSON results and the database keep the absolute path and the hash either way.
- **No more alpha label.** CADMIA follows Semantic Versioning without development stages: before
  1.0.0 a minor version may change commands or output formats (each change is listed here), and
  1.0.0 will mark a stable public interface. The README and the guides install from PyPI.
- **`structure plot` is 2D only.** The 3D kinds `molecule` and `ecn` are removed (3D views of a
  structure belong to a molecular viewer); the default kind is `bonds`, and the interactive HTML
  loads plotly from its CDN, as the `electronic` pages do (kilobytes instead of ~5 MB).
- **Fixed module numbers in the output:** 1 Structure, 2 CDFT, 3 Adsorption, 4 Stability,
  whatever else the report holds.
- **`cadmia report` selects what goes in.** With no options inside a project it takes the
  original results and lists the copies it left out; each module option takes one or more files
  (`--cdft results/cdft.json results/cdft-2.json`, or a shell glob) or `all` for every result of
  that module, and then only the modules given are reported.
- **`cadmia init`** creates `configs/`, `results/` and `reports/` (no more unused `scf/` and
  `adsorption/`); inside a project `geometry` writes to `configs/<PROBE>/` and `prescreen` to
  `prescreened/` next to the configs it read.
- **`cadmia report` writes the CADMIA output.** With no stage options inside a project it reads
  `results/` and writes `reports/cadmia.out`: a fixed-width text file with the header, every
  stage's full report under its own frame, the correlations across stages and the closing block.
  `-o NAME.md` still writes the Markdown summary.
- **One CDFT JSON.** `cdft.json` now also carries the full report's data (`atoms`, `global_fd`,
  `global_koopmans`, `states`), which `--report-format json` wrote to a separate file, and its
  text. `cadmia cdft --report FILE` alone, inside a project, writes the report from
  `results/cdft.json` without recomputing. The same holds for `analyze` and `stability`
  (`--report`/`-r FILE`, with inputs or alone) and for `structure --report FILE`.

### Removed
- **3D views move out of CADMIA.** `structure render` (POV-Ray), `electronic cube` (HTML cube
  viewer), `electronic vmd` (VMD/Tachyon script) and `structure plot --kind molecule|ecn` are
  removed: CADMIA's figures are 2D, and densities, orbitals and cubes open in a wavefunction
  viewer. `scikit-image` leaves the `[full]` extra.
- **`cadmia view`** (the 3Dmol.js page) is removed. For a quick look at a set of structures the
  guide recommends MOLE, a separate lightweight viewer (`pip install mole-viewer`, then
  `mole file.xyz`).

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
