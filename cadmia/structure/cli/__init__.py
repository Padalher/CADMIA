"""argparse handlers behind `cadmia structure`."""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

from cadmia.structure.core.config import load_config
from cadmia.io.cube import parse_cube
from cadmia.io.figures import FIGURE_FORMATS, FIGURE_WIDTHS_IN, resolve_format
from cadmia.io.gaussian import parse_gaussian_geometry
from cadmia.io.orca import parse_orca_geometry
from cadmia.io.xyz import parse_xyz, parse_xyz_database
from cadmia.structure.reports import write_batch_report, write_report
from cadmia.structure.structural.analyzer import analyze_structure


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve examples while showing defaults for CLI options."""

    def _get_help_string(self, action):
        text = action.help
        if text is argparse.SUPPRESS:
            return text
        if "%(default)" not in text and action.option_strings:
            if action.default not in (None, False, argparse.SUPPRESS):
                text += " (default: %(default)s)"
        return text


ROOT_DESCRIPTION = """\
CADMIA structure - Structural analysis

Analyze molecular and cluster coordinate files and write structural reports and
2D plots.
"""

ROOT_EPILOG = """\
Main workflows:
  structure   Analyze geometry, bonds, CN/ECN, local descriptors, bond
              statistics, coordination polyhedra, and distortion descriptors.
  plot        Write interactive HTML plots, or static PNG/SVG/PDF figures,
              for structures and structural descriptors.

Supported structure inputs:
  .xyz        XYZ coordinate files, including concatenated multi-structure XYZ
  .txt        Concatenated XYZ coordinate databases
  .cube       Gaussian cube files; structure is read from the atom block
  .out/.log   Gaussian or ORCA output geometries

Report formats:
  txt         Human-readable structural report
  csv         Flat bond/local-descriptor table
  json        Typed structural result payload

Examples:
  cadmia structure report cluster.xyz -o report.txt
  cadmia structure report *.xyz -o structures.csv --format csv
  cadmia structure report cluster.xyz -o report.json --format json
  cadmia structure report calculation.out --config atomic_data.yaml
  cadmia structure plot cluster.xyz
  cadmia structure plot cluster.xyz --kind distances -o distances.png
  cadmia structure plot clusters/*.xyz --kind bonds -o plots

More detail:
  cadmia structure --help
  cadmia structure plot --help
"""

STRUCTURE_DESCRIPTION = """\
Analyze one structure and write a structural report.

The analysis computes molecular formula, atom labels, cutoff-based bonds,
coordination numbers, Hoppe effective coordination numbers (ECN), local
environment descriptors, directed central-neighbor ECN/distance statistics,
bond statistics, ECN statistics, coordination polyhedra, and distortion
descriptors.
"""

STRUCTURE_EPILOG = """\
Concepts:
  CN           Cutoff-based coordination number.
  ECN          Hoppe self-consistent effective coordination number.
  d_av         Principal average bond length: Hoppe self-consistent
               weighted distance used to compute ECN.
  ECN_act      Mean pair ECN over centers that have that pair.
  ECN_1d       One-decimal ECN display value.
  Bonds        Unique cutoff-filtered bonds; Ti-Ti and Al-Al are not doubled.
  DirB         Directed central-neighbor incidences used for CN/atom.
  MeanAux      Auxiliary arithmetic mean of cutoff-filtered bond distances.

Examples:
  cadmia structure report TiO2_monomer.xyz
  cadmia structure report *.xyz -o structures.csv --format csv
  cadmia structure report *.xyz -o structures.txt --details
  cadmia structure report TiO2_monomer.xyz -o tio2.txt
  cadmia structure report TiO2_monomer.xyz -o tio2.csv --format csv
  cadmia structure report TiO2_monomer.xyz -o tio2.json --format json
  cadmia structure report calculation.log --config atomic_data.yaml
"""

PLOT_DESCRIPTION = """\
Write interactive HTML plots or static PNG/SVG/PDF figures from structure files and
structural reports.

The plots are 2D and expose the same structural analysis the structure command
reports; 3D views of a structure are left to a molecular viewer.
Batch mode accepts multiple structure files, directories, and glob patterns.
ECN trend and heatmap plots read a TXT structural report or a table with
n and *_ECN columns.
"""

PLOT_EPILOG = """\
Plot kinds:
  bonds       Bond-distance histogram by bond type.
  distances   Interatomic-distance heatmap.
  ecn-trend   ECN average by coordination type vs cluster size n.
  ecn-heatmap ECN heatmap by coordination type and cluster size n.

Output format: --format if given, else the -o suffix, else html. html is interactive (plotly);
png, svg and pdf are static, publication-ready figures (matplotlib) for bonds, distances,
ecn-trend and ecn-heatmap; --width single|double sets their width (3.4 in or 6.5 in). In batch
mode -o is a directory, so pick the file type with --format.

Examples:
  cadmia structure plot cluster.xyz
  cadmia structure plot cluster.xyz -o molecule.html
  cadmia structure plot cluster.xyz --kind ecn
  cadmia structure plot cluster.xyz --kind distances
  cadmia structure plot clusters/*.xyz --kind bonds -o plots
  cadmia structure plot structural_report.txt --kind ecn-trend
  cadmia structure plot structural_report.txt --kind ecn-heatmap
  cadmia structure plot structural_report.txt --kind ecn-heatmap --ecn-label rounded
  cadmia structure plot structural_report.txt --kind ecn-trend -o ecn_trend.pdf --width double
  cadmia structure plot clusters/*.xyz --kind bonds --format png -o plots
"""


_PLOT_FORMATS = ("html", *FIGURE_FORMATS)


def _parse_structure(path: str | Path):
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".xyz":
        return parse_xyz(file_path)
    if suffix == ".cube":
        return parse_cube(file_path).structure
    if suffix in {".out", ".log"}:
        try:
            return parse_gaussian_geometry(file_path)
        except ValueError:
            return parse_orca_geometry(file_path)
    raise ValueError(f"Unsupported structure input: {file_path}")


def _parse_structures(path: str | Path):
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".xyz":
        structures = parse_xyz_database(file_path)
        if len(structures) == 1:
            structures[0].structure_id = file_path.stem
        return structures
    if suffix == ".txt":
        return parse_xyz_database(file_path)
    return [_parse_structure(file_path)]


def _expand_structure_inputs(input_values: list[str]) -> list[Path]:
    supported_suffixes = {".xyz", ".cube", ".out", ".log", ".txt"}
    expanded = []

    for value in input_values:
        path = Path(value)
        if path.exists() and path.is_dir():
            # A .txt is taken only when named: in a directory it is usually a report, often
            # the one a previous run wrote there.
            expanded.extend(
                child
                for child in sorted(path.iterdir())
                if child.is_file() and child.suffix.lower() in supported_suffixes - {".txt"}
            )
            continue
        if path.exists() and path.is_file():
            expanded.append(path)
            continue

        matches = [
            Path(match)
            for match in sorted(glob.glob(value))
            if Path(match).is_file()
            and Path(match).suffix.lower() in supported_suffixes
        ]
        expanded.extend(matches)

    unique = []
    seen = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)

    if not unique:
        raise FileNotFoundError(
            "No supported structure files found. Supported suffixes: "
            ".xyz, .cube, .out, .log, .txt"
        )
    return unique


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadmia structure",
        description=ROOT_DESCRIPTION,
        epilog=ROOT_EPILOG,
        formatter_class=_HelpFormatter,
    )
    from cadmia import __version__ as _cadmia_version
    parser.add_argument(
        "--version",
        action="version",
        version=f"CADMIA structure {_cadmia_version}",
        help="Show the version and exit.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="commands",
        description="Run 'cadmia structure COMMAND --help' for command-specific options.",
    )

    structure = subparsers.add_parser(
        "structure",
        help="Analyze bonds, CN/ECN, polyhedra, and write TXT/CSV/JSON reports.",
        description=STRUCTURE_DESCRIPTION,
        epilog=STRUCTURE_EPILOG,
        formatter_class=_HelpFormatter,
    )
    structure.add_argument(
        "input_file",
        nargs="+",
        metavar="INPUT",
        help=(
            "Structure file(s), directory, or glob pattern: .xyz, .cube, "
            "Gaussian .out/.log, ORCA .out/.log, or concatenated XYZ .txt. "
            "For ecn-trend/ecn-heatmap, pass a structural report or table."
        ),
    )
    structure.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help=(
            "Report file, relative to the current folder. Without it the report is "
            "structural_report.<format> in the analyzed folder; inside a project it is "
            "only the module result in results/."
        ),
    )
    structure.add_argument(
        "--format",
        choices=["txt", "csv", "json"],
        metavar="FORMAT",
        help="Report format. If omitted, the output file suffix is used.",
    )
    structure.add_argument(
        "--config",
        metavar="YAML",
        help="Custom YAML configuration with atomic masses, cutoffs, and ECN settings.",
    )
    structure.add_argument(
        "--details",
        action="store_true",
        help=(
            "For TXT batch reports, append full per-structure sections after "
            "the global summary."
        ),
    )
    structure.set_defaults(func=_cmd_structure)


    plot = subparsers.add_parser(
        "plot",
        help="Write interactive HTML plots or static PNG/SVG/PDF figures for structures.",
        description=PLOT_DESCRIPTION,
        epilog=PLOT_EPILOG,
        formatter_class=_HelpFormatter,
    )
    plot.add_argument(
        "input_file",
        nargs="+",
        metavar="INPUT",
        help=(
            "Structure file(s), directory, or glob pattern: .xyz, .cube, "
            "Gaussian .out/.log, ORCA .out/.log, or concatenated XYZ .txt. "
            "For ecn-trend/ecn-heatmap, pass a structural report or table."
        ),
    )
    plot.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help=(
            "Output file for one plot (its suffix picks the format unless --format is given), "
            "or output directory for batch mode. Defaults to the current directory."
        ),
    )
    plot.add_argument(
        "--format",
        choices=_PLOT_FORMATS,
        default=None,
        metavar="FORMAT",
        help=(
            "html (interactive), or png/svg/pdf (static figure). "
            "Default: the -o suffix, else html."
        ),
    )
    plot.add_argument(
        "--width",
        choices=list(FIGURE_WIDTHS_IN),
        default="single",
        help="Static figure width: single = 3.4 in, double = 6.5 in.",
    )
    plot.add_argument(
        "--kind",
        choices=[
            "bonds",
            "distances",
            "ecn-trend",
            "ecn-heatmap",
        ],
        default="bonds",
        metavar="KIND",
        help="Plot kind.",
    )
    plot.add_argument(
        "--title",
        metavar="TEXT",
        help="Optional figure title for molecule and ECN plots.",
    )
    plot.add_argument(
        "--config",
        metavar="YAML",
        help="Custom YAML configuration with atomic masses, cutoffs, and ECN settings.",
    )
    plot.add_argument(
        "--ecn-decimals",
        type=int,
        default=1,
        metavar="N",
        help="Display decimals for rounded ECN labels (heatmap cells; hover text of the html trend).",
    )
    plot.add_argument(
        "--ecn-label",
        choices=["raw", "rounded", "both"],
        default="both",
        metavar="STYLE",
        help=(
            "ECN label style (heatmap cells; hover text of the html trend): raw, rounded, "
            "or both as raw (rounded)."
        ),
    )
    plot.set_defaults(func=_cmd_plot)

    return parser


def _cmd_structure(args) -> int:
    config = load_config(args.config)
    input_paths = _expand_structure_inputs(args.input_file)
    results = []
    for input_path in input_paths:
        for structure in _parse_structures(input_path):
            results.append(analyze_structure(structure, config))

    from cadmia.io.results import save_stage_result
    from cadmia.structure.reports.generator import structural_payload, structural_text
    in_project = False
    for result in results:
        item = _safe_filename_part(result.structure.structure_id)
        saved = save_stage_result("structure", structural_payload(result, config),
                                  structural_text(result, config, program_header=False),
                                  item=item)
        if saved:
            in_project = True
            print(f"Module result: {saved}")
    if in_project and not args.output:
        # results/ already holds the full report; a second copy beside the input would land
        # outside the project, in a folder nobody asked to write to
        return 0

    report_output = _structure_report_output(args.output, input_paths, args.format)
    if len(results) == 1:
        report_path = write_report(
            results[0],
            report_output,
            output_format=args.format,
            config=config,
        )
        print(f"Report written: {report_path}")
        return 0

    report_path = write_batch_report(
        results,
        report_output,
        output_format=args.format,
        config=config,
        include_details=args.details,
    )
    print(f"Batch structural report written: {report_path}")
    print(f"Structures analyzed: {len(results)}")
    return 0


def _structure_report_output(
    requested_output: str | None,
    input_paths: list[Path],
    output_format: str | None,
) -> Path:
    analysis_dir = Path(
        os.path.commonpath([str(path.resolve().parent) for path in input_paths])
    )
    if requested_output:
        # Like every other command: a relative -o is relative to where the command runs.
        return Path(requested_output)

    fmt = output_format or "txt"
    suffix = fmt if fmt in {"txt", "csv", "json"} else "txt"
    return analysis_dir / f"structural_report.{suffix}"


def _safe_filename_part(value: object, fallback: str = "structure") -> str:
    text = str(value or "").strip() or fallback
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or fallback


def _plot_default_name(
    input_file: str | Path,
    structure,
    structure_index: int,
    structures_in_file: int,
    kind: str,
    fmt: str = "html",
) -> str:
    parts = [_safe_filename_part(Path(input_file).stem)]
    if structures_in_file > 1:
        structure_id = _safe_filename_part(structure.structure_id, f"structure_{structure_index + 1}")
        if structure_id == parts[0]:
            structure_id = f"structure_{structure_index + 1}"
        parts.append(structure_id)
    parts.append(kind)
    return "_".join(parts) + f".{fmt}"


def _plot_output_path(
    input_file: str | Path,
    structure,
    structure_index: int,
    structures_in_file: int,
    kind: str,
    requested_output: str | None,
    is_batch: bool,
    fmt: str = "html",
) -> Path:
    filename = _plot_default_name(
        input_file,
        structure,
        structure_index,
        structures_in_file,
        kind,
        fmt,
    )
    if is_batch:
        output_dir = Path(requested_output) if requested_output else Path(".")
        if output_dir.suffix.lower().lstrip(".") in _PLOT_FORMATS:
            raise ValueError(
                f"For batch plotting, --output must be a directory, not {output_dir}; "
                "choose the file type with --format."
            )
        return output_dir / filename

    if requested_output:
        path = Path(requested_output)
        if path.exists() and path.is_dir():
            return path / filename
        return path.with_suffix(f".{fmt}")

    return Path(filename)


def _write_plot(structure, result, kind: str, title: str, output_path: Path,
                fmt: str = "html", width: str = "single") -> None:
    if fmt in FIGURE_FORMATS:
        from cadmia.structure.visualization.figures import (
            bond_histogram_figure, distance_heatmap_figure)
        if kind == "bonds":
            bond_histogram_figure(result, output_path, width=width)
        else:
            distance_heatmap_figure(structure, output_path, width=width,
                                    bonds=result.bonds if result is not None else ())
        return
    try:
        if kind == "bonds":
            from cadmia.structure.visualization.plots import plot_bond_histogram

            plot_bond_histogram(result, output_html=str(output_path))
        elif kind == "distances":
            from cadmia.structure.visualization.plots import plot_distance_heatmap

            plot_distance_heatmap(structure, output_html=str(output_path))
        else:
            raise ValueError(f"Unsupported plot kind: {kind}")
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is not available. Install the optional features with: "
            "pip install '.[full]'"
        ) from exc


def _ecn_table_plot_output_path(kind: str, requested_output: str | None, fmt: str = "html") -> Path:
    filename = f"{'ecn_trend' if kind == 'ecn-trend' else 'ecn_heatmap'}.{fmt}"
    if requested_output is None:
        return Path(filename)
    path = Path(requested_output)
    if path.suffix.lower().lstrip(".") in _PLOT_FORMATS:
        return path.with_suffix(f".{fmt}")
    return path / filename


def _cmd_ecn_table_plot(args) -> int:
    if len(args.input_file) != 1:
        raise ValueError("ECN trend and heatmap plots accept one report/table file.")

    input_path = args.input_file[0]
    fmt = _plot_format(args)
    output_path = _ecn_table_plot_output_path(args.kind, args.output, fmt)
    if fmt in FIGURE_FORMATS:
        from cadmia.structure.visualization.figures import ecn_heatmap_figure, ecn_trend_figure
        width = getattr(args, "width", None) or "single"
        if args.kind == "ecn-trend":
            ecn_trend_figure(input_path, output_path, width=width)
        else:
            ecn_heatmap_figure(input_path, output_path, display_decimals=args.ecn_decimals,
                               label_style=args.ecn_label, width=width)
        print(f"Plot written: {output_path}")
        return 0
    try:
        if args.kind == "ecn-trend":
            from cadmia.structure.visualization.plot_generator import generate_ecn_trend_plot

            generate_ecn_trend_plot(
                input_path,
                output_path,
                display_decimals=args.ecn_decimals,
                label_style=args.ecn_label,
            )
        elif args.kind == "ecn-heatmap":
            from cadmia.structure.visualization.plot_generator import generate_ecn_heatmap

            generate_ecn_heatmap(
                input_path,
                output_path,
                display_decimals=args.ecn_decimals,
                label_style=args.ecn_label,
            )
        else:
            raise ValueError(f"Unsupported ECN table plot kind: {args.kind}")
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is not available. Install the optional features with: "
            "pip install '.[full]'"
        ) from exc

    print(f"Plot written: {output_path}")
    return 0


def _plot_format(args) -> str:
    return resolve_format(getattr(args, "format", None), args.output, _PLOT_FORMATS, "html")


def _cmd_plot(args) -> int:
    if args.kind in {"ecn-trend", "ecn-heatmap"}:
        return _cmd_ecn_table_plot(args)

    fmt = _plot_format(args)
    config = load_config(args.config)
    input_paths = _expand_structure_inputs(args.input_file)
    jobs = []
    for input_path in input_paths:
        structures = _parse_structures(input_path)
        for index, structure in enumerate(structures):
            jobs.append((input_path, structure, index, len(structures)))

    is_batch = len(jobs) > 1
    output_paths = []
    for input_path, structure, index, structures_in_file in jobs:
        result = analyze_structure(structure, config)
        output_path = _plot_output_path(
            input_path,
            structure,
            index,
            structures_in_file,
            args.kind,
            args.output,
            is_batch,
            fmt,
        )
        title = args.title if args.title is not None else structure.formula
        _write_plot(structure, result, args.kind, title, output_path, fmt,
                    getattr(args, "width", None) or "single")
        output_paths.append(output_path)

    if len(output_paths) == 1:
        print(f"Plot written: {output_paths[0]}")
    else:
        print(f"Plots written: {len(output_paths)}")
        for output_path in output_paths:
            print(f"  {output_path}")
    return 0


def main(argv=None):
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {
        "structure",
        "plot",
        "render",
        "-h",
        "--help",
        "--version",
    }
    if argv and argv[0] not in known:
        argv = ["structure", *argv]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
