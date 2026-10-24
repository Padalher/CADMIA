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
from cadmia.io.gaussian import parse_gaussian_geometry
from cadmia.io.orca import parse_orca_geometry
from cadmia.io.xyz import parse_xyz, parse_xyz_database
from cadmia.structure.rendering.geometry import align_to_inertia
from cadmia.structure.rendering.povray import (
    RESOLUTION_PRESETS,
    check_dependencies,
    render_pov,
    write_pov_scene,
)
from cadmia.structure.reports import write_batch_report, write_report
from cadmia.structure.structural.analyzer import analyze_structure
from cadmia.structure.volumetric.cube_io import ISO_PRESETS, extract_isosurfaces, upsample_grid


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

Analyze molecular and cluster coordinate files, compute electronic descriptors,
write structural reports, and prepare POV-Ray render scenes.
"""

ROOT_EPILOG = """\
Main workflows:
  structure   Analyze geometry, bonds, CN/ECN, local descriptors, bond
              statistics, coordination polyhedra, and distortion descriptors.
  plot        Write interactive Plotly HTML plots for structures and
              structural descriptors.
  render      Write a POV-Ray scene from a structure, optionally adding cube
              isosurfaces, upsampling/smoothing, inertia alignment, and PNG
              rendering when POV-Ray is installed.

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
  cadmia structure plot cluster.xyz --kind ecn
  cadmia structure plot clusters/*.xyz --kind bonds -o plots
  cadmia structure render cluster.xyz -o scene
  cadmia structure render cluster.xyz --cube orbital.cube --iso-preset homo_lumo -o orbital
  cadmia structure render cluster.xyz -o scene --render --resolution screen

More detail:
  cadmia structure --help
  cadmia structure plot --help
  cadmia structure render --help
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

RENDER_DESCRIPTION = """\
Write a POV-Ray scene from a structure and optionally render it to PNG.

The renderer can include positive/negative cube isosurfaces, field-specific
isovalue presets, interpolation/smoothing for volumetric grids, and optional
mass-weighted inertia alignment before writing the scene.
"""

RENDER_EPILOG = """\
Isosurface presets:
  homo_lumo  +0.040 / -0.040
  fukui      +0.010 / -0.010
  dual       +0.005 / -0.005
  density    +0.002
  mep        +0.050 / -0.050
  adndp      +10.000 / -26.900

Resolution presets:
  preview       512 x 512
  screen       1024 x 1024
  publication  2560 x 2560
  ultra        5120 x 5120

Examples:
  cadmia structure render TiO2_monomer.xyz -o tio2_scene
  cadmia structure render TiO2_monomer.xyz --align-inertia -o aligned
  cadmia structure render TiO2_monomer.xyz --cube orbital.cube --iso-preset homo_lumo -o orbital
  cadmia structure render TiO2_monomer.xyz --cube fukui.cube --iso-pos 0.01 --iso-neg -0.01 --zoom 3 --sigma 1.0 -o fukui
  cadmia structure render TiO2_monomer.xyz -o tio2_scene --render --resolution publication
"""

PLOT_DESCRIPTION = """\
Write interactive Plotly HTML plots from structure files and structural reports.

The default plot is a 3D molecular view with cutoff-based bonds. Other plot
kinds expose the same structural analysis already used by the structure command.
Batch mode accepts multiple structure files, directories, and glob patterns.
ECN trend and heatmap plots read a TXT structural report or a table with
n and *_ECN columns.
"""

PLOT_EPILOG = """\
Plot kinds:
  molecule    3D molecule colored by element.
  ecn         3D molecule colored by Hoppe ECN.
  bonds       Bond-distance histogram by bond type.
  distances   Interatomic-distance heatmap.
  ecn-trend   ECN average by coordination type vs cluster size n.
  ecn-heatmap ECN heatmap by coordination type and cluster size n.

Examples:
  cadmia structure plot cluster.xyz
  cadmia structure plot cluster.xyz -o molecule.html
  cadmia structure plot cluster.xyz --kind ecn
  cadmia structure plot cluster.xyz --kind distances
  cadmia structure plot clusters/*.xyz --kind bonds -o plots
  cadmia structure plot structural_report.txt --kind ecn-trend
  cadmia structure plot structural_report.txt --kind ecn-heatmap
  cadmia structure plot structural_report.txt --kind ecn-heatmap --ecn-label rounded
"""


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
            "Report filename to create in the analyzed folder. Absolute paths "
            "are honored."
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
        help="Write interactive Plotly HTML plots for structures.",
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
        metavar="HTML",
        help=(
            "Output HTML file for one plot, or output directory for batch mode. "
            "Defaults to the current directory."
        ),
    )
    plot.add_argument(
        "--kind",
        choices=[
            "molecule",
            "ecn",
            "bonds",
            "distances",
            "ecn-trend",
            "ecn-heatmap",
        ],
        default="molecule",
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
        help="Display decimals for rounded ECN labels in ecn-trend/ecn-heatmap.",
    )
    plot.add_argument(
        "--ecn-label",
        choices=["raw", "rounded", "both"],
        default="both",
        metavar="STYLE",
        help=(
            "ECN label style for ecn-trend/ecn-heatmap: raw, rounded, "
            "or both as raw (rounded)."
        ),
    )
    plot.set_defaults(func=_cmd_plot)

    render = subparsers.add_parser(
        "render",
        help="Write or render POV-Ray scenes, with optional cube isosurfaces.",
        description=RENDER_DESCRIPTION,
        epilog=RENDER_EPILOG,
        formatter_class=_HelpFormatter,
    )
    render.add_argument(
        "input_file",
        metavar="INPUT",
        help="Structure file: .xyz, .cube, Gaussian .out/.log, or ORCA .out/.log.",
    )
    render.add_argument(
        "-o",
        "--output",
        default="structure_scene",
        metavar="PATH",
        help="Output scene path. The .pov suffix is added when omitted.",
    )
    render.add_argument(
        "--cube",
        metavar="CUBE",
        help="Optional Gaussian cube file used to extract isosurfaces.",
    )
    render.add_argument(
        "--field-type",
        default="other",
        metavar="NAME",
        help="Field label stored with the cube grid, such as homo, lumo, density, fukui, or mep.",
    )
    render.add_argument(
        "--iso-preset",
        choices=sorted(ISO_PRESETS),
        metavar="PRESET",
        help="Named positive/negative isovalue pair for common fields.",
    )
    render.add_argument(
        "--iso-pos",
        type=float,
        metavar="VALUE",
        help="Positive isovalue. Overrides the positive value from --iso-preset.",
    )
    render.add_argument(
        "--iso-neg",
        type=float,
        metavar="VALUE",
        help="Negative isovalue. Overrides the negative value from --iso-preset.",
    )
    render.add_argument(
        "--zoom",
        type=int,
        default=1,
        metavar="N",
        help="Volumetric upsampling factor for cube isosurfaces.",
    )
    render.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Gaussian smoothing sigma applied after volumetric upsampling.",
    )
    render.add_argument(
        "--align-inertia",
        action="store_true",
        help="Align atoms and isosurfaces to mass-weighted principal inertia axes.",
    )
    render.add_argument(
        "--orientation",
        metavar="JSON",
        help="Orientation JSON exported by `cadmia view`: render in that exact framed pose.",
    )
    render.add_argument(
        "--render",
        action="store_true",
        help="Run POV-Ray after writing the scene and create a PNG.",
    )
    render.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_PRESETS),
        default="preview",
        metavar="PRESET",
        help="PNG render size preset used with --render.",
    )
    render.set_defaults(func=_cmd_render)

    return parser


def _cmd_structure(args) -> int:
    config = load_config(args.config)
    input_paths = _expand_structure_inputs(args.input_file)
    results = []
    for input_path in input_paths:
        for structure in _parse_structures(input_path):
            results.append(analyze_structure(structure, config))

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
        requested_path = Path(requested_output)
        if requested_path.is_absolute():
            return requested_path
        return analysis_dir / requested_path.name

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
) -> str:
    parts = [_safe_filename_part(Path(input_file).stem)]
    if structures_in_file > 1:
        structure_id = _safe_filename_part(structure.structure_id, f"structure_{structure_index + 1}")
        if structure_id == parts[0]:
            structure_id = f"structure_{structure_index + 1}"
        parts.append(structure_id)
    parts.append(kind)
    return "_".join(parts) + ".html"


def _plot_output_path(
    input_file: str | Path,
    structure,
    structure_index: int,
    structures_in_file: int,
    kind: str,
    requested_output: str | None,
    is_batch: bool,
) -> Path:
    filename = _plot_default_name(
        input_file,
        structure,
        structure_index,
        structures_in_file,
        kind,
    )
    if is_batch:
        output_dir = Path(requested_output) if requested_output else Path(".")
        if output_dir.suffix.lower() == ".html":
            raise ValueError("For batch plotting, --output must be a directory.")
        return output_dir / filename

    if requested_output:
        path = Path(requested_output)
        if path.exists() and path.is_dir():
            return path / filename
        if path.suffix.lower() != ".html":
            path = path.with_suffix(".html")
        return path

    return Path(filename)


def _write_plot(structure, result, kind: str, title: str, output_path: Path) -> None:
    try:
        if kind in {"molecule", "ecn"}:
            from cadmia.structure.visualization.molecular import plot_molecule_3d

            color_by = "ecn" if kind == "ecn" else "element"
            plot_molecule_3d(
                structure,
                result=result,
                color_by=color_by,
                title=title,
                output_html=str(output_path),
            )
        elif kind == "bonds":
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


def _ecn_table_plot_output_path(kind: str, requested_output: str | None) -> Path:
    filename = "ecn_trend.html" if kind == "ecn-trend" else "ecn_heatmap.html"
    if requested_output is None:
        return Path(filename)
    path = Path(requested_output)
    if path.suffix.lower() == ".html":
        return path
    return path / filename


def _cmd_ecn_table_plot(args) -> int:
    if len(args.input_file) != 1:
        raise ValueError("ECN trend and heatmap plots accept one report/table file.")

    input_path = args.input_file[0]
    output_path = _ecn_table_plot_output_path(args.kind, args.output)
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


def _cmd_plot(args) -> int:
    if args.kind in {"ecn-trend", "ecn-heatmap"}:
        return _cmd_ecn_table_plot(args)

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
        )
        title = args.title if args.title is not None else structure.structure_id
        _write_plot(structure, result, args.kind, title, output_path)
        output_paths.append(output_path)

    if len(output_paths) == 1:
        print(f"Plot written: {output_paths[0]}")
    else:
        print(f"Plots written: {len(output_paths)}")
        for output_path in output_paths:
            print(f"  {output_path}")
    return 0


def _cmd_render(args) -> int:
    structure = _parse_structure(args.input_file)
    pos_iso = None
    neg_iso = None

    if args.cube:
        grid = parse_cube(args.cube, field_type=args.field_type)
        if args.zoom > 1 or args.sigma > 0:
            grid = upsample_grid(grid, zoom=args.zoom, sigma=args.sigma)
        preset = ISO_PRESETS.get(args.iso_preset) if args.iso_preset else (None, None)
        iso_pos = args.iso_pos if args.iso_pos is not None else preset[0]
        iso_neg = args.iso_neg if args.iso_neg is not None else preset[1]
        pos_iso, neg_iso = extract_isosurfaces(grid, iso_pos=iso_pos, iso_neg=iso_neg)

    if args.align_inertia:
        structure, pos_iso, neg_iso = align_to_inertia(structure, pos_iso, neg_iso)

    orientation = None
    if getattr(args, "orientation", None):
        from cadmia.structure.rendering.orientation import load_orientation
        orientation = load_orientation(args.orientation)

    pov_path = write_pov_scene(structure, pos_iso, neg_iso, args.output, orientation=orientation)
    print(f"POV scene written: {pov_path}")

    if args.render:
        deps = check_dependencies()
        if not deps["povray"]:
            raise RuntimeError("POV-Ray is not available in PATH.")
        render_pov(str(pov_path), RESOLUTION_PRESETS[args.resolution])
        print(f"Rendered PNG: {pov_path.with_suffix('.png')}")
    return 0


def main(argv=None):
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {
        "structure",
        "cdft",
        "cdft-workflow",
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
