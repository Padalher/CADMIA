"""CADMIA CLI — main entry point."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import click
from click.core import ParameterSource

from cadmia import __version__
from cadmia.cli_help_es import EsGroup
from cadmia.core.provenance import command_line
from cadmia.agents.orchestrator import STAGES as _PIPELINE_STAGES
from cadmia.calculators.stages import STAGES as _WORKFLOW_STAGES

_STAGE_TOKENS = tuple(_WORKFLOW_STAGES)


def _cli_provenance(source_file: str | Path, parser: str):
    """Provenance for --db persistence: source file hash + the exact CLI invocation."""
    from cadmia import __version__
    from cadmia.core import Provenance
    return Provenance.from_file(
        Path(source_file).resolve(), parser=parser,
        command=command_line(),
        software_version=f"cadmia {__version__}",
    )


def _record_workflow_run(conn, agent: str, ids=None, status: str = "ok") -> None:
    """Register the current CLI invocation in workflow_runs."""
    from cadmia.db import repository
    repository.insert_workflow_run(
        conn, command=command_line(), agent=agent, status=status, ids=ids,
    )


_BANNER_ART = (
    " ██████╗ █████╗ ██████╗ ███╗   ███╗██╗ █████╗ ",
    "██╔════╝██╔══██╗██╔══██╗████╗ ████║██║██╔══██╗",
    "██║     ███████║██║  ██║██╔████╔██║██║███████║",
    "██║     ██╔══██║██║  ██║██║╚██╔╝██║██║██╔══██║",
    "╚██████╗██║  ██║██████╔╝██║ ╚═╝ ██║██║██║  ██║",
    " ╚═════╝╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝",
)
# (first day, last day) as (month, day), inclusive; the festive dates go first so they win.
_BANNER_THEMES = (
    ((6, 9), (6, 9), "🎂", ("#5fd7ff", "#00d7ff", "#00afff", "#0087ff", "#0087d7", "#005fd7")),
    ((10, 28), (11, 2), "🌼", ("#ffd700", "#ffaf00", "#ff8700", "#ff0087", "#d7008f", "#8700d7")),
    ((12, 1), (12, 31), "🎄", ("#ff0000", "#d70000", "#ffd700", "#00d700", "#00af00", "#008700")),
    ((1, 1), (1, 6), "🎄", ("#ff0000", "#d70000", "#ffd700", "#00d700", "#00af00", "#008700")),
    ((3, 1), (5, 31), "🌸", ("#ffafd7", "#d7afff", "#af87ff", "#af5fd7", "#5fd787", "#00af5f")),
    ((6, 1), (8, 31), "☀", ("#ffff5f", "#ffd700", "#ffaf00", "#ff8700", "#ff5f00", "#d70000")),
    ((9, 1), (11, 30), "🍂", ("#ffd75f", "#ffaf00", "#ff8700", "#d75f00", "#af5f00", "#af3700")),
    ((1, 1), (12, 31), "❄", ("#d7ffff", "#afffff", "#87d7ff", "#5fafff", "#5f87ff", "#5f5fff")),
)


def _banner_theme(today=None) -> tuple:
    """Glyph and six-colour gradient of the season (northern hemisphere) or festivity.

    9 June, when the package was created, keeps CADMIA's own cyan-to-blue.
    """
    import datetime

    d = today or datetime.date.today()
    key = (d.month, d.day)
    return next((glyph, colours) for start, end, glyph, colours in _BANNER_THEMES
                if start <= key <= end)


def _print_banner(today=None) -> None:
    """Print the CADMIA title and program information (shown on bare `cadmia`).

    The gradient follows the season, with its own colours for Día de Muertos and Christmas. rich
    drops the colour by itself when the output is not a terminal or NO_COLOR is set.
    """
    from rich.console import Console
    from cadmia import __version__

    console = Console(highlight=False)
    glyph, gradient = _banner_theme(today)
    coloured = console.color_system is not None
    console.print()
    for line, colour in zip(_BANNER_ART, gradient):
        console.print(line, style=f"bold {colour}", no_wrap=True, crop=False)
    console.print()
    console.print(" Computational Agent-Driven Molecular Intelligence Architecture", style="bold")
    console.print(f" Version {__version__} · GPL-3.0-or-later" + (f"  {glyph}" if coloured else ""),
                  style="dim")
    console.print(" Pablo Daniel Álvarez Hernández — DACB-UJAT", style="dim")
    console.print(" pablo.alvarez@padalher.com · padalher.com", style="dim")
    console.print()


# Curated shortlist shown on bare `cadmia` so the banner stays on screen; the full
# (grouped) command list lives in `cadmia --help`.
_COMMON_COMMANDS = [
    ("pipeline", "run the full study end to end"),
    ("cdft", "reactivity descriptors + reactive-site ranking"),
    ("geometry", "build adsorption configs from the CDFT ranking"),
    ("prescreen", "GFN2-xTB pre-screen of the configs"),
    ("rank", "order DFT outputs by energy → one .xyz"),
    ("prepare · submit · run", "make inputs · queue on SLURM · run locally"),
    ("analyze", "adsorption energetics + thermo + electronic/structural"),
    ("density", "density/HOMO/LUMO/MEP cubes + Fukui/dual/Δρ maps"),
    ("molden", "export .gbw → .molden for NPA (JANPA) / AIM (Multiwfn)"),
    ("report", "consolidate every agent's results into a report"),
    ("correlate", "regress descriptors vs experimental data (r²)"),
    ("doctor", "check the environment (ORCA, SLURM, xtb…)"),
]


def _print_overview() -> None:
    """Compact welcome for bare `cadmia`: common commands + where to find the rest."""
    width = max(len(name) for name, _ in _COMMON_COMMANDS)
    click.echo("Common commands:")
    for name, desc in _COMMON_COMMANDS:
        click.echo(f"  {name:<{width}}  {desc}")
    click.echo("")
    click.echo("All commands : cadmia --help          Manual : cadmia guide")
    click.echo("Help language: --help (English) · --ayuda (español) on any command.")


@click.group(cls=EsGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="cadmia")
@click.pass_context
def main(ctx: click.Context) -> None:
    """CADMIA — Computational Agent-Driven Molecular Intelligence Architecture."""
    if ctx.invoked_subcommand is None:
        _print_banner()
        _print_overview()


def _attach_nuclear_fukui(result, path_np1, path_nm1, path_n, top: int,
                          project_normal: bool = False) -> None:
    """Add the nuclear Fukui block to a CDFT result and print its ranked table."""
    import numpy as np

    from cadmia.cdft.nuclear import directional_phi, nuclear_fukui
    from cadmia.io.engrad import parse_engrad

    if not (path_np1 and path_nm1):
        raise click.UsageError(
            "the nuclear Fukui function is a difference between the two charged states: "
            "--engrad-np1 and --engrad-nm1 are needed together.")

    anion, cation = parse_engrad(path_np1), parse_engrad(path_nm1)
    neutral = parse_engrad(path_n) if path_n else None

    # result.sites is ranked by |Δf|, while the gradients are in the calculation's atom order.
    # Each Φ_A row must name its own nucleus, so the sites are re-indexed by atom_index first.
    by_index = {s.atom_index: s for s in result.sites}
    ordered = [by_index[i] for i in sorted(by_index)]
    if [s.atom_index for s in ordered] != list(range(1, len(ordered) + 1)):
        raise click.UsageError(
            "the CDFT sites are not numbered 1..N, so they cannot be lined up with the "
            "gradients' atom order.")
    # Counting atoms is not enough: the SCF outputs and the .engrad files are separate runs and
    # can come from differently ordered .xyz of the same cluster. Compare element by element, or
    # Phi_A ends up reported against another nucleus.
    from cadmia.core.elements import ATOMIC_NUMBER_TO_SYMBOL
    engrad_symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(z, "?") for z in anion.atomic_numbers]
    if [s.atom_symbol for s in ordered] != engrad_symbols:
        raise click.UsageError(
            "the .engrad atoms do not match the CDFT sites element by element. The gradients and "
            "the SCF outputs must come from the same geometry, in the same atom order.")
    nf = nuclear_fukui(anion, cation, neutral=neutral)
    result.warnings.extend(nf.warnings)

    magnitude = nf.magnitude
    dual_magnitude = (np.linalg.norm(nf.phi_dual, axis=1)
                      if nf.phi_dual is not None else None)

    along = None
    if project_normal:
        # The normals come from the SAME geometry the gradients were computed on, which the
        # .engrad carries; reading them from a separate .xyz would risk a different frame.
        from cadmia.agents.geometry_agent import _site_normal

        coords = anion.positions_bohr * 0.529177210903
        symbols = [ATOMIC_NUMBER_TO_SYMBOL[z] for z in anion.atomic_numbers]
        axes = np.array([_site_normal(symbols, list(coords), i)[0]
                         for i in range(len(symbols))])
        along = directional_phi(nf.phi, axes)
    rows = []
    for i, site in enumerate(ordered):
        row = {
            "atom_index": site.atom_index,
            "atom_symbol": site.atom_symbol,
            "phi": [round(float(v), 6) for v in nf.phi[i]],
            "phi_norm": round(float(magnitude[i]), 6),
        }
        if along is not None:
            row["phi_along"] = round(float(along[i]), 6)
        if nf.phi_plus is not None:
            # Φ⁺ ≠ Φ⁻ at integer N, exactly as f⁺ ≠ f⁻. Reported apart, never averaged.
            row["phi_plus_norm"] = round(float(np.linalg.norm(nf.phi_plus[i])), 6)
            row["phi_minus_norm"] = round(float(np.linalg.norm(nf.phi_minus[i])), 6)
        if nf.phi_dual is not None:
            row["phi_dual"] = [round(float(v), 6) for v in nf.phi_dual[i]]
            row["phi_dual_norm"] = round(float(dual_magnitude[i]), 6)
        rows.append(row)

    result.nuclear = {
        "units": "Phi and Phi(2) in Eh/bohr per electron",
        "has_neutral": neutral is not None,
        "sources": {k: v for k, v in
                    (("N", path_n), ("N+1", path_np1), ("N-1", path_nm1)) if v},
        "sites": rows,
    }

    ranked = sorted(rows, key=lambda r: r["phi_norm"], reverse=True)[:max(top, 1)]
    click.echo("\nNuclear Fukui — ranked by |Phi|")
    header = f"{'atom':>6}  {'df':>9}  {'|Phi|':>10}"
    if dual_magnitude is not None:
        header += f"  {'|Phi(2)|':>10}"
    click.echo(header)
    for row in ranked:
        df = by_index[row["atom_index"]].delta_f
        line = (f"{row['atom_symbol']}{row['atom_index']:<4} "
                f" {('%9.4f' % df) if df is not None else '        -'}"
                f"  {row['phi_norm']:10.6f}")
        if "phi_dual_norm" in row:
            line += f"  {row['phi_dual_norm']:10.6f}"
        click.echo(line)


@main.command("cdft")
@click.option("--n",   "path_n",   default=None, type=click.Path(exists=True),
              help="SCF output file with N electrons.")
@click.option("--np1", "path_np1", default=None, type=click.Path(exists=True),
              help="SCF output file with N+1 electrons (anion).")
@click.option("--nm1", "path_nm1", default=None, type=click.Path(exists=True),
              help="SCF output file with N-1 electrons (cation).")
@click.option("--cluster", default="", show_default=False,
              help="Cluster label for the report and the database. Defaults to the parent directory name.")
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Calculation engine. Auto-detected from file content when omitted.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write JSON output to this file (with --no-table and no -o, the JSON "
                   "goes to stdout).")
@click.option("--report", "-r", "report_path", default=None, type=click.Path(),
              help="Also write the full CDFT report (global + condensed "
                   "per-atom Fukui, local electrophilicity/nucleophilicity, softnesses) here.")
@click.option("--report-format", type=click.Choice(["txt", "json"]), default="txt",
              show_default=True, help="Format for --report.")
@click.option("--tce-homo", "tce_homo", default=None, type=float,
              help="HOMO(TCE) reference in eV for the nucleophilicity index (default: "
                   "Multiwfn B3LYP/6-31G* value). Set to use a same-level reference.")
@click.option("--top", default=5, show_default=True,
              help="Number of sites to show in the ranked table.")
@click.option("--no-table", "no_table", is_flag=True,
              help="Skip the rich table; emit JSON only.")
@click.option("--engrad-np1", "engrad_np1", default=None, type=click.Path(exists=True, dir_okay=False),
              help="ORCA .engrad of the N+1 state, a single point on the SAME geometry as N. "
                   "With --engrad-nm1 it adds the nuclear Fukui function Phi_A, the force "
                   "response to a change in electron count.")
@click.option("--engrad-nm1", "engrad_nm1", default=None, type=click.Path(exists=True, dir_okay=False),
              help="ORCA .engrad of the N-1 state, on the same geometry.")
@click.option("--engrad-n", "engrad_n", default=None, type=click.Path(exists=True, dir_okay=False),
              help="ORCA .engrad of the neutral. Optional: adds the one-sided Phi+ and Phi- "
                   "and the nuclear dual Phi(2), which the two charged states alone cannot give.")
@click.option("--project-normal", "project_normal", is_flag=True,
              help="Also project Phi on each atom's own outward normal, the direction a probe "
                   "approaches along. Keeps the sign, which the magnitude discards: a nucleus "
                   "pulled into the cluster is the opposite case to one pushed out.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist the result (molecule + calculation + descriptors) to the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def cdft_command(
    path_n, path_np1, path_nm1, cluster, engine, output, report_path, report_format,
    tce_homo, top, no_table, engrad_np1, engrad_nm1, engrad_n, project_normal,
    use_db, db_path,
) -> None:
    """Compute CDFT descriptors and rank reactive sites.

    With --report it also writes the full report (global + condensed per-atom
    Fukui, local electrophilicity/nucleophilicity, softnesses), so the complete CDFT analysis
    is available under `cadmia`. Inside a project the result is saved to results/cdft.json,
    and `cadmia cdft --report FILE` alone writes the report from it without recomputing.
    """
    from cadmia.agents import cdft_agent
    from cadmia.display import print_cdft_table
    from cadmia.io.results import save_stage_result, stage_result_path

    if not (path_n and path_np1 and path_nm1):
        saved = stage_result_path("cdft")
        if path_n or path_np1 or path_nm1 or not report_path:
            raise click.UsageError("--n, --np1 and --nm1 are required (inside a project, "
                                   "--report alone writes the report from results/cdft.json).")
        if saved is None or not saved.is_file():
            raise click.ClickException("No results/cdft.json in this project: run cadmia cdft "
                                       "with --n, --np1 and --nm1 first.")
        if report_format == "txt":
            _write_saved_report("cdft", report_path)
            return
        data = json.loads(saved.read_text(encoding="utf-8"))
        content = json.dumps({k: v for k, v in data.items() if k != "report_text"},
                             indent=2, ensure_ascii=False)
        Path(report_path).write_text(content, encoding="utf-8")
        click.echo(f"Full CDFT report written to: {report_path} (from {saved})")
        return

    result = cdft_agent.run(
        path_n=path_n,
        path_np1=path_np1,
        path_nm1=path_nm1,
        cluster=cluster,
        engine=engine,
    )

    if engrad_np1 or engrad_nm1:
        _attach_nuclear_fukui(result, engrad_np1, engrad_nm1, engrad_n, top, project_normal)

    if not no_table:
        print_cdft_table(result, top=top)

    # One file holds everything: the ranking geometry and analyze read, and the full report's
    # data and text, which `cadmia report` and correlate read. Their keys do not overlap.
    from cadmia.cdft.report import cdft_report_payload, cdft_report_text, write_cdft_report
    from cadmia.cdft.workflow import run_workflow_from_files
    data, text, wf = result.to_dict(), "", None
    try:
        wf, q_n, q_np1, q_nm1 = run_workflow_from_files(
            path_n, path_np1, path_nm1, homo_tce_ev=tce_homo,
        )
    except ValueError as exc:
        if report_path:
            raise click.ClickException(f"--report failed: {exc}")
        click.echo(f"  ⚠  Full CDFT report unavailable: {exc}", err=True)
        text = f" Full CDFT report unavailable: {exc}"
    if wf is not None:
        nuclear = result.nuclear.get("sites") if result.nuclear else None
        data.update(cdft_report_payload(wf, q_n, q_np1, q_nm1, nuclear))
        text = cdft_report_text(wf, q_n, q_np1, q_nm1)

    from cadmia.io.results import json_safe
    json_str = json.dumps(json_safe({**data, "report_text": text}), indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table and not report_path:
        click.echo(json_str)

    if report_path:
        write_cdft_report(
            wf, report_path, charges_n=q_n, charges_np1=q_np1, charges_nm1=q_nm1,
            nuclear=nuclear, fmt=report_format,
        )
        click.echo(f"Full CDFT report written to: {report_path}")

    saved = save_stage_result("cdft", data, text)
    if saved:
        click.echo(f"Module result: {saved}")

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        detected_engine = engine or cdft_agent._detect_engine(Path(path_n))
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_cdft_result(
                conn, result, engine=detected_engine, output_path=path_n,
                provenance=_cli_provenance(path_n, "cdft"),
            )
            _record_workflow_run(conn, "cdft", ids)
            conn.commit()
        finally:
            conn.close()
        click.echo(
            f"Persisted to {target}: calculation {ids['calculation_id']}, "
            f"{len(ids['descriptor_ids'])} descriptors."
        )


@main.command("validate")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Calculation engine. Auto-detected when omitted.")
@click.option("--bsse", is_flag=True,
              help="Check for BSSE correction (adsorption calculations only).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write JSON report to this file.")
@click.option("--no-table", "no_table", is_flag=True,
              help="Skip rich table; emit JSON only.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Log each validation outcome to the database run_log.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def validate_command(files, engine, bsse, output, no_table, use_db, db_path) -> None:
    """Validate one or more DFT output files (termination, SCF, spin, frequencies, geometry)."""
    from cadmia.agents import validation_agent
    from cadmia.display import print_validation_report

    reports = []
    for f in files:
        report = validation_agent.run(filepath=f, engine=engine, check_bsse=bsse)
        reports.append(report)
        if not no_table:
            print_validation_report(report)

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            for report in reports:
                failed = [k for k, c in report.checks.items() if c.status == "fail"]
                repository.log_run(
                    conn, agent="validation", action="validate",
                    status=report.overall,
                    message=f"{Path(report.filepath).name}: {report.overall}"
                            + (f" (failed: {', '.join(failed)})" if failed else ""),
                    input_hash=report.filepath,
                )
            _record_workflow_run(conn, "validation",
                                 status="ok" if all(r.overall == "pass" for r in reports) else "warning")
            conn.commit()
        finally:
            conn.close()
        click.echo(f"Logged {len(reports)} validation(s) to {target}.")

    if output:
        import json
        data = [r.to_dict() for r in reports] if len(reports) > 1 else reports[0].to_dict()
        Path(output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table:
        import json
        data = [r.to_dict() for r in reports] if len(reports) > 1 else reports[0].to_dict()
        click.echo(json.dumps(data, indent=2))

    if any(r.overall == "fail" for r in reports):
        raise SystemExit(1)


@main.command("geometry")
@click.option("--cdft", "cdft_json", required=True, type=click.Path(exists=True),
              help="CDFTResult JSON file from `cadmia cdft -o result.json`.")
@click.option("--xyz", "cluster_xyz", required=True, type=click.Path(exists=True),
              help="Cluster geometry: a .xyz file OR a Gaussian/ORCA output (any extension; "
                   "the engine is auto-detected from the file content). The geometry is "
                   "extracted preserving the SCF atom order.")
@click.option("--probe", required=True,
              help="Probe molecule: built-in (H2, CO2, NH3, CO, N2, O2, H2O) or a "
                   "user-defined probe from ~/.cadmia/probes.yaml.")
@click.option("--probes-yaml", "probes_yaml", default=None, type=click.Path(exists=True),
              help="Custom probes file (default: ~/.cadmia/probes.yaml).")
@click.option("--n-sites", default=0, show_default=True,
              help="Number of top CDFT sites to use; 0 means every site in the ranking. "
                   "Narrowing it biases the sampling toward the extremes of the dual "
                   "descriptor, which is not where every probe binds.")
@click.option("--n-configs", default=6, show_default=True,
              help="Azimuthal orientations per site per approach mode.")
@click.option("--polar", "polar_angles", multiple=True, type=float, default=(),
              help="Polar approach angles in degrees (repeatable). Default: 0 20 40 60.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output directory for .xyz files (inside a project: configs/<PROBE>/).")
@click.option("--no-table", "no_table", is_flag=True,
              help="Skip rich table; emit JSON only.")
@click.option("--cluster", "cluster_label", default="",
              help="Cluster label override (default: xyz file stem).")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist the cluster and adsorption configs to the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def geometry_command(
    cdft_json, cluster_xyz, probe, probes_yaml, n_sites, n_configs, polar_angles,
    output, no_table, cluster_label, use_db, db_path,
) -> None:
    """Generate adsorption starting geometries guided by CDFT site ranking."""
    from cadmia.agents import geometry_agent
    from cadmia.display import print_geometry_result

    polars = list(polar_angles) if polar_angles else None

    if probe.upper() not in geometry_agent.available_probes(probes_yaml):
        raise click.BadParameter(
            f"unknown probe '{probe}'. Available: "
            f"{', '.join(geometry_agent.available_probes(probes_yaml))} "
            f"(add more in ~/.cadmia/probes.yaml).",
            param_hint="--probe",
        )

    if output is None:
        from cadmia.io.results import find_project_root
        root = find_project_root()
        output = str(root / "configs" / probe.upper()) if root else None

    result = geometry_agent.run(
        cdft_json=cdft_json,
        cluster_xyz=cluster_xyz,
        probe=probe,
        n_sites=n_sites,
        n_configs=n_configs,
        polar_angles=polars,
        output_dir=output,
        cluster_label=cluster_label,
        probes_yaml=probes_yaml,
    )

    if not no_table:
        print_geometry_result(result)

    if no_table:
        click.echo(result.to_json())
    if result.output_json:
        click.echo(f"  JSON → {result.output_json}")

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_geometry_result(conn, result, cluster_xyz=cluster_xyz)
            _record_workflow_run(conn, "geometry", ids)
            conn.commit()
        finally:
            conn.close()
        click.echo(
            f"Persisted to {target}: cluster {ids['cluster_id'][:12]}…, "
            f"{len(ids['adsorption_ids'])} adsorption configs."
        )


def _announce_db(target: "Path") -> "Path":
    """Say it once when a database file is about to be created, and return the path.

    Persistence is on by default, so the first write can bring a file into
    existence that the user never asked for by name. A silent default is not fixed with a
    second silent default: it costs one line and it says where the file went.
    """
    if not Path(target).exists():
        click.echo(f"  DB  → creating {target}", err=True)
    return Path(target)


_DUP_REASON = re.compile(r"^duplicate of (.+?) \(same minimum\)$")


def _representative_of(reason: str) -> "str | None":
    """The representative named in a prescreen `reason`, or None if it is not a duplicate."""
    m = _DUP_REASON.match(reason or "")
    return m.group(1) if m else None


def _persist_screening(db_path, stage: str, agent: str, rows, *, link_by_xyz: bool) -> int:
    """Write one screening stage to the database; return how many rows were recorded.

    `link_by_xyz` resolves each row back to the `adsorption_configs` entry `geometry` created,
    by the path of the file itself. When that entry is absent — `geometry --db` was never run —
    the row is still written with a null link: losing the screen's own verdict because the
    earlier stage was not persisted would defeat the purpose.
    """
    from cadmia.db.session import init_db, resolve_db_path
    from cadmia.db import repository

    target = _announce_db(resolve_db_path(db_path))
    conn = init_db(target)
    try:
        ids = []
        for row in rows:
            config_id = (repository.find_adsorption_config_by_xyz(conn, row["source"])
                         if link_by_xyz else None)
            ids.append(repository.upsert_screening_result(
                conn, stage=stage, config_id=config_id, **row))
        _record_workflow_run(conn, agent, {"screening_ids": ids})
        conn.commit()
        return len(ids)
    finally:
        conn.close()


@main.command("prescreen")
@click.argument("configs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--charge", default=0, show_default=True, help="Total charge of the adduct.")
@click.option("--mult", "multiplicity", default="auto", show_default=True,
              help="Spin multiplicity (2S+1), or auto: the mult= that geometry writes in each "
                   "config's comment line, else the lowest spin for its electron count.")
@click.option("--gfn", default=2, type=click.IntRange(0, 2), show_default=True,
              help="GFN parameterization (2 = GFN2-xTB, includes D4).")
@click.option("--top", "top_n", default=0, show_default=True,
              help="Keep at most N survivors (0 = no cap, energy window only).")
@click.option("--window", "window_kcal", default=10.0, show_default=True,
              help="Keep configs within this many kcal/mol of the lowest xTB energy.")
@click.option("--rmsd", "dedup_rmsd", default=0.20, show_default=True,
              help="Tolerance (Å) below which two optimized geometries are the same "
                   "minimum: the largest displacement of any single atom after superposition.")
@click.option("--min-keep", "min_keep", default=2, show_default=True,
              help="Guarantee at least N distinct survivors even if the next-lowest minima "
                   "fall outside the window, so DFT gets ≥N candidates to compare (0 = window only).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Directory for the survivors' optimized .xyz (ready for prepare/run). "
                   "Default: ./prescreened, or prescreened/ next to the configs inside a project.")
@click.option("--xtb-path", "xtb_path", default="xtb", show_default=True,
              help="xtb executable name or absolute path.")
@click.option("--jobs", "-j", default=0, show_default=True,
              help="xTB optimizations to run at once (0 = one per available CPU). One xTB run "
                   "on a small cluster does not scale over threads; running many at once does.")
@click.option("--threads", default=1, show_default=True,
              help="OpenMP threads per xTB run. Keep at 1: some xtb builds (e.g. Ubuntu apt 6.6.1) "
                   "SIGSEGV with >1 thread.")
@click.option("--no-sanity", "no_sanity", is_flag=True,
              help="Disable the physical-sanity filter (keep nonphysical xTB minima: "
                   "collapsed atoms or energies far below the median).")
@click.option("--cluster", "cluster_ref", type=click.Path(exists=True, dir_okay=False),
              help="Bare-cluster .xyz, relaxed at the SAME level as this screen. Groups the "
                   "configs by the skeleton their cluster relaxed to and ranks each group "
                   "on its own zero, so a more stable cluster isomer cannot discard the "
                   "adducts that kept the reference skeleton.")
@click.option("--freeze", is_flag=True,
              help="Hold the cluster rigid in xTB, so only the probe relaxes (needs --cluster, "
                   "whose atoms must open every config). Default: everything relaxes, since "
                   "frozen screens drop sites that bind only by relaxing the cluster. DFT is "
                   "never constrained.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist each configuration's fate at this stage to the database, "
                   "linked to the starting configuration `geometry` recorded.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def prescreen_command(configs, charge, multiplicity, gfn, top_n, window_kcal,
                      dedup_rmsd, min_keep, output, xtb_path, jobs, threads, no_sanity,
                      cluster_ref, freeze, use_db, db_path) -> None:
    """Pre-screen adsorption configs with GFN2-xTB before DFT (rank + dedup; Prescreen Agent).

    Cheaply optimizes each starting geometry with xTB, removes duplicates that relax to the
    same minimum, and keeps the lowest-energy survivors within an energy window. xTB energies
    rank/dedup only — survivors must still be re-optimized at DFT (cadmia prepare/run).
    """
    from cadmia.agents import prescreen_agent
    from cadmia.calculators.xtb_runner import xtb_available
    from cadmia.io.results import find_project_root

    if output is None:
        first = Path(configs[0]) if configs else Path(".")
        output = str((first if first.is_dir() else first.parent) / "prescreened"
                     if find_project_root() else "./prescreened")

    if not xtb_available() and xtb_path == "xtb":
        raise click.ClickException(
            "xtb not found on PATH. Install it (https://github.com/grimme-lab/xtb) "
            "or pass --xtb-path /abs/path/to/xtb."
        )

    source = configs[0] if len(configs) == 1 else list(configs)

    click.echo("Pre-screening with GFN2-xTB (one line per configuration as it finishes):")

    def progress(done, total, name, energy, error):
        if error:
            click.echo(f"  [{done}/{total}] {name}  ✗ {error}")
        else:
            click.echo(f"  [{done}/{total}] {name}  E = {energy:.6f} Eh")

    if freeze and cluster_ref is None:
        raise click.UsageError("--freeze needs --cluster: its atoms are the ones held fixed.")
    mult_value, mult_auto = _parse_mult(multiplicity)
    result = prescreen_agent.prescreen(
        source, charge=charge, multiplicity=None if mult_auto else mult_value, gfn=gfn, top_n=top_n,
        energy_window_kcal=window_kcal, dedup_rmsd=dedup_rmsd, min_keep=min_keep,
        output_dir=output, xtb_path=xtb_path, jobs=jobs, threads=threads, sanity=not no_sanity,
        cluster_ref=cluster_ref, freeze=freeze, on_progress=progress,
    )

    click.echo(f"Pre-screen: {result.n_kept}/{result.n_input} kept "
               f"(window {result.window_kcal} kcal/mol) → {result.output_dir}")
    for it in sorted(result.items, key=lambda x: (x.rank is None, x.rank or 0)):
        tag = "✓" if it.kept else " "
        rel = f"+{it.rel_kcal:.2f}" if it.rel_kcal is not None else "  —  "
        click.echo(f"  [{tag}] {Path(it.source_xyz).name:40s} {rel:>9s} kcal/mol  {it.reason}")
    # Candidate report: the survivors selected for DFT, in energy order.
    survivors = sorted(result.survivors, key=lambda x: x.rank or 0)
    if survivors:
        click.echo(f"\nSurvivors for DFT ({len(survivors)}, energy order):")
        click.echo(f"  {'#':>2}  {'ΔE (kcal/mol)':>13}  configuration")
        for it in survivors:
            click.echo(f"  {it.rank:>2}  {it.rel_kcal:>13.3f}  {Path(it.source_xyz).stem}")
        if result.candidates_xyz:
            click.echo(f"\n  all good minima (energy order, tagged [to-DFT]/reserve): "
                       f"{result.candidates_xyz}")
        if result.n_reserve:
            click.echo(f"  + {result.n_reserve} reserve candidate(s) in "
                       f"{result.output_dir}/pool/reserve/ — copy one into "
                       f"{result.output_dir}/ to send it to DFT (no xTB re-run needed).")

    if result.output_json:
        click.echo(f"  JSON → {result.output_json}")

    if use_db:
        n = _persist_screening(
            db_path, "prescreen", "prescreen",
            [dict(source=it.source_xyz, optimized_xyz=it.optimized_xyz,
                  energy_hartree=it.energy_hartree, rel_kcal=it.rel_kcal, rank=it.rank,
                  kept=it.kept, converged=it.converged,
                  duplicate_of=_representative_of(it.reason),
                  cluster_skeleton=it.cluster_skeleton)
             for it in result.items],
            link_by_xyz=True)
        click.echo(f"  DB  → {n} configuration(s) recorded")

    for w in result.warnings:
        click.echo(f"  ⚠  {w}", err=True)


@main.command("density")
@click.option("--gbw", "gbw", default=None, type=click.Path(exists=True),
              help="ORCA wavefunction (.gbw) of one species (single/Fukui mode).")
@click.option("--np1", "np1_gbw", default=None, type=click.Path(exists=True),
              help="N+1 (anion) .gbw — with --nm1, derives the Fukui/dual maps.")
@click.option("--nm1", "nm1_gbw", default=None, type=click.Path(exists=True),
              help="N-1 (cation) .gbw — with --np1, derives the Fukui/dual maps.")
@click.option("--adduct", "adduct_gbw", default=None, type=click.Path(exists=True),
              help="Adduct .gbw — with --cluster/--probe, derives the deformation density Δρ.")
@click.option("--cluster", "cluster_gbw", default=None, type=click.Path(exists=True),
              help="Cluster .gbw at the adduct geometry (Δρ mode).")
@click.option("--probe", "probe_gbw", default=None, type=click.Path(exists=True),
              help="Probe .gbw at the adduct geometry (Δρ mode).")
@click.option("--kind", "kinds", multiple=True,
              type=click.Choice(["density", "homo", "lumo", "mep", "all"]), default=("density",),
              show_default=True, help="Cube(s) to generate for the --gbw species (repeatable).")
@click.option("--homo-index", "homo_index", default=None, type=int,
              help="0-based HOMO orbital number (required for homo/lumo cubes; LUMO = HOMO+1).")
@click.option("--step", default=0.20, show_default=True, type=float,
              help="Grid spacing in Bohr (uniform step; one value keeps N/N±1 cubes voxel-aligned).")
@click.option("--orca-plot-path", "orca_plot_path", default="orca_plot", show_default=True,
              help="orca_plot executable (ships with ORCA).")
@click.option("--output", "-o", default="./cubes", show_default=True, type=click.Path(),
              help="Directory for the generated cubes.")
def density_command(gbw, np1_gbw, nm1_gbw, adduct_gbw, cluster_gbw, probe_gbw,
                    kinds, homo_index, step, orca_plot_path, output) -> None:
    """Generate electron-density / HOMO / LUMO / MEP cubes and Fukui/dual/Δρ maps.

    The mode is inferred from the flags you pass:
      • Species cubes:  --gbw X.gbw --kind density|homo|lumo|mep|all
      • Fukui/dual:     --gbw N.gbw --np1 anion.gbw --nm1 cation.gbw  (f+, f-, f0, dual)
      • Deformation Δρ: --adduct ad.gbw --cluster cl.gbw --probe pr.gbw

    Cube generation needs ORCA's orca_plot (run where ORCA is installed); the difference maps are
    pure grid arithmetic. The grid is sized per axis for a uniform --step, so the lobes stay smooth;
    one --step keeps a Fukui/Δρ set voxel-aligned. For Δρ the cluster and probe single points must
    be run at the adduct geometry WITH GHOST ATOMS (the partner's atoms as ghosts), so all three
    .gbw share the same grid box; otherwise the cubes are not voxel-aligned and Δρ is rejected.
    """
    from cadmia.agents import density_agent
    from cadmia.calculators.orca_plot_runner import OrcaPlotError, orca_plot_available
    from cadmia.io.cube import parse_cube

    _ELECTRON_FIELDS = {"density", "f_plus", "f_minus", "f0", "dual", "deformation"}

    def _echo_cube(label: str, path, field_type: str) -> None:
        s = density_agent.cube_summary(parse_cube(path, field_type=field_type))
        rng = f"[{s['min']:+.3e}, {s['max']:+.3e}]"
        if field_type in _ELECTRON_FIELDS:   # ∫ in electrons only where it is physically meaningful
            click.echo(f"  {label:7s} → {path}   ∫={s['volume_integral']:+.3f} e  {rng}")
        else:                                 # orbital / MEP: the integral is not a count
            click.echo(f"  {label:7s} → {path}   {rng}")

    if not orca_plot_available(orca_plot_path):
        raise click.ClickException(
            f"orca_plot ('{orca_plot_path}') not found on PATH. It ships with ORCA — run this on "
            "the machine where ORCA is installed, or pass --orca-plot-path /abs/path/to/orca_plot."
        )
    selected = list(dict.fromkeys(kinds))
    if "all" in selected:
        selected = ["density", "homo", "lumo", "mep"]

    delta_flags = (adduct_gbw, cluster_gbw, probe_gbw)
    fukui_flags = (np1_gbw, nm1_gbw)

    try:
        # Δρ mode — inferred from --adduct/--cluster/--probe (no dedicated flag).
        if any(delta_flags):
            if any(fukui_flags):
                raise click.UsageError(
                    "don't mix Δρ (--adduct/--cluster/--probe) with Fukui (--np1/--nm1).")
            if not all(delta_flags):
                raise click.UsageError("Δρ needs all three of --adduct, --cluster and --probe .gbw.")
            dst = density_agent.deformation_from_gbw(adduct_gbw, cluster_gbw, probe_gbw, output,
                                                     step=step, orca_plot_path=orca_plot_path)
            _echo_cube("Δρ", dst, "deformation")
            return
        # Fukui/dual mode — inferred from --np1 + --nm1.
        if any(fukui_flags):
            if not all(fukui_flags):
                raise click.UsageError("Fukui/dual needs both --np1 (anion) and --nm1 (cation).")
            if not gbw:
                raise click.UsageError("Fukui/dual needs --gbw (the N state) with --np1/--nm1.")
            written = density_agent.fukui_maps_from_gbw(gbw, np1_gbw, nm1_gbw, output,
                                                        step=step, orca_plot_path=orca_plot_path)
            for name, path in written.items():
                _echo_cube(name, path, name)
            click.echo(f"Fukui/dual maps → {output}/   (∫f± ≈ 1, ∫Δf ≈ 0)")
            return
        # Species-cube mode.
        if not gbw:
            raise click.UsageError(
                "provide --gbw (species cubes / Fukui) or --adduct/--cluster/--probe (Δρ).")
        if ("homo" in selected or "lumo" in selected) and homo_index is None:
            raise click.UsageError("homo/lumo cubes need --homo-index (0-based HOMO orbital number).")
        written = density_agent.generate_species_cubes(gbw, selected, output, step=step,
                                                       homo_index=homo_index,
                                                       orca_plot_path=orca_plot_path)
        for kind, path in written.items():
            _echo_cube(kind, path, kind)
    except OrcaPlotError as exc:
        raise click.ClickException(str(exc))


@main.command("molden")
@click.argument("gbw_files", nargs=-1, type=click.Path(exists=True))
@click.option("--orca-2mkl-path", "orca_2mkl_path", default="orca_2mkl", show_default=True,
              help="orca_2mkl executable (ships with ORCA).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Directory for the .molden files (default: alongside each .gbw).")
def molden_command(gbw_files, orca_2mkl_path, output) -> None:
    """Export ORCA wavefunctions (.gbw) to portable Molden files for external analysis.

    The .gbw is ORCA's binary, version-specific wavefunction; JANPA (NPA) and Multiwfn (AIM/QTAIM)
    read the open Molden format instead. This routes through ORCA's orca_2mkl so the f/g-function
    conventions match what those tools expect — no re-running ORCA, no ORCA-version lock-in.

    Batch-friendly: pass one or more .gbw (the final characterization wavefunctions). Needs
    orca_2mkl, so run it on the machine where ORCA is installed (or pass --orca-2mkl-path /abs/path).
    """
    from cadmia.calculators.orca_2mkl_runner import (
        OrcaToMoldenError,
        gbw_to_molden,
        orca_2mkl_available,
    )

    if not gbw_files:
        raise click.UsageError("give one or more .gbw files to export, e.g. `cadmia molden *.gbw`.")
    if not orca_2mkl_available(orca_2mkl_path):
        raise click.ClickException(
            f"orca_2mkl ('{orca_2mkl_path}') not found on PATH. It ships with ORCA — run this on "
            "the machine where ORCA is installed, or pass --orca-2mkl-path /abs/path/to/orca_2mkl."
        )
    out_dir = Path(output) if output else None
    try:
        for gbw in gbw_files:
            dst = out_dir / f"{Path(gbw).stem}.molden" if out_dir else None
            res = gbw_to_molden(gbw, dst, orca_2mkl_path=orca_2mkl_path)
            click.echo(f"  {Path(gbw).name} → {res.molden_path}")
    except OrcaToMoldenError as exc:
        raise click.ClickException(str(exc))
    click.echo("Molden export done. NPA: `janpa` / NBO7 · AIM: Multiwfn — feed them these .molden.")


@main.command("submit")
@click.option("--jobs", "jobs_source", required=True, type=click.Path(exists=True),
              help="Input file (.gjf or .inp) or directory containing them.")
@click.option("--hpc", "profile_name", required=True,
              help="HPC profile name (see `cadmia profile list`).")
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Force engine; auto-detected from file extension if omitted.")
@click.option("--nprocs", default=8, show_default=True,
              help="CPU cores per job (SLURM --ntasks-per-node).")
@click.option("--mem-g", "mem_gaussian_gb", default=16, show_default=True,
              help="Gaussian memory in GB.")
@click.option("--mem-o", "mem_orca_mb_core", default=2000, show_default=True,
              help="ORCA memory per core in MB.")
@click.option("--output", "-o", default=".", type=click.Path(),
              help="Local directory for output files and jobs.json state.")
@click.option("--wait", is_flag=True,
              help="Block until all jobs finish, then download and validate.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist submitted jobs as calculation records in the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def submit_command(
    jobs_source, profile_name, engine, nprocs, mem_gaussian_gb,
    mem_orca_mb_core, output, wait, use_db, db_path,
) -> None:
    """Submit READY input files (.inp/.gjf) to a SLURM queue; returns once queued (non-blocking).

    Use this on a cluster/workstation with SLURM. It does NOT build inputs — run `cadmia prepare`
    first (or `cadmia geometry`→`prepare`). Without --wait the terminal is freed immediately and
    you poll with `cadmia monitor`; --wait blocks until everything finishes and downloads outputs.
    For running locally without SLURM, use `cadmia run` instead.
    """
    from cadmia.calculators.params import CalcParams
    from cadmia.agents import calculation_agent
    from cadmia.display import print_calculation_result

    params = CalcParams(nprocs=nprocs, mem_gaussian_gb=mem_gaussian_gb,
                        mem_orca_mb_core=mem_orca_mb_core)
    result = calculation_agent.submit(
        jobs_source, profile_name, params,
        engine=engine, output_dir=output, wait=wait,
    )
    print_calculation_result(result)

    state = Path(output) / "cadmia_jobs.json"      # the sidecar submit has always written
    if not wait:
        n = len(getattr(result, "jobs", []) or [])
        click.echo(f"Submitted: {n} job(s) queued on '{profile_name}'. Terminal is free.")
        click.echo(f"Track them with:  cadmia monitor --jobs {state} --hpc {profile_name}")
    click.echo(f"  JSON → {state}")

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_calculation_result(conn, result, task=params.task)
            _record_workflow_run(conn, "calculation", ids,
                                 status="ok" if result.n_failed == 0 else "warning")
            conn.commit()
        finally:
            conn.close()
        click.echo(f"Persisted to {target}: {len(ids['calculation_ids'])} job record(s).")

    if result.n_failed > 0:
        raise SystemExit(1)


def _parse_mult(value):
    """Parse a ``--mult`` value → (multiplicity:int, auto:bool).

    ``"auto"`` (the default) → derive multiplicity from the electron-count parity;
    any integer ≥ 1 → use it verbatim (manual override).
    """
    if value is None or str(value).strip().lower() in ("auto", ""):
        return 1, True
    try:
        m = int(value)
    except (TypeError, ValueError):
        raise click.BadParameter("must be 'auto' or an integer ≥ 1.", param_hint="--mult")
    if m < 1:
        raise click.BadParameter("must be 'auto' or an integer ≥ 1.", param_hint="--mult")
    return m, False


@main.command("run")
@click.option("--jobs", "jobs_source", required=True, type=click.Path(exists=True),
              help="Input file (.gjf/.inp), an .xyz, or a directory of them "
                   "(.xyz files are converted to inputs first).")
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Engine; auto-detected from extension (.xyz defaults to orca).")
@click.option("--engine-path", default=None, type=click.Path(),
              help="Path to the ORCA installation dir or g16 binary (default: PATH).")
@click.option("--task", type=click.Choice(["sp", "opt", "freq", "opt+freq"]),
              default="opt", show_default=True, help="Calculation task for .xyz inputs.")
@click.option("--functional", default="PBE0", show_default=True,
              help="DFT functional written into the generated inputs.")
@click.option("--basis", default="def2-TZVP", show_default=True, help="Basis set.")
@click.option("--dispersion", type=click.Choice(["d3bj", "d4", "none"]),
              default="d4", show_default=True,
              help="Dispersion correction. d4 depends on the atomic charges, d3bj does not.")
@click.option("--charge", default=0, show_default=True, help="Total charge of the system.")
@click.option("--mult", "multiplicity", default="auto", show_default=True,
              help="Spin multiplicity (2S+1), or 'auto' to derive it from the electron count.")
@click.option("--nprocs", default=8, show_default=True,
              help="Cores per job (%pal / %nprocshared) for .xyz inputs.")
@click.option("--mem-g", "mem_gaussian_gb", default=16, show_default=True,
              help="Gaussian memory in GB.")
@click.option("--mem-o", "mem_orca_mb_core", default=2000, show_default=True,
              help="ORCA memory per core in MB (%maxcore).")
@click.option("--extra-orca", default="", help="Extra ORCA simple-input keywords.")
@click.option("--preset", default="analysis", show_default=True,
              type=click.Choice(["none", "cdft", "analysis", "accurate"]),
              help="Analysis output requests to embed in .xyz-generated inputs.")
@click.option("--output", "-o", default=".", type=click.Path(),
              help="Directory for generated inputs, outputs and cadmia_jobs.json.")
@click.option("--timeout", "timeout_s", default=None, type=int,
              help="Per-job wall-time limit in seconds (default: unlimited).")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist the finished jobs as calculation records in the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
@click.option("--no-tidy", is_flag=True,
              help="Keep accessory files in place (by default they move to aux/).")
@click.option("--overwrite", is_flag=True,
              help="Recompute every job, ignoring outputs a previous run already finished.")
def run_command(jobs_source, engine, engine_path, task, functional, basis, dispersion,
                charge, multiplicity, nprocs, mem_gaussian_gb, mem_orca_mb_core,
                extra_orca, preset, output, timeout_s, use_db, db_path, no_tidy,
                overwrite) -> None:
    """Build (from .xyz) and RUN DFT jobs sequentially on THIS machine — blocking, no SLURM.

    Unlike `cadmia submit` (which queues ready inputs on SLURM and returns), `run` executes
    here and now, one job at a time, and blocks until they finish. It accepts .xyz directly
    (builds the input first) as well as .inp/.gjf. Each job uses the cores/memory in its input;
    progress is saved to cadmia_jobs.json after every job, so a re-run resumes where it
    stopped (outputs already finished are skipped; --overwrite recomputes them). Use `submit`
    for a SLURM cluster.
    """
    from cadmia.calculators.params import CalcParams
    from cadmia.calculators.presets import apply_preset
    from cadmia.agents import calculation_agent
    from cadmia.display import print_calculation_result

    mult_value, mult_auto = _parse_mult(multiplicity)
    params = CalcParams(functional=functional, basis=basis, dispersion=dispersion,
                        task=task, charge=charge, multiplicity=mult_value,
                        auto_multiplicity=mult_auto,
                        nprocs=nprocs, mem_gaussian_gb=mem_gaussian_gb,
                        mem_orca_mb_core=mem_orca_mb_core, extra_orca=extra_orca)
    params = apply_preset(params, preset, engine=(engine or "orca"))
    result = calculation_agent.run_local(
        jobs_source, params, engine=engine, output_dir=output,
        engine_path=engine_path, timeout_s=timeout_s, tidy=not no_tidy,
        overwrite=overwrite,
    )
    print_calculation_result(result)

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_calculation_result(conn, result, task=params.task)
            _record_workflow_run(conn, "calculation", ids,
                                 status="ok" if result.n_failed == 0 else "warning")
            conn.commit()
        finally:
            conn.close()
        click.echo(f"Persisted to {target}: {len(ids['calculation_ids'])} job record(s).")

    if result.n_failed > 0:
        raise SystemExit(1)


@main.command("prepare")
@click.argument("stage", type=click.Choice(sorted(_STAGE_TOKENS)))
@click.argument("geometries", nargs=-1, type=click.Path(exists=True))
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default="orca",
              show_default=True,
              help="Engine the generated inputs are written for (both get .inp).")
@click.option("--functional", default="PBE0", show_default=True,
              help="DFT functional (overrides the stage).")
@click.option("--basis", default="def2-TZVP", show_default=True,
              help="Basis set (overrides the stage; the screen stages default to def2-SVP).")
@click.option("--dispersion", type=click.Choice(["d3bj", "d4", "none"]),
              default="d4", show_default=True,
              help="Dispersion correction. d4 depends on the atomic charges, d3bj does not.")
@click.option("--charge", default=0, show_default=True, help="Total charge of the system.")
@click.option("--mult", "multiplicity", default="auto", show_default=True,
              help="Spin multiplicity (2S+1), or 'auto' to derive it from the electron count.")
@click.option("--mult-anion", default="auto", show_default=True,
              help="[stage cdft] Multiplicity of the anion (N+1), or 'auto'.")
@click.option("--mult-cation", default="auto", show_default=True,
              help="[stage cdft] Multiplicity of the cation (N-1), or 'auto'.")
@click.option("--nprocs", default=8, show_default=True,
              help="Cores per job (%pal / %nprocshared).")
@click.option("--mem-g", "mem_gaussian_gb", default=16, show_default=True,
              help="Gaussian memory in GB.")
@click.option("--mem-o", "mem_orca_mb_core", default=2000, show_default=True,
              help="ORCA memory per core in MB (%maxcore).")
@click.option("--extra-orca", default="", help="Extra ORCA simple-input keywords.")
@click.option("--extra-gaussian", default="", help="Extra Gaussian route keywords.")
@click.option("--scf-maxiter", default=0, show_default=True,
              help="SCF iteration limit (0 = engine default). Helps open-shell ions converge.")
@click.option("--geom-maxiter", default=0, show_default=True,
              help="Geometry optimization step limit (0 = engine default, max(50, 3N) in ORCA). "
                   "Raise it when the optimization has to rearrange the substrate.")
@click.option("--chelpg-fine", is_flag=True,
              help="Emit a fine ORCA %chelpg block (GRID 0.20, RMAX 3.50, VDWRADII COSMO).")
@click.option("--preset", default=None,
              type=click.Choice(["none", "cdft", "analysis", "accurate"]),
              help="Analysis output requests to embed (overrides the stage's output).")
@click.option("--cluster", "cp_cluster", default=None, type=click.Path(exists=True),
              help="[stage bsse] Output of the isolated cluster: its atom count splits the "
                   "adduct, and its charge and multiplicity are the fragment's.")
@click.option("--probe-free", "cp_probe", default=None, type=click.Path(exists=True),
              help="[stage bsse] Output of the free probe, for the probe fragment's charge "
                   "and multiplicity.")
@click.option("--overwrite", is_flag=True,
              help="Regenerate inputs even if the target file already exists.")
@click.option("--output", "-o", default=".", type=click.Path(),
              help="Directory for the generated input files.")
def prepare_command(stage, geometries, engine, functional, basis,
                    dispersion, charge, multiplicity, mult_anion, mult_cation,
                    nprocs, mem_gaussian_gb, mem_orca_mb_core, extra_orca,
                    extra_gaussian, scf_maxiter, geom_maxiter, chelpg_fine, preset,
                    cp_cluster, cp_probe, overwrite,
                    output) -> None:
    """Generate DFT input files for a workflow STAGE, WITHOUT running them.

    STAGE standardizes the input for each phase of the workflow, writing one .inp per
    geometry, ready to review or submit (`cadmia submit`/`run`):

    \b
      screen-opt   Pre-screening Opt in def2-SVP to rank xTB candidates by energy.
      screen-freq  Freq at the screen-opt level: drop saddle points before the costly stages.
      opt          Final geometry: Opt at the production level (no analysis).
      freq         Frequencies on the optimized geometry: minimum + thermochemistry.
      opt+freq     Combined final geometry + frequencies in one job.
      char         Electronic characterization single point (Hirshfeld/CHELPG/Mayer/PDOS).
      cdft         The N/N+1/N-1 triplet for finite-difference descriptors.

    The stage owns the task and the accuracy block; customize any other axis with an
    explicit flag (--basis/--functional/--preset/--nprocs …), which overrides the stage.

    GEOMETRIES are .xyz files, engine outputs (.out/.log) — so each stage chains from the
    previous one's result — or ready .inp/.gjf inputs, which pass through unchanged. A
    concatenated .xyz (what `cadmia rank` writes) yields one input per block.

    Both engines get .inp: ORCA takes the filename as an argument and Gaussian reads stdin,
    so the extension is CADMIA's convention rather than a requirement. `run`/`submit` read
    the engine of a ready input from its content, not from its name.

    \b
        cadmia prepare screen-opt  configs/*.xyz    -o inputs/screen
        cadmia prepare screen-freq winners/*.out    -o inputs/screenfreq
        cadmia prepare opt+freq    survivors/*.out  -o inputs/optfreq
        cadmia prepare char        finals/*.out     -o inputs/char
        cadmia prepare cdft        cluster.out      -o inputs/cdft
    """
    from cadmia.calculators.params import CalcParams
    from cadmia.calculators.presets import apply_preset
    from cadmia.calculators.stages import apply_stage, get_stage
    from cadmia.agents import calculation_agent

    sources = [str(g) for g in geometries]
    if not sources:
        raise click.UsageError("give one or more geometries after the stage, "
                               "e.g. `cadmia prepare screen-opt configs/*.xyz`.")

    ctx = click.get_current_context()
    basis_explicit = ctx.get_parameter_source("basis") == ParameterSource.COMMANDLINE

    mult_value, mult_auto = _parse_mult(multiplicity)
    params = CalcParams(functional=functional, basis=basis, dispersion=dispersion,
                        task=get_stage(stage).task, charge=charge, multiplicity=mult_value,
                        auto_multiplicity=mult_auto,
                        nprocs=nprocs, mem_gaussian_gb=mem_gaussian_gb,
                        mem_orca_mb_core=mem_orca_mb_core, extra_orca=extra_orca,
                        extra_gaussian=extra_gaussian, scf_maxiter=scf_maxiter,
                        geom_maxiter=geom_maxiter, chelpg_fine=chelpg_fine)

    # Stage owns task + accuracy + basis + output; an explicit --basis/--preset overrides it.
    params, stage_obj = apply_stage(params, stage, engine=engine,
                                    set_preset=preset is None,
                                    set_basis=not basis_explicit)
    if preset is not None:
        params = apply_preset(params, preset, engine=engine)

    if stage_obj.counterpoise:
        if engine != "orca":
            raise click.UsageError("stage bsse writes ORCA ghost-atom inputs. In Gaussian the "
                                   "adduct job itself does it: Counterpoise=2 in the route and "
                                   "each atom tagged with its fragment, (Fragment=1) for the "
                                   "cluster and (Fragment=2) for the probe; analyze reads the result.")
        if not (cp_cluster and cp_probe):
            raise click.UsageError("stage bsse needs --cluster and --probe-free (the isolated "
                                   "outputs) to split each adduct into its fragments.")
        total = 0
        for src in sources:
            try:
                inputs, warnings = calculation_agent.prepare_counterpoise(
                    src, cp_cluster, cp_probe, params, output_dir=output, overwrite=overwrite)
            except ValueError as exc:
                raise click.UsageError(str(exc))
            for w in warnings:
                click.echo(f"  warning: {w}")
            for inp in inputs:
                click.echo(f"  {inp}")
            total += len(inputs)
        click.echo(f"{total} counterpoise input(s) generated in: {output}. Run them, then "
                   "`cadmia analyze --bsse` finds the four <adduct>_cp_*.out next to the adduct "
                   "or in the working directory.")
        return

    if stage_obj.cdft_triplet:
        _prepare_cdft_triplet(sources, params, engine=engine, charge=charge,
                              multiplicity=multiplicity, mult_anion=mult_anion,
                              mult_cation=mult_cation, overwrite=overwrite,
                              output=output)
        return

    try:
        result = calculation_agent.prepare(
            sources if len(sources) > 1 else sources[0],
            params, engine=engine, output_dir=output, overwrite=overwrite,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc))

    for w in result.warnings:
        click.echo(f"  warning: {w}")
    for inp in result.inputs:
        click.echo(f"  {inp}")
    click.echo(f"{len(result.generated)} input(s) generated in: {output}")
    if result.passed_through:
        click.echo(f"{len(result.passed_through)} ready input(s) passed through unchanged.")


def _prepare_cdft_triplet(sources, params, *, engine, charge, multiplicity,
                          mult_anion, mult_cation, overwrite, output) -> None:
    """Emit the N/N+1/N-1 single-point inputs for each geometry (stage `cdft`)."""
    from cadmia.agents import calculation_agent

    m_n, auto_n = _parse_mult(multiplicity)
    m_a, auto_a = _parse_mult(mult_anion)
    m_c, auto_c = _parse_mult(mult_cation)
    total = 0
    for src in sources:
        inputs, warnings = calculation_agent.prepare_cdft(
            src, params, engine=engine, output_dir=output, charge=charge,
            mult=None if auto_n else m_n,
            mult_anion=None if auto_a else m_a,
            mult_cation=None if auto_c else m_c, overwrite=overwrite,
        )
        for w in warnings:
            click.echo(f"  warning: {w}")
        for inp in inputs:
            click.echo(f"  {inp}")
        total += len(inputs)
    click.echo(f"{total} CDFT input(s) generated in: {output}")


@main.command("monitor")
@click.option("--jobs-file", "--jobs", "jobs_file", default="./cadmia_jobs.json",
              type=click.Path(exists=True),
              help="Path to cadmia_jobs.json written by `cadmia submit` (alias: --jobs).")
@click.option("--hpc", "profile_name", required=True,
              help="HPC profile name.")
@click.option("--output", "-o", default=".", type=click.Path(),
              help="Local directory for downloaded output files.")
@click.option("--once", is_flag=True,
              help="Poll once then exit (useful for scripted workflows).")
@click.option("--interval", default=60, show_default=True,
              help="Seconds between polling cycles.")
def monitor_command(jobs_file, profile_name, output, once, interval) -> None:
    """Poll HPC job status, download completed outputs, and validate them."""
    from cadmia.agents import calculation_agent
    from cadmia.display import print_calculation_result

    result = calculation_agent.monitor(
        jobs_file, profile_name,
        output_dir=output, poll_interval=interval,
        once=once,
    )
    print_calculation_result(result)

    if result.n_failed > 0:
        raise SystemExit(1)


@main.group("profile")
def profile_group() -> None:
    """Manage HPC connection profiles."""


@profile_group.command("list")
def profile_list() -> None:
    """List all saved HPC profiles."""
    from cadmia.config import HPCProfile
    names = HPCProfile.list_profiles()
    if not names:
        click.echo("No profiles found. Run: cadmia profile add")
        return
    for name in names:
        click.echo(f"  {name}")


@profile_group.command("show")
@click.argument("name")
def profile_show(name) -> None:
    """Show details of a saved profile."""
    from cadmia.config import HPCProfile
    try:
        p = HPCProfile.load(name)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)
    click.echo(f"Profile: {p.name}")
    if getattr(p, "local", False):
        click.echo("  mode        : local (SLURM on this machine, no SSH)")
    else:
        click.echo("  mode        : remote (SSH)")
        click.echo(f"  host        : {p.user}@{p.host}:{p.ssh_port}")
    click.echo(f"  work_dir    : {p.work_dir}")
    click.echo(f"  partition   : {p.partition}")
    click.echo(f"  account     : {p.account or '(none)'}")
    click.echo(f"  gaussian    : {p.gaussian_module or '(PATH)'}")
    click.echo(f"  orca        : {p.orca_module or '(PATH)'}")
    if p.orca_path:
        click.echo(f"  orca_path   : {p.orca_path}")
    click.echo(f"  max_time    : {p.max_time}")
    if p.ssh_key:
        click.echo(f"  ssh_key     : {p.ssh_key}")


@profile_group.command("add")
@click.option("--name",       required=True, help="Profile identifier (e.g. hpc1).")
@click.option("--local",      is_flag=True,
              help="Local SLURM on THIS machine (no SSH); host/user not needed.")
@click.option("--host",       default="", help="SSH hostname or IP (remote profiles).")
@click.option("--user",       default="", help="SSH username (remote profiles).")
@click.option("--work-dir",   required=True, help="Scratch directory for jobs.")
@click.option("--partition",  default="general", show_default=True,
              help="SLURM partition name.")
@click.option("--account",    default="", help="SLURM account.")
@click.option("--ssh-key",    default="", help="Path to private SSH key.")
@click.option("--gaussian-module", default=None,
              help="Module to load for Gaussian (default: gaussian/16; none if --local).")
@click.option("--orca-module",     default=None,
              help="Module to load for ORCA (default: orca/6; none if --local).")
@click.option("--orca-path",       default="",     help="Absolute path to ORCA binary dir.")
@click.option("--max-time",        default="24:00:00", show_default=True,
              help="SLURM wall-time limit (HH:MM:SS).")
@click.option("--port",            default=22, show_default=True, help="SSH port.")
def profile_add(
    name, local, host, user, work_dir, partition, account, ssh_key,
    gaussian_module, orca_module, orca_path, max_time, port,
) -> None:
    """Save a new SLURM profile (local or HPC) to ~/.cadmia/profiles/<name>.yaml."""
    from cadmia.config import HPCProfile
    # Modules default per backend: clusters use environment modules, a local
    # workstation usually has the engine on PATH (orca_path / g16), so none.
    if gaussian_module is None:
        gaussian_module = "" if local else "gaussian/16"
    if orca_module is None:
        orca_module = "" if local else "orca/6"
    try:
        profile = HPCProfile(
            name=name, host=host, user=user, work_dir=work_dir,
            partition=partition, account=account, ssh_key=ssh_key,
            gaussian_module=gaussian_module, orca_module=orca_module,
            orca_path=orca_path, max_time=max_time, ssh_port=port, local=local,
        )
        saved = profile.save()
        kind = "local" if local else "remote (SSH)"
        click.echo(f"Profile '{name}' [{kind}] saved to {saved}")
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)


@profile_group.command("ping")
@click.argument("name")
def profile_ping(name) -> None:
    """Check a profile: SSH reachability (remote) or local SLURM (sbatch present)."""
    from cadmia.config import HPCProfile
    from cadmia.agents.calculation_agent import _make_runner
    try:
        profile = HPCProfile.load(name)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)
    runner = _make_runner(profile)
    if getattr(profile, "local", False):
        click.echo("Checking local SLURM (sbatch) ...", nl=False)
    else:
        click.echo(f"Pinging {profile.user}@{profile.host} ...", nl=False)
    if runner.ping():
        click.echo(" OK")
    else:
        click.echo(" FAILED", err=True)
        raise SystemExit(1)


@main.group("benchmark")
def benchmark_group() -> None:
    """Systematic Gaussian vs ORCA comparison (Benchmark Agent)."""


@benchmark_group.command("generate")
@click.option("--xyz", "xyz_path", required=True, type=click.Path(exists=True),
              help="Cluster .xyz file.")
@click.option("--functional", default="PBE0", show_default=True,
              help="DFT functional keyword (e.g. PBE0, B3LYP).")
@click.option("--basis", default="def2-TZVP", show_default=True,
              help="Basis set name (e.g. def2-TZVP).")
@click.option("--dispersion", default="d3bj", show_default=True,
              type=click.Choice(["d3bj", "d4", "none"]),
              help="Empirical dispersion correction.")
@click.option("--task", default="opt", show_default=True,
              type=click.Choice(["sp", "opt", "freq", "opt+freq"]),
              help="Calculation type.")
@click.option("--charge", default=0, show_default=True,
              help="Total molecular charge.")
@click.option("--mult", "multiplicity", default="auto", show_default=True,
              help="Spin multiplicity (2S+1), or 'auto' to derive it from the electron count.")
@click.option("--nprocs", default=8, show_default=True,
              help="CPU cores requested in the input file.")
@click.option("--mem-g", "mem_gaussian_gb", default=16, show_default=True,
              help="Memory for Gaussian (%mem), in GB.")
@click.option("--mem-o", "mem_orca_mb_core", default=2000, show_default=True,
              help="Memory per core for ORCA (%maxcore), in MB.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output directory for .gjf and .inp files.")
@click.option("--cluster", "cluster_label", default="",
              help="Cluster label override.")
def benchmark_generate(
    xyz_path, functional, basis, dispersion, task, charge, multiplicity,
    nprocs, mem_gaussian_gb, mem_orca_mb_core, output, cluster_label,
) -> None:
    """Generate Gaussian .gjf and ORCA .inp input files for the same geometry."""
    from cadmia.calculators.params import CalcParams
    from cadmia.agents import benchmark_agent

    mult_value, mult_auto = _parse_mult(multiplicity)
    params = CalcParams(
        functional=functional, basis=basis, dispersion=dispersion, task=task,
        charge=charge, multiplicity=mult_value, auto_multiplicity=mult_auto, nprocs=nprocs,
        mem_gaussian_gb=mem_gaussian_gb, mem_orca_mb_core=mem_orca_mb_core,
    )
    result = benchmark_agent.generate(xyz_path, params, output, cluster_label)

    click.echo(f"Gaussian input: {result.gaussian_input}")
    click.echo(f"ORCA input:     {result.orca_input}")

    if result.output_json:
        click.echo(f"  JSON → {result.output_json}")


@benchmark_group.command("compare")
@click.option("--gaussian", "gaussian_out", required=True, type=click.Path(exists=True),
              help="Gaussian output file (.log or .out).")
@click.option("--orca", "orca_out", required=True, type=click.Path(exists=True),
              help="ORCA output file (.out).")
@click.option("--cluster", "cluster_label", default="", help="Cluster label override.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write comparison report JSON to this file.")
@click.option("--markdown", "md_out", default=None, type=click.Path(),
              help="Write Markdown comparison table to this file.")
@click.option("--no-table", "no_table", is_flag=True,
              help="Skip rich table; emit JSON only.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist both engine calculations and the comparison to the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def benchmark_compare(gaussian_out, orca_out, cluster_label, output, md_out, no_table,
                      use_db, db_path) -> None:
    """Compare already-computed Gaussian and ORCA outputs (ΔE, RMSD, timing)."""
    from cadmia.agents import benchmark_agent
    from cadmia.display import print_benchmark_result

    result = benchmark_agent.compare(gaussian_out, orca_out, cluster_label=cluster_label)

    if not no_table:
        print_benchmark_result(result)

    if output:
        Path(output).write_text(result.to_json(), encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table:
        click.echo(result.to_json())

    if md_out:
        Path(md_out).write_text(result.to_markdown(), encoding="utf-8")
        click.echo(f"Markdown written to: {md_out}")

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_benchmark_result(conn, result)
            _record_workflow_run(conn, "benchmark", ids)
            conn.commit()
        finally:
            conn.close()
        click.echo(f"Persisted to {target}: {len(ids['calculation_ids'])} calculation(s) + comparison log.")

    if result.comparison and not result.comparison.engine_agreement:
        raise SystemExit(1)


def _resolve_adducts(adducts, adduct, adduct_dir) -> list:
    """Adduct outputs from the positional paths (files, directories or a shell glob).

    A directory contributes its *.out files, so `analyze co2/` and `analyze co2/*.out` are
    the same run. --adduct / --adduct-dir remain as the older spellings of one and many.
    """
    given = [g for g in (bool(adducts), bool(adduct), bool(adduct_dir)) if g]
    if len(given) != 1:
        raise click.UsageError(
            "Give the adduct outputs as arguments (files, a directory, or a glob: "
            "cadmia analyze co2/*.out), or exactly one of --adduct / --adduct-dir.")
    if adduct:
        return [Path(adduct)]
    roots = [Path(adduct_dir)] if adduct_dir else [Path(a) for a in adducts]
    out: list = []
    for r in roots:
        out.extend(sorted(r.glob("*.out")) if r.is_dir() else [r])
    if not out:
        raise click.UsageError(f"No *.out files found in {', '.join(str(r) for r in roots)}")
    return out


def _analysis_stage_json(result, adduct: Path) -> "tuple[str, str]":
    """The analysis JSON (with its full text report) and that text; inside a project it is
    also saved to results/analysis/<adduct>.json."""
    from cadmia.agents.text_report import analysis_report_text
    from cadmia.io.results import save_stage_result
    from cadmia.io.results import json_safe
    data = result.to_dict()
    text = analysis_report_text(data)
    saved = save_stage_result("analysis", data, text, item=Path(adduct).stem)
    if saved:
        click.echo(f"Module result: {saved}")
        data["config_id"] = saved.stem
    return json.dumps(json_safe({**data, "report_text": text}), indent=2, ensure_ascii=False), text


def _write_saved_report(stage: str, report_path) -> None:
    """`cadmia <stage> --report FILE` with no inputs: the report from the project's results/."""
    from cadmia.io.results import saved_report
    try:
        text, where = saved_report(stage)
    except LookupError as exc:
        raise click.ClickException(str(exc))
    Path(report_path).write_text(text, encoding="utf-8")
    click.echo(f"Full {stage} report written to: {report_path} (from {where})")


@main.command("analyze")
@click.argument("adducts", nargs=-1, type=click.Path(exists=True))
@click.option("--adduct", default=None, type=click.Path(exists=True),
              help="Adduct (cluster+probe) output file (single mode).")
@click.option("--adduct-dir", "adduct_dir", default=None, type=click.Path(exists=True),
              help="Directory of adduct output files (batch mode): analyzes every *.out against "
                   "the same --cluster/--probe-free and writes one JSON each into -o (a directory).")
@click.option("--cluster", default=None, type=click.Path(exists=True),
              help="Isolated cluster output file.")
@click.option("--probe-free", default=None, type=click.Path(exists=True),
              help="Free (isolated) probe output file.")
@click.option("--probe", default=None,
              help="Probe label (e.g. CO2, NH3). Inferred from the free-probe formula if omitted.")
@click.option("--bsse", is_flag=True,
              help="Correct E_ads for BSSE: from the four ORCA single points of `prepare bsse` next "
                   "to the adduct output, or the BSSE a Gaussian Counterpoise job prints.")
@click.option("--cluster-deformed", default=None, type=click.Path(exists=True),
              help="Cluster single-point at the adduct geometry (for deformation energy).")
@click.option("--probe-deformed", default=None, type=click.Path(exists=True),
              help="Probe single-point at the adduct geometry (for deformation energy).")
@click.option("--cdft", "cdft_json", default=None, type=click.Path(exists=True),
              help="CDFTResult JSON to cross-reference the binding site against the ranking.")
@click.option("--config-index", default=None, type=int,
              help="Configuration index for this (cluster, probe) triplet.")
@click.option("--bend-threshold", "bend_threshold", default=10.0, show_default=True,
              type=click.FloatRange(min=0, min_open=True),
              help="Bending (degrees) of a linear probe that counts as activation.")
@click.option("--label", "cluster_label", default="",
              help="Cluster label override (default: from the cluster output).")
@click.option("--report", "-r", "report_path", default=None, type=click.Path(),
              help="Also write the full text report here. Alone, inside a project, it writes "
                   "it from results/analysis/ without recomputing.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write AnalysisResult JSON to this file.")
@click.option("--no-table", "no_table", is_flag=True,
              help="Skip the rich table; emit JSON only.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist the analysis as an adsorption row in the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def analyze_command(
    adducts, adduct, adduct_dir, cluster, probe_free, probe, bsse, cluster_deformed,
    probe_deformed, cdft_json, config_index, cluster_label, output, no_table, use_db, db_path,
    report_path=None, bend_threshold=10.0,
) -> None:
    """Post-calculation adsorption energetics and probe activation (Analysis Agent).

    Takes the adduct outputs as arguments — files, a directory, or a shell glob:

      cadmia analyze co2/*.out --cluster cluster.out --probe-free co2.out -o analysis_co2/

    One adduct prints the table; several run in batch against the same cluster/probe and
    write one JSON each into -o (a directory), with the config index assigned by sorted
    filename. --adduct and --adduct-dir are the older spellings of one and many.
    """
    from cadmia.agents import analysis_agent
    from cadmia.display import print_analysis_result

    if report_path and not (adducts or adduct or adduct_dir):
        _write_saved_report("analysis", report_path)
        return
    paths = _resolve_adducts(adducts, adduct, adduct_dir)
    if not (cluster and probe_free):
        raise click.UsageError("--cluster and --probe-free are required (inside a project, "
                               "--report alone writes the report from results/analysis/).")
    texts = []

    def _persist(result, adduct_path):
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_analysis_result(
                conn, result, provenance=_cli_provenance(adduct_path, "analysis"),
            )
            _record_workflow_run(conn, "analysis", ids)
            conn.commit()
        finally:
            conn.close()
        return ids, target

    # ── Batch mode ──
    if len(paths) > 1:
        if cluster_deformed or probe_deformed:
            raise click.UsageError("--cluster-deformed/--probe-deformed are per-config; "
                                   "they cannot be combined with several adducts.")
        out_dir = Path(output) if output else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        failed = []
        for i, ad in enumerate(paths):
            dst = out_dir / f"analysis_{ad.stem}.json"
            try:
                result = analysis_agent.analyze(
                    str(ad), cluster, probe_free, probe=probe, bsse=bsse,
                    cdft_json=cdft_json, config_index=i, cluster_label=cluster_label,
                    bend_threshold_deg=bend_threshold,
                )
                json_str, text = _analysis_stage_json(result, ad)
                dst.write_text(json_str, encoding="utf-8")
            except (ValueError, OSError, KeyError, IndexError) as exc:
                # One unreadable adduct must not cost the analysis of all the others.
                failed.append(ad.name)
                click.echo(f"  ✗ {ad.name}: {type(exc).__name__}: {exc}", err=True)
                continue
            texts.append(f" ── {ad.name} ".ljust(72, "─") + "\n" + text)
            tag = f"{result.e_ads_kcal:+.2f} kcal/mol" if result.e_ads_kcal is not None else "n/a"
            click.echo(f"  {ad.name} → {dst.name}  (E_ads {tag})")
            if use_db:
                _persist(result, str(ad))
        click.echo(f"Analyzed {len(paths) - len(failed)} of {len(paths)} adduct(s) → {out_dir}/"
                   if failed else f"Analyzed {len(paths)} adduct(s) → {out_dir}/")
        if report_path and texts:
            Path(report_path).write_text("\n\n".join(texts) + "\n", encoding="utf-8")
            click.echo(f"Full analysis report written to: {report_path}")
        if failed:
            click.echo(f"  Failed: {', '.join(failed)}", err=True)
            if len(failed) == len(paths):
                raise click.ClickException("No adduct could be analyzed.")
        return

    # ── Single mode ──
    try:
        result = analysis_agent.analyze(
            str(paths[0]), cluster, probe_free,
            probe=probe, bsse=bsse,
            cluster_deformed=cluster_deformed, probe_deformed=probe_deformed,
            cdft_json=cdft_json, config_index=config_index, cluster_label=cluster_label,
            bend_threshold_deg=bend_threshold,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))

    if not no_table:
        print_analysis_result(result)

    try:
        json_str, text = _analysis_stage_json(result, paths[0])
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if report_path:
        Path(report_path).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Full analysis report written to: {report_path}")
    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table:
        click.echo(json_str)

    if use_db:
        ids, target = _persist(result, str(paths[0]))
        click.echo(f"Persisted to {target}: adsorption row {ids['adsorption_id'][:12]}….")


@main.command("stability")
@click.option("--mixed", default=None, type=click.Path(exists=True),
              help="Output file of the mixed oxide whose stability is being measured.")
@click.option("--ref", "references", multiple=True, type=click.Path(exists=True),
              help="Pure-oxide reference output (repeatable, e.g. --ref TiO2.out --ref Al2O3.out).")
@click.option("--coeff", "coefficients", multiple=True,
              help="Stoichiometric coefficient per --ref (same order): 1, 0.5 or 1/2. "
                   "Auto-solved if omitted (½, ½ for references of twice the size).")
@click.option("--label", "mixed_label", default="",
              help="Mixed-oxide label override.")
@click.option("--energy", "energy_basis", default="E0", show_default=True,
              type=click.Choice(["E0", "E", "H", "G", "Gq"]),
              help="Energy used in every term: E0 = E + ZPE, E electronic, H enthalpy, "
                   "G Gibbs as printed, Gq Gibbs with Grimme's quasi-RRHO entropy. Falls back "
                   "to E (with a warning) if any output has no frequencies.")
@click.option("--report", "-r", "report_path", default=None, type=click.Path(),
              help="Also write the full text report here. Alone, inside a project, it writes "
                   "it from results/stability.json without recomputing.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write StabilityResult JSON to this file.")
@click.option("--no-table", "no_table", is_flag=True,
              help="Skip the rich table; emit JSON only.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist the relative stability to the database.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def stability_command(
    mixed, references, coefficients, mixed_label, energy_basis, output, no_table, use_db, db_path,
    report_path=None,
) -> None:
    """Relative (formation) stability of a mixed oxide vs. pure-oxide references."""
    from cadmia.agents import stability_agent
    from cadmia.display import print_stability_result

    if report_path and not (mixed or references):
        _write_saved_report("stability", report_path)
        return
    if not (mixed and references):
        raise click.UsageError("--mixed and --ref are required (inside a project, --report "
                               "alone writes the report from results/stability.json).")

    coeffs = list(coefficients) if coefficients else None
    try:
        result = stability_agent.compute_stability(
            mixed, list(references), coefficients=coeffs, mixed_label=mixed_label,
            energy=energy_basis,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))

    if not no_table:
        print_stability_result(result)

    from cadmia.agents.text_report import stability_report_text
    from cadmia.io.results import json_safe, save_stage_result
    data = {**result.to_dict(), "sources": {"mixed": str(Path(mixed).resolve()),
                                            "references": [str(Path(r).resolve()) for r in references]}}
    text = stability_report_text(data)
    json_str = json.dumps(json_safe({**data, "report_text": text}), indent=2, ensure_ascii=False)
    saved = save_stage_result("stability", data, text)
    if saved:
        click.echo(f"Module result: {saved}")
    if report_path:
        Path(report_path).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Full stability report written to: {report_path}")
    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table:
        click.echo(json_str)

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        from cadmia.db import repository
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = repository.persist_stability_result(
                conn, result, provenance=_cli_provenance(mixed, "stability"),
            )
            _record_workflow_run(conn, "stability", ids)
            conn.commit()
        finally:
            conn.close()
        click.echo(f"Persisted to {target}: stability row {ids['stability_id'][:12]}….")


@main.command("stability-series")
@click.option("--cluster", "clusters", required=True, multiple=True, type=click.Path(exists=True),
              help="Cluster output of one family, mixed or pure (repeatable). n is inferred "
                   "from stoichiometry; clusters with the same n are isomers.")
@click.option("--ref", "references", multiple=True, type=click.Path(exists=True),
              help="Pure-oxide output of any size (repeatable), matched by stoichiometry: size 1 "
                   "gives E_b, the same n gives ΔE_mix(n), equal size gives ΔE_mix,eq per f.u.")
@click.option("--energy", "energy_basis", default="E0", show_default=True,
              type=click.Choice(["E0", "E", "H", "G", "Gq"]),
              help="Energy used in every term: E0 = E + ZPE, E electronic, H enthalpy, "
                   "G Gibbs as printed, Gq Gibbs with Grimme's quasi-RRHO entropy. Falls back "
                   "to E (with a warning) if any output has no frequencies. The lowest isomer "
                   "in this basis represents each n.")
@click.option("--pop", "populations", is_flag=True,
              help="Boltzmann populations of the isomers of each n, in G (σ from the geometry) "
                   "and in E0, with g = 2 if chiral, at the outputs' temperature, plus the "
                   "descriptors averaged with the G populations. Indicative: valid only for the "
                   "isomers given.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write SeriesStabilityResult JSON to this file.")
@click.option("--no-table", "no_table", is_flag=True, help="Skip the rich table; emit JSON only.")
def stability_series_command(clusters, references, energy_basis, populations, output,
                             no_table) -> None:
    """Size-dependent stability along a cluster series: E_b, ΔE_mix, E_b,fam, E_f, Δ²E."""
    from cadmia.agents import stability_agent
    from cadmia.display import print_series_stability

    result = stability_agent.compute_series_stability(
        list(clusters), reference_outs=list(references), energy=energy_basis,
        populations=populations,
    )
    if not no_table:
        print_series_stability(result)
    json_str = result.to_json()
    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table:
        click.echo(json_str)


@main.command("bde")
@click.option("--molecule", required=True, type=click.Path(exists=True),
              help="Parent molecule output file.")
@click.option("--fragment", "fragments", required=True, multiple=True, type=click.Path(exists=True),
              help="Dissociation fragment output (repeatable; usually two).")
@click.option("--label", "molecule_label", default="", help="Molecule label override.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write BDEResult JSON to this file.")
@click.option("--no-table", "no_table", is_flag=True, help="Skip the rich table; emit JSON only.")
def bde_command(molecule, fragments, molecule_label, output, no_table) -> None:
    """Bond dissociation energy: Σ E(fragment) − E(molecule), from fragment outputs."""
    from cadmia.agents import bde_agent
    from cadmia.display import print_bde_result

    result = bde_agent.compute_bde(
        molecule, list(fragments), molecule_label=molecule_label,
    )
    if not no_table:
        print_bde_result(result)
    json_str = result.to_json()
    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(f"JSON written to: {output}")
    elif no_table:
        click.echo(json_str)


@main.command("pipeline")
@click.option("--n",   "path_n",   default=None, type=click.Path(exists=True),
              help="SCF output with N electrons (required unless --resume).")
@click.option("--np1", "path_np1", default=None, type=click.Path(exists=True),
              help="SCF output with N+1 electrons, anion (required unless --resume).")
@click.option("--nm1", "path_nm1", default=None, type=click.Path(exists=True),
              help="SCF output with N-1 electrons, cation (required unless --resume).")
@click.option("--cluster", default="", help="Cluster label (e.g. Al2TiO5).")
@click.option("--probe", default="NH3", show_default=True,
              help="Adsorption probe: built-in (H2, CO2, NH3, CO, N2, O2, H2O) or a "
                   "user-defined probe from ~/.cadmia/probes.yaml.")
@click.option("--ref", "references", multiple=True, type=click.Path(exists=True),
              help="Pure-oxide reference output for the stability stage (repeatable).")
@click.option("--probe-free", "probe_free", default=None, type=click.Path(exists=True),
              help="Isolated-probe output; without it the analyze stage is skipped.")
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Engine; auto-detected from the outputs when omitted.")
@click.option("--engine-path", default=None, type=click.Path(),
              help="ORCA install dir or g16 binary for the calculation stage.")
@click.option("--task",
              type=click.Choice(["sp", "opt", "freq", "opt+freq",
                                  "opt_tight", "opt_tight+freq"]),
              default="opt+freq", show_default=True,
              help="Task for the adsorption configs. opt+freq gives the production geometry "
                   "plus the frequencies the adsorption thermochemistry (ΔG/ΔH) needs. "
                   "TightOpt is not the default because weakly bound probes have near-free "
                   "rotational modes whose step never meets its displacement tolerance, so "
                   "the run exhausts its optimization cycles and never reaches the Freq.")
@click.option("--preset", "preset",
              type=click.Choice(["none", "analysis", "accurate"]),
              default="accurate", show_default=True,
              help="Output/accuracy bundle for the calculation stage. accurate adds the "
                   "production accuracy block (RIJCOSX, def2/J, TightSCF, DefGrid3) on top "
                   "of the analysis property set, so adduct energies match the reoptimized "
                   "cluster/probe. analysis keeps only the properties, and none adds neither.")
@click.option("--functional", default="PBE0", show_default=True,
              help="DFT functional for every generated input in the run.")
@click.option("--basis", default="def2-TZVP", show_default=True, help="Basis set.")
@click.option("--dispersion", type=click.Choice(["d3bj", "d4", "none"]),
              default="d4", show_default=True,
              help="Dispersion correction. d4 depends on the atomic charges, d3bj does not.")
@click.option("--charge", default=0, show_default=True, type=int,
              help="Total charge of the cluster; the adducts add the probe's (0 for every "
                   "built-in probe).")
@click.option("--nprocs", default=8, show_default=True, help="Cores per adsorption job.")
@click.option("--mem-o", "mem_orca_mb_core", default=2000, show_default=True,
              help="ORCA memory per core in MB.")
@click.option("--bsse", is_flag=True, help="BSSE-correct the adsorption energies.")
@click.option("--latex", is_flag=True, help="Also write LaTeX tables in the report.")
@click.option("--timeout", "timeout_s", default=None, type=int,
              help="Per-job wall-time limit in seconds (calculation stage).")
@click.option("--until", type=click.Choice(list(_PIPELINE_STAGES)), default=None,
              help="Stop after this stage.")
@click.option("--resume", is_flag=True,
              help="Resume from pipeline.json in --workdir, skipping finished stages.")
@click.option("--dry-run", "dry_run", is_flag=True,
              help="Print the stage plan without executing anything.")
@click.option("--workdir", "-o", default="./pipeline", type=click.Path(),
              help="Working directory for the manifest and all stage outputs.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Record the pipeline run in the database (workflow_runs).")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def pipeline_command(
    path_n, path_np1, path_nm1, cluster, probe, references, probe_free,
    engine, engine_path, task, preset, functional, basis, dispersion, nprocs,
    mem_orca_mb_core, bsse, latex, timeout_s, until, resume, dry_run,
    workdir, use_db, db_path, charge=0,
) -> None:
    """Run the full pipeline end to end (Orchestrator Agent): validate → cdft →
    geometry → run → analyze → stability → report.

    Resumable: progress is saved to pipeline.json after every stage. The calculation
    stage needs a quantum-chemistry engine; without one it is marked 'blocked' and the
    run stops cleanly, so validate/cdft/geometry can run anywhere and the rest resume
    later with `cadmia pipeline --resume -o WORKDIR --engine-path ...`.
    """
    from cadmia.agents import orchestrator
    from cadmia.agents import geometry_agent
    from cadmia.calculators.params import CalcParams

    if not resume and not all([path_n, path_np1, path_nm1]):
        raise click.UsageError("--n, --np1 and --nm1 are required unless --resume is used.")

    if probe.upper() not in geometry_agent.available_probes():
        raise click.BadParameter(
            f"unknown probe '{probe}'. Available: "
            f"{', '.join(geometry_agent.available_probes())} "
            f"(add more in ~/.cadmia/probes.yaml).",
            param_hint="--probe",
        )

    from cadmia.calculators.presets import apply_preset
    params = CalcParams(functional=functional, basis=basis, dispersion=dispersion,
                        task=task, nprocs=nprocs, mem_orca_mb_core=mem_orca_mb_core,
                        charge=charge, auto_multiplicity=True)
    if engine is None:
        manifest = Path(workdir) / "pipeline.json"
        n_out = path_n or (orchestrator.PipelineState.load(manifest).inputs["n"]
                           if manifest.exists() else None)
        engine = orchestrator.detect_engine(n_out) if n_out else None
    params = apply_preset(params, preset, engine=engine or "orca")

    def on_event(stage: str, msg: str) -> None:
        click.echo(f"  [{stage:<9}] {msg}")

    # On resume the saved level of theory stands unless a level option was given explicitly.
    level_options = ("task", "preset", "functional", "basis", "dispersion", "charge")
    cli_ctx = click.get_current_context()
    explicit = any(cli_ctx.get_parameter_source(o) == click.core.ParameterSource.COMMANDLINE
                   for o in level_options)

    click.echo(f"CADMIA pipeline → {workdir}")
    try:
        state = orchestrator.run_pipeline(
            n=path_n, np1=path_np1, nm1=path_nm1, cluster=cluster, probe=probe,
            refs=list(references), probe_free=probe_free, workdir=workdir,
            engine=engine, engine_path=engine_path, params=params, bsse=bsse,
            latex=latex, timeout_s=timeout_s, until=until, resume=resume,
            dry_run=dry_run, on_event=on_event, params_explicit=explicit or not resume,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))

    if dry_run:
        return

    blocked = state.blocked_stage
    if blocked:
        click.echo(f"\nStopped at '{blocked}': {state.stages[blocked].note}")
    elif len(state.completed) < len(orchestrator.STAGES):
        pending = [n for n in orchestrator.STAGES if n not in state.completed]
        click.echo(f"\nStopped after '{state.completed[-1] if state.completed else '—'}' "
                   f"(--until). Still to run: {', '.join(pending)}. Continue with "
                   f"cadmia pipeline --resume --workdir {workdir}")
    else:
        done = ", ".join(state.completed)
        click.echo(f"\nPipeline complete. Stages: {done}")
        report = state.stages["report"].outputs.get("report_md")
        if report:
            click.echo(f"Report: {report}")
        from cadmia.citation import citation_notice
        click.echo("\n" + citation_notice(), err=True)

    if use_db:
        from cadmia.db.session import init_db, resolve_db_path
        target = _announce_db(resolve_db_path(db_path))
        conn = init_db(target)
        try:
            ids = {n: state.stages[n].status for n in _PIPELINE_STAGES}
            _record_workflow_run(conn, "orchestrator", ids,
                                 status="warning" if blocked else "ok")
            conn.commit()
        finally:
            conn.close()
        click.echo(f"Recorded pipeline run in {target}.")

    if blocked and state.stages[blocked].status == "failed":
        raise SystemExit(1)


class _ManyValues(click.Option):
    """An option that takes every value after it up to the next option, so that
    `--cdft cdft.json cdft_ejemplo.json` and a shell glob (`--cdft results/cdft*.json`) work.
    Repeating it (`--cdft a --cdft b`) adds up the same."""

    def add_to_parser(self, parser, ctx):
        ret = super().add_to_parser(parser, ctx)
        opts = getattr(parser, "_long_opt", None)
        opt = next((opts[n] for n in self.opts if opts and n in opts), None)
        if opt is not None:                     # else: plain repeatable option
            previous = opt.process

            def process(value, state):
                values = [value]
                while state.rargs and not any(state.rargs[0].startswith(p) for p in opt.prefixes):
                    values.append(state.rargs.pop(0))
                previous(tuple(values), state)
            opt.process = process
        return ret


def _module_files(values) -> list:
    """The flattened values of a _ManyValues option."""
    return [v for group in values for v in (group if isinstance(group, tuple) else (group,))]


_MODULE_HELP = ("JSON file(s) of the {what}, one or more after the option; 'all' takes every "
                "{what} saved in the project's results/, copies included.")


@main.command("report")
@click.option("--structure", "structure_json", cls=_ManyValues, multiple=True, type=click.UNPROCESSED,
              help=_MODULE_HELP.format(what="structural report"))
@click.option("--cdft", "cdft_json", cls=_ManyValues, multiple=True, type=click.UNPROCESSED,
              help=_MODULE_HELP.format(what="CDFT result"))
@click.option("--analysis", "analysis_jsons", cls=_ManyValues, multiple=True, type=click.UNPROCESSED,
              help=_MODULE_HELP.format(what="adsorption result"))
@click.option("--analysis-dir", "analysis_dir", default=None, type=click.Path(exists=True),
              help="Directory of AnalysisResult JSONs (adds every *.json in it). Avoids "
                   "listing them one by one with --analysis.")
@click.option("--stability", "stability_json", cls=_ManyValues, multiple=True, type=click.UNPROCESSED,
              help=_MODULE_HELP.format(what="stability result"))
@click.option("--benchmark", "benchmark_json", default=None, type=click.Path(exists=True),
              help="BenchmarkResult JSON.")
@click.option("--literature", "literature_json", default=None, type=click.Path(exists=True),
              help="LiteratureResult JSON (cadmia literature --format json) → adds a References section.")
@click.option("--dos", "dos_files", multiple=True, type=click.Path(exists=True),
              help="DOS file from `cadmia electronic dos` (repeatable) → linked in the report.")
@click.option("--pdos", "pdos_files", multiple=True, type=click.Path(exists=True),
              help="PDOS file from `cadmia electronic pdos` (repeatable) → linked in the report.")
@click.option("--title", default="CADMIA report", help="Report title.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output file. A .md suffix writes the Markdown summary instead of the full "
                   "text output (default: reports/cadmia.out inside a project, else cadmia.out).")
@click.option("--latex", is_flag=True, help="Also write LaTeX tables (.tex).")
@click.option("--figures", is_flag=True,
              help="Also write correlation figures (matplotlib; CSV fallback if absent).")
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Engine to record in the run-information header.")
@click.option("--functional", default=None, help="Functional to record in the header (e.g. PBE0).")
@click.option("--basis", default=None, help="Basis set to record in the header (e.g. def2-TZVP).")
@click.option("--dispersion", default=None, help="Dispersion correction to record (e.g. D3BJ).")
@click.option("--simple", is_flag=True,
              help="Lean Markdown summary (-o NAME.md): result tables only, without the process header/footer.")
@click.option("--narrate", is_flag=True,
              help="Experimental. Append an AI-drafted discussion section (requires LLM "
                   "configuration; see ~/.cadmia/llm.yaml and the provider API key env variable).")
@click.option("--language", type=click.Choice(["es", "en"]), default="es", show_default=True,
              help="Language for the --narrate discussion draft.")
def report_command(structure_json, cdft_json, analysis_jsons, analysis_dir, stability_json, benchmark_json,
                   literature_json, dos_files, pdos_files, title, output, latex, figures,
                   engine, functional, basis, dispersion, simple, narrate, language) -> None:
    """Assemble the CADMIA output from the results of every module (Report Agent).

    Inside a project, with no module options, it reads what the modules saved to results/
    (the originals; copies made from other inputs are left out and listed) and writes
    reports/cadmia.out: the header, each module's full report under its own numbered frame
    (1 Structure, 2 CDFT, 3 Adsorption, 4 Stability), the correlations across modules and the
    closing block. A module option takes one or more JSON files, or 'all' for every result of
    that module in results/, and then only the modules given are reported:

      cadmia report --cdft results/cdft.json results/cdft-2.json --structure all

    With -o NAME.md it writes the Markdown summary (use --simple for its lean variant).
    """
    from cadmia.agents import report_agent
    from cadmia.io.results import find_project_root, load_stage_results

    root = find_project_root()
    saved = None
    explicit = {"structure": _module_files(structure_json), "cdft": _module_files(cdft_json),
                "analysis": _module_files(analysis_jsons), "stability": _module_files(stability_json)}
    if analysis_dir:
        found = sorted(str(p) for p in Path(analysis_dir).glob("*.json"))
        if not found:
            raise click.UsageError(f"--analysis-dir: no *.json files in {analysis_dir}")
        explicit["analysis"] += found

    if any(explicit.values()) or benchmark_json:
        stages = {}
        for name, values in explicit.items():
            items = []
            for v in values:
                if v == "all":
                    if root is None:
                        raise click.UsageError(f"--{name} all reads the project's results/; run it "
                                               "inside a `cadmia init` project.")
                    saved = saved if saved is not None else load_stage_results(root)
                    items += saved.get(name, [])
                elif not Path(v).is_file():
                    raise click.BadParameter(f"no such file: {v}", param_hint=f"--{name}")
                else:
                    items.append({**report_agent._load(v), "_file": v})
            if items:
                stages[name] = items
    elif root is None:
        raise click.UsageError("Outside a project, provide at least one of --structure/--cdft/"
                               "--analysis/--analysis-dir/--stability/--benchmark.")
    else:
        stages = load_stage_results(root, copies=False)
        for err in stages.pop("_errors", []):
            click.echo(f"  ⚠  Skipped {err}", err=True)
        left_out = stages.pop("_copies", [])
        if not stages:
            raise click.ClickException(f"No module results in {root / 'results'} yet: run a "
                                       "module (structure report, cdft, analyze, stability) first.")
        if left_out:
            click.echo(f"  ⓘ  Copies left out ({', '.join(left_out)}); add them with --cdft all, "
                       "--analysis all…, or by name.", err=True)
        conflicts, others = report_agent.cluster_mismatches(stages)
        if conflicts:
            raise click.ClickException(
                "results/ mixes clusters, so they cannot share one output:\n  "
                + "\n  ".join(conflicts) + "\nMove or delete the results that belong to "
                "another study, or give the module files explicitly (--cdft, --analysis…).")
        for other in others:
            click.echo(f"  ⚠  Structure of another formula included: {other}", err=True)
    analyses = stages.get("analysis", [])
    if output is None:
        output = str(root / "reports" / "cadmia.out") if root else "cadmia.out"
    markdown = Path(output).suffix.lower() == ".md"
    if simple and not markdown:
        raise click.UsageError("--simple applies to the Markdown summary (-o NAME.md).")
    from cadmia.io.results import report_paths_mode
    if report_paths_mode() == "absolute":
        click.echo("  ⚠  Paths: absolute — the output shows full paths, with user and "
                   "folder names; check it before sharing ('cadmia paths relative' hides them).",
                   err=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    import os
    report_dir = Path(output).parent
    def _rel(p):
        return os.path.relpath(p, report_dir)
    meta = {
        "engine": engine,
        "functional": functional,
        "basis": basis,
        "dispersion": dispersion,
        "command": command_line(),
        "files": [f for f in [*(v for vs in explicit.values() for v in vs if v != "all"),
                              benchmark_json, literature_json] if isinstance(f, str)],
        "dos_files": [_rel(p) for p in dos_files],
        "pdos_files": [_rel(p) for p in pdos_files],
    }

    first = lambda stage: (stages.get(stage) or [None])[0]
    report = report_agent.build_report(
        structure=first("structure"), cdft=first("cdft"), analyses=analyses,
        stability=first("stability"), benchmark=benchmark_json, literature=literature_json,
        title=title, meta=meta, full=not simple,
    )

    if narrate:
        from cadmia.llm import LLMError, narrate_report
        try:
            narration = narrate_report(report, language=language)
        except LLMError as exc:
            raise click.ClickException(f"--narrate failed: {exc}")
        report.markdown = report.markdown.rstrip() + "\n" + narration.section
        meta["narration"] = narration.section
        click.echo(f"Discussion draft: {narration.provider}/{narration.model}")

    if markdown:
        written = report.to_files(output, latex=latex, figures=figures)
        click.echo(f"Markdown report: {written['markdown']}")
    else:
        Path(output).write_text(report_agent.build_output(
            stages, meta={**meta, "title": title, "root": str(root) if root else None},
            summary=report), encoding="utf-8")
        click.echo(f"CADMIA output:   {output}")
        written = (report.to_files(Path(output).with_suffix(".md"), latex=latex, figures=figures)
                   if latex or figures else {"latex": None, "figures": [], "csv": []})
    if written["latex"]:
        click.echo(f"LaTeX tables:    {written['latex']}")
    for fig in written["figures"]:
        click.echo(f"Figure:          {fig}")
    for csv in written["csv"]:
        click.echo(f"Correlation CSV: {csv}")
    for c in report.correlations:
        r = f"{c.pearson_r:.3f}" if c.pearson_r is not None else "n/a"
        click.echo(f"Correlation {c.name}: Pearson r = {r} ({len(c.x)} points)")
    for w in report.warnings:
        click.echo(f"  ⚠  {w}")


@main.command("guide")
@click.option("-es", "lang", flag_value="es", default=True, help="Spanish guide (default).")
@click.option("-en", "lang", flag_value="en", help="English guide.")
@click.option("--show", is_flag=True, help="Print the full guide instead of its path.")
def guide_command(lang: str, show: bool) -> None:
    """Show the bundled CADMIA usage guide, or print its file path.

    Language: -es (default) / -en. Examples: `cadmia guide -en --show`.
    """
    import cadmia
    name = "usage_guide.txt" if lang == "es" else "usage_guide.en.txt"
    path = Path(cadmia.__file__).resolve().parent / name
    if not path.exists():
        raise click.ClickException(f"Usage guide ({lang}) not bundled at {path}")
    if show:
        click.echo(path.read_text(encoding="utf-8"))
    else:
        click.echo(str(path))
        click.echo("Tip: 'cadmia guide --show' prints it; add '-en' for English.")


@main.command("paths")
@click.argument("mode", required=False, type=click.Choice(["relative", "absolute"]))
def paths_command(mode: str | None) -> None:
    """How text reports (.txt, cadmia.out) print file paths: relative (default) or absolute.

    Without MODE, shows the current setting. JSON results and the database always keep
    the absolute path and the file hash. Examples: `cadmia paths`, `cadmia paths absolute`.
    """
    from cadmia.io import results
    if mode is None:
        click.echo(f"report paths: {results.report_paths_mode()}")
        return
    path = results.set_report_paths_mode(mode)
    click.echo(f"report paths: {mode}  ({path})")
    if mode == "absolute":
        click.echo("  ⚠  Text reports will show full paths, with user and folder names; check them "
                   "before sharing. 'cadmia paths relative' goes back.", err=True)


@main.command("literature")
@click.argument("query", required=False, default="")
@click.option("--doi", "dois", multiple=True,
              help="Explicit DOI to fetch (repeatable). Works with or without a query.")
@click.option("--source", "sources", multiple=True,
              type=click.Choice(["crossref", "openalex", "semanticscholar"]), default=("crossref",),
              show_default=True, help="Bibliographic API(s) to query (repeatable).")
@click.option("--limit", default=10, show_default=True, help="Max references after dedupe/rank.")
@click.option("--from-year", "from_year", default=None, type=int,
              help="Keep only works published on/after this year.")
@click.option("--sort", type=click.Choice(["relevance", "citations", "year"]),
              default="relevance", show_default=True, help="Ranking of the result list.")
@click.option("--format", "fmt", type=click.Choice(["md", "bibtex", "json"]),
              default="md", show_default=True, help="Output format.")
@click.option("--narrate", is_flag=True,
              help="Experimental. Add an AI relevance note (needs LLM config — DOIs/numbers "
                   "always come from the API, never the model).")
@click.option("--language", type=click.Choice(["es", "en"]), default="en", show_default=True,
              help="Language for the --narrate note.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write to this file instead of stdout.")
def literature_command(query, dois, sources, limit, from_year, sort, fmt,
                       narrate, language, output) -> None:
    """Discover references from free bibliographic APIs — no API key (Literature Agent).

    Examples:
      cadmia literature "conceptual DFT heterogeneous catalysis" --from-year 2024 --sort citations
      cadmia literature --doi 10.1021/cr990029p --format bibtex
    """
    from cadmia.agents import literature_agent

    if not query and not dois:
        raise click.UsageError("Provide a search QUERY and/or at least one --doi.")

    result = literature_agent.run(
        query=query, sources=list(sources), limit=limit, from_year=from_year,
        sort=sort, dois=list(dois), narrate=narrate, language=language,
    )

    if fmt == "json":
        text = result.to_json()
    elif fmt == "bibtex":
        text = literature_agent.to_bibtex(result.references)
    else:
        text = literature_agent.to_markdown(result)

    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"{len(result.references)} references written to: {output}")
    else:
        click.echo(text)

    for w in result.warnings:
        click.echo(f"  ⚠  {w}", err=True)


def _echo_diagnosis(diag) -> None:
    """One diagnosis, human-readable."""
    click.echo(f"Diagnosis: {diag.filepath} [{diag.engine}]")
    facts = []
    if diag.energy_hartree is not None:
        facts.append(f"E = {diag.energy_hartree:.6f} Ha")
    if diag.gap_ev is not None:
        facts.append(f"gap = {diag.gap_ev:.2f} eV")
    if diag.scf_n_iterations is not None:
        facts.append(f"{diag.scf_n_iterations} SCF iterations")
    facts.append("no frequencies" if diag.n_imaginary is None
                 else f"{diag.n_imaginary} imaginary")
    if diag.multiplicity is not None:
        facts.append(f"mult = {diag.multiplicity}")
    if facts:
        click.echo("  " + " | ".join(facts))

    if diag.healthy:
        click.echo("  [\u2713] No convergence symptoms detected.")
        return
    for s in diag.symptoms:
        icon = "\u2717" if s.severity == "critical" else "!"
        click.echo(f"  [{icon}] {s.code}: {s.message}")
        if s.evidence:
            click.echo(f"        {s.evidence}")


@main.command("diagnose")
@click.argument("output_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Calculation engine. Auto-detected when omitted.")
@click.option("--no-llm", is_flag=True,
              help="Symptoms only; skip the LLM fix suggestions.")
@click.option("--language", type=click.Choice(["es", "en"]), default="es", show_default=True,
              help="Language for the LLM suggestions.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write the Diagnosis JSON to this file (a JSON array when several "
                   "outputs are given).")
def diagnose_command(output_files, engine, no_llm, language, output) -> None:
    """Diagnose problematic DFT outputs: symptoms (deterministic) + fix suggestions (LLM).

    Takes one or more outputs, so a whole campaign can be triaged at once:
    cadmia diagnose *.out — or cadmia diagnose co2/*.out nh3/*.out.

    Symptom extraction is pure parsing and always runs; the LLM layer (experimental) is
    optional and only proposes convergence strategies for you to verify — nothing is executed. It runs
    only on a single output, so a batch never fans out into one provider call per file.
    """
    import json

    from cadmia.agents import diagnosis_agent

    diags = [diagnosis_agent.diagnose(f, engine=engine) for f in output_files]
    for diag in diags:
        _echo_diagnosis(diag)

    if output:
        Path(output).write_text(diags[0].to_json() if len(diags) == 1
                                else json.dumps([d.to_dict() for d in diags], indent=2),
                                encoding="utf-8")
        click.echo(f"JSON written to: {output}")

    if len(diags) > 1:
        n_bad = sum(1 for d in diags if not d.healthy)
        click.echo(f"\n{len(diags)} output(s) diagnosed, {n_bad} with symptoms.")
        return

    diag = diags[0]
    if no_llm or diag.healthy:
        return
    from cadmia.llm import LLMError, suggest_fixes
    try:
        suggestion = suggest_fixes(diag, language=language)
    except LLMError as exc:
        click.echo(f"\nLLM suggestions unavailable: {exc}")
        return
    click.echo(f"\nSuggested strategies ({suggestion.provider}/{suggestion.model} — verify before use):\n")
    click.echo(suggestion.text)


@main.group("db")
def db_group() -> None:
    """Manage the local CADMIA database."""


@db_group.command("init")
@click.option("--path", default=None, type=click.Path(),
              help="Database path (default: ~/.cadmia/cadmia.db).")
def db_init(path) -> None:
    """Create the database and initialize the schema."""
    from cadmia.db.session import init_db, resolve_db_path
    db_path = Path(path) if path else DEFAULT_DB
    conn = init_db(db_path)
    conn.close()
    click.echo(f"Database initialized: {db_path}")


@db_group.command("status")
@click.option("--path", default=None, type=click.Path())
def db_status(path) -> None:
    """Show record counts for each table."""
    from cadmia.db.session import get_connection, DEFAULT_DB
    db_path = Path(path) if path else DEFAULT_DB
    if not db_path.exists():
        click.echo(f"No database found at {db_path}. Run: cadmia db init")
        return
    tables = ["molecules", "calculations", "descriptors",
              "bond_descriptors", "adsorption", "stability", "run_log",
              "projects", "systems", "structures", "files", "provenance",
              "adsorption_configs", "adsorption_results", "hpc_jobs",
              "workflow_runs"]
    with get_connection(db_path) as conn:
        click.echo(f"Database: {db_path}")
        for table in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                click.echo(f"  {table:<20} {n:>6} rows")
            except Exception:
                click.echo(f"  {table:<20}  (table not found)")


@db_group.command("ask")
@click.argument("question")
@click.option("--path", default=None, type=click.Path(),
              help="Database path (default: ~/.cadmia/cadmia.db).")
@click.option("--limit", default=50, show_default=True,
              help="Maximum number of rows to display.")
@click.option("--sql-only", is_flag=True,
              help="Show the generated SQL without executing it.")
def db_ask(question, path, limit, sql_only) -> None:
    """Query the database in natural language (experimental; an LLM writes a read-only SELECT).

    The model only writes SQL text; it is validated to be a single SELECT and run
    on a read-only connection, so the database cannot be modified. All numbers
    come from the database, never from the model.
    """
    from cadmia.db.session import DEFAULT_DB
    from cadmia.llm import LLMError
    from cadmia.llm.sql import ask_database, database_schema, generate_sql

    db_path = Path(path) if path else DEFAULT_DB
    if not db_path.exists():
        raise click.ClickException(f"No database found at {db_path}. Run: cadmia db init")

    try:
        if sql_only:
            sql, response = generate_sql(question, database_schema(db_path))
            click.echo(f"-- {response.provider}/{response.model}")
            click.echo(sql)
            return
        result = ask_database(question, db_path, limit=limit)
    except LLMError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"SQL ({result.provider}/{result.model}):\n  {result.sql}\n")
    if not result.rows:
        click.echo("(no rows)")
        return
    widths = [max(len(str(c)), *(len(str(r[i])) for r in result.rows))
              for i, c in enumerate(result.columns)]
    click.echo("  ".join(str(c).ljust(w) for c, w in zip(result.columns, widths)))
    click.echo("  ".join("─" * w for w in widths))
    for row in result.rows:
        click.echo("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    if len(result.rows) == limit:
        click.echo(f"(showing first {limit} rows; use --limit to see more)")


@main.command("doctor")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path to check (default: the resolved project or global one).")
def doctor_command(db_path) -> None:
    """Diagnose the CADMIA environment: Python, dependencies, config, database, external tools.

    Exits with code 1 if any required check fails; warnings (optional features) do not fail.
    """
    import importlib
    import importlib.metadata as importlib_metadata
    import shutil
    import sys

    OK, WARN, FAIL = "✓", "!", "✗"
    failures = 0

    def report(status: str, label: str, detail: str = "") -> None:
        click.echo(f"  [{status}] {label:<34} {detail}")

    click.echo("CADMIA doctor")

    click.echo("\nPython & required dependencies:")
    py = sys.version_info
    if py >= (3, 11):
        report(OK, "python", f"{py.major}.{py.minor}.{py.micro}")
    elif py >= (3, 9):
        report(WARN, "python", f"{py.major}.{py.minor}.{py.micro} (3.11+ recommended; 3.9+ supported)")
    else:
        report(FAIL, "python", f"{py.major}.{py.minor} (>= 3.9 required)")
        failures += 1
    # Version comes from the installed distribution, not from a `__version__` attribute: Click
    # drops its own in 9.1, and not every package carries one.
    _DIST = {"yaml": "pyyaml"}
    for mod in ("numpy", "click", "yaml"):
        try:
            importlib.import_module(mod)
        except ImportError:
            report(FAIL, mod, "missing — pip install " + _DIST.get(mod, mod))
            failures += 1
            continue
        try:
            version = importlib_metadata.version(_DIST.get(mod, mod))
        except importlib_metadata.PackageNotFoundError:
            version = ""
        report(OK, mod, version)

    click.echo("\nCADMIA packages:")
    for pkg in ("cadmia.core", "cadmia.io", "cadmia.cdft", "cadmia.electronic",
                "cadmia.structure", "cadmia.agents", "cadmia.db"):
        try:
            importlib.import_module(pkg)
            report(OK, pkg)
        except Exception as exc:
            report(FAIL, pkg, str(exc))
            failures += 1

    click.echo("\nConfiguration:")
    try:
        from cadmia.structure.core.config import default_config_path
        cfg = default_config_path()
        if cfg.exists():
            report(OK, "atomic_data.yaml", str(cfg))
        else:
            report(FAIL, "atomic_data.yaml", f"not found at {cfg}")
            failures += 1
    except Exception as exc:
        report(FAIL, "atomic_data.yaml", str(exc))
        failures += 1

    click.echo("\nOptional dependencies (features degrade gracefully):")
    optional = {
        "rich": "formatted CLI tables",
        "matplotlib": "static figures (PNG/SVG/PDF)",
        "plotly": "interactive 2D plots",
        "scipy": "convex-hull shape descriptors",
    }
    for mod, feature in optional.items():
        try:
            importlib.import_module(mod)
            report(OK, mod, feature)
        except ImportError:
            report(WARN, mod, f"missing — {feature} disabled")

    click.echo("\nLLM (optional — text features only):")
    try:
        import os
        from cadmia.llm import DEFAULT_CONFIG_PATH, load_config
        llm_cfg = load_config()
        cfg_state = str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else "defaults (no llm.yaml)"
        report(OK, "config", f"{llm_cfg.provider}/{llm_cfg.resolved_model()} — {cfg_state}")
        env = llm_cfg.api_key_env()
        if os.environ.get(env, "").strip():
            report(OK, env, "set")
        else:
            report(WARN, env, "not set — --narrate and other LLM features disabled")
    except Exception as exc:
        report(WARN, "llm config", str(exc))

    click.echo("\nExternal tools (PATH):")
    for tool, feature in (("orca", "ORCA engine"), ("g16", "Gaussian 16 engine"),
                          ("xtb", "GFN2-xTB pre-screening (cadmia prescreen)"),
                          ("ssh", "HPC submission")):
        found = shutil.which(tool)
        if tool == "orca" and found and Path(found).resolve().stat().st_size < 5_000_000:
            # /usr/bin/orca on Linux desktops is usually the GNOME screen reader
            report(WARN, tool, f"{found} — looks like the GNOME screen reader, not ORCA QC")
            continue
        report(OK if found else WARN, tool, found or f"not found — {feature} unavailable")

    click.echo("\nSLURM scheduler (for cadmia submit / monitor):")
    sbatch = shutil.which("sbatch")
    squeue = shutil.which("squeue")
    if sbatch and squeue:
        report(OK, "slurm", f"sbatch at {sbatch}")
    else:
        report(WARN, "slurm",
               "sbatch/squeue not found — local SLURM submit unavailable "
               "(remote-SSH profiles run sbatch on the cluster, not here)")

    click.echo("\nDatabase:")
    from cadmia.db.session import resolve_db_path
    # doctor INSPECTS; it must not announce a creation it is not going to perform.
    target = resolve_db_path(db_path)
    if not target.exists():
        report(OK, str(target), "not created yet — the first command that records a result creates it")
    else:
        import sqlite3
        try:
            conn = sqlite3.connect(str(target))
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            required = {"molecules", "calculations", "descriptors", "adsorption",
                        "stability", "run_log", "structures", "provenance"}
            missing = required - tables
            if missing:
                report(WARN, str(target),
                       f"schema outdated (missing: {', '.join(sorted(missing))}) — run: cadmia db init")
            else:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(calculations)")}
                if "provenance_id" in cols:
                    report(OK, str(target), "schema v2 (traceability enabled)")
                else:
                    report(WARN, str(target), "v1 columns — run: cadmia db init to migrate")
            conn.close()
        except Exception as exc:
            report(FAIL, str(target), str(exc))
            failures += 1

    click.echo()
    if failures:
        click.echo(f"{failures} required check(s) failed.")
        raise SystemExit(1)
    click.echo("All required checks passed.")


@main.command("init")
@click.argument("directory", default=".", type=click.Path())
@click.option("--name", default="", help="Project name (default: directory name).")
@click.option("--no-db", "no_db", is_flag=True, help="Do not create the local .cadmia/cadmia.db.")
@click.option("--overwrite", is_flag=True, help="Overwrite an existing manifest/database.")
def init_command(directory, name, no_db, overwrite) -> None:
    """Scaffold a CADMIA project: pipeline folders, manifest, and a local database.

    Inside the project every stage (structure report, cdft, analyze, stability) saves its
    result, data and full report in one JSON, to results/, and `cadmia report` assembles them
    into reports/cadmia.out.
    """
    from cadmia.project import init_project

    summary = init_project(directory, name=name, with_db=not no_db, overwrite=overwrite)
    click.echo(f"Project '{summary['name']}' at {summary['root']}")
    for c in summary["created"]:
        click.echo(f"  + {c}")
    for s in summary["skipped"]:
        click.echo(f"  · {s} (exists, kept)")
    if summary["db"]:
        click.echo(f"Database: {summary['db']}")
    click.echo("Next: import outputs with `cadmia import *.out --db-path "
               f"{summary['db'] or '~/.cadmia/cadmia.db'}`")


@main.command("import")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Calculation engine. Auto-detected per file when omitted.")
@click.option("--functional", default=None, help="Record this functional on each calculation.")
@click.option("--basis", default=None, help="Record this basis set on each calculation.")
@click.option("--dispersion", default=None, help="Record this dispersion correction.")
@click.option("--task", default="scf", show_default=True, help="Calculation task label.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: ~/.cadmia/cadmia.db).")
@click.option("--no-table", "no_table", is_flag=True, help="Emit JSON instead of a table.")
def import_command(files, engine, functional, basis, dispersion, task, db_path, no_table) -> None:
    """Import existing SCF outputs into the database (structure + calculation + provenance)."""
    from cadmia.agents import import_agent

    result = import_agent.import_outputs(
        list(files), db_path=db_path, engine=engine, functional=functional,
        basis=basis, dispersion=dispersion, task=task, command=command_line(),
    )

    if no_table:
        click.echo(result.to_json())
    else:
        for it in result.items:
            if it.error:
                click.echo(f"  ✗ {Path(it.path).name}: {it.error}")
            else:
                e = f"{it.energy_au:.6f}" if it.energy_au is not None else "—"
                sid = (it.structure_id or "")[:12]
                click.echo(f"  ✓ {Path(it.path).name}  {it.engine}  {it.formula}  "
                           f"E={e}  {it.status}  [{sid}…]")
        click.echo(f"Imported {result.n_ok}/{len(result.items)} into "
                   f"{db_path or '~/.cadmia/cadmia.db'}.")

    if result.n_ok < len(result.items):
        raise SystemExit(1)


@main.command("bonds")
@click.argument("output_file", type=click.Path(exists=True))
@click.option("--top", default=0, help="Show only the N strongest bonds (0 = all).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write the result to this file instead of the screen; a .json extension "
                   "writes JSON (the full analysis), anything else the table.")
def bonds_command(output_file, top, output) -> None:
    """Report Mayer bond orders from an ORCA output (needs Print[P_Mayer] 1)."""
    from cadmia.io.mayer import parse_mayer

    result = parse_mayer(output_file)
    if result is None or not result.bonds:
        raise click.ClickException(
            "No Mayer block found. Add `%output Print[P_Mayer] 1 end` to the ORCA input "
            "(Gaussian does not print this analysis)."
        )

    bonds = sorted(result.bonds, key=lambda b: b.order, reverse=True)
    if top:
        bonds = bonds[:top]

    if output and Path(output).suffix.lower() == ".json":
        import json as _json
        from cadmia.core.sidecar import write_json_sidecar
        # JSON carries the whole analysis; --top is a screen cap, not a filter on the data.
        written, err = write_json_sidecar(output, _json.dumps(result.to_dict(), indent=2))
        if err:
            raise click.ClickException(err)
        click.echo(f"JSON → {written}")
        return

    lines = [f"Mayer bond orders (> {result.threshold:g})  —  {Path(output_file).name}", ""]
    for b in bonds:
        lines.append(f"  {b.atom_i:>3}-{b.symbol_i:<2}  {b.atom_j:>3}-{b.symbol_j:<2}  {b.order:7.4f}")
    lines.append("")
    lines.append(f"{len(bonds)} bonds, {len(result.atoms)} atoms.")
    text = "\n".join(lines)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Written to: {output}")
    else:
        click.echo(text)


@main.command("xyz")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--engine", type=click.Choice(["gaussian", "orca"]), default=None,
              help="Force the engine parser; auto-detected from content when omitted.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output .xyz (default: <input>.xyz next to the source).")
@click.option("--comment", default="", help="Comment line for the .xyz (line 2).")
def xyz_command(input_file, engine, output, comment) -> None:
    """Extract a clean .xyz from a Gaussian/ORCA output (engine auto-detected by content)."""
    from cadmia.io import final_energy_from_file, structure_from_file

    structure = structure_from_file(input_file, engine=engine)
    energy = final_energy_from_file(input_file, engine=engine)
    # Carry the converged energy into the comment (line 2) by default, as ORCA's own .xyz does;
    # an explicit --comment wins. rank re-reads energy from the .out, so this is a label, not a source.
    if not comment and energy is not None:
        comment = f"E = {energy:.8f} Eh"
    out = Path(output) if output else Path(input_file).with_suffix(".xyz")
    out.write_text(structure.to_xyz(comment=comment), encoding="utf-8")
    tail = f", E = {energy:.8f} Eh" if energy is not None else ""
    click.echo(f"{structure.n_atoms} atoms ({structure.formula}){tail} → {out}")


@main.command("rank")
@click.argument("sources", nargs=-1, required=True, type=click.Path())
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Single .xyz to write (default: ranked_by_energy.xyz in the cwd; "
                   "a directory writes ranked_by_energy.xyz inside it). You may also give "
                   "the output name as a trailing .xyz argument.")
@click.option("--window", "window_kcal", default=25.0, show_default=True, type=float,
              help="Keep only geometries within this ΔE (kcal/mol) of the lowest (0 = no window). "
                   "The default is generous on purpose: it drops SVP outliers without cutting "
                   "conformers that may reorder at the production level.")
@click.option("--top", "top_n", default=15, show_default=True, type=int,
              help="Keep at most this many lowest-energy unique minima (0 = no cap).")
@click.option("--dedup-rmsd", "dedup_rmsd", default=0.20, show_default=True, type=float,
              help="Group outputs that relaxed to the same minimum (Å, largest atomic "
                   "displacement after superposition; 0 disables). "
                   "Distinct starting configurations routinely funnel to one minimum, so "
                   "without this --top counts files instead of structures.")
@click.option("--cluster", "cluster_ref", type=click.Path(exists=True, dir_okay=False),
              help="Bare-cluster .xyz, optimized at the SAME level as these outputs. Groups "
                   "the minima by the skeleton their cluster relaxed to and gives each group "
                   "its own zero, window and cap, so a more stable cluster isomer cannot "
                   "discard the adducts that kept the reference skeleton.")
@click.option("--db/--no-db", "use_db", default=True, show_default=True,
              help="Persist each output's fate to the database: final energy, which\n"
                   "distinct minimum it belongs to, its skeleton class and whether it\n"
                   "passed the window.")
@click.option("--db-path", default=None, type=click.Path(),
              help="Database path (default: the project's own .cadmia/cadmia.db if the "
                   "working directory is inside one, else ~/.cadmia/cadmia.db).")
def rank_command(sources, output, window_kcal, top_n, dedup_rmsd, cluster_ref,
                 use_db, db_path) -> None:
    """Order many Gaussian/ORCA outputs by energy → one .xyz in energy order.

    Universal and engine-agnostic: reads each output's final electronic energy and geometry,
    sorts them low→high, and concatenates the geometries into a single .xyz with the energy
    gap (kcal/mol) in every block's comment. No optimization — just discrimination by energy
    of results you already computed (e.g. a cheap PBE/def2-SVP screen before high-level DFT).

    --window/--top trim the high-energy tail so the .xyz carries only the winners (feed it
    straight to `cadmia prepare`); the full ordering is still saved in the JSON sidecar.

    Examples:
      cadmia rank *.out ranked.xyz
      cadmia rank *.out --window 30 -o rank/co2.xyz
      cadmia rank *.out --top 5 ranked.xyz
      cadmia rank *.out --cluster cluster_dft.xyz ranked.xyz

    With --cluster the window is measured within each skeleton class instead of against a
    single global minimum. The reference must come from the same level of theory as the
    outputs: for DFT results that is the DFT-optimized bare cluster, not the xTB one.
    """
    from cadmia.agents import rank_agent

    # .out/.log/etc. are inputs; a single .xyz argument is the output name (rank never
    # takes .xyz as input — those carry no energy), so the two usage forms both work.
    srcs = [s for s in sources if not str(s).lower().endswith(".xyz")]
    xyz_args = [s for s in sources if str(s).lower().endswith(".xyz")]
    out = output
    if out is None:
        if len(xyz_args) == 1:
            out = xyz_args[0]
        elif len(xyz_args) > 1:
            raise click.ClickException(
                "Several .xyz arguments given; pass exactly one output name (or use -o)."
            )
        else:
            out = "ranked_by_energy.xyz"
    elif xyz_args:
        raise click.ClickException("Give the output via -o OR a trailing .xyz, not both.")

    if not srcs:
        raise click.ClickException("No input outputs given (expected .out/.log files).")
    missing = [s for s in srcs if not Path(s).exists()]
    if missing:
        raise click.ClickException("Not found: " + ", ".join(missing))

    result = rank_agent.rank(srcs, output=out, window_kcal=window_kcal, top_n=top_n,
                             dedup_rmsd=dedup_rmsd, cluster_ref=cluster_ref)

    click.echo(f"Ranked {result.n_ranked}/{result.n_input} output(s) by energy"
               + (f" → {result.output_xyz}" if result.output_xyz else ""))
    if result.n_unique != result.n_ranked:
        click.echo(f"  {result.n_ranked} output(s) → {result.n_unique} distinct minima "
                   f"(no atom farther than {dedup_rmsd:g} Å); the duplicates are in the "
                   "JSON with the representative they collapsed onto.")
    cut = result.n_selected != result.n_unique          # window/top actually trimmed something
    marked = result.n_selected != result.n_ranked        # some output is absent from the .xyz
    if cut:
        crit = " / ".join(c for c in (f"ΔE ≤ {window_kcal:g} kcal/mol" if window_kcal > 0 else "",
                                      f"top {top_n}" if top_n > 0 else "") if c)
        click.echo(f"  kept {result.n_selected}/{result.n_unique} in the .xyz ({crit}); "
                   "the rest are in the JSON only.")
    if result.ranked:
        mark = "keep" if marked else ""
        click.echo(f"  {mark:>4} {'#':>3}  {'ΔE (kcal/mol)':>13}  {'E (Eh)':>15}  "
                   f"{'Opt':>8}  {'Neg Freq':>8}  output")
        for it in result.ranked:
            # "—" = the output has no Freq at all; 0 = frequencies computed, true minimum.
            imag = "—" if it.n_imaginary is None else str(it.n_imaginary)
            # "—" = single point, no optimization to converge in the first place.
            opt = "—" if it.geom_converged is None else ("ok" if it.geom_converged else "NO-CONV")
            # "≡" = collapsed onto an earlier minimum; "·" = trimmed by the window/top cut.
            flag = "  ✓ " if it.selected else ("  ≡ " if it.duplicate_of else "  · ")
            keep = flag if marked else ""
            click.echo(f"  {keep:>4} {it.rank:>3}  {it.rel_kcal:>13.3f}  {it.energy_hartree:>15.6f}  "
                       f"{opt:>8}  {imag:>8}  {Path(it.source).name}")
        if marked and any(it.duplicate_of for it in result.ranked):
            click.echo("  ≡ ⇒ same minimum as an earlier entry (see duplicate_of in the JSON).")
        if result.n_unconverged:
            click.echo("  Opt NO-CONV ⇒ the optimization never reached a stationary point: the "
                       "energy is provisional and the geometry needs reoptimizing before it is "
                       "used as a result (it is still usable as a starting guess).")
        if any(it.n_imaginary for it in result.ranked):
            click.echo("  Neg Freq > 0 ⇒ saddle point, not a minimum (discard or reoptimize).")

    if result.output_json:
        click.echo(f"  JSON → {result.output_json}")

    if use_db:
        n = _persist_screening(
            db_path, "rank", "rank",
            [dict(source=it.source, energy_hartree=it.energy_hartree, rel_kcal=it.rel_kcal,
                  rank=it.rank, kept=it.selected, converged=it.geom_converged,
                  duplicate_of=it.duplicate_of, cluster_skeleton=it.cluster_skeleton)
             for it in result.items],
            link_by_xyz=False)
        click.echo(f"  DB  → {n} output(s) recorded")

    for w in result.warnings:
        click.echo(f"  ⚠  {w}", err=True)
    if result.n_ranked == 0:
        # A script chaining rank → prepare must not take "nothing ranked" for success.
        raise click.ClickException(f"None of the {result.n_input} output(s) could be ranked "
                                   "(see the warnings above).")


def _correlate_plot(result, out_dir: Path) -> tuple[list[str], bool]:
    """Write one scatter+fit figure per pair (matplotlib); fall back to CSV of points."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception:
        have_mpl = False
    for p in result.pairs:
        if len(p.x) < 2:
            continue
        stem = out_dir / f"{p.descriptor.replace('@', '_at_')}__vs__{p.observable}"
        if have_mpl:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.scatter(p.x, p.y)
            if p.slope is not None:
                xs = [min(p.x), max(p.x)]
                ax.plot(xs, [p.slope * v + p.intercept for v in xs])
            ax.set_xlabel(p.descriptor)
            ax.set_ylabel(p.observable)
            ax.set_title(p.equation(), fontsize=7)
            fig.tight_layout()
            png = stem.with_suffix(".png")
            fig.savefig(png, dpi=150)
            plt.close(fig)
            written.append(str(png))
        else:
            csvp = stem.with_suffix(".csv")
            lines = ["system,x,y"] + [f"{s},{xi},{yi}" for s, xi, yi in zip(p.systems, p.x, p.y)]
            csvp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(str(csvp))
    return written, have_mpl


@main.command("correlate")
@click.option("--experimental", "-e", default=None, type=click.Path(exists=True),
              help="CSV with a 'system' column + observable columns. Optional: omit it to read the "
                   "observable straight from the JSON with --y <descriptor> (theory↔theory).")
@click.option("--descriptors", "-d", "descriptors", multiple=True, required=True,
              type=click.Path(), help="Descriptor JSON files, a directory, or a glob "
                   "(cdft/analyze outputs). Repeatable. Merged per system id.")
@click.option("--cdft", "cdft", default=None, type=click.Path(exists=True),
              help="The cluster CDFT JSON (cdft -o, or results/cdft.json) to broadcast over every config, "
                   "so atom-resolved descriptors like dual@site resolve at each binding site.")
@click.option("--x", "x", default="all",
              help="Descriptor alias (mu, eta, omega, omega_minus, omega_plus, N, ip, ea, gap, "
                   "e_ads, dG_ads, dH_ads, delta_n, e_stab, delta_gap, dq_probe, dq_site, "
                   "f_plus@site, dual@N, phi_norm@site…) or 'all' to scan every available one. Default: all.")
@click.option("--y", "y", default=None,
              help="Observable: a CSV column with -e, or a descriptor alias (e.g. dG_ads) without it. "
                   "With -e, optional if the table has a single observable.")
@click.option("--abs-x", "abs_x", is_flag=True, help="Correlate |x| (e.g. |ΔN|).")
@click.option("--log-x", "log_x", is_flag=True, help="Correlate ln(x) (x>0).")
@click.option("--log-y", "log_y", is_flag=True, help="Correlate ln(y) (y>0).")
@click.option("--plot", is_flag=True, help="Write scatter+fit figures (matplotlib; CSV fallback).")
@click.option("--format", "fmt", type=click.Choice(["txt", "json", "csv"]), default="txt",
              show_default=True, help="Output format.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output file (txt/json/csv). Figures go next to it (or ./correlate).")
def correlate_command(experimental, descriptors, cdft, x, y, abs_x, log_x, log_y, plot, fmt, output) -> None:
    """Correlate computed descriptors against an observable (terminal step of a study).

    Fits a least-squares line descriptor → observable across the systems of a study and reports
    slope, intercept, r, r² and a two-tailed p-value. In scan mode (--x all) it ranks every
    available descriptor against one observable by |r|. Consumes the JSON already produced by
    `cadmia cdft` / `cadmia analyze`; it computes nothing new.

    The observable can be experimental (a -e CSV column) or another computed descriptor read
    from the same JSON (--y dG_ads, no CSV). Pass --cdft to spread one cluster's CDFT report over
    all its configs so dual@site / f_plus@site resolve at each binding site.

    Examples:
      cadmia correlate -e mayr.csv -d study/ --y mayr_N --x omega_minus --plot
      cadmia correlate -d adducts/ --cdft cluster_cdft.json --x dual@site --y dG_ads --plot
    """
    from cadmia.agents import correlate_agent

    x_transform = "abs" if abs_x else ("log" if log_x else "none")
    y_transform = "log" if log_y else "none"
    try:
        result = correlate_agent.correlate(
            experimental, list(descriptors), x=x, y=y, cdft=cdft,
            x_transform=x_transform, y_transform=y_transform,
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc))

    if fmt == "json":
        text = result.to_json()
    elif fmt == "csv":
        rows = ["descriptor,observable,n,slope,intercept,r,r2,p_value"]
        for p in result.ranked:
            rows.append(f"{p.descriptor},{p.observable},{p.n},"
                        f"{_csvnum(p.slope)},{_csvnum(p.intercept)},{_csvnum(p.r)},"
                        f"{_csvnum(p.r2)},{_csvnum(p.p_value)}")
        text = "\n".join(rows)
    else:
        lines = [f"CADMIA correlate — {result.n_systems_matched} system(s) matched", ""]
        header = f"  {'descriptor':<16} {'obs':<10} {'n':>3} {'slope':>11} {'intercept':>11} {'r':>7} {'r²':>7} {'p':>9}"
        lines += [header, "  " + "-" * (len(header) - 2)]
        for p in result.ranked:
            lines.append(
                f"  {p.descriptor:<16} {p.observable:<10} {p.n:>3} "
                f"{_fmtnum(p.slope, 11, 4)} {_fmtnum(p.intercept, 11, 4)} "
                f"{_fmtnum(p.r, 7, 3)} {_fmtnum(p.r2, 7, 3)} {_fmtnum(p.p_value, 9, 4)}"
                + (f"   ⚠ {p.note}" if p.note else "")
            )
        best = result.ranked[0] if result.ranked else None
        if best is not None and best.slope is not None:
            lines += ["", "Best fit:", "  " + best.equation()]
        text = "\n".join(lines)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Correlation written → {output}")
    else:
        click.echo(text)

    if plot:
        out_dir = Path(output).parent / "figures" if output else Path("correlate")
        written, have_mpl = _correlate_plot(result, out_dir)
        if written:
            click.echo(("Figures" if have_mpl else "Point CSVs (matplotlib missing)")
                       + f" written → {out_dir}")

    for w in result.warnings:
        click.echo(f"  ⚠  {w}", err=True)


def _fmtnum(v, width: int, prec: int) -> str:
    return f"{v:>{width}.{prec}f}" if isinstance(v, (int, float)) else f"{'—':>{width}}"


def _csvnum(v) -> str:
    return f"{v:.6g}" if isinstance(v, (int, float)) else ""


def _dispatch_legacy(handler, **kwargs) -> None:
    """Build the argparse-style namespace a legacy `_cmd_*` handler expects and run it.

    Native click commands below reuse the mature handlers in cadmia.electronic.cli /
    cadmia.structure.cli unchanged — only the parsing layer becomes click (so every
    subcommand gains --help / --ayuda through EsGroup), with no logic duplicated.
    """
    from types import SimpleNamespace
    try:
        code = handler(SimpleNamespace(**kwargs))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc))
    raise SystemExit(code or 0)


# ── electronic (native click group: dos | pdos | ir | mep) ──────────────────
@main.group("electronic")
def electronic_group() -> None:
    """Electronic properties: density of states, cube viewers and VMD export."""


@electronic_group.command("dos")
@click.argument("input_file")
@click.option("--sigma", type=float, default=0.1, show_default=True,
              help="Gaussian broadening width in eV.")
@click.option("--points", type=int, default=2000, show_default=True,
              help="Number of energy-grid points.")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Write the DOS curve (.dat/.csv/.json/.html).")
@click.option("--format", "fmt", type=click.Choice(["dat", "csv", "json", "html", "png", "svg", "pdf"]),
              default=None,
              help="Output format; html = interactive plot, png/svg/pdf = static figure (matplotlib). "
                   "Default: the -o suffix, else dat.")
@click.option("--width", type=click.Choice(["single", "double"]), default="single", show_default=True,
              help="Static figure width: single (3.4 in) or double (6.5 in) column.")
def electronic_dos(input_file, sigma, points, output, fmt, width) -> None:
    """Total DOS from the orbital eigenvalues of a Gaussian/ORCA output."""
    from cadmia.electronic.cli import _cmd_dos
    _dispatch_legacy(_cmd_dos, input_file=input_file, sigma=sigma, points=points,
                     output=output, format=fmt, width=width)


@electronic_group.command("pdos")
@click.argument("input_file")
@click.option("--sigma", type=float, default=0.1, show_default=True,
              help="Gaussian broadening width in eV.")
@click.option("--points", type=int, default=2000, show_default=True,
              help="Number of energy-grid points.")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Write the PDOS curves (.dat/.csv/.json/.html).")
@click.option("--format", "fmt", type=click.Choice(["dat", "csv", "json", "html", "png", "svg", "pdf"]),
              default=None,
              help="Output format; html = interactive plot, png/svg/pdf = static figure (matplotlib). "
                   "Default: the -o suffix, else dat.")
@click.option("--width", type=click.Choice(["single", "double"]), default="single", show_default=True,
              help="Static figure width: single (3.4 in) or double (6.5 in) column.")
def electronic_pdos(input_file, sigma, points, output, fmt, width) -> None:
    """Element-projected DOS from an ORCA output with Print[P_OrbPopMO_L] 1."""
    from cadmia.electronic.cli import _cmd_pdos
    _dispatch_legacy(_cmd_pdos, input_file=input_file, sigma=sigma, points=points,
                     output=output, format=fmt, width=width)


@electronic_group.command("ir")
@click.argument("input_file")
@click.option("--scale", type=float, default=1.0, show_default=True,
              help="Empirical frequency scaling factor (e.g. 0.96 for hybrid DFT).")
@click.option("--fwhm", type=float, default=16.0, show_default=True,
              help="Peak full width at half maximum in cm⁻¹.")
@click.option("--lineshape", type=click.Choice(["lorentzian", "gaussian"]),
              default="lorentzian", show_default=True, help="Broadening profile.")
@click.option("--points", type=int, default=4000, show_default=True,
              help="Number of grid points.")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Write the spectrum (.dat/.csv/.json/.html).")
@click.option("--format", "fmt", type=click.Choice(["dat", "csv", "json", "html", "png", "svg", "pdf"]),
              default=None,
              help="Output format; html = interactive plot, png/svg/pdf = static figure (matplotlib). "
                   "Default: the -o suffix, else dat.")
@click.option("--width", type=click.Choice(["single", "double"]), default="single", show_default=True,
              help="Static figure width: single (3.4 in) or double (6.5 in) column.")
def electronic_ir(input_file, scale, fwhm, lineshape, points, output, fmt, width) -> None:
    """IR spectrum from the normal-mode IR intensities of an ORCA Freq output."""
    from cadmia.electronic.cli import _cmd_ir
    _dispatch_legacy(_cmd_ir, input_file=input_file, scale=scale, fwhm=fwhm,
                     lineshape=lineshape, points=points, output=output, format=fmt, width=width)


@electronic_group.command("mep")
@click.argument("input_file")
@click.option("--density", default=None, type=click.Path(exists=True),
              help="Electron-density .cube used to define the molecular surface (optional).")
@click.option("--isovalue", type=float, default=0.001, show_default=True,
              help="Density isovalue defining the surface (e/bohr³).")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Write the MEP report to this file (default: print to stdout).")
def electronic_mep(input_file, density, isovalue, output) -> None:
    """MEP analysis from a .cube: surface extrema, σ² statistics and polarity index."""
    from cadmia.electronic.cli import _cmd_mep
    _dispatch_legacy(_cmd_mep, input_file=input_file, density=density,
                     isovalue=isovalue, output=output)


# ── structure (native click group: structure | report | plot | compare) ────────────────
@main.group("structure", invoke_without_command=True)
@click.option("--report", "-r", "report_path", default=None, type=click.Path(),
              help="Write the full structural report, inside a project, from what the module saved "
                   "to results/structure/, without recomputing.")
@click.pass_context
def structure_group(ctx, report_path) -> None:
    """Structural analysis: ECN/coordination, reports, interactive and static plots."""
    if ctx.invoked_subcommand is not None:
        if report_path:
            raise click.UsageError("structure --report reads results/structure/ on its own; "
                                   "for a report of new inputs use structure report FILE -o NAME.")
        return
    if report_path:
        _write_saved_report("structure", report_path)
    else:
        click.echo(ctx.get_help())


@structure_group.command("report")
@click.argument("input_file", nargs=-1, required=True)
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Report file, relative to the current folder (default: structural_report.<format> "
                   "in the analyzed folder).")
@click.option("--format", "fmt", type=click.Choice(["txt", "csv", "json"]), default=None,
              help="Report format. If omitted, uses the output file suffix.")
@click.option("--config", default=None, type=click.Path(),
              help="Custom YAML (atomic masses, cutoffs, ECN settings).")
@click.option("--details", is_flag=True,
              help="For TXT batch reports, append full per-structure sections.")
def structure_report(input_file, output, fmt, config, details) -> None:
    """Analyze bonds, CN/ECN, polyhedra and write TXT/CSV/JSON reports."""
    from cadmia.structure.cli import _cmd_structure
    _dispatch_legacy(_cmd_structure, input_file=list(input_file), output=output,
                     format=fmt, config=config, details=details)


# Back-compat alias: `cadmia structure structure` still works (hidden in help).
_structure_alias = copy.copy(structure_report)
_structure_alias.hidden = True
structure_group.add_command(_structure_alias, name="structure")


@structure_group.command("plot")
@click.argument("input_file", nargs=-1, required=True)
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Output file (one plot) or directory (batch). Without --format the suffix picks "
                   "the format: .html interactive, .png/.svg/.pdf static (2D kinds). Defaults to current dir.")
@click.option("--format", "fmt", type=click.Choice(["html", "png", "svg", "pdf"]), default=None,
              help="Output format (needed for static figures in a batch). Default: the -o suffix, else html.")
@click.option("--width", type=click.Choice(["single", "double"]), default="single", show_default=True,
              help="Static figure width: single (3.4 in) or double (6.5 in) column.")
@click.option("--kind", type=click.Choice(["bonds", "distances", "ecn-trend", "ecn-heatmap"]),
              default="bonds", show_default=True, help="Plot kind (2D).")
@click.option("--title", default=None, help="Optional figure title.")
@click.option("--config", default=None, type=click.Path(),
              help="Custom YAML (atomic masses, cutoffs, ECN settings).")
@click.option("--ecn-decimals", "ecn_decimals", type=int, default=1, show_default=True,
              help="Display decimals for rounded ECN labels.")
@click.option("--ecn-label", "ecn_label", type=click.Choice(["raw", "rounded", "both"]),
              default="both", show_default=True, help="ECN label style.")
def structure_plot(input_file, output, kind, title, config, ecn_decimals, ecn_label, fmt, width) -> None:
    """Interactive HTML plots, or static PNG/SVG/PDF figures of the 2D kinds (needs [full])."""
    from cadmia.structure.cli import _cmd_plot
    _dispatch_legacy(_cmd_plot, input_file=list(input_file), output=output, kind=kind,
                     title=title, config=config, ecn_decimals=ecn_decimals, ecn_label=ecn_label,
                     format=fmt, width=width)


@structure_group.command("compare")
@click.argument("reference", type=click.Path(exists=True))
@click.argument("candidates", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Emit the comparison as JSON.")
def structure_compare(reference, candidates, as_json) -> None:
    """Compare structures against a reference: same isomer, and how far apart.

    Answers two questions that a bare RMSD cannot separate. The SKELETON says whether the
    connectivity is the same — whether a bond was made or broken — and is invariant to
    rotation, translation and atom order. The DEVIATION, from an optimal superposition, says
    how far the atoms moved. A structure that merely bent has the same skeleton and a large
    deviation; one that rearranged has a different skeleton. Distance alone cannot tell those
    apart, because a deformation routinely moves atoms as far as a rearrangement does.

    Atoms are paired by ELEMENT, not by their position in the file, so the same structure
    written in a different atom order still compares as identical. Both numbers come from that
    one pairing. Structures with a different atom multiset are reported as not comparable
    rather than silently given a meaningless number.
    """
    import json as _json
    from collections import Counter

    import numpy as np

    from cadmia.io.xyz import parse_xyz
    from cadmia.utils.chemistry import skeleton
    from cadmia.utils.geometry import kabsch_paired_deviations

    ref = parse_xyz(reference)
    ref_signature, _, ref_borderline = skeleton(ref.atoms, ref.positions)
    ref_counts = Counter(ref.atoms)

    rows = []
    for path in candidates:
        other = parse_xyz(path)
        if Counter(other.atoms) != ref_counts:
            rows.append({"file": str(path), "comparable": False,
                         "reason": "different atom composition"})
            continue
        signature, _, borderline = skeleton(other.atoms, other.positions)
        # One superposition and one pairing for both numbers. Taking the maximum from an
        # element-paired comparison and the RMSD from a position-paired one would report a
        # large RMSD next to a zero deviation whenever the two files order their atoms
        # differently, and nothing in the output would say why.
        deviations = kabsch_paired_deviations(ref.positions, other.positions,
                                              ref.atoms, other.atoms)
        if deviations.size == 0:
            rows.append({"file": str(path), "comparable": False,
                         "reason": "too few atoms agree to fit a common frame"})
            continue
        rows.append({
            "file": str(path),
            "comparable": True,
            "same_skeleton": signature == ref_signature,
            "rmsd_ang": round(float(np.sqrt((deviations ** 2).mean())), 4),
            "max_deviation_ang": round(float(deviations.max()), 4),
            "borderline_bond": bool(borderline or ref_borderline),
        })

    if as_json:
        click.echo(_json.dumps({"reference": str(reference), "comparisons": rows}, indent=2))
        return

    click.echo(f"Reference: {reference}")
    click.echo(f"{'File':<44}{'Skeleton':<12}{'RMSD':>8}{'MaxDev':>9}")
    for row in rows:
        name = Path(row["file"]).name
        if not row["comparable"]:
            click.echo(f"{name:<44}{'n/a':<12}{row['reason']}")
            continue
        verdict = "same" if row["same_skeleton"] else "DIFFERENT"
        click.echo(f"{name:<44}{verdict:<12}{row['rmsd_ang']:>8.3f}{row['max_deviation_ang']:>9.3f}")
        if row["borderline_bond"]:
            click.echo("    warning: a contact sits near the bond cutoff, so the skeleton "
                       "verdict is not safe here.")
