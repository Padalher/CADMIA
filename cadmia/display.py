"""Terminal display utilities using rich."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from cadmia.agents.analysis_agent import AnalysisResult
    from cadmia.agents.stability_agent import StabilityResult, SeriesStabilityResult
    from cadmia.agents.bde_agent import BDEResult
    from cadmia.agents.benchmark_agent import BenchmarkResult
    from cadmia.agents.calculation_agent import CalculationResult
    from cadmia.agents.cdft_agent import CDFTResult
    from cadmia.agents.geometry_agent import GeometryResult
    from cadmia.agents.validation_agent import ValidationReport

_CHARACTER_STYLE = {
    "electrophilic": "bold red",
    "nucleophilic":  "bold blue",
    "ambiphilic":   "yellow",
    "unknown":       "dim",
}


def print_cdft_table(result: "CDFTResult", top: int = 5) -> None:
    """Print ranked site table and global descriptors to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_print(result, top)
        return

    console = Console()
    console.print(
        f"\n[bold cyan]CADMIA — CDFT Agent[/bold cyan]  |  "
        f"cluster: [bold]{result.cluster}[/bold]  |  "
        f"scheme: {result.charge_scheme}  |  "
        f"id: {result.calculation_id}"
    )

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("rank",      justify="right", style="dim")
    table.add_column("atom",      justify="left")
    table.add_column("f⁺",        justify="right")
    table.add_column("f⁻",        justify="right")
    table.add_column("Δf",        justify="right")
    table.add_column("character", justify="left")
    table.add_column("acid/base", justify="left")

    for site in result.sites[:top]:
        style = _CHARACTER_STYLE.get(site.character, "")
        table.add_row(
            str(site.rank),
            f"{site.atom_symbol}({site.atom_index})",
            f"{site.fukui_plus:.4f}"  if site.fukui_plus  is not None else "—",
            f"{site.fukui_minus:.4f}" if site.fukui_minus is not None else "—",
            f"{site.delta_f:+.4f}"   if site.delta_f     is not None else "—",
            site.character,
            (site.acid_base or "—").replace("lewis_", "").replace("_", " "),
            style=style,
        )

    console.print(table)

    gd = result.global_descriptors
    console.print("[bold]Global descriptors[/bold]")
    pairs = [
        ("HOMO",           gd.get("homo_eV"),           "eV"),
        ("LUMO",           gd.get("lumo_eV"),            "eV"),
        ("Gap",            gd.get("gap_eV"),             "eV"),
        ("IP",             gd.get("ip_eV"),              "eV"),
        ("EA",             gd.get("ea_eV"),              "eV"),
        ("η (hardness)",   gd.get("hardness_eV"),        "eV"),
        ("μ (potential)",  gd.get("mu_eV"),              "eV"),
        ("ω (electrophil)",gd.get("omega_eV"),           "eV"),
        ("N (nucleophil)", gd.get("nucleophilicity_eV"), "eV"),
    ]
    for label, val, unit in pairs:
        if val is not None:
            console.print(f"  {label:<22} {val:+.4f} {unit}")

    ab = result.acid_base or {}
    acid, base = ab.get("strongest_acid_site"), ab.get("strongest_base_site")
    if acid or base:
        parts = []
        if acid:
            parts.append(f"acid [bold]{acid['atom_symbol']}({acid['atom_index']})[/bold]")
        if base:
            parts.append(f"base [bold]{base['atom_symbol']}({base['atom_index']})[/bold]")
        console.print(f"[bold]Lewis sites[/bold]  " + "  |  ".join(parts))
        if ab.get("acid_coordination_note"):
            console.print(f"  [dim]{ab['acid_coordination_note']}[/dim]")

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")

    console.print()


def print_validation_report(report: "ValidationReport") -> None:
    """Print a validation report to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_validation_print(report)
        return

    console = Console()
    overall_style = {"pass": "bold green", "fail": "bold red", "warning": "bold yellow"}.get(
        report.overall, "white"
    )
    console.print(
        f"\n[bold cyan]CADMIA — Validation Agent[/bold cyan]  |  "
        f"[{overall_style}]{report.overall.upper()}[/{overall_style}]  |  "
        f"{Path(report.filepath).name}  ({report.engine})"
    )

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("check",   style="dim", width=22)
    table.add_column("status",  justify="center", width=10)
    table.add_column("message")

    _ICONS = {
        "pass":    "[green]✓[/green]",
        "fail":    "[red]✗[/red]",
        "warning": "[yellow]⚠[/yellow]",
        "skip":    "[dim]–[/dim]",
    }

    for name, result in report.checks.items():
        table.add_row(name.replace("_", " "), _ICONS.get(result.status, result.status), result.message)

    console.print(table)


def print_calculation_result(result: "CalculationResult") -> None:
    """Print a summary table of submitted/monitored HPC jobs."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_calculation_print(result)
        return

    console = Console()
    console.print(
        f"\n[bold cyan]CADMIA — Calculation Agent[/bold cyan]  |  "
        f"profile: [bold]{result.profile}[/bold]  |  "
        f"{len(result.jobs)} jobs  "
        f"({result.n_done} done, {result.n_failed} failed)"
    )

    _STATUS_STYLE = {
        "done":    "[green]done[/green]",
        "failed":  "[red]failed[/red]",
        "running": "[yellow]running[/yellow]",
        "queued":  "[cyan]queued[/cyan]",
        "pending": "[dim]pending[/dim]",
        "unknown": "[dim]unknown[/dim]",
    }
    _VALID_ICON = {"True": "[green]✓[/green]", "False": "[red]✗[/red]", "None": "[dim]—[/dim]"}

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("label",    width=20)
    table.add_column("engine",   width=10)
    table.add_column("job_id",   width=10, justify="right")
    table.add_column("status",   width=12)
    table.add_column("valid",    width=7,  justify="center")
    table.add_column("output",   width=30)

    for j in result.jobs:
        table.add_row(
            j.label,
            j.engine,
            j.job_id or "—",
            _STATUS_STYLE.get(j.status, j.status),
            _VALID_ICON.get(str(j.validation_ok), "—"),
            Path(j.output_path).name if j.output_path else "—",
        )

    console.print(table)

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")
    console.print()


def _fallback_calculation_print(result: "CalculationResult") -> None:
    click.echo(f"\nCADMIA Calculation — {result.profile} — {len(result.jobs)} jobs")
    for j in result.jobs:
        valid = "✓" if j.validation_ok else ("✗" if j.validation_ok is False else "—")
        click.echo(f"  [{j.status:^8}] {j.label:<20} {j.engine:<10} job={j.job_id or '—'} valid={valid}")


def print_benchmark_result(result: "BenchmarkResult") -> None:
    """Print a benchmark comparison table to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_benchmark_print(result)
        return

    console = Console()
    p = result.params
    if p is not None:
        level = f"{p.functional}/{p.basis}"
        if p.dispersion not in ("none", ""):
            level += f" + {p.dispersion.upper()}"
        meta = f"level: {level}  |  task: {p.task}"
    else:
        meta = "level: unknown (parsed from pre-computed outputs)"
    console.print(
        f"\n[bold cyan]CADMIA — Benchmark Agent[/bold cyan]  |  "
        f"cluster: [bold]{result.cluster}[/bold]  |  "
        f"{meta}"
    )

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("metric",      width=22)
    table.add_column("Gaussian",    justify="right", width=16)
    table.add_column("ORCA",        justify="right", width=16)
    table.add_column("Δ / ratio",   justify="right", width=16)

    def _fv(v, fmt=".6f"):
        return f"{v:{fmt}}" if v is not None else "—"

    g, o, c = result.gaussian_result, result.orca_result, result.comparison

    table.add_row("Energy (Ha)",
                  _fv(g.energy_hartree if g else None),
                  _fv(o.energy_hartree if o else None),
                  _fv(c.delta_e_kcal if c else None, ".4f") + (" kcal/mol" if c and c.delta_e_kcal is not None else ""))
    table.add_row("Geometry RMSD (Å)", "—", "—",
                  _fv(c.rmsd_ang if c else None, ".4f"))
    table.add_row("Wall time (s)",
                  _fv(g.wall_time_s if g else None, ".1f"),
                  _fv(o.wall_time_s if o else None, ".1f"),
                  _fv(c.timing_ratio if c else None, ".3f") + ("×" if c and c.timing_ratio is not None else ""))
    table.add_row("Freq RMSD (cm⁻¹)", "—", "—",
                  _fv(c.freq_rmsd_cm1 if c else None, ".2f"))

    console.print(table)

    if c:
        agree_style = "bold green" if c.engine_agreement else "bold red"
        agree_str   = "YES" if c.engine_agreement else "NO"
        console.print(f"  Engine agreement: [{agree_style}]{agree_str}[/{agree_style}]  |  "
                      f"Recommendation: [bold]{c.recommendation}[/bold]")
        if c.flags:
            console.print("[bold yellow]Flags:[/bold yellow]")
            for flag in c.flags:
                console.print(f"  ⚠  {flag}")

    console.print()


def _fallback_benchmark_print(result: "BenchmarkResult") -> None:
    p = result.params
    click.echo(f"\nCADMIA Benchmark — {result.cluster} — {p.functional}/{p.basis} {p.task}")
    g, o, c = result.gaussian_result, result.orca_result, result.comparison
    if g:
        click.echo(f"  Gaussian energy: {g.energy_hartree}")
    if o:
        click.echo(f"  ORCA energy:     {o.energy_hartree}")
    if c:
        click.echo(f"  ΔE (kcal/mol):   {c.delta_e_kcal}")
        click.echo(f"  RMSD (Å):        {c.rmsd_ang}")
        click.echo(f"  Agreement:       {c.engine_agreement}")
        click.echo(f"  Recommendation:  {c.recommendation}")


def print_geometry_result(result: "GeometryResult") -> None:
    """Print a summary of generated adsorption configurations to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_geometry_print(result)
        return

    console = Console()
    console.print(
        f"\n[bold cyan]CADMIA — Geometry Agent[/bold cyan]  |  "
        f"cluster: [bold]{result.cluster}[/bold]  |  "
        f"probe: [bold]{result.probe}[/bold]  |  "
        f"{len(result.configs)} configurations"
    )

    from collections import defaultdict
    by_site: dict = defaultdict(list)
    for c in result.configs:
        by_site[(c.site_atom_idx, c.site_atom_symbol)].append(c)

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("site",     justify="left")
    table.add_column("modes",    justify="right")
    table.add_column("configs",  justify="right")
    table.add_column("dist (Å)", justify="right")
    table.add_column("orientations (sample)", justify="left")

    for (idx, sym), cfgs in sorted(by_site.items()):
        dists = sorted({f"{c.placement_dist:.3f}" for c in cfgs})
        modes = sorted({c.orientation.split("_p")[0] for c in cfgs})
        sample = ", ".join(c.orientation for c in cfgs[:2])
        if len(cfgs) > 2:
            sample += f", … (+{len(cfgs)-2})"
        table.add_row(
            f"{sym}({idx})",
            str(len(modes)),
            str(len(cfgs)),
            "/".join(dists),
            sample,
        )

    console.print(table)

    # Both counters were being computed and written to the sidecar without ever reaching
    # the terminal, so a site quietly losing a quarter of its sampling looked like a
    # complete run.
    if result.n_pushed_out:
        console.print(f"  {result.n_pushed_out} configuration(s) moved outward to clear "
                      f"an overlap (see pushed_out_A)")
    if result.n_rejected_clash:
        console.print(f"  [yellow]{result.n_rejected_clash} configuration(s) dropped: "
                      f"no room for the probe at that site[/yellow]")

    if result.output_dir:
        console.print(f"  .xyz files written to: [dim]{result.output_dir}[/dim]")

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")

    console.print()


def _fallback_geometry_print(result: "GeometryResult") -> None:
    click.echo(f"\nCADMIA Geometry — {result.cluster} + {result.probe} — {len(result.configs)} configs")
    from collections import defaultdict
    by_site: dict = defaultdict(list)
    for c in result.configs:
        by_site[(c.site_atom_idx, c.site_atom_symbol)].append(c)
    for (idx, sym), cfgs in sorted(by_site.items()):
        click.echo(f"  site {sym}({idx}): {len(cfgs)} configs, dist={cfgs[0].placement_dist:.3f} Å")
    if result.n_pushed_out:
        click.echo(f"  {result.n_pushed_out} config(s) moved outward to clear an overlap")
    if result.n_rejected_clash:
        click.echo(f"  {result.n_rejected_clash} config(s) dropped: no room for the probe")


def _fallback_validation_print(report: "ValidationReport") -> None:
    click.echo(f"\nCADMIA Validation — {Path(report.filepath).name} — {report.overall.upper()}")
    for name, result in report.checks.items():
        click.echo(f"  [{result.status:^7}]  {name:<22}  {result.message}")


def _fallback_print(result: "CDFTResult", top: int) -> None:
    click.echo(f"\nCADMIA CDFT — {result.cluster} ({result.charge_scheme})")
    click.echo(f"{'rank':>4}  {'atom':<10} {'f+':>8} {'f-':>8} {'Δf':>9}  {'character':<13} acid/base")
    for site in result.sites[:top]:
        fp = f"{site.fukui_plus:.4f}"  if site.fukui_plus  is not None else "   —"
        fm = f"{site.fukui_minus:.4f}" if site.fukui_minus is not None else "   —"
        df = f"{site.delta_f:+.4f}"   if site.delta_f     is not None else "   —"
        label = f"{site.atom_symbol}({site.atom_index})"
        ab = (site.acid_base or "—").replace("lewis_", "").replace("_", " ")
        click.echo(f"{site.rank:>4}  {label:<10} {fp:>8} {fm:>8} {df:>9}  {site.character:<13} {ab}")


def print_analysis_result(result: "AnalysisResult") -> None:
    """Print adsorption energetics and activation summary to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_analysis_print(result)
        return

    console = Console()
    site = (f"{result.site_atom_symbol}({result.site_atom_index})"
            if result.site_atom_index is not None else "—")
    console.print(
        f"\n[bold cyan]CADMIA — Analysis Agent[/bold cyan]  |  "
        f"cluster: [bold]{result.cluster}[/bold]  |  probe: [bold]{result.probe}[/bold]  |  "
        f"binding site: [bold]{site}[/bold]  |  engine: {result.engine}"
    )

    def _kcal(v):
        return f"{v:+.3f}" if v is not None else "—"

    energetics = Table(box=box.SIMPLE_HEAD, show_footer=False)
    energetics.add_column("quantity", justify="left")
    energetics.add_column("kcal/mol", justify="right")
    energetics.add_row("E_ads" if result.e_ads_bsse_kcal is not None else "E_ads (no BSSE)",
                       _kcal(result.e_ads_kcal))
    if result.e_ads_bsse_kcal is not None:
        energetics.add_row("E_ads (BSSE)", _kcal(result.e_ads_bsse_kcal))
    if result.e_def_cluster_kcal is not None:
        energetics.add_row("E_def(cluster)", _kcal(result.e_def_cluster_kcal))
    if result.e_def_probe_kcal is not None:
        energetics.add_row("E_def(probe)", _kcal(result.e_def_probe_kcal))
    if result.e_int_kcal is not None:
        energetics.add_row("E_int", _kcal(result.e_int_kcal))
    if result.delta_h_ads_kcal is not None:
        energetics.add_row("ΔH_ads", _kcal(result.delta_h_ads_kcal))
    if result.delta_g_ads_kcal is not None:
        energetics.add_row("ΔG_ads", _kcal(result.delta_g_ads_kcal))
    if result.minus_t_delta_s_kcal is not None:
        energetics.add_row("−TΔS", _kcal(result.minus_t_delta_s_kcal))
    console.print(energetics)

    if result.probe_bonds:
        bonds = Table(box=box.SIMPLE_HEAD, show_footer=False)
        bonds.add_column("bond", justify="left")
        bonds.add_column("free (Å)", justify="right")
        bonds.add_column("adduct (Å)", justify="right")
        bonds.add_column("Δ (Å)", justify="right")
        bonds.add_column("%", justify="right")
        for b in result.probe_bonds:
            bonds.add_row(b.label, f"{b.free_A:.3f}", f"{b.adduct_A:.3f}",
                          f"{b.delta_A:+.3f}", f"{b.percent:+.1f}")
        console.print(bonds)
        flag = ("[bold red]ACTIVATED[/bold red]" if result.activated
                else "[dim]no significant activation[/dim]")
        console.print(f"  max elongation: {result.max_elongation_pct:+.1f}%  → {flag}")

    if result.cdft_agreement is not None:
        verdict = ("[green]agrees[/green]" if result.cdft_agreement
                   else "[yellow]discrepancy[/yellow]")
        console.print(
            f"  CDFT cross-ref: binding site is the #{result.cdft_site_rank} "
            f"{result.cdft_site_character} site → {verdict}"
        )

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")
    console.print()


def _fallback_analysis_print(result: "AnalysisResult") -> None:
    site = (f"{result.site_atom_symbol}({result.site_atom_index})"
            if result.site_atom_index is not None else "—")
    click.echo(f"\nCADMIA Analysis — {result.cluster} + {result.probe} — binding site {site}")
    def _k(v): return f"{v:+.3f}" if v is not None else "—"
    _bsse = result.e_ads_bsse_kcal is not None
    click.echo(f"  E_ads{'' if _bsse else ' (no BSSE)'} = {_k(result.e_ads_kcal)} kcal/mol",
               nl=False)
    if _bsse:
        click.echo(f"  | E_ads(BSSE) = {_k(result.e_ads_bsse_kcal)}", nl=False)
    click.echo()
    if result.delta_g_ads_kcal is not None or result.delta_h_ads_kcal is not None:
        click.echo(f"  ΔH_ads = {_k(result.delta_h_ads_kcal)}  | "
                   f"ΔG_ads = {_k(result.delta_g_ads_kcal)}  | "
                   f"−TΔS = {_k(result.minus_t_delta_s_kcal)} kcal/mol")
    for b in result.probe_bonds:
        click.echo(f"  bond {b.label}: {b.free_A:.3f} → {b.adduct_A:.3f} Å ({b.percent:+.1f}%)")
    if result.max_elongation_pct is not None:
        click.echo(f"  max elongation {result.max_elongation_pct:+.1f}% → "
              f"{'ACTIVATED' if result.activated else 'no activation'}")
    if result.cdft_agreement is not None:
        click.echo(f"  CDFT: #{result.cdft_site_rank} {result.cdft_site_character} site → "
              f"{'agrees' if result.cdft_agreement else 'discrepancy'}")
    for w in result.warnings:
        click.echo(f"  ! {w}")


def _terms_line(result) -> str:
    """'label mult/imag' for every term; imag '—' when there was no frequency calculation."""
    return "  multiplicity / imaginary modes: " + " · ".join(
        f"{label} {mult}/{'—' if result.n_imaginary.get(label) is None else result.n_imaginary[label]}"
        for label, mult in result.multiplicities.items())


def print_stability_result(result: "StabilityResult") -> None:
    """Print relative (formation) stability vs. pure-oxide references."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_stability_print(result)
        return

    console = Console()
    bal = "[green]balanced[/green]" if result.balanced else "[yellow]UNBALANCED[/yellow]"
    console.print(
        f"\n[bold cyan]CADMIA — Stability Agent[/bold cyan]  |  "
        f"mixed: [bold]{result.mixed}[/bold] ({result.formula}, {result.n_atoms} atoms)  |  "
        f"engine: {result.engine}  |  stoichiometry: {bal}  |  energy: [bold]{result.energy_basis}[/bold]"
    )

    refs = Table(box=box.SIMPLE_HEAD, show_footer=False)
    refs.add_column("reference", justify="left")
    refs.add_column("formula", justify="left")
    refs.add_column("coeff", justify="right")
    refs.add_column(f"{result.energy_basis} (Ha)", justify="right")
    for r in result.references:
        refs.add_row(r.label, r.formula, _coeff(r.coefficient),
                     f"{r.e_hartree:.6f}" if r.e_hartree is not None else "—")
    console.print(refs)
    if result.multiplicities:
        console.print(_terms_line(result))

    if result.e_rel_kcal is not None:
        verdict = ("[green]more stable than pure oxides[/green]" if result.e_rel_kcal < 0
                   else "[red]less stable than pure oxides[/red]")
        console.print(
            f"  E_rel({result.energy_basis}) = [bold]{result.e_rel_kcal:+.3f}[/bold] kcal/mol  "
            f"({result.e_rel_eV:+.4f} eV, {result.e_rel_eV_per_atom:+.4f} eV/atom)  → {verdict}"
        )
        if result.formula_units > 1 or result.reference_size == "equal-size":
            console.print(f"  per formula unit (n = {result.formula_units}, references "
                          f"{result.reference_size}): [bold]{result.e_rel_kcal_per_fu:+.3f}[/bold] kcal/mol")
        if len(result.e_rel_by_basis_kcal) > 1:
            console.print("  by basis (kcal/mol): " + "   ".join(
                f"{basis} {value:+.2f}" for basis, value in result.e_rel_by_basis_kcal.items()))
            if result.formula_units > 1:
                console.print("  by basis, per f.u.:  " + "   ".join(
                    f"{basis} {value / result.formula_units:+.2f}"
                    for basis, value in result.e_rel_by_basis_kcal.items()))

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")
    console.print()


def _fallback_stability_print(result: "StabilityResult") -> None:
    bal = "balanced" if result.balanced else "UNBALANCED"
    click.echo(f"\nCADMIA Stability — {result.mixed} ({result.formula}) — stoichiometry {bal}")
    for r in result.references:
        e = f"{r.e_hartree:.6f}" if r.e_hartree is not None else "—"
        click.echo(f"  {_coeff(r.coefficient)}× {r.label} ({r.formula})  E={e} Ha")
    if result.multiplicities:
        click.echo(_terms_line(result))
    if result.e_rel_kcal is not None:
        click.echo(f"  E_rel({result.energy_basis}) = {result.e_rel_kcal:+.3f} kcal/mol "
              f"({result.e_rel_eV_per_atom:+.4f} eV/atom; {result.e_rel_kcal_per_fu:+.3f} "
              f"kcal/mol per f.u., references {result.reference_size})")
    for w in result.warnings:
        click.echo(f"  ! {w}")


def _fmt(value, spec="+.3f") -> str:
    return format(value, spec) if value is not None else "—"


def _coeff(c) -> str:
    from fractions import Fraction
    return str(Fraction(c).limit_denominator(1000))


_SERIES_COLUMNS = (("E_b", "e_b_kcal"), ("ΔE_mix(n)", "de_mix_kcal"),
                   ("ΔE_mix,eq/f.u.", "de_mix_eq_kcal"), ("E_b,fam", "e_b_family_kcal"),
                   ("E_f", "e_f_kcal"), ("Δ²E", "delta2_kcal"))


def _series_columns(members):
    """Descriptor columns that carry a value in some member: no column of dashes only."""
    return [(name, attr) for name, attr in _SERIES_COLUMNS
            if attr in ("e_b_family_kcal", "e_f_kcal", "delta2_kcal")
            or any(getattr(m, attr) is not None for m in members)]


def _isomer_columns(result):
    bases = list(dict.fromkeys(b for i in result.isomers for b in i.delta_kcal))
    return bases, any(i.population is not None for i in result.isomers)


def print_series_stability(result: "SeriesStabilityResult") -> None:
    """Print size-dependent stability descriptors along a series, and its isomers."""
    columns = _series_columns(result.members)
    bases, with_population = _isomer_columns(result)
    weighted_title = (f"population-weighted average (weights: {result.population_basis} "
                      f"populations; not the free energy of the ensemble, no mixing entropy)")
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        click.echo(f"\nCADMIA Stability series (kcal/mol, energy {result.energy_basis})")
        for title, members in (("", result.members), (weighted_title, result.weighted_members)):
            if not members:
                continue
            if title:
                click.echo(f"\n  {title}")
            click.echo("  n   formula        " + " ".join(f"{name:>14}" for name, _ in columns))
            for m in members:
                click.echo(f"  {m.n:<3} {m.formula:<14} "
                           + " ".join(f"{_fmt(getattr(m, attr)):>14}" for _, attr in columns))
        if result.isomers:
            click.echo("\n  isomers: n, label, point group, " + ", ".join(f"Δ{b}" for b in bases)
                       + (f", g, P({result.population_basis}), P(E0)" if with_population else ""))
            for i in result.isomers:
                click.echo(f"  {i.n:<3} {i.label}{' *' if i.representative else ''}  "
                           f"{i.point_group or '—'}  "
                           + "  ".join(_fmt(i.delta_kcal.get(b), ".2f") for b in bases)
                           + (f"  {i.degeneracy or '—'}  {_fmt(i.population and 100 * i.population, '.1f')} %"
                              f"  {_fmt(i.population_e0 and 100 * i.population_e0, '.1f')} %"
                              if with_population else ""))
        if result.multiplicities:
            click.echo(_terms_line(result))
        for note in result.notes:
            click.echo(f"  · {note}")
        for w in result.warnings:
            click.echo(f"  ! {w}")
        return

    console = Console()
    console.print(
        f"\n[bold cyan]CADMIA — Stability series[/bold cyan]  |  "
        f"engine: {result.engine or '—'}  |  energy: [bold]{result.energy_basis}[/bold]  |  "
        f"descriptors in kcal/mol (eV in JSON)"
    )

    def series_table(members, title=None):
        table = Table(box=box.SIMPLE_HEAD, title=title)
        table.add_column("n", justify="right")
        if result.isomers:
            table.add_column("cluster", justify="left")
        table.add_column("formula", justify="left")
        table.add_column(f"{result.energy_basis} (Ha)", justify="right")
        for name, _ in columns:
            table.add_column(name, justify="right")
        for m in members:
            table.add_row(
                str(m.n), *([m.label] if result.isomers else []), m.formula,
                f"{m.e_hartree:.6f}" if m.e_hartree is not None else "—",
                *(_fmt(getattr(m, attr)) for _, attr in columns),
            )
        console.print(table)

    series_table(result.members)
    if result.isomers:
        iso = Table(box=box.SIMPLE_HEAD,
                    title=f"Isomers (Δ vs. the lowest of each n, kcal/mol; * = represents n in "
                          f"{result.energy_basis})")
        for name in ("n", "isomer", "point group", "chiral",
                     *(f"Δ{b}" for b in bases),
                     *(("g", f"P {result.population_basis} (%)", "P E0 (%)") if with_population else ())):
            iso.add_column(name, justify="left" if name in ("isomer", "point group") else "right")
        for i in result.isomers:
            iso.add_row(str(i.n), i.label + (" *" if i.representative else ""),
                        i.point_group or "—", "—" if i.chiral is None else ("yes" if i.chiral else "no"),
                        *(_fmt(i.delta_kcal.get(b), ".2f") for b in bases),
                        *((str(i.degeneracy or "—"),
                           _fmt(i.population and 100 * i.population, ".1f"),
                           _fmt(i.population_e0 and 100 * i.population_e0, ".1f"))
                          if with_population else ()))
        console.print(iso)
        if with_population:
            console.print(f"  Populations: in {result.population_basis} (σ from the geometry) and in E0, "
                          f"g = 2 if chiral, T = {result.population_temperature_k} K. Valid only "
                          "for the isomers given; indicative, since an error of 1 kcal/mol in ΔG "
                          "changes them a lot.")
    if result.weighted_members:
        series_table(result.weighted_members, title=weighted_title)
    if result.multiplicities:
        console.print(_terms_line(result))
    console.print("  E_b, ΔE_mix < 0: more stable than the pure references  ·  E_b,fam > 0: bound "
                  "vs. its own monomer  ·  E_f > 0: stable to fragmentation")
    for note in result.notes:
        console.print(f"  [cyan]·[/cyan] {note}")
    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")
    console.print()


def print_bde_result(result: "BDEResult") -> None:
    """Print a bond dissociation energy result."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        click.echo(f"\nCADMIA BDE — {result.molecule} ({result.formula})")
        for f in result.fragments:
            e = f"{f.e_hartree:.6f}" if f.e_hartree is not None else "—"
            click.echo(f"  fragment {f.label} ({f.formula})  E={e} Ha")
        if result.bde_kcal is not None:
            click.echo(f"  BDE = {result.bde_kcal:.3f} kcal/mol ({result.bde_eV:.4f} eV)")
        for w in result.warnings:
            click.echo(f"  ! {w}")
        return

    console = Console()
    cons = "[green]atoms conserved[/green]" if result.atoms_conserved else "[yellow]atom MISMATCH[/yellow]"
    console.print(
        f"\n[bold cyan]CADMIA — BDE[/bold cyan]  |  molecule: [bold]{result.molecule}[/bold] "
        f"({result.formula})  |  engine: {result.engine}  |  {cons}"
    )
    frags = Table(box=box.SIMPLE_HEAD)
    frags.add_column("fragment", justify="left")
    frags.add_column("formula", justify="left")
    frags.add_column("E (Ha)", justify="right")
    for f in result.fragments:
        frags.add_row(f.label, f.formula,
                      f"{f.e_hartree:.6f}" if f.e_hartree is not None else "—")
    console.print(frags)
    if result.bde_kcal is not None:
        console.print(f"  BDE = [bold]{result.bde_kcal:.3f}[/bold] kcal/mol "
                      f"({result.bde_eV:.4f} eV)")
    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  ⚠  {w}")
    console.print()
