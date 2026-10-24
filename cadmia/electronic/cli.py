"""argparse handlers behind `cadmia electronic` (DOS, PDOS, IR, MEP, cubes, VMD)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadmia.electronic.dos import dos_from_output, pdos_from_output, write_dos, write_pdos


def _cmd_ir(args) -> int:
    from cadmia.electronic.ir_spectrum import (
        ir_spectrum_from_output, write_ir_spectrum, _peak_table,
    )
    spectrum = ir_spectrum_from_output(
        args.input_file, scale=args.scale, fwhm=args.fwhm,
        lineshape=args.lineshape, n_points=args.points,
    )
    if spectrum.peak_positions.size == 0:
        print("No IR-active vibrational modes found. "
              "Was this a frequency (! Freq) calculation?")
        return 1

    if spectrum.n_imaginary:
        # Imaginary modes carry no IR intensity, so they vanish from the curve on their own and
        # a saddle point plots like a minimum. The spectrum is still drawn; what it is a
        # spectrum OF is the thing worth saying.
        print(f"WARNING: {spectrum.n_imaginary} imaginary mode(s) — this geometry is not a "
              "minimum, and they contribute nothing to the curve below.")

    if args.format == "html":
        from cadmia.electronic.plot import write_ir_html
        out = args.output or (Path(args.input_file).stem + "_ir.html")
        write_ir_html(spectrum, out, title=f"IR spectrum — {Path(args.input_file).name}")
        print(f"IR plot written to: {out}")
    elif args.output:
        write_ir_spectrum(spectrum, args.output, fmt=args.format)
        print(f"IR spectrum written to: {args.output}")
    else:
        peaks = _peak_table(spectrum)
        scale_note = "" if args.scale == 1.0 else f" (scaled ×{args.scale:g})"
        print(f"IR spectrum{scale_note}: {len(peaks)} active modes  |  "
              f"{spectrum.lineshape}, FWHM {spectrum.fwhm_cm1:g} cm⁻¹")
        print("  cm⁻¹      km/mol")
        for p in peaks[:12]:
            print(f"  {p['wavenumber_cm1']:8.1f}  {p['ir_intensity_km_mol']:8.2f}")
        if len(peaks) > 12:
            print(f"  … {len(peaks) - 12} more")
        print("(use --output to write the full curve as .dat/.csv/.json/.html)")
    return 0


def _cmd_dos(args) -> int:
    dos = dos_from_output(args.input_file, sigma=args.sigma, n_points=args.points)

    if args.format == "html":
        from cadmia.electronic.plot import write_dos_html
        out = args.output or (Path(args.input_file).stem + "_dos.html")
        write_dos_html(dos, out, title=f"DOS — {Path(args.input_file).name}")
        print(f"Interactive DOS plot written to: {out}")
        return 0
    if args.format == "json":
        payload = {
            "energy_ev": dos.energy_ev.tolist(),
            "dos_total": dos.dos_total.tolist(),
            "dos_alpha": dos.dos_alpha.tolist(),
            "dos_beta": dos.dos_beta.tolist() if dos.dos_beta is not None else None,
            "spin_polarized": dos.is_spin_polarized,
            "homo_ev": dos.homo_ev,
            "lumo_ev": dos.lumo_ev,
            "gap_ev": dos.gap_ev,
        }
        text = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"JSON written to: {args.output}")
        else:
            print(text)
    elif args.output:
        write_dos(dos, args.output, fmt=args.format)
        print(f"DOS written to: {args.output}")
    else:
        gap = f"{dos.gap_ev:.4f}" if dos.gap_ev is not None else "—"
        print(f"DOS: {len(dos.energy_ev)} points over "
              f"[{dos.energy_ev[0]:.2f}, {dos.energy_ev[-1]:.2f}] eV  |  "
              f"HOMO={dos.homo_ev:.4f}  LUMO={dos.lumo_ev:.4f}  gap={gap} eV  |  "
              f"{'spin-polarized' if dos.is_spin_polarized else 'closed-shell'}")
        print("(use --output to write the full curve as .dat/.csv/.json)")
    return 0


def _cmd_mep(args) -> int:
    from cadmia.io.cube import parse_cube
    from cadmia.electronic.mep import analyze_mep, write_mep_report
    mep_grid = parse_cube(args.input_file)
    density_grid = parse_cube(args.density) if args.density else None
    stats = analyze_mep(mep_grid, density_grid=density_grid, isovalue=args.isovalue)
    if args.output:
        write_mep_report(stats, args.output)
        print(f"MEP report written to: {args.output}")
    else:
        print(str(stats))
    return 0


def _cmd_cube(args) -> int:
    from cadmia.electronic.cube_plot import write_cube_html
    out = args.output or (Path(args.input_file).stem + "_view.html")
    warnings = write_cube_html(
        args.input_file, out, isovalue=args.isovalue,
        max_points=args.max_points, field_type=args.field_type,
    )
    print(f"Interactive cube viewer written to: {out}")
    for w in warnings:
        print(f"  ! {w}")
    return 0


def _cmd_vmd(args) -> int:
    from cadmia.electronic.vmd_export import write_vmd_script
    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        print(f"Invalid --resolution '{args.resolution}'; expected WxH (e.g. 1920x1440).")
        return 1
    out = args.output or (Path(args.input_file).stem + "_vmd.tcl")
    warnings = write_vmd_script(
        args.input_file, out, isovalue=args.isovalue, resolution=(w, h),
        material=args.material, render_output=args.render_output, tachyon=args.tachyon,
    )
    print(f"VMD/Tachyon script written to: {out}")
    print(f"  Run it with:  vmd -e {out}    then in the VMD console:  render_hires")
    for warn in warnings:
        print(f"  ! {warn}")
    return 0


def _cmd_pdos(args) -> int:
    pdos = pdos_from_output(args.input_file, sigma=args.sigma, n_points=args.points)

    if args.format == "html":
        from cadmia.electronic.plot import write_pdos_html
        out = args.output or (Path(args.input_file).stem + "_pdos.html")
        write_pdos_html(pdos, out, title=f"PDOS — {Path(args.input_file).name}")
        print(f"Interactive PDOS plot written to: {out}")
        return 0
    if args.format == "json":
        payload = {
            "energy_ev": pdos.energy_ev.tolist(),
            "series": {e: s.tolist() for e, s in pdos.series.items()},
            "dos_total": pdos.total.tolist(),
            "spin_polarized": pdos.spin_polarized,
            "homo_ev": pdos.homo_ev,
            "lumo_ev": pdos.lumo_ev,
            "gap_ev": pdos.gap_ev,
        }
        text = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"JSON written to: {args.output}")
        else:
            print(text)
    elif args.output:
        write_pdos(pdos, args.output, fmt=args.format)
        print(f"PDOS written to: {args.output}")
    else:
        gap = f"{pdos.gap_ev:.4f}" if pdos.gap_ev is not None else "—"
        print(f"PDOS: elements {', '.join(sorted(pdos.series))}  |  "
              f"{len(pdos.energy_ev)} points over "
              f"[{pdos.energy_ev[0]:.2f}, {pdos.energy_ev[-1]:.2f}] eV  |  "
              f"gap={gap} eV  |  "
              f"{'spin-polarized' if pdos.spin_polarized else 'closed-shell'}")
        print("(use --output to write the full curves as .dat/.csv/.json)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadmia electronic",
        description="Electronic-property analysis: DOS, PDOS, IR, MEP and cube tools.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    dos = sub.add_parser("dos", help="Total DOS from the orbital eigenvalues of a Gaussian/ORCA output.")
    dos.add_argument("input_file", metavar="OUTPUT", help="Gaussian or ORCA output file.")
    dos.add_argument("--sigma", type=float, default=0.1, help="Gaussian broadening width in eV (default: 0.1).")
    dos.add_argument("--points", type=int, default=2000, help="Minimum number of energy-grid points; more are used for a step of sigma/4 (default: 2000).")
    dos.add_argument("-o", "--output", default=None,
                     help="Write the DOS curve to this file (.dat/.csv/.json/.html).")
    dos.add_argument("--format", choices=["dat", "csv", "json", "html"], default="dat",
                     help="Output format; html writes an interactive plot.")
    dos.set_defaults(func=_cmd_dos)

    pdos = sub.add_parser(
        "pdos",
        help="Element-projected DOS from an ORCA output with Print[P_OrbPopMO_L] 1.",
    )
    pdos.add_argument("input_file", metavar="OUTPUT", help="ORCA output file with per-MO Loewdin populations.")
    pdos.add_argument("--sigma", type=float, default=0.1, help="Gaussian broadening width in eV (default: 0.1).")
    pdos.add_argument("--points", type=int, default=2000, help="Minimum number of energy-grid points; more are used for a step of sigma/4 (default: 2000).")
    pdos.add_argument("-o", "--output", default=None,
                      help="Write the PDOS curves to this file (.dat/.csv/.json/.html).")
    pdos.add_argument("--format", choices=["dat", "csv", "json", "html"], default="dat",
                      help="Output format; html writes an interactive plot.")
    pdos.set_defaults(func=_cmd_pdos)

    ir = sub.add_parser(
        "ir",
        help="IR spectrum from the normal-mode IR intensities of an ORCA Freq output.",
    )
    ir.add_argument("input_file", metavar="OUTPUT", help="ORCA output of a frequency (! Freq) run.")
    ir.add_argument("--scale", type=float, default=1.0,
                    help="Empirical frequency scaling factor (e.g. 0.96 for hybrid DFT; default: 1.0).")
    ir.add_argument("--fwhm", type=float, default=16.0,
                    help="Peak full width at half maximum in cm⁻¹ (default: 16).")
    ir.add_argument("--lineshape", choices=["lorentzian", "gaussian"], default="lorentzian",
                    help="Broadening profile (default: lorentzian).")
    ir.add_argument("--points", type=int, default=4000, help="Number of grid points (default: 4000).")
    ir.add_argument("-o", "--output", default=None,
                    help="Write the spectrum to this file (.dat/.csv/.json/.html).")
    ir.add_argument("--format", choices=["dat", "csv", "json", "html"], default="dat",
                    help="Output format; html writes an interactive plot.")
    ir.set_defaults(func=_cmd_ir)

    cube = sub.add_parser(
        "cube",
        help="Interactive HTML viewer for a .cube field (orbital/density/MEP isosurfaces).",
    )
    cube.add_argument("input_file", metavar="CUBE", help="Gaussian/ORCA .cube file.")
    cube.add_argument("-o", "--output", default=None,
                      help="Output HTML file (default: <cube>_view.html).")
    cube.add_argument("--isovalue", type=float, default=None,
                      help="Isosurface value (default: 15%% of the field's max |value|).")
    cube.add_argument("--max-points", type=int, default=60000,
                      help="Max grid points after downsampling (default: 60000).")
    cube.add_argument("--field-type", default="other",
                      choices=["other", "density", "orbital", "mep"],
                      help="Field label used in the title/metadata.")
    cube.set_defaults(func=_cmd_cube)

    vmd = sub.add_parser(
        "vmd",
        help="Write a VMD/Tachyon .tcl script to render a .cube (publication-quality figures).",
    )
    vmd.add_argument("input_file", metavar="CUBE", help="Gaussian/ORCA .cube file.")
    vmd.add_argument("-o", "--output", default=None,
                     help="Output .tcl script (default: <cube>_vmd.tcl).")
    vmd.add_argument("--isovalue", type=float, default=None,
                     help="Isosurface value (default: 15%% of the field's max |value|).")
    vmd.add_argument("--resolution", default="1920x1440",
                     help="Render resolution WxH (default: 1920x1440).")
    vmd.add_argument("--material", default="Transparent",
                     help="VMD material for the isosurfaces (default: Transparent).")
    vmd.add_argument("--render-output", default="render.tga",
                     help="Image file the render_hires proc writes (default: render.tga).")
    vmd.add_argument("--tachyon", default=None,
                     help="External Tachyon executable; omit to use VMD's built-in renderer.")
    vmd.set_defaults(func=_cmd_vmd)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
