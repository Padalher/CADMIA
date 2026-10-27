"""argparse handlers behind `cadmia electronic` (DOS, PDOS, IR, MEP)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cadmia.electronic.dos import dos_from_output, pdos_from_output, write_dos, write_pdos
from cadmia.io.figures import FIGURE_FORMATS, FIGURE_WIDTHS_IN, resolve_format

_FORMATS = ("dat", "csv", "json", "html", *FIGURE_FORMATS)


def _format(args) -> str:
    return resolve_format(getattr(args, "format", None), args.output, _FORMATS, "dat")


def _figure_width(args) -> str:
    return getattr(args, "width", None) or "single"


def _rounded(array) -> list:
    return np.round(array, 6).tolist()


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

    fmt = _format(args)
    if fmt in FIGURE_FORMATS:
        from cadmia.electronic.figures import write_ir_figure
        out = args.output or (Path(args.input_file).stem + f"_ir.{fmt}")
        try:
            write_ir_figure(spectrum, out, title=f"IR spectrum — {Path(args.input_file).name}",
                            width=_figure_width(args))
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 1
        print(f"IR spectrum figure written to: {out}")
        return 0
    if fmt == "html":
        from cadmia.electronic.plot import write_ir_html
        out = args.output or (Path(args.input_file).stem + "_ir.html")
        write_ir_html(spectrum, out, title=f"IR spectrum — {Path(args.input_file).name}")
        print(f"IR plot written to: {out}")
    elif args.output:
        write_ir_spectrum(spectrum, args.output, fmt=fmt)
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
        print("(use --output to write the full curve: .dat/.csv/.json/.html or a .png/.svg/.pdf figure)")
    return 0


def _cmd_dos(args) -> int:
    dos = dos_from_output(args.input_file, sigma=args.sigma, n_points=args.points)

    fmt = _format(args)
    if fmt in FIGURE_FORMATS:
        from cadmia.electronic.figures import write_dos_figure
        out = args.output or (Path(args.input_file).stem + f"_dos.{fmt}")
        try:
            write_dos_figure(dos, out, title=f"DOS — {Path(args.input_file).name}",
                             width=_figure_width(args))
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 1
        print(f"DOS figure written to: {out}")
        return 0
    if fmt == "html":
        from cadmia.electronic.plot import write_dos_html
        out = args.output or (Path(args.input_file).stem + "_dos.html")
        write_dos_html(dos, out, title=f"DOS — {Path(args.input_file).name}")
        print(f"Interactive DOS plot written to: {out}")
        return 0
    if fmt == "json":
        payload = {
            "energy_ev": np.round(dos.energy_ev, 4).tolist(),
            "dos_total": _rounded(dos.dos_total),
            "dos_alpha": _rounded(dos.dos_alpha),
            "dos_beta": _rounded(dos.dos_beta) if dos.dos_beta is not None else None,
            "spin_polarized": dos.is_spin_polarized,
            "homo_ev": dos.homo_ev,
            "lumo_ev": dos.lumo_ev,
            "gap_ev": dos.gap_ev,
        }
        text = json.dumps(payload)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"JSON written to: {args.output}")
        else:
            print(text)
    elif args.output:
        write_dos(dos, args.output, fmt=fmt)
        print(f"DOS written to: {args.output}")
    else:
        gap = f"{dos.gap_ev:.4f}" if dos.gap_ev is not None else "—"
        print(f"DOS: {len(dos.energy_ev)} points over "
              f"[{dos.energy_ev[0]:.2f}, {dos.energy_ev[-1]:.2f}] eV  |  "
              f"HOMO={dos.homo_ev:.4f}  LUMO={dos.lumo_ev:.4f}  gap={gap} eV  |  "
              f"{'spin-polarized' if dos.is_spin_polarized else 'closed-shell'}")
        print("(use --output to write the full curve: .dat/.csv/.json/.html or a .png/.svg/.pdf figure)")
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


def _cmd_pdos(args) -> int:
    pdos = pdos_from_output(args.input_file, sigma=args.sigma, n_points=args.points)

    fmt = _format(args)
    if fmt in FIGURE_FORMATS:
        from cadmia.electronic.figures import write_pdos_figure
        out = args.output or (Path(args.input_file).stem + f"_pdos.{fmt}")
        try:
            write_pdos_figure(pdos, out, title=f"PDOS — {Path(args.input_file).name}",
                              width=_figure_width(args))
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 1
        print(f"PDOS figure written to: {out}")
        return 0
    if fmt == "html":
        from cadmia.electronic.plot import write_pdos_html
        out = args.output or (Path(args.input_file).stem + "_pdos.html")
        write_pdos_html(pdos, out, title=f"PDOS — {Path(args.input_file).name}")
        print(f"Interactive PDOS plot written to: {out}")
        return 0
    if fmt == "json":
        payload = {
            "energy_ev": np.round(pdos.energy_ev, 4).tolist(),
            "series": {e: _rounded(s) for e, s in pdos.series.items()},
            "dos_total": _rounded(pdos.total),
            "spin_polarized": pdos.spin_polarized,
            "homo_ev": pdos.homo_ev,
            "lumo_ev": pdos.lumo_ev,
            "gap_ev": pdos.gap_ev,
        }
        text = json.dumps(payload)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"JSON written to: {args.output}")
        else:
            print(text)
    elif args.output:
        write_pdos(pdos, args.output, fmt=fmt)
        print(f"PDOS written to: {args.output}")
    else:
        gap = f"{pdos.gap_ev:.4f}" if pdos.gap_ev is not None else "—"
        print(f"PDOS: elements {', '.join(sorted(pdos.series))}  |  "
              f"{len(pdos.energy_ev)} points over "
              f"[{pdos.energy_ev[0]:.2f}, {pdos.energy_ev[-1]:.2f}] eV  |  "
              f"gap={gap} eV  |  "
              f"{'spin-polarized' if pdos.spin_polarized else 'closed-shell'}")
        print("(use --output to write the full curves: .dat/.csv/.json/.html or a .png/.svg/.pdf figure)")
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
                     help="Write the DOS curve to this file (.dat/.csv/.json/.html/.png/.svg/.pdf).")
    dos.add_argument("--format", choices=_FORMATS, default=None,
                     help="Output format; html writes an interactive plot, png/svg/pdf a static "
                          "figure (matplotlib). Default: the -o suffix, else dat.")
    dos.add_argument("--width", choices=list(FIGURE_WIDTHS_IN), default="single",
                     help="Static figure width: single = 3.4 in, double = 6.5 in (default: single).")
    dos.set_defaults(func=_cmd_dos)

    pdos = sub.add_parser(
        "pdos",
        help="Element-projected DOS from an ORCA output with Print[P_OrbPopMO_L] 1.",
    )
    pdos.add_argument("input_file", metavar="OUTPUT", help="ORCA output file with per-MO Loewdin populations.")
    pdos.add_argument("--sigma", type=float, default=0.1, help="Gaussian broadening width in eV (default: 0.1).")
    pdos.add_argument("--points", type=int, default=2000, help="Minimum number of energy-grid points; more are used for a step of sigma/4 (default: 2000).")
    pdos.add_argument("-o", "--output", default=None,
                      help="Write the PDOS curves to this file (.dat/.csv/.json/.html/.png/.svg/.pdf).")
    pdos.add_argument("--format", choices=_FORMATS, default=None,
                      help="Output format; html writes an interactive plot, png/svg/pdf a static "
                           "figure (matplotlib). Default: the -o suffix, else dat.")
    pdos.add_argument("--width", choices=list(FIGURE_WIDTHS_IN), default="single",
                      help="Static figure width: single = 3.4 in, double = 6.5 in (default: single).")
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
                    help="Write the spectrum to this file (.dat/.csv/.json/.html/.png/.svg/.pdf).")
    ir.add_argument("--format", choices=_FORMATS, default=None,
                    help="Output format; html writes an interactive plot, png/svg/pdf a static "
                         "figure (matplotlib). Default: the -o suffix, else dat.")
    ir.add_argument("--width", choices=list(FIGURE_WIDTHS_IN), default="single",
                    help="Static figure width: single = 3.4 in, double = 6.5 in (default: single).")
    ir.set_defaults(func=_cmd_ir)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
