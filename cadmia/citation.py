"""How to cite CADMIA: the notice printed when a report or a full pipeline finishes."""
import random
import textwrap

from cadmia import __version__

# Zenodo concept DOI: it always resolves to the latest version. Add the article DOI when it exists.
REFERENCES = (
    f"P. D. Alvarez Hernandez, CADMIA (version {__version__}), Zenodo, "
    "https://doi.org/10.5281/zenodo.23608477",
)

# One is printed under the notice, as Gaussian does at the end of a job. Every entry carries its
# source; keep it that way when adding one.
QUOTES = (
    ("The underlying physical laws necessary for the mathematical theory of a large part of "
     "physics and the whole of chemistry are thus completely known, and the difficulty is only "
     "that the exact application of these laws leads to equations much too complicated to be "
     "soluble.", "P. A. M. Dirac, Proc. R. Soc. Lond. A 123, 714 (1929)"),
    ("The purpose of computing is insight, not numbers.",
     "R. W. Hamming, Numerical Methods for Scientists and Engineers (1962)"),
    ("The first principle is that you must not fool yourself — and you are the easiest person "
     "to fool.", "R. P. Feynman, \"Cargo Cult Science\", Caltech commencement address (1974)"),
    ("All models are wrong, but some are useful.", "G. E. P. Box (1979)"),
    ("God made the bulk; the surface was invented by the devil.", "attributed to W. Pauli"),
    ("Electronegativity is the negative of the chemical potential of density functional theory.",
     "R. G. Parr, R. A. Donnelly, M. Levy, W. E. Palke, J. Chem. Phys. 68, 3801 (1978)"),
    ("Absolute hardness was defined from the curvature of E(N): η = ½(∂²E/∂N²)_v.",
     "R. G. Parr, R. G. Pearson, J. Am. Chem. Soc. 105, 7512 (1983)"),
    ("The Fukui function f(r) = (dρ/dN)_v was introduced in 1984.",
     "R. G. Parr, W. Yang, J. Am. Chem. Soc. 106, 4049 (1984)"),
    ("The electrophilicity index ω = μ²/2η was proposed in 1999.",
     "R. G. Parr, L. v. Szentpály, S. Liu, J. Am. Chem. Soc. 121, 1922 (1999)"),
    ("The dual descriptor Δf = f+ − f− tells electrophilic from nucleophilic sites in one map.",
     "C. Morell, A. Grand, A. Toro-Labbé, J. Phys. Chem. A 109, 205 (2005)"),
    ("Kenichi Fukui and Roald Hoffmann shared the 1981 Nobel Prize in Chemistry for their "
     "theories on the course of chemical reactions.", "nobelprize.org"),
    ("Walter Kohn (density functional theory) and John Pople (computational methods) shared the "
     "1998 Nobel Prize in Chemistry.", "nobelprize.org"),
    ("Irving Langmuir received the 1932 Nobel Prize in Chemistry for his discoveries and "
     "investigations in surface chemistry.", "nobelprize.org"),
    ("Gerhard Ertl received the 2007 Nobel Prize in Chemistry for his studies of chemical "
     "processes on solid surfaces.", "nobelprize.org"),
)


def cite_block() -> str:
    rule = "-" * 71
    lines = [rule, f" Thank you for using CADMIA {__version__}",
             " If you publish results obtained with CADMIA, please cite:"]
    lines += [f" {i}. {ref}" for i, ref in enumerate(REFERENCES, 1)]
    return "\n".join(lines + [rule])


def quote_block(quote: "tuple[str, str] | None" = None) -> str:
    text, source = quote or random.choice(QUOTES)
    lines = textwrap.wrap(text, 70, initial_indent=" ", subsequent_indent=" ")
    return "\n".join(lines + [f"     — {source}"])


def citation_notice(quote: "tuple[str, str] | None" = None) -> str:
    return cite_block() + "\n" + quote_block(quote)
