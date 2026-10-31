#!/usr/bin/env python3
"""Render the CADMIA usage guide (plain text) as a LaTeX document.

The TXT guide is the single source of truth: `cadmia/usage_guide.txt` (ES) and
`cadmia/usage_guide.en.txt` (EN) ship inside the package and back `--ayuda`.
This script derives the typeset version from them so the PDF can never drift
from the guide the program actually installs — the previous hand-written
`usage_guide.tex` had fallen a month behind.

Section and subsection numbers come from the TXT, not from LaTeX counters: the
guide numbers its own sections (and the appendix uses A.1…A.12), so every
heading is unnumbered with a manual label. That keeps the PDF's numbering
identical to the text guide by construction.

    python3 tools/guide_to_latex.py cadmia/usage_guide.txt    -o docs/latex/usage_guide_es.tex
    python3 tools/guide_to_latex.py cadmia/usage_guide.en.txt -o docs/latex/usage_guide_en.tex

`docs/latex/` is gitignored: the .tex and the .pdf are derived artifacts, only the
.txt guides are versioned.

The output is pdfLaTeX-compatible (the preamble maps the Unicode symbols the
guide uses onto math/text commands).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

RULE = re.compile(r"^(=|-|\.){5,}\s*$")
EQ_RULE = re.compile(r"^={5,}\s*$")
DASH_RULE = re.compile(r"^-{5,}\s*$")
DOT_RULE = re.compile(r"^\.{5,}\s*$")

SECTION_NUM = re.compile(r"^(\d+)\.\s+(.*)$")
SUBSECTION = re.compile(r"^(\d+\.\d+)\s+(.*)$")
APPENDIX_SUB = re.compile(r"^(A\.\d+)\s+(.*)$")
INDEX_ENTRY = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
INDEX_APPENDIX_ENTRY = re.compile(r"^\s*(A\.\d+)\s+(.+?)\s*$")
REFERENCE_ITEM = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
# A line of its own in capitals ("QUÉ HACE", "ALGORITMO, PASO A PASO") heads a block of prose.
MINOR_HEAD = re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9][A-ZÁÉÍÓÚÜÑ0-9 ,;:()/'+&.¿?¡!—–-]{2,70}$")
LIST_ITEM = re.compile(r"^(\s{0,6})([-•*]|\d{1,2}[.)])\s+(\S.*)$")
# Three spaces inside a line align columns: such a line is a table and keeps its layout.
ALIGNED = re.compile(r"\S {3,}\S")

# Words the guide writes in caps that must stay in caps when a heading is
# converted from ALL CAPS back to sentence case.
ACRONYMS = {
    "CADMIA", "CDFT", "DFT", "SCF", "DOS", "PDOS", "HPC", "LLM", "ECN", "CN",
    "HOMO", "LUMO", "MEP", "BDE", "BSSE", "HSAB", "SLURM", "SSH", "API", "CLI",
    "JSON", "CSV", "HTML", "XYZ", "PDF", "SQL", "DB", "IP", "EA", "ZPE", "NBO",
    "NPA", "AIM", "TD-DFT", "ORCA", "SP", "MO", "RMSD",
}

# Non-ASCII characters that appear inside code blocks. `listings` does not read
# the document's inputenc, so every one of them must be mapped here or pdfLaTeX
# aborts with "Invalid UTF-8 byte sequence". test_guide_to_latex.py asserts the
# guides introduce no character outside this table, so a new symbol fails a test
# instead of a build.
LISTINGS_LITERATE = {
    "á": r"\'a", "é": r"\'e", "í": r"\'i", "ó": r"\'o", "ú": r"\'u",
    "Á": r"\'A", "É": r"\'E", "Í": r"\'I", "Ó": r"\'O", "Ú": r"\'U",
    "ñ": r"\~n", "Ñ": r"\~N", "ü": r'\"u', "Ü": r'\"U',
    "¿": r"?`", "¡": r"!`", "º": r"\textordmasculine", "ª": r"\textordfeminine",
    "°": r"\textdegree", "·": r"$\cdot$", "Å": r"\AA",
    "²": r"$^2$", "³": r"$^3$", "½": r"$\frac{1}{2}$",
    "Δ": r"$\Delta$", "Σ": r"$\Sigma$", "Ω": r"$\Omega$", "Π": r"$\Pi$",
    "α": r"$\alpha$", "β": r"$\beta$", "η": r"$\eta$", "μ": r"$\mu$",
    "ν": r"$\nu$", "ρ": r"$\rho$", "σ": r"$\sigma$", "χ": r"$\chi$",
    "ω": r"$\omega$", "π": r"$\pi$", "θ": r"$\theta$", "λ": r"$\lambda$",
    "–": "--", "—": "---", "…": r"\ldots",
    "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
    "−": "-", "±": r"$\pm$", "×": r"$\times$",
    "∠": r"$\angle$", "≠": r"$\neq$", "≡": r"$\equiv$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≈": r"$\approx$",
    "∫": r"$\int$", "∞": r"$\infty$", "√": r"$\surd$",
    "¹": r"$^1$", "⁻": r"$^-$", "⟨": r"$\langle$", "⟩": r"$\rangle$",
}

# Same idea for prose, where inputenc does handle UTF-8 but pdfLaTeX still needs
# a definition for symbols outside the Latin-1 font encoding.
UNICODE_TEXT = {
    "α": r"\ensuremath{\alpha}", "β": r"\ensuremath{\beta}",
    "η": r"\ensuremath{\eta}", "μ": r"\ensuremath{\mu}",
    "ν": r"\ensuremath{\nu}", "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}", "χ": r"\ensuremath{\chi}",
    "ω": r"\ensuremath{\omega}", "π": r"\ensuremath{\pi}",
    "θ": r"\ensuremath{\theta}", "λ": r"\ensuremath{\lambda}",
    "Δ": r"\ensuremath{\Delta}", "Σ": r"\ensuremath{\Sigma}",
    "Ω": r"\ensuremath{\Omega}", "Γ": r"\ensuremath{\Gamma}", "Π": r"\ensuremath{\Pi}",
    "→": r"\ensuremath{\rightarrow}", "←": r"\ensuremath{\leftarrow}",
    "↔": r"\ensuremath{\leftrightarrow}", "⇒": r"\ensuremath{\Rightarrow}",
    "×": r"\ensuremath{\times}", "±": r"\ensuremath{\pm}",
    "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}",
    "≠": r"\ensuremath{\neq}", "≡": r"\ensuremath{\equiv}",
    "≈": r"\ensuremath{\approx}", "∠": r"\ensuremath{\angle}",
    "∫": r"\ensuremath{\int}", "∞": r"\ensuremath{\infty}",
    "∂": r"\ensuremath{\partial}", "√": r"\ensuremath{\surd}",
    "∑": r"\ensuremath{\sum}", "−": "-",
    "⟨": r"\ensuremath{\langle}", "⟩": r"\ensuremath{\rangle}",
    "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}", "½": r"\ensuremath{\frac{1}{2}}",
    "⁺": r"\textsuperscript{+}", "⁻": r"\textsuperscript{-}",
    "⁰": r"\textsuperscript{0}", "¹": r"\textsuperscript{1}", "⁸": r"\textsuperscript{8}",
    "₁": r"\textsubscript{1}", "₂": r"\textsubscript{2}",
    "₃": r"\textsubscript{3}", "₄": r"\textsubscript{4}",
    "₅": r"\textsubscript{5}", "₆": r"\textsubscript{6}",
    "°": r"\textdegree", "·": r"\textperiodcentered",
    "Å": r"\AA{}", "…": r"\ldots{}", "•": r"\textbullet",
    "–": "--", "—": "---",
    "“": "``", "”": "''", "‘": "`", "’": "'",
    "º": r"\textordmasculine", "ª": r"\textordfeminine",
    " ": "~",
}

def _brace(tex: str) -> str:
    """Terminate a control word with {} so it cannot swallow the next letter.

    Without it `\\textperiodcentered` immediately followed by "TiO" is read as
    the (undefined) command \\textperiodcenteredTiO.
    """
    return tex + "{}" if tex and tex[-1].isalpha() and "\\" in tex else tex


LISTINGS_LITERATE = {c: _brace(t) for c, t in LISTINGS_LITERATE.items()}
UNICODE_TEXT = {c: _brace(t) for c, t in UNICODE_TEXT.items()}

LANG = {
    "es": {
        "babel": "spanish,es-noshorthands",
        "index": "ÍNDICE",
        "guide": "GUÍA DE USO",
        "appendix": "APÉNDICE",
        "references": "REFERENCIAS",
        "toc": "Índice",
    },
    "en": {
        "babel": "english",
        "index": "TABLE OF CONTENTS",
        "guide": "USER GUIDE",
        "appendix": "APPENDIX",
        "references": "REFERENCES",
        "toc": "Contents",
    },
}


@dataclass
class Block:
    kind: str                      # cover | section | subsection | minor | para | code | refs | list
    text: str = ""
    label: str = ""
    lines: List[str] = field(default_factory=list)
    items: list = field(default_factory=list)   # list: (marker, text, code lines) per item


@dataclass
class Document:
    title: str = ""
    cover: List[str] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    index: dict = field(default_factory=dict)


def sentence_case(title: str) -> str:
    """Turn an ALL-CAPS heading back into sentence case, preserving acronyms.

    Used only as a fallback: headings normally take their capitalization from
    the guide's own index, which is already written in prose case.
    """
    if title != title.upper():
        return title
    words = title.split(" ")
    out = []
    for i, w in enumerate(words):
        bare = w.strip("():,.—-")
        if bare in ACRONYMS or (len(bare) > 1 and not bare.isalpha()):
            out.append(w)
        elif i == 0:
            out.append(w.capitalize())
        else:
            out.append(w.lower())
    return " ".join(out)


def parse_index(lines: List[str], start: int, end: int, appendix_word: str = "") -> dict:
    """Map section/appendix numbers to their prose-case titles from the index.

    The body writes headings in ALL CAPS; the index writes the same titles in
    prose case, so it is the better source for a typeset heading. The appendix
    banner is stored under the key "APPENDIX".
    """
    index = {}
    for raw in lines[start:end]:
        if appendix_word and raw.strip().startswith(appendix_word):
            index["APPENDIX"] = raw.strip()
            continue
        m = INDEX_ENTRY.match(raw)
        if m and raw.startswith((" ", "\t")) or (m and raw[0].isdigit()):
            index[m.group(1)] = m.group(2).strip()
            continue
        m = INDEX_APPENDIX_ENTRY.match(raw)
        if m:
            index[m.group(1)] = m.group(2).strip()
    return index


def _is_heading(lines: List[str], i: int) -> bool:
    return (
        EQ_RULE.match(lines[i] or "")
        and i + 2 < len(lines)
        and bool((lines[i + 1] or "").strip())
        and bool(EQ_RULE.match(lines[i + 2] or ""))
    )


def parse_guide(text: str, lang: str) -> Document:
    words = LANG[lang]
    lines = text.splitlines()
    doc = Document()

    # Cover: everything before the first "==== / title / ====" heading.
    first = next((i for i in range(len(lines)) if _is_heading(lines, i)), 0)
    doc.cover = [ln for ln in lines[:first] if not EQ_RULE.match(ln)]

    # Index block: capture titles, then skip it (LaTeX builds its own toc).
    for i in range(len(lines)):
        if _is_heading(lines, i) and lines[i + 1].strip() == words["index"]:
            stop = next((j for j in range(i + 3, len(lines)) if _is_heading(lines, j)),
                        len(lines))
            doc.index = parse_index(lines, i + 3, stop, words["appendix"])
            break

    i = first
    para: List[str] = []
    code: List[str] = []
    lst: list = []               # open list: [marker, text, code lines, indent]
    in_appendix = False

    def flush_para():
        if para:
            doc.blocks.append(Block("para", text=" ".join(x.strip() for x in para)))
            para.clear()

    def flush_code():
        while code and not code[-1].strip():
            code.pop()
        if code:
            doc.blocks.append(Block("code", lines=list(code)))
            code.clear()

    def flush_list():
        if lst:
            doc.blocks.append(Block("list", items=[(m, t, c) for m, t, c, _ in lst]))
            lst.clear()

    def flush():
        flush_para()
        flush_code()
        flush_list()

    while i < len(lines):
        line = lines[i]

        # ---- "==== / TITLE / ====" heading -----------------------------------
        if _is_heading(lines, i):
            flush()
            title = lines[i + 1].strip()
            i += 3
            if title == words["index"]:
                while i < len(lines) and not _is_heading(lines, i):
                    i += 1
                continue
            if title == words["guide"]:
                doc.title = title
                continue
            # The appendix banner is itself a ==== heading; from here on, A.N
            # headings are underlined with dashes rather than dots.
            if title.startswith(words["appendix"]):
                doc.blocks.append(
                    Block("section", text=doc.index.get("APPENDIX") or sentence_case(title))
                )
                in_appendix = True
                continue
            m = SECTION_NUM.match(title)
            if m:
                num, rest = m.group(1), m.group(2)
                pretty = doc.index.get(num) or sentence_case(rest)
                doc.blocks.append(Block("section", text=pretty, label=f"{num}."))
            else:
                doc.blocks.append(Block("section", text=sentence_case(title)))
            continue

        stripped = line.strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        # ---- appendix marker -------------------------------------------------
        if stripped.startswith(words["appendix"]) and not line.startswith(" "):
            flush()
            title = doc.index.get("APPENDIX") or sentence_case(stripped)
            doc.blocks.append(Block("section", text=title))
            in_appendix = True
            i += 1
            continue

        # ---- underlined subsections -----------------------------------------
        if stripped and (DOT_RULE.match(nxt) or (in_appendix and DASH_RULE.match(nxt))):
            m = SUBSECTION.match(stripped) or APPENDIX_SUB.match(stripped)
            if m:
                num, rest = m.group(1), m.group(2)
                pretty = doc.index.get(num) or sentence_case(rest)
                flush()
                doc.blocks.append(Block("subsection", text=pretty, label=num))
                i += 2
                continue

        # ---- references list -------------------------------------------------
        if stripped.startswith(words["references"]) and not line.startswith(" "):
            flush()
            items: List[str] = []
            i += 1
            while i < len(lines):
                cur = lines[i]
                if not cur.strip():
                    if items and i + 1 < len(lines) and not lines[i + 1].startswith(" "):
                        break
                    i += 1
                    continue
                if _is_heading(lines, i) or REFERENCE_ITEM.match(cur) is None and not cur.startswith(" "):
                    break
                m = REFERENCE_ITEM.match(cur)
                if m:
                    items.append(f"[{m.group(1)}] {m.group(2).strip()}")
                elif items:
                    items[-1] += " " + cur.strip()
                i += 1
            doc.blocks.append(Block("refs", text=stripped, lines=items))
            continue

        # ---- horizontal rule -------------------------------------------------
        if DASH_RULE.match(line):
            flush()
            i += 1
            continue

        # ---- lists: "- item" / "1. step", with their indented continuation ---
        item = LIST_ITEM.match(line)
        indent = len(line) - len(line.lstrip(" "))
        if item and not code:
            flush_para()
            flush_code()
            lst.append([item.group(2), item.group(3).strip(), [], len(item.group(1))])
            i += 1
            continue
        if lst and stripped and indent > lst[-1][3]:
            text_col = lst[-1][3] + len(lst[-1][0]) + 1
            if indent >= text_col + 3 or ALIGNED.search(stripped):
                lst[-1][2].append(line[text_col:] if len(line) > text_col else stripped)
            elif lst[-1][2]:
                lst[-1][2].append(line[text_col:])
            else:
                lst[-1][1] += " " + stripped
            i += 1
            continue
        if lst and not stripped:
            ahead = next((lines[j] for j in range(i + 1, len(lines)) if lines[j].strip()), "")
            nxt_item = LIST_ITEM.match(ahead)
            ahead_indent = len(ahead) - len(ahead.lstrip(" "))
            if not ((nxt_item and len(nxt_item.group(1)) == lst[-1][3])
                    or ahead_indent > lst[-1][3]):
                flush_list()
            i += 1
            continue
        if lst:
            flush_list()

        # ---- a heading line of its own, in capitals -----------------------------
        if (stripped and not line.startswith(" ") and MINOR_HEAD.match(stripped)
                and sum(c.isalpha() for c in stripped) >= 4 and not nxt.startswith("    ")
                or stripped and not line.startswith(" ") and MINOR_HEAD.match(stripped)
                and sum(c.isalpha() for c in stripped) >= 4 and not para and not code):
            flush()
            doc.blocks.append(Block("minor", text=stripped))
            i += 1
            continue

        # ---- code (indented 4+, or aligned in columns) vs prose -----------------
        if stripped and (line.startswith("    ") or (ALIGNED.search(stripped) and not para)):
            flush_para()
            code.append(line[4:] if line.startswith("    ") else line)
            i += 1
            continue

        if not stripped:
            if code:
                code.append("")
            else:
                flush_para()
            i += 1
            continue

        flush_code()
        para.append(line)
        i += 1

    flush()
    return doc


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    """Escape LaTeX metacharacters and spell out non-Latin-1 symbols.

    Symbols are substituted here rather than declared with
    \\DeclareUnicodeCharacter so the emitted .tex depends on nothing but a stock
    pdfLaTeX: what the file contains is what gets typeset.
    """
    out = []
    s = s.replace("--", "-\x00-")          # a CLI option: two hyphens, not an en dash
    for c in s:
        if c == "\x00":
            out.append("{}")
        elif c in _ESCAPES:
            out.append(_ESCAPES[c])
        elif c in UNICODE_TEXT:
            out.append(UNICODE_TEXT[c])
        else:
            out.append(c)
    return "".join(out)


def render_inline(s: str) -> str:
    """Escape prose, turning `backticked` spans into \\texttt{}."""
    out = []
    for i, part in enumerate(s.split("`")):
        if i % 2:
            out.append(r"\texttt{" + tex_escape(part) + "}")
        else:
            out.append(tex_escape(part))
    text = "".join(out)
    # Straight double quotes render as two right-quotes in LaTeX; pair them up.
    def _quotes(m):
        return "``" + m.group(1) + "''"
    return re.sub(r'"([^"]*)"', _quotes, text)


PREAMBLE = r"""%% Generated by tools/guide_to_latex.py from {source} — do not edit by hand.
%% Re-run the script after editing the text guide.
\documentclass[11pt,a4paper]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[{babel}]{{babel}}
\usepackage{{lmodern}}
\usepackage{{geometry}}
\usepackage{{listings}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\usepackage{{parskip}}
\usepackage{{textcomp}}
\usepackage[hidelinks]{{hyperref}}

\geometry{{margin=2.5cm}}

\definecolor{{codebg}}{{gray}}{{0.96}}
\definecolor{{codeframe}}{{gray}}{{0.55}}
\lstset{{
  basicstyle=\ttfamily\small,
  backgroundcolor=\color{{codebg}},
  frame=single,
  rulecolor=\color{{codeframe}},
  breaklines=true,
  breakatwhitespace=false,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  extendedchars=true,
  literate=
{literate}
}}

\setlist[itemize]{{leftmargin=1.4em,itemsep=2pt,topsep=3pt}}
\hypersetup{{pdftitle={{CADMIA}},pdfauthor={{Pablo Daniel Alvarez Hernandez}}}}

\begin{{document}}
"""


def literate_block(chars=None) -> str:
    """Render the lstlisting `literate` table, four entries per line."""
    items = [(c, tex) for c, tex in LISTINGS_LITERATE.items()
             if chars is None or c in chars]
    out, row = [], []
    for c, tex in items:
        row.append("{%s}{{%s}}1" % (c, tex))
        if len(row) == 4:
            out.append("    " + " ".join(row))
            row = []
    if row:
        out.append("    " + " ".join(row))
    return "\n".join(out)


def render_latex(doc: Document, lang: str, source: str) -> str:
    words = LANG[lang]
    out = [PREAMBLE.format(babel=words["babel"], source=source,
                           literate=literate_block())]

    cover = "\n".join(doc.cover).strip("\n")
    out.append("\\begin{center}\n\\begin{footnotesize}\n\\begin{verbatim}\n"
               + cover + "\n\\end{verbatim}\n\\end{footnotesize}\n\\end{center}\n")
    out.append("\\vspace{1em}\n\\tableofcontents\n\\newpage\n")

    for b in doc.blocks:
        if b.kind == "section":
            label = f"{b.label} {b.text}".strip()
            out.append(f"\n\\section*{{{render_inline(label)}}}\n"
                       f"\\addcontentsline{{toc}}{{section}}{{{render_inline(label)}}}\n")
        elif b.kind == "subsection":
            label = f"{b.label} {b.text}".strip()
            out.append(f"\n\\subsection*{{{render_inline(label)}}}\n"
                       f"\\addcontentsline{{toc}}{{subsection}}{{{render_inline(label)}}}\n")
        elif b.kind == "minor":
            out.append(f"\n\\subsubsection*{{{render_inline(sentence_case(b.text))}}}\n")
        elif b.kind == "para":
            # a blank line ends the paragraph: with a single newline LaTeX runs them all together
            out.append(render_inline(b.text) + "\n\n")
        elif b.kind == "list":
            ordered = b.items and b.items[0][0][0].isdigit()
            out.append("\\begin{itemize}[leftmargin=2em]\n")
            for marker, text, code_lines in b.items:
                label = f"[{marker}]" if ordered else ""
                out.append(f"  \\item{label} " + render_inline(text) + "\n")
                if code_lines:
                    body = "\n".join(code_lines).rstrip()
                    out.append("\\begin{lstlisting}\n" + body + "\n\\end{lstlisting}\n")
            out.append("\\end{itemize}\n")
        elif b.kind == "code":
            body = "\n".join(b.lines)
            out.append("\\begin{lstlisting}\n" + body + "\n\\end{lstlisting}\n")
        elif b.kind == "refs":
            out.append(f"\n\\paragraph{{{render_inline(b.text)}}}\\mbox{{}}\n")
            out.append("\\begin{itemize}[label={}]\n")
            for item in b.lines:
                out.append("  \\item " + render_inline(item) + "\n")
            out.append("\\end{itemize}\n")

    out.append("\n\\end{document}\n")
    return "".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path, help="usage_guide.txt or usage_guide.en.txt")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output .tex")
    ap.add_argument("--lang", choices=sorted(LANG), help="guide language (autodetected)")
    args = ap.parse_args(argv)

    text = args.source.read_text(encoding="utf-8")
    lang = args.lang or ("en" if LANG["en"]["guide"] in text else "es")
    doc = parse_guide(text, lang)
    args.output.write_text(render_latex(doc, lang, args.source.name), encoding="utf-8")
    n_sec = sum(1 for b in doc.blocks if b.kind == "section")
    n_sub = sum(1 for b in doc.blocks if b.kind == "subsection")
    print(f"{args.output}: {lang}, {n_sec} sections, {n_sub} subsections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
