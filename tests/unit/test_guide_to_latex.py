"""Tests for tools/guide_to_latex.py — the TXT guide is the source of the PDF.

The point of these is that a change to the usage guide fails a test rather than
a LaTeX build: the guide is edited far more often than the converter.
"""
from __future__ import annotations

import collections
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE_ES = REPO / "cadmia" / "usage_guide.txt"
GUIDE_EN = REPO / "cadmia" / "usage_guide.en.txt"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "guide_to_latex", REPO / "tools" / "guide_to_latex.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["guide_to_latex"] = mod
    spec.loader.exec_module(mod)
    return mod


g2l = _load_module()

GUIDES = [(GUIDE_ES, "es"), (GUIDE_EN, "en")]


def _doc(path: Path, lang: str):
    return g2l.parse_guide(path.read_text(encoding="utf-8"), lang)


@pytest.mark.parametrize("path,lang", GUIDES)
def test_all_seventeen_sections_plus_appendix(path, lang):
    doc = _doc(path, lang)
    sections = [b for b in doc.blocks if b.kind == "section"]
    assert len(sections) == 18                      # 17 numbered + the appendix
    assert [b.label for b in sections[:17]] == [f"{i}." for i in range(1, 18)]


@pytest.mark.parametrize("path,lang", GUIDES)
def test_appendix_subsections_are_complete(path, lang):
    doc = _doc(path, lang)
    labels = [b.label for b in doc.blocks
              if b.kind == "subsection" and b.label.startswith("A.")]
    assert labels == [f"A.{i}" for i in range(1, 13)]


@pytest.mark.parametrize("path,lang", GUIDES)
def test_headings_use_the_index_capitalization(path, lang):
    """Body headings are ALL CAPS; the typeset ones must not be."""
    doc = _doc(path, lang)
    for b in doc.blocks:
        if b.kind in ("section", "subsection") and len(b.text) > 12:
            assert b.text != b.text.upper(), b.text


@pytest.mark.parametrize("path,lang", GUIDES)
def test_every_code_glyph_is_covered_by_the_literate_table(path, lang):
    """listings ignores inputenc: an unmapped glyph is a fatal pdfLaTeX error."""
    doc = _doc(path, lang)
    used = {c for b in doc.blocks if b.kind == "code"
            for line in b.lines for c in line if ord(c) > 127}
    missing = used - set(g2l.LISTINGS_LITERATE)
    assert not missing, f"add to LISTINGS_LITERATE: {sorted(missing)}"


@pytest.mark.parametrize("path,lang", GUIDES)
def test_prose_carries_no_unmapped_non_ascii(path, lang):
    """Everything outside Latin-1 must be spelled out as a LaTeX command."""
    doc = _doc(path, lang)
    rendered = "".join(
        g2l.render_inline(b.text) for b in doc.blocks
        if b.kind in ("para", "section", "subsection", "refs")
    )
    stray = {c for c in rendered if ord(c) > 0xFF}
    assert not stray, f"add to UNICODE_TEXT: {sorted(stray)}"


@pytest.mark.parametrize("path,lang", GUIDES)
def test_conversion_loses_no_prose(path, lang):
    """Every word of the body must survive into some block."""
    doc = _doc(path, lang)
    captured = []
    for b in doc.blocks:
        if b.kind in ("para", "section", "subsection"):
            captured.append(b.text)
        elif b.kind == "code":
            captured.extend(b.lines)
        elif b.kind == "refs":
            captured.append(b.text)
            captured.extend(b.lines)

    def words(s):
        return re.findall(r"[^\W\d_]{4,}", s.lower())

    got = collections.Counter(words(" ".join(captured)))
    body = path.read_text(encoding="utf-8").split(g2l.LANG[lang]["index"], 1)[1]
    body = body.split("=" * 68, 2)[-1]
    want = collections.Counter(words(body))
    missing = sum(n - got[w] for w, n in want.items() if n > got[w])
    # A couple of words legitimately differ: section 16's index entry and its
    # body heading are not worded identically, and the index wins.
    assert missing <= 3, f"{missing} words lost in conversion"


def test_no_replacement_ends_in_a_bare_letter():
    """\\textdegree followed by "C" would parse as the command \\textdegreeC.

    A replacement is safe when it ends in a delimiter — `}`, `$` or `{}` — so
    the rule is simply that none may end in a letter.
    """
    for table in (g2l.LISTINGS_LITERATE, g2l.UNICODE_TEXT):
        for char, tex in table.items():
            assert not tex[-1:].isalpha(), f"{char!r} -> {tex!r} can swallow the next letter"


def test_tex_escape_handles_metacharacters():
    out = g2l.tex_escape("100% & $5 _x #1 {a}")
    for raw in ("100\\%", "\\&", "\\$5", "\\_x", "\\#1", "\\{a\\}"):
        assert raw in out


def test_backticks_become_texttt():
    assert g2l.render_inline("run `cadmia doctor` now") == r"run \texttt{cadmia doctor} now"


def test_underscores_inside_backticks_are_escaped():
    assert g2l.render_inline("`e_ads_kcal`") == r"\texttt{e\_ads\_kcal}"


@pytest.mark.parametrize("path,lang", GUIDES)
def test_renders_a_complete_document(path, lang):
    tex = g2l.render_latex(_doc(path, lang), lang, path.name)
    assert tex.startswith("%% Generated by tools/guide_to_latex.py")
    assert "\\begin{document}" in tex and "\\end{document}" in tex
    assert "\\tableofcontents" in tex
    assert tex.count("\\begin{lstlisting}") == tex.count("\\end{lstlisting}")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
@pytest.mark.parametrize("path,lang", GUIDES)
def test_output_compiles_with_pdflatex(path, lang, tmp_path):
    tex = g2l.render_latex(_doc(path, lang), lang, path.name)
    src = tmp_path / f"guide_{lang}.tex"
    src.write_text(tex, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
        errors="replace",                      # TeX logs are not valid UTF-8
    )
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (tmp_path / f"guide_{lang}.pdf").exists()
