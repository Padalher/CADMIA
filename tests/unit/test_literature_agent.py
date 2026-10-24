"""Tests for cadmia.agents.literature_agent (no network — _http_get_json is monkeypatched)."""
from __future__ import annotations

import json

import pytest

from cadmia.agents import literature_agent as lit
from cadmia.agents.literature_agent import (
    LiteratureError,
    LiteratureResult,
    Reference,
    dedupe,
    rank,
    to_bibtex,
    to_markdown,
)

_CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "title": ["Conceptual Density Functional Theory"],
                "author": [{"given": "P.", "family": "Geerlings"},
                           {"given": "F.", "family": "De Proft"}],
                "issued": {"date-parts": [[2003]]},
                "container-title": ["Chemical Reviews"],
                "DOI": "10.1021/cr990029p",
                "URL": "https://doi.org/10.1021/cr990029p",
                "is-referenced-by-count": 5000,
                "type": "journal-article",
            },
            {
                "title": ["Multiwfn: A Multifunctional Wavefunction Analyzer"],
                "author": [{"given": "T.", "family": "Lu"}, {"given": "F.", "family": "Chen"}],
                "issued": {"date-parts": [[2012]]},
                "container-title": ["Journal of Computational Chemistry"],
                "DOI": "10.1002/jcc.22885",
                "URL": "https://doi.org/10.1002/jcc.22885",
                "is-referenced-by-count": 12000,
                "type": "journal-article",
            },
        ]
    }
}

_OPENALEX_PAYLOAD = {
    "results": [
        {
            "title": "Conceptual Density Functional Theory",
            "authorships": [{"author": {"display_name": "Paul Geerlings"}}],
            "publication_year": 2003,
            "primary_location": {"source": {"display_name": "Chemical Reviews"}},
            "doi": "https://doi.org/10.1021/cr990029p",   # same DOI → dedupe with Crossref
            "cited_by_count": 4800,
            "type": "article",
            # inverted index → "A review of reactivity"
            "abstract_inverted_index": {"A": [0], "review": [1], "of": [2], "reactivity": [3]},
        }
    ]
}

_SEMANTIC_PAYLOAD = {
    "data": [
        {
            "title": "Electrophilicity Index",
            "authors": [{"name": "R. G. Parr"}, {"name": "L. von Szentpály"}, {"name": "S. Liu"}],
            "year": 1999,
            "venue": "Journal of the American Chemical Society",
            "externalIds": {"DOI": "10.1021/ja983494x"},
            "url": "https://www.semanticscholar.org/paper/abc",
            "citationCount": 6000,
            "publicationTypes": ["JournalArticle"],
            "abstract": "The electrophilicity index measures the energy lowering on saturation.",
        }
    ]
}


@pytest.fixture
def fake_apis(monkeypatch):
    def _fake_get(url, params, *, timeout_s):
        if "crossref" in url and not url.rstrip("/").endswith("works"):
            # by-DOI endpoint: works/<doi>
            return {"message": _CROSSREF_PAYLOAD["message"]["items"][0]}
        if "crossref" in url:
            return _CROSSREF_PAYLOAD
        if "openalex" in url:
            return _OPENALEX_PAYLOAD
        if "semanticscholar" in url:
            return _SEMANTIC_PAYLOAD
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(lit, "_http_get_json", _fake_get)


def test_crossref_search_normalizes(fake_apis):
    refs = lit.search_crossref("conceptual dft", rows=5)
    assert len(refs) == 2
    r = refs[0]
    assert r.title == "Conceptual Density Functional Theory"
    assert r.year == 2003
    assert r.doi == "10.1021/cr990029p"        # normalized (lowercase, no https)
    assert r.citations == 5000
    assert r.authors == ["P. Geerlings", "F. De Proft"]


def test_run_ranks_by_citations(fake_apis):
    result = lit.run("conceptual dft", sources=["crossref"], limit=10, sort="citations")
    assert isinstance(result, LiteratureResult)
    assert [r.doi for r in result.references] == ["10.1002/jcc.22885", "10.1021/cr990029p"]


def test_run_dedupes_across_sources(fake_apis):
    result = lit.run("conceptual dft", sources=["crossref", "openalex"], limit=10)
    dois = [r.doi for r in result.references]
    assert dois.count("10.1021/cr990029p") == 1   # merged across the two APIs


def test_fetch_by_doi(fake_apis):
    result = lit.run("", dois=["10.1021/cr990029p"])
    assert len(result.references) == 1
    assert result.references[0].doi == "10.1021/cr990029p"


def test_from_year_filter_passed(monkeypatch):
    captured = {}

    def _fake_get(url, params, *, timeout_s):
        captured.update(params)
        return _CROSSREF_PAYLOAD

    monkeypatch.setattr(lit, "_http_get_json", _fake_get)
    lit.search_crossref("x", from_year=2024)
    assert captured.get("filter") == "from-pub-date:2024-01-01"


def test_network_error_becomes_warning(monkeypatch):
    def _boom(url, params, *, timeout_s):
        raise LiteratureError("Could not reach the API")

    monkeypatch.setattr(lit, "_http_get_json", _boom)
    result = lit.run("anything", sources=["crossref"])
    assert result.references == []
    assert any("crossref" in w for w in result.warnings)


def test_bibtex_render():
    refs = [Reference(title="Conceptual Density Functional Theory",
                      authors=["P. Geerlings", "F. De Proft"], year=2003,
                      container="Chemical Reviews", doi="10.1021/cr990029p",
                      type="journal-article")]
    bib = to_bibtex(refs)
    assert bib.startswith("@article{geerlings2003")
    assert "doi = {10.1021/cr990029p}" in bib
    assert "author = {P. Geerlings and F. De Proft}" in bib


def test_markdown_render_lists_doi():
    result = LiteratureResult(query="x", references=[
        Reference(title="T", authors=["A B", "C D", "E F", "G H"], year=2025,
                  container="J", doi="10.1/x", citations=3),
    ])
    md = to_markdown(result)
    assert "DOI: 10.1/x" in md
    assert "et al." in md            # >3 authors truncated
    assert "cited by 3" in md


def test_rank_by_year():
    refs = [Reference(title="a", year=2003), Reference(title="b", year=2025)]
    assert [r.year for r in rank(refs, by="year")] == [2025, 2003]


def test_dedupe_keeps_richer_record():
    a = Reference(title="Same Paper", doi="10.1/x", citations=None, authors=["A"])
    b = Reference(title="Same Paper", doi="10.1/x", citations=50, authors=["A", "B"])
    out = dedupe([a, b])
    assert len(out) == 1
    assert out[0].citations == 50


# ── Semantic Scholar + abstracts + narration ──

def test_semantic_scholar_normalizes(fake_apis):
    refs = lit.search_semantic_scholar("electrophilicity", rows=5)
    assert len(refs) == 1
    r = refs[0]
    assert r.title == "Electrophilicity Index"
    assert r.year == 1999
    assert r.doi == "10.1021/ja983494x"
    assert r.citations == 6000
    assert r.source == "semanticscholar"
    assert r.abstract.startswith("The electrophilicity index")


def test_run_with_semanticscholar_source(fake_apis):
    result = lit.run("electrophilicity", sources=["semanticscholar"], limit=5)
    assert [r.doi for r in result.references] == ["10.1021/ja983494x"]


def test_openalex_reconstructs_abstract(fake_apis):
    refs = lit.search_openalex("conceptual dft")
    assert refs[0].abstract == "A review of reactivity"


def test_crossref_strips_jats_abstract():
    ref = lit._ref_from_crossref({
        "title": ["X"],
        "abstract": "<jats:p>Hello <jats:italic>world</jats:italic>.</jats:p>",
    })
    assert ref.abstract == "Hello world ."


def test_dedupe_preserves_abstract():
    a = Reference(title="Same", doi="10.1/x", abstract="", citations=10)
    b = Reference(title="Same", doi="10.1/x", abstract="rich abstract", citations=1)
    out = dedupe([a, b])
    assert len(out) == 1
    assert out[0].abstract == "rich abstract"


def test_markdown_shows_abstract():
    result = LiteratureResult(query="x", references=[
        Reference(title="T", authors=["A B"], year=2025, doi="10.1/x",
                  abstract="A short abstract."),
    ])
    md = to_markdown(result)
    assert "> A short abstract." in md


def test_narrate_uses_provider_text_only(monkeypatch, fake_apis):
    class _FakeResp:
        text = "This list covers conceptual DFT reactivity descriptors."

    class _FakeProvider:
        def complete(self, user, system=""):
            return _FakeResp()

    monkeypatch.setattr(lit, "_narrate",
                        lambda refs, topic, language: _FakeProvider().complete("").text)
    result = lit.run("conceptual dft", sources=["crossref"], narrate=True)
    assert result.narrative.startswith("This list covers")
