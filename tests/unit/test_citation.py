from cadmia import __version__
from cadmia.citation import citation_notice


def test_notice_names_the_version_and_the_repository():
    notice = citation_notice()
    assert f"CADMIA {__version__}" in notice
    assert "https://github.com/Padalher/CADMIA" in notice


def test_notice_ends_with_a_sourced_quote():
    from cadmia.citation import QUOTES
    notice = citation_notice(QUOTES[1])
    assert notice.splitlines()[-1] == f"     — {QUOTES[1][1]}"
    assert all(text and source for text, source in QUOTES)
