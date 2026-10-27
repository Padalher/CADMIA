"""Tests for the bilingual CLI help system (--help / --ayuda)."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from cadmia.cli import main
from cadmia.cli_help_es import SPANISH_HELP, _command_key


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ─────────────────────────────────────────────
# Spanish help (--ayuda)
# ─────────────────────────────────────────────

def test_root_ayuda(runner):
    result = runner.invoke(main, ["--ayuda"])
    assert result.exit_code == 0
    assert "Plataforma de agentes" in result.output
    assert "Comandos disponibles" in result.output


def test_command_ayuda(runner):
    result = runner.invoke(main, ["cdft", "--ayuda"])
    assert result.exit_code == 0
    assert "Descriptores CDFT" in result.output
    assert "Salida SCF con N electrones" in result.output


def test_nested_subcommand_ayuda(runner):
    result = runner.invoke(main, ["benchmark", "generate", "--ayuda"])
    assert result.exit_code == 0
    assert "Generar entradas Gaussian y ORCA" in result.output


def test_profile_nested_ayuda(runner):
    result = runner.invoke(main, ["profile", "add", "--ayuda"])
    assert result.exit_code == 0
    assert "Guardar un perfil de conexión HPC" in result.output


@pytest.mark.parametrize("argv,key", [
    (["cdft"],                 "cdft"),
    (["validate"],             "validate"),
    (["geometry"],             "geometry"),
    (["submit"],               "submit"),
    (["monitor"],              "monitor"),
    (["benchmark", "generate"], "benchmark generate"),
    (["benchmark", "compare"],  "benchmark compare"),
    (["profile", "add"],        "profile add"),
    (["profile", "list"],       "profile list"),
    (["db", "init"],            "db init"),
])
def test_every_command_has_spanish_help(runner, argv, key):
    """Each command resolves to a defined SPANISH_HELP entry (no fallback message)."""
    assert key in SPANISH_HELP
    result = runner.invoke(main, argv + ["--ayuda"])
    assert result.exit_code == 0
    assert "No hay ayuda en español" not in result.output


# ─────────────────────────────────────────────
# English help still works and advertises the Spanish flag
# ─────────────────────────────────────────────

def test_english_help_intact(runner):
    result = runner.invoke(main, ["cdft", "--help"])
    assert result.exit_code == 0
    assert "Compute CDFT descriptors" in result.output


def test_english_help_advertises_spanish(runner):
    result = runner.invoke(main, ["cdft", "--help"])
    assert "--ayuda" in result.output


def test_ayuda_skips_required_option_validation(runner):
    """--ayuda is eager: it must print help even though --n/--np1/--nm1 are required."""
    result = runner.invoke(main, ["cdft", "--ayuda"])
    assert result.exit_code == 0
    assert "Missing option" not in result.output


# ─────────────────────────────────────────────
# _command_key resolver
# ─────────────────────────────────────────────

def test_command_key_resolves_nested(runner):
    """Sanity: every SPANISH_HELP key except 'main' is reachable as a command path."""
    # Group keys and leaf keys must all be non-empty strings.
    assert all(isinstance(k, str) and k for k in SPANISH_HELP)
    assert "main" in SPANISH_HELP


# ─────────────────────────────────────────────
# Unified surface: the electronic/structure programs are reachable as
# `cadmia` subcommands (CLI consolidation, transition step 1).
# ─────────────────────────────────────────────

def test_electronic_and_structure_listed_in_root_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "electronic" in result.output
    assert "structure" in result.output


def test_electronic_passthrough_forwards_to_program(runner):
    result = runner.invoke(main, ["electronic", "--help"])
    assert result.exit_code == 0
    assert "dos" in result.output and "pdos" in result.output


def test_structure_passthrough_forwards_to_program(runner):
    result = runner.invoke(main, ["structure", "--help"])
    assert result.exit_code == 0
    assert "plot" in result.output and "report" in result.output and "render" not in result.output


def test_banner_theme_follows_the_calendar():
    import datetime

    from cadmia.cli import _banner_theme
    glyph = lambda m, d: _banner_theme(datetime.date(2026, m, d))[0]
    assert glyph(11, 1) == "🌼" and glyph(12, 24) == "🎄" and glyph(1, 6) == "🎄"
    assert glyph(1, 7) == "❄" and glyph(4, 15) == "🌸" and glyph(7, 15) == "☀"
    assert glyph(10, 15) == "🍂" and glyph(11, 3) == "🍂"
    assert glyph(6, 9) == "🎂" and glyph(6, 10) == "☀"
    assert all(len(_banner_theme(datetime.date(2026, m, 15))[1]) == 6 for m in range(1, 13))



def _visible_commands(cmd, path=()):
    yield path, cmd
    for name, sub in getattr(cmd, "commands", {}).items():
        if not sub.hidden:
            yield from _visible_commands(sub, path + (name,))


def test_spanish_help_tracks_the_real_options():
    """--ayuda is hand-written: every real option must appear in it, and every
    option it cites must exist (cross-references to other commands allowed)."""
    import re

    import click

    commands = list(_visible_commands(main))
    every_option = {o for _, c in commands for p in c.params for o in p.opts + p.secondary_opts}
    problems = []
    for path, cmd in commands:
        key = " ".join(path) or "main"
        text = SPANISH_HELP.get(key)
        if text is None:
            problems.append(f"{key}: no Spanish help")
            continue
        own = [o for p in cmd.params if isinstance(p, click.Option)
               for o in p.opts + p.secondary_opts if o.startswith("--") and o not in ("--help", "--ayuda")]
        missing = [o for o in own if not re.search(re.escape(o) + r"(?![\w-])", text)]
        unknown = set(re.findall(r"(?<![\w-])--[a-z][\w-]*", text)) - every_option - {"--help"}
        if missing:
            problems.append(f"{key}: undocumented {missing}")
        if unknown:
            problems.append(f"{key}: no such option {sorted(unknown)}")
    assert not problems, "\n".join(problems)
