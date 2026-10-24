"""Shared pytest fixtures for CADMIA tests."""
from pathlib import Path

import pytest

from cadmia.structure.core.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture(autouse=True)
def _never_touch_the_real_database(tmp_path, monkeypatch):
    """Point every default database write at a throwaway file.

    Since persistence is ON by default, so a command invoked in a test without an
    explicit --db-path would write to the user's own ~/.cadmia/cadmia.db — their actual
    campaign. Autouse on purpose: the protection must not depend on each test remembering
    to ask for it.

    `resolve_db_path` is also pinned, otherwise a test running from inside a tree that has
    its own .cadmia/cadmia.db would adopt that one instead.
    """
    from cadmia.db import session

    sandbox = tmp_path / "sandbox.cadmia.db"
    original = session.resolve_db_path

    def _sandboxed(explicit=None, start=None):
        # A test that passes `start` is testing the resolver itself, so it gets the real one.
        # The CLI never passes it, so every command still lands in the sandbox.
        if start is not None:
            return original(explicit, start)
        return Path(explicit) if explicit else sandbox

    monkeypatch.setattr(session, "DEFAULT_DB", sandbox)
    monkeypatch.setattr(session, "resolve_db_path", _sandboxed)
    return sandbox
