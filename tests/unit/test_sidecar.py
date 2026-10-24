"""Tests for atomic text writes and defensive JSON reads (core/sidecar)."""
from __future__ import annotations

import json

import pytest

from cadmia import __version__
from cadmia.core import provenance
from cadmia.core.sidecar import (artifact_fingerprint, atomic_write_text, check_input_directory,
                                 file_lock, read_json, verify_sidecar, write_json_sidecar)


def test_atomic_write_text_roundtrip(tmp_path):
    dest = tmp_path / "sub" / "state.json"
    atomic_write_text(dest, '{"a": 1}')
    assert dest.read_text() == '{"a": 1}'          # parent dir created


def test_atomic_write_text_replaces_existing_and_leaves_no_tmp(tmp_path):
    dest = tmp_path / "state.json"
    atomic_write_text(dest, "old")
    atomic_write_text(dest, "new")
    assert dest.read_text() == "new"
    # no leftover temp files next to the destination
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_read_json_roundtrip(tmp_path):
    dest = tmp_path / "x.json"
    atomic_write_text(dest, '{"k": [1, 2, 3]}')
    assert read_json(dest) == {"k": [1, 2, 3]}


def test_read_json_raises_actionable_on_truncated_file(tmp_path):
    dest = tmp_path / "half.json"
    dest.write_text('{"a": 1, "b":')             # interrupted mid-write
    with pytest.raises(ValueError) as exc:
        read_json(dest)
    assert "half.json" in str(exc.value)         # names the file and the remedy
    assert "corrupted" in str(exc.value) or "truncated" in str(exc.value)


def test_write_json_sidecar_appends_newline(tmp_path):
    written, err = write_json_sidecar(tmp_path / "s.json", '{"a": 1}')
    assert err is None
    text = (tmp_path / "s.json").read_text()
    assert text.endswith("\n")
    assert json.loads(text)["a"] == 1      # provenance is added alongside, not instead


def test_file_lock_is_exclusive(tmp_path):
    """A second acquirer of a held lock fails (non-blocking) — no lost updates."""
    import fcntl
    import os
    target = tmp_path / "cadmia_jobs.json"
    fd = os.open(tmp_path / "cadmia_jobs.json.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)                     # another process holds it
    try:
        with pytest.raises(BlockingIOError):
            with file_lock(target, blocking=False):
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_file_lock_releases_on_exit(tmp_path):
    target = tmp_path / "s.json"
    with file_lock(target):
        pass
    with file_lock(target, blocking=False):           # freed → acquirable again
        pass


# ── A sidecar must say which code wrote it ──────────────────────────

def test_the_sidecar_records_which_code_wrote_it(tmp_path):
    dest = tmp_path / "prescreen.json"
    write_json_sidecar(dest, json.dumps({"n_kept": 3}))
    prov = json.loads(dest.read_text())["provenance"]
    assert prov["software_version"].startswith("cadmia ")
    assert prov["command"] and prov["timestamp"]


def test_a_sidecar_that_is_not_an_object_is_written_untouched(tmp_path):
    dest = tmp_path / "list.json"
    write_json_sidecar(dest, "[1, 2, 3]")
    assert json.loads(dest.read_text()) == [1, 2, 3]


def test_a_caller_that_stamps_its_own_provenance_wins(tmp_path):
    dest = tmp_path / "s.json"
    write_json_sidecar(dest, json.dumps({"provenance": {"command": "mine"}}))
    assert json.loads(dest.read_text())["provenance"] == {"command": "mine"}


def test_provenance_survives_a_missing_git(tmp_path, monkeypatch):
    # A tarball install has no checkout to ask; that is not an error.
    monkeypatch.setattr(provenance, "_git_commit", lambda: None)
    assert provenance.software_version() == f"cadmia {__version__}"


def test_a_glob_expanded_command_is_truncated(tmp_path):
    # The shell expands configs/*.xyz before the process sees it: argv can carry a thousand
    # paths, and repeating them in every sidecar buries the record.
    long_cmd = "cadmia prescreen " + " ".join(f"cfg/c{i:04d}.xyz" for i in range(1144))
    prov = provenance.Provenance.for_run(command=long_cmd)
    assert len(prov.command) < 500
    assert "arguments)" in prov.command
    assert prov.command.startswith("cadmia prescreen cfg/c0000.xyz")


def test_a_short_command_is_kept_whole(tmp_path):
    prov = provenance.Provenance.for_run(command="cadmia doctor")
    assert prov.command == "cadmia doctor"


# ─────────────────────────────────────────────
# Artifact fingerprint: does the sidecar still describe what is on disk?
# ─────────────────────────────────────────────

def _xyz(path, comment="cfg"):
    path.write_text(f"1\n{comment}\nH 0.0 0.0 0.0\n", encoding="utf-8")
    return path


def test_fingerprint_changes_when_a_listed_file_changes(tmp_path):
    a, b = _xyz(tmp_path / "a.xyz"), _xyz(tmp_path / "b.xyz")
    before = artifact_fingerprint([a, b])
    assert before["n"] == 2 and before["sha256"]

    _xyz(a, comment="regenerated")
    assert artifact_fingerprint([a, b])["sha256"] != before["sha256"]


def test_fingerprint_does_not_depend_on_the_order_given(tmp_path):
    a, b = _xyz(tmp_path / "a.xyz"), _xyz(tmp_path / "b.xyz", "other")
    assert artifact_fingerprint([a, b])["sha256"] == artifact_fingerprint([b, a])["sha256"]


def test_fingerprint_lists_what_it_could_not_read(tmp_path):
    a = _xyz(tmp_path / "a.xyz")
    result = artifact_fingerprint([a, tmp_path / "gone.xyz"])
    assert result["n"] == 1
    assert result["missing"] == ["gone.xyz"]


def test_fingerprint_of_nothing_is_none():
    assert artifact_fingerprint([]) is None


def _sidecar_with(tmp_path, files):
    target = tmp_path / "geometry.json"
    write_json_sidecar(target, json.dumps({"n_configs": len(files)}), artifacts=files)
    return target


def test_verify_is_quiet_when_everything_matches(tmp_path):
    files = [_xyz(tmp_path / "a.xyz"), _xyz(tmp_path / "b.xyz", "b")]
    assert verify_sidecar(_sidecar_with(tmp_path, files)) == []


def test_verify_reports_a_regenerated_file(tmp_path):
    files = [_xyz(tmp_path / "a.xyz"), _xyz(tmp_path / "b.xyz", "b")]
    sidecar = _sidecar_with(tmp_path, files)
    _xyz(files[0], comment="regenerated after the sidecar was written")
    problems = verify_sidecar(sidecar)
    assert any("not the ones it describes" in p for p in problems)


def test_verify_reports_a_missing_file(tmp_path):
    files = [_xyz(tmp_path / "a.xyz"), _xyz(tmp_path / "b.xyz", "b")]
    sidecar = _sidecar_with(tmp_path, files)
    files[1].unlink()
    assert any("are gone" in p for p in verify_sidecar(sidecar))


def test_verify_reports_leftovers_from_an_earlier_run(tmp_path):
    """The case that actually bites.

    A second run with fewer sites overwrites the names it produces and leaves the rest of the
    previous run in place. The sidecar lists what this run wrote; the next stage globs the
    directory and reads more than that.
    """
    files = [_xyz(tmp_path / "a.xyz")]
    sidecar = _sidecar_with(tmp_path, files)
    _xyz(tmp_path / "from_the_previous_run.xyz", "older")
    problems = verify_sidecar(sidecar)
    assert any("not listed" in p and "earlier run" in p for p in problems)


def test_verify_says_nothing_about_a_sidecar_written_before_fingerprints(tmp_path):
    target = tmp_path / "geometry.json"
    write_json_sidecar(target, json.dumps({"n_configs": 2}))      # no artifacts given
    _xyz(tmp_path / "a.xyz")
    assert verify_sidecar(target) == []


def test_check_input_directory_finds_the_sidecar_by_name(tmp_path):
    files = [_xyz(tmp_path / "a.xyz")]
    _sidecar_with(tmp_path, files)
    _xyz(tmp_path / "stale.xyz", "older")
    assert any("not listed" in p for p in check_input_directory(tmp_path))
    assert check_input_directory(tmp_path / "does_not_exist") == []
