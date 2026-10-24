from __future__ import annotations

from pathlib import Path

from cadmia.calculators.tidy import (AUX_SUFFIXES, slurm_tidy_block,
                                     tidy_job_outputs)


def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_text("x", encoding="utf-8")
    return p


def test_tidy_moves_only_accessory_files(tmp_path):
    stem = "job1"
    # essentials that must stay
    keep = [f"{stem}.inp", f"{stem}.out", f"{stem}.gbw",
            f"{stem}.densities", f"{stem}.densitiesinfo"]
    # accessory that must move
    move = [f"{stem}.bibtex", f"{stem}.property.txt", f"{stem}.chelpg.xyz",
            f"{stem}.pc_chelpg", f"{stem}.vpot", f"{stem}.xyz"]
    for n in keep + move:
        _touch(tmp_path, n)

    moved = tidy_job_outputs(tmp_path, stem)

    assert sorted(moved) == sorted(move)
    for n in keep:
        assert (tmp_path / n).is_file(), f"{n} should stay in place"
    for n in move:
        assert not (tmp_path / n).exists(), f"{n} should have moved"
        assert (tmp_path / "aux" / n).is_file()


def test_tidy_densities_never_moved(tmp_path):
    # orca_plot reads the density from these — they must never be tucked away.
    assert ".densities" not in AUX_SUFFIXES
    assert ".densitiesinfo" not in AUX_SUFFIXES


def test_tidy_noop_when_nothing_to_move(tmp_path):
    _touch(tmp_path, "job.out")
    assert tidy_job_outputs(tmp_path, "job") == []
    assert not (tmp_path / "aux").exists()   # aux/ only created when needed


def test_tidy_only_touches_matching_stem(tmp_path):
    _touch(tmp_path, "keep.bibtex")
    _touch(tmp_path, "job.bibtex")
    tidy_job_outputs(tmp_path, "job")
    assert (tmp_path / "keep.bibtex").is_file()
    assert (tmp_path / "aux" / "job.bibtex").is_file()


def test_slurm_tidy_block_uses_stem_and_workdir():
    block = slurm_tidy_block("/remote/work/job1.inp", "/remote/work")
    assert "/remote/work/aux" in block
    assert "job1" in block
    assert ".bibtex" in block and ".vpot" in block


def test_the_users_own_xyz_is_not_tucked_away(tmp_path):
    """<stem>.xyz is a byproduct only when the ENGINE rewrote it.

    In a single point or a frequency job ORCA writes no geometry, so the file of that name is
    the one the user supplied — the geometry agent's output. Moving it into aux/ takes it out of
    the directory `prepare` rescans.
    """
    import time

    from cadmia.calculators.tidy import tidy_job_outputs

    (tmp_path / "cfg.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    time.sleep(0.01)
    (tmp_path / "cfg.inp").write_text("! PBE0\n", encoding="utf-8")   # written after the xyz
    (tmp_path / "cfg.bibtex").write_text("x", encoding="utf-8")

    moved = tidy_job_outputs(tmp_path, "cfg")
    assert "cfg.xyz" not in moved
    assert (tmp_path / "cfg.xyz").exists()
    assert "cfg.bibtex" in moved                   # a real byproduct still moves


def test_an_engine_written_xyz_is_still_tidied(tmp_path):
    import time

    from cadmia.calculators.tidy import tidy_job_outputs

    (tmp_path / "opt.inp").write_text("! PBE0 Opt\n", encoding="utf-8")
    time.sleep(0.01)
    (tmp_path / "opt.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")   # newer than the input

    moved = tidy_job_outputs(tmp_path, "opt")
    assert "opt.xyz" in moved
    assert not (tmp_path / "opt.xyz").exists()
