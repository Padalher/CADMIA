

# ── Concurrency (parallel xTB across configurations) ──────────────────────────

from cadmia.agents.prescreen_agent import _resolve_jobs


def test_jobs_never_exceeds_the_number_of_configs():
    assert _resolve_jobs(64, 3) == 3
    assert _resolve_jobs(0, 2) <= 2


def test_explicit_jobs_wins_over_auto():
    assert _resolve_jobs(4, 100) == 4


def test_jobs_is_at_least_one():
    assert _resolve_jobs(0, 1) == 1
    # Anything non-positive means "auto", so a nonsense value still yields a usable count.
    assert _resolve_jobs(-5, 10) >= 1


def test_auto_jobs_follows_the_affinity_mask_not_the_core_count(monkeypatch):
    # Under SLURM a job pinned to 8 cores on a 128-core node must not launch 128 xtb.
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr("os.cpu_count", lambda: 128)
    assert _resolve_jobs(0, 100) == 8
