"""The exclusivity guard, which had no tests at all.

It exists because the failure happened here: a calibration run was launched
while a sweep was still going, and both had to be discarded. The hazard is
specifically that nothing downstream can detect it -- two interleaved runs
produce self-consistent replicates and tight intervals, so the numbers look
fine and are wrong by an amount that depends on what the other run was doing.

A guard against an undetectable failure has to be right on its own, because
nothing else will notice when it is not.
"""

from __future__ import annotations

import json
import os

import pytest

from losscolumn.thrusts.kernel import bench
from losscolumn.thrusts.kernel.bench import MeasurementLock


@pytest.fixture(autouse=True)
def _isolated_lock(tmp_path, monkeypatch):
    """Point the lock at a temp file so tests never touch the real one."""
    monkeypatch.setenv(bench._LOCK_ENV, str(tmp_path / "test.lock"))
    return tmp_path / "test.lock"


def _write_holder(path, pid, purpose="other run"):
    path.write_text(json.dumps({
        "pid": pid, "purpose": purpose, "started": "2026-01-01T00:00:00Z",
    }), encoding="utf-8")


# ------------------------------------------------------------- the happy path


def test_an_uncontended_lock_is_acquired_and_released(_isolated_lock):
    with MeasurementLock("sweep") as lk:
        assert lk.acquired
        assert lk.state["exclusive"] is True
        assert _isolated_lock.exists()
        held = json.loads(_isolated_lock.read_text(encoding="utf-8"))
        assert held["pid"] == os.getpid()
        assert held["purpose"] == "sweep"
    assert not _isolated_lock.exists(), "the lock must be released on exit"


def test_the_lock_is_released_even_when_the_body_raises(_isolated_lock):
    with pytest.raises(ValueError):
        with MeasurementLock("sweep"):
            raise ValueError("measurement blew up")
    assert not _isolated_lock.exists(), (
        "a crashed run must not leave a lock that blocks the next one")


# ----------------------------------------------------------------- contention


def test_a_live_holder_refuses_a_second_measurement(_isolated_lock, monkeypatch):
    _write_holder(_isolated_lock, 999999, "campaign")
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: True)

    with pytest.raises(RuntimeError, match="another losscolumn measurement"):
        MeasurementLock("sweep").__enter__()


def test_the_refusal_names_who_holds_it(_isolated_lock, monkeypatch):
    """So the operator can wait for the right thing rather than guess."""
    _write_holder(_isolated_lock, 999999, "the communication campaign")
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: True)

    with pytest.raises(RuntimeError) as e:
        MeasurementLock("sweep").__enter__()
    msg = str(e.value)
    assert "999999" in msg and "the communication campaign" in msg


def test_non_strict_records_contention_instead_of_refusing(_isolated_lock, monkeypatch):
    _write_holder(_isolated_lock, 999999, "campaign")
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: True)

    with MeasurementLock("sweep", strict=False) as lk:
        assert lk.state["exclusive"] is False
        assert lk.state["contended_by"]["pid"] == 999999


def test_a_contended_run_does_not_steal_the_holders_lock(_isolated_lock, monkeypatch):
    """The dangerous version of proceeding anyway.

    Overwriting the file with our own pid means our exit deletes it, and the
    run that actually holds the lock is left unprotected against a third.
    """
    _write_holder(_isolated_lock, 999999, "campaign")
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: True)

    with MeasurementLock("sweep", strict=False):
        held = json.loads(_isolated_lock.read_text(encoding="utf-8"))
        assert held["pid"] == 999999, "the holder's record must survive"

    assert _isolated_lock.exists(), "exiting must not remove someone else's lock"
    held = json.loads(_isolated_lock.read_text(encoding="utf-8"))
    assert held["pid"] == 999999


# --------------------------------------------------------------- stale locks


def test_a_stale_lock_is_cleared_and_reclaimed(_isolated_lock, monkeypatch):
    """A killed run must not block the machine forever."""
    _write_holder(_isolated_lock, 999999, "killed run")
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: False)

    with MeasurementLock("sweep") as lk:
        assert lk.acquired
        assert lk.state["exclusive"] is True
        held = json.loads(_isolated_lock.read_text(encoding="utf-8"))
        assert held["pid"] == os.getpid()


def test_an_unreadable_lock_is_treated_as_stale(_isolated_lock):
    _isolated_lock.write_text("{ this is not json", encoding="utf-8")
    with MeasurementLock("sweep") as lk:
        assert lk.acquired


def test_our_own_pid_in_the_lock_is_not_contention(_isolated_lock):
    """Re-entering under the same process is not two measurements."""
    _write_holder(_isolated_lock, os.getpid(), "earlier stage")
    with MeasurementLock("sweep") as lk:
        assert lk.acquired
        assert lk.state["exclusive"] is True


# ------------------------------------------------------------- atomic claim


def test_the_claim_is_atomic_not_check_then_write(_isolated_lock, monkeypatch):
    """The bug this replaced.

    Checking `exists()` and then writing is two operations. Between them a
    second process makes the same check, reaches the same conclusion, and
    writes the same file -- so both proceed believing they hold the lock, which
    is exactly the contention the class exists to prevent.

    Simulated by having the file appear after the existence check would have
    passed: an atomic create-or-fail notices, a check-then-write does not.
    """
    import errno

    real_open = os.open
    calls = {"n": 0}

    def racing_open(path, flags, *a, **kw):
        # First claim attempt loses the race: something else created it first.
        if flags & os.O_EXCL and calls["n"] == 0:
            calls["n"] += 1
            _write_holder(_isolated_lock, 999999, "winner")
            raise FileExistsError(errno.EEXIST, "File exists")
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: True)

    with pytest.raises(RuntimeError, match="another losscolumn measurement"):
        MeasurementLock("sweep").__enter__()
    assert calls["n"] == 1, "the claim must go through an exclusive create"


def test_losing_the_race_to_reclaim_a_stale_lock_is_reported(_isolated_lock, monkeypatch):
    """Two runs clear the same stale lock; only one may end up holding it."""
    import errno

    real_open = os.open
    state = {"n": 0}

    def racing_open(path, flags, *a, **kw):
        if flags & os.O_EXCL:
            state["n"] += 1
            raise FileExistsError(errno.EEXIST, "File exists")
        return real_open(path, flags, *a, **kw)

    _write_holder(_isolated_lock, 999999, "dead run")
    monkeypatch.setattr(bench, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(os, "open", racing_open)

    with pytest.raises(RuntimeError, match="claimed the lock while"):
        MeasurementLock("sweep").__enter__()
    assert state["n"] == 2, "one claim, then one retry after clearing the stale lock"


# ---------------------------------------------------------- pid liveness


def test_pid_alive_says_yes_for_this_process():
    assert bench._pid_alive(os.getpid()) is True


def test_pid_alive_says_no_for_an_impossible_pid():
    """Fail-safe direction: an unknown pid must not read as alive.

    Reading a dead pid as alive blocks a machine that is free, which is
    annoying; reading a live one as dead permits two concurrent measurements,
    which is the failure that cost two runs.
    """
    assert bench._pid_alive(-1) is False
