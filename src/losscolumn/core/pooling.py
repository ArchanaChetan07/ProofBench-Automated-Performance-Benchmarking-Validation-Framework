"""The pooling guard: sessions do not get combined until they are shown to be.

LC-8.1 says pooled measurements come from one session or from sessions shown
comparable. A rule nobody can violate by accident is worth writing down; this
is the thing that makes it one, by refusing at the point where the pooling
happens rather than noting it in a report afterwards.

The distinction the guard exists to protect is three-way, and collapsing it to
two is what caused the original failure:

``INVALID``      the measurement is broken -- no timing, a non-positive median,
                 a crashed rank. Nothing can be built on it.
``INCOMPARABLE`` the measurement is perfectly good and describes a machine
                 state that some other measurement does not share. Usable
                 alone, never averaged with the other.
``COMPARABLE``   poolable.

The middle state is the one that gets lost. A pipeline with only valid/invalid
either throws away good data or, far worse, keeps it and pools it -- which is
precisely how a 32-point densification made a fit measurably worse.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.stability import (
    ComparabilityVerdict,
    SentinelReading,
    SessionComparison,
    compare_sessions,
)

__all__ = ["PoolingViolation", "SessionLedger", "PoolingDecision", "guard_pool"]


class PoolingViolation(RuntimeError):
    """Raised when records from sessions not shown comparable would be combined.

    An exception rather than a warning. The failure this prevents is silent by
    nature -- the pooled fit looks fine, converges, reports a plausible error --
    so a diagnostic that can be scrolled past is not enough.
    """


@dataclass
class PoolingDecision:
    """What the guard decided, and on what evidence."""

    allowed: bool
    sessions: list[str] = field(default_factory=list)
    comparisons: list[SessionComparison] = field(default_factory=list)
    n_records_in: int = 0
    n_records_out: int = 0
    n_invalid: int = 0
    n_incomparable: int = 0
    reason: str = ""
    criterion: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed, "sessions": self.sessions,
            "criterion": self.criterion,
            "n_records_in": self.n_records_in, "n_records_out": self.n_records_out,
            "n_invalid": self.n_invalid, "n_incomparable": self.n_incomparable,
            "reason": self.reason,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


@dataclass
class SessionLedger:
    """Which sessions exist, and which of them may be combined.

    Built once from sentinel readings, then consulted wherever records are about
    to be pooled. Keeping the decision in one place means the criterion cannot
    quietly differ between two call sites.
    """

    criterion: float
    sentinels: dict[str, list[SentinelReading]] = field(default_factory=dict)
    _cache: dict[tuple[str, str], SessionComparison] = field(default_factory=dict)

    @classmethod
    def from_readings(cls, readings: Iterable[SentinelReading], *,
                      criterion: float) -> SessionLedger:
        led = cls(criterion=criterion)
        for r in readings:
            led.sentinels.setdefault(r.session_id, []).append(r)
        return led

    @property
    def sessions(self) -> list[str]:
        return sorted(self.sentinels)

    def comparison(self, a: str, b: str) -> SessionComparison:
        if a == b:
            return SessionComparison(
                ComparabilityVerdict.COMPARABLE, self.criterion, observed_ratio=1.0,
                session_a=a, session_b=b,
                reason="a session is comparable with itself",
            )
        key = (a, b) if a <= b else (b, a)
        if key not in self._cache:
            if a not in self.sentinels or b not in self.sentinels:
                missing = a if a not in self.sentinels else b
                self._cache[key] = SessionComparison(
                    ComparabilityVerdict.INVALID, self.criterion,
                    session_a=a, session_b=b,
                    reason=(
                        f"session {missing!r} has no sentinel readings. Without them "
                        "there is no evidence either way, and absence of evidence is "
                        "not comparability."
                    ),
                )
            else:
                self._cache[key] = compare_sessions(
                    self.sentinels[key[0]], self.sentinels[key[1]],
                    criterion=self.criterion)
        return self._cache[key]

    def verdict(self, a: str, b: str) -> ComparabilityVerdict:
        return self.comparison(a, b).verdict

    def poolable_set(self, sessions: Sequence[str]) -> bool:
        """Every pair must be comparable, not merely every pair with the first.

        Comparability is not transitive: A within tolerance of B and B of C says
        nothing about A and C, which can be two tolerances apart.
        """
        ss = sorted(set(sessions))
        return all(self.comparison(a, b).poolable
                   for i, a in enumerate(ss) for b in ss[i + 1:])

    def to_dict(self) -> dict[str, Any]:
        ss = self.sessions
        return {
            "criterion": self.criterion,
            "sessions": ss,
            "n_sentinel_readings": {k: len(v) for k, v in self.sentinels.items()},
            "pairs": [self.comparison(a, b).to_dict()
                      for i, a in enumerate(ss) for b in ss[i + 1:]],
            "all_poolable": self.poolable_set(ss) if len(ss) > 1 else True,
        }


def guard_pool(records: Sequence[Any], ledger: SessionLedger | None, *,
               session_of: Callable[[Any], str] = lambda r: getattr(r, "session_id", ""),
               valid_of: Callable[[Any], bool] = lambda r: getattr(r, "valid", True),
               strict: bool = True) -> tuple[list[Any], PoolingDecision]:
    """Filter records to a poolable set, or refuse.

    Returns the records that may legitimately be combined together with the
    decision that produced them. With ``strict`` the guard raises instead of
    quietly returning a subset, because a caller that asked to pool four
    sessions and silently got two has been given an answer to a question it did
    not ask.
    """
    valid = [r for r in records if valid_of(r)]
    dec = PoolingDecision(
        allowed=True, n_records_in=len(records), n_invalid=len(records) - len(valid),
        criterion=ledger.criterion if ledger else float("nan"),
    )
    sessions = sorted({session_of(r) for r in valid})
    dec.sessions = sessions

    if len(sessions) <= 1:
        dec.n_records_out = len(valid)
        dec.reason = (
            "a single session, so no cross-session pooling occurs"
            if sessions else "no valid records"
        )
        return valid, dec

    if ledger is None:
        dec.allowed = False
        dec.n_records_out = 0
        dec.reason = (
            f"{len(sessions)} sessions and no stability evidence at all. Pooling "
            "them would assume the comparability that the sentinel exists to "
            "establish."
        )
        if strict:
            raise PoolingViolation(dec.reason)
        return [], dec

    dec.comparisons = [ledger.comparison(a, b)
                       for i, a in enumerate(sessions) for b in sessions[i + 1:]]
    bad = [c for c in dec.comparisons if not c.poolable]
    if not bad:
        dec.n_records_out = len(valid)
        dec.reason = (
            f"all {len(sessions)} sessions mutually comparable within "
            f"{ledger.criterion:.2f}x"
        )
        return valid, dec

    dec.allowed = False
    dec.n_records_out = 0
    worst = max(bad, key=lambda c: (c.observed_ratio
                                    if c.observed_ratio == c.observed_ratio else 0.0))
    dec.n_incomparable = len(
        [r for r in valid
         if session_of(r) in {worst.session_a, worst.session_b}])
    dec.reason = (
        f"{len(bad)} of {len(dec.comparisons)} session pair(s) are not poolable; "
        f"worst is {worst.session_a} against {worst.session_b}. {worst.reason} "
        "These records are valid and remain usable within their own session."
    )
    if strict:
        raise PoolingViolation(dec.reason)
    return [], dec
