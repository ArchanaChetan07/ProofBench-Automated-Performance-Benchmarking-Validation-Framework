"""Exact interval algebra over profiler timelines.

Overlap efficiency is defined in the proposal as the fraction of collective
time hidden behind compute. Computed naively -- by summing kernel durations --
it is wrong whenever kernels on the same stream overlap in the trace, which
they routinely do once you include NCCL streams and CUDA graphs. The honest
computation is set-theoretic:

    exposed        = measure( union(comm) \\ union(compute) )
    overlap_eff    = 1 - exposed / measure(union(comm))

This module implements union / difference / intersection / measure on sets of
half-open intervals with a sweep, in O(n log n), with no float accumulation
error beyond the endpoints themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

Interval = tuple[float, float]


@dataclass(frozen=True)
class IntervalSet:
    """A canonical (sorted, disjoint, non-empty) set of half-open intervals."""

    intervals: tuple[Interval, ...] = ()

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_spans(cls, spans: Iterable[Sequence[float]]) -> IntervalSet:
        """Build from arbitrary ``(start, end)`` pairs; merges and drops empties."""
        raw: list[Interval] = []
        for s in spans:
            a, b = float(s[0]), float(s[1])
            if b > a:
                raw.append((a, b))
        return cls._merge(raw)

    @staticmethod
    def _merge(raw: list[Interval]) -> IntervalSet:
        if not raw:
            return IntervalSet(())
        raw.sort()
        out: list[Interval] = [raw[0]]
        for a, b in raw[1:]:
            la, lb = out[-1]
            if a <= lb:  # touching or overlapping -> merge
                if b > lb:
                    out[-1] = (la, b)
            else:
                out.append((a, b))
        return IntervalSet(tuple(out))

    # ---- algebra ----------------------------------------------------------

    def union(self, other: IntervalSet) -> IntervalSet:
        return IntervalSet._merge(list(self.intervals) + list(other.intervals))

    def intersect(self, other: IntervalSet) -> IntervalSet:
        out: list[Interval] = []
        i = j = 0
        A, B = self.intervals, other.intervals
        while i < len(A) and j < len(B):
            lo = max(A[i][0], B[j][0])
            hi = min(A[i][1], B[j][1])
            if hi > lo:
                out.append((lo, hi))
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return IntervalSet(tuple(out))

    def difference(self, other: IntervalSet) -> IntervalSet:
        """``self \\ other``."""
        out: list[Interval] = []
        j = 0
        B = other.intervals
        for a, b in self.intervals:
            cur = a
            while j < len(B) and B[j][1] <= cur:
                j += 1
            k = j
            while k < len(B) and B[k][0] < b:
                if B[k][0] > cur:
                    out.append((cur, min(B[k][0], b)))
                cur = max(cur, B[k][1])
                if cur >= b:
                    break
                k += 1
            if cur < b:
                out.append((cur, b))
        return IntervalSet(tuple(out))

    # ---- measures ---------------------------------------------------------

    def measure(self) -> float:
        """Total covered length."""
        return float(sum(b - a for a, b in self.intervals))

    def span(self) -> float:
        """Extent from first start to last end, including the gaps."""
        if not self.intervals:
            return 0.0
        return self.intervals[-1][1] - self.intervals[0][0]

    def gaps(self) -> IntervalSet:
        """Idle intervals inside the span -- the bubbles."""
        if len(self.intervals) < 2:
            return IntervalSet(())
        out = [
            (self.intervals[i][1], self.intervals[i + 1][0])
            for i in range(len(self.intervals) - 1)
            if self.intervals[i + 1][0] > self.intervals[i][1]
        ]
        return IntervalSet(tuple(out))

    def clip(self, lo: float, hi: float) -> IntervalSet:
        return self.intersect(IntervalSet(((lo, hi),)))

    def shift(self, dt: float) -> IntervalSet:
        return IntervalSet(tuple((a + dt, b + dt) for a, b in self.intervals))

    # ---- dunder -----------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.intervals)

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Iterator[Interval]:
        return iter(self.intervals)

    def __or__(self, other: IntervalSet) -> IntervalSet:
        return self.union(other)

    def __and__(self, other: IntervalSet) -> IntervalSet:
        return self.intersect(other)

    def __sub__(self, other: IntervalSet) -> IntervalSet:
        return self.difference(other)

    def to_list(self) -> list[list[float]]:
        return [[a, b] for a, b in self.intervals]


def coverage_profile(sets: dict[str, IntervalSet], lo: float, hi: float) -> list[dict]:
    """Piecewise-constant map of which categories are live over ``[lo, hi)``.

    Used to render the timeline strip in the overlap artifact, and to sanity
    check attribution: if some wall-clock is covered by neither compute nor
    communication, it is genuine idle and must be reported, not absorbed.
    """
    edges = {lo, hi}
    for s in sets.values():
        for a, b in s.clip(lo, hi):
            edges.add(a)
            edges.add(b)
    xs = sorted(edges)
    out: list[dict] = []
    for a, b in zip(xs[:-1], xs[1:], strict=False):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        live = sorted(
            k for k, s in sets.items() if any(x <= mid < y for x, y in s.intervals)
        )
        out.append({"start": a, "end": b, "live": live})
    return out
