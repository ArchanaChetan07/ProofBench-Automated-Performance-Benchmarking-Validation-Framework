"""Detecting a configuration that has two behaviours rather than one.

Every cost model in this project has the form ``t = f(nbytes)``: one message
size, one predicted time. That is a modelling assumption and it is usually
right. Where it is wrong it fails silently and expensively, because a
single-valued function fitted through a region with two behaviours returns
something near the middle -- a value the system never actually produces.

This is what the medium regime turned out to be. Between 512 KiB and about
900 KiB the gloo all_reduce at world 2 takes one of two paths, and individual
calls at a fixed size land 0.16x to 5.55x the point's own median. At exactly
1 MiB the time steps by a factor of ten and the variability collapses again, so
the band is not noise around a smooth curve -- it is the neighbourhood of an
algorithm-selection threshold, where the choice is not determined by the message
size alone.

Two consequences follow, and they are the reason this module exists rather than
a note in a report:

**More repeats do not help.** Averaging over a bistable point estimates the
*mixture proportion* more precisely. The proportion is not a property of the
message size, so a better estimate of it is a better estimate of nothing
durable.

**More model flexibility does not help either.** The obstacle is not a shape the
family cannot express. Any function of one variable is single-valued, and a
region with two behaviours is not.

So a bimodal cell cannot grade a model, in the same way and for a stronger
reason than a cell whose noise exceeds the gate: there is no target for the
model to be right about. It is reported as such rather than being fitted
through.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ModeSplit", "BimodalReport", "detect_bimodal", "find_threshold_band"]

MIN_SAMPLES = 8
"""Below this a gap in the sorted values is ordinary order-statistic spacing."""

GAP_FRACTION = 0.30
"""Share of the total spread one gap must take to count as a separation."""

EDGE_GUARD = 2
"""How far from either end the gap must sit. A gap after the first point is an
outlier, not a second mode, and calling it one would flag every noisy cell."""

MIN_MODE_SHARE = 0.15
"""Smallest share of the calls the minority mode must hold."""

SEPARATION = 1.8
"""How far apart the two mode centres must be. Below this the 'modes' are two
halves of one wide distribution."""


@dataclass
class ModeSplit:
    """One point's timings, tested for having two centres rather than one."""

    nbytes: int
    n: int = 0
    bimodal: bool = False
    low_centre: float = float("nan")
    high_centre: float = float("nan")
    separation: float = float("nan")
    minority_share: float = float("nan")
    gap_fraction: float = float("nan")
    cv: float = float("nan")
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "nbytes": self.nbytes, "n": self.n, "bimodal": self.bimodal,
            "low_centre": self.low_centre, "high_centre": self.high_centre,
            "separation": self.separation, "minority_share": self.minority_share,
            "gap_fraction": self.gap_fraction, "cv": self.cv,
            "reason": self.reason,
        }


def detect_bimodal(timings: Sequence[float], *, nbytes: int = 0) -> ModeSplit:
    """Is this point's timing distribution better described by two centres?

    Deliberately conservative. Timing distributions are right-skewed even when
    perfectly well behaved -- a scheduling hiccup produces a long tail, not a
    second mode -- so a split has to clear four separate conditions before it is
    called bimodal, and a wide unimodal spread fails all but the first.
    """
    t = np.sort(np.array([x for x in timings if x == x and x > 0], dtype=float))
    ms = ModeSplit(nbytes=nbytes, n=int(t.size))
    if t.size < MIN_SAMPLES:
        ms.reason = f"{t.size} timings, below the {MIN_SAMPLES} a split needs"
        return ms

    ms.cv = float(t.std() / t.mean()) if t.mean() > 0 else float("nan")
    spread = float(t[-1] - t[0])
    if spread <= 0:
        ms.reason = "every timing identical"
        return ms

    gaps = np.diff(t)
    # Only gaps away from the ends: a jump after the first or before the last
    # point is a tail, and tails are what timing distributions have.
    lo, hi = EDGE_GUARD, t.size - EDGE_GUARD - 1
    if hi <= lo:
        ms.reason = "too few interior positions to place a split"
        return ms
    interior = gaps[lo:hi]
    gi = int(np.argmax(interior)) + lo
    ms.gap_fraction = float(gaps[gi] / spread)

    left, right = t[:gi + 1], t[gi + 1:]
    ms.low_centre = float(np.median(left))
    ms.high_centre = float(np.median(right))
    ms.separation = (ms.high_centre / ms.low_centre
                     if ms.low_centre > 0 else float("inf"))
    ms.minority_share = float(min(left.size, right.size) / t.size)

    checks = [
        (ms.gap_fraction >= GAP_FRACTION,
         f"the largest interior gap holds {ms.gap_fraction:.0%} of the spread, "
         f"below {GAP_FRACTION:.0%}"),
        (ms.minority_share >= MIN_MODE_SHARE,
         f"the smaller group holds {ms.minority_share:.0%} of the calls, below "
         f"{MIN_MODE_SHARE:.0%}: a few stragglers, not a mode"),
        (ms.separation >= SEPARATION,
         f"the two centres differ by {ms.separation:.2f}x, below "
         f"{SEPARATION:.1f}x: one wide distribution, not two"),
    ]
    failed = [why for ok, why in checks if not ok]
    if failed:
        ms.reason = failed[0]
        return ms

    ms.bimodal = True
    ms.reason = (
        f"{ms.minority_share:.0%} of calls sit at {ms.low_centre * 1e3:.3f} ms and "
        f"the rest at {ms.high_centre * 1e3:.3f} ms, {ms.separation:.1f}x apart, "
        f"separated by a gap holding {ms.gap_fraction:.0%} of the spread"
    )
    return ms


@dataclass
class BimodalReport:
    """Which sizes in a group have two behaviours, and what that implies."""

    group: str = ""
    splits: list[ModeSplit] = field(default_factory=list)
    band_lo: int = 0
    band_hi: int = 0
    step_at: int = 0
    step_prev: int = 0
    step_factor: float = float("nan")
    inversions: list[tuple[int, int, float]] = field(default_factory=list)
    """Adjacent size pairs where the larger message was measurably faster.

    The most robust signature of the three, and the only one that survives
    averaging. A cost surface must be non-decreasing in message size, so a
    decrease is either noise or a configuration that is not choosing the same
    algorithm at both sizes. Per-point bimodality, by contrast, is progressively
    hidden as each timing averages more calls: the two modes blend into one
    mixture mean, the distribution looks unimodal, and the underlying
    instability is untouched.
    """
    notes: list[str] = field(default_factory=list)

    @property
    def bimodal_sizes(self) -> list[int]:
        return sorted({s.nbytes for s in self.splits if s.bimodal})

    @property
    def has_threshold(self) -> bool:
        return (bool(self.bimodal_sizes) or math.isfinite(self.step_factor)
                or bool(self.inversions))

    @property
    def flagged_sizes(self) -> list[int]:
        """Every size implicated, individually. Not a hull.

        The band endpoints are kept for reporting, but a hull spanning three
        decades because of two unrelated events at either end is not a band --
        it is an accident of taking min and max.
        """
        out = set(self.bimodal_sizes)
        for a, b, _ in self.inversions:
            out |= {a, b}
        if self.step_at:
            out.add(self.step_at)
            if self.step_prev:
                out.add(self.step_prev)
        return sorted(out)

    def covers(self, nbytes: int) -> bool:
        return nbytes in set(self.flagged_sizes)

    def flagged_regimes(self, regime_of) -> tuple[str, ...]:
        """The regimes containing a flagged size, and only those."""
        return tuple(sorted({regime_of(n) for n in self.flagged_sizes}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "bimodal_sizes": self.bimodal_sizes,
            "band_lo": self.band_lo, "band_hi": self.band_hi,
            "step_at": self.step_at, "step_prev": self.step_prev,
            "step_factor": self.step_factor,
            "flagged_sizes": self.flagged_sizes,
            "inversions": [list(x) for x in self.inversions],
            "has_threshold": self.has_threshold,
            "splits": [s.to_dict() for s in self.splits],
            "notes": self.notes,
        }


def find_threshold_band(records: Sequence[Any], *, group: str = "",
                        step_ratio: float = 3.0,
                        inversion_ratio: float = 1.30) -> BimodalReport:
    """Locate an algorithm-selection threshold and the unstable band around it.

    Three signatures, any of which is sufficient.

    A *step*: the median jumps by more than `step_ratio` between adjacent sizes,
    which no bandwidth term can produce over a small size increment.

    An *inversion*: a larger message measured faster than a smaller one by more
    than `inversion_ratio`. This is the signature to trust, because it survives
    averaging -- it is a statement about medians, not about the shape of a
    distribution.

    *Bistability*: individual sizes whose calls split into two groups. The most
    direct evidence, and the most fragile: as each timing averages more calls
    the two modes blend into a mixture mean and the point looks unimodal while
    behaving exactly as badly.
    """
    rep = BimodalReport(group=group)
    usable = [r for r in records
              if getattr(r, "valid", True) and len(getattr(r, "timings_s", []) or []) >= MIN_SAMPLES]
    if not usable:
        rep.notes.append("no point carries enough timings to test")
        return rep

    for r in sorted(usable, key=lambda x: x.nbytes):
        rep.splits.append(detect_bimodal(r.timings_s, nbytes=r.nbytes))

    by_size: dict[int, list[float]] = {}
    for r in usable:
        m = float(np.median(r.timings_s))
        if m > 0:
            by_size.setdefault(r.nbytes, []).append(m)
    sizes = sorted(by_size)
    med = {n: float(np.median(v)) for n, v in by_size.items()}
    # The LARGEST step, not the first: a small early jump would otherwise mask
    # the threshold that matters.
    best_step = 0.0
    for a, b in zip(sizes, sizes[1:], strict=False):
        if med[a] <= 0:
            continue
        ratio = med[b] / med[a]
        if ratio >= step_ratio and ratio > best_step:
            best_step = ratio
            rep.step_at, rep.step_prev, rep.step_factor = b, a, ratio
        if ratio <= 1.0 / inversion_ratio:
            rep.inversions.append((a, b, ratio))

    # Reported for orientation only. What implicates a regime is
    # `flagged_sizes`, which does not join unrelated events into one span.
    marks = rep.flagged_sizes
    if marks:
        rep.band_lo, rep.band_hi = min(marks), max(marks)

    if rep.has_threshold:
        bits = []
        if math.isfinite(rep.step_factor):
            bits.append(f"the median steps by {rep.step_factor:.1f}x at "
                        f"{rep.step_at} bytes")
        if rep.inversions:
            worst = min(rep.inversions, key=lambda x: x[2])
            bits.append(
                f"{len(rep.inversions)} size pair(s) run backwards, worst "
                f"{worst[0]}B to {worst[1]}B at {1 / worst[2]:.2f}x faster for "
                "the larger message")
        if rep.bimodal_sizes:
            bits.append(f"{len(rep.bimodal_sizes)} size(s) split their calls "
                        "into two groups")
        rep.notes.append("; ".join(bits))
        rep.notes.append(
            "A single-valued cost model cannot describe this band. Neither more "
            "repeats nor more segments address it: repeats estimate the mixture "
            "proportion, which is not a property of the message size, and any "
            "function of one variable is single-valued."
        )
    return rep
