"""Latency-throughput frontiers, and how to compare two of them honestly.

Requirement LC-3 forbids single-number headline claims. For serving engines the
right object is the latency-throughput Pareto frontier: an engine is not
"1.4x faster", it dominates over some range of latency budgets and loses over
others, and which range you land in is a property of your SLO, not of the
engine.

Two frontiers are compared through their **attainment curves**: for each
latency budget t, the best throughput achievable at or under t. The ratio of
two attainment curves is a function of the budget, and the region where that
ratio dips below one *is* the loss column for Thrust II -- derived, not
editorially chosen.

Uncertainty is carried through as two frontiers rather than error bars on one:

``robust``    non-dominated even using each point's pessimistic corner.
``possible``  non-dominated using optimistic corners -- a point here may belong
              on the frontier but the data does not establish it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class OperatingPoint:
    """One tuned configuration, measured."""

    config: dict[str, Any]
    latency_ms: float                      # the SLO metric, lower is better
    throughput: float                      # tok/s or req/s, higher is better
    latency_ci: tuple[float, float] | None = None
    throughput_ci: tuple[float, float] | None = None
    label: str = ""
    trial: int | None = None
    tag: str = ""                          # e.g. "default" or "tuned"
    meta: dict[str, Any] = field(default_factory=dict)

    def optimistic(self) -> tuple[float, float]:
        lo_lat = self.latency_ci[0] if self.latency_ci else self.latency_ms
        hi_thr = self.throughput_ci[1] if self.throughput_ci else self.throughput
        return lo_lat, hi_thr

    def pessimistic(self) -> tuple[float, float]:
        hi_lat = self.latency_ci[1] if self.latency_ci else self.latency_ms
        lo_thr = self.throughput_ci[0] if self.throughput_ci else self.throughput
        return hi_lat, lo_thr

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _skyline(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated set for (minimise x, maximise y)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    keep: list[int] = []
    best_y = -math.inf
    for i in order:
        _, y = points[i]
        if y > best_y + 1e-12:
            keep.append(i)
            best_y = y
    return keep


@dataclass
class ParetoFrontier:
    """A tuned engine's latency-throughput frontier."""

    system: str
    points: list[OperatingPoint] = field(default_factory=list)
    latency_metric: str = "p99_latency_ms"
    throughput_metric: str = "output_tok_s"

    # ---- frontier extraction ---------------------------------------------

    def frontier_idx(self, mode: str = "point") -> list[int]:
        if mode == "point":
            pts = [(p.latency_ms, p.throughput) for p in self.points]
        elif mode == "robust":
            pts = [p.pessimistic() for p in self.points]
        elif mode == "possible":
            pts = [p.optimistic() for p in self.points]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return _skyline(pts)

    def frontier(self, mode: str = "point") -> list[OperatingPoint]:
        return [self.points[i] for i in self.frontier_idx(mode)]

    # ---- attainment -------------------------------------------------------

    def attainment(self, budgets: np.ndarray, mode: str = "point") -> np.ndarray:
        """Best throughput achievable at or under each latency budget.

        NaN where no measured configuration meets the budget: an engine that
        cannot hit a 50 ms SLO at all must show as absent, not as zero, because
        zero would silently average into a favourable comparison.
        """
        fr = self.frontier(mode)
        pairs = sorted(
            [(p.pessimistic() if mode == "robust" else
              p.optimistic() if mode == "possible" else
              (p.latency_ms, p.throughput)) for p in fr]
        )
        out = np.full(len(budgets), np.nan)
        best = -math.inf
        j = 0
        for k, t in enumerate(budgets):
            while j < len(pairs) and pairs[j][0] <= t:
                best = max(best, pairs[j][1])
                j += 1
            out[k] = best if best > -math.inf else np.nan
        return out

    def best_throughput(self) -> OperatingPoint | None:
        return max(self.points, key=lambda p: p.throughput, default=None)

    def best_latency(self) -> OperatingPoint | None:
        return min(self.points, key=lambda p: p.latency_ms, default=None)

    def tagged(self, tag: str) -> OperatingPoint | None:
        for p in self.points:
            if p.tag == tag:
                return p
        return None

    def default_to_tuned_gap(self) -> dict[str, Any] | None:
        """How much of the engine's performance is locked behind tuning.

        Reported as a primary result, not a footnote: most production
        deployments run near library defaults, so an engine that is fast only
        after a 40-trial search is making a different offer than one that is
        fast out of the box.
        """
        d, t = self.tagged("default"), self.tagged("tuned")
        if d is None or t is None:
            return None
        return {
            "default": d.to_dict(),
            "tuned": t.to_dict(),
            "throughput_gain_pct": (t.throughput / d.throughput - 1) * 100
            if d.throughput
            else float("nan"),
            "latency_change_pct": (t.latency_ms / d.latency_ms - 1) * 100
            if d.latency_ms
            else float("nan"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "latency_metric": self.latency_metric,
            "throughput_metric": self.throughput_metric,
            "points": [p.to_dict() for p in self.points],
            "frontier_point": [i for i in self.frontier_idx("point")],
            "frontier_robust": [i for i in self.frontier_idx("robust")],
            "default_to_tuned": self.default_to_tuned_gap(),
        }


def pareto_frontier(system: str, points: Sequence[OperatingPoint], **kw: Any) -> ParetoFrontier:
    return ParetoFrontier(system=system, points=list(points), **kw)


@dataclass
class FrontierComparison:
    """Ratio of two attainment curves, budget by budget."""

    method: str
    baseline: str
    budgets: np.ndarray
    ratio: np.ndarray                      # method / baseline throughput at budget
    method_attain: np.ndarray
    baseline_attain: np.ndarray
    mode: str = "point"

    @property
    def loss_budgets(self) -> np.ndarray:
        return self.budgets[np.nan_to_num(self.ratio, nan=1.0) < 1.0]

    def crossover(self) -> float | None:
        """Latency budget at which the ordering flips, if it flips once."""
        r = np.nan_to_num(self.ratio, nan=1.0)
        sign = np.sign(r - 1.0)
        flips = np.where(np.diff(sign) != 0)[0]
        if len(flips) != 1:
            return None
        i = int(flips[0])
        return float(self.budgets[i])

    def summary(self) -> dict[str, Any]:
        r = self.ratio[np.isfinite(self.ratio)]
        return {
            "mode": self.mode,
            "n_budgets": int(len(self.budgets)),
            "frac_budgets_method_wins": float(np.mean(r > 1.0)) if r.size else float("nan"),
            "frac_budgets_method_loses": float(np.mean(r < 1.0)) if r.size else float("nan"),
            "max_win_pct": float((np.nanmax(r) - 1) * 100) if r.size else float("nan"),
            "max_loss_pct": float((np.nanmin(r) - 1) * 100) if r.size else float("nan"),
            "crossover_ms": self.crossover(),
            "budgets_method_cannot_meet": [
                float(b)
                for b, m in zip(self.budgets, self.method_attain)
                if not np.isfinite(m)
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "baseline": self.baseline,
            "budgets_ms": self.budgets.tolist(),
            "ratio": [None if not np.isfinite(x) else float(x) for x in self.ratio],
            "method_attainment": [None if not np.isfinite(x) else float(x) for x in self.method_attain],
            "baseline_attainment": [None if not np.isfinite(x) else float(x) for x in self.baseline_attain],
            "summary": self.summary(),
        }


def compare_frontiers(
    method: ParetoFrontier,
    baseline: ParetoFrontier,
    *,
    n_budgets: int = 40,
    mode: str = "point",
    budgets: np.ndarray | None = None,
) -> FrontierComparison:
    """Compare two engines across the whole SLO range they both address."""
    if budgets is None:
        lats = [p.latency_ms for p in method.points + baseline.points if math.isfinite(p.latency_ms)]
        if not lats:
            raise ValueError("no finite latencies to compare")
        budgets = np.geomspace(max(min(lats), 1e-3), max(lats), n_budgets)
    ma = method.attainment(budgets, mode=mode)
    ba = baseline.attainment(budgets, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = ma / ba
    return FrontierComparison(
        method=method.system,
        baseline=baseline.system,
        budgets=np.asarray(budgets, float),
        ratio=np.asarray(ratio, float),
        method_attain=ma,
        baseline_attain=ba,
        mode=mode,
    )
