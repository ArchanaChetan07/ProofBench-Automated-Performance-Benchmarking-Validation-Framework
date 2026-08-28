"""Discontinuity ("cliff") detection over a swept envelope.

RQ1 asks whether commonly-recommended configurations sit at an optimum or
*adjacent to a cliff*. That is a question about the local topology of the
performance surface, and it is not answerable from aggregate throughput -- the
usual reporting practice averages exactly the structure that matters away.

Definitions used here:

**Step.** Between adjacent levels of one factor, holding all others fixed, the
log-ratio of the metric, oriented so positive means the surface got worse.

**Cliff.** A step whose degradation exceeds both an absolute floor and a robust
threshold from the pooled step distribution on that axis, *and* whose bootstrap
interval clears the floor. Both conditions are required: real surfaces are
noisy, and a single unlucky replicate should not manufacture a discontinuity.

**Exposure.** For any configuration, the worst degradation reachable by moving
one level along any axis. This is the number that answers RQ1 directly: a
config with 3% headroom and 40% exposure is a bad recommendation regardless of
how well it benchmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.envelope import Cell, Envelope


def _pct(x: float) -> str:
    """Keep "the configuration stops running" distinct from "it got slower"."""
    return "cannot run" if not math.isfinite(x) else f"{x:+.1f}%"


@dataclass
class Cliff:
    """One statistically supported discontinuity between adjacent levels."""

    axis: str
    from_level: Any
    to_level: Any
    fixed: dict[str, Any]
    degradation_pct: float
    ci_lo_pct: float
    ci_hi_pct: float
    robust_z: float
    from_cell: Cell
    to_cell: Cell
    direction: str = "increase"          # which way along the axis the fall happens

    def describe(self) -> str:
        fixed = ", ".join(f"{k}={v}" for k, v in self.fixed.items())
        at = f" at {fixed}" if fixed else ""
        if not math.isfinite(self.degradation_pct):
            return (
                f"{self.axis} {self.from_level} -> {self.to_level}{at}: the configuration "
                f"stops running entirely"
            )
        return (
            f"{self.axis} {self.from_level} -> {self.to_level}{at}: "
            f"{self.degradation_pct:+.1f}% "
            f"(95% CI {self.ci_lo_pct:+.1f}..{self.ci_hi_pct:+.1f}, z={self.robust_z:.1f})"
        )

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["from_cell"] = list(self.from_cell)
        d["to_cell"] = list(self.to_cell)
        d["description"] = self.describe()
        return d


@dataclass
class CliffReport:
    system: str
    metric: str
    cliffs: list[Cliff] = field(default_factory=list)
    exposure: dict[str, float] = field(default_factory=dict)   # cell label -> worst % reachable
    threshold_pct: float = 0.0
    n_steps_tested: int = 0

    @property
    def n_cliffs(self) -> int:
        return len(self.cliffs)

    def worst(self, k: int = 5) -> list[Cliff]:
        return sorted(self.cliffs, key=lambda c: -c.degradation_pct)[:k]

    def exposure_of(self, env: Envelope, cell: Cell) -> float:
        return self.exposure.get(env.label(cell), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "metric": self.metric,
            "threshold_pct": self.threshold_pct,
            "n_steps_tested": self.n_steps_tested,
            "n_cliffs": self.n_cliffs,
            "cliffs": [c.to_dict() for c in self.cliffs],
            "exposure": self.exposure,
        }

    def to_markdown(self, limit: int = 8) -> str:
        lines = [
            f"### Discontinuities -- {self.system} ({self.metric})",
            "",
            f"{self.n_cliffs} cliff(s) in {self.n_steps_tested} adjacent-level steps, "
            f"threshold {self.threshold_pct:.1f}%.",
            "",
            "| Axis | Transition | Held fixed | Degradation | 95% CI |",
            "|------|------------|------------|-------------|--------|",
        ]
        for c in self.worst(limit):
            fixed = ", ".join(f"{k}={v}" for k, v in c.fixed.items()) or "-"
            ci = (
                "-"
                if not math.isfinite(c.ci_lo_pct)
                else f"{c.ci_lo_pct:+.1f}..{c.ci_hi_pct:+.1f}"
            )
            lines.append(
                f"| {c.axis} | {c.from_level} -> {c.to_level} | {fixed} | "
                f"{_pct(c.degradation_pct)} | {ci} |"
            )
        if not self.cliffs:
            lines.append("| - | _no supported discontinuity_ | - | - | - |")
        return "\n".join(lines)


def detect_cliffs(
    env: Envelope,
    system: str,
    *,
    min_drop: float = 0.15,
    robust_z: float = 3.0,
    n_boot: int = 2000,
    seed: int = 0,
    axes: list[str] | None = None,
) -> CliffReport:
    """Find supported discontinuities in ``system``'s surface.

    ``min_drop`` is the absolute floor (0.15 = a 15% degradation across one
    level of one factor). ``robust_z`` sets how far outside the pooled step
    distribution a step must sit; a surface that degrades smoothly everywhere
    has no cliffs no matter how steep it is, which is the intended behaviour --
    a cliff is a *surprise*, not a slope.
    """
    rng = np.random.default_rng(seed)
    sign = -1.0 if env.metric.higher_is_better else 1.0
    axes = axes or [f.name for f in env.factors if f.ordered and len(f.levels) > 1]
    report = CliffReport(system=system, metric=env.metric.name, threshold_pct=min_drop * 100)

    exposure: dict[str, float] = {env.label(c): 0.0 for c in env.cells()}

    for axis in axes:
        steps: list[tuple[float, Cell, Cell, dict[str, Any]]] = []
        for fixed, base in env.fibers(axis):
            cells = env.fiber_cells(axis, base)
            for c0, c1 in zip(cells[:-1], cells[1:]):
                a = env.replicates_at(system, c0)
                b = env.replicates_at(system, c1)
                if a.size == 0 or b.size == 0:
                    # A level the system cannot run at all is the steepest
                    # cliff there is; record it as unbounded degradation.
                    if a.size and not b.size:
                        steps.append((math.inf, c0, c1, fixed))
                    continue
                d = sign * (math.log(float(np.median(b))) - math.log(float(np.median(a))))
                steps.append((d, c0, c1, fixed))

        finite = [s[0] for s in steps if math.isfinite(s[0])]
        if not finite:
            continue
        med = float(np.median(finite))
        mad = float(np.median(np.abs(np.asarray(finite) - med))) * 1.4826
        thresh_robust = med + robust_z * mad if mad > 0 else math.inf
        floor = math.log1p(min_drop)
        report.n_steps_tested += len(steps)

        for d, c0, c1, fixed in steps:
            if not math.isfinite(d):
                report.cliffs.append(
                    Cliff(
                        axis=axis,
                        from_level=env.coords(c0)[axis],
                        to_level=env.coords(c1)[axis],
                        fixed=fixed,
                        degradation_pct=float("inf"),
                        ci_lo_pct=float("inf"),
                        ci_hi_pct=float("inf"),
                        robust_z=float("inf"),
                        from_cell=c0,
                        to_cell=c1,
                    )
                )
                exposure[env.label(c0)] = float("inf")
                continue
            if d < floor or d < thresh_robust:
                if d > 0:
                    lbl = env.label(c0)
                    exposure[lbl] = max(exposure[lbl], (math.exp(d) - 1) * 100)
                continue

            lo, hi = _step_ci(env, system, c0, c1, sign, n_boot, rng)
            if lo <= floor:  # interval does not clear the floor -> not supported
                continue
            z = (d - med) / mad if mad > 0 else float("inf")
            report.cliffs.append(
                Cliff(
                    axis=axis,
                    from_level=env.coords(c0)[axis],
                    to_level=env.coords(c1)[axis],
                    fixed=fixed,
                    degradation_pct=(math.exp(d) - 1) * 100,
                    ci_lo_pct=(math.exp(lo) - 1) * 100,
                    ci_hi_pct=(math.exp(hi) - 1) * 100,
                    robust_z=float(z),
                    from_cell=c0,
                    to_cell=c1,
                )
            )
            lbl = env.label(c0)
            exposure[lbl] = max(exposure[lbl], (math.exp(d) - 1) * 100)

    report.exposure = exposure
    return report


def _step_ci(
    env: Envelope,
    system: str,
    c0: Cell,
    c1: Cell,
    sign: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    a = np.log(env.replicates_at(system, c0))
    b = np.log(env.replicates_at(system, c1))
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boots = sign * (b[ib].mean(axis=1) - a[ia].mean(axis=1))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def cliff_adjacency(
    env: Envelope, report: CliffReport, configs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each named configuration, its headroom and its one-step exposure.

    This is the RQ1 answer in tabular form: a recommended config is dangerous
    when a single-level change in any direction costs far more than the config
    itself gains.
    """
    out: list[dict[str, Any]] = []
    for cfg in configs:
        name = cfg.get("name", "config")
        coords = {k: v for k, v in cfg.items() if k in env.factor_names}
        try:
            cell = env.cell_of(**coords)
        except (KeyError, ValueError):
            out.append({"name": name, "error": "configuration is outside the swept envelope"})
            continue
        exposure = report.exposure.get(env.label(cell), 0.0)
        adjacent = [
            c.describe()
            for c in report.cliffs
            if c.from_cell == cell or c.to_cell == cell
        ]
        out.append(
            {
                "name": name,
                "coords": coords,
                "exposure_pct": exposure,
                "cliff_adjacent": bool(adjacent),
                "adjacent_cliffs": adjacent,
            }
        )
    return out
