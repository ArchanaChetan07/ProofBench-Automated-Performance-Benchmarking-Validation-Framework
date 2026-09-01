"""Is the surface modellable here, and what would it cost to find out?

The coverage gate asks for held-out error at or under some threshold. Whether
that is achievable is not a question about models: it is a question about how
precisely this machine can be interrogated, and it has an arithmetic answer.

Measurement variance on this machine arrives at two levels, and they behave
completely differently under more work:

* **within a run** -- averages down with repeats, as 1/sqrt(n)
* **at the process launch** -- does not. Repeating inside one process cannot
  touch it; only more launches can.

So there is a floor: a campaign that adds repeats forever converges on the
launch-level dispersion and stops. If that floor sits above the gate, no amount
of repeating clears it and no model family is at fault. That is worth knowing
before anyone spends a week on a better model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NoiseBudget", "ReplicatePlan", "CellReachability", "Modellability",
           "decompose_noise", "plan_replicates", "assess_modellability"]

MEDIAN_SE_FACTOR = 1.2533
"""sqrt(pi/2): the standard error of a median relative to that of a mean."""


@dataclass
class NoiseBudget:
    """Measurement variance split into the part that averages and the part that does not."""

    within_run_cv: float = float("nan")
    across_restart_cv: float = float("nan")
    n_repeats: int = 0
    n_launches: int = 0

    @property
    def expected_restart_cv(self) -> float:
        """What the restart level would show if repeats were its only source."""
        if not (self.within_run_cv > 0 and self.n_repeats > 0):
            return float("nan")
        return MEDIAN_SE_FACTOR * self.within_run_cv / math.sqrt(self.n_repeats)

    @property
    def launch_cv(self) -> float:
        """The launch-level component, net of what repeats already explain.

        Variances subtract, not the coefficients themselves. Where the observed
        restart dispersion is at or below what averaging predicts there is no
        launch component to speak of, and this is zero rather than imaginary.
        """
        a, b = self.across_restart_cv, self.expected_restart_cv
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan")
        return math.sqrt(max(a * a - b * b, 0.0))

    def point_noise(self, n_repeats: int, n_launches: int = 1) -> float:
        """Dispersion of an estimate built from this many repeats and launches."""
        if not (self.within_run_cv > 0):
            return float("nan")
        w = MEDIAN_SE_FACTOR * self.within_run_cv / math.sqrt(max(n_repeats, 1))
        lv = self.launch_cv
        lv = 0.0 if not math.isfinite(lv) else lv / math.sqrt(max(n_launches, 1))
        return math.sqrt(w * w + lv * lv)

    @property
    def floor(self) -> float:
        """The best achievable with unlimited repeats in a single launch.

        The number that decides whether a gate is reachable at all by the usual
        remedy of measuring harder.
        """
        lv = self.launch_cv
        return lv if math.isfinite(lv) else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "within_run_cv": self.within_run_cv,
            "across_restart_cv": self.across_restart_cv,
            "expected_restart_cv": self.expected_restart_cv,
            "launch_cv": self.launch_cv, "floor": self.floor,
            "n_repeats": self.n_repeats, "n_launches": self.n_launches,
        }


@dataclass
class ReplicatePlan:
    """How much work a target precision costs, and whether it is buyable."""

    target: float
    repeats_needed: int = 0
    launches_needed: int = 1
    reachable_by_repeats_alone: bool = False
    reachable_at_all: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target, "repeats_needed": self.repeats_needed,
            "launches_needed": self.launches_needed,
            "reachable_by_repeats_alone": self.reachable_by_repeats_alone,
            "reachable_at_all": self.reachable_at_all, "note": self.note,
        }


def decompose_noise(envelope: dict[str, Any]) -> NoiseBudget:
    """Read a NoiseBudget out of a measured stability envelope."""
    wr = envelope.get("within_run", {}) or {}
    ar = envelope.get("across_restart", {}) or {}
    return NoiseBudget(
        within_run_cv=float(wr.get("cv", float("nan"))),
        across_restart_cv=float(ar.get("cv", float("nan"))),
        n_repeats=int(wr.get("n_aggregated", 0) or 0),
        n_launches=int(ar.get("n_aggregated", 0) or 0),
    )


def plan_replicates(budget: NoiseBudget, target: float, *,
                    max_repeats: int = 400, max_launches: int = 40) -> ReplicatePlan:
    """The cheapest (repeats, launches) that reaches `target`, if any does.

    Repeats first because they are far cheaper than launches -- a launch pays
    process startup and transport setup for every probe in it. Launches are
    added only where the floor makes them unavoidable.
    """
    plan = ReplicatePlan(target=target)
    if not (budget.within_run_cv > 0) or target <= 0:
        plan.note = "no noise estimate, so no plan can be costed"
        return plan

    floor = budget.floor
    if math.isfinite(floor) and floor > target:
        plan.reachable_by_repeats_alone = False
        # Repeats have hit their floor; buy the rest with launches.
        need_l = math.ceil((MEDIAN_SE_FACTOR * floor / target) ** 2) if target > 0 else 0
        for n in range(1, max_repeats + 1):
            if budget.point_noise(n, need_l) <= target:
                plan.repeats_needed, plan.launches_needed = n, need_l
                plan.reachable_at_all = need_l <= max_launches
                break
        plan.launches_needed = max(plan.launches_needed, need_l)
        plan.reachable_at_all = plan.launches_needed <= max_launches and bool(
            plan.repeats_needed)
        plan.note = (
            f"repeats alone converge on {floor:.1%}, above the {target:.0%} target: "
            f"the launch-level component does not average down inside one process, "
            f"so this needs about {plan.launches_needed} launches per point as well"
        )
        return plan

    for n in range(1, max_repeats + 1):
        if budget.point_noise(n, 1) <= target:
            plan.repeats_needed = n
            plan.launches_needed = 1
            plan.reachable_by_repeats_alone = True
            plan.reachable_at_all = True
            plan.note = (
                f"{n} repeats in one launch reaches {target:.0%}; the launch-level "
                f"floor is {floor:.1%}, below the target, so no extra launches are "
                "required"
            )
            return plan
    plan.note = f"not reachable within {max_repeats} repeats"
    return plan


@dataclass
class CellReachability:
    """Whether one cell's gate is achievable, and why not when it is not."""

    group: str
    regime: str
    kind: str
    heldout_err: float
    noise_cv: float
    gate: float
    verdict: str = ""
    reason: str = ""
    repeats_needed: int = 0
    launches_needed: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group, "regime": self.regime, "kind": self.kind,
            "heldout_err": self.heldout_err, "noise_cv": self.noise_cv,
            "gate": self.gate, "verdict": self.verdict, "reason": self.reason,
            "repeats_needed": self.repeats_needed,
            "launches_needed": self.launches_needed,
        }


@dataclass
class Modellability:
    """The determination: is the required surface modellable on this machine?"""

    budget: NoiseBudget = field(default_factory=NoiseBudget)
    plans: dict[str, ReplicatePlan] = field(default_factory=dict)
    cells: list[CellReachability] = field(default_factory=list)
    gate: float = 0.15
    verdict: str = "UNDETERMINED"
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cells:
            out[c.verdict] = out.get(c.verdict, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "plans": {k: v.to_dict() for k, v in self.plans.items()},
            "gate": self.gate, "verdict": self.verdict, "summary": self.summary,
            "counts": self.counts,
            "cells": [c.to_dict() for c in self.cells],
            "notes": self.notes,
        }


def _with_floor(budget: NoiseBudget, floor: float) -> NoiseBudget:
    """A cell-level budget carrying the machine's launch floor.

    The launch component belongs to the machine, so it is the same in every
    cell; only the within-run part is the cell's own. Applied by choosing the
    restart-level dispersion that reproduces the wanted floor at this cell's
    repeat count, since that is what `launch_cv` inverts.
    """
    if not math.isfinite(floor):
        return budget
    exp = budget.expected_restart_cv
    if not math.isfinite(exp):
        return budget
    return NoiseBudget(
        within_run_cv=budget.within_run_cv,
        across_restart_cv=math.sqrt(exp * exp + floor * floor),
        n_repeats=budget.n_repeats, n_launches=budget.n_launches)


def assess_modellability(envelope: dict[str, Any], debt: dict[str, Any], *,
                         gate: float = 0.15,
                         targets: tuple[float, ...] = (0.20, 0.15, 0.10),
                         max_repeats: int = 400, max_launches: int = 40,
                         expensive_repeats: int = 30,
                         expensive_launches: int = 4,
                         ) -> Modellability:
    """Decide whether the coverage gate is reachable, cell by cell.

    Three outcomes per cell, and the middle one is the point of the exercise:

    ``BUYABLE``       the residual is noise and more measurement would remove it
    ``UNREACHABLE``   the noise floor is at or above the gate, so no amount of
                      measuring or modelling clears it here
    ``MODEL_WORK``    the measurement is clean and the gap is genuinely the
                      model's
    """
    m = Modellability(budget=decompose_noise(envelope), gate=gate)
    for t in targets:
        m.plans[f"{t:.0%}"] = plan_replicates(m.budget, t)

    for item in debt.get("items", []):
        c = CellReachability(
            group=item["group"], regime=item["regime"], kind=item["kind"],
            heldout_err=item.get("heldout_err", float("nan")),
            noise_cv=item.get("noise_cv", float("nan")), gate=gate,
        )
        cv, err = c.noise_cv, c.heldout_err

        # Cost this cell with its OWN within-run variation against the machine's
        # shared launch floor, rather than calling it unreachable because its raw
        # CV exceeds the gate. A noisy cell is not beyond reach; it is expensive,
        # and the difference between those two is the entire point of costing it.
        # Repeats shrink the within-run part as 1/sqrt(n) and the launch part not
        # at all, so a cell is only truly unreachable when the floor defeats it.
        plan = None
        if cv == cv and cv > 0:
            plan = plan_replicates(
                _with_floor(NoiseBudget(within_run_cv=cv,
                                        n_repeats=m.budget.n_repeats,
                                        n_launches=m.budget.n_launches),
                            m.budget.floor),
                gate, max_repeats=max_repeats, max_launches=max_launches)
            c.repeats_needed = plan.repeats_needed
            c.launches_needed = plan.launches_needed

        if plan is not None and not plan.reachable_at_all:
            c.verdict = "UNREACHABLE"
            c.reason = (
                f"run-to-run variation {cv:.1%} against a launch floor of "
                f"{m.budget.floor:.1%}: the {gate:.0%} gate is out of reach within "
                f"{max_repeats} repeats and {max_launches} launches per point"
            )
        elif plan is not None and (plan.repeats_needed > expensive_repeats
                                   or plan.launches_needed > expensive_launches):
            c.verdict = "EXPENSIVE"
            c.reason = (
                f"reachable, at {plan.repeats_needed} repeats across "
                f"{plan.launches_needed} launch(es) per point, for a cell whose "
                f"run-to-run variation is {cv:.1%}"
            )
        elif item["kind"] == "noise_limited":
            c.verdict = "BUYABLE"
            c.reason = (
                f"the residual ({err:.1%}) is the instrument's; "
                f"{c.repeats_needed} repeats per point reaches the gate"
            )
        elif item["kind"] == "model_limited":
            c.verdict = "MODEL_WORK"
            c.reason = (
                f"the measurement is clean ({cv:.1%}) and the best family still "
                f"misses by {err:.1%}: a real modelling gap"
            )
        else:
            c.verdict = "UNDETERMINED"
            c.reason = item.get("rationale", "")
        m.cells.append(c)

    n = m.counts
    unreachable = n.get("UNREACHABLE", 0)
    expensive = n.get("EXPENSIVE", 0)
    model_work = n.get("MODEL_WORK", 0)
    floor = m.budget.floor

    if math.isfinite(floor) and floor >= gate:
        m.verdict = "NOT_MODELLABLE_AT_THIS_GATE"
        m.summary = (
            f"The launch-level noise floor is {floor:.1%} and the gate asks for "
            f"{gate:.0%}. Repeats inside one process converge on that floor and "
            "stop, so the gate is not reachable by measuring harder, and it is not "
            "a modelling failure that it has not been met."
        )
    elif unreachable:
        m.verdict = "PARTLY_MODELLABLE"
        m.summary = (
            f"{unreachable} of {len(m.cells)} uncovered cells cannot be graded to "
            f"the {gate:.0%} gate by any practical amount of measurement. The other "
            f"{len(m.cells) - unreachable} can: {model_work} need modelling work, "
            f"{expensive} need far more replicates than the campaign carried, and "
            "the rest need a modest increase."
        )
    elif expensive:
        m.verdict = "MODELLABLE_AT_A_COST"
        m.summary = (
            f"Every uncovered cell is reachable, but {expensive} of {len(m.cells)} "
            f"need more than {expensive_repeats} repeats or more than "
            f"{expensive_launches} launches per point. {model_work} need modelling "
            "work on top of that."
        )
    elif model_work:
        m.verdict = "MODELLABLE_WITH_WORK"
        m.summary = (
            f"Every uncovered cell is measurable to the {gate:.0%} gate. "
            f"{model_work} of them need a model family that fits; the rest need "
            "replicates."
        )
    else:
        m.verdict = "MODELLABLE"
        m.summary = "Every uncovered cell is reachable with more replicates alone."
    return m
