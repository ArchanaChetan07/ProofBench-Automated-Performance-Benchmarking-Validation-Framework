"""Feasibility as a state, not as a number.

A configuration that will not run is not a configuration that runs at zero
tokens per second. Encoding it that way was a real defect in this project: the
comparison paths skip zero-valued cells, so four cells the model had predicted
infeasible vanished, the predicted loss map came back empty, and the study
reported it could establish nothing -- when the model had made four falsifiable
predictions and two of them were wrong.

The rule this module exists to enforce: **a number never carries a meaning that
is not numeric**. Feasibility is a state with its own type; throughput is
``None`` when there is no throughput; and a run that legitimately measures zero
tokens per second is *feasible with a throughput of zero*, which is a different
fact from *did not run*.

Five states, because the platform forces the distinction:

``FEASIBLE``        ran to completion inside the device-memory budget.
``INFEASIBLE``      the allocator refused inside the budget. A real answer.
``HOST_FALLBACK``   completed, but only because the driver served the
                    allocation from host memory. Not device-feasible, and the
                    difference matters: on this platform a fallback run does
                    not fail, it thrashes for hours, so counting it as feasible
                    would make "fits" mean "eventually returned".
``INVALID``         the measurement could not be trusted -- the budget was
                    never enforced, or the failure was not a memory failure.
                    Rejected rather than guessed at.
``NOT_MEASURED``    nobody looked. Distinct from every answer above.

Only ``FEASIBLE`` counts as feasible. The remaining four are not
interchangeable, and collapsing them is how a hardware refusal becomes a slow
success.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Feasibility",
    "FeasibilityObservation",
    "FeasibilityCell",
    "FeasibilityMatrix",
    "DEFAULT_DANGER_WEIGHT",
]


class Feasibility(str, Enum):
    """Whether a configuration runs, and if not, why the answer is what it is."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    HOST_FALLBACK = "host_fallback"
    INVALID = "invalid"
    NOT_MEASURED = "not_measured"

    @property
    def is_feasible(self) -> bool:
        """Only an unqualified success is feasible."""
        return self is Feasibility.FEASIBLE

    @property
    def is_definite(self) -> bool:
        """Whether this state answers the question at all.

        ``HOST_FALLBACK``, ``INVALID`` and ``NOT_MEASURED`` do not. They are
        excluded from classification metrics rather than being rounded to the
        nearest verdict, which would silently turn an unanswered cell into
        evidence.
        """
        return self in (Feasibility.FEASIBLE, Feasibility.INFEASIBLE)

    @property
    def ran(self) -> bool:
        """Whether the work completed, regardless of where the memory came from."""
        return self in (Feasibility.FEASIBLE, Feasibility.HOST_FALLBACK)


# How much worse a false win is than a false loss, for the weighted score in
# :meth:`FeasibilityMatrix.dangerous_error_score`. A false win kills a job that
# was launched on the model's advice; a false loss leaves capacity unused. Ten
# is a policy choice, not a measurement -- it is stated here, reported in every
# artifact, and the raw counts are always shown beside it so a reader who
# disagrees can reweigh them.
DEFAULT_DANGER_WEIGHT = 10.0


@dataclass(frozen=True)
class FeasibilityObservation:
    """One determination of feasibility, and how it was reached.

    ``throughput`` is ``None`` unless the configuration ran. It is deliberately
    *not* zero: zero is a legitimate measured throughput, and a type that
    cannot tell "no value" from "the value is zero" is the type this module
    exists to replace.
    """

    status: Feasibility
    throughput: float | None = None
    peak_bytes: int | None = None
    budget_bytes: int | None = None
    method: str = "unknown"
    detail: str = ""
    runtime_config: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float | None = None

    def __post_init__(self) -> None:
        if self.status.ran is False and self.throughput is not None:
            raise ValueError(
                f"a {self.status.value} observation carries a throughput of "
                f"{self.throughput}. A configuration that did not run has no "
                "throughput; encoding one is how 'did not run' becomes 'ran slowly'."
            )

    @property
    def is_feasible(self) -> bool:
        return self.status.is_feasible

    @property
    def usable_for_timing(self) -> bool:
        """Whether this observation's throughput may enter a performance claim."""
        return self.status is Feasibility.FEASIBLE and self.throughput is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "throughput": self.throughput,
            "peak_bytes": self.peak_bytes,
            "budget_bytes": self.budget_bytes,
            "method": self.method,
            "detail": self.detail,
            "runtime_config": self.runtime_config,
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeasibilityObservation:
        return cls(
            status=Feasibility(d.get("status", "not_measured")),
            throughput=d.get("throughput"),
            peak_bytes=d.get("peak_bytes"),
            budget_bytes=d.get("budget_bytes"),
            method=d.get("method", "unknown"),
            detail=d.get("detail", ""),
            runtime_config=d.get("runtime_config", {}) or {},
            elapsed_s=d.get("elapsed_s"),
        )

    @classmethod
    def not_measured(cls, detail: str = "") -> FeasibilityObservation:
        return cls(status=Feasibility.NOT_MEASURED, detail=detail, method="none")

    @classmethod
    def predicted(cls, fits: bool, *, detail: str = "",
                  predicted_bytes: int | None = None,
                  budget_bytes: int | None = None) -> FeasibilityObservation:
        """A model's opinion. Carries no throughput: a model did not run anything."""
        return cls(
            status=Feasibility.FEASIBLE if fits else Feasibility.INFEASIBLE,
            throughput=None,
            peak_bytes=predicted_bytes,
            budget_bytes=budget_bytes,
            method="model",
            detail=detail,
        )


# --------------------------------------------------------------------------
# per-cell agreement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibilityCell:
    """A predicted feasibility and a measured one, held apart.

    The two are stored independently and never merged into a single "correct"
    flag, because the two ways of being wrong are not equally bad and a reader
    must be able to see which one occurred.
    """

    key: Any
    predicted: FeasibilityObservation
    measured: FeasibilityObservation
    coords: dict[str, Any] = field(default_factory=dict)

    @property
    def comparable(self) -> bool:
        """Whether both sides gave a definite answer."""
        return self.predicted.status.is_definite and self.measured.status.is_definite

    @property
    def correct_feasible(self) -> bool:
        return (
            self.comparable
            and self.predicted.is_feasible
            and self.measured.is_feasible
        )

    @property
    def correct_infeasible(self) -> bool:
        return (
            self.comparable
            and not self.predicted.is_feasible
            and not self.measured.is_feasible
        )

    @property
    def false_win(self) -> bool:
        """Model predicts feasible, measurement says infeasible. The dangerous one."""
        return (
            self.comparable and self.predicted.is_feasible and not self.measured.is_feasible
        )

    @property
    def false_loss(self) -> bool:
        """Model predicts infeasible, measurement says feasible. Merely wasteful."""
        return (
            self.comparable and not self.predicted.is_feasible and self.measured.is_feasible
        )

    @property
    def outcome(self) -> str:
        if not self.comparable:
            return "not_comparable"
        if self.false_win:
            return "false_win"
        if self.false_loss:
            return "false_loss"
        return "correct_feasible" if self.correct_feasible else "correct_infeasible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "coords": self.coords,
            "predicted": self.predicted.to_dict(),
            "measured": self.measured.to_dict(),
            "outcome": self.outcome,
            "comparable": self.comparable,
        }


# --------------------------------------------------------------------------
# the confusion matrix and the asymmetric metrics
# --------------------------------------------------------------------------


@dataclass
class FeasibilityMatrix:
    """Classification outcomes for a feasibility study.

    Every metric here is computable without any continuous quantity: a
    feasibility prediction is a classification, and demanding a regression
    slope before reporting an intersection over union refuses four statistics
    that are perfectly well defined. That refusal was a real defect too.
    """

    cells: list[FeasibilityCell] = field(default_factory=list)
    danger_weight: float = DEFAULT_DANGER_WEIGHT

    # ---- counts ----------------------------------------------------------

    @property
    def comparable(self) -> list[FeasibilityCell]:
        return [c for c in self.cells if c.comparable]

    @property
    def excluded(self) -> list[FeasibilityCell]:
        return [c for c in self.cells if not c.comparable]

    @property
    def n_total(self) -> int:
        return len(self.cells)

    @property
    def n_comparable(self) -> int:
        return len(self.comparable)

    @property
    def n_false_win(self) -> int:
        return sum(1 for c in self.cells if c.false_win)

    @property
    def n_false_loss(self) -> int:
        return sum(1 for c in self.cells if c.false_loss)

    @property
    def n_correct_feasible(self) -> int:
        return sum(1 for c in self.cells if c.correct_feasible)

    @property
    def n_correct_infeasible(self) -> int:
        return sum(1 for c in self.cells if c.correct_infeasible)

    @property
    def n_predicted_feasible(self) -> int:
        return sum(1 for c in self.comparable if c.predicted.is_feasible)

    @property
    def n_measured_feasible(self) -> int:
        return sum(1 for c in self.comparable if c.measured.is_feasible)

    @property
    def n_predicted_infeasible(self) -> int:
        return self.n_comparable - self.n_predicted_feasible

    @property
    def n_measured_infeasible(self) -> int:
        return self.n_comparable - self.n_measured_feasible

    # ---- rates -----------------------------------------------------------

    def _rate(self, n: int, denom: int) -> float:
        return n / denom if denom else float("nan")

    @property
    def accuracy(self) -> float:
        return self._rate(
            self.n_correct_feasible + self.n_correct_infeasible, self.n_comparable
        )

    @property
    def false_win_rate(self) -> float:
        """False wins as a share of what the model *cleared*.

        Denominated on the predictions rather than on the grid, because that is
        the quantity a user of the model experiences: of the configurations it
        told me to run, how many died.
        """
        return self._rate(self.n_false_win, self.n_predicted_feasible)

    @property
    def false_loss_rate(self) -> float:
        return self._rate(self.n_false_loss, self.n_predicted_infeasible)

    @property
    def false_win_area(self) -> float:
        """False wins as a share of the whole comparable grid."""
        return self._rate(self.n_false_win, self.n_comparable)

    @property
    def false_loss_area(self) -> float:
        return self._rate(self.n_false_loss, self.n_comparable)

    @property
    def infeasible_iou(self) -> float:
        """Intersection over union of the predicted and measured infeasible sets.

        Undefined -- not 1.0 -- when neither set contains anything. Two empty
        sets agreeing is not agreement, and returning 1.0 for it is how a study
        that could not have failed reports a pass.
        """
        pred = {c.key for c in self.comparable if not c.predicted.is_feasible}
        meas = {c.key for c in self.comparable if not c.measured.is_feasible}
        union = pred | meas
        if not union:
            return float("nan")
        return len(pred & meas) / len(union)

    @property
    def dangerous_error_score(self) -> float:
        """A single number that refuses to let false wins hide behind accuracy.

        ``(w * false_wins + false_losses) / comparable``, with ``w`` the danger
        weight. Reported alongside the raw counts, never instead of them: the
        weighting is a policy, and a reader who weighs the two errors
        differently needs the inputs.
        """
        if not self.n_comparable:
            return float("nan")
        return (
            self.danger_weight * self.n_false_win + self.n_false_loss
        ) / self.n_comparable

    @property
    def safe(self) -> bool:
        """Whether the model can be used to decide what to launch."""
        return self.n_comparable > 0 and self.n_false_win == 0

    # ---- reporting -------------------------------------------------------

    def confusion(self) -> dict[str, dict[str, int]]:
        return {
            "predicted_feasible": {
                "measured_feasible": self.n_correct_feasible,
                "measured_infeasible": self.n_false_win,
            },
            "predicted_infeasible": {
                "measured_feasible": self.n_false_loss,
                "measured_infeasible": self.n_correct_infeasible,
            },
        }

    def excluded_by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.excluded:
            for side, obs in (("predicted", c.predicted), ("measured", c.measured)):
                if not obs.status.is_definite:
                    out[f"{side}:{obs.status.value}"] = (
                        out.get(f"{side}:{obs.status.value}", 0) + 1
                    )
        return out

    def verdict(self) -> str:
        if not self.n_comparable:
            return (
                "NOT COMPUTED: no cell has a definite answer on both sides, so no "
                "classification metric is defined"
            )
        if self.n_false_win:
            return (
                f"UNSAFE FOR LAUNCH DECISIONS: {self.n_false_win} of "
                f"{self.n_predicted_feasible} configurations the model cleared were "
                f"refused by the hardware ({self.false_win_rate:.0%}). Overall accuracy "
                f"is {self.accuracy:.0%}, which is the number that hides this."
            )
        if self.n_false_loss:
            return (
                f"SAFE BUT CONSERVATIVE: no false wins; {self.n_false_loss} "
                f"configuration(s) the model rejected would in fact have run "
                f"({self.false_loss_rate:.0%} of its rejections)."
            )
        return f"EXACT: the model agrees with the hardware on all {self.n_comparable} cells."

    # ---- boundary displacement -------------------------------------------

    def boundary_displacement(self, axis: str) -> dict[str, Any]:
        """How far the predicted feasibility boundary sits from the measured one.

        Walks each fiber of the grid -- the cells that differ only along
        ``axis`` -- and finds the index at which each map first turns
        infeasible. The difference is the displacement, in levels, and its sign
        says which way the model errs: positive means the model holds on too
        long, which is the direction that produces false wins.

        Reported per axis rather than pooled, because "one level late on micro
        batch" is actionable and "one level late on average" is not.
        """
        rows: dict[tuple, list[FeasibilityCell]] = {}
        for c in self.comparable:
            if axis not in c.coords:
                continue
            key = tuple(sorted((k, v) for k, v in c.coords.items() if k != axis))
            rows.setdefault(key, []).append(c)

        shifts: list[int] = []
        pred_only = meas_only = 0
        for cells in rows.values():
            ordered = sorted(cells, key=lambda c: c.coords[axis])
            p = next((i for i, c in enumerate(ordered)
                      if not c.predicted.is_feasible), None)
            m = next((i for i, c in enumerate(ordered)
                      if not c.measured.is_feasible), None)
            if p is None and m is None:
                continue
            if p is None:
                meas_only += 1
                continue
            if m is None:
                pred_only += 1
                continue
            shifts.append(p - m)

        import statistics

        return {
            "axis": axis,
            "n_fibers_compared": len(shifts),
            "n_fibers_predicted_boundary_only": pred_only,
            "n_fibers_measured_boundary_only": meas_only,
            "mean_signed_shift": statistics.fmean(shifts) if shifts else float("nan"),
            "mean_abs_shift": statistics.fmean([abs(x) for x in shifts])
            if shifts else float("nan"),
            "max_abs_shift": max((abs(x) for x in shifts), default=float("nan")),
            "shifts": shifts,
        }

    def boundary_report(self, axes: Sequence[str]) -> list[dict[str, Any]]:
        return [self.boundary_displacement(a) for a in axes]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeasibilityMatrix:
        """Rebuild from a published artifact, so a report can be regenerated."""
        cells = [
            FeasibilityCell(
                key=tuple(c["key"]) if isinstance(c.get("key"), list) else c.get("key"),
                predicted=FeasibilityObservation.from_dict(c["predicted"]),
                measured=FeasibilityObservation.from_dict(c["measured"]),
                coords=c.get("coords", {}) or {},
            )
            for c in d.get("cells", [])
        ]
        return cls(cells=cells, danger_weight=d.get("danger_weight", DEFAULT_DANGER_WEIGHT))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_comparable": self.n_comparable,
            "n_excluded": len(self.excluded),
            "excluded_by_status": self.excluded_by_status(),
            "confusion": self.confusion(),
            "counts": {
                "correct_feasible": self.n_correct_feasible,
                "correct_infeasible": self.n_correct_infeasible,
                "false_win": self.n_false_win,
                "false_loss": self.n_false_loss,
            },
            "rates": {
                "accuracy": self.accuracy,
                "false_win_rate": self.false_win_rate,
                "false_loss_rate": self.false_loss_rate,
                "false_win_area": self.false_win_area,
                "false_loss_area": self.false_loss_area,
            },
            "infeasible_iou": self.infeasible_iou,
            "dangerous_error_score": self.dangerous_error_score,
            "danger_weight": self.danger_weight,
            "danger_weight_policy": (
                "A false win kills a job launched on the model's advice; a false loss "
                "leaves capacity unused. The weight is a policy choice, not a "
                "measurement, and the raw counts are reported beside it so a reader who "
                "weighs them differently can do so."
            ),
            "safe": self.safe,
            "verdict": self.verdict(),
            "boundary_displacement": self.boundary_report(self._axes()),
            "cells": [c.to_dict() for c in self.cells],
        }

    def _axes(self) -> list[str]:
        seen: list[str] = []
        for c in self.cells:
            for k in c.coords:
                if k not in seen:
                    seen.append(k)
        return seen

    def to_markdown(self) -> str:
        lines = [
            f"**{self.verdict()}**",
            "",
            "| | measured feasible | measured infeasible |",
            "|---|---|---|",
            f"| **predicted feasible** | {self.n_correct_feasible} correct | "
            f"**{self.n_false_win} FALSE WIN** |",
            f"| **predicted infeasible** | {self.n_false_loss} false loss | "
            f"{self.n_correct_infeasible} correct |",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | {self.accuracy:.0%} |",
            f"| False-win count | **{self.n_false_win}** of {self.n_comparable} |",
            f"| False-win rate (of cleared) | {self.false_win_rate:.0%} |",
            f"| False-win area (of grid) | {self.false_win_area:.0%} |",
            f"| False-loss count | {self.n_false_loss} of {self.n_comparable} |",
            f"| False-loss rate (of rejected) | {self.false_loss_rate:.0%} |",
            f"| False-loss area (of grid) | {self.false_loss_area:.0%} |",
            f"| Infeasible-region IoU | {self.infeasible_iou:.2f} |",
            f"| Dangerous-error score (w={self.danger_weight:g}) | "
            f"{self.dangerous_error_score:.2f} |",
            "",
        ]
        bd = [b for b in self.boundary_report(self._axes())
              if b["n_fibers_compared"]]
        if bd:
            lines += ["**Boundary displacement** &mdash; positive means the model "
                      "holds on too long, the direction that produces false wins.", "",
                      "| Axis | Mean signed shift | Max |shift| | Fibers |",
                      "|---|---|---|---|"]
            for b in bd:
                lines.append(
                    f"| `{b['axis']}` | {b['mean_signed_shift']:+.2f} level(s) | "
                    f"{b['max_abs_shift']:.0f} | {b['n_fibers_compared']} |"
                )
            lines.append("")
        excl = self.excluded_by_status()
        if excl:
            lines += [
                f"{len(self.excluded)} cell(s) excluded from every metric because one "
                "side gave no definite answer: "
                + ", ".join(f"`{k}` x{v}" for k, v in sorted(excl.items()))
                + ". They are excluded rather than rounded to the nearest verdict.",
                "",
            ]
        return "\n".join(lines)


def build_matrix(cells: Iterable[FeasibilityCell],
                 danger_weight: float = DEFAULT_DANGER_WEIGHT) -> FeasibilityMatrix:
    return FeasibilityMatrix(cells=list(cells), danger_weight=danger_weight)
