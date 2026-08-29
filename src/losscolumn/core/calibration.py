"""Comparing a predicted loss map against a measured one.

A simulated thrust produces a claim of a particular kind: *if the model is
right, here is where the method loses*. That is a prediction, and a prediction
is only worth something once somebody has checked one. This module is the
check.

It takes the model's predicted comparisons and the measured comparisons over
the same cells and asks six questions, which are the six the request named:

* **Region overlap** -- do the predicted and measured loss regions cover the
  same cells?
* **Boundary error** -- where they disagree, by how many levels along which
  axis is the model's boundary displaced?
* **False win regions** -- cells the model called a win that measurement calls
  a loss. These are the dangerous errors: the model told you to ship something
  that regresses.
* **False loss regions** -- cells the model called a loss that measurement
  clears. Wasteful rather than dangerous, but still wrong.
* **Cliff agreement** -- did the model put its discontinuities where the
  hardware puts them?
* **Calibration error** -- on the continuous effect, not just the verdict: is
  the model biased, and does it exaggerate or understate?

The last is the one that feeds back. A slope below one means the model
overstates every effect by a constant factor, which is a correctable parameter
error rather than a wrong model, and :func:`suggest_update` says by how much.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.envelope import Cell, Envelope
from losscolumn.core.stats import CellComparison

# Verdicts that count as "the method is worse here".
_LOSS = ("loss",)
_WIN = ("win",)


@dataclass
class BoundaryShift:
    """How far the predicted loss boundary sits from the measured one, per axis."""

    axis: str
    n_fibers_compared: int = 0
    n_fibers_predicted_only: int = 0
    n_fibers_measured_only: int = 0
    mean_abs_shift: float = float("nan")
    mean_signed_shift: float = float("nan")
    shifts: list[int] = field(default_factory=list)

    def describe(self) -> str:
        if not math.isfinite(self.mean_abs_shift):
            return f"{self.axis}: no fiber has a boundary in both maps"
        direction = "early" if self.mean_signed_shift < 0 else "late"
        return (
            f"{self.axis}: the model puts the boundary {abs(self.mean_signed_shift):.2f} "
            f"level(s) too {direction} on average "
            f"(|shift| {self.mean_abs_shift:.2f} over {self.n_fibers_compared} fiber(s))"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "n_fibers_compared": self.n_fibers_compared,
            "n_fibers_predicted_only": self.n_fibers_predicted_only,
            "n_fibers_measured_only": self.n_fibers_measured_only,
            "mean_abs_shift": self.mean_abs_shift,
            "mean_signed_shift": self.mean_signed_shift,
            "shifts": self.shifts,
        }


@dataclass
class CalibrationResult:
    """What a calibration study establishes about a model's reliability."""

    n_cells: int = 0
    n_compared: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    agreement: float = float("nan")
    kappa: float = float("nan")

    region_iou: float = float("nan")
    n_predicted_loss: int = 0
    n_measured_loss: int = 0
    false_loss_cells: list[dict[str, Any]] = field(default_factory=list)
    false_win_cells: list[dict[str, Any]] = field(default_factory=list)
    boundary: dict[str, BoundaryShift] = field(default_factory=dict)

    cliff_precision: float = float("nan")
    cliff_recall: float = float("nan")
    n_cliffs_predicted: int = 0
    n_cliffs_measured: int = 0

    bias_pct: float = float("nan")
    mae_pct: float = float("nan")
    rmse_pct: float = float("nan")
    slope: float = float("nan")
    intercept_pct: float = float("nan")
    r_squared: float = float("nan")
    within_mde: float = float("nan")

    mde: float = 0.05
    notes: list[str] = field(default_factory=list)

    @property
    def n_false_win(self) -> int:
        return len(self.false_win_cells)

    @property
    def n_false_loss(self) -> int:
        return len(self.false_loss_cells)

    def verdict(self) -> str:
        """A one-line statement of how far the model can be trusted."""
        if not math.isfinite(self.rmse_pct):
            return "no overlapping cells: the model has not been calibrated at all"
        if self.n_false_win:
            return (
                f"NOT SAFE FOR DECISIONS: the model calls {self.n_false_win} cell(s) a win "
                f"that measurement calls a loss. Its typical error is {self.rmse_pct:.1f} "
                "percentage points."
            )
        if self.rmse_pct <= self.mde * 100:
            return (
                f"USABLE FOR RANKING: no false wins, and a typical error of "
                f"{self.rmse_pct:.1f} percentage points, inside the {self.mde:.0%} MDE."
            )
        return (
            f"DIRECTIONAL ONLY: no false wins, but a typical error of {self.rmse_pct:.1f} "
            f"percentage points exceeds the {self.mde:.0%} MDE, so the model orders "
            "configurations without sizing the gaps."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_compared": self.n_compared,
            "confusion": self.confusion,
            "agreement": self.agreement,
            "kappa": self.kappa,
            "region_overlap": {
                "iou": self.region_iou,
                "n_predicted_loss": self.n_predicted_loss,
                "n_measured_loss": self.n_measured_loss,
                "n_false_loss": self.n_false_loss,
                "n_false_win": self.n_false_win,
                "false_loss_cells": self.false_loss_cells,
                "false_win_cells": self.false_win_cells,
            },
            "boundary_error": {k: v.to_dict() for k, v in self.boundary.items()},
            "cliffs": {
                "precision": self.cliff_precision,
                "recall": self.cliff_recall,
                "n_predicted": self.n_cliffs_predicted,
                "n_measured": self.n_cliffs_measured,
            },
            "calibration_error": {
                "bias_pct": self.bias_pct,
                "mae_pct": self.mae_pct,
                "rmse_pct": self.rmse_pct,
                "slope": self.slope,
                "intercept_pct": self.intercept_pct,
                "r_squared": self.r_squared,
                "fraction_within_mde": self.within_mde,
            },
            "verdict": self.verdict(),
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "### Calibration -- predicted against measured",
            "",
            f"**{self.verdict()}**",
            "",
            f"- {self.n_compared} of {self.n_cells} cells measured and compared.",
            f"- Verdict agreement {self.agreement:.0%} (Cohen's kappa {self.kappa:.2f}).",
            f"- Loss-region overlap (IoU) {self.region_iou:.2f}: "
            f"{self.n_predicted_loss} predicted, {self.n_measured_loss} measured.",
            f"- **{self.n_false_win} false win(s)** -- predicted a win, measured a loss.",
            f"- {self.n_false_loss} false loss(es) -- predicted a loss, measurement cleared it.",
            "",
            "| Calibration statistic | Value | Reading |",
            "|---|---|---|",
            f"| Bias | {self.bias_pct:+.2f} pp | "
            f"{'model is optimistic' if self.bias_pct < 0 else 'model is pessimistic'} |",
            f"| Mean absolute error | {self.mae_pct:.2f} pp | typical miss |",
            f"| RMSE | {self.rmse_pct:.2f} pp | miss including the tails |",
            f"| Slope | {self.slope:.2f} | "
            f"{'exaggerates effects' if self.slope < 0.9 else 'understates effects' if self.slope > 1.1 else 'scales correctly'} |",
            f"| R-squared | {self.r_squared:.2f} | share of variation the model tracks |",
            f"| Within MDE | {self.within_mde:.0%} | cells predicted to within {self.mde:.0%} |",
            "",
        ]
        if self.boundary:
            lines += ["**Boundary displacement**", ""]
            lines += [f"- {b.describe()}" for b in self.boundary.values()]
            lines.append("")
        if math.isfinite(self.cliff_recall):
            lines += [
                f"**Cliffs**: {self.cliff_recall:.0%} of measured discontinuities were "
                f"predicted; {self.cliff_precision:.0%} of predicted ones are real "
                f"({self.n_cliffs_predicted} predicted, {self.n_cliffs_measured} measured).",
                "",
            ]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _bucket(v: str) -> str:
    """Collapse verdicts to the three states a decision turns on."""
    if v in _LOSS:
        return "loss"
    if v in _WIN:
        return "win"
    return "neither"


def _kappa(conf: dict[str, dict[str, int]], labels: Sequence[str]) -> float:
    total = sum(conf[a][b] for a in labels for b in labels)
    if total == 0:
        return float("nan")
    po = sum(conf[a][a] for a in labels) / total
    pe = sum(
        (sum(conf[a].values()) / total) * (sum(conf[x][a] for x in labels) / total)
        for a in labels
    )
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boundary_error(
    env: Envelope, predicted_loss: set[Cell], measured_loss: set[Cell], axis: str
) -> BoundaryShift:
    """How many levels the predicted loss boundary is displaced along one axis.

    Walks each 1-D fiber, finds the index where each map first flips into a
    loss, and reports the difference. This is the statistic a practitioner
    actually wants: not "the model was wrong somewhere" but "the model puts the
    cliff one sequence length too early".
    """
    out = BoundaryShift(axis=axis)
    for _fixed, base in env.fibers(axis):
        cells = env.fiber_cells(axis, base)
        p = next((i for i, c in enumerate(cells) if c in predicted_loss), None)
        m = next((i for i, c in enumerate(cells) if c in measured_loss), None)
        if p is None and m is None:
            continue
        if p is None:
            out.n_fibers_measured_only += 1
            continue
        if m is None:
            out.n_fibers_predicted_only += 1
            continue
        out.shifts.append(p - m)
        out.n_fibers_compared += 1
    if out.shifts:
        arr = np.array(out.shifts, dtype=float)
        out.mean_abs_shift = float(np.mean(np.abs(arr)))
        out.mean_signed_shift = float(np.mean(arr))
    return out


def calibrate(
    envelope: Envelope,
    predicted: Sequence[CellComparison],
    measured: Sequence[CellComparison],
    *,
    mde: float = 0.05,
    predicted_cliffs: Sequence[Any] = (),
    measured_cliffs: Sequence[Any] = (),
) -> CalibrationResult:
    """Score a model's predicted loss map against measurement.

    Only cells present and conclusive in *both* maps are compared. A cell the
    model predicted but nobody measured is not evidence either way, and
    silently treating it as agreement would inflate every statistic here.
    """
    pred = {tuple(c.cell): c for c in predicted}
    meas = {tuple(c.cell): c for c in measured}
    shared = sorted(set(pred) & set(meas))

    res = CalibrationResult(n_cells=len(pred), n_compared=len(shared), mde=mde)
    if not shared:
        res.notes.append(
            "no cell was both predicted and measured, so nothing here is calibrated"
        )
        return res

    labels = ("loss", "win", "neither")
    conf = {a: dict.fromkeys(labels, 0) for a in labels}
    for cell in shared:
        conf[_bucket(pred[cell].verdict)][_bucket(meas[cell].verdict)] += 1
    res.confusion = conf
    res.agreement = sum(conf[a][a] for a in labels) / len(shared)
    res.kappa = _kappa(conf, labels)

    p_loss = {c for c in shared if _bucket(pred[c].verdict) == "loss"}
    m_loss = {c for c in shared if _bucket(meas[c].verdict) == "loss"}
    res.n_predicted_loss, res.n_measured_loss = len(p_loss), len(m_loss)
    union = p_loss | m_loss
    res.region_iou = len(p_loss & m_loss) / len(union) if union else 1.0

    for cell in sorted(m_loss - p_loss):
        # The model said this was safe. It is not. These are the errors that
        # cost something, so they are listed individually rather than counted.
        res.false_win_cells.append({
            "cell": envelope.label(cell),
            "coords": envelope.coords(cell),
            "predicted_pct": (math.exp(pred[cell].effect) - 1) * 100,
            "measured_pct": (math.exp(meas[cell].effect) - 1) * 100,
            "predicted_verdict": pred[cell].verdict,
        })
    for cell in sorted(p_loss - m_loss):
        res.false_loss_cells.append({
            "cell": envelope.label(cell),
            "coords": envelope.coords(cell),
            "predicted_pct": (math.exp(pred[cell].effect) - 1) * 100,
            "measured_pct": (math.exp(meas[cell].effect) - 1) * 100,
            "measured_verdict": meas[cell].verdict,
        })

    for f in envelope.factors:
        if len(f.levels) >= 2:
            res.boundary[f.name] = boundary_error(envelope, p_loss, m_loss, f.name)

    # Continuous calibration, on the log-ratio effect rather than the verdict.
    # A model can get every verdict right and still be useless for sizing, and
    # a model can get verdicts wrong near the threshold while tracking the
    # effect well; these are different failures and are reported separately.
    x = np.array([pred[c].effect for c in shared], dtype=float)
    y = np.array([meas[c].effect for c in shared], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size >= 2:
        err_pct = (np.exp(y) - np.exp(x)) * 100
        res.bias_pct = float(np.mean(err_pct))
        res.mae_pct = float(np.mean(np.abs(err_pct)))
        res.rmse_pct = float(np.sqrt(np.mean(err_pct**2)))
        res.within_mde = float(np.mean(np.abs(err_pct) <= mde * 100))
        if np.ptp(x) > 1e-12:
            slope, intercept = np.polyfit(x, y, 1)
            res.slope, res.intercept_pct = float(slope), float((math.exp(intercept) - 1) * 100)
            pred_y = slope * x + intercept
            ss_res = float(np.sum((y - pred_y) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            res.r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
        else:
            res.notes.append(
                "the predicted effects have no spread, so no slope can be fitted: "
                "the model is being checked at a single operating point"
            )

    if predicted_cliffs or measured_cliffs:
        pc = {(tuple(c.from_cell), tuple(c.to_cell)) for c in predicted_cliffs}
        mc = {(tuple(c.from_cell), tuple(c.to_cell)) for c in measured_cliffs}
        res.n_cliffs_predicted, res.n_cliffs_measured = len(pc), len(mc)
        res.cliff_precision = len(pc & mc) / len(pc) if pc else float("nan")
        res.cliff_recall = len(pc & mc) / len(mc) if mc else float("nan")

    return res


def suggest_update(res: CalibrationResult, *, parameter: str = "achievable_mfu",
                   current: float | None = None) -> dict[str, Any]:
    """Turn a calibration result into a concrete parameter correction.

    A slope away from one is the signature of a scale error in a single
    parameter rather than a wrong model, and a scale error is correctable. This
    reports the correction and, deliberately, refuses to apply it: the update
    belongs in a new sealed protocol, because a model retuned against the same
    measurements it is then validated on has been fitted, not calibrated.
    """
    out: dict[str, Any] = {
        "parameter": parameter,
        "current": current,
        "slope": res.slope,
        "bias_pct": res.bias_pct,
        "applied": False,
        "why_not_applied": (
            "A model refitted on the measurements used to validate it is fitted, not "
            "calibrated. The correction is recorded here and takes effect only in a "
            "newly sealed protocol, evaluated against cells this study did not touch."
        ),
    }
    if math.isfinite(res.slope) and res.slope > 0:
        out["multiplicative_correction"] = 1.0 / res.slope
        if current is not None:
            out["suggested"] = current / res.slope
    out["reading"] = (
        "the model tracks the measured effects but overstates their size"
        if math.isfinite(res.slope) and res.slope < 0.9
        else "the model understates the measured effects"
        if math.isfinite(res.slope) and res.slope > 1.1
        else "the model scales correctly; residual error is not a scale error"
    )
    return out
