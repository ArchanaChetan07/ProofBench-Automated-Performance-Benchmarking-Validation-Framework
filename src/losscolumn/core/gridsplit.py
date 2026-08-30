"""Calibration and validation grids, kept apart by construction.

A model fitted on the cells it is then graded by has been fitted, not
validated, and its apparent accuracy is circular. Keeping the two sets apart is
easy to intend and easy to forget, so this module makes forgetting raise rather
than merely be regrettable:

* the split is decided **once**, before any measurement, and sealed by a hash;
* asking for a validation cell while the model is unfrozen raises;
* fitting a parameter after the model is frozen raises;
* a cell that appears in both sets is an error at construction time.

Also here: the preflight. Measuring a grid is expensive, and a grid that cannot
produce a meaningful answer is expensive and worthless. The preflight asks the
model, which is free, what it expects to find -- and says whether the study can
be informative at all before anything is run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.provenance import content_hash, utcnow

__all__ = ["GridSplit", "SplitViolation", "PreflightReport", "preflight", "select_boundary_grid"]


class SplitViolation(RuntimeError):
    """Raised when calibration and validation data are about to be confused."""


@dataclass
class GridSplit:
    """A frozen partition of cells into calibration and validation.

    ``frozen`` is the state change that matters. Before it, calibration cells
    may be read and parameters fitted; validation cells may not be touched.
    After it, the model is fixed and validation cells become readable. There is
    no state in which both are true.
    """

    calibration: tuple[Any, ...]
    validation: tuple[Any, ...]
    rationale: dict[Any, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    frozen: bool = False
    frozen_at: str | None = None
    _fitted: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        cal, val = set(map(_key, self.calibration)), set(map(_key, self.validation))
        overlap = cal & val
        if overlap:
            raise SplitViolation(
                f"{len(overlap)} cell(s) appear in both the calibration and the "
                f"validation grid: {sorted(overlap)[:5]}. A cell used to choose a "
                "parameter cannot also be evidence that the parameter was right."
            )
        if not cal:
            raise SplitViolation("the calibration grid is empty")
        if not val:
            raise SplitViolation(
                "the validation grid is empty; there would be nothing to validate on"
            )

    # ---- membership -------------------------------------------------------

    def is_calibration(self, cell: Any) -> bool:
        return _key(cell) in {_key(c) for c in self.calibration}

    def is_validation(self, cell: Any) -> bool:
        return _key(cell) in {_key(c) for c in self.validation}

    def check_readable(self, cell: Any, *, purpose: str) -> None:
        """Raise if reading this cell for this purpose would break the split."""
        if purpose == "calibration":
            if self.is_validation(cell):
                raise SplitViolation(
                    f"cell {cell!r} belongs to the validation grid and was requested "
                    "for calibration. Fitting on it would make the later validation "
                    "circular."
                )
        elif purpose == "validation":
            if not self.frozen:
                raise SplitViolation(
                    f"cell {cell!r} was requested for validation, but the model is not "
                    "frozen. Reading validation data before freezing is how a "
                    "validation set quietly becomes a training set."
                )
            if self.is_calibration(cell):
                raise SplitViolation(
                    f"cell {cell!r} belongs to the calibration grid and was requested "
                    "for validation. It has already informed the model."
                )
        else:
            raise ValueError(f"unknown purpose {purpose!r}")

    # ---- freezing ---------------------------------------------------------

    def record_fit(self, parameter: str) -> None:
        """Record that a parameter was fitted. Raises once the model is frozen."""
        if self.frozen:
            raise SplitViolation(
                f"parameter {parameter!r} was fitted after the model was frozen. "
                "Anything learned after freezing has necessarily come from the "
                "validation data, which is precisely what freezing forbids."
            )
        self._fitted.append(parameter)

    def freeze(self) -> GridSplit:
        if self.frozen:
            return self
        self.frozen = True
        self.frozen_at = utcnow()
        return self

    # ---- identity ---------------------------------------------------------

    def digest(self) -> str:
        return content_hash({
            "calibration": sorted(_key(c) for c in self.calibration),
            "validation": sorted(_key(c) for c in self.validation),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "frozen": self.frozen,
            "frozen_at": self.frozen_at,
            "digest": self.digest(),
            "n_calibration": len(self.calibration),
            "n_validation": len(self.validation),
            "fitted_parameters": list(self._fitted),
            "calibration": [list(c) for c in self.calibration],
            "validation": [list(c) for c in self.validation],
            "rationale": {str(k): v for k, v in self.rationale.items()},
        }

    def to_markdown(self) -> str:
        return "\n".join([
            f"**Grid split** `{self.digest()[7:23]}...` &mdash; "
            f"{len(self.calibration)} calibration cells, "
            f"{len(self.validation)} validation cells, "
            f"{'FROZEN ' + (self.frozen_at or '') if self.frozen else 'not yet frozen'}.",
            "",
            "The validation cells are unreadable until the model is frozen, and the "
            "calibration cells are excluded from validation. Both are enforced, not "
            "merely intended.",
        ])


def _key(cell: Any) -> tuple:
    if isinstance(cell, tuple):
        return cell
    if isinstance(cell, list):
        return tuple(cell)
    return (cell,)


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


@dataclass
class PreflightReport:
    """What the model expects to find, before anything is measured."""

    n_cells: int = 0
    n_predicted_feasible: int = 0
    n_predicted_infeasible: int = 0
    n_boundary: int = 0
    boundary_threshold: float = 0.0
    informative: bool = False
    reasons: list[str] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)
    label: str = ""

    @property
    def can_classify(self) -> bool:
        """Whether classification metrics could come out either way.

        A grid on which the model predicts the same thing everywhere can still
        produce a confusion matrix -- measurement may disagree -- but it can
        never produce an infeasible-region overlap or a boundary displacement,
        because the predicted set has no boundary. Both facts are reported.
        """
        return self.n_predicted_feasible > 0 and self.n_predicted_infeasible > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_cells": self.n_cells,
            "n_predicted_feasible": self.n_predicted_feasible,
            "n_predicted_infeasible": self.n_predicted_infeasible,
            "n_boundary": self.n_boundary,
            "boundary_threshold": self.boundary_threshold,
            "informative": self.informative,
            "can_classify": self.can_classify,
            "reasons": self.reasons,
            "cells": self.cells,
        }

    def to_markdown(self) -> str:
        head = "informative" if self.informative else "NOT INFORMATIVE"
        lines = [
            f"**Preflight -- {self.label}: {head}**",
            "",
            f"- {self.n_cells} cells: {self.n_predicted_feasible} predicted feasible, "
            f"{self.n_predicted_infeasible} predicted infeasible.",
            f"- {self.n_boundary} within {self.boundary_threshold:.0%} of the decision "
            "boundary, where being slightly wrong changes the answer.",
            f"- Predicted set has a boundary: {'yes' if self.can_classify else 'no'}.",
        ]
        if self.reasons:
            lines += [""] + [f"- {r}" for r in self.reasons]
        return "\n".join(lines)


def preflight(
    cells: Sequence[Any],
    predict_fits: Callable[[Any], bool],
    boundary_distance: Callable[[Any], float],
    *,
    label: str = "grid",
    boundary_threshold: float = 0.25,
    describe: Callable[[Any], dict[str, Any]] | None = None,
) -> PreflightReport:
    """Ask the model what it expects, before spending measurement time.

    Refuses only when a metric is genuinely impossible. A grid where the model
    predicts everything feasible is still worth measuring -- the measurement can
    disagree, and a confusion matrix with false wins in it is exactly the
    finding that matters. What it cannot produce is a *predicted* infeasible
    region, so overlap and boundary displacement are unavailable, and the
    report says which metrics survive rather than refusing wholesale.
    """
    rep = PreflightReport(n_cells=len(cells), boundary_threshold=boundary_threshold,
                          label=label)
    for c in cells:
        fits = bool(predict_fits(c))
        dist = float(boundary_distance(c))
        near = dist <= boundary_threshold
        rep.n_predicted_feasible += int(fits)
        rep.n_predicted_infeasible += int(not fits)
        rep.n_boundary += int(near)
        entry: dict[str, Any] = {
            "cell": list(_key(c)), "predicted_feasible": fits,
            "boundary_distance": dist, "near_boundary": near,
        }
        if describe is not None:
            entry.update(describe(c))
        rep.cells.append(entry)

    if not cells:
        rep.reasons.append("the grid is empty")
    if rep.n_predicted_infeasible == 0:
        rep.reasons.append(
            "the model predicts every cell feasible, so there is no predicted "
            "infeasible region: overlap and boundary displacement will be undefined. "
            "The confusion matrix, accuracy and false-win rate remain available, and "
            "a false win here would still be a finding"
        )
    if rep.n_predicted_feasible == 0:
        rep.reasons.append(
            "the model predicts every cell infeasible, so a false win is impossible "
            "by construction and the grid cannot test the dangerous direction"
        )
    if rep.n_boundary == 0:
        rep.reasons.append(
            f"no cell lies within {boundary_threshold:.0%} of the decision boundary; "
            "the grid tests only regimes where the answer is obvious"
        )
    # Informative means: the measurement could come out in more than one way,
    # and at least one cell is somewhere the model could plausibly be wrong.
    rep.informative = bool(cells) and rep.n_boundary > 0
    return rep


def select_boundary_grid(
    candidates: Sequence[Any],
    boundary_distance: Callable[[Any], float],
    predict_fits: Callable[[Any], bool],
    *,
    n_boundary: int = 6,
    n_clear_feasible: int = 2,
    n_clear_infeasible: int = 2,
    exclude: Sequence[Any] = (),
) -> tuple[list[Any], dict[Any, str]]:
    """Choose cells around the predicted boundary, plus anchors on each side.

    Not the largest Cartesian product: most of a big grid sits deep inside a
    region where the answer is obvious and would be obvious under almost any
    parameters. What discriminates between models is the neighbourhood of
    ``predicted_memory ~ available_memory``. Anchors on both sides are kept so
    that accuracy and the confusion matrix are not computed entirely on the
    hardest cells.
    """
    banned = {_key(c) for c in exclude}
    pool = [c for c in candidates if _key(c) not in banned]
    ranked = sorted(pool, key=boundary_distance)

    chosen: dict[Any, str] = {}
    for c in ranked[:n_boundary]:
        chosen[c] = f"boundary: |log(required/budget)| = {boundary_distance(c):.3f}"

    feasible = [c for c in reversed(ranked) if predict_fits(c) and c not in chosen]
    for c in feasible[:n_clear_feasible]:
        chosen[c] = "anchor: predicted feasible with margin"

    infeasible = [c for c in reversed(ranked) if not predict_fits(c) and c not in chosen]
    for c in infeasible[:n_clear_infeasible]:
        chosen[c] = "anchor: predicted infeasible with margin"

    return list(chosen), chosen
