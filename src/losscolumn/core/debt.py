"""Coverage debt: what is missing, why, and what would actually fix it.

A readiness verdict of NOT READY is a true statement and a useless one on its
own. The question it leaves open is the only one worth acting on: *what would
have to change*, and is that a measurement problem, a model problem, or a
hardware problem?

Those three have different remedies and very different costs:

**noise-limited** -- the measurement is too variable for any model to be graded
against. More repeats or a quieter machine. No modelling work will help.

**model-limited** -- the measurement is clean and no available family fits.
Either a family that can express the shape, or an admission that the surface is
not modellable at this granularity. More repeats will not help.

**evidence-limited** -- too few points in a regime to fit or validate. More
sizes, in that regime specifically.

Separating them stops the most expensive mistake in this kind of work, which is
spending a week on a better model for a surface whose measurement noise already
exceeds the error you are trying to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["DebtKind", "DebtItem", "CoverageDebt", "assess_debt"]


class DebtKind(str, Enum):
    GROUP_MODEL_REJECTED = "group_model_rejected"
    """No model was accepted for this cell's group, so the cell cannot be covered
    whatever its own measurement quality.

    Kept separate from `model_limited` because the constraint is not local. The
    selection rule scores a family on its worst regime, by design -- it exists so
    that one useless regime cannot hide behind three good ones. The consequence
    is that a single ungradable regime rejects the family for the whole group,
    and cells that are cleanly measured and well fitted are then uncovered for a
    reason that has nothing to do with them. Reporting those as noise-limited
    would send someone to buy repeats that cannot help."""

    NOISE_LIMITED = "noise_limited"
    MODEL_LIMITED = "model_limited"
    EVIDENCE_LIMITED = "evidence_limited"
    NOT_MEASURED = "not_measured"
    UNKNOWN = "unknown"

    @property
    def remedy(self) -> str:
        return {
            DebtKind.GROUP_MODEL_REJECTED:
                "nothing local to this cell. Either a family that fits the "
                "group's worst regime, or a decision about what the worst-regime "
                "rule should do when a regime is too noisy to grade at all -- "
                "which is a question about the gate, to be settled deliberately "
                "rather than by widening it",
            DebtKind.NOISE_LIMITED:
                "more repeats, or a quieter machine. No modelling work will help: "
                "the residual is already below the measurement's own variation",
            DebtKind.MODEL_LIMITED:
                "a model family that can express this shape, or an admission that "
                "the surface is not modellable at this granularity. More repeats "
                "will not help: the measurement is already clean",
            DebtKind.EVIDENCE_LIMITED:
                "more message sizes in this regime specifically, so the family can "
                "be fitted and then validated there",
            DebtKind.NOT_MEASURED: "measure it",
            DebtKind.UNKNOWN: "diagnose before spending",
        }[self]


@dataclass
class DebtItem:
    """One uncovered cell, classified by what is actually blocking it."""

    group: str
    regime: str
    kind: DebtKind
    status: str = ""
    heldout_err: float = float("nan")
    noise_cv: float = float("nan")
    n_points: int = 0
    n_validation_points: int = 0
    gap: float = float("nan")
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group, "regime": self.regime, "kind": self.kind.value,
            "status": self.status, "heldout_err": self.heldout_err,
            "noise_cv": self.noise_cv, "n_points": self.n_points,
            "n_validation_points": self.n_validation_points,
            "gap": self.gap, "rationale": self.rationale,
            "remedy": self.kind.remedy,
        }


@dataclass
class CoverageDebt:
    """The whole ledger, and what each kind of debt would cost to clear."""

    items: list[DebtItem] = field(default_factory=list)
    n_required_cells: int = 0
    n_covered_cells: int = 0
    max_err: float = 0.15
    max_noise: float = 0.20
    notes: list[str] = field(default_factory=list)

    def by_kind(self) -> dict[str, list[DebtItem]]:
        out: dict[str, list[DebtItem]] = {}
        for i in self.items:
            out.setdefault(i.kind.value, []).append(i)
        return out

    def by_group(self) -> dict[str, list[DebtItem]]:
        out: dict[str, list[DebtItem]] = {}
        for i in self.items:
            out.setdefault(i.group, []).append(i)
        return out

    @property
    def dominant_kind(self) -> DebtKind:
        counts = {k: len(v) for k, v in self.by_kind().items()}
        if not counts:
            return DebtKind.UNKNOWN
        return DebtKind(max(counts, key=lambda k: counts[k]))

    def next_experiment(self) -> str:
        """The single most discriminating thing to do next.

        Deliberately one thing. A list of everything that could be improved is
        how a campaign turns into an open-ended project; the point of
        classifying the debt is to identify the measurement that would change a
        verdict.
        """
        kinds = self.by_kind()
        if kinds.get("model_limited"):
            worst = min(kinds["model_limited"], key=lambda i: i.noise_cv)
            return (
                f"{worst.group}, {worst.regime} regime: measurement noise is "
                f"{worst.noise_cv:.1%} while the best model misses by "
                f"{worst.heldout_err:.1%}. The gap is the model's, not the "
                "instrument's, so the discriminating experiment is to add message "
                "sizes in that regime and re-fit: if a richer segmentation then "
                "clears the gate, the surface has more structure than the current "
                "families express; if it does not, the surface is not modellable "
                "at this granularity and that is the finding."
            )
        if kinds.get("noise_limited"):
            worst = max(kinds["noise_limited"], key=lambda i: i.noise_cv)
            return (
                f"{worst.group}, {worst.regime} regime: the residual "
                f"({worst.heldout_err:.1%}) is at or below the measurement's own "
                f"variation ({worst.noise_cv:.1%}). No model can do better than the "
                "instrument, so the discriminating experiment is more repeats there "
                "-- if the noise falls and the error follows, it was noise; if the "
                "error stays, it was never noise-limited."
            )
        if kinds.get("evidence_limited"):
            worst = min(kinds["evidence_limited"], key=lambda i: i.n_points)
            return (
                f"{worst.group}, {worst.regime} regime: only {worst.n_points} valid "
                "point(s). Nothing can be concluded until there are enough to fit "
                "and validate separately."
            )
        return "no debt: every required cell is covered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_required_cells": self.n_required_cells,
            "n_covered_cells": self.n_covered_cells,
            "n_debt_items": len(self.items),
            "by_kind": {k: len(v) for k, v in self.by_kind().items()},
            "dominant_kind": self.dominant_kind.value,
            "next_experiment": self.next_experiment(),
            "items": [i.to_dict() for i in self.items],
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        kinds = self.by_kind()
        lines = [
            f"**Coverage debt: {len(self.items)} uncovered cell(s) of "
            f"{self.n_required_cells}.**",
            "",
            "| Kind | Cells | Remedy |",
            "|---|---|---|",
        ]
        for k, items in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| `{k}` | {len(items)} | {DebtKind(k).remedy} |")
        lines += ["", "| Group | Regime | Kind | held-out | noise | gap | why |",
                  "|---|---|---|---|---|---|---|"]
        for i in sorted(self.items, key=lambda x: (x.group, x.regime)):
            err = f"{i.heldout_err:.1%}" if i.heldout_err == i.heldout_err else "—"
            cv = f"{i.noise_cv:.1%}" if i.noise_cv == i.noise_cv else "—"
            gap = f"{i.gap:+.1%}" if i.gap == i.gap else "—"
            lines.append(
                f"| {i.group} | {i.regime} | `{i.kind.value}` | {err} | {cv} | "
                f"{gap} | {i.rationale} |"
            )
        lines += ["", "**The single most discriminating next experiment**", "",
                  self.next_experiment(), ""]
        if self.notes:
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


# WITHDRAWN. A two-sample CV was corrected by a single factor of 1/0.545,
# measured against the sentinel's seven repeats. A pre-registered re-measurement
# of the same grid at twenty-one repeats falsified it: the recorded CV went to
# 19.0% where 10.7% was predicted.
#
# The factor was not merely mis-estimated. It cannot exist. Within that one
# session the excess splits into 2.25x from the estimator and a further 1.35x
# from timescale -- twenty-one repeats span more wall-clock than two adjacent
# ones and see slower variation. So the quantity a correction is meant to
# recover grows with the window it is measured over, and the factor depends on
# an arbitrary choice of reference: 1.84x against seven repeats, 2.25x against
# twenty-one.
#
# Preserved at artifacts/history/2026-08-31-two-sample-cv-correction. The
# remedy is not a better factor; it is to measure with enough repeats that no
# correction is needed, and to record how many were used.
CV_N2_BIAS_WITHDRAWN = 0.545
"""Kept only so the withdrawal can be read against the number it withdraws."""

MIN_TRUSTWORTHY_REPEATS = 5
"""Below this a recorded CV is not usable as a noise estimate.

Not a correction, a refusal. Two adjacent repeats measure how much one call
differs from its neighbour, which is a different quantity from how much the
call varies over a run -- and it is the second that a coverage gate is asking
about. There is no factor that turns the first into the second.
"""


class UntrustworthyNoiseEstimate(ValueError):
    """Raised when a debt would be classified from too few repeats.

    An exception rather than a warning, because the failure mode is silent and
    expensive: an understated CV makes a cell look cleanly measured, a cleanly
    measured cell with a large residual is classified model-limited, and
    somebody is then sent to build a model family for a surface whose instrument
    cannot resolve the error they are chasing. That is exactly what happened
    here, to six of sixteen cells.
    """


def _worst_regime(group: Any) -> tuple[str, float] | None:
    """The noisiest regime in a group: the one most likely to have sunk it."""
    best = None
    for reg, rc in getattr(group, "regimes", {}).items():
        cv = getattr(rc, "noise_cv", float("nan"))
        if cv == cv and (best is None or cv > best[1]):
            best = (reg, cv)
    return best


def assess_debt(coverage: Any, selections: dict[str, Any], *,
                max_err: float = 0.15, max_noise: float = 0.20,
                min_points: int = 3, cv_from_n: int = 0,
                allow_untrustworthy: bool = False) -> CoverageDebt:
    """Classify every uncovered cell by what is actually blocking it.

    The classification turns on comparing the best available model's error in
    that regime against the measurement's own variation there. A residual at or
    below the noise is the instrument's; a residual well above it, on a clean
    measurement, is the model's.

    `cv_from_n` is the number of repeats each recorded CV was computed from.
    Fewer than `MIN_TRUSTWORTHY_REPEATS` and this refuses to classify, because a
    CV from two adjacent repeats measures a different quantity from the one the
    gate asks about and no correction converts between them.
    """
    if cv_from_n and cv_from_n < MIN_TRUSTWORTHY_REPEATS and not allow_untrustworthy:
        raise UntrustworthyNoiseEstimate(
            f"the recorded CVs come from {cv_from_n} repeat(s), below the "
            f"{MIN_TRUSTWORTHY_REPEATS} needed for a usable noise estimate. Two "
            "adjacent repeats measure how much a call differs from its neighbour, "
            "not how much it varies over a run, and classifying a debt from the "
            "first while reading it as the second is what produced the ledger "
            "preserved at artifacts/history/2026-08-31-two-sample-cv-correction. "
            "Re-measure with more repeats, or pass allow_untrustworthy=True and "
            "say so in the report."
        )
    debt = CoverageDebt(
        n_required_cells=coverage.n_required_cells,
        n_covered_cells=coverage.n_covered_cells,
        max_err=max_err, max_noise=max_noise,
    )
    if cv_from_n:
        debt.notes.append(
            f"Recorded CVs come from {cv_from_n} repeats per point, and are used "
            "as measured. No correction is applied: an earlier ledger corrected a "
            "two-repeat CV by a single factor and a pre-registered re-measurement "
            "falsified it."
            + ("" if cv_from_n >= MIN_TRUSTWORTHY_REPEATS else
               " These are below the trustworthy threshold and were admitted "
               "explicitly.")
        )

    for g in coverage.groups.values():
        sel = selections.get(g.key, {})
        best_per_regime: dict[str, float] = {}
        for c in sel.get("candidates", []):
            for reg, err in (c.get("heldout_per_tier_err") or {}).items():
                if err == err and (reg not in best_per_regime or err < best_per_regime[reg]):
                    best_per_regime[reg] = err

        for reg, rc in g.regimes.items():
            if rc.covered:
                continue
            err = rc.heldout_err
            if err != err:
                err = best_per_regime.get(reg, float("nan"))
            cv = rc.noise_cv
            item = DebtItem(
                group=g.key, regime=reg, kind=DebtKind.UNKNOWN,
                status=rc.status.value, heldout_err=err, noise_cv=cv,
                n_points=rc.n_points, n_validation_points=rc.n_validation_points,
            )
            if err == err:
                item.gap = err - max_err

            # Status first. A cell in a group with no accepted model is
            # uncovered for that reason, and grading it on some rejected
            # candidate's error describes a model nobody is allowed to use.
            if rc.status.value == "model_rejected":
                item.kind = DebtKind.GROUP_MODEL_REJECTED
                blocker = _worst_regime(g)
                item.rationale = (
                    "the group has no accepted model"
                    + (f", blocked by its {blocker[0]} regime ({blocker[1]:.1%} "
                       f"run-to-run variation)" if blocker else "")
                    + f"; this cell's own measurement is {cv:.1%}"
                    if cv == cv else "; this cell's own measurement quality is unknown"
                )
                debt.items.append(item)
                continue
            if rc.n_points == 0:
                item.kind = DebtKind.NOT_MEASURED
                item.rationale = "no valid measurement in this regime"
            elif rc.n_points < min_points or rc.n_validation_points == 0:
                item.kind = DebtKind.EVIDENCE_LIMITED
                item.rationale = (
                    f"{rc.n_points} valid point(s), {rc.n_validation_points} of them "
                    "held out: too few to fit and validate separately"
                )
            elif cv == cv and cv > max_noise:
                item.kind = DebtKind.NOISE_LIMITED
                item.rationale = (
                    f"run-to-run variation {cv:.1%} exceeds the {max_noise:.0%} the "
                    "protocol allows; nothing can be graded here"
                )
            elif err == err and cv == cv and err <= cv * 1.5:
                item.kind = DebtKind.NOISE_LIMITED
                item.rationale = (
                    f"the best model misses by {err:.1%} against run-to-run variation "
                    f"of {cv:.1%}; the residual is the instrument's, not the model's"
                )
            elif err == err:
                item.kind = DebtKind.MODEL_LIMITED
                item.rationale = (
                    f"the measurement is clean ({cv:.1%}) and the best available "
                    f"family still misses by {err:.1%}: the gap is the model's"
                )
            else:
                item.rationale = "no model could be fitted; cause not established"
            debt.items.append(item)
    return debt
