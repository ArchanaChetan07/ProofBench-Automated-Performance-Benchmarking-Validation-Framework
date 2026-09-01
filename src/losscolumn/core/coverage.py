"""Three different questions that a single word keeps collapsing.

The communication study reached a state where one group out of six had an
accepted model, and nothing in the code distinguished that from a calibrated
subsystem. The distinction is not pedantic: acting on it means renting eight
A100s.

So the three questions are separate types here, and none of them is
convertible into another:

**Parameter validity** -- did *this* fitted model pass its quality gate?
Answered per model. ``ACCEPTED``, ``DIAGNOSTIC`` or ``REJECTED``.

**Model coverage** -- which parts of the required operating surface do the
accepted models actually speak for? Answered per group and per regime. A group
with an accepted model that was never validated in one regime is not covered in
that regime, and extrapolating into it is not coverage either.

**Subsystem readiness** -- is enough of the preregistered surface covered to
support the work downstream? Answered once, for the whole subsystem, and only
by conjunction: every required group, every required regime, accepted models
only, validation complete. There is deliberately no way to express "mostly
ready".

The asymmetry that motivates all of this: an over-strict readiness verdict
costs another local afternoon, and an over-permissive one costs an eight-GPU
allocation spent debugging instead of measuring.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ParameterVerdict",
    "RegimeStatus",
    "CoverageVerdict",
    "ReadinessVerdict",
    "RegimeCoverage",
    "GroupCoverage",
    "CommunicationCoverage",
]


class ParameterVerdict(str, Enum):
    """What a quality gate decided about one fitted model."""

    ACCEPTED = "accepted"
    DIAGNOSTIC = "diagnostic"
    REJECTED = "rejected"

    @property
    def usable(self) -> bool:
        return self is ParameterVerdict.ACCEPTED


class RegimeStatus(str, Enum):
    """Why one regime of one group is or is not spoken for."""

    COVERED = "covered"
    NOT_MEASURED = "not_measured"
    NOT_VALIDATED = "not_validated"
    """Measured, but no held-out point in this regime graded the model here."""
    EXTRAPOLATED = "extrapolated"
    """The accepted model reaches this regime only by extending past its data."""
    MODEL_REJECTED = "model_rejected"
    ERROR_TOO_HIGH = "error_too_high"
    TOO_NOISY = "too_noisy"
    INSUFFICIENT_POINTS = "insufficient_points"

    @property
    def covered(self) -> bool:
        return self is RegimeStatus.COVERED


class CoverageVerdict(str, Enum):
    COVERED = "covered"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    NOT_MEASURED = "not_measured"


class ReadinessVerdict(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"

    @property
    def ready(self) -> bool:
        return self is ReadinessVerdict.READY


@dataclass
class RegimeCoverage:
    """One (group, regime) cell of the coverage matrix."""

    regime: str
    status: RegimeStatus = RegimeStatus.NOT_MEASURED
    n_points: int = 0
    n_validation_points: int = 0
    heldout_err: float = float("nan")
    noise_cv: float = float("nan")
    detail: str = ""

    @property
    def covered(self) -> bool:
        return self.status.covered

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime, "status": self.status.value,
            "covered": self.covered, "n_points": self.n_points,
            "n_validation_points": self.n_validation_points,
            "heldout_err": self.heldout_err, "noise_cv": self.noise_cv,
            "detail": self.detail,
        }


@dataclass
class GroupCoverage:
    """One required communication group, and every regime it must speak for."""

    collective: str
    world: int
    parameter_verdict: ParameterVerdict = ParameterVerdict.REJECTED
    model_family: str = ""
    estimator: str = ""
    n_segments: int = 0
    worst_regime_err: float = float("nan")
    regimes: dict[str, RegimeCoverage] = field(default_factory=dict)
    detail: str = ""

    @property
    def key(self) -> str:
        return f"{self.collective}/world{self.world}"

    @property
    def verdict(self) -> CoverageVerdict:
        if not self.regimes:
            return CoverageVerdict.NOT_MEASURED
        if all(r.covered for r in self.regimes.values()):
            return CoverageVerdict.COVERED
        if all(r.status is RegimeStatus.NOT_MEASURED for r in self.regimes.values()):
            return CoverageVerdict.NOT_MEASURED
        return CoverageVerdict.INSUFFICIENT_COVERAGE

    @property
    def covered(self) -> bool:
        return self.verdict is CoverageVerdict.COVERED

    def uncovered(self) -> list[RegimeCoverage]:
        return [r for r in self.regimes.values() if not r.covered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "collective": self.collective, "world": self.world,
            "parameter_verdict": self.parameter_verdict.value,
            "coverage_verdict": self.verdict.value,
            "covered": self.covered,
            "model_family": self.model_family, "estimator": self.estimator,
            "n_segments": self.n_segments,
            "worst_regime_err": self.worst_regime_err,
            "regimes": {k: v.to_dict() for k, v in self.regimes.items()},
            "detail": self.detail,
        }


@dataclass
class CommunicationCoverage:
    """The coverage matrix, and the one readiness verdict it supports."""

    required_collectives: tuple[str, ...]
    required_worlds: tuple[int, ...]
    required_regimes: tuple[str, ...]
    groups: dict[str, GroupCoverage] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    stability: dict[str, Any] | None = None
    """The stability envelope, when one has been measured.

    A plain dictionary rather than an imported type, so that `coverage` does not
    depend on the measurement layer: a reader grading someone else's artifact
    has the envelope as JSON and nothing else.
    """

    # ---- construction -----------------------------------------------------

    @classmethod
    def empty(cls, collectives: Iterable[str], worlds: Iterable[int],
              regimes: Iterable[str]) -> CommunicationCoverage:
        cov = cls(tuple(collectives), tuple(worlds), tuple(regimes))
        for c in cov.required_collectives:
            for w in cov.required_worlds:
                g = GroupCoverage(collective=c, world=w)
                for r in cov.required_regimes:
                    g.regimes[r] = RegimeCoverage(regime=r)
                cov.groups[g.key] = g
        return cov

    # ---- the three questions ---------------------------------------------

    @property
    def n_required_groups(self) -> int:
        return len(self.required_collectives) * len(self.required_worlds)

    @property
    def accepted_groups(self) -> list[GroupCoverage]:
        """Parameter validity. Says nothing about coverage."""
        return [g for g in self.groups.values() if g.parameter_verdict.usable]

    @property
    def covered_groups(self) -> list[GroupCoverage]:
        """Model coverage. A superset question to parameter validity."""
        return [g for g in self.groups.values() if g.covered]

    @property
    def n_required_cells(self) -> int:
        return self.n_required_groups * len(self.required_regimes)

    @property
    def n_covered_cells(self) -> int:
        return sum(
            1 for g in self.groups.values() for r in g.regimes.values() if r.covered
        )

    @property
    def coverage_fraction(self) -> float:
        return self.n_covered_cells / self.n_required_cells if self.n_required_cells else 0.0

    def readiness(self) -> tuple[ReadinessVerdict, list[str]]:
        """Subsystem readiness. Conjunctive, and deliberately unforgiving.

        Returns the verdict and the list of blocking reasons. A partial result
        is never READY: there is no arithmetic here that trades a well-covered
        group against a missing one, because the work downstream needs the
        whole surface and would fail on the part that is missing.
        """
        blocking: list[str] = []

        # LC-1.2. Added after the sentinel established that no two measurement
        # sessions on this machine have been shown comparable. This does not
        # invalidate evidence already gathered inside one session; it says that
        # the debt cannot be paid down by measuring more in a second one.
        if self.stability is not None and not self.stability.get("pooling_permitted"):
            kind = self.stability.get("drift_kind", "undetermined")
            n_ok = self.stability.get("n_poolable_pairs")
            n_all = self.stability.get("n_session_pairs")
            blocking.append(
                f"the machine does not permit cross-session pooling (drift is "
                f"{kind}"
                + (f"; {n_ok} of {n_all} session pairs poolable" if n_all else "")
                + "), so coverage cannot be extended by measuring again later: any "
                "new evidence has to be gathered in the same session as everything "
                "it will be compared against"
            )

        missing_groups = [g for g in self.groups.values() if not g.covered]
        if missing_groups:
            blocking.append(
                f"{len(missing_groups)} of {self.n_required_groups} required groups "
                f"are not covered: "
                + ", ".join(f"{g.key} ({g.verdict.value})" for g in missing_groups[:6])
            )

        unaccepted = [g for g in self.groups.values() if not g.parameter_verdict.usable]
        if unaccepted:
            blocking.append(
                f"{len(unaccepted)} group(s) have no accepted model: "
                + ", ".join(g.key for g in unaccepted[:6])
            )

        uncovered_cells = [
            (g.key, r) for g in self.groups.values() for r in g.uncovered()
        ]
        if uncovered_cells:
            by_reason: dict[str, int] = {}
            for _, r in uncovered_cells:
                by_reason[r.status.value] = by_reason.get(r.status.value, 0) + 1
            blocking.append(
                f"{len(uncovered_cells)} of {self.n_required_cells} "
                f"(group x regime) cells are uncovered: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(by_reason.items()))
            )

        never_validated = [
            (g.key, r.regime) for g in self.groups.values()
            for r in g.regimes.values() if r.n_validation_points == 0
        ]
        if never_validated:
            blocking.append(
                f"{len(never_validated)} cell(s) have no held-out validation point, so "
                "no independent evidence grades the model there"
            )

        return (
            (ReadinessVerdict.NOT_READY, blocking) if blocking
            else (ReadinessVerdict.READY, [])
        )

    # ---- reporting --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        verdict, blocking = self.readiness()
        return {
            "required_collectives": list(self.required_collectives),
            "required_worlds": list(self.required_worlds),
            "required_regimes": list(self.required_regimes),
            "n_required_groups": self.n_required_groups,
            "n_required_cells": self.n_required_cells,
            "parameter_validity": {
                "n_accepted": len(self.accepted_groups),
                "accepted": [g.key for g in self.accepted_groups],
            },
            "model_coverage": {
                "n_covered_groups": len(self.covered_groups),
                "n_covered_cells": self.n_covered_cells,
                "coverage_fraction": self.coverage_fraction,
                "covered": [g.key for g in self.covered_groups],
            },
            "subsystem_readiness": {
                "verdict": verdict.value,
                "ready": verdict.ready,
                "blocking": blocking,
            },
            "groups": {k: g.to_dict() for k, g in self.groups.items()},
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        verdict, blocking = self.readiness()
        head = "READY" if verdict.ready else "NOT READY"
        lines = [
            f"**Subsystem readiness: {head}**",
            "",
            f"Parameter validity: {len(self.accepted_groups)} of "
            f"{self.n_required_groups} groups have an accepted model. "
            f"Model coverage: {self.n_covered_cells} of {self.n_required_cells} "
            f"(group x regime) cells ({self.coverage_fraction:.0%}). "
            "These are different numbers and neither implies readiness on its own.",
            "",
            "| Collective | World | "
            + " | ".join(r for r in self.required_regimes)
            + " | Model | Verdict | Covered |",
            "|---" * (len(self.required_regimes) + 5) + "|",
        ]
        for c in self.required_collectives:
            for w in self.required_worlds:
                g = self.groups.get(f"{c}/world{w}")
                if g is None:
                    continue
                cells = []
                for r in self.required_regimes:
                    rc = g.regimes.get(r)
                    if rc is None:
                        cells.append("—")
                    elif rc.covered:
                        cells.append(f"{rc.heldout_err:.0%}" if rc.heldout_err == rc.heldout_err else "ok")
                    else:
                        cells.append(f"_{rc.status.value}_")
                model = f"`{g.model_family}`" if g.model_family else "—"
                lines.append(
                    f"| {c} | {w} | " + " | ".join(cells)
                    + f" | {model} | {g.parameter_verdict.value} | "
                    + ("yes" if g.covered else "**no**") + " |"
                )
        lines.append("")
        if blocking:
            lines += ["**Blocking gates**", ""]
            lines += [f"- {b}" for b in blocking]
            lines.append("")
        if self.notes:
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)
