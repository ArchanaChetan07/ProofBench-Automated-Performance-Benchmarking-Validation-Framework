"""Tuning-budget parity: the control that makes an engine comparison mean anything.

The untuned-baseline problem is not solved by promising to be fair. It is
solved by an append-only ledger that records, for every system, how many
configurations were tried, from what space, by which search procedure, under
whose hands, on what hardware -- and by a certificate that refuses to validate
when those do not match.

Two grades of parity are distinguished, because they are not the same claim:

**Nominal parity.** Equal trial counts, same search procedure, same operator,
same hardware. Checkable, and checked here.

**Effective parity.** Equal trial counts *relative to the difficulty of each
search space*. Forty trials over a 3-dimensional space is a dense search;
forty over a 9-dimensional one is a sample. The certificate reports trials per
dimension and fraction of space covered so a reader can see when nominal parity
flatters one engine, and flags the gap rather than pretending to fix it.

Operator skill remains a residual confound. It is recorded, not eliminated;
the proposal says as much, and so does the certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from losscolumn.core.provenance import Provenance, content_hash, utcnow


@dataclass
class SearchSpace:
    """The set of knobs a system was allowed to be tuned over."""

    name: str
    dims: dict[str, list[Any]]           # knob -> candidate values
    notes: str | None = None

    @property
    def n_dims(self) -> int:
        return len(self.dims)

    @property
    def cardinality(self) -> int:
        n = 1
        for v in self.dims.values():
            n *= max(len(v), 1)
        return n

    def digest(self) -> str:
        return content_hash({"dims": {k: list(v) for k, v in sorted(self.dims.items())}})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dims": self.dims,
            "n_dims": self.n_dims,
            "cardinality": self.cardinality,
            "digest": self.digest(),
            "notes": self.notes,
        }


@dataclass
class Trial:
    """One configuration evaluated during tuning."""

    system: str
    index: int
    config: dict[str, Any]
    objective: float                      # higher is better, by convention
    outcome: str = "ok"                   # ok | oom | error | timeout
    wall_time_s: float = 0.0
    started_at: str = field(default_factory=utcnow)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TuningLedger:
    """Append-only record of a tuning campaign for one system."""

    system: str
    space: SearchSpace
    search_algorithm: str
    search_seed: int
    operator: str
    hardware_fingerprint: str
    workload_digest: str
    objective_name: str
    group: str = ""          # e.g. the workload this campaign was run against
    trials: list[Trial] = field(default_factory=list)
    budget_trials: int = 0
    started_at: str = field(default_factory=utcnow)
    finished_at: str | None = None

    def record(self, trial: Trial) -> None:
        if self.finished_at is not None:
            raise RuntimeError(
                f"ledger for {self.system!r} is sealed; a trial recorded after sealing "
                "would break the parity guarantee"
            )
        if trial.index != len(self.trials):
            raise ValueError(
                f"out-of-order trial {trial.index} (expected {len(self.trials)}); "
                "the ledger is append-only by design"
            )
        self.trials.append(trial)

    def seal(self) -> None:
        self.finished_at = utcnow()

    # ---- derived quantities ----------------------------------------------

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_ok(self) -> int:
        return sum(1 for t in self.trials if t.outcome == "ok")

    @property
    def wall_time_s(self) -> float:
        return float(sum(t.wall_time_s for t in self.trials))

    @property
    def trials_per_dim(self) -> float:
        return self.n_trials / self.space.n_dims if self.space.n_dims else float("inf")

    @property
    def space_coverage(self) -> float:
        seen = {content_hash(t.config) for t in self.trials}
        return len(seen) / self.space.cardinality if self.space.cardinality else 0.0

    def best(self) -> Trial | None:
        ok = [t for t in self.trials if t.outcome == "ok" and math.isfinite(t.objective)]
        return max(ok, key=lambda t: t.objective, default=None)

    def improvement_curve(self) -> list[float]:
        """Best-so-far objective after each trial: the honest picture of whether
        the budget was enough, or whether the search was still climbing when it
        was cut off."""
        out: list[float] = []
        best = -math.inf
        for t in self.trials:
            if t.outcome == "ok" and math.isfinite(t.objective):
                best = max(best, t.objective)
            out.append(best)
        return out

    def still_improving(self, tail: int = 10, rel: float = 0.01) -> bool:
        """True if the last ``tail`` trials still gained more than ``rel``.

        A search that was still improving when the budget ran out did not find
        the system's optimum, and the resulting comparison understates it. This
        flag appears in the certificate for every system.
        """
        curve = self.improvement_curve()
        if len(curve) <= tail or not math.isfinite(curve[-1]) or curve[-1] <= 0:
            return False
        prior = curve[-tail - 1]
        if not math.isfinite(prior) or prior <= 0:
            return True
        return (curve[-1] / prior - 1.0) > rel

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "space": self.space.to_dict(),
            "search_algorithm": self.search_algorithm,
            "search_seed": self.search_seed,
            "operator": self.operator,
            "hardware_fingerprint": self.hardware_fingerprint,
            "workload_digest": self.workload_digest,
            "objective_name": self.objective_name,
            "group": self.group,
            "budget_trials": self.budget_trials,
            "n_trials": self.n_trials,
            "n_ok": self.n_ok,
            "wall_time_s": self.wall_time_s,
            "trials_per_dim": self.trials_per_dim,
            "space_coverage": self.space_coverage,
            "still_improving": self.still_improving(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "best": self.best().to_dict() if self.best() else None,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass
class ParityViolation:
    code: str
    severity: str                          # fatal | warning
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ParityCertificate:
    """The artifact that says whether a multi-system comparison was fair."""

    ledgers: list[TuningLedger]
    violations: list[ParityViolation] = field(default_factory=list)
    issued_at: str = field(default_factory=utcnow)
    provenance: Provenance | None = None

    @property
    def ok(self) -> bool:
        return not any(v.severity == "fatal" for v in self.violations)

    @property
    def systems(self) -> list[str]:
        return [l.system for l in self.ledgers]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "parity_certificate",
            "issued_at": self.issued_at,
            "ok": self.ok,
            "systems": self.systems,
            "ledgers": [l.to_dict() for l in self.ledgers],
            "violations": [v.to_dict() for v in self.violations],
            "digest": content_hash([l.to_dict() for l in self.ledgers]),
        }

    def to_markdown(self) -> str:
        lines = [
            "### Tuning-budget parity certificate",
            "",
            f"Status: **{'PARITY HELD' if self.ok else 'PARITY VIOLATED'}** "
            f"({len(self.violations)} finding(s))",
            "",
            "| System | Group | Trials | ok | Space (dims / size) | Trials/dim | Coverage | "
            "Tuning wall-clock | Still improving |",
            "|--------|-------|--------|----|---------------------|------------|----------|"
            "-------------------|-----------------|",
        ]
        for l in self.ledgers:
            lines.append(
                f"| {l.system} | {l.group or '-'} | {l.n_trials} | {l.n_ok} | "
                f"{l.space.n_dims} / {l.space.cardinality} | {l.trials_per_dim:.1f} | "
                f"{l.space_coverage:.1%} | {l.wall_time_s / 60:.1f} min | "
                f"{'YES -- budget was binding' if l.still_improving() else 'no'} |"
            )
        if self.violations:
            lines += ["", "| Severity | Code | Finding |", "|----------|------|---------|"]
            for v in self.violations:
                lines.append(f"| {v.severity} | `{v.code}` | {v.message} |")
        return "\n".join(lines)


def certify(ledgers: Sequence[TuningLedger], *, tolerance_trials: int = 0) -> ParityCertificate:
    """Check nominal parity across systems and report effective-parity risks."""
    cert = ParityCertificate(ledgers=list(ledgers))
    v = cert.violations
    if len(ledgers) < 2:
        v.append(ParityViolation("PAR-000", "fatal", "parity requires at least two systems"))
        return cert

    def _spread(name: str, vals: Iterable[Any]) -> set:
        return set(vals)

    counts = {l.system: l.n_trials for l in ledgers}
    if max(counts.values()) - min(counts.values()) > tolerance_trials:
        v.append(
            ParityViolation(
                "PAR-001",
                "fatal",
                f"unequal trial counts: {counts}. Equal budgets are the whole control.",
                {"counts": counts},
            )
        )

    algos = _spread("algo", (l.search_algorithm for l in ledgers))
    if len(algos) > 1:
        v.append(
            ParityViolation(
                "PAR-002", "fatal", f"different search procedures used: {sorted(algos)}",
                {"algorithms": sorted(algos)},
            )
        )

    ops = _spread("operator", (l.operator for l in ledgers))
    if len(ops) > 1:
        v.append(
            ParityViolation(
                "PAR-003",
                "warning",
                f"different operators tuned different systems ({sorted(ops)}); operator "
                "skill is a documented, uneliminated confound",
                {"operators": sorted(ops)},
            )
        )

    hw = _spread("hw", (l.hardware_fingerprint for l in ledgers))
    if len(hw) > 1:
        v.append(
            ParityViolation(
                "PAR-004", "fatal",
                "systems were tuned on different hardware; results are not comparable",
                {"fingerprints": sorted(hw)},
            )
        )

    wl = _spread("workload", (l.workload_digest for l in ledgers))
    if len(wl) > 1:
        v.append(
            ParityViolation(
                "PAR-005", "fatal", "systems were tuned against different workload traces",
                {"workloads": sorted(wl)},
            )
        )

    objs = _spread("obj", (l.objective_name for l in ledgers))
    if len(objs) > 1:
        v.append(
            ParityViolation(
                "PAR-006", "fatal", f"systems were tuned against different objectives: {sorted(objs)}"
            )
        )

    for l in ledgers:
        if l.finished_at is None:
            v.append(
                ParityViolation(
                    "PAR-007", "fatal", f"ledger for {l.system} was never sealed", {"system": l.system}
                )
            )
        if l.still_improving():
            v.append(
                ParityViolation(
                    "PAR-008",
                    "warning",
                    f"{l.system} was still improving when its budget ran out; its measured "
                    "performance is a lower bound and the comparison understates it",
                    {"system": l.system},
                )
            )
        if l.n_ok < 0.5 * max(l.n_trials, 1):
            v.append(
                ParityViolation(
                    "PAR-009",
                    "warning",
                    f"{l.system} completed only {l.n_ok}/{l.n_trials} trials; the effective "
                    "budget was smaller than the nominal one",
                    {"system": l.system},
                )
            )

    tpd = {l.system: l.trials_per_dim for l in ledgers}
    if tpd and min(tpd.values()) > 0 and max(tpd.values()) / min(tpd.values()) > 2.0:
        v.append(
            ParityViolation(
                "PAR-010",
                "warning",
                f"nominal parity but unequal search difficulty: trials per dimension {tpd}. "
                "Equal budgets do not mean equal search density.",
                {"trials_per_dim": tpd},
            )
        )

    cov = {l.system: l.space_coverage for l in ledgers}
    if cov and min(cov.values()) > 0 and max(cov.values()) / min(cov.values()) > 2.0:
        v.append(
            ParityViolation(
                "PAR-011",
                "warning",
                "equal trial counts explored unequal fractions of each search space: "
                + ", ".join(f"{k} {x:.1%}" for k, x in sorted(cov.items()))
                + ". The engine with the larger space received a sparser search from the "
                "same budget, which understates it.",
                {"space_coverage": cov},
            )
        )

    wall = {l.system: l.wall_time_s for l in ledgers}
    if wall and min(wall.values()) > 0 and max(wall.values()) / min(wall.values()) > 3.0:
        v.append(
            ParityViolation(
                "PAR-012",
                "warning",
                "equal trials cost very unequal wall-clock: "
                + ", ".join(f"{k} {x / 60:.0f} min" for k, x in sorted(wall.items()))
                + ". Under a fixed time budget rather than a fixed trial budget the "
                "comparison would come out differently, and a reader choosing under time "
                "pressure should know which budget was held constant.",
                {"wall_time_s": wall},
            )
        )
    return cert


def certify_grouped(
    groups: dict[str, Sequence[TuningLedger]], *, tolerance_trials: int = 0
) -> ParityCertificate:
    """Certify parity within each group, then merge into one certificate.

    Tuning happens per workload, so ledgers for different workloads are not
    comparable to each other -- checking them together would trip the
    same-workload rule for a design that is correct. Each group is certified on
    its own and the findings are merged, tagged with the group they came from.
    """
    all_ledgers: list[TuningLedger] = []
    violations: list[ParityViolation] = []
    for name, leds in sorted(groups.items()):
        sub = certify(leds, tolerance_trials=tolerance_trials)
        all_ledgers += list(leds)
        for viol in sub.violations:
            violations.append(
                ParityViolation(
                    code=viol.code,
                    severity=viol.severity,
                    message=f"[{name}] {viol.message}",
                    detail={**viol.detail, "group": name},
                )
            )
    # Deduplicate findings that repeat identically across every group: a
    # structural asymmetry stated once is clearer than the same sentence four
    # times, and burying the reader is its own kind of non-reporting.
    seen: dict[tuple[str, str], int] = {}
    merged: list[ParityViolation] = []
    for viol in violations:
        key = (viol.code, viol.message.split("] ", 1)[-1])
        seen[key] = seen.get(key, 0) + 1
    emitted: set[tuple[str, str]] = set()
    for viol in violations:
        key = (viol.code, viol.message.split("] ", 1)[-1])
        if seen[key] == len(groups) and len(groups) > 1:
            if key in emitted:
                continue
            emitted.add(key)
            merged.append(
                ParityViolation(
                    code=viol.code,
                    severity=viol.severity,
                    message=f"[all {len(groups)} groups] {key[1]}",
                    detail=viol.detail,
                )
            )
        else:
            merged.append(viol)
    return ParityCertificate(ledgers=all_ledgers, violations=merged)
