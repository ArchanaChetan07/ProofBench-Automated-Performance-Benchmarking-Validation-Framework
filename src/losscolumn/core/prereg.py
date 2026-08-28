"""Pre-registration: sealing the analysis before the data exists.

Post-hoc selection of the favourable comparison is the failure mode that no
amount of statistical care fixes, because it happens before the statistics. The
defence is ordering: publish the protocol, seal it, and only then collect.

A seal here is a content hash over the *normative* fields of the protocol --
hypotheses, metrics, factor grids, the MDE, the FDR level, the decision rules,
the exclusion rules, the stopping rule -- plus a timestamp and an optional
external anchor (a git tag, an OSF id, a blockchain-style timestamp receipt,
whatever the venue accepts). Editorial fields such as prose rationale are
deliberately excluded from the hash so a typo fix does not break the seal.

Verification then answers three questions a reader actually has:

1. Is this the document that was sealed? (hash match)
2. Was the data collected after it was sealed? (timestamp ordering)
3. Was the analysis the one that was registered? (parameter match, with every
   difference reported as a declared or undeclared deviation)

Undeclared deviations are fatal. Declared ones are not -- research changes --
but they appear in the artifact next to the result they affect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Sequence

from losscolumn.core.provenance import content_hash, utcnow

# Fields that are part of the scientific commitment and therefore hashed.
NORMATIVE_FIELDS = (
    "title",
    "hypotheses",
    "metrics",
    "factors",
    "systems",
    "mde",
    "q_level",
    "alpha",
    "replicates",
    "interleaved",
    "decision_rules",
    "exclusion_rules",
    "stopping_rule",
    "primary_outcome",
    "secondary_outcomes",
    "tuning_budget_trials",
    "search_algorithm",
    "loss_column_policy",
)


@dataclass
class Deviation:
    """A departure from the sealed protocol, with its justification."""

    field: str
    registered: Any
    actual: Any
    reason: str
    declared_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class PreRegistration:
    """The sealed protocol for one thrust."""

    title: str
    hypotheses: dict[str, str]
    metrics: list[dict[str, Any]]
    factors: list[dict[str, Any]]
    systems: list[str]
    mde: float
    q_level: float = 0.05
    alpha: float = 0.05
    replicates: int = 7
    interleaved: bool = True
    decision_rules: list[str] = field(default_factory=list)
    exclusion_rules: list[str] = field(default_factory=list)
    stopping_rule: str = "fixed grid; no interim analysis"
    primary_outcome: str = ""
    secondary_outcomes: list[str] = field(default_factory=list)
    tuning_budget_trials: int | None = None
    search_algorithm: str | None = None
    loss_column_policy: str = (
        "Every region meeting the MDE at the stated FDR level is published in the "
        "loss column of the main artifact, at equal prominence to wins."
    )
    # Non-normative context: editable without breaking the seal.
    rationale: str = ""
    authors: list[str] = field(default_factory=list)
    version: str = "1"
    # Populated by seal().
    sealed_at: str | None = None
    seal_hash: str | None = None
    anchor: str | None = None
    deviations: list[Deviation] = field(default_factory=list)

    # ---- sealing ----------------------------------------------------------

    def normative(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in NORMATIVE_FIELDS}

    def compute_hash(self) -> str:
        return content_hash(self.normative())

    @property
    def is_sealed(self) -> bool:
        return self.seal_hash is not None

    def declare_deviation(self, field_: str, actual: Any, reason: str) -> Deviation:
        d = Deviation(
            field=field_, registered=getattr(self, field_, None), actual=actual, reason=reason
        )
        self.deviations.append(d)
        return d

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "preregistration"
        d["normative_hash"] = self.compute_hash()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreRegistration:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        devs = [Deviation(**x) for x in d.pop("deviations", [])]
        obj = cls(**d)
        obj.deviations = devs
        return obj

    def to_markdown(self) -> str:
        lines = [
            f"# Pre-registration -- {self.title}",
            "",
            f"- Version: `{self.version}`",
            f"- Sealed: `{self.sealed_at or 'NOT SEALED'}`",
            f"- Seal hash: `{self.seal_hash or '-'}`",
            f"- Anchor: `{self.anchor or '-'}`",
            f"- Authors: {', '.join(self.authors) or '-'}",
            "",
            "## Hypotheses",
            "",
        ]
        for k, v in self.hypotheses.items():
            lines.append(f"- **{k}.** {v}")
        lines += [
            "",
            "## Primary outcome",
            "",
            self.primary_outcome or "_unspecified_",
            "",
            "## Design",
            "",
            f"- Systems compared: {', '.join(self.systems)}",
            f"- Replicates per cell: {self.replicates} "
            f"({'interleaved A/B' if self.interleaved else 'blocked'})",
            f"- Minimum effect worth calling (MDE): {self.mde * 100:.1f}%",
            f"- FDR level q: {self.q_level}; interval alpha: {self.alpha}",
            f"- Tuning budget: "
            f"{self.tuning_budget_trials if self.tuning_budget_trials is not None else 'n/a'} "
            f"trials per system via `{self.search_algorithm or 'n/a'}`",
            "",
            "### Factors",
            "",
            "| Factor | Levels | Unit |",
            "|--------|--------|------|",
        ]
        for f_ in self.factors:
            lv = f_.get("levels", [])
            shown = ", ".join(str(x) for x in lv[:8]) + (" ..." if len(lv) > 8 else "")
            lines.append(f"| `{f_.get('name')}` | {shown} | {f_.get('unit') or '-'} |")
        lines += ["", "### Metrics", "", "| Metric | Unit | Direction |", "|--------|------|-----------|"]
        for m in self.metrics:
            lines.append(
                f"| `{m.get('name')}` | {m.get('unit')} | "
                f"{'higher is better' if m.get('higher_is_better') else 'lower is better'} |"
            )
        lines += ["", "## Decision rules", ""]
        lines += [f"{i}. {r}" for i, r in enumerate(self.decision_rules, 1)] or ["_none stated_"]
        lines += ["", "## Exclusion rules", ""]
        lines += [f"- {r}" for r in self.exclusion_rules] or ["_none stated_"]
        lines += ["", "## Stopping rule", "", self.stopping_rule, ""]
        lines += ["## Loss-column policy", "", self.loss_column_policy, ""]
        if self.rationale:
            lines += ["## Rationale (non-normative)", "", self.rationale, ""]
        if self.deviations:
            lines += ["## Declared deviations", "",
                      "| Field | Registered | Actual | Reason | Declared |",
                      "|-------|-----------|--------|--------|----------|"]
            for d in self.deviations:
                lines.append(
                    f"| `{d.field}` | `{d.registered}` | `{d.actual}` | {d.reason} | {d.declared_at} |"
                )
        return "\n".join(lines)


def seal(prereg: PreRegistration, *, anchor: str | None = None) -> PreRegistration:
    """Stamp the protocol. Re-sealing an already-sealed document is refused."""
    if prereg.is_sealed:
        raise RuntimeError(
            "protocol is already sealed; publish a new version with an explicit "
            "deviation record instead of resealing"
        )
    prereg.sealed_at = utcnow()
    prereg.seal_hash = prereg.compute_hash()
    prereg.anchor = anchor
    return prereg


@dataclass
class VerificationResult:
    ok: bool
    findings: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, severity: str, message: str, **detail: Any) -> None:
        self.findings.append(
            {"code": code, "severity": severity, "message": message, "detail": detail}
        )
        if severity == "fatal":
            self.ok = False

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "findings": self.findings}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def verify(
    prereg: PreRegistration,
    *,
    analysis: dict[str, Any] | None = None,
    collected_at: str | Sequence[str] | None = None,
) -> VerificationResult:
    """Check a result set against the protocol that was supposed to govern it."""
    res = VerificationResult(ok=True)

    if not prereg.is_sealed:
        res.add("PRE-001", "fatal", "protocol was never sealed; ordering cannot be established")
        return res

    if prereg.compute_hash() != prereg.seal_hash:
        res.add(
            "PRE-002",
            "fatal",
            "normative fields do not match the seal: the protocol was edited after sealing",
            recomputed=prereg.compute_hash(),
            sealed=prereg.seal_hash,
        )

    sealed_dt = _parse(prereg.sealed_at)
    stamps = [collected_at] if isinstance(collected_at, str) else list(collected_at or [])
    for ts in stamps:
        dt = _parse(ts)
        if dt is None:
            res.add("PRE-003", "warning", f"unparseable collection timestamp {ts!r}")
        elif sealed_dt and dt < sealed_dt:
            res.add(
                "PRE-004",
                "fatal",
                f"data collected at {ts} predates the seal at {prereg.sealed_at}",
                collected_at=ts,
            )

    if analysis:
        declared = {d.field for d in prereg.deviations}
        for key in ("mde", "q_level", "alpha", "replicates", "interleaved"):
            if key in analysis and analysis[key] != getattr(prereg, key):
                sev = "warning" if key in declared else "fatal"
                res.add(
                    "PRE-005",
                    sev,
                    f"analysis used {key}={analysis[key]!r} but the protocol registered "
                    f"{getattr(prereg, key)!r}"
                    + (" (declared deviation)" if key in declared else " (UNDECLARED)"),
                    field=key,
                )
        reg_systems = set(prereg.systems)
        got = set(analysis.get("systems", []) or [])
        if got and not got <= reg_systems:
            res.add(
                "PRE-006",
                "fatal" if "systems" not in declared else "warning",
                f"analysis covers systems not in the protocol: {sorted(got - reg_systems)}",
            )
        if got and reg_systems - got:
            res.add(
                "PRE-007",
                "warning",
                f"registered systems missing from the analysis: {sorted(reg_systems - got)}; "
                "a system that was registered and then dropped must be accounted for",
            )
        reg_factors = {f["name"]: set(map(str, f.get("levels", []))) for f in prereg.factors}
        for f_ in analysis.get("factors", []) or []:
            name = f_.get("name")
            lv = set(map(str, f_.get("levels", [])))
            if name in reg_factors and not lv <= reg_factors[name]:
                res.add(
                    "PRE-008",
                    "warning",
                    f"factor `{name}` was analysed at levels outside the registered grid: "
                    f"{sorted(lv - reg_factors[name])}",
                )
            if name in reg_factors and reg_factors[name] - lv:
                res.add(
                    "PRE-009",
                    "warning" if "factors" in declared else "fatal",
                    f"registered levels of `{name}` were not analysed: "
                    f"{sorted(reg_factors[name] - lv)}. Dropping levels after seeing data is "
                    "the selection this protocol exists to prevent.",
                )
    return res
