"""The reporting standard, as a program.

Section 5 of the proposal lists five requirements. A prose requirement that
nobody can mechanically check is a preference; this module turns each one into
a rule with an identifier, a severity, and a test that runs against a claim
document produced by any tooling, not just this package.

    LC-1  Loss column            -- forecloses selective reporting
    LC-2  Tuning-budget parity   -- forecloses the untuned baseline
    LC-3  Envelope, not point    -- forecloses overgeneralisation
    LC-4  Pre-registration       -- forecloses post-hoc selection
    LC-5  One-command reproduction -- forecloses unfalsifiability by inaccessibility

Rules numbered LC-Qn are quality checks. They never block conformance on their
own, but they are printed in the same table, because a conforming artifact
whose measurement noise exceeds its claimed effect is conforming and useless.

Grades:

``conforming``                 no fatal findings, no warnings
``conforming-with-warnings``   no fatal findings
``non-conforming``             at least one fatal finding
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from losscolumn.spec import registry
from losscolumn.version import STANDARD_VERSION

Severity = str  # "fatal" | "warning" | "info"


@dataclass
class Finding:
    rule: str
    requirement: str
    severity: Severity
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ConformanceReport:
    claim_id: str
    standard_version: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed and f.severity == "fatal"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed and f.severity == "warning"]

    @property
    def not_applicable(self) -> list[Finding]:
        return [f for f in self.findings if f.detail.get("not_applicable")]

    @property
    def checked(self) -> list[Finding]:
        """Rules that actually applied to this claim."""
        return [f for f in self.findings if not f.detail.get("not_applicable")]

    @property
    def grade(self) -> str:
        if self.failures:
            return "non-conforming"
        return "conforming-with-warnings" if self.warnings else "conforming"

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "conformance_report",
            "claim_id": self.claim_id,
            "standard_version": self.standard_version,
            "grade": self.grade,
            "standard_digest": registry.compute_digest(),
            "n_rules_checked": len(self.checked),
            "n_not_applicable": len(self.not_applicable),
            "n_fatal": len(self.failures),
            "n_warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_markdown(self) -> str:
        badge = {
            "conforming": "PASS",
            "conforming-with-warnings": "PASS (with warnings)",
            "non-conforming": "FAIL",
        }[self.grade]
        lines = [
            f"### Conformance -- `{self.claim_id}` against {self.standard_version}",
            "",
            f"**{badge}** -- {len(self.failures)} fatal, {len(self.warnings)} warning(s).",
            "",
            "| Rule | Requirement | Result | Finding |",
            "|------|-------------|--------|---------|",
        ]
        for f in self.findings:
            if f.detail.get("not_applicable"):
                mark = "n/a"
            else:
                mark = "pass" if f.passed else ("FAIL" if f.severity == "fatal" else "warn")
            lines.append(f"| `{f.rule}` | {f.requirement} | {mark} | {f.message} |")
        return "\n".join(lines)


# --------------------------------------------------------------------------


class Validator:
    """Grade a claim document against the standard."""

    def __init__(self, *, min_cells: int = 8, min_replicates: int = 5,
                 max_noise_to_effect: float = 0.5) -> None:
        self.min_cells = min_cells
        self.min_replicates = min_replicates
        self.max_noise_to_effect = max_noise_to_effect

    # ---- entry point ------------------------------------------------------

    def validate(self, claim: dict[str, Any]) -> ConformanceReport:
        rep = ConformanceReport(
            claim_id=claim.get("id", "<unidentified>"),
            standard_version=claim.get("standard_version", "unknown"),
        )
        add = _adder(rep)

        declared = claim.get("standard_version")
        add("LC-0.1", declared == STANDARD_VERSION,
            f"claim declares standard `{declared}`, validator implements `{STANDARD_VERSION}`"
            if declared != STANDARD_VERSION
            else f"claim declares `{declared}`",
        )

        ev = claim.get("evidence_class")
        add("LC-0.2", ev in {"measured", "simulated", "mixed"},
            f"evidence_class is {ev!r}; it must be one of measured, simulated, mixed"
            if ev not in {"measured", "simulated", "mixed"}
            else f"evidence class: {ev}",
        )
        if ev in {"simulated", "mixed"}:
            add("LC-0.3", bool(claim.get("evidence_note")),
                "a simulated or mixed claim must state what was modelled and what was "
                "measured" if not claim.get("evidence_note")
                else "the simulated component is described",
            )

        self._check_loss_column(claim, add)
        self._check_parity(claim, add)
        self._check_envelope(claim, add)
        self._check_prereg(claim, add)
        self._check_reproduction(claim, add)
        self._check_quality(claim, add)
        return rep

    # ---- LC-1 -------------------------------------------------------------

    def _check_loss_column(self, c: dict[str, Any], add: Callable) -> None:
        lc = c.get("loss_column")
        add("LC-1.1", isinstance(lc, dict) and bool(lc),
            "no loss column present" if not lc else "loss column present")
        if not isinstance(lc, dict):
            return

        counts = lc.get("counts", {}) or {}
        n_loss = int(counts.get("loss", 0))
        regions = lc.get("regions", []) or []
        add("LC-1.2", not (n_loss > 0 and not regions),
            f"{n_loss} losing cells but no region is described"
            if (n_loss > 0 and not regions)
            else "losing cells are covered by described regions",
        )

        notes = " ".join(lc.get("notes", []) or [])
        add("LC-1.3", "INCOMPLETE" not in notes,
            "the loss column declares itself incomplete" if "INCOMPLETE" in notes
            else "loss column is complete over its own cells",
        )

        placement = c.get("loss_column_placement", "main")
        add("LC-1.4", placement == "main",
            f"loss column is placed in `{placement}`; the standard requires equal prominence "
            "to the wins" if placement != "main" else "loss column is in the main body",
        )

        unattributed = [r for r in regions if not r.get("attribution")]
        add("LC-1.5", not unattributed,
            f"{len(unattributed)}/{len(regions)} loss region(s) have no attributed cause"
            if unattributed else "every loss region carries an attributed cause",
        )

        # Missing cells are losses of the harshest kind and must be visible.
        miss = int(counts.get("missing", 0))
        described = any("missing" in (r.get("missing_reasons") and "x" or "") for r in regions) or (
            "could not be measured" in notes
        )
        add("LC-1.6", miss == 0 or described,
            f"{miss} unmeasurable cell(s) are not accounted for in the column"
            if (miss and not described) else "unmeasurable cells are accounted for",
        )

        # LC-1.9. A region is an assertion; its own numbers have to support it.
        # Three mechanical ways a region can be internally unsupported: more
        # losing cells than cells, a worst case milder than its own median, or
        # a q-value above the threshold the claim itself declared.
        q_level = float(lc.get("q_level") or 0.05)
        mde_pct = float(lc.get("mde_pct") or 0.0)
        bad: list[str] = []
        for i, r in enumerate(regions, 1):
            n_cells_r, n_loss_r = int(r.get("n_cells", 0)), int(r.get("n_loss", 0))
            worst, med = r.get("worst_regression_pct"), r.get("median_regression_pct")
            mq = r.get("max_q")
            if n_loss_r > n_cells_r:
                bad.append(f"region {i} reports {n_loss_r} losing cells in {n_cells_r} cells")
            if isinstance(worst, (int, float)) and isinstance(med, (int, float)):
                if math.isfinite(worst) and math.isfinite(med) and worst < med - 1e-9:
                    bad.append(f"region {i} has a worst case milder than its median")
                if math.isfinite(worst) and worst < mde_pct - 1e-9:
                    bad.append(
                        f"region {i} worst case {worst:.1f}% is below the claim's own "
                        f"MDE of {mde_pct:.1f}%"
                    )
            if isinstance(mq, (int, float)) and math.isfinite(mq) and mq > q_level + 1e-12:
                bad.append(
                    f"region {i} max q={mq:.3g} exceeds the declared threshold {q_level:g}"
                )
        add("LC-1.9", not bad, "; ".join(bad) if bad
            else "every region's support meets the claim's declared thresholds")

        design = lc.get("design_check") or {}
        add("LC-1.8", design.get("can_reject", True),
            design.get("message", "")
            or "the design is capable of declaring a loss if one exists",
        )

        add("LC-1.7", float(lc.get("inconclusive_fraction", 0.0)) <= 0.35,
            f"{float(lc.get('inconclusive_fraction', 0)):.0%} of the envelope is inconclusive; "
            "the sweep may lack the power to support either a win or a loss claim"
            if float(lc.get("inconclusive_fraction", 0.0)) > 0.35
            else "inconclusive fraction is within tolerance",
        )

    # ---- LC-2 -------------------------------------------------------------

    def _check_parity(self, c: dict[str, Any], add: Callable) -> None:
        par = c.get("parity")
        needs_parity = bool(par) or _is_tuned_comparison(c)

        if not needs_parity:
            # Every parity rule is inapplicable to an untuned single-system
            # measurement. Marking them n/a rather than passing keeps the
            # coverage matrix honest: a rule nobody could have failed here is
            # not evidence that the artifact satisfies it.
            for rid in ("LC-2.1", "LC-2.2", "LC-2.3", "LC-2.4", "LC-2.5",
                        "LC-2.6", "LC-2.7"):
                add(rid, True, "no tuning was involved; parity does not apply", na=True)
            return

        add("LC-2.1", isinstance(par, dict) and bool(par),
            "a tuned comparison ships no parity certificate" if not par
            else "parity certificate present")
        if not isinstance(par, dict):
            return

        add("LC-2.2", bool(par.get("ok")),
            "parity certificate reports a violation" if not par.get("ok")
            else "parity certificate holds")

        ledgers = par.get("ledgers", []) or []
        add("LC-2.3", len(ledgers) >= 2,
            f"only {len(ledgers)} tuning ledger(s); parity needs at least two systems"
            if len(ledgers) < 2 else f"{len(ledgers)} systems tuned under one budget")

        counts = {led.get("system"): led.get("n_trials") for led in ledgers}
        add("LC-2.4", len(set(counts.values())) <= 1,
            f"trial counts differ: {counts}" if len(set(counts.values())) > 1
            else f"equal trial budgets ({next(iter(counts.values()), 0)} per system)")

        binding = [led.get("system") for led in ledgers if led.get("still_improving")]
        add("LC-2.5", not binding,
            f"budget was binding for {binding}: their performance is a lower bound"
            if binding else "no system was still improving when its budget ended")

        ops = {led.get("operator") for led in ledgers}
        add("LC-2.6", True,
            f"operator(s) of record: {sorted(x for x in ops if x)} "
            "(skill is documented, not eliminated)")

        # A baseline may be a *derived* comparator -- "the best of the tuned
        # alternatives at each cell" -- which has no ledger of its own but
        # inherits one from each constituent. The requirement is that nothing
        # in the comparison went untuned, so the check follows the derivation
        # rather than insisting on a literal ledger for the baseline's name.
        tuned_systems = {led.get("system") for led in ledgers}
        derived = (c.get("supporting", {}) or {}).get("baseline_derived_from") or []
        baseline = c.get("baseline")
        if baseline in tuned_systems:
            add("LC-2.7", True,
                f"the baseline `{baseline}` was tuned under the same budget")
        elif derived:
            missing = [d for d in derived if d not in tuned_systems]
            add("LC-2.7", not missing,
                f"the derived baseline `{baseline}` draws on {missing}, which have no "
                f"tuning ledger" if missing
                else f"the derived baseline `{baseline}` is composed of tuned systems "
                     f"({', '.join(sorted(derived))}), each with its own ledger")
        else:
            add("LC-2.7", False,
                f"the baseline `{baseline}` has no tuning ledger and declares no "
                "derivation: this is the untuned-baseline failure the requirement targets")

        # LC-2.8. A claim may state that an arm ran at library defaults. If a
        # ledger shows that arm was tuned, one of the two is false, and the
        # direction that flatters the author is the one worth catching: a tuned
        # configuration described as "out of the box".
        policies = (c.get("supporting", {}) or {}).get("tuning_policy") or {}
        trials_by_system = {led.get("system"): int(led.get("n_trials") or 0)
                            for led in ledgers}
        contradictions = []
        for system, policy in policies.items():
            claims_tuned = "default" not in str(policy).lower()
            n = trials_by_system.get(system)
            if n is None:
                continue
            if not claims_tuned and n > 0:
                contradictions.append(
                    f"`{system}` is described as `{policy}` but its ledger records "
                    f"{n} tuning trial(s)"
                )
            if claims_tuned and n == 0:
                contradictions.append(
                    f"`{system}` is described as `{policy}` but its ledger records no trials"
                )
        add("LC-2.8", not contradictions,
            "; ".join(contradictions) if contradictions
            else "declared tuning state matches the ledgers")

    # ---- LC-3 -------------------------------------------------------------

    def _check_envelope(self, c: dict[str, Any], add: Callable) -> None:
        lc = c.get("loss_column", {}) or {}
        env = c.get("envelope") or {}
        factors = env.get("factors") or [
            {"name": k, "levels": v} for k, v in (lc.get("factor_levels") or {}).items()
        ]
        n_cells = int(lc.get("n_cells", 0)) or _cells_from_factors(factors)

        add("LC-3.1", len(factors) >= 2,
            f"only {len(factors)} factor(s) swept; a point estimate cannot support a "
            "general claim" if len(factors) < 2 else f"{len(factors)} factors swept")

        add("LC-3.2", n_cells >= self.min_cells,
            f"{n_cells} cells is below the floor of {self.min_cells}"
            if n_cells < self.min_cells else f"{n_cells} cells measured")

        h = c.get("headline") or {}
        has_range = all(k in h for k in ("best_pct", "worst_pct", "median_pct"))
        add("LC-3.3", has_range,
            "headline does not report best/worst/median across the envelope"
            if not has_range else "headline reports the range, not a single number")

        if has_range:
            same = math.isclose(float(h["best_pct"]), float(h["worst_pct"]), rel_tol=1e-9)
            add("LC-3.4", not same,
                "headline best and worst are identical, which suggests a point estimate "
                "dressed as a range" if same else "headline range is non-degenerate")
            add("LC-3.5", bool(h.get("conditions_worst")),
                "the worst case is stated without the conditions that produce it"
                if not h.get("conditions_worst") else "worst case names its conditions")

        # LC-3.7. The headline is the sentence a reader will quote, so it is the
        # one most worth checking against the artifact's own numbers. A worst
        # case milder than the loss column's worst region is the signature of a
        # summary written by hand and never reconciled with the measurements.
        lc3 = c.get("loss_column") or {}
        regions3 = lc3.get("regions") or []
        finite_worsts = [
            float(r["worst_regression_pct"]) for r in regions3
            if isinstance(r.get("worst_regression_pct"), (int, float))
            and math.isfinite(float(r["worst_regression_pct"]))
        ]
        worst_region = max(finite_worsts) if finite_worsts else None
        problems: list[str] = []
        if has_range:
            hw, hb = float(h.get("worst_pct", 0)), float(h.get("best_pct", 0))
            # The headline speaks in the reader's direction; the loss column is
            # oriented so positive means worse. A headline worst of -30%
            # corresponds to a region worst of +30%.
            if worst_region is not None and math.isfinite(hw) and -hw < worst_region - 1.0:
                problems.append(
                    f"headline reports a worst case of {hw:+.1f}% but the loss column "
                    f"contains a region at {-worst_region:+.1f}%"
                )
            if math.isfinite(hw) and math.isfinite(hb) and hb < hw - 1e-9:
                problems.append("headline best is worse than headline worst")
            n_loss3 = int((lc3.get("counts") or {}).get("loss", 0))
            if n_loss3 > 0 and math.isfinite(hw) and hw >= 0:
                problems.append(
                    f"headline reports no adverse case, but the loss column has "
                    f"{n_loss3} losing cell(s)"
                )
        add("LC-3.7", not problems, "; ".join(problems) if problems
            else "headline extremes agree with the loss column")

        stmt = str(h.get("statement", ""))
        bare = re.search(r"\b(up to|as much as)\b", stmt, re.I)
        add("LC-3.6", bare is None,
            f"headline uses an unbounded superlative ({bare.group(0)!r}) that the envelope "
            "cannot support" if bare else "headline avoids unbounded superlatives")

    # ---- LC-4 -------------------------------------------------------------

    def _check_prereg(self, c: dict[str, Any], add: Callable) -> None:
        pre = c.get("prereg")
        add("LC-4.1", isinstance(pre, dict) and bool(pre),
            "no pre-registration document" if not pre else "pre-registration present")
        if not isinstance(pre, dict):
            return

        add("LC-4.2", bool(pre.get("sealed_at")),
            "protocol was never sealed" if not pre.get("sealed_at")
            else f"sealed at {pre.get('sealed_at')}")

        add("LC-4.3", pre.get("seal_hash") == pre.get("normative_hash"),
            "normative fields no longer hash to the seal: the protocol was edited after "
            "sealing" if pre.get("seal_hash") != pre.get("normative_hash")
            else "protocol matches its seal")

        ver = c.get("prereg_verification") or {}
        add("LC-4.4", bool(ver),
            "no verification of results against the protocol was run" if not ver
            else "results were verified against the protocol")
        if ver:
            fatal = [f for f in ver.get("findings", []) if f.get("severity") == "fatal"]
            add("LC-4.5", not fatal,
                "; ".join(f["message"] for f in fatal) if fatal
                else "no fatal protocol deviations")

        devs = pre.get("deviations", []) or []
        add("LC-4.6", not devs,
            f"{len(devs)} declared deviation(s) from the sealed protocol" if devs
            else "no deviations from the sealed protocol")

        add("LC-4.7", bool(pre.get("anchor")),
            "the seal has no external anchor; ordering rests on the author's own timestamp"
            if not pre.get("anchor") else f"seal anchored to `{pre.get('anchor')}`")

    # ---- LC-5 -------------------------------------------------------------

    def _check_reproduction(self, c: dict[str, Any], add: Callable) -> None:
        r = c.get("reproduction") or {}
        add("LC-5.1", bool(r.get("command")),
            "no reproduction command" if not r.get("command")
            else f"`{r.get('command')}`")

        cmd = str(r.get("command", ""))
        multi = bool(re.search(r"(&&|;\s*\S|\|\|)", cmd)) or "\n" in cmd.strip()
        add("LC-5.2", not multi,
            "reproduction is a shell pipeline, not one command; wrap it in an entry point"
            if multi else "reproduction is a single command")

        add("LC-5.3", bool(r.get("commit")),
            "no commit pinned" if not r.get("commit") else f"pinned to `{str(r.get('commit'))[:12]}`")

        add("LC-5.4", r.get("dirty") is not True,
            "the producing working tree was dirty; the pinned commit does not identify the "
            "code that ran" if r.get("dirty") else "producing tree was clean")

        add("LC-5.5", bool(r.get("container_image")),
            "no container image pinned; the software environment is not reproducible from "
            "this record alone" if not r.get("container_image")
            else f"image `{r.get('container_image')}`")

        add("LC-5.6", bool(r.get("hardware")),
            "hardware is not documented" if not r.get("hardware")
            else f"hardware: {r.get('hardware')}")

        add("LC-5.7", r.get("estimated_runtime_min") is not None,
            "no runtime estimate; a reproducer cannot tell whether this is 10 minutes or "
            "10 days" if r.get("estimated_runtime_min") is None
            else f"~{r.get('estimated_runtime_min')} min")

    # ---- LC-Q -------------------------------------------------------------

    def _check_quality(self, c: dict[str, Any], add: Callable) -> None:
        env = c.get("envelope") or {}
        reps = int(env.get("replicates", 0) or 0)
        if reps:
            add("LC-Q1", reps >= self.min_replicates,
                f"{reps} replicates per cell is below the floor of {self.min_replicates}"
                if reps < self.min_replicates else f"{reps} replicates per cell")
            add("LC-Q2", bool(env.get("interleaved")),
                "systems were measured in blocks, not interleaved; drift between blocks is "
                "confounded with the effect" if not env.get("interleaved")
                else "measurements were interleaved A/B")

            cov = env.get("coverage") or {}
            worst = min(cov.values()) if cov else 1.0
            add("LC-Q3", worst >= 0.9,
                f"lowest per-system cell coverage is {worst:.0%}" if worst < 0.9
                else f"cell coverage >= {worst:.0%} for every system")

        lc = c.get("loss_column") or {}
        noise = (c.get("supporting", {}) or {}).get("noise", {}) or {}
        cv = noise.get("median_cv")
        worst_eff = abs(float(lc.get("regions", [{}])[0].get("worst_regression_pct", 0)) / 100) \
            if lc.get("regions") else None
        if cv is not None and worst_eff:
            ratio = float(cv) / max(worst_eff, 1e-9)
            add("LC-Q4", ratio <= self.max_noise_to_effect,
                f"within-cell noise (CV {float(cv):.1%}) is large relative to the headline "
                f"effect; ratio {ratio:.2f}" if ratio > self.max_noise_to_effect
                else f"noise/effect ratio {ratio:.2f}")

        # LC-Q6. Gated tightly so it cannot fire on a legitimately low-power
        # sweep: it requires that the design *had* the power to reject, that the
        # envelope is large enough for the absence to mean something, that
        # equivalence was never once established, and that most of the envelope
        # is undetermined. A correct equivalence test on a real surface does not
        # do that. An inverted one does it every time -- which is how this rule
        # came to exist, from a real bug in this package's own statistics.
        lcq = c.get("loss_column") or {}
        counts_q = lcq.get("counts") or {}
        n_q = int(lcq.get("n_cells") or 0)
        inconclusive_frac = float(lcq.get("inconclusive_fraction") or 0.0)
        powered = bool((lcq.get("design_check") or {}).get("can_reject", False))
        broken_tost = (
            powered
            and n_q >= 20
            and int(counts_q.get("tie", 0)) == 0
            and inconclusive_frac > 0.5
        )
        add("LC-Q6", not broken_tost,
            f"the design has the power to reject, yet equivalence is established in 0 of "
            f"{n_q} cells while {inconclusive_frac:.0%} are inconclusive: the signature of "
            "an inverted or broken two-one-sided-tests procedure"
            if broken_tost else "equivalence is established where the data supports it")

        add("LC-Q5", bool(c.get("limitations")),
            "no limitations section" if not c.get("limitations")
            else f"{len(c.get('limitations'))} limitation(s) stated")


def _adder(rep: ConformanceReport) -> Callable:
    """Emit a finding, taking severity from the frozen registry.

    The call site names the rule and says whether it passed. It does not get to
    choose the severity: that is the whole content of freezing the standard,
    and before the registry existed one rule really was ``info`` on one code
    path and ``fatal`` on another.

    ``na=True`` marks a rule that does not apply to this claim -- an untuned
    single-system measurement has no parity to check. Not-applicable is not
    passing and not failing; it is counted separately so a coverage matrix can
    tell "we checked and it was fine" from "we never looked".
    """

    def add(rule: str, passed: bool, message: str, *, na: bool = False,
            **detail: Any) -> None:
        r = registry.get(rule)
        rep.findings.append(
            Finding(
                rule=r.id,
                requirement=r.requirement,
                severity=registry.INFO if na else r.severity,
                passed=True if na else bool(passed),
                message=message,
                detail={**detail, "not_applicable": True} if na else detail,
            )
        )

    return add


def _cells_from_factors(factors: list[dict[str, Any]]) -> int:
    n = 1
    for f in factors:
        n *= max(len(f.get("levels", [])), 1)
    return n if factors else 0


def _is_tuned_comparison(c: dict[str, Any]) -> bool:
    """Whether the claim is of a kind where tuning parity is the live threat."""
    if c.get("parity"):
        return True
    thrust = str(c.get("thrust", "")).upper()
    if thrust in {"II", "2"}:
        return True
    text = " ".join(
        [str(c.get("title", "")), str((c.get("headline") or {}).get("statement", ""))]
    ).lower()
    return any(w in text for w in ("tuned", "engine", "serving", "throughput of"))


def validate_claim(claim: dict[str, Any], **kw: Any) -> ConformanceReport:
    return Validator(**kw).validate(claim)
