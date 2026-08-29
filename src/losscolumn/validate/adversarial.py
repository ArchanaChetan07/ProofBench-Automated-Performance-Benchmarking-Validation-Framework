"""An adversarial corpus for LC-1.0: artifacts that are deliberately wrong.

A conformance checker that has only ever been shown conforming documents is a
rubber stamp. This module ships the counter-examples: a known-good claim, and a
catalogue of mutations that each break it in one specific, named way, with the
rule that is supposed to catch each one.

It lives in the package rather than in ``tests/`` on purpose. The corpus is
part of the standard's deliverable: anyone implementing LC-1.0 in another
language can run their validator against these documents and check that it
reaches the same verdicts. A standard whose only implementation is its author's
is not a standard.

Every defect here corresponds to a real failure mode, and several were found in
this project's own artifacts before they were rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from losscolumn.spec import registry
from losscolumn.version import STANDARD_VERSION

# --------------------------------------------------------------------------
# the known-good baselines
# --------------------------------------------------------------------------


def conforming_claim() -> dict[str, Any]:
    """A claim that passes every applicable rule. Every defect mutates this."""
    return {
        "kind": "claim",
        "standard_version": STANDARD_VERSION,
        "id": "adversarial-baseline",
        "title": "A conforming baseline claim",
        "thrust": "III",
        "method": "candidate",
        "baseline": "reference",
        "evidence_class": "measured",
        "evidence_note": "",
        "loss_column_placement": "main",
        "authors": ["corpus"],
        "headline": {
            "statement": "Faster in some regions of the envelope and slower in others.",
            "best_pct": 18.0,
            "worst_pct": -24.0,
            "median_pct": 1.5,
            "conditions_best": {"seq_len": 2048, "dtype": "float16"},
            "conditions_worst": {"seq_len": 128, "dtype": "bfloat16"},
        },
        "loss_column": {
            "metric": "throughput",
            "unit": "TFLOP/s",
            "method": "candidate",
            "baseline": "reference",
            "mde_pct": 5.0,
            "q_level": 0.05,
            "n_cells": 24,
            "counts": {"loss": 3, "win": 5, "tie": 14, "inconclusive": 2, "missing": 0},
            "loss_fraction": 3 / 24,
            "inconclusive_fraction": 2 / 24,
            "win_fraction": 5 / 24,
            "n_unrunnable": 0,
            "design_check": {
                "can_reject": True,
                "replicates": 11,
                "n_tests": 24,
                "min_replicates": 9,
                "p_floor": 2**-11,
                "message": "",
            },
            "regions": [
                {
                    "label": "seq_len=[128]",
                    "bounds": {"seq_len": [128]},
                    "n_cells": 4,
                    "n_loss": 3,
                    "frac_of_envelope": 4 / 24,
                    "median_regression_pct": 19.0,
                    "worst_regression_pct": 24.0,
                    "max_q": 0.004,
                    "attribution": "launch bound at short sequences",
                    "missing_reasons": [],
                }
            ],
            "notes": [],
        },
        "envelope": {
            "kind": "envelope",
            "replicates": 11,
            "interleaved": True,
            "systems": ["candidate", "reference"],
            "coverage": {"candidate": 1.0, "reference": 1.0},
            "metric": {"name": "throughput", "unit": "TFLOP/s", "higher_is_better": True},
            "factors": [
                {"name": "seq_len", "levels": [128, 512, 2048]},
                {"name": "head_dim", "levels": [64, 128]},
                {"name": "batch", "levels": [1, 4]},
                {"name": "dtype", "levels": ["float16", "bfloat16"]},
            ],
        },
        "reproduction": {
            "command": "losscolumn run thrust3",
            "repo": "https://example.invalid/losscolumn",
            "commit": "0123456789abcdef",
            "dirty": False,
            "container_image": "ghcr.io/example/losscolumn:1.0",
            "hardware": "1x H100 80GB",
            "estimated_runtime_min": 45,
        },
        "prereg": {
            "sealed_at": "2026-01-01T00:00:00Z",
            "seal_hash": "sha256:0f0f",
            "normative_hash": "sha256:0f0f",
            "anchor": "git:v1.0.0",
            "deviations": [],
            "mde": 0.05,
            "q_level": 0.05,
            "replicates": 11,
        },
        "prereg_verification": {"ok": True, "findings": []},
        "limitations": ["forward pass only"],
        # Present so LC-Q4 is exercised rather than skipped: a rule that is
        # never emitted has no passing case, and the coverage matrix should
        # show that as a gap rather than hide it.
        "supporting": {"noise": {"median_cv": 0.012, "p95_cv": 0.03, "n": 24}},
    }


def conforming_tuned_claim() -> dict[str, Any]:
    """A conforming claim of the kind where parity rules apply."""
    c = conforming_claim()
    c["id"] = "adversarial-baseline-tuned"
    c["thrust"] = "II"
    c["method"] = "engine_a"
    c["baseline"] = "engine_b"
    c["loss_column"]["method"] = "engine_a"
    c["loss_column"]["baseline"] = "engine_b"
    c["supporting"] = {
        "tuning_policy": {"engine_a": "tuned, 40 trials", "engine_b": "tuned, 40 trials"}
    }
    c["parity"] = {
        "ok": True,
        "issued_at": "2026-01-02T00:00:00Z",
        "systems": ["engine_a", "engine_b"],
        "violations": [],
        "ledgers": [
            {"system": "engine_a", "n_trials": 40, "n_ok": 40, "operator": "op",
             "still_improving": False, "group": "chat",
             "space": {"n_dims": 5, "cardinality": 192},
             "trials_per_dim": 8.0, "space_coverage": 0.2, "wall_time_s": 600},
            {"system": "engine_b", "n_trials": 40, "n_ok": 40, "operator": "op",
             "still_improving": False, "group": "chat",
             "space": {"n_dims": 5, "cardinality": 192},
             "trials_per_dim": 8.0, "space_coverage": 0.2, "wall_time_s": 600},
        ],
    }
    return c


def conforming_simulated_claim() -> dict[str, Any]:
    """A conforming claim whose evidence is modelled, and says so.

    Needed for coverage: LC-0.3 only fires on a simulated or mixed claim, so a
    corpus of measured claims alone would leave it with no passing case.
    """
    c = conforming_claim()
    c["id"] = "adversarial-baseline-simulated"
    c["evidence_class"] = "simulated"
    c["evidence_note"] = (
        "Produced by a calibratable roofline model, not measured on the target "
        "hardware; the model's parameters are the quantities a microbenchmark measures."
    )
    return c


# --------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------


@dataclass
class Defect:
    """One deliberately broken artifact, and the rule that must catch it."""

    name: str
    description: str
    expect_fail: tuple[str, ...]
    mutate: Callable[[dict[str, Any]], None]
    base: str = "plain"           # "plain" | "tuned"
    severity: str = registry.FATAL
    notes: str = ""

    def build(self) -> dict[str, Any]:
        doc = conforming_tuned_claim() if self.base == "tuned" else conforming_claim()
        self.mutate(doc)
        doc["id"] = f"adversarial-{self.name}"
        return doc


def _set(d: Any, path: str, value: Any) -> None:
    """Set a dotted path, where an integer component indexes a list."""
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else cur[p]
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


# The nine defects named in the freeze request come first, in that order, then
# the remainder needed to give every rule a failing case.
CORPUS: tuple[Defect, ...] = (
    # ---- the named nine --------------------------------------------------
    Defect(
        "insufficient-replicates",
        "Three replicates on a 24-cell grid: the sign-flip floor of 2^-3 = 0.125 "
        "cannot clear the Benjamini-Hochberg threshold of 0.05/24, so no cell could "
        "ever be declared a loss.",
        ("LC-1.8", "LC-Q1"),
        lambda d: (
            _set(d, "loss_column.design_check", {
                "can_reject": False, "replicates": 3, "n_tests": 24,
                "min_replicates": 9, "p_floor": 0.125,
                "message": "design cannot reject: with r=3 the smallest attainable "
                           "p-value is 0.125, but Benjamini-Hochberg over 24 cells "
                           "requires p <= 0.0021",
            }),
            _set(d, "envelope.replicates", 3),
        ),
    ),
    Defect(
        "bh-impossible-power",
        "Eleven replicates, but 100000 simultaneous tests: q/m falls below the "
        "sign-flip floor, so the grid is too wide for the replicate count.",
        ("LC-1.8",),
        lambda d: _set(d, "loss_column.design_check", {
            "can_reject": False, "replicates": 11, "n_tests": 100000,
            "min_replicates": 21, "p_floor": 2**-11,
            "message": "design cannot reject: Benjamini-Hochberg over 100000 cells "
                       "requires p <= 5e-07 for the most extreme cell",
        }),
    ),
    Defect(
        "inverted-tost-bounds",
        "A powered 24-cell sweep in which equivalence is never once established and "
        "most cells are inconclusive: the signature of two-one-sided tests whose "
        "alternatives point outward instead of inward.",
        ("LC-Q6",),
        lambda d: (
            _set(d, "loss_column.counts",
                 {"loss": 3, "win": 2, "tie": 0, "inconclusive": 19, "missing": 0}),
            _set(d, "loss_column.inconclusive_fraction", 19 / 24),
        ),
        notes="This defect was real in this package before it was a rule.",
    ),
    Defect(
        "mde-differs-from-seal",
        "The analysis used a 5% minimum effect while the sealed protocol registered "
        "10%, undeclared.",
        ("LC-4.5",),
        lambda d: _set(d, "prereg_verification", {
            "ok": False,
            "findings": [{
                "code": "PRE-005", "severity": "fatal",
                "message": "analysis used mde=0.05 but the protocol registered 0.1 "
                           "(UNDECLARED)",
                "detail": {"field": "mde"},
            }],
        }),
    ),
    Defect(
        "simulated-unlabeled",
        "A modelled result presented without saying so.",
        ("LC-0.2",),
        lambda d: _set(d, "evidence_class", "estimated"),
    ),
    Defect(
        "simulated-unexplained",
        "Labelled simulated, but never says what was modelled and what measured.",
        ("LC-0.3",),
        lambda d: (_set(d, "evidence_class", "simulated"),
                   _set(d, "evidence_note", "")),
    ),
    Defect(
        "tuned-baseline-called-default",
        "The baseline is described as running at library defaults while its ledger "
        "records forty tuning trials.",
        ("LC-2.8",),
        lambda d: _set(d, "supporting.tuning_policy",
                       {"engine_a": "tuned, 40 trials", "engine_b": "library defaults"}),
        base="tuned",
    ),
    Defect(
        "unsupported-headline",
        "The headline reports a worst case of -4% while the loss column contains a "
        "region at -24%.",
        ("LC-3.7",),
        lambda d: _set(d, "headline.worst_pct", -4.0),
    ),
    Defect(
        "dirty-tree",
        "Produced from a modified working tree, so the pinned commit does not "
        "identify the code that ran.",
        ("LC-5.4",),
        lambda d: _set(d, "reproduction.dirty", True),
    ),
    Defect(
        "unsupported-loss-region",
        "A region asserted as a loss whose q-value exceeds the threshold the claim "
        "itself declares, and whose worst case is milder than its own median.",
        ("LC-1.9",),
        lambda d: _set(d, "loss_column.regions", [{
            "label": "seq_len=[128]", "bounds": {"seq_len": [128]},
            "n_cells": 4, "n_loss": 3, "frac_of_envelope": 4 / 24,
            "median_regression_pct": 19.0, "worst_regression_pct": 8.0,
            "max_q": 0.4, "attribution": "asserted", "missing_reasons": [],
        }]),
    ),

    # ---- remaining coverage ---------------------------------------------
    Defect("wrong-standard-version", "Declares a revision the validator does not implement.",
           ("LC-0.1",), lambda d: _set(d, "standard_version", "LC-0.9")),
    Defect("no-loss-column", "Ships no loss column at all.",
           ("LC-1.1",), lambda d: d.pop("loss_column")),
    Defect("losses-without-regions", "Reports losing cells but describes no region.",
           ("LC-1.2",), lambda d: _set(d, "loss_column.regions", [])),
    Defect("self-declared-incomplete", "The column says it is incomplete.",
           ("LC-1.3",), lambda d: _set(d, "loss_column.notes",
                                       ["the column is INCOMPLETE and must not be published"])),
    Defect("loss-column-in-appendix", "The column is relegated to an appendix.",
           ("LC-1.4",), lambda d: _set(d, "loss_column_placement", "appendix")),
    Defect("unattributed-region", "A region with no attributed cause.",
           ("LC-1.5",), lambda d: _set(d, "loss_column.regions.0.attribution", ""),
           severity=registry.WARNING),
    Defect("unaccounted-missing-cells", "Unmeasurable cells that nothing explains.",
           ("LC-1.6",),
           lambda d: _set(d, "loss_column.counts",
                          {"loss": 3, "win": 5, "tie": 12, "inconclusive": 2, "missing": 2}),
           severity=registry.WARNING),
    Defect("mostly-inconclusive", "Over a third of the envelope is undetermined.",
           ("LC-1.7",),
           lambda d: _set(d, "loss_column.inconclusive_fraction", 0.6),
           severity=registry.WARNING),
    Defect("no-parity-certificate", "A tuned comparison with no parity certificate.",
           ("LC-2.1",), lambda d: d.pop("parity"), base="tuned"),
    Defect("parity-violated", "The certificate reports a violation.",
           ("LC-2.2",), lambda d: _set(d, "parity.ok", False), base="tuned"),
    Defect("one-system-only", "A single ledger presented as a comparison.",
           ("LC-2.3",),
           lambda d: _set(d, "parity.ledgers", [d["parity"]["ledgers"][0]]), base="tuned"),
    Defect("unequal-trial-counts", "One engine received twice the budget.",
           ("LC-2.4",),
           lambda d: _set(d, "parity.ledgers.0.n_trials", 80),
           base="tuned", severity=registry.WARNING),
    Defect("budget-was-binding", "An engine was still improving when its budget ended.",
           ("LC-2.5",),
           lambda d: _set(d, "parity.ledgers.0.still_improving", True),
           base="tuned", severity=registry.WARNING),
    Defect("untuned-baseline", "The baseline has no ledger and declares no derivation.",
           ("LC-2.7",),
           lambda d: (_set(d, "baseline", "never_tuned"),
                      _set(d, "supporting.tuning_policy",
                           {"engine_a": "tuned, 40 trials"})),
           base="tuned"),
    Defect("single-factor", "One factor swept.",
           ("LC-3.1",),
           lambda d: _set(d, "envelope.factors", [{"name": "seq_len", "levels": [1, 2, 3]}])),
    Defect("too-few-cells", "Below the minimum cell count.",
           ("LC-3.2",), lambda d: _set(d, "loss_column.n_cells", 4)),
    Defect("headline-without-range", "No best/worst/median.",
           ("LC-3.3",), lambda d: d["headline"].pop("worst_pct")),
    Defect("degenerate-range", "Best equals worst: a point estimate dressed as a range.",
           ("LC-3.4",),
           lambda d: (_set(d, "headline.best_pct", -24.0),
                      _set(d, "headline.worst_pct", -24.0)),
           severity=registry.WARNING),
    Defect("worst-without-conditions", "A worst case nobody can locate.",
           ("LC-3.5",), lambda d: _set(d, "headline.conditions_worst", {}),
           severity=registry.WARNING),
    Defect("unbounded-superlative", "'up to 2x faster'.",
           ("LC-3.6",), lambda d: _set(d, "headline.statement", "Up to 2x faster."),
           severity=registry.WARNING),
    Defect("no-preregistration", "No protocol document.",
           ("LC-4.1",), lambda d: _set(d, "prereg", None)),
    Defect("unsealed-protocol", "A protocol that was never sealed.",
           ("LC-4.2",), lambda d: _set(d, "prereg.sealed_at", None)),
    Defect("protocol-edited-after-sealing", "Normative fields no longer hash to the seal.",
           ("LC-4.3",), lambda d: _set(d, "prereg.normative_hash", "sha256:beef")),
    Defect("unverified-protocol", "Nobody checked the results against the seal.",
           ("LC-4.4",), lambda d: _set(d, "prereg_verification", None)),
    Defect("declared-deviation", "A declared departure from the sealed protocol.",
           ("LC-4.6",),
           lambda d: _set(d, "prereg.deviations",
                          [{"field": "factors", "registered": "a", "actual": "b",
                            "reason": "hardware", "declared_at": "2026-01-03T00:00:00Z"}]),
           severity=registry.WARNING),
    Defect("unanchored-seal", "The seal rests on the author's own clock.",
           ("LC-4.7",), lambda d: _set(d, "prereg.anchor", None),
           severity=registry.WARNING),
    Defect("no-reproduction-command", "No command.",
           ("LC-5.1",), lambda d: _set(d, "reproduction.command", "")),
    Defect("pipeline-not-command", "A shell pipeline rather than an entry point.",
           ("LC-5.2",),
           lambda d: _set(d, "reproduction.command", "python a.py && python b.py"),
           severity=registry.WARNING),
    Defect("unpinned-commit", "No commit pinned.",
           ("LC-5.3",), lambda d: _set(d, "reproduction.commit", None)),
    Defect("unpinned-image", "No container image.",
           ("LC-5.5",), lambda d: _set(d, "reproduction.container_image", None),
           severity=registry.WARNING),
    Defect("undocumented-hardware", "Hardware not stated.",
           ("LC-5.6",), lambda d: _set(d, "reproduction.hardware", "")),
    Defect("no-runtime-estimate", "A reproducer cannot budget for the attempt.",
           ("LC-5.7",), lambda d: _set(d, "reproduction.estimated_runtime_min", None),
           severity=registry.WARNING),
    Defect("blocked-measurement", "Systems measured in blocks, not interleaved.",
           ("LC-Q2",), lambda d: _set(d, "envelope.interleaved", False),
           severity=registry.WARNING),
    Defect("sparse-coverage", "A third of the cells were never measured.",
           ("LC-Q3",),
           lambda d: _set(d, "envelope.coverage", {"candidate": 0.66, "reference": 1.0}),
           severity=registry.WARNING),
    Defect("noise-swamps-effect", "Within-cell noise comparable to the headline effect.",
           ("LC-Q4",),
           lambda d: _set(d, "supporting", {"noise": {"median_cv": 0.9, "n": 24}}),
           severity=registry.WARNING),
    Defect("no-limitations", "A claim with no stated bounds.",
           ("LC-Q5",), lambda d: _set(d, "limitations", []),
           severity=registry.WARNING),
)


# Rules that cannot be made to fail, with the reason. Kept explicit so the
# coverage matrix distinguishes "no failing test" from "no failing test is
# possible", which are very different states for a standard to be in.
UNFAILABLE: dict[str, str] = {
    "LC-2.6": "informational: names the operators of record and never fails",
}


@dataclass
class CorpusResult:
    defect: str
    expected: tuple[str, ...]
    caught: bool
    grade: str
    failed_rules: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


def run_corpus() -> list[CorpusResult]:
    """Grade every defective artifact and check the right rule caught it."""
    from losscolumn.validate.validator import validate_claim

    out: list[CorpusResult] = []
    for d in CORPUS:
        rep = validate_claim(d.build())
        failed = tuple(f.rule for f in rep.findings
                       if not f.passed and not f.detail.get("not_applicable"))
        missing = tuple(r for r in d.expect_fail if r not in failed)
        out.append(
            CorpusResult(
                defect=d.name,
                expected=d.expect_fail,
                caught=not missing,
                grade=rep.grade,
                failed_rules=failed,
                missing=missing,
                detail={"severity": d.severity, "base": d.base},
            )
        )
    return out


def coverage() -> dict[str, dict[str, Any]]:
    """For every registered rule: was it seen passing, and seen failing?"""
    from losscolumn.validate.validator import validate_claim

    seen_pass: set[str] = set()
    seen_fail: set[str] = set()
    seen_na: set[str] = set()

    for doc in (conforming_claim(), conforming_tuned_claim(),
                conforming_simulated_claim()):
        for f in validate_claim(doc).findings:
            if f.detail.get("not_applicable"):
                seen_na.add(f.rule)
            elif f.passed:
                seen_pass.add(f.rule)
            else:
                seen_fail.add(f.rule)

    for d in CORPUS:
        for f in validate_claim(d.build()).findings:
            if f.detail.get("not_applicable"):
                seen_na.add(f.rule)
            elif f.passed:
                seen_pass.add(f.rule)
            else:
                seen_fail.add(f.rule)

    return {
        r.id: {
            "requirement": r.requirement,
            "severity": r.severity,
            "summary": r.summary,
            "passing_case": r.id in seen_pass,
            "failing_case": r.id in seen_fail or r.id in UNFAILABLE,
            "unfailable_reason": UNFAILABLE.get(r.id, ""),
            "seen_not_applicable": r.id in seen_na,
        }
        for r in (registry.RULES[k] for k in sorted(registry.RULES))
    }


def coverage_markdown() -> str:
    cov = coverage()
    n = len(cov)
    both = sum(1 for v in cov.values() if v["passing_case"] and v["failing_case"])
    lines = [
        "# Validator coverage matrix -- LC-1.0",
        "",
        f"Registry digest `{registry.compute_digest()}` &mdash; {n} rules, "
        f"{registry.N_FATAL} fatal.",
        "",
        f"**{both}/{n}** rules have both a passing and a failing case. "
        f"The corpus contains {len(CORPUS)} deliberately defective artifacts.",
        "",
        "A rule with no failing case is a rule nobody has shown can fire. "
        "Where that is structural, the reason is given.",
        "",
        "| Rule | Requirement | Severity | Pass | Fail | Note |",
        "|------|-------------|----------|------|------|------|",
    ]
    for rid, v in cov.items():
        note = v["unfailable_reason"] or ("" if v["failing_case"] else "**no failing case**")
        lines.append(
            f"| `{rid}` | {v['requirement']} | {v['severity']} | "
            f"{'yes' if v['passing_case'] else '—'} | "
            f"{'yes' if v['failing_case'] else '**no**'} | {note} |"
        )
    lines += ["", "## Defect corpus", "",
              "| Defect | Expected rule(s) | Description |",
              "|--------|------------------|-------------|"]
    for d in CORPUS:
        lines.append(
            f"| `{d.name}` | {', '.join(f'`{r}`' for r in d.expect_fail)} | {d.description} |"
        )
    return "\n".join(lines)
