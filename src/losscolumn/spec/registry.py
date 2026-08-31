"""The frozen rule registry for LC-1.0.

Freezing a standard means more than writing "v1.0" at the top of a document.
It means the rule identifiers, their severities and the shape of the documents
they grade cannot drift without the drift being detectable. This module is
where that is made structural:

* Every rule this validator can emit is declared here, once, with its
  identifier, requirement, severity and the failure mode it forecloses.
* The validator looks severity up here instead of passing it at the call site,
  so a severity cannot be quietly changed in one branch of one function -- the
  bug that existed before this file, where ``LC-2.1`` was ``info`` on one path
  and ``fatal`` on another.
* :data:`FROZEN_DIGEST` is a content hash over the whole table. A test asserts
  it. Any change to a rule id or severity fails that test, which forces the
  change to be a deliberate version bump rather than an edit.

Adding a rule is a **minor** version bump and requires a new digest.
Changing an existing rule's id or severity is a **major** version bump: it
changes the meaning of a grade that other people's artifacts already carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from losscolumn.core.provenance import content_hash

FATAL = "fatal"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Rule:
    """One frozen conformance rule."""

    id: str
    requirement: str
    severity: str
    summary: str
    forecloses: str = ""

    def key(self) -> tuple[str, str, str]:
        """The part of the rule that is frozen. Prose may be corrected."""
        return (self.id, self.requirement, self.severity)


# --------------------------------------------------------------------------
# The table. Order is not significant; identifiers and severities are.
# --------------------------------------------------------------------------

LC_1_0_RULES: tuple[Rule, ...] = (
    # --- LC-0: the claim declares what it is ------------------------------
    Rule("LC-0.1", "Standard version", FATAL,
         "The claim declares the standard revision it is graded against.",
         "grading an artifact against rules it was never written for"),
    Rule("LC-0.2", "Evidence class", FATAL,
         "evidence_class is one of measured, simulated, mixed.",
         "a modelled number being read as a measured one"),
    Rule("LC-0.3", "Evidence class", FATAL,
         "A simulated or mixed claim states what was modelled and what measured.",
         "an unexplained simulation passing as evidence"),

    # --- LC-1: loss column -------------------------------------------------
    Rule("LC-1.1", "Loss column", FATAL, "A loss column is present.",
         "selective reporting"),
    Rule("LC-1.2", "Loss column", FATAL,
         "If any cell lost, at least one region describes where.",
         "acknowledging losses without locating them"),
    Rule("LC-1.3", "Loss column", FATAL,
         "The column does not declare itself incomplete.",
         "publishing a column known to be partial"),
    Rule("LC-1.4", "Loss column", FATAL,
         "The column is in the main body, not an appendix.",
         "burying adverse regimes"),
    Rule("LC-1.5", "Loss column", WARNING,
         "Every loss region carries an attributed cause.",
         "a location without a mechanism"),
    Rule("LC-1.6", "Loss column", WARNING,
         "Cells that could not be measured are accounted for.",
         "an out-of-memory regime vanishing from the record"),
    Rule("LC-1.7", "Loss column", WARNING,
         "The inconclusive fraction is stated and tolerable.",
         "an underdetermined envelope reading as a settled one"),
    Rule("LC-1.8", "Loss column", FATAL,
         "The design is capable of declaring a loss if one exists.",
         "an empty loss column produced by an underpowered sweep"),
    Rule("LC-1.9", "Loss column", FATAL,
         "Every region's statistical support meets the claim's own thresholds.",
         "a region asserted as a loss on evidence weaker than the claim declares"),

    # --- LC-2: tuning-budget parity ----------------------------------------
    Rule("LC-2.1", "Tuning-budget parity", FATAL,
         "A tuned comparison ships a parity certificate.",
         "the untuned baseline"),
    Rule("LC-2.2", "Tuning-budget parity", FATAL,
         "The parity certificate reports no violation.",
         "a comparison known to be unfair"),
    Rule("LC-2.3", "Tuning-budget parity", FATAL,
         "At least two systems were tuned under the budget.",
         "calling a single-system measurement a comparison"),
    Rule("LC-2.4", "Tuning-budget parity", WARNING,
         "Trial counts are equal across systems.",
         "unequal effort presented as equal"),
    Rule("LC-2.5", "Tuning-budget parity", WARNING,
         "No system was still improving when its budget ended.",
         "a lower bound reported as an optimum"),
    Rule("LC-2.6", "Tuning-budget parity", INFO,
         "The operators of record are named.",
         "an undocumented skill confound"),
    Rule("LC-2.7", "Tuning-budget parity", FATAL,
         "The baseline was tuned, or derives from systems that were.",
         "the untuned baseline, in its most direct form"),
    Rule("LC-2.8", "Tuning-budget parity", FATAL,
         "The declared tuning state of each arm matches its ledger.",
         "a tuned configuration presented as library defaults"),

    # --- LC-3: envelope, not point -----------------------------------------
    Rule("LC-3.1", "Envelope, not point", FATAL,
         "At least two factors were swept.",
         "a point estimate supporting a general claim"),
    Rule("LC-3.2", "Envelope, not point", FATAL,
         "The envelope meets the minimum cell count.",
         "a surface asserted from too few samples"),
    Rule("LC-3.3", "Envelope, not point", FATAL,
         "The headline reports best, worst and median.",
         "a single number as a headline"),
    Rule("LC-3.4", "Envelope, not point", WARNING,
         "The headline range is non-degenerate.",
         "a point estimate dressed as a range"),
    Rule("LC-3.5", "Envelope, not point", WARNING,
         "The worst case names the conditions that produce it.",
         "an unlocatable worst case"),
    Rule("LC-3.6", "Envelope, not point", WARNING,
         "The headline avoids unbounded superlatives.",
         "'up to 2x faster'"),
    Rule("LC-3.7", "Envelope, not point", FATAL,
         "The headline's extremes are consistent with the loss column.",
         "a summary the artifact's own measurements do not support"),

    # --- LC-4: pre-registration --------------------------------------------
    Rule("LC-4.1", "Pre-registration", FATAL, "A pre-registration document is present.",
         "post-hoc selection of the favourable comparison"),
    Rule("LC-4.2", "Pre-registration", FATAL, "The protocol was sealed.",
         "a protocol written after the data"),
    Rule("LC-4.3", "Pre-registration", FATAL,
         "The normative fields still hash to the seal.",
         "editing the protocol after sealing"),
    Rule("LC-4.4", "Pre-registration", FATAL,
         "The results were verified against the protocol.",
         "a seal nobody checked"),
    Rule("LC-4.5", "Pre-registration", FATAL,
         "No fatal protocol deviation was found.",
         "analysing under different parameters than were registered"),
    Rule("LC-4.6", "Pre-registration", WARNING,
         "No deviation from the sealed protocol was declared.",
         "undisclosed drift from the plan"),
    Rule("LC-4.7", "Pre-registration", WARNING,
         "The seal carries an external anchor.",
         "ordering resting on the author's own clock"),

    # --- LC-5: one-command reproduction ------------------------------------
    Rule("LC-5.1", "One-command reproduction", FATAL, "A reproduction command is given.",
         "unfalsifiability by inaccessibility"),
    Rule("LC-5.2", "One-command reproduction", WARNING,
         "Reproduction is one command, not a pipeline.",
         "a recipe only its author can follow"),
    Rule("LC-5.3", "One-command reproduction", FATAL, "A commit is pinned.",
         "an unversioned result"),
    Rule("LC-5.4", "One-command reproduction", FATAL,
         "The producing source tree was clean.",
         "a pinned commit that does not identify the code that ran"),
    Rule("LC-5.5", "One-command reproduction", WARNING, "A container image is pinned.",
         "an irreproducible software environment"),
    Rule("LC-5.6", "One-command reproduction", FATAL, "The hardware is documented.",
         "a result that cannot be situated"),
    Rule("LC-5.7", "One-command reproduction", WARNING, "A runtime estimate is given.",
         "a reproducer unable to budget for the attempt"),

    # --- LC-Q: measurement quality -----------------------------------------
    Rule("LC-Q1", "Measurement quality", WARNING, "Replicate count meets the floor.",
         "an underpowered design"),
    Rule("LC-Q2", "Measurement quality", WARNING, "Measurements were interleaved.",
         "drift confounded with the effect"),
    Rule("LC-Q3", "Measurement quality", WARNING, "Per-system cell coverage is high.",
         "a sparse envelope described as full"),
    Rule("LC-Q4", "Measurement quality", WARNING,
         "Measurement noise is small relative to the claimed effect.",
         "an effect indistinguishable from the noise floor"),
    Rule("LC-Q5", "Measurement quality", WARNING, "Limitations are stated.",
         "a claim presented without its bounds"),
    Rule("LC-Q6", "Measurement quality", FATAL,
         "Equivalence is established somewhere when the design has the power to.",
         "an inverted or broken equivalence test, which makes every tie look inconclusive"),
)

# --------------------------------------------------------------------------
# LC-1.1 additions
# --------------------------------------------------------------------------
#
# LC-1.0 above is not edited. Every rule it defines keeps its identifier, its
# severity and its meaning, and `LC_1_0_DIGEST` still verifies against it, so a
# claim graded under 1.0 still means exactly what it meant.
#
# These seven rules exist because measurement found seven new ways to be wrong
# that 1.0 could not express. Each was discovered as a concrete defect in this
# project's own work before it was written as a rule, and each ships with an
# artifact that violates it.

LC_1_1_ADDITIONS: tuple[Rule, ...] = (
    Rule("LC-6.1", "Semantic state", FATAL,
         "Feasibility is a state, never a sentinel value.",
         "an out-of-memory configuration encoded as a throughput of zero, which "
         "comparison paths then skip"),
    Rule("LC-6.2", "Semantic state", FATAL,
         "A cell that did not run carries no measurement.",
         "'did not run' and 'ran and measured zero' collapsing into one number"),
    Rule("LC-6.3", "Semantic state", FATAL,
         "A completed run served from host memory is not device-feasible.",
         "a run that thrashes for hours being recorded as fitting"),
    Rule("LC-7.1", "Model validation", FATAL,
         "Calibration and validation data are disjoint and the split is recorded.",
         "a model graded on the data that chose its parameters"),
    Rule("LC-7.2", "Model validation", FATAL,
         "No parameter is fitted after the model is frozen.",
         "a validation set quietly becoming a training set"),
    Rule("LC-7.3", "Model validation", FATAL,
         "A parameter that failed its quality gate is not used.",
         "a rejected fit promoted into an active model because nobody re-read "
         "its provenance"),
    Rule("LC-7.4", "Model validation", WARNING,
         "A fit over a log-scaled domain reports per-regime error, not only a "
         "pooled figure.",
         "one useless regime hiding behind three good ones, and R^2 dominated "
         "by the largest points"),
)

_RULES: tuple[Rule, ...] = LC_1_0_RULES + LC_1_1_ADDITIONS

RULES: dict[str, Rule] = {r.id: r for r in _RULES}

REQUIREMENT_ORDER = (
    "Standard version",
    "Evidence class",
    "Loss column",
    "Tuning-budget parity",
    "Envelope, not point",
    "Pre-registration",
    "One-command reproduction",
    "Measurement quality",
    "Semantic state",
    "Model validation",
)


def frozen_table(rules: tuple[Rule, ...] | None = None) -> list[list[str]]:
    """The frozen part of a rule set, in a stable order."""
    return sorted([list(r.key()) for r in (rules if rules is not None else _RULES)])


def compute_digest() -> str:
    """Digest of the CURRENT standard revision."""
    return content_hash({"standard": "LC-1.1", "rules": frozen_table()})


def compute_lc_1_0_digest() -> str:
    """Digest of LC-1.0, unchanged.

    Asserted by test alongside the 1.1 digest. A claim graded under 1.0 must go
    on meaning what it meant, so 1.0's table is verified independently of
    whatever 1.1 adds.
    """
    return content_hash({"standard": "LC-1.0", "rules": frozen_table(LC_1_0_RULES)})


# Asserted by tests/test_standard_frozen.py. Changing a rule id or a severity
# changes this hash, which fails that test, which forces a version bump rather
# than a silent edit.
# LC-1.0, unchanged since it was frozen. Not to be edited.
LC_1_0_DIGEST = "sha256:1483dc1b908fc25155cdf160dd85c04a6d2b995bac3d61ab074446dcc9468df3"

# The current revision. Adding a rule is a minor bump and a new digest;
# changing an existing rule's id or severity would be a major bump, and the
# LC-1.0 digest above exists so that such a change cannot pass unnoticed.
FROZEN_DIGEST = "sha256:c167b1a695a81a4cb388142f277c42cdc5667ac31661c2ff92947c6a36ab5684"

N_RULES = len(_RULES)
N_FATAL = sum(1 for r in _RULES if r.severity == FATAL)


def get(rule_id: str) -> Rule:
    try:
        return RULES[rule_id]
    except KeyError:
        raise KeyError(
            f"{rule_id!r} is not a registered rule of LC-1.0. Emitting an unregistered "
            "rule would put a grade on an artifact that the standard does not define; "
            "add it to losscolumn.spec.registry and bump the version."
        ) from None


def by_requirement() -> dict[str, list[Rule]]:
    out: dict[str, list[Rule]] = {k: [] for k in REQUIREMENT_ORDER}
    for r in _RULES:
        out.setdefault(r.requirement, []).append(r)
    return {k: sorted(v, key=lambda r: r.id) for k, v in out.items() if v}


def to_markdown() -> str:
    lines = [
        "# LC-1.1 rule registry (frozen)",
        "",
        f"{N_RULES} rules, {N_FATAL} fatal. Digest `{compute_digest()}`.",
        "",
        f"Supersedes LC-1.0 (`{compute_lc_1_0_digest()}`), which is unchanged: "
        f"LC-1.1 adds {len(LC_1_1_ADDITIONS)} rules "
        f"({', '.join(r.id for r in LC_1_1_ADDITIONS)}) and edits none. A claim "
        "graded under 1.0 still means what it meant.",
        "",
    ]
    for req, rules in by_requirement().items():
        lines += [f"## {req}", "", "| Rule | Severity | Requirement | Forecloses |",
                  "|------|----------|-------------|------------|"]
        for r in rules:
            lines.append(
                f"| `{r.id}` | {r.severity} | {r.summary} | {r.forecloses or '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def to_dict() -> dict[str, Any]:
    return {
        "standard": "LC-1.1",
        "digest": compute_digest(),
        "supersedes": "LC-1.0",
        "lc_1_0_digest": compute_lc_1_0_digest(),
        "added_in_1_1": [r.id for r in LC_1_1_ADDITIONS],
        "n_rules": N_RULES,
        "n_fatal": N_FATAL,
        "rules": [
            {
                "id": r.id,
                "requirement": r.requirement,
                "severity": r.severity,
                "summary": r.summary,
                "forecloses": r.forecloses,
            }
            for r in sorted(_RULES, key=lambda x: x.id)
        ],
    }
