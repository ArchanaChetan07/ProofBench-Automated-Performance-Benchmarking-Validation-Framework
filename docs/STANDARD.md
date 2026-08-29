# The Loss Column Reporting Standard

**Version LC-1.0**

A conforming performance claim about a machine-learning system ships five
things. Each requirement targets a specific, documented failure mode in how
systems performance is currently reported. None of them expresses a general
preference for rigour, and none of them is satisfied by an assurance in prose:
every requirement below is checkable by a program, and
[`losscolumn/validate/validator.py`](../src/losscolumn/validate/validator.py)
is that program.

The standard is deliberately narrow. It does not tell you what to measure, how
to design a kernel, or which engine to prefer. It tells you what a claim must
carry before a reader can decide whether to believe it.

---

## Why a standard rather than a checklist

Three structural features make performance claims in this field resistant to
falsification, and they are not fixed by trying harder.

**The untuned-baseline problem.** Authors optimise their own system and compare
against a baseline at default settings. Effort asymmetry alone can manufacture
the entire reported gap, and nothing in a typical paper lets a reader separate
the two.

**Point estimates for a high-dimensional surface.** Inference performance is a
function of at least batch size, sequence length, model architecture,
precision, KV-cache policy, hardware generation and concurrency. Reporting one
point in that space and generalising is not supported by the measurement.

**Systematic non-publication of losses.** No incentive exists to report the
regimes where a method regresses. The result is a literature of methods that
appear to strictly dominate, which cannot all be true simultaneously.

Each requirement below closes one of these, plus the two that make the others
enforceable.

---

## LC-1 — Loss column

> **Every claim ships an explicit map of the regions where the method
> underperforms, at equal prominence to the wins, with an attributed cause for
> each region.**

*Forecloses:* selective reporting.

### What conformance requires

| Rule | Requirement |
|------|-------------|
| `LC-1.1` | A loss column is present. |
| `LC-1.2` | If any cell lost, at least one region describes where. |
| `LC-1.3` | The column does not declare itself incomplete. |
| `LC-1.4` | The column is in the main body, not an appendix. |
| `LC-1.5` | Every region carries an attributed cause. *(warning)* |
| `LC-1.6` | Cells that could not be measured are accounted for. *(warning)* |
| `LC-1.7` | The inconclusive fraction is stated and under 35%. *(warning)* |
| `LC-1.8` | The design is *capable* of declaring a loss. |

### Equal prominence, operationally

"Equal prominence" is not a matter of taste. In this implementation the loss
column is rendered immediately below the headline and above every other
section, so a reader cannot reach the good news without passing the bad news.
`LC-1.4` fails any claim that marks its column as belonging to an appendix.

### The five verdicts

A cell is not win-or-lose. Each cell of the envelope receives one of:

| Verdict | Meaning |
|---------|---------|
| `loss` | Significantly worse, by at least the pre-registered minimum effect. |
| `win` | Significantly better, by at least that effect. |
| `tie` | Statistically *equivalent* within ±MDE, by two one-sided tests. |
| `inconclusive` | The data cannot distinguish the three above. |
| `missing` | Neither system could be measured here. |

`tie` has to be earned. A non-significant difference is **not** evidence of
equivalence, and reporting it as one is the most common statistical error in
benchmark papers. The `inconclusive` bucket is reported and counted, because an
envelope that is 40% inconclusive is a different object from one that is 5%.

A configuration that **cannot run at all** — out of memory, unsupported shape,
crash — is a loss at that cell, and the harshest kind. It is recorded with its
failure reason, never silently omitted.

### LC-1.8, and why it is the rule that matters most

The most dangerous artifact this standard can produce is a **conforming,
well-formatted, entirely empty loss column produced by a sweep that could never
have found anything.**

An exact paired sign-flip test over *r* replicates cannot produce a p-value
below `2^-r`, no matter how large the true effect. Benjamini–Hochberg over *m*
cells rejects the most extreme one only if its p-value is at most `q/m`.
Together:

```
r ≥ log₂(m / q)
```

For a 48-cell sweep at q = 0.05, that is **r ≥ 10**. A sweep at r = 5 will
report an empty loss column against a 171% regression — this is not
hypothetical; it happened during development of this package and is what
prompted the rule. `LC-1.8` is fatal, and every loss column carries the
arithmetic that distinguishes "we looked and found nothing" from "we could not
have found anything."

---

## LC-2 — Tuning-budget parity

> **Every system in a comparison receives the same number of configuration
> trials, from the same search procedure, against the same workload, on the
> same hardware, run by the same operator — and the record of that is
> published.**

*Forecloses:* the untuned baseline.

### Nominal versus effective parity

**Nominal parity** is equal trial counts, same procedure, same operator, same
hardware, same workload. It is checkable and is checked (`PAR-001` … `PAR-007`).

**Effective parity** is equal search *density*. Forty trials over a
three-dimensional space is a dense search; forty over a nine-dimensional one is
a sample. The certificate reports trials per dimension, fraction of space
covered, and tuning wall-clock, and raises a finding where they diverge
(`PAR-010`, `PAR-011`, `PAR-012`). It does not pretend to fix the asymmetry —
it makes it visible.

`PAR-012` deserves particular attention: an engine that must be recompiled per
configuration can consume 36× the wall-clock for the same trial count. Under a
fixed *time* budget rather than a fixed *trial* budget, the comparison would
come out differently, and a reader choosing under time pressure needs to know
which budget was held constant.

### Derived baselines

A baseline may be *derived* — "the best of the tuned alternatives at each
cell" — and then has no ledger of its own. It must declare
`supporting.baseline_derived_from`, and every constituent must have a ledger.
The requirement is that nothing in the comparison went untuned.

### Operator skill

Recorded, not eliminated. One operator running every search is the best
available control; two operators is a `PAR-003` warning. The tuning protocol
and every trial are published so a third party can re-tune any system and
contest the result — which is what a contested-result reserve is *for*.

---

## LC-3 — Envelope, not point

> **Results are surfaces. A single number is not permitted as a headline
> claim.**

*Forecloses:* overgeneralisation from one configuration.

At least two factors, at least eight cells, and a headline that carries its own
best, worst and median with the conditions that produce each. Unbounded
superlatives — "up to 2× faster" — raise `LC-3.6`, because an envelope cannot
support an unbounded claim.

In this implementation the headline is **derived from the loss column**, not
written by hand. It is therefore not possible to state a summary the
measurements do not support.

Replicates are kept, never pre-averaged. An artifact that ships only means
cannot be re-analysed by a skeptic, and every downstream statistic needs the
spread.

---

## LC-4 — Pre-registration

> **The protocol is published and sealed before data collection. The analysis
> is verified against the seal.**

*Forecloses:* post-hoc selection of the favourable comparison.

The seal is a content hash over the *normative* fields — hypotheses, metrics,
factor grids, the MDE, the FDR level, decision rules, exclusion rules, the
stopping rule. Prose rationale is deliberately outside the hash, so a typo fix
does not break the commitment.

Verification answers three questions:

1. Is this the document that was sealed? (hash match)
2. Was the data collected after it was sealed? (timestamp ordering)
3. Was the analysis the one registered? (parameter match)

**Undeclared deviations are fatal. Declared ones are not** — research changes —
but they appear in the artifact next to the result they affect.

### The architectural point

Pre-registration is a filing exercise unless the analysis actually *reads* its
parameters from the seal. In this implementation it does: the MDE, the FDR
level and the replicate count are inputs to the sweep, taken from the sealed
protocol, so there is exactly one place those numbers live. An analysis that
silently used a different effect size than the one registered is not merely
detected after the fact — it cannot happen.

---

## LC-5 — One-command reproduction

> **One command, a pinned commit, a pinned image, documented hardware, and a
> runtime estimate.**

*Forecloses:* unfalsifiability by inaccessibility.

A dirty working tree is fatal: the pinned commit no longer identifies the code
that ran. "Dirty" is evaluated over source paths only — a run writes artifacts,
and a dirtiness check that always fires teaches people to ignore it.

A reproduction that is a shell pipeline rather than one command raises
`LC-5.2`. A missing runtime estimate raises `LC-5.7`: a reproducer cannot
otherwise tell whether this is ten minutes or ten days.

---

## Evidence class

Beyond the five requirements, every claim declares `evidence_class`:
`measured`, `simulated` or `mixed`. A model-generated number and a measured one
are both useful and are not the same kind of evidence. A `simulated` or `mixed`
claim must also carry an `evidence_note` saying what was modelled and what was
measured (`LC-0.3`, fatal), and the renderer banners it at the top of the page.

This is the cheapest possible defence against the two being confused later —
by a reader, or by the author six months on.

---

## Grades

| Grade | Meaning |
|-------|---------|
| `conforming` | No fatal findings, no warnings. |
| `conforming-with-warnings` | No fatal findings. |
| `non-conforming` | At least one fatal finding. |

Quality rules (`LC-Q*`) never block conformance on their own, but they are
printed in the same table, because a conforming artifact whose measurement
noise exceeds its claimed effect is conforming and useless.

---

## Adopting the standard

The validator grades **documents, not objects**. A claim produced by anyone's
tooling is gradable:

```bash
losscolumn validate their-claim.json
```

The document schema is whatever `Claim.to_dict()` emits; the fields the
validator reads are exactly those listed in the rule tables above. Nothing in
the validator is specific to attention kernels, serving engines or sharding —
those are the three worked reference implementations, not the scope.

## What this standard does not do

It does not make a measurement correct. It does not detect a mis-specified
model, a mis-attributed cause, or a benchmark that measures the wrong thing.
It constrains what a claim must *disclose*, which is a narrower and more
achievable goal than constraining what is true. A conforming claim can still
be wrong — but it can be *shown* to be wrong, which is the point.
