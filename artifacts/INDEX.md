# Artifact index

Generated 2026-08-31T04:16:16Z by `losscolumn index`. Do not edit: an index maintained by hand drifts from the files it describes, and an index that lists an artifact nobody produced is the kind of unfalsifiable furniture this project exists to argue against.

**4 claims** &mdash; 2 conforming, 2 measured, 2 simulated.

---

## STANDARD

LC-c167b1a695a8... &mdash; **54 rules**, 34 fatal.

| Document | What it is |
|---|---|
| [`rules.md`](standard/rules.md) | Every rule, its severity, and the failure mode it forecloses |
| [`validator-coverage.md`](standard/validator-coverage.md) | Which rules have a passing case and a failing case |
| [`adversarial-corpus.json`](standard/adversarial-corpus.json) | Deliberately defective artifacts, for checking another implementation |
| [`schema/`](standard/schema) | The four frozen document shapes as JSON Schema |

Registry digest `sha256:c167b1a695a81a4cb388142f277c42cdc5667ac31661c2ff92947c6a36ab5684`  
Schema digest `sha256:49729bd2610b893b183ba5aa27a7df0f7a6319ea496c9265ae172b43a595f529`

Both are asserted by `tests/test_standard_frozen.py`. Changing a rule id or a severity fails that test, which makes a version bump a deliberate act rather than an edit that slips through.

---

## PROTOCOLS

### Thrust I -- the overlap envelope

**Registered protocol** &mdash; [`prereg-thrust-I.json`](prereg/prereg-thrust-I.json) (revision 1), sealed 2026-08-28T22:48:06Z  
MDE 5% &middot; q 0.05 &middot; 11 replicates &middot; seal `b86a36613c8bab14...`

| Claim | Evidence | Grade | Loss cells | Artifacts |
|---|---|---|---|---|
| **The overlap envelope: where the recommended sharding configuration breaks down**<br><code>full_shard/tp1</code> vs <code>best_of_swept</code> | simulated | conforming-with-warnings (1w) | 12/48 | [conformance.json](thrust1-overlap-envelope.conformance.json) [html](thrust1-overlap-envelope.html) [md](thrust1-overlap-envelope.md) |

### Thrust II -- the equal-tuning audit

**Registered protocol** &mdash; [`prereg-thrust-II.json`](prereg/prereg-thrust-II.json) (revision 1), sealed 2026-08-28T22:48:06Z  
MDE 5% &middot; q 0.05 &middot; 11 replicates &middot; seal `aad2fa03dfc12a86...`

| Claim | Evidence | Grade | Loss cells | Artifacts |
|---|---|---|---|---|
| **The equal-tuning audit: engine differences that survive budget parity**<br><code>vllm</code> vs <code>best_alternative</code> | simulated | conforming-with-warnings (2w) | 18/24 | [conformance.json](thrust2-equal-tuning-audit.conformance.json) [html](thrust2-equal-tuning-audit.html) [md](thrust2-equal-tuning-audit.md) |

### Thrust III -- sparse attention, prefill and decode

**Registered protocol** &mdash; [`prereg-thrust-III.json`](prereg/prereg-thrust-III.json) (revision 2), sealed 2026-08-29T07:01:56Z  
MDE 10% &middot; q 0.05 &middot; 11 replicates &middot; seal `fd87925555851b65...`

Superseded and still verifiable: [`prereg-thrust-III-v1-20260828T224806Z.json`](prereg/archive/prereg-thrust-III-v1-20260828T224806Z.json)

| Claim | Evidence | Grade | Loss cells | Artifacts |
|---|---|---|---|---|
| **Sparse attention, portable implementation: a loss map over prefill and decode**<br><code>implementation</code> vs <code>reference</code> | measured | conforming | 44/80 | [conformance.json](thrust3a-portable-attention.conformance.json) [html](thrust3a-portable-attention.html) [md](thrust3a-portable-attention.md) |
| **Sparse attention, triton implementation: a loss map over prefill and decode**<br><code>implementation</code> vs <code>reference</code> | measured | conforming | 8/80 | [conformance.json](thrust3b-triton-attention.conformance.json) [html](thrust3b-triton-attention.html) [md](thrust3b-triton-attention.md) |

---

## CALIBRATION

Evidence *about* a claim rather than a claim. A simulated thrust predicts a loss map; a calibration study measures a subset of it and reports how far the prediction can be trusted.

- [`calibration-thrust1-communication.json`](calibration-thrust1-communication.json) &mdash; 
- [`calibration-thrust1-compute.json`](calibration-thrust1-compute.json) &mdash; ESTABLISHES NOTHING: this study could not have detected a disagreement. neither the predicted nor the measured map contains a loss, so the region overlap of 1.00 is two empty sets agreeing and the boundary statistics are undefined; the predicted effects have no spread, so no slope can be fitted and the model was checked at a single operating point. A wider grid, or one chosen so the model predicts a loss somewhere, is needed before any statement about the model's reliability is supported.
- [`calibration-thrust1-memory-v2.json`](calibration-thrust1-memory-v2.json) &mdash; 
- [`calibration-thrust1-memory.json`](calibration-thrust1-memory.json) &mdash; REGION STATISTICS ONLY: 5 false win(s) and 0 false loss(es) over 8 cells, overlap 0.17. The magnitude statistics are absent by construction: this is a feasibility study, so the prediction is a classification and there is no effect size to regress. EVERY false win is a configuration the model cleared and the hardware refused.

---

## WITHDRAWN AND DISCARDED

Preserved rather than deleted. Nothing in this section is a claim, and nothing here carries a conformance grade: grading a withdrawn artifact would invite it to be cited.

Integrity: **all preserved files match their withdrawal digests**.

| Withdrawn | Reason | Superseded by | Record |
|---|---|---|---|
| Sparse attention, Triton implementation (run of 2026-08-29 11:09) | `unfair-comparison` | `lc-thrust3b-triton-attention @ 0d6d57b` | [NOTE.md](history/2026-08-29-unfair-baseline/NOTE.md) |
| Thrust I compute-model calibration (run of 2026-08-29 18:20) | `vacuous-result` | `calibration-thrust1-compute @ 0d6d57b` | [NOTE.md](history/2026-08-29-vacuous-calibration/NOTE.md) |

**Discarded runs** &mdash; no artifact was published, so there is nothing to withdraw; what is recorded is that the run happened and what changed as a result.

- [Two benchmark processes sharing one GPU](history/incidents/2026-08-29-contended-gpu.md) &mdash; 2026-08-29T10:16:50-07:00 / 10:22:22-07:00

---

## REPRODUCTION

```bash
scripts/reproduce.sh --full
```

Clones the repository into a fresh directory, builds an isolated environment, installs from the pinned commit, runs the pipeline, and compares what comes out against what is tracked here.

It runs from a clone rather than the working tree on purpose. Running in place proves the pipeline works on a machine that already has everything it needs; the failures worth catching -- uncommitted files, unpinned dependencies, paths that exist only on one disk -- are invisible from inside the tree.

Structure is compared, not numbers. Two benchmark runs on the same machine never produce identical timings, and a check that demanded they did would fail every time and be switched off. What must reproduce is the standard version, the factor lattice, the protocol seal and the conformance verdict.
