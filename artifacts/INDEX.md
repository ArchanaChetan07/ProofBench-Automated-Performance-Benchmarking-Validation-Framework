# Artifact index

Generated 2026-08-29T17:45:00Z by `losscolumn index`. Do not edit: an index maintained by hand drifts from the files it describes, and an index that lists an artifact nobody produced is the kind of unfalsifiable furniture this project exists to argue against.

**2 claims** &mdash; 0 conforming, 0 measured, 2 simulated.

---

## STANDARD

LC-1483dc1b908f... &mdash; **47 rules**, 28 fatal.

| Document | What it is |
|---|---|
| [`rules.md`](standard/rules.md) | Every rule, its severity, and the failure mode it forecloses |
| [`validator-coverage.md`](standard/validator-coverage.md) | Which rules have a passing case and a failing case |
| [`adversarial-corpus.json`](standard/adversarial-corpus.json) | Deliberately defective artifacts, for checking another implementation |
| [`schema/`](standard/schema) | The four frozen document shapes as JSON Schema |

Registry digest `sha256:1483dc1b908fc25155cdf160dd85c04a6d2b995bac3d61ab074446dcc9468df3`  
Schema digest `sha256:0a5260d50e842389dcd01008f1cfba0cdb3bda1aa54b6ce32a099c8dea52b9ea`

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

_No claim published._

---

## REPRODUCTION

```bash
scripts/reproduce.sh --full
```

Clones the repository into a fresh directory, builds an isolated environment, installs from the pinned commit, runs the pipeline, and compares what comes out against what is tracked here.

It runs from a clone rather than the working tree on purpose. Running in place proves the pipeline works on a machine that already has everything it needs; the failures worth catching -- uncommitted files, unpinned dependencies, paths that exist only on one disk -- are invisible from inside the tree.

Structure is compared, not numbers. Two benchmark runs on the same machine never produce identical timings, and a check that demanded they did would fail every time and be switched off. What must reproduce is the standard version, the factor lattice, the protocol seal and the conformance verdict.
