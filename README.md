# losscolumn

**Falsifiable performance claims for LLM training and inference systems.**

A reporting standard, and its reference implementation, for a field where
performance claims are largely unfalsifiable. A paper or vendor benchmark
reports one configuration, on one hardware target, against a baseline the
authors had no incentive to tune, and omits the regions where the method loses.
The reader cannot tell whether a reported 2× speedup is a contribution or an
artifact of the comparison.

This package makes the alternative concrete. Every result it produces carries a
**loss column**: an explicit, statistically supported map of where the method
underperforms, published above the wins, with an attributed cause for each
region.

```bash
pip install -e ".[torch]"

losscolumn doctor                # what can this machine actually measure?
losscolumn prereg seal III       # seal the protocol BEFORE collecting data
losscolumn run thrust3           # run it, publish a conforming artifact
losscolumn validate artifacts/*.claim.json
```

---

## The idea in one table

Every result this package emits looks like this. It is not an appendix; it is
the first thing on the page.

| # | Where it loses | Cells | Median | Worst | max q | Attributed to |
|---|----------------|-------|--------|-------|-------|---------------|
| 1 | `micro_batch ≤ 2 and seq_len=512 and world_size=32` | 4/4 (8%) | +126.9% | +258.6% | 0.0022 | overlap efficiency 0.45; 54.5% of the step is exposed communication; beaten by `hybrid_shard/tp1` |
| 2 | `micro_batch ≥ 2 and seq_len=8192 and checkpointing=False` | 6/6 (12%) | cannot run | cannot run | — | OOM: needs ~101 GB/rank against 80 GB HBM |

Three things distinguish this from a limitations paragraph:

- **It is derived, not written.** Regions come from greedy maximal-box covering
  over the factor lattice, seeded from the worst uncovered losing cell. Nobody
  chose which losses to mention.
- **It is statistically supported.** Paired sign-flip permutation tests, BCa
  bootstrap intervals, Benjamini–Hochberg FDR control across all cells, and a
  minimum effect size fixed in advance.
- **It knows when it cannot see.** A sweep that lacks the power to detect a
  loss says so, loudly, instead of reporting a clean bill of health.

---

## What is here

### The standard — [`docs/STANDARD.md`](docs/STANDARD.md)

**LC-1.2.** Eight requirements, each targeting a documented failure mode, each
implemented as an executable conformance rule rather than as prose. LC-1.0 and
LC-1.1 are frozen and unedited: every revision adds rules and changes none, so a
claim graded under an earlier one still means what it meant.

| | Requirement | What it forecloses | Since |
|---|-------------|--------------------|-------|
| **LC-1** | Loss column | Selective reporting | 1.0 |
| **LC-2** | Tuning-budget parity | The untuned baseline | 1.0 |
| **LC-3** | Envelope, not point | Overgeneralisation | 1.0 |
| **LC-4** | Pre-registration | Post-hoc selection | 1.0 |
| **LC-5** | One-command reproduction | Unfalsifiability by inaccessibility | 1.0 |
| **LC-6** | Semantic state | Infeasibility encoded as a number | 1.1 |
| **LC-7** | Model validation | A model graded on the data that chose it | 1.1 |
| **LC-8** | Session comparability | Two good sessions averaged into a bad surface | 1.2 |

The last three were not designed. Each was forced by a defect that got through
the revision before it, found in this repository's own work — which is the
intended way for the standard to grow, since a rule with no artifact behind it
is a guess about what might go wrong.

```bash
losscolumn validate anyones-claim.json
losscolumn standard export          # schemas, rules, coverage matrix, corpus
```

The validator grades **documents, not objects**. Nothing in it is specific to
this project's subject matter.

**What "frozen" means here.** Rule identifiers and severities carry meaning
into other people's artifacts: a claim graded `conforming` last year has to
mean the same thing this year. So the freeze is structural rather than a
promise. Every rule lives in one registry with its severity; the validator
looks severity up there instead of passing it at the call site, which is how
`LC-2.1` came to be `info` on one code path and `fatal` on another before the
registry existed. A content hash over the table is asserted by a test, so
changing an id or a severity fails CI and forces a version bump.

| | |
|---|---|
| Rules | **47** (28 fatal, 17 warning, 2 informational) |
| Document schemas | 4, frozen: envelope, loss column, claim, protocol seal |
| Adversarial corpus | **46** deliberately defective artifacts, all caught |
| Rule coverage | **47/47** have both a passing case and a failing case |

The corpus ships **in the package**, not in `tests/`. Anyone implementing
the standard in another language can run their validator against those documents and
check that it reaches the same verdicts; a standard whose only implementation
is its author's is not a standard. See
[`artifacts/standard/validator-coverage.md`](artifacts/standard/validator-coverage.md).

### Three worked reference implementations

| Thrust | Question | Harness |
|--------|----------|---------|
| **I** | Where does communication–computation overlap break down, and do recommended sharding configurations sit next to a cliff? | [`thrusts/overlap`](src/losscolumn/thrusts/overlap) |
| **II** | How much engine-to-engine difference survives an identical tuning budget? | [`thrusts/engines`](src/losscolumn/thrusts/engines) |
| **III** | Over (seq_len x batch x sparsity pattern x density x phase), where does a from-scratch attention implementation lose to the reference computing the same pattern, and why? | [`thrusts/kernel`](src/losscolumn/thrusts/kernel) |

---

## Three ideas worth stealing

Even if you never adopt the standard, three pieces of this are reusable.

### 1. A sweep can be arithmetically incapable of finding anything

An exact paired sign-flip test over `r` replicates cannot produce a p-value
below `2⁻ʳ`. Benjamini–Hochberg over `m` cells rejects the most extreme only if
`p ≤ q/m`. So:

```python
from losscolumn.core.stats import design_can_reject
design_can_reject(replicates=5, n_tests=48, q=0.05)
# {'can_reject': False, 'min_replicates': 10,
#  'message': 'design cannot reject: with r=5 the smallest attainable p-value
#              is 0.031, but Benjamini-Hochberg over 48 cells requires
#              p <= 0.001 for the most extreme cell...'}
```

A 48-cell sweep at 5 replicates will report an empty loss column against a
**+171% regression**. This is not hypothetical — it happened while building
this package, and the check exists because of it. `LC-1.8` makes it fatal.

### 2. Overlap efficiency is a set-theoretic quantity

Summing NCCL kernel durations double-counts concurrent collectives; subtracting
summed compute from summed communication can go negative. The honest
computation is interval algebra:

```python
exposed = comm_intervals - compute_intervals      # set difference
overlap_efficiency = 1 - exposed.measure() / comm_intervals.measure()
```

Always in `[0, 1]`, exact, `O(n log n)`. See
[`core/intervals.py`](src/losscolumn/core/intervals.py).

### 3. Tolerances derived from the arithmetic, not fitted to the result

A correctness tolerance chosen by running the test and rounding up until it
passes is not a tolerance; it is a record of the bug you decided to keep. With
fp32 accumulation the error budget is `4u_storage + 8·√seq·u_accumulator`. The
loose alternative — assuming the accumulator has the storage format's precision
— yields a tolerance ~150× the error a correct kernel shows, and would happily
accept a kernel that accumulates in fp16.

The test suite verifies this by feeding the checker a deliberately broken
fp16-accumulating kernel and asserting it **fails**.

---

## Hardware tiering

Every harness is written against a backend protocol with three tiers, and every
artifact states which one produced it:

| Tier | Runs on | `evidence_class` |
|------|---------|------------------|
| `synthetic` | Anything, including CI with no GPU | `simulated` |
| local | Whatever accelerator is present | `measured` |
| cluster | The funded 8×A100 / 8×H100 allocation | `measured` |

The synthetic backends are **calibratable models**, not toys: their parameters
(achievable FLOP/s, bus bandwidth, collective latency, per-step scheduler
overhead) are exactly the quantities `nccl-tests` and a microbenchmark measure,
and `calibrate()` replaces them with measured values. Where the calibrated
model mispredicts is itself the interesting systems result.

This directly implements the proposal's own risk mitigation — validate the
measurement pipeline on cheap hardware before committing multi-GPU time — one
level further down: validate the *analysis* before committing any GPU time.

A `simulated` claim is banner-flagged at the top of its page and is a fatal
conformance error if it does not explain itself.

### Turning a simulation into a prediction that gets checked

```bash
losscolumn calibrate thrust1
```

A simulated claim says something weaker than a measured one: *if this model is
right, the method loses here*. That is a prediction, and a prediction is worth
the track record of the model that made it — which, before any calibration, is
nothing.

So Thrust I's question is restricted to the axes one GPU can vary, the model is
asked for its prediction **before** anything is measured, and the predicted loss
map is then scored against the measured one on six axes: region overlap,
per-axis boundary displacement, false wins (the model called it safe and it
regresses), false losses, cliff agreement, and calibration error on the effect
rather than the verdict.

The correction that falls out is computed and deliberately **not applied**. A
model refitted on the measurements used to validate it has been fitted, not
calibrated; the update belongs in a newly sealed protocol, evaluated on cells
the calibration study did not touch. See
[`docs/FROM_SIMULATION_TO_SCIENCE.md`](docs/FROM_SIMULATION_TO_SCIENCE.md).

What this does **not** do is as important. It calibrates the compute half of
the model completely and the communication half not at all — and Thrust I's
loss regions are driven by exposed communication. The study makes that thrust
better founded, not founded, and the report says so in those words.

---

## Layout

```
src/losscolumn/
  core/          hardware-free analysis; numpy only
    envelope.py    factorial measurement grids, replicates kept
    stats.py       sign-flip tests, BCa bootstrap, BH-FDR, design power
    losscolumn.py  loss-region extraction  ← the flagship algorithm
    cliffs.py      discontinuity detection
    pareto.py      frontiers and attainment curves
    parity.py      tuning ledgers and certification
    prereg.py      sealing and verification
    intervals.py   exact interval algebra
  spec/          the claim document model
  validate/      the standard, as a program
  report/        SVG loss maps, timelines, frontiers; standalone HTML
  thrusts/       the three harnesses
```

`losscolumn.core` depends on **numpy alone**, so the analysis layer installs
and runs on a laptop, in CI, and on a login node with no CUDA and no network.

---

## Reproducing this repository's own artifacts

```bash
scripts/reproduce.sh --full
```

That is the honest version: it clones the repository into a fresh directory,
builds an isolated environment, installs from the pinned commit, runs the
pipeline, and compares what comes out against what is tracked here. Running the
pipeline in the working tree proves it works on a machine that already has
everything it needs; the failures worth catching — uncommitted files, unpinned
dependencies, paths that exist only on one disk — are invisible from inside the
tree, and the script refuses to start from a dirty one.

Structure is compared, not numbers. Two benchmark runs on the same machine
never produce identical timings, and a check that demanded they did would fail
every time and be switched off within a week. What must reproduce is the
standard version, the factor lattice, the protocol seal and the conformance
verdict.

The individual steps, if you want them:

```bash
losscolumn doctor                     # what can this machine actually measure?
losscolumn prereg seal III            # commit to a protocol before collecting data
scripts/measure_all.sh                # every measurement, in series
losscolumn index                      # regenerate the artifact index
```

`measure_all.sh` runs in series deliberately. Two benchmark processes on one
device contend at the scheduler and in cache and produce timings that are wrong
by an amount nothing downstream can detect — the replicates stay
self-consistent and the intervals stay tight. A lock enforces it; this happened
here once and both sets of numbers had to be discarded.

## Tests

```bash
pytest -q
```

The tests that carry weight are the ones that check the machinery can still
tell truth from falsehood:

- **Detectors, against ground truth.** Five surfaces built so the right answer
  is known by construction: a steep but smooth decline yields no cliff at any
  slope; noise yields none across seeds and noise levels; multiple cliffs are
  found independently; and a catastrophic cliff does not mask a moderate one
  next to it. A planted loss region is recovered with the right bounds, and an
  unperturbed surface produces none.
- **The validator, against defects.** All 46 corpus artifacts, each breaking
  one rule in one named way, plus a passing case for every rule.
- **The freeze.** Registry and schema digests, asserted literally.

## Status and honesty

This is a research instrument, not a benchmark result.

**Thrust III is measured.** Both implementations were swept over the full
registered lattice — 144 cells, 80 of them applicable configurations — at 11
replicates, on a local sm_75 card. Both claims grade **conforming with zero
fatal findings and zero warnings**, and both pass **80/80** correctness shapes
against an fp64 ground truth before any timing.

| | portable (eager ops) | Triton (fused, block-skipping) |
|---|---|---|
| Prefill speedup, median | 0.38x | **5.76x** |
| Decode speedup, median | 0.96x | **3.46x** |
| Losing cells | 44 / 80 | **8 / 80** |
| Worst case | 493x slower | 69x slower |
| Correctness | 80/80 | 80/80 |

Both lose in the same place — the **dense** pattern, where there are no blocks
to skip and the comparison is purely code generation against a fused kernel
that has had years of work. Everywhere sparsity is real, block-skipping wins.

**Two findings from this repository's own artifacts are worth more than the
numbers.**

*The baseline was unfair, and the loss column found it.* An earlier revision
routed the reference through an explicit attention mask for **every** pattern,
including dense — and an explicit mask precludes torch's fused backend. Against
that handicapped baseline the Triton kernel reported **zero losses**: a clean
sweep. Giving the reference the backend torch would have chosen turned that
into **8 losses, worst case 69x**. The empty loss column was an artifact of the
comparator, which is the exact failure this project exists to make visible.

*The device has no tensor cores, and compute capability does not say so.* The
T1000 reports sm_75, but TU117 ships without the units: measured here, fp16
GEMM runs at 0.33 TFLOP/s against 1.83 for fp32 — five times **slower**, on an
emulated path. Every timing above is fp16. The comparison stays internally
valid because both arms took the same path, but the absolute rates do not
transfer to a tensor-core device and neither necessarily does the ordering.
This is detected by microbenchmark and printed as the first limitation on both
claims.

**Thrusts I and II are simulated** and stamped `evidence_class="simulated"`,
with a banner on their own front page. The calibration study for Thrust I ran
and reports **ESTABLISHES NOTHING**: neither the predicted nor the measured map
contained a loss, so its region overlap of 1.00 was two empty sets agreeing,
and no slope could be fitted. That is the LC-1.8 failure turned on the
calibration itself, and it is stated at the top of the report rather than
buried under statistics that look like agreement. What the study *does*
establish stands: the model's flat 4/3 activation-checkpointing multiplier
overstates the measured penalty of 1.22x.

The numbers that would appear in a paper come from the funded allocation on
Hopper, where the FA-3 mechanisms this kernel cannot express actually exist.
The standard, the analysis and the conformance machinery are what is finished
here.

If Triton fails to import under Anaconda on Windows, see
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — the cause is a stale bundled
Visual C++ redistributable, not the Triton package.

## License

Apache-2.0.
