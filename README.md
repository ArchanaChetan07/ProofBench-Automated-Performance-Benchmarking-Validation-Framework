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

Five requirements, each targeting a documented failure mode, each implemented
as an executable conformance rule rather than as prose.

| | Requirement | What it forecloses |
|---|-------------|--------------------|
| **LC-1** | Loss column | Selective reporting |
| **LC-2** | Tuning-budget parity | The untuned baseline |
| **LC-3** | Envelope, not point | Overgeneralisation |
| **LC-4** | Pre-registration | Post-hoc selection |
| **LC-5** | One-command reproduction | Unfalsifiability by inaccessibility |

```bash
losscolumn validate anyones-claim.json
```

The validator grades **documents, not objects**. Nothing in it is specific to
this project's subject matter.

### Three worked reference implementations

| Thrust | Question | Harness |
|--------|----------|---------|
| **I** | Where does communication–computation overlap break down, and do recommended sharding configurations sit next to a cliff? | [`thrusts/overlap`](src/losscolumn/thrusts/overlap) |
| **II** | How much engine-to-engine difference survives an identical tuning budget? | [`thrusts/engines`](src/losscolumn/thrusts/engines) |
| **III** | Where does a faithful from-scratch attention kernel lose to the reference, and why? | [`thrusts/kernel`](src/losscolumn/thrusts/kernel) |

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
losscolumn prereg seal I && losscolumn prereg seal II && losscolumn prereg seal III
losscolumn run all
losscolumn validate artifacts/*.claim.json
```

Requirement LC-5 would be hollow if this project's own artifacts were not
produced that way. Each claim records the exact invocation that made it, the
commit, the image, and the hardware.

## Tests

```bash
pytest -q
```

The tests that matter are the ones that verify the detectors detect: a planted
loss region is recovered with the right bounds; a flat surface produces no
region; a smoothly declining surface produces no cliff; a deliberately broken
kernel fails the correctness gate; an underpowered design flags itself. Every
validator rule is exercised by a document that violates it — a conformance
checker that has never been shown a non-conforming document is a rubber stamp.

## Status and honesty

This is a research instrument, not a benchmark result. The artifacts in
`artifacts/` are produced from the calibratable models plus real single-GPU
measurement for Thrust III, and they say so on their front pages. The numbers
that would appear in a paper come from the funded allocation; the analysis,
the standard and the conformance machinery are what is finished here.

## License

Apache-2.0.
