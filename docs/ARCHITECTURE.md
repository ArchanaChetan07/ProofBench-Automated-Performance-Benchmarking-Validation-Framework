# Architecture

Three layers, with a strict dependency direction: analysis knows nothing about
hardware, the standard knows nothing about the thrusts, and the thrusts know
nothing about each other.

```
                         ┌──────────────────────────────┐
   thrusts/overlap  ──┐  │  losscolumn.core             │
   thrusts/engines  ──┼─▶│  numpy only, no GPU, no I/O  │
   thrusts/kernel   ──┘  └──────────────┬───────────────┘
        │                               │
        │  Envelope + CellComparison    │  LossColumn
        ▼                               ▼
   ┌─────────────┐              ┌───────────────┐
   │  pipeline   │─── Claim ───▶│   validate    │──▶ ConformanceReport
   └─────────────┘              └───────────────┘
        │                               │
        └──────────▶  report  ◀─────────┘
                        │
              markdown + standalone HTML
```

## Why the direction matters

`losscolumn.core` depends on numpy and nothing else. That is not minimalism for
its own sake — it means the analysis path can be exercised, tested and
inspected on a laptop, in CI, and on a cluster login node with no CUDA and no
network. The measurement code is the part that needs a GPU; the part that
decides what the measurement *means* does not, and keeping them separate is
what allows the whole reasoning chain to be validated before an hour of
allocation is spent.

It also means the same code path runs on measured and modelled data. When the
two disagree, the disagreement is about physics rather than about tooling.

---

## The central data structure

Everything flows through `Envelope`: a full-factorial grid over the operating
factors, holding replicate measurements for every system under comparison.

```python
env = Envelope.allocate(
    factors=(Factor("seq_len", (512, 2048, 8192)),
             Factor("dtype", ("fp16", "bf16"), ordered=False)),
    metric=Metric("throughput", "tok/s", higher_is_better=True),
    systems=["method", "baseline"],
    replicates=11,
    interleaved=True,
)
env.put("method", cell, measurements)          # replicates, never a mean
env.mark_missing("method", cell, "OOM at 81 GB")
```

Three commitments are encoded in the type:

**Replicates are kept, never pre-averaged.** Paired tests, bootstrap intervals
and cliff reproducibility all need the spread, and an artifact that ships only
means cannot be re-analysed by a skeptic.

**Cells may be missing, with a reason.** Real sweeps OOM. "The method cannot
run here" is exactly the kind of loss the standard exists to surface, so it is
recorded, carried through the statistics as a loss, and rendered as *cannot
run* rather than as a very large number.

**Measurement order is recorded.** Paired statistics are only valid if systems
were measured interleaved rather than in blocks. The envelope carries the flag,
`compare_cells` respects it, and the validator checks it (`LC-Q2`).

---

## The analysis pipeline

```
Envelope
   │  compare_cells()          paired sign-flip / Welch, BCa CI, TOST, BH-FDR
   ▼
list[CellComparison]           one of {loss, win, tie, inconclusive, missing}
   │  extract_loss_column()    greedy maximal-box covering
   ▼
LossColumn                     regions + design-power check
   │  assemble()               headline DERIVED from the comparisons
   ▼
Claim
   │  validate_claim()
   ▼
ConformanceReport
```

### Loss-region extraction

The question: given a few hundred cells each with a verdict, what is the
smallest set of human-readable statements covering every losing cell?

Greedy maximal-box covering over the factor lattice — close in spirit to bump
hunting (PRIM) and rule induction — with three properties the standard needs:

1. **Complete.** Every loss cell lands in at least one region; a singleton box
   is always admissible, so the loop terminates with full coverage.
2. **Pure.** A region may absorb `inconclusive` cells (region boundaries are
   genuinely fuzzy) but never a `win` or a `tie`.
3. **Worst-first.** Regions are emitted by severity × coverage, not by
   discovery order.

A redundancy pass drops any region whose losing cells are already covered by
others: greedy growth can produce a box that overlaps an earlier one so heavily
it contributes nothing, and keeping it would inflate the apparent number of
distinct failure modes — the same dishonesty in the other direction.

### Statistics

All in **log space**. Ratios of timings and throughputs are multiplicative and
right-skewed; differences of logs are approximately symmetric, compose across
factors, and give an effect measure that reads the same whether the metric was
framed as latency or as throughput.

Orientation, used without exception: **`effect > 0` means the method is
worse.**

| Choice | Why |
|--------|-----|
| Paired sign-flip permutation test | Exact at r = 5…15 by enumeration; assumes symmetry, not normality; the right null for interleaved A/B timing. |
| BCa bootstrap intervals | Percentile intervals are visibly wrong at r ≤ 8, which is where every GPU sweep lives. |
| Benjamini–Hochberg FDR | A sweep is hundreds of simultaneous tests. Without it, pure noise produces a confident-looking loss region at any α. |
| TOST for equivalence | "No significant difference" is not evidence of equivalence. `tie` has to be earned. |
| Student's *t* hand-rolled | Keeps `core` numpy-only. Verified against scipy to 1e-10 in the test suite. |

---

## Backend protocols

Each thrust defines a protocol and ships at least two implementations.

```python
class OverlapBackend(Protocol):
    name: str
    evidence_class: str          # "measured" | "simulated"
    def feasible(self, cfg) -> tuple[bool, str]: ...
    def measure(self, cfg, *, steps, seed) -> list[StepTrace]: ...
```

| Thrust | Simulated | Measured |
|--------|-----------|----------|
| I | `SyntheticOverlapBackend` — discrete-event step model | `TorchProfilerBackend` — real FSDP under `torch.profiler` |
| II | `SyntheticEngine` — closed-loop roofline serving model | `ExternalEngine` — vLLM / SGLang / TRT-LLM over one OpenAI client |
| III | — | `flash_torch` (portable) and `flash_attention_triton` |

The simulated backends emit the **same objects** the measured ones do —
`StepTrace` with spans, `ServeResult` with percentiles — so the attribution,
the envelope, the statistics and the report are byte-for-byte the same code on
both.

### Why one benchmark client for all three engines

Each engine project ships its own benchmark script, and they differ in how they
generate load, whether the prompt counts toward throughput, and how percentiles
are computed. Using each project's own script would compare three measurement
methodologies, not three engines.

---

## Determinism

Reproducibility is a requirement, so the sources of nondeterminism are closed
deliberately:

- **No `hash()` on strings for seeding.** Python salts it per process; a sweep
  seeded that way is unreproducible across runs. `zlib.crc32` over a canonical
  label is stable forever.
- **Canonical JSON everywhere.** Sorted keys, tight separators. Content hashes
  are only meaningful if serialisation is stable, so every hash routes through
  one function.
- **Per-(system, cell, replicate) seeds** derived from one master seed recorded
  in the pre-registration.
- **Bitwise determinism is a correctness check**, not an assumption: the kernel
  suite asserts two invocations give identical outputs, because a kernel with a
  race can pass a tolerance check on Tuesday.

---

## Timing hygiene

`thrusts/kernel/bench.py` exists because four common mistakes each move results
by more than the effects usually reported:

| Mistake | Fix |
|---------|-----|
| `perf_counter` around an async launch | CUDA events recorded on the stream |
| Warm L2 flattering small shapes | Flush a buffer larger than L2 between iterations |
| Blocked A/B confounding drift with the effect | Interleave every replicate |
| Mean of a right-skewed distribution | Median per iteration, replicate spread kept |

Clock and temperature are recorded before and after. The harness does not
pretend to control throttling; it reports it so a reader can discount.

---

## Report rendering

Self-contained by construction: SVG generated by hand (no matplotlib), CSS
inlined, no CDN, no external fonts. An artifact has to survive being emailed,
archived, and opened offline in five years.

Section order is **normative, not cosmetic**. The loss column sits immediately
under the headline, above the wins, above the method. `LC-1.4` fails a claim
that marks its column as belonging to an appendix.

## Extending

A fourth thrust needs to produce an `Envelope` and a list of `CellComparison`.
Everything downstream — extraction, assembly, validation, rendering, synthesis
— follows without modification. `synthesis.py` reads whatever claim documents
are present, so it picks up a new thrust with no change at all.
