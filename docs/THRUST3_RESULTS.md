# Thrust III on tensor cores

The measurement this project was built to make, made for the first time.

Everything before this ran on a T1000: 8.6 GB, no tensor cores, compute
capability 7.5. A sparse-attention kernel measured on a card with no tensor
cores is measured against a reference that has none either, so the comparison
that matters -- hand-written blocking against the vendor's fused path -- never
happened. It has now, on an RTX PRO 6000 Blackwell, sm_120, 96 GB.

The registered lattice, unchanged: seq_len x batch x pattern x density x
phase, 144 cells, 11 replicates. 80 cells are gradable and 64 are inapplicable
(a density below 1.0 means nothing for a dense pattern).

## The Triton kernel

**44 wins, 28 losses, 8 ties** of 80 gradable cells.
Graded `conforming`, no fatal findings.

It wins where there is sparsity to exploit and loses where there is not. Both
regions are attributed rather than described:

**16 losses**, worst 1959%, median 1065% -- pattern ['dense'], phase ['prefill', 'decode'].

> dense pattern: no blocks are skipped, so the comparison is purely code generation against the fused reference

**8 losses**, worst 106%, median 63% -- pattern ['sliding_window', 'block_sparse'], phase ['decode'].

> decode only: one query row is padded to a 64-wide block, so 98% of every tile is wasted work a prefill-shaped kernel cannot avoid

**8 losses**, worst 106%, median 75% -- pattern ['sliding_window', 'block_sparse'], phase ['decode'].

> decode only: one query row is padded to a 64-wide block, so 98% of every tile is wasted work a prefill-shaped kernel cannot avoid

The dense region is the honest half of the result. With no blocks to skip the
comparison is hand-written code generation against a fused vendor kernel, and
the hand-written kernel is ten to twenty times slower. That is what the loss
column is for.

## The portable implementation

**80 losses, 0 wins** of 80 cells -- the FlashAttention algorithm
written out in eager PyTorch ops. Correct everywhere, portable anywhere, and
not fast. Its loss map measures what per-tile dispatch costs.

## The memory model, in domain at last

`block-memory-v2` was fitted on 8.6 GB. On 96 GB it is **8 of 8 correct**, with a
dangerous-error score of **0.0** and no false wins. Prediction error runs
-13% to +14%.

One thing that result does not establish. The `ckpt=True` and `ckpt=False`
cells measure identical peaks, because the calibration block has one layer and
checkpointing one layer saves nothing at the peak: backward recomputes to the
same high-water mark. The model subtracts a saving the workload cannot realise,
which is exactly why every `ckpt=True` cell errs positive. **The checkpointing
term is still unvalidated**, and a grid with one layer cannot validate it.

## What the rental cost to find

Two bugs, both of the kind that only appear on real hardware:

**`kill(-1, 0)` is not a liveness check.** On POSIX it addresses every process
the caller may signal, so it succeeds and reads as alive. Windows asks
tasklist, finds nothing, returns False -- which is why three weeks of
development never saw it. A lock file that parses without a pid falls back to
-1, so on Linux a truncated lock would have blocked every later measurement on
the machine.

**A previous tenant's vLLM engine held 87 of 97 GB.** The budget is a fraction
of total memory, so the calibration computed 93.8 GB, met the allocator's
refusal near 11 GB, and recorded six cells INFEASIBLE with a dangerous-error
score of 7.5. Cleared of the other process the identical grid scores 8 of 8
and zero. The model was blamed for a verdict the machine never had room to
give. Measurement now refuses a device someone else is using, and records the
occupancy so a reader can tell a crowded machine from a wrong model.

## Not measured

The stability sentinel and the communication campaign need four devices and
this box had one. They are recorded in `stages-not-run.json` rather than
omitted: an absent artifact and a stage nobody attempted look identical
afterwards, and only one of them means the measurement was not made.

## The checkpointing term, measured

The 8-of-8 calibration result is silent about checkpointing, and depth is what
makes the term observable. Measured at 1, 2, 4 and 8 layers with three
predictions sealed first:

| layers | measured off | measured on | measured saving | predicted saving |
|---|---|---|---|---|
| 1 | 1.44 GB | 1.44 GB | 0.0% | 10.0% |
| 2 | 2.41 GB | 1.54 GB | 36.3% | 17.3% |
| 4 | 4.35 GB | 1.72 GB | 60.4% | 27.2% |
| 8 | 8.21 GB | 2.09 GB | 74.5% | 38.0% |

**P1 held.** One layer saves nothing, to the byte. Checkpointing a single block
discards intermediates and recomputes them, reaching the same peak.

**P2 held.** The saving grows with depth, which is what the term describes.

**P3 falsified by 36.6 percentage points.** At eight layers the registered
`checkpoint_retained_fraction` of 0.25 predicts a 38.0% saving; 74.5% was
measured. Checkpointing is worth roughly twice what the model credits it.

Separating the two sides shows the error is not one mistake but two, and only
one of them is safe:

| layers | predicted off vs measured | predicted on vs measured |
|---|---|---|
| 1 | +55.7% | +40.1% |
| 2 | +8.1% | +40.2% |
| 4 | −23.6% | +40.5% |
| 8 | **−42.2%** | +40.8% |

Under checkpointing the model over-predicts by a near-constant 40% at every
depth, which is the signature of a single wrong constant and is the *safe*
direction: it reserves more than is needed.

Without checkpointing the error **changes sign**. The model over-predicts at one
layer and under-predicts by 42% at eight, because its per-layer growth is too
small — measured memory rises about 0.97 GB per layer and the model's rises far
slower. Under-prediction is the direction that says *this fits* about something
that does not, and it gets worse the deeper the stack. A production model has
tens of layers, not eight.

None of this was visible in the calibration grid, and the grid said so: its own
preflight recorded `"informative": false` and `"can_classify": false`, because
every cell fits trivially on 96 GB and no cell lies near the decision boundary.
An 8-of-8 score on a grid that cannot fail is not evidence that the model is
right.

The model's own registered domain is `n_layers: (1, 1)`, and the excursion
detector flags eight layers as **8.0x** outside the fitted box against a
`MAX_EXCURSION` of 4.0. The machinery was correct; the term had simply never
been checked anywhere it could be.

**No parameter was refitted.** Every version-2 parameter is registered rather
than fitted, and a disagreement found in validation is a finding, not an input.
