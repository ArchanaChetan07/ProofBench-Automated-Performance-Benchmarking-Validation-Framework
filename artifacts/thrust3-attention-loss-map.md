# Reproduction with a loss map: a from-scratch attention kernel

> A faithful from-scratch implementation of the FlashAttention algorithm (flash_torch (portable)), validated for numerical correctness against an fp64 ground truth before any timing, then swept across (head_dim, seq_len, batch, dtype) on NVIDIA T1000 8GB (sm_75, 9 GB). Median -230.9% across the swept envelope; best -46.9% at head_dim=128, seq_len=128, batch=4, dtype=float16; worst -1470.8% at head_dim=32, seq_len=512, batch=1, dtype=float16.

`lc-thrust3-attention-loss-map` -- thrust III -- reimplementation vs reference -- standard LC-1.0

## Loss column

_Where this result does not hold. Published above the wins, by requirement LC-1._

### Loss column -- reimplementation vs reference (attention_throughput, TFLOP/s)

reimplementation loses to reference on 100% of the swept envelope (36/36 cells), across 1 region(s); worst measurable case +1470.8% on attention_throughput.

| # | Where it loses | Cells | Median | Worst | max q | Attributed to |
|---|----------------|-------|--------|-------|-------|---------------|
| 1 | the entire envelope | 36/36 (100% of envelope) | +230.9% | +1470.8% | 0.000488 | the implementation dispatches ~1456 eager kernels per call (128x64 tiling) where the reference launches one fused kernel; dispatch and intermediate write-back dominate |

- Envelope: 36 cells. Losses 36, wins 0, ties 0, inconclusive 0, missing 0.
- Decision rule: regression >= 10.0% (pre-registered MDE) and Benjamini-Hochberg q <= 0.05.

### Attributed causes

- **32 <= head_dim <= 128 and 128 <= seq_len <= 2048 and 1 <= batch <= 4 and dtype in {float16, bfloat16}** -- the implementation dispatches ~1456 eager kernels per call (128x64 tiling) where the reference launches one fused kernel; dispatch and intermediate write-back dominate

## Where this fails as a study

- FlashAttention-3's headline mechanisms -- warp specialisation, TMA-driven asynchronous copy, ping-pong scheduling of softmax against the next GEMM -- are Hopper features that Triton's programming model does not expose. This is a reimplementation of the algorithm, not of the implementation, and the loss map is the measurement of what that distinction costs.
- Forward pass only. The backward pass has a different arithmetic intensity and a different loss map, and is out of scope here.
- SM clocks are recorded but not controlled. A run taken while the card is throttling is not comparable to one taken cold, and the difference routinely exceeds published effect sizes.

## Correctness suite

<h4>Correctness -- flash<i>torch (portable) on NVIDIA T1000 8GB (sm</i>75, 9 GB)</h4>
<p>36/36 shapes pass. Tolerances are derived from the arithmetic (4u + 8<i>sqrt(seq)</i>u_acc), not fitted to the results.</p>
<div class="scroll"><table><thead><tr>
<th>Shape</th>
<th>Result</th>
<th>max rel err</th>
<th>Tolerance</th>
<th>Failed checks</th>
</tr></thead><tbody>
<tr><td><code>b1xh8xs128xd32/float16</code></td><td>pass</td><td>2.45e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs128xd32/bfloat16</code></td><td>pass</td><td>1.93e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs128xd32/float16</code></td><td>pass</td><td>2.18e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs128xd32/bfloat16</code></td><td>pass</td><td>1.38e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs512xd32/float16</code></td><td>pass</td><td>3.56e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs512xd32/bfloat16</code></td><td>pass</td><td>2.47e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs512xd32/float16</code></td><td>pass</td><td>1.71e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs512xd32/bfloat16</code></td><td>pass</td><td>2.21e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs2048xd32/float16</code></td><td>pass</td><td>2.97e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs2048xd32/bfloat16</code></td><td>pass</td><td>2.51e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs2048xd32/float16</code></td><td>pass</td><td>1.88e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs2048xd32/bfloat16</code></td><td>pass</td><td>2.04e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs128xd64/float16</code></td><td>pass</td><td>2.72e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs128xd64/bfloat16</code></td><td>pass</td><td>2.17e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs128xd64/float16</code></td><td>pass</td><td>2.22e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs128xd64/bfloat16</code></td><td>pass</td><td>2.83e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs512xd64/float16</code></td><td>pass</td><td>2.55e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs512xd64/bfloat16</code></td><td>pass</td><td>3.33e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs512xd64/float16</code></td><td>pass</td><td>3.03e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs512xd64/bfloat16</code></td><td>pass</td><td>2.91e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs2048xd64/float16</code></td><td>pass</td><td>2.65e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs2048xd64/bfloat16</code></td><td>pass</td><td>2.12e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs2048xd64/float16</code></td><td>pass</td><td>3.55e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs2048xd64/bfloat16</code></td><td>pass</td><td>1.88e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs128xd128/float16</code></td><td>pass</td><td>2.72e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs128xd128/bfloat16</code></td><td>pass</td><td>2.16e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs128xd128/float16</code></td><td>pass</td><td>3.60e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs128xd128/bfloat16</code></td><td>pass</td><td>2.50e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs512xd128/float16</code></td><td>pass</td><td>3.08e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs512xd128/bfloat16</code></td><td>pass</td><td>2.42e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs512xd128/float16</code></td><td>pass</td><td>2.71e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs512xd128/bfloat16</code></td><td>pass</td><td>1.78e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b1xh8xs2048xd128/float16</code></td><td>pass</td><td>2.67e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh8xs2048xd128/bfloat16</code></td><td>pass</td><td>1.80e-03</td><td>1.56e-02</td><td>-</td></tr>
<tr><td><code>b4xh8xs2048xd128/float16</code></td><td>pass</td><td>3.14e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b4xh8xs2048xd128/bfloat16</code></td><td>pass</td><td>1.66e-03</td><td>1.56e-02</td><td>-</td></tr>
</tbody></table></div>

## Pre-registration

- Sealed: `2026-08-28T22:48:06Z`
- Seal hash: `sha256:4e8e472017ff9e1645aa00f034abe3633e86cce891cfd50ae777ef78ee1eab4c`
- Verification: **PASS** (0 finding(s))


## Reproduction

```bash
losscolumn run thrust3
```

- Repository: `local checkout` at commit `87e209643ccb4cda209ac714c3eed4f6d84bf714`
- Image: `ghcr.io/archanachetan07/losscolumn:0.1.0`
- Hardware: NVIDIA T1000 8GB (sm_75, 9 GB)
- Estimated runtime: 220 min (~$704.00)

## Conformance

### Conformance -- `lc-thrust3-attention-loss-map` against LC-1.0

**PASS** -- 0 fatal, 0 warning(s).

| Rule | Requirement | Result | Finding |
|------|-------------|--------|---------|
| `LC-0` | LC-1.0 | pass | claim declares `LC-1.0` |
| `LC-0.2` | LC-1.0 | pass | evidence class: measured |
| `LC-1.1` | Loss column | pass | loss column present |
| `LC-1.2` | Loss column | pass | losing cells are covered by described regions |
| `LC-1.3` | Loss column | pass | loss column is complete over its own cells |
| `LC-1.4` | Loss column | pass | loss column is in the main body |
| `LC-1.5` | Loss column | pass | every loss region carries an attributed cause |
| `LC-1.6` | Loss column | pass | unmeasurable cells are accounted for |
| `LC-1.8` | Loss column | pass | the design is capable of declaring a loss if one exists |
| `LC-1.7` | Loss column | pass | inconclusive fraction is within tolerance |
| `LC-2.1` | Tuning-budget parity | pass | no tuning was involved; parity certificate not applicable |
| `LC-3.1` | Envelope, not point | pass | 4 factors swept |
| `LC-3.2` | Envelope, not point | pass | 36 cells measured |
| `LC-3.3` | Envelope, not point | pass | headline reports the range, not a single number |
| `LC-3.4` | Envelope, not point | pass | headline range is non-degenerate |
| `LC-3.5` | Envelope, not point | pass | worst case names its conditions |
| `LC-3.6` | Envelope, not point | pass | headline avoids unbounded superlatives |
| `LC-4.1` | Pre-registration | pass | pre-registration present |
| `LC-4.2` | Pre-registration | pass | sealed at 2026-08-28T22:48:06Z |
| `LC-4.3` | Pre-registration | pass | protocol matches its seal |
| `LC-4.4` | Pre-registration | pass | results were verified against the protocol |
| `LC-4.5` | Pre-registration | pass | no fatal protocol deviations |
| `LC-4.6` | Pre-registration | pass | no deviations from the sealed protocol |
| `LC-4.7` | Pre-registration | pass | seal anchored to `git:a0bdfe6` |
| `LC-5.1` | One-command reproduction | pass | `losscolumn run thrust3` |
| `LC-5.2` | One-command reproduction | pass | reproduction is a single command |
| `LC-5.3` | One-command reproduction | pass | pinned to `87e209643ccb` |
| `LC-5.4` | One-command reproduction | pass | producing tree was clean |
| `LC-5.5` | One-command reproduction | pass | image `ghcr.io/archanachetan07/losscolumn:0.1.0` |
| `LC-5.6` | One-command reproduction | pass | hardware: NVIDIA T1000 8GB (sm_75, 9 GB) |
| `LC-5.7` | One-command reproduction | pass | ~220 min |
| `LC-Q1` | Measurement quality | pass | 11 replicates per cell |
| `LC-Q2` | Measurement quality | pass | measurements were interleaved A/B |
| `LC-Q3` | Measurement quality | pass | cell coverage >= 100% for every system |
| `LC-Q4` | Measurement quality | pass | noise/effect ratio 0.00 |
| `LC-Q5` | Measurement quality | pass | 3 limitation(s) stated |

## Provenance

- Captured: `2026-08-29T02:06:14Z` on `Windows-10-10.0.26200-SP0`, Python 3.11.5
- Hardware fingerprint: `cc72e8eab7a5d10b` (1x NVIDIA T1000 8GB)
- Packages: `torch=2.6.0+cu124`, `numpy=1.26.4`, `transformers=5.12.1`
