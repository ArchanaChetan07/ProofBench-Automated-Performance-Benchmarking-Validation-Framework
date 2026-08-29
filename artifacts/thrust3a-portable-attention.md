# Sparse attention, portable implementation: a loss map over prefill and decode

> The portable implementation &mdash; the FlashAttention algorithm written out in eager PyTorch ops: correct everywhere, portable to any device, and not fast. Its loss map measures what per-tile dispatch costs &mdash; swept over the registered (seq_len x batch x pattern x density x phase) lattice on NVIDIA T1000 8GB (sm_75, 9 GB), against the reference computing the same pattern. Median -45.8% across the swept envelope; best +82.7% at seq_len=512, batch=8, pattern=block_sparse, density=0.125, phase=decode; worst -1038.2% at seq_len=1024, batch=1, pattern=dense, density=1.0, phase=prefill.

`lc-thrust3a-portable-attention` -- thrust III -- implementation vs reference -- standard LC-1.0

## Loss column

_Where this result does not hold. Published above the wins, by requirement LC-1._

### Loss column -- implementation vs reference (latency_ms, ms)

implementation loses to reference on 55% of the swept envelope (44/80 cells), across 4 region(s); worst measurable case +1038.2% on latency_ms.

| # | Where it loses | Cells | Median | Worst | max q | Attributed to |
|---|----------------|-------|--------|-------|-------|---------------|
| 1 | pattern=dense and density=1.0 | 16/16 (20% of envelope) | +455.1% | +1038.2% | 0.000888 | dense pattern: no blocks are skipped, so the comparison is purely code generation against the fused reference; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel |
| 2 | batch=1 and pattern in {sliding_window, block_sparse} and 0.125 <= density <= 0.25 and phase=prefill | 16/16 (20% of envelope) | +217.6% | +473.7% | 0.000888 | sparse at density 0.125: too few blocks per query row to amortise the per-block prologue; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel |
| 3 | batch=1 and pattern in {sliding_window, block_sparse} and density=0.25 | 16/16 (20% of envelope) | +222.8% | +473.7% | 0.000888 | sparse at density 0.25: too few blocks per query row to amortise the per-block prologue; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel |
| 4 | 2048 <= seq_len <= 4096 and batch=1 and pattern in {sliding_window, block_sparse} and 0.125 <= density <= 0.25 | 16/16 (20% of envelope) | +196.9% | +428.3% | 0.000888 | sparse at density 0.125: too few blocks per query row to amortise the per-block prologue; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel |

- Envelope: 80 cells. Losses 44, wins 34, ties 2, inconclusive 0, missing 0, inapplicable 64.
- Decision rule: regression >= 10.0% (pre-registered MDE) and Benjamini-Hochberg q <= 0.05.

### Attributed causes

- **512 <= seq_len <= 4096 and 1 <= batch <= 8 and pattern=dense and density=1.0 and phase in {prefill, decode}** -- dense pattern: no blocks are skipped, so the comparison is purely code generation against the fused reference; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel
- **512 <= seq_len <= 4096 and batch=1 and pattern in {sliding_window, block_sparse} and 0.125 <= density <= 0.25 and phase=prefill** -- sparse at density 0.125: too few blocks per query row to amortise the per-block prologue; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel
- **512 <= seq_len <= 4096 and batch=1 and pattern in {sliding_window, block_sparse} and density=0.25 and phase in {prefill, decode}** -- sparse at density 0.25: too few blocks per query row to amortise the per-block prologue; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel
- **2048 <= seq_len <= 4096 and batch=1 and pattern in {sliding_window, block_sparse} and 0.125 <= density <= 0.25 and phase in {prefill, decode}** -- sparse at density 0.125: too few blocks per query row to amortise the per-block prologue; the portable implementation dispatches one eager op per tile pair where the reference launches a single fused kernel

## Where this fails as a study

- Measured on NVIDIA T1000 8GB (sm_75), which predates the architecture FlashAttention-3 targets. FA-3's warp specialisation, TMA and ping-pong scheduling do not exist on this hardware for either arm to use, so this is not evidence about them.
- Forward pass only. The backward pass has a different arithmetic intensity and a different loss map.
- One head dimension (64) and one dtype (float16). Both are held fixed to keep the sparsity lattice tractable; widening either is a new registration, not an extension of this one.
- The block-sparse pattern is one seeded random draw per shape, not an average over draws. A different draw would give a different pattern with the same density, and the artifact records the seed rather than claiming generality over patterns.
- SM clocks are recorded but not controlled. A run taken while the card is throttling is not comparable to one taken cold.

## Speedup by phase

<p>Latency means something different in each phase, so it is reported per phase rather than pooled: at prefill it is the time-to-first-token contribution, at decode it is the inter-token latency. A speedup below 1.00x is a loss.</p><div class="scroll"><table><thead><tr><th>Phase</th><th>Cells</th><th>Median latency</th><th>Reference</th><th>Median speedup</th><th>Worst</th><th>Best</th></tr></thead><tbody><tr><td><b>prefill</b><span class='sub'>time-to-first-token contribution</span></td><td>40/72</td><td>35.995 ms</td><td>25.188 ms</td><td>0.63x</td><td>0.08x</td><td>5.20x</td></tr><tr><td><b>decode</b><span class='sub'>inter-token latency</span></td><td>40/72</td><td>2.020 ms</td><td>1.488 ms</td><td>0.90x</td><td>0.09x</td><td>5.80x</td></tr></tbody></table></div>

## Correctness suite

<h4>Correctness -- flash<i>torch</i>sparse (portable eager ops) on NVIDIA T1000 8GB (sm_75, 9 GB)</h4>
<p>80/80 shapes pass. Tolerances are derived from the arithmetic (4u + 8<i>sqrt(seq)</i>u_acc), not fitted to the results.</p>
<div class="scroll"><table><thead><tr>
<th>Shape</th>
<th>Result</th>
<th>max rel err</th>
<th>Tolerance</th>
<th>Failed checks</th>
</tr></thead><tbody>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.09e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.68e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.72e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>1.53e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.09e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.85e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.72e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>1.53e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.72e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.69e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.52e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.58e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.77e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.84e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.52e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.81e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.77e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.84e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.57e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.93e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>1.58e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.02e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>1.58e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.09e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>1.58e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.78e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.88e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.29e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.61e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.02e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>1.81e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.28e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.49e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.74e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>1.95e-04</td><td>1.97e-03</td><td>-</td></tr>
</tbody></table></div>

## Pre-registration

- Sealed: `2026-08-29T07:01:56Z`
- Seal hash: `sha256:fd87925555851b655abfbe9df0c7ec2e5cd065a1cd6f8be9573a8ca114ac6a24`
- Verification: **PASS** (0 finding(s))


## Reproduction

```bash
losscolumn run thrust3
```

- Repository: `local checkout` at commit `7c28f6afb5883db3f479096bf41ae0793cc5bfb3`
- Image: `ghcr.io/archanachetan07/losscolumn:0.1.0`
- Hardware: NVIDIA T1000 8GB (sm_75, 9 GB)
- Estimated runtime: 25 min

## Conformance

### Conformance -- `lc-thrust3a-portable-attention` against LC-1.0

**PASS** -- 0 fatal, 0 warning(s).

| Rule | Requirement | Result | Finding |
|------|-------------|--------|---------|
| `LC-0.1` | Standard version | pass | claim declares `LC-1.0` |
| `LC-0.2` | Evidence class | pass | evidence class: measured |
| `LC-1.1` | Loss column | pass | loss column present |
| `LC-1.2` | Loss column | pass | losing cells are covered by described regions |
| `LC-1.3` | Loss column | pass | loss column is complete over its own cells |
| `LC-1.4` | Loss column | pass | loss column is in the main body |
| `LC-1.5` | Loss column | pass | every loss region carries an attributed cause |
| `LC-1.6` | Loss column | pass | unmeasurable cells are accounted for |
| `LC-1.9` | Loss column | pass | every region's support meets the claim's declared thresholds |
| `LC-1.8` | Loss column | pass | the design is capable of declaring a loss if one exists |
| `LC-1.7` | Loss column | pass | inconclusive fraction is within tolerance |
| `LC-2.1` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-2.2` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-2.3` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-2.4` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-2.5` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-2.6` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-2.7` | Tuning-budget parity | n/a | no tuning was involved; parity does not apply |
| `LC-3.1` | Envelope, not point | pass | 5 factors swept |
| `LC-3.2` | Envelope, not point | pass | 80 cells measured |
| `LC-3.3` | Envelope, not point | pass | headline reports the range, not a single number |
| `LC-3.4` | Envelope, not point | pass | headline range is non-degenerate |
| `LC-3.5` | Envelope, not point | pass | worst case names its conditions |
| `LC-3.7` | Envelope, not point | pass | headline extremes agree with the loss column |
| `LC-3.6` | Envelope, not point | pass | headline avoids unbounded superlatives |
| `LC-4.1` | Pre-registration | pass | pre-registration present |
| `LC-4.2` | Pre-registration | pass | sealed at 2026-08-29T07:01:56Z |
| `LC-4.3` | Pre-registration | pass | protocol matches its seal |
| `LC-4.4` | Pre-registration | pass | results were verified against the protocol |
| `LC-4.5` | Pre-registration | pass | no fatal protocol deviations |
| `LC-4.6` | Pre-registration | pass | no deviations from the sealed protocol |
| `LC-4.7` | Pre-registration | pass | seal anchored to `git:257b179` |
| `LC-5.1` | One-command reproduction | pass | `losscolumn run thrust3` |
| `LC-5.2` | One-command reproduction | pass | reproduction is a single command |
| `LC-5.3` | One-command reproduction | pass | pinned to `7c28f6afb588` |
| `LC-5.4` | One-command reproduction | pass | producing tree was clean |
| `LC-5.5` | One-command reproduction | pass | image `ghcr.io/archanachetan07/losscolumn:0.1.0` |
| `LC-5.6` | One-command reproduction | pass | hardware: NVIDIA T1000 8GB (sm_75, 9 GB) |
| `LC-5.7` | One-command reproduction | pass | ~25 min |
| `LC-Q1` | Measurement quality | pass | 11 replicates per cell |
| `LC-Q2` | Measurement quality | pass | measurements were interleaved A/B |
| `LC-Q3` | Measurement quality | pass | cell coverage >= 100% for every system |
| `LC-Q4` | Measurement quality | pass | noise/effect ratio 0.00 |
| `LC-Q6` | Measurement quality | pass | equivalence is established where the data supports it |
| `LC-Q5` | Measurement quality | pass | 5 limitation(s) stated |

## Provenance

- Captured: `2026-08-29T18:10:41Z` on `Windows-10-10.0.26200-SP0`, Python 3.11.5
- Hardware fingerprint: `cc72e8eab7a5d10b` (1x NVIDIA T1000 8GB)
- Packages: `torch=2.6.0+cu124`, `triton=3.2.0`, `numpy=1.26.4`, `transformers=5.12.1`
