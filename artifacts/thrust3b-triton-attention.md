# Sparse attention, triton implementation: a loss map over prefill and decode

> The triton implementation &mdash; the same algorithm as a fused Triton kernel with real block skipping. Its loss map measures what the code generator achieves &mdash; swept over the registered (seq_len x batch x pattern x density x phase) lattice on NVIDIA T1000 8GB (sm_75, 9 GB), against the reference computing the same pattern. Median +77.2% across the swept envelope; best +93.6% at seq_len=4096, batch=1, pattern=block_sparse, density=0.125, phase=prefill; worst -6849.2% at seq_len=4096, batch=8, pattern=dense, density=1.0, phase=decode.

`lc-thrust3b-triton-attention` -- thrust III -- implementation vs reference -- standard LC-1.0

## Loss column

_Where this result does not hold. Published above the wins, by requirement LC-1._

### Loss column -- implementation vs reference (latency_ms, ms)

implementation loses to reference on 10% of the swept envelope (8/80 cells), across 1 region(s); worst measurable case +6849.2% on latency_ms.

| # | Where it loses | Cells | Median | Worst | max q | Attributed to |
|---|----------------|-------|--------|-------|-------|---------------|
| 1 | pattern=dense and density=1.0 and phase=decode | 8/8 (10% of envelope) | +2037.0% | +6849.2% | 0.00434 | decode only: one query row is padded to a 64-wide block, so 98% of every tile is wasted work a prefill-shaped kernel cannot avoid |

- Envelope: 80 cells. Losses 8, wins 64, ties 8, inconclusive 0, missing 0, inapplicable 64.
- Decision rule: regression >= 10.0% (pre-registered MDE) and Benjamini-Hochberg q <= 0.05.

### Attributed causes

- **512 <= seq_len <= 4096 and 1 <= batch <= 8 and pattern=dense and density=1.0 and phase=decode** -- decode only: one query row is padded to a 64-wide block, so 98% of every tile is wasted work a prefill-shaped kernel cannot avoid

## Where this fails as a study

- THIS DEVICE HAS NO TENSOR CORES. It reports compute capability sm_75, but compute capability names an instruction set rather than the units behind it, and this die ships without them: fp16 GEMM measures 0.323 TFLOP/s against 1.828 TFLOP/s for fp32, a ratio of 0.177 where a tensor-core part exceeds 2. Every timing here is fp16, so both arms ran an emulated path that nobody deploys. The comparison stays internally valid -- both arms took the same path -- but the absolute rates do not transfer to a tensor-core device, and neither necessarily does the ordering: on real hardware the reference's fused backend and the kernel's tl.dot would both reach units that are absent here, and there is no reason to assume they would gain equally.
- Measured on NVIDIA T1000 8GB (sm_75), which predates the architecture FlashAttention-3 targets. FA-3's warp specialisation, TMA and ping-pong scheduling do not exist on this hardware for either arm to use, so this is not evidence about them.
- Forward pass only. The backward pass has a different arithmetic intensity and a different loss map.
- One head dimension (64) and one dtype (float16). Both are held fixed to keep the sparsity lattice tractable; widening either is a new registration, not an extension of this one.
- The reference for a sparse pattern is torch computing that same pattern densely under an explicit mask, because that is the only way torch computes this function. An explicit mask precludes the fused backend, so the reference is not a sparsity-optimised kernel and the margin here is not a margin over one. Against a production block-sparse implementation it would be smaller, and this artifact is not evidence about that comparison.
- The block-sparse pattern is one seeded random draw per shape, not an average over draws. A different draw would give a different pattern with the same density, and the artifact records the seed rather than claiming generality over patterns.
- SM clocks are recorded but not controlled. A run taken while the card is throttling is not comparable to one taken cold.

## Speedup by phase

<p>Latency means something different in each phase, so it is reported per phase rather than pooled: at prefill it is the time-to-first-token contribution, at decode it is the inter-token latency. A speedup below 1.00x is a loss.</p><div class="scroll"><table><thead><tr><th>Phase</th><th>Cells</th><th>Median latency</th><th>Reference</th><th>Median speedup</th><th>Worst</th><th>Best</th></tr></thead><tbody><tr><td><b>prefill</b><span class='sub'>time-to-first-token contribution</span></td><td>40/72</td><td>3.919 ms</td><td>16.983 ms</td><td>5.76x</td><td>0.96x</td><td>15.61x</td></tr><tr><td><b>decode</b><span class='sub'>inter-token latency</span></td><td>40/72</td><td>0.356 ms</td><td>1.038 ms</td><td>3.46x</td><td>0.01x</td><td>8.19x</td></tr></tbody></table></div>

## Correctness suite

<h4>Correctness -- flash<i>attention</i>triton<i>sparse (fused) on NVIDIA T1000 8GB (sm</i>75, 9 GB)</h4>
<p>80/80 shapes pass. Tolerances are derived from the arithmetic (4u + 8<i>sqrt(seq)</i>u_acc), not fitted to the results.</p>
<div class="scroll"><table><thead><tr>
<th>Shape</th>
<th>Result</th>
<th>max rel err</th>
<th>Tolerance</th>
<th>Failed checks</th>
</tr></thead><tbody>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.39e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.55e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>3.39e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.71e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.39e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.47e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>3.39e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.71e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.72e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.69e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.52e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.93e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.94e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.96e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.52e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.86e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.94e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.96e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs512xd64/float16/causal</code></td><td>pass</td><td>2.57e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.57e-04</td><td>1.96e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.21e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.21e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.21e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.89e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.21e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.87e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.88e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.29e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>2.61e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.82e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.01e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>4.46e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>3.49e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.74e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1024xd64/float16/causal</code></td><td>pass</td><td>3.18e-04</td><td>1.97e-03</td><td>-</td></tr>
<tr><td><code>b1xh2xs1xd64/float16/causal</code></td><td>pass</td><td>2.23e-04</td><td>1.97e-03</td><td>-</td></tr>
</tbody></table></div>

## Pre-registration

- Sealed: `2026-08-29T07:01:56Z`
- Seal hash: `sha256:fd87925555851b655abfbe9df0c7ec2e5cd065a1cd6f8be9573a8ca114ac6a24`
- Verification: **PASS** (0 finding(s))


## Reproduction

```bash
losscolumn run thrust3
```

- Repository: `local checkout` at commit `ced1ad46b08e0026787352e966ecae1a1fe8fa64`
- Image: `ghcr.io/archanachetan07/losscolumn:0.1.0`
- Hardware: NVIDIA T1000 8GB (sm_75, 9 GB)
- Estimated runtime: 25 min

## Conformance

### Conformance -- `lc-thrust3b-triton-attention` against LC-1.0

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
| `LC-5.3` | One-command reproduction | pass | pinned to `ced1ad46b08e` |
| `LC-5.4` | One-command reproduction | pass | producing tree was clean |
| `LC-5.5` | One-command reproduction | pass | image `ghcr.io/archanachetan07/losscolumn:0.1.0` |
| `LC-5.6` | One-command reproduction | pass | hardware: NVIDIA T1000 8GB (sm_75, 9 GB) |
| `LC-5.7` | One-command reproduction | pass | ~25 min |
| `LC-Q1` | Measurement quality | pass | 11 replicates per cell |
| `LC-Q2` | Measurement quality | pass | measurements were interleaved A/B |
| `LC-Q3` | Measurement quality | pass | cell coverage >= 100% for every system |
| `LC-Q4` | Measurement quality | pass | noise/effect ratio 0.00 |
| `LC-Q6` | Measurement quality | pass | equivalence is established where the data supports it |
| `LC-Q5` | Measurement quality | pass | 7 limitation(s) stated |

## Provenance

- Captured: `2026-08-30T00:32:34Z` on `Windows-10-10.0.26200-SP0`, Python 3.11.5
- Hardware fingerprint: `cc72e8eab7a5d10b` (1x NVIDIA T1000 8GB)
- Packages: `torch=2.6.0+cu124`, `triton=3.2.0`, `numpy=1.26.4`, `transformers=5.12.1`
