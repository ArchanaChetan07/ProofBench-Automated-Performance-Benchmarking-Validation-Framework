# Pre-registration -- Thrust III -- sparse attention, prefill and decode, with a loss map

- Version: `2`
- Sealed: `2026-08-29T07:01:56Z`
- Seal hash: `sha256:fd87925555851b655abfbe9df0c7ec2e5cd065a1cd6f8be9573a8ca114ac6a24`
- Anchor: `git:257b179`
- Authors: Archana Suresh Patil

## Hypotheses

- **H0.** Across the registered lattice, each from-scratch implementation is within the MDE of the reference computing the same attention pattern.
- **H1.** It is not, and the regions where it loses are attributable to identifiable mechanisms: block-skipping that fails to amortise at low density, query-block padding at decode, and launch-bound regimes at short sequence.

## Primary outcome

A loss column over (seq_len, batch, pattern, density, phase) on per-call attention latency, published separately for each implementation, with an attributed mechanism for every region.

## Design

- Systems compared: implementation, reference
- Replicates per cell: 11 (interleaved A/B)
- Minimum effect worth calling (MDE): 10.0%
- FDR level q: 0.05; interval alpha: 0.05
- Tuning budget: 0 trials per system via `none: neither arm is tuned per cell`

### Factors

| Factor | Levels | Unit |
|--------|--------|------|
| `seq_len` | 512, 1024, 2048, 4096 | tokens |
| `batch` | 1, 8 | - |
| `pattern` | dense, sliding_window, block_sparse | - |
| `density` | 0.125, 0.25, 1.0 | - |
| `phase` | prefill, decode | - |

### Metrics

| Metric | Unit | Direction |
|--------|------|-----------|
| `latency_ms` | ms | lower is better |
| `attention_throughput` | TFLOP/s | higher is better |
| `peak_memory_mb` | MB | lower is better |
| `max_relative_error` | ratio | lower is better |

## Decision rules

1. The two implementations -- portable eager ops and the fused Triton kernel -- are swept separately and published as separate claims. They answer different questions and are never pooled into one number.
2. The comparator for a sparse pattern is the reference computing THAT pattern densely under a mask, never unmasked dense attention: the two compute different functions, and comparing against the wrong one turns the pattern's density into a spurious speedup.
3. Correctness is evaluated against an fp64 ground truth before timing at every cell. A cell that fails is recorded as unmeasurable for that implementation and enters the comparison as a loss; its timing is never reported.
4. Tolerances are derived from the arithmetic -- 4u for the storage format plus 8*sqrt(seq)*u for the accumulator -- fixed before measurement and never adjusted to make a cell pass.
5. Throughput counts the FLOPs of the blocks actually computed, not the dense equivalent.
6. Block size is fixed at 64x64 and is not autotuned: it is part of the sparsity format, and tuning over it would change the pattern being measured.
7. Timing uses CUDA events, an L2 flush between iterations, and interleaved A/B replicates; the statistic is the median of per-iteration device times.
8. A cell is a loss when the paired regression is at least the MDE and its Benjamini-Hochberg q-value over all measured cells is at most 0.05.

## Exclusion rules

- Combinations of pattern and density that are not a configuration -- dense below full density, and any sparse pattern at full density -- are recorded as inapplicable with that reason. They are not silently dropped: the artifact must not claim a full factorial it did not run.
- Shapes that exceed device memory are recorded as unmeasurable with the allocation failure, for both arms.
- No cell is excluded after its timing is seen.

## Stopping rule

Fixed full-factorial lattice, fixed replicate count, no interim analysis and no adaptive allocation of replicates.

## Loss-column policy

Each implementation's loss map is the artifact, and losing regions are marked on the map itself rather than listed in an appendix. Where an implementation is slower than the reference the artifact says so on its front page, and says that a reader wanting speed should use the reference.

## Rationale (non-normative)

BASELINE TUNING POLICY. Neither arm receives per-cell tuning, and the policy is symmetric by construction rather than by intention. The reference is given its own best backend for each shape by torch's dispatcher, with no backend disabled and no manual restriction; the Triton implementation autotunes only its scheduling parameters (warps and pipeline stages) from a fixed configuration list, with block size held constant. Neither arm is hand-tuned per cell by the operator, so the effort asymmetry that tuning-budget parity exists to control is absent here -- which is why no parity certificate accompanies this thrust and LC-2 is recorded as not applicable.

GPU BUDGET. The registered lattice is 144 cells, of which 80 are applicable configurations, swept twice (once per implementation) at 11 replicates of 10 timed iterations. On a single device that is roughly 25 minutes of measurement per implementation, and the registration commits to running it complete rather than stopping when a result appears.

REPLICATE COUNT. Eleven is set by the arithmetic, not by convention. An exact paired sign-flip test over r replicates cannot produce a p-value below 2**-r, and Benjamini-Hochberg over the applicable cells needs p <= 0.05/m for the most extreme cell; at m = 80 that requires r >= 11. A sweep at r = 5 would report an empty loss column however large the regression.

MDE. Ten percent rather than five: an implementation within 10% of the reference is a practical substitute for most purposes, and the finding of interest is the region where it is far outside that. A tighter MDE would fill the loss column with regions no reader would act on.
