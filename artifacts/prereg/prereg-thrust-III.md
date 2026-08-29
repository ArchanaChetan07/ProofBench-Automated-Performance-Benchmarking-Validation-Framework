# Pre-registration -- Thrust III -- reproduction with a loss map

- Version: `1`
- Sealed: `2026-08-28T22:48:06Z`
- Seal hash: `sha256:4e8e472017ff9e1645aa00f034abe3633e86cce891cfd50ae777ef78ee1eab4c`
- Anchor: `git:a0bdfe6`
- Authors: Archana Suresh Patil

## Hypotheses

- **H0.** A faithful from-scratch implementation of the FlashAttention algorithm matches the reference within the MDE across the swept shape space.
- **H1.** It does not, and the regions where it loses are attributable to specific implementation mechanisms the reimplementation does not express.

## Primary outcome

A complete performance map over (head_dim, seq_len, batch, dtype) with every losing region marked and an attributed cause for each.

## Design

- Systems compared: reimplementation, reference
- Replicates per cell: 11 (interleaved A/B)
- Minimum effect worth calling (MDE): 10.0%
- FDR level q: 0.05; interval alpha: 0.05
- Tuning budget: n/a trials per system via `n/a`

### Factors

| Factor | Levels | Unit |
|--------|--------|------|
| `head_dim` | 32, 64, 128 | - |
| `seq_len` | 128, 512, 2048 | tokens |
| `batch` | 1, 4 | - |
| `dtype` | float16, bfloat16 | - |

### Metrics

| Metric | Unit | Direction |
|--------|------|-----------|
| `attention_throughput` | TFLOP/s | higher is better |
| `max_relative_error` | ratio | lower is better |

## Decision rules

1. Correctness is evaluated before timing at every shape. A shape that fails correctness is recorded as unmeasurable for the reimplementation and enters the comparison as a loss; its timing is never reported.
2. Correctness tolerances are derived from the arithmetic -- 4u for the storage format plus 8*sqrt(seq)*u for the accumulator -- and are fixed before any measurement. They are not adjusted to make a shape pass.
3. Timing uses CUDA events, an L2 flush between iterations, and interleaved A/B replicates; the statistic is the median of per-iteration device times.
4. The reimplementation is expected to lose over much of the space. The result is the map and the attribution, not the aggregate.

## Exclusion rules

- Shapes that exceed device memory are recorded as unmeasurable with the allocation failure, for both arms.
- No shape is excluded after seeing its timing.

## Stopping rule

Fixed full-factorial grid; correctness gate is evaluated first.

## Loss-column policy

The performance map is the artifact. Losing regions are marked on the map itself, not listed in an appendix, and the artifact states plainly that a reader wanting the fastest kernel should use the reference.

## Rationale (non-normative)

The MDE is 10% rather than 5% because a kernel within 10% of the reference is a practical substitute; the finding of interest is the region where it is far outside that, and a tighter MDE would fill the loss column with regions no reader would act on. Matching a mature hand-tuned kernel is not the goal and would not be credible.
