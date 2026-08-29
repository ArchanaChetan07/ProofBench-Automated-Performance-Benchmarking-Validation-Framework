# Pre-registration -- Thrust I -- the overlap envelope

- Version: `1`
- Sealed: `2026-08-28T22:48:06Z`
- Seal hash: `sha256:b86a36613c8bab1494c655395c3077e4421cf5bd276045c122c6d0e5eee4f335`
- Anchor: `git:a0bdfe6`
- Authors: Archana Suresh Patil

## Hypotheses

- **H0.** The commonly recommended FSDP configuration (full shard, no tensor parallelism) is within the MDE of the best configuration in the swept space at every point of the operating envelope.
- **H1.** There exist regions of the envelope where the recommended configuration is beaten by at least the MDE, and those regions are identifiable from communication-computation overlap.

## Primary outcome

The loss column of the recommended configuration against the best swept alternative, on cluster training throughput.

## Design

- Systems compared: full_shard/tp1, hybrid_shard/tp1, full_shard/tp2, full_shard/tp8, best_of_swept
- Replicates per cell: 11 (interleaved A/B)
- Minimum effect worth calling (MDE): 5.0%
- FDR level q: 0.05; interval alpha: 0.05
- Tuning budget: n/a trials per system via `n/a`

### Factors

| Factor | Levels | Unit |
|--------|--------|------|
| `micro_batch` | 1, 2, 4, 8 | - |
| `seq_len` | 512, 2048, 8192 | tokens |
| `checkpointing` | False, True | - |
| `world_size` | 8, 32 | GPUs |

### Metrics

| Metric | Unit | Direction |
|--------|------|-----------|
| `cluster_throughput` | tok/s | higher is better |
| `overlap_efficiency` | fraction | higher is better |

## Decision rules

1. A cell is a loss when the paired regression is at least the MDE and its Benjamini-Hochberg q-value over all cells is at most 0.05.
2. A cell is a tie only when two one-sided tests establish equivalence within the MDE; a non-significant difference is reported as inconclusive, not as a tie.
3. A configuration that cannot run in a cell is recorded as a loss in that cell, with the failure reason.
4. A discontinuity is reported only when the degradation exceeds 15% and sits at least three robust deviations outside the pooled step distribution on its axis, and its bootstrap interval clears 15%.

## Exclusion rules

- The first five steps of every run are discarded as warm-up, by count, before any statistic is computed.
- A replicate whose measured step time exceeds five times the median of its cell is retained and reported, not dropped; outliers are the phenomenon.

## Stopping rule

Fixed full-factorial grid, fixed replicate count, no interim analysis.

## Loss-column policy

Every region meeting the MDE at the stated FDR level is published in the main artifact above the wins, with an attributed mechanism drawn from the overlap measurement rather than from narrative.

## Rationale (non-normative)

The replicate count is set by the arithmetic, not by convention: an exact paired sign-flip test over r replicates cannot produce a p-value below 2**-r, and Benjamini-Hochberg over a 48-cell grid needs p <= 0.05/48 for the most extreme cell. That requires r >= 10. A sweep at r = 5 would report an empty loss column no matter how badly the recommendation lost.
