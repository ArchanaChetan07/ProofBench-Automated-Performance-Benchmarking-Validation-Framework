# The overlap envelope: where the recommended sharding configuration breaks down

> **SIMULATED EVIDENCE.** Produced by the package's calibratable model, not by measurement on the target hardware. The model's parameters -- achievable FLOP/s, bus bandwidth, collective latency, per-step scheduler overhead -- are the quantities a microbenchmark measures, and the funded run replaces them with measured values. Every analysis step downstream of the numbers is the same code that runs on measured data; this artifact exists to show that path works before GPU time is committed to it.

> Across a 48-cell sharding envelope for a 7B-class model, the commonly recommended FSDP configuration (full_shard/tp1) is at or near the best swept alternative over most of the space and collapses in identifiable corners of it. Median -0.1% across the swept envelope; best +0.2% at micro_batch=4, seq_len=8192, checkpointing=True, world_size=8; worst -inf% at micro_batch=2, seq_len=8192, checkpointing=False, world_size=8.

`lc-thrust1-overlap-envelope` -- thrust I -- full_shard/tp1 vs best_of_swept -- standard LC-1.0

## Loss column

_Where this result does not hold. Published above the wins, by requirement LC-1._

### Loss column -- full_shard/tp1 vs best_of_swept (cluster_throughput, tok/s)

full_shard/tp1 loses to best_of_swept on 25% of the swept envelope (12/48 cells), across 3 region(s); worst measurable case +258.1% on cluster_throughput, and cannot run at all in 8 cell(s).

| # | Where it loses | Cells | Median | Worst | max q | Attributed to |
|---|----------------|-------|--------|-------|-------|---------------|
| 1 | 2 <= micro_batch <= 8 and seq_len=8192 and checkpointing=False | 6/6 (12% of envelope) | cannot run | cannot run | n/a | the configuration cannot run here at all; OOM: model needs ~101 GB/rank against 80 GB HBM; beaten by full_shard/tp2, full_shard/tp8 |
| 2 | micro_batch=8 and 2048 <= seq_len <= 8192 and checkpointing=False | 4/4 (8% of envelope) | cannot run | cannot run | n/a | the configuration cannot run here at all; OOM: model needs ~101 GB/rank against 80 GB HBM; beaten by full_shard/tp2, full_shard/tp8 |
| 3 | 1 <= micro_batch <= 2 and seq_len=512 and world_size=32 | 4/4 (8% of envelope) | +127.1% | +258.1% | 0.00244 | overlap efficiency 0.45; 54.5% of the step is exposed communication; beaten by hybrid_shard/tp1 |

- Envelope: 48 cells. Losses 12, wins 0, ties 35, inconclusive 1, missing 0.
- Decision rule: regression >= 5.0% (pre-registered MDE) and Benjamini-Hochberg q <= 0.05.

### Attributed causes

- **2 <= micro_batch <= 8 and seq_len=8192 and checkpointing=False and 8 <= world_size <= 32** -- the configuration cannot run here at all; OOM: model needs ~101 GB/rank against 80 GB HBM; beaten by full_shard/tp2, full_shard/tp8
- **micro_batch=8 and 2048 <= seq_len <= 8192 and checkpointing=False and 8 <= world_size <= 32** -- the configuration cannot run here at all; OOM: model needs ~101 GB/rank against 80 GB HBM; beaten by full_shard/tp2, full_shard/tp8
- **1 <= micro_batch <= 2 and seq_len=512 and False <= checkpointing <= True and world_size=32** -- overlap efficiency 0.45; 54.5% of the step is exposed communication; beaten by hybrid_shard/tp1

## Where this fails as a study

- A single 8-GPU node cannot exhibit the inter-node bandwidth effects that dominate at cluster scale. The world_size=32 column is modelled; on the funded allocation it is out of budget, and the artifact states that here rather than in a footnote.
- The comparator is an oracle -- the best of four swept configurations at each cell -- which flatters the alternatives, since a practitioner would have to find that configuration themselves.
- Overlap efficiency is measured on one rank. A straggler on another rank appears here as compute time, not as exposed communication.

## Discontinuities

<h4>Discontinuities -- full<i>shard/tp1 (cluster</i>throughput)</h4>
<p>14 cliff(s) in 82 adjacent-level steps, threshold 15.0%.</p>
<div class="scroll"><table><thead><tr>
<th>Axis</th>
<th>Transition</th>
<th>Held fixed</th>
<th>Degradation</th>
<th>95% CI</th>
</tr></thead><tbody>
<tr><td>micro_batch</td><td>4 -&gt; 8</td><td>seq<i>len=2048, checkpointing=False, world</i>size=8</td><td>cannot run</td><td>-</td></tr>
<tr><td>micro_batch</td><td>4 -&gt; 8</td><td>seq<i>len=2048, checkpointing=False, world</i>size=32</td><td>cannot run</td><td>-</td></tr>
<tr><td>micro_batch</td><td>1 -&gt; 2</td><td>seq<i>len=8192, checkpointing=False, world</i>size=8</td><td>cannot run</td><td>-</td></tr>
<tr><td>micro_batch</td><td>1 -&gt; 2</td><td>seq<i>len=8192, checkpointing=False, world</i>size=32</td><td>cannot run</td><td>-</td></tr>
<tr><td>seq_len</td><td>2048 -&gt; 8192</td><td>micro<i>batch=2, checkpointing=False, world</i>size=8</td><td>cannot run</td><td>-</td></tr>
<tr><td>seq_len</td><td>2048 -&gt; 8192</td><td>micro<i>batch=2, checkpointing=False, world</i>size=32</td><td>cannot run</td><td>-</td></tr>
<tr><td>seq_len</td><td>2048 -&gt; 8192</td><td>micro<i>batch=4, checkpointing=False, world</i>size=8</td><td>cannot run</td><td>-</td></tr>
<tr><td>seq_len</td><td>2048 -&gt; 8192</td><td>micro<i>batch=4, checkpointing=False, world</i>size=32</td><td>cannot run</td><td>-</td></tr>
<tr><td>seq_len</td><td>512 -&gt; 2048</td><td>micro<i>batch=8, checkpointing=False, world</i>size=8</td><td>cannot run</td><td>-</td></tr>
<tr><td>seq_len</td><td>512 -&gt; 2048</td><td>micro<i>batch=8, checkpointing=False, world</i>size=32</td><td>cannot run</td><td>-</td></tr>
</tbody></table></div>

## Cliff adjacency of named configurations

<p>RQ1 in tabular form. A configuration is dangerous when a single-level change in any direction costs far more than the configuration itself gains.</p><div class="scroll"><table><thead><tr><th>Configuration</th><th>Adjacent to a cliff</th><th>One-step exposure</th><th>The cliff</th></tr></thead><tbody><tr><td>textbook single-node recipe</td><td>no</td><td>23.0%</td><td>&ndash;</td></tr><tr><td>long-context default</td><td>no</td><td>0.0%</td><td>&ndash;</td></tr><tr><td>same recipe scaled to 4 nodes</td><td>no</td><td>22.7%</td><td>&ndash;</td></tr><tr><td>small-batch fine-tune at scale</td><td>yes</td><td>0.0%</td><td>world_size 8 -> 32 at micro_batch=1, seq_len=512, checkpointing=False: -10.4% (95% CI -10.6..-10.0, z=63.6)</td></tr></tbody></table></div>

## Pre-registration

- Sealed: `2026-08-28T22:48:06Z`
- Seal hash: `sha256:b86a36613c8bab1494c655395c3077e4421cf5bd276045c122c6d0e5eee4f335`
- Verification: **PASS** (0 finding(s))


## Reproduction

```bash
losscolumn run thrust1
```

- Repository: `local checkout` at commit `222dec09b09b51d0386d4ad7b96147460cfc0554` (**dirty tree**)
- Image: `ghcr.io/archanachetan07/losscolumn:0.1.0`
- Hardware: 8x A100 80GB (measured path) / calibratable model (this artifact)
- Estimated runtime: 120 min (~$1560.00)

## Conformance

### Conformance -- `lc-thrust1-overlap-envelope` against LC-1.0

**FAIL** -- 1 fatal, 1 warning(s).

| Rule | Requirement | Result | Finding |
|------|-------------|--------|---------|
| `LC-0` | LC-1.0 | pass | claim declares `LC-1.0` |
| `LC-0.2` | LC-1.0 | pass | evidence class: simulated |
| `LC-0.3` | LC-1.0 | pass | the simulated component is described |
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
| `LC-3.2` | Envelope, not point | pass | 48 cells measured |
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
| `LC-5.1` | One-command reproduction | pass | `losscolumn run thrust1` |
| `LC-5.2` | One-command reproduction | pass | reproduction is a single command |
| `LC-5.3` | One-command reproduction | pass | pinned to `222dec09b09b` |
| `LC-5.4` | One-command reproduction | FAIL | the producing working tree was dirty; the pinned commit does not identify the code that ran |
| `LC-5.5` | One-command reproduction | pass | image `ghcr.io/archanachetan07/losscolumn:0.1.0` |
| `LC-5.6` | One-command reproduction | pass | hardware: 8x A100 80GB (measured path) / calibratable model (this artifact) |
| `LC-5.7` | One-command reproduction | pass | ~120 min |
| `LC-Q1` | Measurement quality | pass | 11 replicates per cell |
| `LC-Q2` | Measurement quality | pass | measurements were interleaved A/B |
| `LC-Q3` | Measurement quality | warn | lowest per-system cell coverage is 83% |
| `LC-Q4` | Measurement quality | pass | noise/effect ratio 0.00 |
| `LC-Q5` | Measurement quality | pass | 3 limitation(s) stated |

## Provenance

- Captured: `2026-08-29T01:56:45Z` on `Windows-10-10.0.26200-SP0`, Python 3.11.5
- Hardware fingerprint: `cc72e8eab7a5d10b` (1x NVIDIA T1000 8GB)
- Packages: `torch=2.6.0+cu124`, `numpy=1.26.4`, `transformers=5.12.1`
