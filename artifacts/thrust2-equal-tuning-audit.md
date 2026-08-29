# The equal-tuning audit: engine differences that survive budget parity

> **SIMULATED EVIDENCE.** Produced by the package's calibratable model, not by measurement on the target hardware. The model's parameters -- achievable FLOP/s, bus bandwidth, collective latency, per-step scheduler overhead -- are the quantities a microbenchmark measures, and the funded run replaces them with measured values. Every analysis step downstream of the numbers is the same code that runs on measured data; this artifact exists to show that path works before GPU time is committed to it.

> Three inference engines, each given an identical 40-trial search from the same procedure against the same traces on the same hardware, compared across their full latency-throughput frontiers on four workload shapes. Median -11.0% across the swept envelope; best -3.5% at workload=summarize, latency_budget_ms=250; worst -inf% at workload=summarize, latency_budget_ms=10.

`lc-thrust2-equal-tuning-audit` -- thrust II -- vllm vs best_alternative -- standard LC-1.0

## Loss column

_Where this result does not hold. Published above the wins, by requirement LC-1._

### Loss column -- vllm vs best_alternative (attainable_throughput, tok/s)

vllm loses to best_alternative on 75% of the swept envelope (18/24 cells), across 3 region(s); worst measurable case +107.7% on attainable_throughput, and cannot run at all in 1 cell(s).

| # | Where it loses | Cells | Median | Worst | max q | Attributed to |
|---|----------------|-------|--------|-------|-------|---------------|
| 1 | workload in {chat, rag, agentic} and 10 <= latency_budget_ms <= 100 | 13/15 (62% of envelope) | +17.9% | +107.7% | 0.000488 | vllm cannot meet a 10 ms/token normalised p99 budget on rag at any measured concurrency; rag has 15% prefix reuse, which the alternative caches and vllm does not at its tuned settings; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings |
| 2 | workload in {chat, agentic} | 12/12 (50% of envelope) | +17.9% | +94.8% | 0.000488 | agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings |
| 3 | workload in {chat, summarize, agentic} and 10 <= latency_budget_ms <= 25 | 9/9 (38% of envelope) | +47.3% | +94.8% | 0.000488 | vllm cannot meet a 10 ms/token normalised p99 budget on summarize at any measured concurrency; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings |

- Envelope: 24 cells. Losses 18, wins 0, ties 4, inconclusive 0, missing 2.
- Decision rule: regression >= 5.0% (pre-registered MDE) and Benjamini-Hochberg q <= 0.05.
- 2 cell(s) could not be measured for either system and are excluded from the comparison rather than counted as wins.

### Attributed causes

- **workload in {chat, rag, agentic} and 10 <= latency_budget_ms <= 100** -- vllm cannot meet a 10 ms/token normalised p99 budget on rag at any measured concurrency; rag has 15% prefix reuse, which the alternative caches and vllm does not at its tuned settings; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings
- **workload in {chat, agentic} and 10 <= latency_budget_ms <= 250** -- agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings
- **workload in {chat, summarize, agentic} and 10 <= latency_budget_ms <= 25** -- vllm cannot meet a 10 ms/token normalised p99 budget on summarize at any measured concurrency; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings

## Where this fails as a study

- Equal tuning budget is not equal tuning skill. One operator ran all three searches; familiarity is a residual confound the protocol documents and cannot remove.
- Engine releases move weekly. This is a snapshot against pinned commits and is designed to be re-run rather than cited indefinitely.
- Equal trial counts are nominal parity. The certificate reports where they were not effective parity -- unequal search-space coverage and a large wall-clock asymmetry from per-configuration engine builds.
- Prompts are synthetic sequences of the measured length. Length drives the systems behaviour under test, but real text would change cache behaviour for the prefix-caching engines.

## Default-to-tuned gap

<h4>Default-to-tuned gap</h4>
<p>How much of each engine's performance is locked behind a tuning search. Reported as a primary result because production deployments run near defaults.</p>
<div class="scroll"><table><thead><tr>
<th>Engine</th>
<th>Workload</th>
<th>Default tok/s</th>
<th>Tuned tok/s</th>
<th>Gain from tuning</th>
<th>p99 latency change</th>
</tr></thead><tbody>
<tr><td>sglang</td><td>agentic</td><td>3,186</td><td>3,186</td><td>-0.0%</td><td>-0.0%</td></tr>
<tr><td>sglang</td><td>chat</td><td>4,037</td><td>4,132</td><td>+2.4%</td><td>-2.5%</td></tr>
<tr><td>sglang</td><td>rag</td><td>1,578</td><td>1,524</td><td>-3.4%</td><td>+22.4%</td></tr>
<tr><td>sglang</td><td>summarize</td><td>926</td><td>947</td><td>+2.3%</td><td>-7.9%</td></tr>
<tr><td>trtllm</td><td>agentic</td><td>3,176</td><td>3,176</td><td>-0.0%</td><td>+0.4%</td></tr>
<tr><td>trtllm</td><td>chat</td><td>4,578</td><td>4,580</td><td>+0.0%</td><td>-0.3%</td></tr>
<tr><td>trtllm</td><td>rag</td><td>1,657</td><td>1,627</td><td>-1.8%</td><td>+14.1%</td></tr>
<tr><td>trtllm</td><td>summarize</td><td>997</td><td>964</td><td>-3.3%</td><td>+11.6%</td></tr>
<tr><td>vllm</td><td>agentic</td><td>2,754</td><td>2,991</td><td>+8.6%</td><td>-30.8%</td></tr>
<tr><td>vllm</td><td>chat</td><td>3,881</td><td>3,883</td><td>+0.0%</td><td>-0.3%</td></tr>
<tr><td>vllm</td><td>rag</td><td>1,497</td><td>1,472</td><td>-1.6%</td><td>+13.2%</td></tr>
<tr><td>vllm</td><td>summarize</td><td>909</td><td>920</td><td>+1.3%</td><td>-6.1%</td></tr>
</tbody></table></div>

## Workload traces

<p>The same request trace is replayed against every engine; the digest is what proves the inputs were identical. Prefix reuse is a property of the workload, not of the engine, and it is the quantity a radix cache monetises &mdash; publishing an engine comparison without stating it is the easiest way to make either engine look dominant.</p><div class="scroll"><table><thead><tr><th>Workload</th><th>Requests</th><th>Median prompt</th><th>Median output</th><th>Prefix reuse</th><th>Digest</th></tr></thead><tbody><tr><td><b>chat</b><br><span style='color:var(--muted)'>Short interactive turns; the shape most published comparisons use.</span></td><td>400</td><td>232</td><td>179</td><td>0%</td><td><code>16731a73c449</code></td></tr><tr><td><b>rag</b><br><span style='color:var(--muted)'>Long retrieval-augmented prefill with a shared system prompt and short answers.</span></td><td>400</td><td>5225</td><td>92</td><td>15%</td><td><code>df72f3731410</code></td></tr><tr><td><b>summarize</b><br><span style='color:var(--muted)'>Long input, long output; decode dominated with a heavy prefill.</span></td><td>400</td><td>12136</td><td>1017</td><td>0%</td><td><code>74f010ee0eae</code></td></tr><tr><td><b>agentic</b><br><span style='color:var(--muted)'>Many short calls reusing one long tool-definition prefix; high concurrency.</span></td><td>800</td><td>1496</td><td>42</td><td>82%</td><td><code>581ad3a342b1</code></td></tr></tbody></table></div>

## Where each engine wins

<p>Attainment-curve comparison, budget by budget. The crossover is the latency budget at which the ordering flips; where there is no single crossover, the two frontiers interleave and no ordering statement is available at all.</p><div class="scroll"><table><thead><tr><th>Comparison</th><th>Budgets vLLM wins</th><th>Best</th><th>Worst</th><th>Crossover</th></tr></thead><tbody><tr><td>agentic vs sglang</td><td>3%</td><td>+16%</td><td>-51%</td><td>no single crossover</td></tr><tr><td>agentic vs trtllm</td><td>44%</td><td>+2035%</td><td>-8%</td><td>no single crossover</td></tr><tr><td>chat vs sglang</td><td>0%</td><td>-6%</td><td>-75%</td><td>7.6 ms/token</td></tr><tr><td>chat vs trtllm</td><td>0%</td><td>-15%</td><td>-94%</td><td>7.5 ms/token</td></tr><tr><td>rag vs sglang</td><td>0%</td><td>-2%</td><td>-47%</td><td>19.3 ms/token</td></tr><tr><td>rag vs trtllm</td><td>0%</td><td>-8%</td><td>-53%</td><td>19.0 ms/token</td></tr><tr><td>summarize vs sglang</td><td>0%</td><td>-3%</td><td>-48%</td><td>9.3 ms/token</td></tr><tr><td>summarize vs trtllm</td><td>0%</td><td>-7%</td><td>-49%</td><td>9.9 ms/token</td></tr></tbody></table></div>

## Tuning-budget parity

### Tuning-budget parity certificate

Status: **PARITY HELD** (3 finding(s))

| System | Group | Trials | ok | Space (dims / size) | Trials/dim | Coverage | Tuning wall-clock | Still improving |
|--------|-------|--------|----|---------------------|------------|----------|-------------------|-----------------|
| vllm | agentic | 40 | 40 | 6 / 576 | 6.7 | 6.9% | 8.0 min | YES -- budget was binding |
| sglang | agentic | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 8.0 min | no |
| trtllm | agentic | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 288.0 min | no |
| vllm | chat | 40 | 40 | 6 / 576 | 6.7 | 6.9% | 8.0 min | no |
| sglang | chat | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 8.0 min | no |
| trtllm | chat | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 288.0 min | no |
| vllm | rag | 40 | 40 | 6 / 576 | 6.7 | 6.9% | 8.0 min | no |
| sglang | rag | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 8.0 min | no |
| trtllm | rag | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 288.0 min | no |
| vllm | summarize | 40 | 40 | 6 / 576 | 6.7 | 6.9% | 8.0 min | no |
| sglang | summarize | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 8.0 min | no |
| trtllm | summarize | 40 | 40 | 5 / 192 | 8.0 | 20.8% | 288.0 min | no |

| Severity | Code | Finding |
|----------|------|---------|
| warning | `PAR-008` | [agentic] vllm was still improving when its budget ran out; its measured performance is a lower bound and the comparison understates it |
| warning | `PAR-011` | [all 4 groups] equal trial counts explored unequal fractions of each search space: sglang 20.8%, trtllm 20.8%, vllm 6.9%. The engine with the larger space received a sparser search from the same budget, which understates it. |
| warning | `PAR-012` | [all 4 groups] equal trials cost very unequal wall-clock: sglang 8 min, trtllm 288 min, vllm 8 min. Under a fixed time budget rather than a fixed trial budget the comparison would come out differently, and a reader choosing under time pressure should know which budget was held constant. |

## Pre-registration

- Sealed: `2026-08-28T22:48:06Z`
- Seal hash: `sha256:aad2fa03dfc12a8628d665a123351eb5f942000c9cc5be11536f12358a5875dc`
- Verification: **PASS** (0 finding(s))


## Reproduction

```bash
losscolumn run thrust2
```

- Repository: `local checkout` at commit `222dec09b09b51d0386d4ad7b96147460cfc0554` (**dirty tree**)
- Image: `ghcr.io/archanachetan07/losscolumn:0.1.0`
- Hardware: 8x H100 (measured path) / calibratable model (this artifact)
- Estimated runtime: 80 min (~$2000.00)

## Conformance

### Conformance -- `lc-thrust2-equal-tuning-audit` against LC-1.0

**FAIL** -- 1 fatal, 2 warning(s).

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
| `LC-2.1` | Tuning-budget parity | pass | parity certificate present |
| `LC-2.2` | Tuning-budget parity | pass | parity certificate holds |
| `LC-2.3` | Tuning-budget parity | pass | 12 systems tuned under one budget |
| `LC-2.4` | Tuning-budget parity | pass | equal trial budgets (40 per system) |
| `LC-2.5` | Tuning-budget parity | warn | budget was binding for ['vllm']: their performance is a lower bound |
| `LC-2.6` | Tuning-budget parity | pass | operator(s) of record: ['A. S. Patil'] (skill is documented, not eliminated) |
| `LC-2.7` | Tuning-budget parity | pass | the derived baseline `best_alternative` is composed of tuned systems (sglang, trtllm), each with its own ledger |
| `LC-3.1` | Envelope, not point | pass | 2 factors swept |
| `LC-3.2` | Envelope, not point | pass | 24 cells measured |
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
| `LC-5.1` | One-command reproduction | pass | `losscolumn run thrust2` |
| `LC-5.2` | One-command reproduction | pass | reproduction is a single command |
| `LC-5.3` | One-command reproduction | pass | pinned to `222dec09b09b` |
| `LC-5.4` | One-command reproduction | FAIL | the producing working tree was dirty; the pinned commit does not identify the code that ran |
| `LC-5.5` | One-command reproduction | pass | image `ghcr.io/archanachetan07/losscolumn:0.1.0` |
| `LC-5.6` | One-command reproduction | pass | hardware: 8x H100 (measured path) / calibratable model (this artifact) |
| `LC-5.7` | One-command reproduction | pass | ~80 min |
| `LC-Q1` | Measurement quality | pass | 11 replicates per cell |
| `LC-Q2` | Measurement quality | pass | measurements were interleaved A/B |
| `LC-Q3` | Measurement quality | warn | lowest per-system cell coverage is 88% |
| `LC-Q4` | Measurement quality | pass | noise/effect ratio 0.00 |
| `LC-Q5` | Measurement quality | pass | 4 limitation(s) stated |

## Provenance

- Captured: `2026-08-29T01:56:57Z` on `Windows-10-10.0.26200-SP0`, Python 3.11.5
- Hardware fingerprint: `cc72e8eab7a5d10b` (1x NVIDIA T1000 8GB)
- Packages: `torch=2.6.0+cu124`, `numpy=1.26.4`, `transformers=5.12.1`
