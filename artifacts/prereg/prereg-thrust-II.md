# Pre-registration -- Thrust II -- the equal-tuning audit

- Version: `1`
- Sealed: `2026-08-28T22:48:06Z`
- Seal hash: `sha256:aad2fa03dfc12a8628d665a123351eb5f942000c9cc5be11536f12358a5875dc`
- Anchor: `git:a0bdfe6`
- Authors: Archana Suresh Patil

## Hypotheses

- **H0.** Under identical tuning budgets, the measured differences between inference engines are within the MDE across the latency-budget range.
- **H1.** A measurable share of published engine-to-engine difference does not survive tuning-budget parity, and the differences that do survive are confined to identifiable workload and latency regions.

## Primary outcome

The loss column of vLLM against the best alternative engine, on throughput attainable under a normalised p99 latency budget, across four workload traces.

## Design

- Systems compared: vllm, sglang, trtllm, best_alternative
- Replicates per cell: 11 (interleaved A/B)
- Minimum effect worth calling (MDE): 5.0%
- FDR level q: 0.05; interval alpha: 0.05
- Tuning budget: 40 trials per system via `random-explore+coordinate-refine@1.0(n=40,explore=0.6)`

### Factors

| Factor | Levels | Unit |
|--------|--------|------|
| `workload` | chat, rag, summarize, agentic | - |
| `latency_budget_ms` | 10, 15, 25, 50, 100, 250 | ms per output token |

### Metrics

| Metric | Unit | Direction |
|--------|------|-----------|
| `attainable_throughput` | tok/s | higher is better |
| `normalised_p99_latency` | ms/token | lower is better |

## Decision rules

1. Every engine receives exactly 40 trials per workload from the same search procedure, with per-engine seeds derived from one master seed.
2. The tuning objective is goodput at concurrency 256 -- throughput that meets the workload's stated SLO -- identically for every engine.
3. Every engine is measured twice per workload: at library defaults and at its tuned optimum. Both are reported.
4. Engines are compared through attainment curves, not at a single operating point; the loss column is derived from where the attainment ratio falls below one.
5. A budget that is still improving at exhaustion is reported as binding, and the affected engine's result is labelled a lower bound.

## Exclusion rules

- A configuration that fails to start or runs out of memory is recorded with its failure and counted against its budget; it is not retried at a milder setting.
- The first 32 requests of every replay are warm-up and excluded from timing.

## Stopping rule

Fixed trial budget per engine per workload; no adaptive allocation.

## Loss-column policy

Every region meeting the MDE at the stated FDR level is published in the loss column of the main artifact, at equal prominence to wins.

## Rationale (non-normative)

Tuning is per workload so that 'this engine is worse here' is not confounded with 'this engine was tuned somewhere else'. Equal trial counts are nominal parity only; trials per dimension, fraction of space covered and tuning wall-clock are reported because equal budgets can still favour the engine with the smaller search space or the cheaper trial.
