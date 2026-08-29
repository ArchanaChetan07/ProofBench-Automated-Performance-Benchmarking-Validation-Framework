# The Loss Column -- synthesis

For every claim tested, whether it survived. A null result is reported with the same prominence as a refutation: if the audited claims hold under controlled comparison, that is itself a finding about the evidentiary health of the field.

| Claim | Thrust | Evidence | Verdict | Basis | Conformance |
|-------|--------|----------|---------|-------|-------------|
| Reproduction with a loss map: a from-scratch attention kernel | III | measured | **held with exceptions** | holds over 94% of the envelope; 1 cell(s) regress | conforming-with-warnings (0F/1W) |
| The overlap envelope: where the recommended sharding configuration breaks down | I | simulated | **did not hold** | regresses over 25% of the swept envelope | conforming-with-warnings (0F/1W) |
| The equal-tuning audit: engine differences that survive budget parity | II | simulated | **did not hold** | regresses over 75% of the swept envelope | conforming-with-warnings (0F/2W) |

## Loss columns

### Reproduction with a loss map: a from-scratch attention kernel

reimplementation loses to reference on 6% of the swept envelope (1/18 cells), across 1 region(s); worst measurable case +32.7% on attention_throughput.

1. **head_dim=[128], seq_len=[128], batch=[1], dtype=['float16']** -- +32.7%; launch-bound region: at these shapes the kernel runs for tens of microseconds and fixed per-launch cost is a large share of it

### The overlap envelope: where the recommended sharding configuration breaks down

full_shard/tp1 loses to best_of_swept on 25% of the swept envelope (12/48 cells), across 3 region(s); worst measurable case +258.1% on cluster_throughput, and cannot run at all in 8 cell(s).

1. **micro_batch=[2, 4, 8], seq_len=[8192], checkpointing=[False], world_size=[8, 32]** -- cannot run; the configuration cannot run here at all; OOM: model needs ~101 GB/rank against 80 GB HBM; beaten by full_shard/tp2, full_shard/tp8
2. **micro_batch=[8], seq_len=[2048, 8192], checkpointing=[False], world_size=[8, 32]** -- cannot run; the configuration cannot run here at all; OOM: model needs ~101 GB/rank against 80 GB HBM; beaten by full_shard/tp2, full_shard/tp8
3. **micro_batch=[1, 2], seq_len=[512], checkpointing=[False, True], world_size=[32]** -- +258.1%; overlap efficiency 0.45; 54.5% of the step is exposed communication; beaten by hybrid_shard/tp1

### The equal-tuning audit: engine differences that survive budget parity

vllm loses to best_alternative on 75% of the swept envelope (18/24 cells), across 3 region(s); worst measurable case +107.7% on attainable_throughput, and cannot run at all in 1 cell(s).

1. **workload=['chat', 'rag', 'agentic'], latency_budget_ms=[10, 15, 25, 50, 100]** -- +107.7%; vllm cannot meet a 10 ms/token normalised p99 budget on rag at any measured concurrency; rag has 15% prefix reuse, which the alternative caches and vllm does not at its tuned settings; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings
2. **workload=['chat', 'agentic'], latency_budget_ms=[10, 15, 25, 50, 100, 250]** -- +94.8%; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings
3. **workload=['chat', 'summarize', 'agentic'], latency_budget_ms=[10, 15, 25]** -- +94.8%; vllm cannot meet a 10 ms/token normalised p99 budget on summarize at any measured concurrency; agentic has 82% prefix reuse, which the alternative caches and vllm does not at its tuned settings
