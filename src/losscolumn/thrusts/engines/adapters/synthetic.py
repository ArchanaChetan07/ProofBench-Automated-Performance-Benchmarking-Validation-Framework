"""Simulated serving engines with distinct, physically-motivated behaviour.

Stamped ``evidence_class="simulated"`` everywhere it touches an artifact. Its
job is to exercise the audit end to end -- tuning under parity, Pareto
frontiers, attainment-curve comparison, loss-column extraction -- so that all
of that is known to work before an hour of H100 time is spent, and so the
harness has a reference case whose right answer is known.

The three engines are not re-skins of one model. Each carries the mechanism
that its literature actually claims, and each carries the cost of that
mechanism:

**vLLM / PagedAttention.** Paged KV blocks give high memory utilisation and
therefore large batches. The scheduler is Python, so per-step overhead is
comparatively large and matters most at small batch. Prefix caching is
available but off by default -- which is itself a finding, since the default is
what most deployments run.

**SGLang / RadixAttention.** A radix tree over KV prefixes converts repeated
system prompts into cache hits. It wins on workloads with reuse and pays a
small bookkeeping cost on workloads without it. Whether it wins is therefore a
property of the trace, and the audit reports reuse fraction per workload.

**TensorRT-LLM.** A compiled C++ runtime with the lowest per-step overhead and
the best kernel efficiency, at the cost of scheduling rigidity: heterogeneous
sequence lengths pack worse into its static engine shapes, and it must be
rebuilt per configuration -- a real cost the audit charges to its tuning time
rather than hiding.

The serving model itself is a closed-loop roofline: decode is memory bound and
reads the weights once per step regardless of batch, prefill is compute bound,
KV capacity bounds the batch, and per-step overhead is charged explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.parity import SearchSpace
from losscolumn.thrusts.engines.adapters.base import ServeResult
from losscolumn.thrusts.engines.workloads import Workload


@dataclass
class ServingModel:
    """Hardware and model constants. Representative H100-80GB values."""

    name: str = "h100-80gb"
    params_b: float = 7.0
    n_layers: int = 32
    n_kv_heads: int = 8               # grouped-query attention
    head_dim: int = 128
    hbm_gb: float = 80.0
    mem_bw_gbs: float = 3350.0
    achievable_bw: float = 0.82
    peak_tflops: float = 989.0        # bf16 dense with sparsity off
    achievable_mfu: float = 0.45

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * 2

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * 2

    @property
    def eff_bw(self) -> float:
        return self.mem_bw_gbs * 1e9 * self.achievable_bw

    @property
    def eff_flops(self) -> float:
        return self.peak_tflops * 1e12 * self.achievable_mfu


@dataclass
class EngineProfile:
    """The knobs that distinguish one engine from another in the model."""

    overhead_step_ms: float           # fixed per-iteration scheduler cost
    overhead_seq_us: float            # per-sequence cost inside an iteration
    kernel_eff: float                 # fraction of achievable bandwidth reached
    kv_efficiency: float              # usable fraction of allocated KV memory
    prefix_cache_default: bool
    prefix_cache_available: bool
    heterogeneity_penalty: float      # sensitivity to length variance in a batch
    build_time_s: float = 0.0         # per-configuration compile cost
    cache_overhead_ms: float = 0.0    # bookkeeping when the cache never hits


PROFILES: dict[str, EngineProfile] = {
    "vllm": EngineProfile(
        overhead_step_ms=1.15, overhead_seq_us=5.5, kernel_eff=0.88,
        kv_efficiency=0.96, prefix_cache_default=False, prefix_cache_available=True,
        heterogeneity_penalty=0.05,
    ),
    "sglang": EngineProfile(
        overhead_step_ms=0.85, overhead_seq_us=4.5, kernel_eff=0.90,
        kv_efficiency=0.94, prefix_cache_default=True, prefix_cache_available=True,
        heterogeneity_penalty=0.05, cache_overhead_ms=0.18,
    ),
    "trtllm": EngineProfile(
        overhead_step_ms=0.32, overhead_seq_us=2.0, kernel_eff=1.0,
        kv_efficiency=0.93, prefix_cache_default=False, prefix_cache_available=False,
        heterogeneity_penalty=0.16, build_time_s=420.0,
    ),
}

SPACES: dict[str, dict[str, list[Any]]] = {
    "vllm": {
        "max_num_seqs": [64, 128, 256, 512],
        "max_num_batched_tokens": [2048, 4096, 8192, 16384],
        "gpu_memory_utilization": [0.85, 0.90, 0.95],
        "enable_chunked_prefill": [True, False],
        "enable_prefix_caching": [True, False],
        "block_size": [8, 16, 32],
    },
    "sglang": {
        "max_running_requests": [64, 128, 256, 512],
        "chunked_prefill_size": [2048, 4096, 8192, 16384],
        "mem_fraction_static": [0.80, 0.86, 0.92],
        "schedule_policy": ["lpm", "fcfs"],
        "disable_radix_cache": [True, False],
    },
    "trtllm": {
        "max_batch_size": [64, 128, 256, 512],
        "max_num_tokens": [2048, 4096, 8192, 16384],
        "kv_cache_free_gpu_mem_fraction": [0.80, 0.88, 0.94],
        "enable_chunked_context": [True, False],
        "scheduler_policy": ["max_utilization", "guaranteed_no_evict"],
    },
}

DEFAULTS: dict[str, dict[str, Any]] = {
    "vllm": {
        "max_num_seqs": 256, "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.90, "enable_chunked_prefill": True,
        "enable_prefix_caching": False, "block_size": 16,
    },
    "sglang": {
        "max_running_requests": 256, "chunked_prefill_size": 8192,
        "mem_fraction_static": 0.86, "schedule_policy": "lpm",
        "disable_radix_cache": False,
    },
    "trtllm": {
        "max_batch_size": 256, "max_num_tokens": 8192,
        "kv_cache_free_gpu_mem_fraction": 0.88,
        "enable_chunked_context": True, "scheduler_policy": "guaranteed_no_evict",
    },
}


@dataclass
class SyntheticEngine:
    """A simulated serving engine."""

    name: str
    model: ServingModel = field(default_factory=ServingModel)
    noise: float = 0.012
    evidence_class: str = "simulated"

    @property
    def profile(self) -> EngineProfile:
        return PROFILES[self.name]

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "evidence_class": self.evidence_class,
            "model": self.model.name,
            "note": "simulated roofline serving model, not a measurement",
            "profile": self.profile.__dict__,
        }

    def search_space(self) -> SearchSpace:
        return SearchSpace(name=self.name, dims=SPACES[self.name])

    def defaults(self) -> dict[str, Any]:
        return dict(DEFAULTS[self.name])

    # ---- normalising the vocabulary ---------------------------------------

    def _norm(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Map each engine's own knob names onto the shared model variables.

        The engines genuinely name the same concept differently; forcing them
        into one vocabulary here is what lets the tuning loop stay identical
        across all three.
        """
        n = self.name
        if n == "vllm":
            return {
                "max_batch": cfg["max_num_seqs"],
                "max_tokens": cfg["max_num_batched_tokens"],
                "kv_fraction": cfg["gpu_memory_utilization"],
                "chunked": cfg["enable_chunked_prefill"],
                "prefix": cfg["enable_prefix_caching"],
                "fragment": {8: 0.02, 16: 0.035, 32: 0.07}.get(cfg["block_size"], 0.04),
            }
        if n == "sglang":
            return {
                "max_batch": cfg["max_running_requests"],
                "max_tokens": cfg["chunked_prefill_size"],
                "kv_fraction": cfg["mem_fraction_static"],
                "chunked": True,
                "prefix": not cfg["disable_radix_cache"],
                "fragment": 0.03,
                "longest_prefix_first": cfg["schedule_policy"] == "lpm",
            }
        return {
            "max_batch": cfg["max_batch_size"],
            "max_tokens": cfg["max_num_tokens"],
            "kv_fraction": cfg["kv_cache_free_gpu_mem_fraction"],
            "chunked": cfg["enable_chunked_context"],
            "prefix": False,
            "fragment": 0.025,
            "conservative": cfg["scheduler_policy"] == "guaranteed_no_evict",
        }

    # ---- the model --------------------------------------------------------

    def serve(
        self,
        config: dict[str, Any],
        workload: Workload,
        *,
        concurrency: int,
        seed: int = 0,
    ) -> ServeResult:
        rng = np.random.default_rng(seed)
        m, p = self.model, self.profile
        c = self._norm(config)

        prompts = np.array([r.prompt_tokens for r in workload.requests], dtype=float)
        outs = np.array([r.output_tokens for r in workload.requests], dtype=float)
        if prompts.size == 0:
            return ServeResult(*([float("nan")] * 6), error="empty workload")

        # --- KV capacity bounds the batch -------------------------------
        kv_gb = m.hbm_gb * c["kv_fraction"] - m.weight_bytes / 1e9 - 2.0
        if kv_gb <= 0:
            return ServeResult(
                *([float("nan")] * 6),
                error=f"KV cache does not fit at kv_fraction={c['kv_fraction']}",
            )
        kv_tokens = kv_gb * 1e9 * p.kv_efficiency * (1 - c["fragment"]) / m.kv_bytes_per_token
        mean_ctx = float(prompts.mean() + outs.mean() / 2)
        kv_batch = int(kv_tokens / max(mean_ctx, 1))
        batch = int(max(min(concurrency, c["max_batch"], kv_batch), 1))
        kv_limited = batch < min(concurrency, c["max_batch"])

        # --- prefix cache -------------------------------------------------
        reuse = workload.prefix_reuse_fraction if c["prefix"] else 0.0
        if c["prefix"] and c.get("longest_prefix_first"):
            reuse *= 1.12  # scheduling by longest prefix raises the hit rate
        reuse = min(reuse, 0.95)
        eff_prompts = prompts * (1 - reuse)

        # --- decode step: memory bound ------------------------------------
        weights_ms = m.weight_bytes / m.eff_bw * 1e3
        kv_read_ms = batch * mean_ctx * m.kv_bytes_per_token / m.eff_bw * 1e3
        het = float(np.std(prompts) / max(np.mean(prompts), 1.0))
        t_step = (weights_ms + kv_read_ms) / p.kernel_eff
        t_step += p.overhead_step_ms + p.overhead_seq_us * batch / 1000.0
        t_step *= 1.0 + p.heterogeneity_penalty * het
        if c["prefix"] and reuse < 0.02:
            t_step += p.cache_overhead_ms       # bookkeeping with nothing to show for it
        if not c["chunked"]:
            # Without chunked prefill a long prompt blocks the decode loop, and
            # every in-flight request pays for it.
            t_step += float(np.percentile(eff_prompts, 90)) / m.eff_flops * 2 * m.params_b * 1e9 * 1e3 * 0.35

        # --- prefill: compute bound ---------------------------------------
        flops_per_token = 2 * m.params_b * 1e9
        prefill_ms = eff_prompts * flops_per_token / m.eff_flops * 1e3
        chunk_penalty = np.maximum(eff_prompts / c["max_tokens"], 1.0) * 0.06
        prefill_ms = prefill_ms * (1 + chunk_penalty)

        # --- queueing: closed loop with `concurrency` clients ---------------
        queue_factor = max(concurrency / batch, 1.0)
        service_ms = prefill_ms + outs * t_step
        wait_ms = (queue_factor - 1.0) * float(np.median(service_ms))

        ttft = wait_ms + prefill_ms
        e2e = ttft + outs * t_step
        jitter = 1.0 + rng.normal(0, self.noise, size=e2e.shape)
        e2e = e2e * jitter
        ttft = ttft * jitter

        out_tok_s = float(outs.sum() / (e2e.sum() / concurrency) * 1e3) if e2e.sum() else 0.0
        tot_tok_s = float((outs.sum() + prompts.sum()) / (e2e.sum() / concurrency) * 1e3)
        req_s = float(len(outs) / (e2e.sum() / concurrency) * 1e3)

        meets = np.ones(len(outs), dtype=bool)
        if workload.slo_ttft_ms:
            meets &= ttft <= workload.slo_ttft_ms
        if workload.slo_tpot_ms:
            meets &= t_step <= workload.slo_tpot_ms
        goodput = req_s * float(meets.mean())

        norm = e2e / np.maximum(outs, 1.0)
        return ServeResult(
            ttft_ms_p50=float(np.percentile(ttft, 50)),
            ttft_ms_p99=float(np.percentile(ttft, 99)),
            tpot_ms_p50=float(t_step),
            tpot_ms_p99=float(t_step * 1.08),
            e2e_ms_p50=float(np.percentile(e2e, 50)),
            e2e_ms_p99=float(np.percentile(e2e, 99)),
            norm_latency_ms_p50=float(np.percentile(norm, 50)),
            norm_latency_ms_p99=float(np.percentile(norm, 99)),
            output_throughput=out_tok_s,
            total_throughput=tot_tok_s,
            request_throughput=req_s,
            goodput=goodput,
            n_ok=len(outs),
            n_failed=0,
            peak_kv_utilisation=min(batch * mean_ctx / max(kv_tokens, 1), 1.0),
            prefix_hit_rate=reuse,
            meta={
                "batch": batch,
                "kv_limited": kv_limited,
                "kv_tokens": kv_tokens,
                "t_step_ms": t_step,
                "build_time_s": p.build_time_s,
                "heterogeneity": het,
            },
        )


def synthetic_engines(model: ServingModel | None = None) -> dict[str, SyntheticEngine]:
    m = model or ServingModel()
    return {n: SyntheticEngine(name=n, model=m) for n in PROFILES}
