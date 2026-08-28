"""A calibratable step model that emits synthetic profiler traces.

This is a *model*, and every artifact produced from it is stamped
``evidence_class="simulated"`` so it can never be mistaken for a measurement.
It exists for three reasons the proposal itself argues for:

1. Thrust III is sequenced first precisely so the measurement pipeline is
   validated before multi-GPU time is committed. The same logic applies one
   level down: the analysis code should be validated before *any* GPU time is
   committed, and that needs data with known ground truth.
2. It gives the loss-column extractor and cliff detector a surface whose
   discontinuities are known analytically, so the tests can assert recovery
   rather than assert nothing.
3. Its parameters -- achievable FLOP/s, bus bandwidth, latency -- are exactly
   the quantities ``nccl-tests`` and a microbenchmark measure. Calibrating the
   model on the real cluster and then comparing model to measurement is a
   sharper result than either alone: where the model mispredicts is where the
   interesting systems behaviour is.

The physics is standard and deliberately simple:

* compute per layer from GEMM and attention FLOPs at an achievable utilisation
* FSDP all-gather / reduce-scatter volumes from the sharded parameter bytes
* ring-collective time as ``latency + (N-1)/N * bytes / busbw``
* prefetch depth 1: layer i's compute can hide layer i+1's all-gather, nothing
  more, which is what the FSDP implementation actually does
* tensor-parallel all-reduces sit on the critical path and do not overlap

The cliffs this produces are not hand-placed. They fall out of the model where
per-layer compute stops being long enough to cover the next all-gather.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.thrusts.overlap.attrib import Span, StepTrace

BYTES_PER_PARAM = 2  # bf16


@dataclass
class ModelSpec:
    """A 7B-class decoder, the size the proposal names for Thrust I."""

    name: str = "llama-7b-class"
    n_layers: int = 32
    hidden: int = 4096
    ffn_hidden: int = 11008
    n_heads: int = 32
    vocab: int = 32000

    @property
    def params_per_layer(self) -> int:
        attn = 4 * self.hidden * self.hidden
        mlp = 3 * self.hidden * self.ffn_hidden
        return attn + mlp

    @property
    def params(self) -> int:
        return self.n_layers * self.params_per_layer + 2 * self.vocab * self.hidden


@dataclass
class Fabric:
    """Hardware and interconnect parameters.

    Defaults are representative A100-80GB SXM numbers; they are *model
    parameters*, not measurements, and ``calibrate()`` replaces them with
    values from nccl-tests and a GEMM microbenchmark on the target cluster.
    """

    name: str = "a100-80gb-nvlink3"
    peak_tflops: float = 312.0            # bf16 dense
    achievable_mfu: float = 0.48          # what a real step sustains
    intra_busbw_gbs: float = 235.0        # NVLink all-reduce bus bandwidth
    inter_busbw_gbs: float = 22.0         # per-rank effective IB bandwidth
    intra_latency_us: float = 8.0
    inter_latency_us: float = 32.0
    gpus_per_node: int = 8
    kernel_launch_us: float = 4.0

    def busbw(self, world: int, inter_node: bool | None = None) -> float:
        crosses = world > self.gpus_per_node if inter_node is None else inter_node
        return self.inter_busbw_gbs if crosses else self.intra_busbw_gbs

    def latency_us(self, world: int, inter_node: bool | None = None) -> float:
        crosses = world > self.gpus_per_node if inter_node is None else inter_node
        return self.inter_latency_us if crosses else self.intra_latency_us

    def collective_ms(
        self,
        nbytes: float,
        world: int,
        kind: str = "all_gather",
        *,
        inter_node: bool | None = None,
    ) -> float:
        """Ring-collective time. ``nbytes`` is the full buffer being moved.

        ``inter_node`` must be given whenever a *small* group nonetheless spans
        nodes -- a hybrid-shard replica group of 4 ranks, one per node, runs at
        InfiniBand speed, not NVLink speed. Inferring the fabric from the group
        size alone silently prices that collective an order of magnitude too
        cheaply, which is exactly the kind of error that makes a sharding
        recommendation look better than it is.
        """
        if world <= 1 or nbytes <= 0:
            return 0.0
        bw = self.busbw(world, inter_node) * 1e9
        lat = self.latency_us(world, inter_node) * 1e-6
        # Ring cost factors: all-gather and reduce-scatter move (N-1)/N of the
        # full buffer; all-reduce is the two composed.
        factor = {"all_gather": 1.0, "reduce_scatter": 1.0, "all_reduce": 2.0}.get(kind, 1.0)
        steps = (world - 1) / world
        return (lat * math.log2(max(world, 2)) + factor * steps * nbytes / bw) * 1e3


@dataclass
class ShardingConfig:
    """One point in the Thrust I sweep."""

    strategy: str = "full_shard"           # full_shard | hybrid_shard | tp_only | ddp
    world_size: int = 8
    tp_degree: int = 1
    micro_batch: int = 4
    seq_len: int = 2048
    activation_checkpointing: bool = False
    prefetch_depth: int = 1

    @property
    def dp_degree(self) -> int:
        return max(self.world_size // max(self.tp_degree, 1), 1)

    @property
    def shard_group(self) -> int:
        """Number of ranks a parameter is sharded across."""
        if self.strategy == "ddp":
            return 1
        if self.strategy == "hybrid_shard":
            return min(self.dp_degree, 8)
        return self.dp_degree

    @property
    def tokens_per_step(self) -> int:
        """Tokens the whole job advances per step.

        Tensor-parallel ranks cooperate on one micro-batch, so they multiply
        neither tokens nor throughput. Reporting per-rank throughput under
        tensor parallelism is a common way to make TP look free; the cluster
        figure is the only one that means anything.
        """
        return self.dp_degree * self.micro_batch * self.seq_len

    def label(self) -> str:
        return (
            f"{self.strategy}/tp{self.tp_degree}/mb{self.micro_batch}/s{self.seq_len}"
            f"{'/ckpt' if self.activation_checkpointing else ''}"
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class LayerBudget:
    compute_ms: float
    ag_ms: float
    rs_ms: float
    tp_ms: float


class StepSimulator:
    """Turns a sharding configuration into a plausible, self-consistent trace."""

    def __init__(self, model: ModelSpec | None = None, fabric: Fabric | None = None,
                 seed: int = 0, jitter: float = 0.035) -> None:
        self.model = model or ModelSpec()
        self.fabric = fabric or Fabric()
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter

    # ---- analytic budgets -------------------------------------------------

    def layer_budget(self, cfg: ShardingConfig) -> LayerBudget:
        m, f = self.model, self.fabric
        b, s = cfg.micro_batch, cfg.seq_len
        h, hf = m.hidden, m.ffn_hidden
        tp = max(cfg.tp_degree, 1)

        # GEMM FLOPs per layer, forward+backward (factor 3 of forward), sharded
        # over the tensor-parallel group.
        gemm_fwd = 2 * b * s * (4 * h * h + 3 * h * hf)
        attn_fwd = 4 * b * m.n_heads * s * s * (h // m.n_heads)
        flops = 3.0 * (gemm_fwd + attn_fwd) / tp
        if cfg.activation_checkpointing:
            flops *= 4.0 / 3.0  # one extra forward
        compute_ms = flops / (f.peak_tflops * 1e12 * f.achievable_mfu) * 1e3
        compute_ms += f.kernel_launch_us * 1e-3 * 12  # ~12 kernels per layer

        shard = cfg.shard_group
        layer_bytes = m.params_per_layer * BYTES_PER_PARAM / tp
        if shard > 1:
            ag = f.collective_ms(layer_bytes, shard, "all_gather")
            rs = f.collective_ms(layer_bytes, shard, "reduce_scatter")
        else:
            ag = 0.0
            rs = f.collective_ms(layer_bytes, cfg.dp_degree, "all_reduce") if cfg.dp_degree > 1 else 0.0

        # Hybrid shard replicates across nodes: gradients still need an
        # inter-node all-reduce. The replica group has one rank per node, so it
        # is small but crosses the slow fabric.
        if cfg.strategy == "hybrid_shard" and cfg.dp_degree > f.gpus_per_node:
            replicas = cfg.dp_degree // f.gpus_per_node
            rs += f.collective_ms(
                layer_bytes / shard, replicas, "all_reduce", inter_node=True
            )

        # Tensor parallelism: two all-reduces of activations per layer forward,
        # two more in backward. These sit between dependent GEMMs.
        tp_ms = 0.0
        if tp > 1:
            act_bytes = b * s * h * BYTES_PER_PARAM
            tp_ms = 4 * f.collective_ms(act_bytes, tp, "all_reduce")
        return LayerBudget(compute_ms, ag, rs, tp_ms)

    # ---- trace generation -------------------------------------------------

    def simulate_step(self, cfg: ShardingConfig, *, step_index: int = 0) -> StepTrace:
        """Emit one step as spans, on two streams, with explicit dependencies.

        A small discrete-event schedule rather than a formula: the compute
        stream and the communication stream each have their own busy-until
        clock, collectives serialise on the comm stream, and compute blocks
        only where the real implementation blocks -- on the all-gather of the
        layer it is about to run, and on tensor-parallel all-reduces. Exposed
        communication is then whatever the interval algebra finds, not
        something the model asserts.
        """
        lb = self.layer_budget(cfg)
        L = self.model.n_layers
        j = lambda x: float(max(x * (1.0 + self.rng.normal(0, self.jitter)), 0.0))  # noqa: E731

        spans: list[Span] = []
        t = 0.0            # compute stream clock
        comm_free = 0.0    # communication stream clock
        ag_done: dict[int, float] = {}

        def enqueue(name: str, earliest: float, dur: float) -> float:
            nonlocal comm_free
            start = max(earliest, comm_free)
            end = start + dur
            spans.append(Span(name, start, end, stream=2, category="comm"))
            comm_free = end
            return end

        # The first all-gather has no prior compute to hide behind: this is the
        # prologue bubble every FSDP step pays, and it is why short steps
        # overlap badly no matter how the rest is tuned.
        if lb.ag_ms > 0:
            ag_done[0] = enqueue("ncclDevKernel_AllGather[prologue,layer0]", 0.0, j(lb.ag_ms))
            t = ag_done[0]

        for i in range(L):
            if lb.ag_ms > 0 and i in ag_done:
                t = max(t, ag_done[i])          # wait for this layer's parameters
            c_dur = j(lb.compute_ms)
            spans.append(
                Span(f"ampere_bf16_gemm[layer{i}]", t, t + c_dur, stream=0, category="compute")
            )
            # Prefetch the next layers' all-gathers while this layer computes.
            if lb.ag_ms > 0:
                for k in range(1, cfg.prefetch_depth + 1):
                    nxt = i + k
                    if nxt < L and nxt not in ag_done:
                        ag_done[nxt] = enqueue(
                            f"ncclDevKernel_AllGather[layer{nxt}]", t + 0.02 * c_dur, j(lb.ag_ms)
                        )
            t += c_dur
            if lb.tp_ms > 0:  # blocks the compute stream between dependent GEMMs
                t = enqueue(f"ncclDevKernel_AllReduce[tp,fwd,layer{i}]", t, j(lb.tp_ms))

        # Backward: gradient compute per layer, each followed by a
        # reduce-scatter that is launched asynchronously and hides under the
        # next layer's compute -- until the comm stream saturates.
        for i in reversed(range(L)):
            c_dur = j(lb.compute_ms * 4.0 / 3.0)   # backward ~2x forward FLOPs, minus recompute
            spans.append(
                Span(f"ampere_bf16_wgrad[layer{i}]", t, t + c_dur, stream=0, category="compute")
            )
            if lb.rs_ms > 0:
                enqueue(f"ncclDevKernel_ReduceScatter[layer{i}]", t + 0.35 * c_dur, j(lb.rs_ms))
            t += c_dur
            if lb.tp_ms > 0:
                t = enqueue(f"ncclDevKernel_AllReduce[tp,bwd,layer{i}]", t, j(lb.tp_ms))

        # The optimizer cannot start until the last gradient has landed.
        t = max(t, comm_free)
        d = j(0.35 + self.model.params * BYTES_PER_PARAM / max(cfg.shard_group, 1) / 1.2e12 * 1e3)
        spans.append(Span("at::native::fused_adamw", t, t + d, stream=0, category="compute"))
        t += d

        return StepTrace(
            spans=spans,
            step_start=0.0,
            step_end=max(t, comm_free),
            rank=0,
            config={
                **cfg.to_dict(),
                "step_index": step_index,
                "model": self.model.name,
                "fabric": self.fabric.name,
            },
        )

    def simulate(self, cfg: ShardingConfig, *, steps: int = 5) -> list[StepTrace]:
        return [self.simulate_step(cfg, step_index=i) for i in range(steps)]

    # ---- feasibility ------------------------------------------------------

    def memory_gb(self, cfg: ShardingConfig) -> float:
        """Rough peak memory per rank, used to decide which cells OOM.

        Cells the configuration cannot run are as much a part of the envelope
        as the ones it can, and they are recorded as unmeasurable rather than
        omitted.
        """
        m = self.model
        shard = max(cfg.shard_group, 1)
        tp = max(cfg.tp_degree, 1)
        # Sharded optimizer state: bf16 params + bf16 grads + fp32 master
        # weights and two Adam moments.
        state = m.params * BYTES_PER_PARAM * (1 + 1 + 6) / shard / tp
        # Saved activations per token per layer, assuming a fused attention
        # kernel (so the s x s score matrix is never materialised).
        per_tok_layer = m.hidden * BYTES_PER_PARAM * 20
        act = cfg.micro_batch * cfg.seq_len * per_tok_layer * m.n_layers / tp
        if cfg.activation_checkpointing:
            # Only layer boundaries are kept, plus one layer live for recompute.
            act = cfg.micro_batch * cfg.seq_len * m.hidden * BYTES_PER_PARAM * (
                m.n_layers + 20
            ) / tp
        return (state + act) / 1e9 + 2.0  # + CUDA context, fragmentation, comm buffers

    def feasible(self, cfg: ShardingConfig, *, hbm_gb: float = 80.0) -> tuple[bool, str]:
        need = self.memory_gb(cfg)
        if need > hbm_gb:
            return False, f"OOM: model needs ~{need:.0f} GB/rank against {hbm_gb:.0f} GB HBM"
        if cfg.tp_degree > self.fabric.gpus_per_node:
            return False, "tensor parallel group would span nodes"
        if cfg.world_size % max(cfg.tp_degree, 1) != 0:
            return False, "world size is not divisible by the tensor-parallel degree"
        return True, ""


def calibrate(fabric: Fabric, *, measured_busbw_gbs: float | None = None,
              measured_tflops: float | None = None) -> Fabric:
    """Replace model parameters with measured ones from the target cluster.

    ``measured_busbw_gbs`` comes from ``all_reduce_perf`` in nccl-tests at the
    message size the model actually uses; ``measured_tflops`` from a GEMM
    microbenchmark at the shapes in the layer budget. Calibrating at the wrong
    message size is the usual way this model goes quietly wrong.
    """
    f = Fabric(**{**fabric.__dict__})
    if measured_busbw_gbs is not None:
        if f.gpus_per_node >= 8:
            f.intra_busbw_gbs = measured_busbw_gbs
        else:
            f.inter_busbw_gbs = measured_busbw_gbs
    if measured_tflops is not None:
        f.achievable_mfu = measured_tflops / f.peak_tflops
    f.name = f"{fabric.name}+calibrated"
    return f
