"""The measured backend for Thrust I: real FSDP steps under the PyTorch profiler.

This is the path that runs on the 8xA100 allocation. It produces
``StepTrace`` objects with exactly the same shape as the simulator's, so the
attribution, the envelope, the cliff detection and the loss-column extraction
downstream are byte-for-byte the same code on measured and modelled data. When
the two disagree, the disagreement is about physics rather than about tooling.

Two details that decide whether the numbers mean anything:

**Steps are dropped, not averaged.** The first steps of a run include lazy
initialisation, autotuning and allocator growth, and they are several times
slower than steady state. They are discarded by count, and the count is
recorded, rather than being smoothed away into a mean.

**Kernel spans come from the device timeline, not the CPU one.** The profiler
reports both; the CPU ranges include launch queuing and would make every
collective look longer and better overlapped than it is.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from losscolumn.thrusts.overlap.attrib import Span, StepTrace, classify
from losscolumn.thrusts.overlap.simulate import ModelSpec, ShardingConfig

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False


def distributed_state() -> dict[str, Any]:
    """Rank/world information from the launcher environment."""
    return {
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "launched": "RANK" in os.environ,
    }


def spans_from_profile(prof: Any, *, device_only: bool = True) -> list[Span]:
    """Extract kernel spans from a ``torch.profiler`` result.

    Uses the exported Chrome trace rather than the aggregated key averages:
    averages have already thrown away the start times, and start times are the
    entire content of an overlap measurement.
    """
    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8") as fh:
            events = json.load(fh).get("traceEvents", [])
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)

    spans: list[Span] = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if device_only and cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = str(e.get("name", ""))
        ts = float(e.get("ts", 0.0)) / 1000.0        # us -> ms
        dur = float(e.get("dur", 0.0)) / 1000.0
        if dur <= 0:
            continue
        spans.append(
            Span(
                name=name,
                start=ts,
                end=ts + dur,
                stream=e.get("args", {}).get("stream", e.get("tid")),
                category=classify(name),
                bytes=e.get("args", {}).get("bytes"),
            )
        )
    return spans


def split_steps(spans: list[Span], n_steps: int, *, drop_first: int = 0) -> list[StepTrace]:
    """Cut a multi-step timeline into per-step traces at the optimizer kernels.

    Step boundaries are taken at the optimizer update, which is the one
    operation guaranteed to appear exactly once per step and to depend on
    everything before it.
    """
    marks = sorted(
        s.end for s in spans if any(k in s.name.lower() for k in ("adam", "sgd", "optimizer"))
    )
    if len(marks) < 2:
        lo = min((s.start for s in spans), default=0.0)
        hi = max((s.end for s in spans), default=0.0)
        edges = [lo + (hi - lo) * i / max(n_steps, 1) for i in range(n_steps + 1)]
    else:
        edges = [min(s.start for s in spans)] + marks

    traces: list[StepTrace] = []
    for a, b in zip(edges[:-1], edges[1:]):
        sub = [s for s in spans if s.start < b and s.end > a]
        if sub:
            traces.append(StepTrace(spans=sub, step_start=a, step_end=b))
    return traces[drop_first:]


@dataclass
class TorchProfilerBackend:
    """Measure real FSDP steps. Requires a distributed launch to see collectives."""

    build_step: Callable[[ShardingConfig], Callable[[], None]] | None = None
    model: ModelSpec = field(default_factory=ModelSpec)
    hbm_gb: float = 80.0
    warmup_steps: int = 5
    name: str = "torch-profiler"
    evidence_class: str = "measured"

    def feasible(self, cfg: ShardingConfig) -> tuple[bool, str]:
        """Feasibility is decided by running it, not by predicting it.

        The simulator estimates memory; here the honest answer is that an OOM
        is discovered by attempting the configuration. Only the structurally
        impossible cases are rejected up front.
        """
        if not HAS_TORCH:
            return False, "torch is not installed"
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        state = distributed_state()
        if cfg.world_size > 1 and not state["launched"]:
            return False, (
                f"world_size={cfg.world_size} requires a torchrun launch; this process "
                "sees no RANK in its environment"
            )
        if cfg.world_size > 1 and cfg.world_size != state["world_size"]:
            return False, (
                f"configuration asks for world_size={cfg.world_size} but the launcher "
                f"provided {state['world_size']}"
            )
        if cfg.tp_degree > 1 and cfg.world_size % cfg.tp_degree:
            return False, "world size is not divisible by the tensor-parallel degree"
        return True, ""

    def measure(self, cfg: ShardingConfig, *, steps: int = 3, seed: int = 0) -> list[StepTrace]:
        if self.build_step is None:
            raise RuntimeError(
                "TorchProfilerBackend needs a build_step callable that returns a closure "
                "running one training step for the given configuration. The harness "
                "deliberately does not guess how your model is constructed."
            )
        torch.manual_seed(seed)
        step_fn = self.build_step(cfg)

        for _ in range(self.warmup_steps):
            step_fn()
        torch.cuda.synchronize()

        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(steps):
                step_fn()
            torch.cuda.synchronize()

        spans = spans_from_profile(prof)
        traces = split_steps(spans, steps)
        state = distributed_state()
        for t in traces:
            t.rank = state["rank"]
            t.config = {**cfg.to_dict(), "backend": self.name, "world_size": state["world_size"]}
        return traces


def build_fsdp_step(cfg: ShardingConfig, model_spec: ModelSpec | None = None) -> Callable[[], None]:
    """Construct one FSDP training step for the given sharding configuration.

    Kept deliberately small and explicit rather than wrapping a training
    framework: the point of Thrust I is to measure what a sharding choice does,
    and a framework's own scheduling would be an uncontrolled variable sitting
    between the configuration and the timeline.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required")
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    spec = model_spec or ModelSpec()
    state = distributed_state()
    if cfg.world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(state["local_rank"])

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(spec.hidden, 4 * spec.hidden, bias=False)
            self.proj = nn.Linear(spec.hidden, spec.hidden, bias=False)
            self.up = nn.Linear(spec.hidden, spec.ffn_hidden, bias=False)
            self.down = nn.Linear(spec.ffn_hidden, spec.hidden, bias=False)
            self.norm = nn.LayerNorm(spec.hidden)

        def forward(self, x: Any) -> Any:
            h = self.norm(x)
            qkv = self.attn(h)
            q, k, v = qkv.split(qkv.shape[-1] // 4, dim=-1)[:3]
            a = torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
            ).squeeze(1)
            x = x + self.proj(a)
            return x + self.down(torch.nn.functional.silu(self.up(self.norm(x))))

    net = nn.Sequential(*[Block() for _ in range(spec.n_layers)]).cuda()
    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        "ddp": ShardingStrategy.NO_SHARD,
    }.get(cfg.strategy, ShardingStrategy.FULL_SHARD)
    if cfg.world_size > 1:
        net = FSDP(net, sharding_strategy=strategy, use_orig_params=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True)
    x = torch.randn(cfg.micro_batch, cfg.seq_len, spec.hidden, device="cuda", dtype=torch.bfloat16)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
            loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()

    return step
