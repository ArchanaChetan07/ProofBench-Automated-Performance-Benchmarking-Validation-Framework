"""Thrust III -- the registered sparse-attention sweep.

RQ3, in the form the sealed protocol registers it: over the lattice

    seq_len x batch x sparsity pattern x density x phase

where does an implementation lose to the reference computing the same pattern,
and what accounts for the gap?

Two implementations are swept separately and never pooled:

**A. portable** -- the FlashAttention algorithm in eager PyTorch ops. It is the
reference implementation of the *algorithm*: correct everywhere, portable to
any device, and not fast. Its loss map measures what dispatch overhead costs.

**B. triton** -- the same algorithm as a fused kernel with real block skipping.
Its loss map measures what the code generator achieves.

They answer different questions and are published as separate claims. Merging
them would produce a single number describing neither.

The comparator for a sparse pattern is SDPA computing *that same pattern*
densely under a mask -- not unmasked dense attention. The two compute different
functions, and comparing against the wrong one is how a sparse kernel is made
to show a speedup that is really just its density.

Four metrics are recorded per cell, and the phase decides what two of them
mean: at prefill the latency is the time-to-first-token contribution, at decode
it is the inter-token latency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.losscolumn import LossColumn, extract_loss_column
from losscolumn.core.provenance import Provenance
from losscolumn.core.stats import CellComparison, compare_cells
from losscolumn.thrusts.kernel.bench import (
    clock_stability,
    device_description,
    free_memory,
    paired_ab,
    peak_memory_mb,
)
from losscolumn.thrusts.kernel.correctness import CorrectnessResult, CorrectnessSuite, ShapeSpec
from losscolumn.thrusts.kernel.reference import (
    error_budget,
    flash_torch_sparse,
    math_reference_sparse,
    sparse_reference,
)
from losscolumn.thrusts.kernel.sparsity import (
    attention_flops_sparse,
    build_layout,
    is_applicable,
)
from losscolumn.thrusts.kernel.triton_fa import device_limits, triton_available

METHOD = "implementation"
BASELINE = "reference"

IMPLEMENTATIONS = ("portable", "triton")


@dataclass
class SparseSweepResult:
    """Everything one implementation's registered sweep produces."""

    implementation: str
    latency: Envelope
    throughput: Envelope
    memory: Envelope
    correctness: CorrectnessSuite
    comparisons: list[CellComparison] = field(default_factory=list)
    loss_column: LossColumn | None = None
    layouts: dict[str, dict[str, Any]] = field(default_factory=dict)
    device: str = ""
    clocks: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    phase_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    evidence_class: str = "measured"

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "device": self.device,
            "evidence_class": self.evidence_class,
            "limits": self.limits,
            "clocks": self.clocks,
            "latency_envelope": self.latency.to_dict(),
            "throughput_envelope": self.throughput.to_dict(include_replicates=False),
            "memory_envelope": self.memory.to_dict(include_replicates=False),
            "correctness": self.correctness.to_dict(),
            "loss_column": self.loss_column.to_dict() if self.loss_column else None,
            "layouts": self.layouts,
            "phase_summary": self.phase_summary,
        }


def select(implementation: str) -> tuple[Callable, str]:
    """Resolve an implementation name to its sparse entry point."""
    if implementation == "portable":
        return flash_torch_sparse, "flash_torch_sparse (portable eager ops)"
    if implementation == "triton":
        ok, why = triton_available()
        if not ok:
            raise RuntimeError(f"the triton implementation is unavailable: {why}")
        from losscolumn.thrusts.kernel.triton_fa import flash_attention_triton_sparse

        return flash_attention_triton_sparse, "flash_attention_triton_sparse (fused)"
    raise ValueError(f"unknown implementation {implementation!r}; expected one of "
                     f"{IMPLEMENTATIONS}")


def _factors(seq_lens, batches, patterns, densities, phases) -> tuple[Factor, ...]:
    return (
        Factor("seq_len", tuple(seq_lens), unit="tokens", log_scale=True),
        Factor("batch", tuple(batches)),
        Factor("pattern", tuple(patterns), ordered=False),
        Factor("density", tuple(densities)),
        Factor("phase", tuple(phases), ordered=False),
    )


def _check_correctness(impl, q, k, v, layout, causal, spec: ShapeSpec) -> CorrectnessResult:
    """Correctness against fp64 ground truth for the same sparse pattern.

    Runs before any timing, every cell, exactly as the protocol registers. A
    shape that fails here is never timed: its loss is that it is wrong, and a
    speed number for a wrong kernel is worse than no number.
    """
    import torch

    res = CorrectnessResult(shape=spec, passed=False)
    try:
        out = impl(q, k, v, layout, causal=causal)
        ref = math_reference_sparse(q, k, v, layout, causal=causal)
        denom = ref.abs().amax().clamp_min(1e-12)
        tol = error_budget(spec.torch_dtype, spec.seq_k or spec.seq_len, spec.head_dim)
        err = float(((out.to(torch.float64) - ref).abs().amax() / denom).item())
        finite = bool(torch.isfinite(out).all().item())
        res.checks["output"] = {
            "passed": err <= tol["rel_tol"] and finite,
            "max_rel_err": err,
            "tolerance": tol["rel_tol"],
            "derivation": tol["derivation"],
            "finite": finite,
        }
        out2 = impl(q, k, v, layout, causal=causal)
        res.checks["determinism"] = {
            "passed": bool(torch.equal(out, out2)),
            "note": "bitwise identical across two invocations",
        }
        res.checks["dtype"] = {
            "passed": out.dtype == q.dtype and tuple(out.shape) == tuple(q.shape),
            "got": f"{out.dtype}{tuple(out.shape)}",
        }
        res.passed = all(c.get("passed", False) for c in res.checks.values())
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        res.passed = False
    return res


def run_sparse_sweep(
    *,
    implementation: str = "triton",
    seq_lens: Sequence[int] = (512, 1024, 2048, 4096),
    batches: Sequence[int] = (1, 8),
    patterns: Sequence[str] = ("dense", "sliding_window", "block_sparse"),
    densities: Sequence[float] = (0.125, 0.25, 1.0),
    phases: Sequence[str] = ("prefill", "decode"),
    head_dim: int = 64,
    heads: int = 8,
    dtype: str = "float16",
    causal: bool = True,
    replicates: int = 11,
    iters: int = 10,
    device: str = "cuda",
    mde: float = 0.10,
    q_level: float = 0.05,
    seed: int = 20260101,
) -> SparseSweepResult:
    """Run the registered lattice for one implementation."""
    import torch

    impl, impl_label = select(implementation)
    factors = _factors(seq_lens, batches, patterns, densities, phases)

    def _alloc(metric: Metric, systems: list[str]) -> Envelope:
        return Envelope.allocate(
            factors, metric, systems, replicates,
            interleaved=True,
            workload=f"attention forward, heads={heads}, head_dim={head_dim}, "
                     f"dtype={dtype}, causal={causal}",
            provenance=Provenance.capture(),
            meta={
                "implementation": implementation,
                "implementation_label": impl_label,
                "reference": "torch SDPA computing the same pattern densely under a mask",
                "heads": heads, "head_dim": head_dim, "dtype": dtype, "causal": causal,
                "baseline_tuning_policy": (
                    "the reference is given its own best backend per shape by torch's "
                    "dispatcher; no backend is disabled and no manual restriction is "
                    "applied"
                ),
            },
        )

    latency = _alloc(
        Metric("latency_ms", "ms", higher_is_better=False,
               description="per-call attention latency; TTFT contribution at prefill, "
                           "inter-token latency at decode"),
        [METHOD, BASELINE],
    )
    throughput = _alloc(
        Metric("attention_throughput", "TFLOP/s", higher_is_better=True,
               description="FLOPs over blocks actually computed, per second"),
        [METHOD, BASELINE],
    )
    memory = _alloc(
        Metric("peak_memory_mb", "MB", higher_is_better=False,
               description="peak device allocation during one call"),
        [METHOD, BASELINE],
    )

    suite = CorrectnessSuite(implementation=impl_label, device=device_description(device))
    result = SparseSweepResult(
        implementation=implementation,
        latency=latency, throughput=throughput, memory=memory,
        correctness=suite,
        device=device_description(device),
        clocks={"before": clock_stability(device)},
        limits=device_limits(),
    )

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    for cell in latency.cells():
        c = latency.coords(cell)
        seq, bs = int(c["seq_len"]), int(c["batch"])
        pattern, density, phase = str(c["pattern"]), float(c["density"]), str(c["phase"])

        ok, why = is_applicable(pattern, density)
        if not ok:
            for env in (latency, throughput, memory):
                env.mark_missing(METHOD, cell, why)
                env.mark_missing(BASELINE, cell, why)
            continue

        seq_q = seq if phase == "prefill" else 1
        layout = build_layout(seq_q, seq, pattern=pattern, density=density,
                              causal=causal, seed=seed)
        spec = ShapeSpec(batch=bs, heads=heads, seq_len=seq_q, head_dim=head_dim,
                         dtype=dtype, causal=causal, seq_k=seq)
        free_memory()

        try:
            g = torch.Generator(device=device).manual_seed(seed)
            q = torch.randn(bs, heads, seq_q, head_dim, generator=g,
                            device=device, dtype=torch.float32).to(dt)
            k = torch.randn(bs, heads, seq, head_dim, generator=g,
                            device=device, dtype=torch.float32).to(dt)
            v = torch.randn(bs, heads, seq, head_dim, generator=g,
                            device=device, dtype=torch.float32).to(dt)
        except Exception as e:
            for env in (latency, throughput, memory):
                env.mark_missing(METHOD, cell, f"allocation failed: {e}")
                env.mark_missing(BASELINE, cell, f"allocation failed: {e}")
            continue

        cr = _check_correctness(impl, q, k, v, layout, causal, spec)
        suite.results.append(cr)
        result.layouts[latency.label(cell)] = layout.to_dict()

        run_method = partial(impl, q, k, v, layout, causal=causal)
        run_reference = partial(sparse_reference, q, k, v, layout, causal=causal)

        if not cr.passed:
            reason = f"correctness failed: {', '.join(cr.failures()) or cr.error}"
            for env in (latency, throughput, memory):
                env.mark_missing(METHOD, cell, reason)
            b_only, _ = paired_ab(run_reference, run_reference, replicates=replicates,
                                  iters=iters, device=device)
            latency.put(BASELINE, cell, b_only)
            del q, k, v
            continue

        a_ms, b_ms = paired_ab(run_method, run_reference, replicates=replicates,
                               iters=iters, device=device)
        flops = attention_flops_sparse(bs, heads, head_dim, layout, seq_q, seq)
        latency.put(METHOD, cell, a_ms)
        latency.put(BASELINE, cell, b_ms)
        throughput.put(METHOD, cell, [_tflops(flops, t) for t in a_ms])
        throughput.put(BASELINE, cell, [_tflops(flops, t) for t in b_ms])

        # Peak memory is a property of one call, not of a replicate, so it is
        # measured once per arm and broadcast rather than pretending to have a
        # distribution it does not have.
        memory.put(METHOD, cell, [_peak(run_method)] * replicates)
        memory.put(BASELINE, cell, [_peak(run_reference)] * replicates)
        del q, k, v

    result.clocks["after"] = clock_stability(device)
    result.comparisons = compare_cells(latency, METHOD, BASELINE, mde=mde, q=q_level,
                                       seed=seed, paired=True)
    result.loss_column = extract_loss_column(
        latency, result.comparisons, method=METHOD, baseline=BASELINE,
        mde=mde, q_level=q_level,
    )
    _attribute(result, heads=heads, head_dim=head_dim, causal=causal)
    result.phase_summary = _phase_summary(result)
    return result


def _tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12 if ms and math.isfinite(ms) and ms > 0 else float("nan")


def _peak(fn) -> float:
    import torch

    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return peak_memory_mb()


def _phase_summary(res: SparseSweepResult) -> dict[str, dict[str, float]]:
    """Headline numbers per phase, in the units that phase is judged in."""
    env = res.latency
    out: dict[str, dict[str, float]] = {}
    ax = env.axis("phase")
    for level in env.factor("phase").levels:
        cells = [c for c in env.cells() if env.coords(c)["phase"] == level]
        m = [float(np.median(env.replicates_at(METHOD, c))) for c in cells
             if env.is_measured(METHOD, c)]
        b = [float(np.median(env.replicates_at(BASELINE, c))) for c in cells
             if env.is_measured(BASELINE, c)]
        pairs = [
            (float(np.median(env.replicates_at(METHOD, c))),
             float(np.median(env.replicates_at(BASELINE, c))))
            for c in cells
            if env.is_measured(METHOD, c) and env.is_measured(BASELINE, c)
        ]
        speedups = [bb / mm for mm, bb in pairs if mm > 0]
        out[str(level)] = {
            "n_cells": len(cells),
            "n_measured": len(pairs),
            "median_latency_ms": float(np.median(m)) if m else float("nan"),
            "median_reference_latency_ms": float(np.median(b)) if b else float("nan"),
            "median_speedup": float(np.median(speedups)) if speedups else float("nan"),
            "worst_speedup": float(np.min(speedups)) if speedups else float("nan"),
            "best_speedup": float(np.max(speedups)) if speedups else float("nan"),
            # At prefill this latency is the TTFT contribution; at decode it is
            # the inter-token latency. Named here so the artifact does not make
            # the reader infer it.
            "interpretation": (
                "time-to-first-token contribution" if level == "prefill"
                else "inter-token latency"
            ),
        }
    _ = ax
    return out


def _attribute(res: SparseSweepResult, *, heads: int, head_dim: int, causal: bool) -> None:
    """Attach a mechanism to each loss region, from the measurement."""
    if res.loss_column is None:
        return
    env = res.latency
    for region in res.loss_column.regions:
        cells = [c for c in env.cells()
                 if all(env.coords(c)[k] in v for k, v in region.bounds.items())]
        coords = [env.coords(c) for c in cells]
        phases = {str(x["phase"]) for x in coords}
        patterns = {str(x["pattern"]) for x in coords}
        seqs = [int(x["seq_len"]) for x in coords]
        parts: list[str] = []

        if region.contains_missing and region.missing_reasons:
            parts.append(region.missing_reasons[0])
        elif phases == {"decode"}:
            parts.append(
                f"decode only: one query row is padded to a {64}-wide block, so "
                f"{63 / 64:.0%} of every tile is wasted work a prefill-shaped kernel "
                "cannot avoid"
            )
        else:
            densities = [float(x["density"]) for x in coords]
            if patterns == {"dense"}:
                parts.append("dense pattern: no blocks are skipped, so the comparison "
                             "is purely code generation against the fused reference")
            elif densities and max(densities) < 0.3:
                parts.append(
                    f"sparse at density {min(densities):.3g}: too few blocks per query "
                    "row to amortise the per-block prologue"
                )
            if seqs and max(seqs) <= 1024:
                parts.append(
                    "short sequences: the kernel runs for tens of microseconds and "
                    "fixed per-launch cost is a large share of it"
                )
        if res.implementation == "portable":
            parts.append(
                "the portable implementation dispatches one eager op per tile pair "
                "where the reference launches a single fused kernel"
            )
        region.attribution = "; ".join(dict.fromkeys(parts)) or \
            "no mechanism identified from the measurement"
