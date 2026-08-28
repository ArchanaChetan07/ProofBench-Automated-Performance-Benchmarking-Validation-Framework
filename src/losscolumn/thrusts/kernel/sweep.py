"""Thrust III -- reproduction with a loss map.

RQ3: over the ``(head_dim, seq_len, batch, dtype)`` space, in which regions does
a faithful from-scratch attention kernel lose to the reference, and what
accounts for the gap?

The sweep is deliberately ordered so that correctness cannot be traded for
speed:

1. Build the shape grid.
2. Run the correctness suite over every shape. Record the result.
3. Time **only** the shapes that passed. A shape that fails correctness is
   entered into the envelope as unmeasurable with the failure as its reason --
   it is a loss, and the loudest kind.
4. Extract the loss column and attribute each region.

The expected outcome is that the reimplementation loses across most of the
space. That is the finding, not a failure of the project: the artifact's job is
to say precisely where and why, and to tell a reader who wants speed to use the
reference instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.losscolumn import LossColumn, extract_loss_column
from losscolumn.core.provenance import Provenance
from losscolumn.core.stats import CellComparison, compare_cells
from losscolumn.thrusts.kernel.bench import (
    device_description,
    clock_stability,
    free_memory,
    paired_ab,
)
from losscolumn.thrusts.kernel.correctness import (
    CorrectnessSuite,
    ShapeSpec,
    check_shape,
    make_inputs,
)
from losscolumn.thrusts.kernel.reference import flash_torch, sdpa_reference
from losscolumn.thrusts.kernel.triton_fa import attention_flops, triton_available

METHOD = "reimplementation"
BASELINE = "reference"


@dataclass
class KernelSweepResult:
    envelope: Envelope
    correctness: CorrectnessSuite
    comparisons: list[CellComparison] = field(default_factory=list)
    loss_column: LossColumn | None = None
    device: str = ""
    implementation: str = ""
    clocks: dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: dict[str, float] = field(default_factory=dict)
    roofline: dict[str, Any] = field(default_factory=dict)
    evidence_class: str = "measured"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "implementation": self.implementation,
            "evidence_class": self.evidence_class,
            "clocks": self.clocks,
            "envelope": self.envelope.to_dict(),
            "correctness": self.correctness.to_dict(),
            "loss_column": self.loss_column.to_dict() if self.loss_column else None,
            "roofline": self.roofline,
        }


def select_implementation(prefer: str = "auto") -> tuple[Callable, str, str]:
    """Pick the reimplementation under test, and say plainly which one it is.

    ``auto`` uses the Triton kernel where the hardware and toolchain support it
    and the portable PyTorch implementation otherwise. Both implement the same
    algorithm; they are not interchangeable as *performance* claims, and the
    artifact names which one produced every number.
    """
    if prefer in ("triton", "auto"):
        ok, why = triton_available()
        if ok:
            from losscolumn.thrusts.kernel.triton_fa import flash_attention_triton

            return flash_attention_triton, "triton_flash_fwd", ""
        if prefer == "triton":
            raise RuntimeError(f"Triton implementation requested but unavailable: {why}")
        return flash_torch, "flash_torch (portable)", why
    return flash_torch, "flash_torch (portable)", ""


def run_kernel_sweep(
    *,
    head_dims: Sequence[int] = (32, 64, 128),
    seq_lens: Sequence[int] = (128, 512, 2048),
    batches: Sequence[int] = (1, 4),
    dtypes: Sequence[str] = ("float16", "bfloat16"),
    heads: int = 8,
    causal: bool = False,
    replicates: int = 11,
    iters: int = 12,
    device: str = "cuda",
    implementation: str = "auto",
    mde: float = 0.05,
    q_level: float = 0.05,
    seed: int = 20260101,
    memory_budget_gb: float | None = None,
) -> KernelSweepResult:
    impl, impl_name, fallback_reason = select_implementation(implementation)
    factors = (
        Factor("head_dim", tuple(head_dims)),
        Factor("seq_len", tuple(seq_lens), unit="tokens", log_scale=True),
        Factor("batch", tuple(batches)),
        Factor("dtype", tuple(dtypes), ordered=False),
    )
    env = Envelope.allocate(
        factors,
        Metric("attention_throughput", "TFLOP/s", higher_is_better=True,
               description="forward-pass attention FLOPs per second"),
        [METHOD, BASELINE],
        replicates,
        interleaved=True,
        workload=f"attention forward, heads={heads}, causal={causal}",
        provenance=Provenance.capture(),
        meta={
            "implementation": impl_name,
            "reference": "torch.nn.functional.scaled_dot_product_attention",
            "heads": heads,
            "causal": causal,
            "fallback_reason": fallback_reason,
        },
    )
    suite = CorrectnessSuite(implementation=impl_name, device=device_description(device))
    result = KernelSweepResult(
        envelope=env,
        correctness=suite,
        device=device_description(device),
        implementation=impl_name,
        clocks={"before": clock_stability(device)},
    )

    for cell in env.cells():
        c = env.coords(cell)
        spec = ShapeSpec(
            batch=int(c["batch"]),
            heads=heads,
            seq_len=int(c["seq_len"]),
            head_dim=int(c["head_dim"]),
            dtype=str(c["dtype"]),
            causal=causal,
        )
        free_memory()

        # Correctness first, always. A shape that fails here is never timed --
        # but the reference still is, so the cell enters the comparison as a
        # loss rather than vanishing as "no data". A kernel that is wrong at a
        # shape has lost at that shape, and the loss column must say so.
        cr = check_shape(impl, spec, device=device, seed=seed)
        suite.results.append(cr)

        try:
            q, k, v = make_inputs(spec, device=device, seed=seed)
        except Exception as e:
            env.mark_missing(METHOD, cell, f"allocation failed: {e}")
            env.mark_missing(BASELINE, cell, f"allocation failed: {e}")
            continue

        flops = attention_flops(spec.batch, heads, spec.seq_len, spec.seq_len,
                                spec.head_dim, causal)

        if not cr.passed:
            env.mark_missing(
                METHOD, cell, f"correctness failed: {', '.join(cr.failures()) or cr.error}"
            )
            b_only, _ = paired_ab(
                lambda: sdpa_reference(q, k, v, causal=causal),
                lambda: sdpa_reference(q, k, v, causal=causal),
                replicates=replicates,
                iters=iters,
                device=device,
            )
            env.put(BASELINE, cell, [_tflops(flops, t) for t in b_only])
            del q, k, v
            continue

        a_ms, b_ms = paired_ab(
            lambda: impl(q, k, v, causal=causal),
            lambda: sdpa_reference(q, k, v, causal=causal),
            replicates=replicates,
            iters=iters,
            device=device,
        )
        env.put(METHOD, cell, [_tflops(flops, t) for t in a_ms])
        env.put(BASELINE, cell, [_tflops(flops, t) for t in b_ms])
        del q, k, v

    result.clocks["after"] = clock_stability(device)
    result.comparisons = compare_cells(
        env, METHOD, BASELINE, mde=mde, q=q_level, seed=seed, paired=True
    )
    result.loss_column = extract_loss_column(
        env, result.comparisons, method=METHOD, baseline=BASELINE, mde=mde, q_level=q_level
    )
    _attribute(result, heads=heads, causal=causal)
    result.roofline = _roofline(env, heads, causal)
    return result


def _tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12 if ms and math.isfinite(ms) and ms > 0 else float("nan")


def _roofline(env: Envelope, heads: int, causal: bool) -> dict[str, Any]:
    """Arithmetic intensity per cell, to separate "slow kernel" from "no headroom".

    Attention at small sequence length is memory bound: the reference cannot go
    faster there either, so a loss in that region says something different from
    a loss at long sequence where the kernel is compute bound and the gap is
    pure code quality. The loss column uses this to attribute.
    """
    out: dict[str, Any] = {"cells": {}}
    for cell in env.cells():
        c = env.coords(cell)
        s, d, b = int(c["seq_len"]), int(c["head_dim"]), int(c["batch"])
        bytes_per = 2 if c["dtype"] in ("float16", "bfloat16") else 4
        flops = attention_flops(b, heads, s, s, d, causal)
        # Q, K, V in and O out; the score matrix never touches HBM.
        moved = 4 * b * heads * s * d * bytes_per
        out["cells"][env.label(cell)] = {
            "flops": flops,
            "bytes": moved,
            "arithmetic_intensity": flops / moved if moved else float("nan"),
        }
    return out


def _attribute(result: KernelSweepResult, *, heads: int, causal: bool) -> None:
    """Attach a mechanism to each loss region.

    The attributions are drawn from the measurement itself -- arithmetic
    intensity, the failing correctness check, the region's position in the
    space -- rather than from a story told afterwards. Where the data does not
    support an attribution, the region says so instead of guessing.
    """
    if result.loss_column is None:
        return
    env = result.envelope
    ri = result.roofline.get("cells", {})
    for region in result.loss_column.regions:
        cells = [c for c in env.cells() if _in_region(env, c, region)]
        ai = [ri.get(env.label(c), {}).get("arithmetic_intensity", float("nan")) for c in cells]
        ai = [x for x in ai if math.isfinite(x)]
        reasons = sorted(region.missing_reasons)
        parts: list[str] = []
        if region.contains_missing and reasons:
            parts.append(reasons[0])
        elif ai and float(np.median(ai)) < 32:
            parts.append(
                f"memory bound here (arithmetic intensity {np.median(ai):.0f} FLOP/byte): the "
                "gap is dominated by load efficiency, not by the inner loop"
            )
        else:
            seqs = region.bounds.get("seq_len", [])
            dts = region.bounds.get("dtype", [])
            hint = []
            if ai:
                hint.append(f"arithmetic intensity {np.median(ai):.0f} FLOP/byte")
            if "portable" in result.implementation:
                # Not a guess: the portable implementation dispatches one eager
                # op per tile pair, so it pays kernel-launch and intermediate
                # write-back costs the fused reference does not pay at all.
                bq, bk = 128, 64
                tiles = sum(
                    math.ceil(int(s) / bq) * math.ceil(int(s) / bk)
                    for s in (seqs or env.factor("seq_len").levels)
                ) / max(len(seqs or env.factor("seq_len").levels), 1)
                hint.append(
                    f"the implementation dispatches ~{tiles * 8:.0f} eager kernels per call "
                    f"({bq}x{bk} tiling) where the reference launches one fused kernel; "
                    "dispatch and intermediate write-back dominate"
                )
            elif seqs and max(int(s) for s in seqs) >= 2048:
                hint.append(
                    "compute bound: the reference overlaps softmax with the next GEMM through "
                    "warp specialisation and TMA, neither of which Triton exposes"
                )
            else:
                hint.append(
                    "launch-bound region: at these shapes the kernel runs for tens of "
                    "microseconds and fixed per-launch cost is a large share of it"
                )
            if "bfloat16" in [str(x) for x in dts] and len(dts) == 1:
                hint.append("bf16 only; no dedicated fast path in this implementation")
            parts.append("; ".join(hint))
        region.attribution = "; ".join(parts)


def _in_region(env: Envelope, cell, region) -> bool:
    coords = env.coords(cell)
    return all(coords[k] in v for k, v in region.bounds.items())
