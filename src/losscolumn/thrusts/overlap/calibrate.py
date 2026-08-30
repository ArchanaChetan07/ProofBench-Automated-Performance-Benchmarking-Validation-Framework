"""Calibrating the Thrust I step model against measurement.

Thrust I's envelope is simulated, because the measured version needs a cluster.
That makes its claim conditional: *if the model is right, the recommended
configuration loses here*. A conditional claim is worth something only once
somebody has checked how often the condition holds. This module is that check.

The design problem is that the model's headline decision -- which FSDP sharding
to use -- collapses on a single device. But the *shape* of the decision does
not. Thrust I asks:

    at each point of the envelope, is the configuration people are told to use
    beaten by another configuration in the swept space?

and that question restricts cleanly to the two axes one GPU can vary: micro
batch and activation checkpointing. So this study runs Thrust I's own question
at reduced scope, gets a predicted loss map from the model and a measured loss
map from the hardware, and scores one against the other.

What that establishes, and what it does not:

* It establishes how well the model predicts *which configuration wins* and *by
  how much* on the axes it can be checked on -- the compute half.
* It does not touch the communication half. Bus bandwidth, the per-collective
  latency floor, and whether a real runtime achieves the overlap the model
  assumes are all unmeasurable here, and Thrust I's loss regions are driven by
  exposed communication. The parameters that matter most to its published
  conclusion remain unchecked, and the report says so in those words.

Half a model checked is worth more than a whole model asserted, provided nobody
is misled about which half.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.calibration import CalibrationResult, calibrate, suggest_update
from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.provenance import Provenance
from losscolumn.core.stats import CellComparison, compare_cells
from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description, paired_ab
from losscolumn.thrusts.overlap.simulate import Fabric, ModelSpec

RECOMMENDED = "recommended"
BEST = "best_of_swept"


@dataclass
class ThrustOneCalibration:
    """How reliable Thrust I's simulated loss map is, where it can be checked."""

    predicted: Envelope
    measured: Envelope
    result: CalibrationResult
    peak_tflops: float = float("nan")
    measured_mfu: float = float("nan")
    assumed_mfu: float = float("nan")
    checkpoint_penalty: dict[str, Any] = field(default_factory=dict)
    update: dict[str, Any] = field(default_factory=dict)
    device: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    uncalibrated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "scope": self.scope,
            "peak_tflops": self.peak_tflops,
            "measured_mfu": self.measured_mfu,
            "assumed_mfu": self.assumed_mfu,
            "checkpoint_penalty": self.checkpoint_penalty,
            "calibration": self.result.to_dict(),
            "model_update": self.update,
            "uncalibrated": self.uncalibrated,
            "predicted_envelope": self.predicted.to_dict(include_replicates=False),
            "measured_envelope": self.measured.to_dict(),
        }

    def to_markdown(self) -> str:
        cp = self.checkpoint_penalty
        lines = [
            "### Calibration study -- Thrust I, compute half",
            "",
            f"Measured on {self.device}. The model was asked for its prediction before "
            "anything was measured, and was not adjusted afterwards.",
            "",
            "| Model parameter | Assumed | Measured | Ratio |",
            "|---|---|---|---|",
            f"| Achievable MFU | {self.assumed_mfu:.3f} | {self.measured_mfu:.3f} | "
            f"{self.measured_mfu / self.assumed_mfu:.2f}x |",
        ]
        if cp:
            lines.append(
                f"| Checkpointing penalty | {cp.get('assumed', float('nan')):.2f}x | "
                f"{cp.get('measured_median', float('nan')):.2f}x | "
                f"{cp.get('ratio', float('nan')):.2f}x |"
            )
        lines += ["", self.result.to_markdown(), ""]
        if cp.get("varies"):
            lines += [
                "The model applies a single flat multiplier for activation "
                f"checkpointing. Measured, the penalty ranges from "
                f"{cp['measured_min']:.2f}x to {cp['measured_max']:.2f}x across the "
                "grid, so the flat multiplier is not merely mis-sized but the wrong "
                "shape: it cannot reproduce a penalty that depends on the operating "
                "point.",
                "",
            ]
        lines += ["**Not calibrated by this study**", ""]
        lines += [f"- {u}" for u in self.uncalibrated]
        return "\n".join(lines)


def deterministic_comparisons(
    env: Envelope, method: str, baseline: str, *, mde: float
) -> list[CellComparison]:
    """Verdicts for a deterministic model, without inventing sampling noise.

    A closed-form model has no replicate variance. Running the paired
    statistics over its output would manufacture a confidence interval from
    numbers that never varied, so the verdict here is a threshold on the
    predicted effect and the interval is the point itself. Being explicit about
    this is the difference between a prediction and a fake measurement.
    """
    out: list[CellComparison] = []
    sign = -1.0 if env.metric.higher_is_better else 1.0
    for cell in env.cells():
        if not env.is_measured(baseline, cell):
            continue
        # A configuration the model says cannot run is a LOSS, not a cell to
        # skip. Skipping it was a real bug here: four cells the model predicted
        # infeasible vanished from the comparison, the predicted loss map came
        # back empty, and the calibration reported that it could establish
        # nothing -- when in fact the model had made four falsifiable
        # predictions and two of them were wrong.
        if not env.is_measured(method, cell):
            reason = (env.missing.get(method, {}) or {}).get(tuple(cell), "unavailable")
            out.append(CellComparison(
                cell=cell, coords=env.coords(cell), effect=float("inf"),
                ci_lo=float("inf"), ci_hi=float("inf"),
                p_worse=0.0, p_better=1.0, q_worse=0.0, q_better=1.0, p_equiv=1.0,
                verdict="loss", n_method=0, n_baseline=1,
                method_point=float("nan"),
                baseline_point=float(np.median(env.replicates_at(baseline, cell))),
                paired=False, reason=reason,
                meta={"deterministic": True, "unrunnable": True},
            ))
            continue
        m = float(np.median(env.replicates_at(method, cell)))
        b = float(np.median(env.replicates_at(baseline, cell)))
        if b <= 0:
            # No reference to compare against; nothing can be said about this cell.
            continue
        if m <= 0:
            # A measured zero is a MEASUREMENT, not an absence. Skipping it was
            # the defect that made predicted losses vanish from the map: a
            # configuration that produced no throughput is the most extreme loss
            # there is, and dropping it reported the opposite.
            out.append(CellComparison(
                cell=cell, coords=env.coords(cell), effect=float("inf"),
                ci_lo=float("inf"), ci_hi=float("inf"),
                p_worse=0.0, p_better=1.0, q_worse=0.0, q_better=1.0, p_equiv=1.0,
                verdict="loss", n_method=1, n_baseline=1,
                method_point=m, baseline_point=b, paired=False,
                reason="measured a throughput of zero",
                meta={"deterministic": True, "zero_throughput": True},
            ))
            continue
        effect = sign * (math.log(m) - math.log(b))
        verdict = "loss" if effect >= math.log1p(mde) else (
            "win" if effect <= -math.log1p(mde) else "tie"
        )
        out.append(CellComparison(
            cell=cell, coords=env.coords(cell), effect=effect,
            ci_lo=effect, ci_hi=effect,
            p_worse=0.0 if verdict == "loss" else 1.0,
            p_better=0.0 if verdict == "win" else 1.0,
            q_worse=0.0 if verdict == "loss" else 1.0,
            q_better=0.0 if verdict == "win" else 1.0,
            p_equiv=0.0 if verdict == "tie" else 1.0,
            verdict=verdict, n_method=1, n_baseline=1,
            method_point=m, baseline_point=b, paired=False,
            reason="deterministic model prediction; no sampling uncertainty",
            meta={"deterministic": True},
        ))
    return out


def measure_peak_gemm(*, device: str = "cuda", dtype: str = "float16",
                      n: int = 4096, replicates: int = 5, iters: int = 20) -> float:
    """Sustained large-GEMM throughput, in TFLOP/s.

    Measured rather than quoted: a datasheet peak assumes a clock and a duty
    cycle no real workload sustains, and the model's ``peak_tflops *
    achievable_mfu`` product is trying to predict this number, not that one.
    """
    import torch

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    a = torch.randn(n, n, device=device, dtype=dt)
    b = torch.randn(n, n, device=device, dtype=dt)

    def gemm() -> None:
        a.matmul(b)

    times, _ = paired_ab(gemm, gemm, replicates=replicates, iters=iters, device=device)
    ms = float(np.median(times))
    torch.cuda.empty_cache()
    return 2.0 * n**3 / (ms * 1e-3) / 1e12 if ms > 0 else float("nan")


def _block(hidden: int, ffn: int, heads: int, dtype, device: str):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
            self.proj = nn.Linear(hidden, hidden, bias=False)
            self.up = nn.Linear(hidden, ffn, bias=False)
            self.down = nn.Linear(ffn, hidden, bias=False)
            self.h = heads

        def forward(self, x):
            b, s, _ = x.shape
            qkv = self.qkv(x).view(b, s, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a.transpose(1, 2).reshape(b, s, -1))
            return x + self.down(torch.nn.functional.gelu(self.up(x)))

    return Block().to(device=device, dtype=dtype)


def run_thrust_one_calibration(
    *,
    # Sized for a development card. The registered Thrust I grid is wider, and
    # the point of a calibration study is not to reproduce the grid -- it is to
    # get enough spread in the predicted effect to fit a slope against measured
    # reality. On a 50W card the full grid costs about three hours, most of it
    # in one corner, and a study nobody can afford to run is not a check.
    micro_batches: Sequence[int] = (1, 2, 4),
    seq_lens: Sequence[int] = (512, 1024),
    checkpointing: Sequence[bool] = (False, True),
    hidden: int = 1024,
    heads: int = 16,
    ffn_mult: int = 4,
    spec: ModelSpec | None = None,
    fabric: Fabric | None = None,
    device: str = "cuda",
    dtype: str = "float16",
    replicates: int = 11,
    iters: int = 3,
    mde: float = 0.05,
    q_level: float = 0.05,
    seed: int = 20260101,
    progress: bool = True,
) -> ThrustOneCalibration:
    """Run Thrust I's question at reduced scope, predicted and measured.

    The recommended configuration is the largest micro batch -- the advice a
    practitioner is most often given, and the one Thrust I's H0 is about. The
    baseline is the best of the swept micro batches at that cell. Where the
    recommendation is beaten, that cell is a loss.
    """
    import torch

    spec = spec or ModelSpec()
    fabric = fabric or Fabric()
    ffn = ffn_mult * hidden
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    factors = (
        Factor("seq_len", tuple(seq_lens), unit="tokens", log_scale=True),
        Factor("checkpointing", tuple(checkpointing)),
    )
    metric = Metric("layer_throughput", "tok/s", higher_is_better=True,
                    description="tokens per second through one transformer block, "
                                "forward and backward")

    def _alloc(reps: int, interleaved: bool, note: str) -> Envelope:
        return Envelope.allocate(
            factors, metric, [RECOMMENDED, BEST], reps, interleaved=interleaved,
            workload=f"transformer block, hidden={hidden}, heads={heads}, ffn={ffn}",
            provenance=Provenance.capture(),
            meta={"note": note, "micro_batches": list(micro_batches),
                  "recommended_rule": "largest micro batch in the swept set"},
        )

    predicted = _alloc(1, False, "closed-form model prediction; deterministic")
    measured = _alloc(replicates, True, "measured on the local device")

    # ---- the model's prediction, computed before anything is measured -----
    local = ModelSpec(**{**spec.__dict__, "hidden": hidden, "n_heads": heads,
                         "ffn_hidden": ffn})
    for cell in predicted.cells():
        c = predicted.coords(cell)
        s, ckpt = int(c["seq_len"]), bool(c["checkpointing"])
        tputs = {mb: _model_tokens_per_s(local, fabric, mb, s, ckpt)
                 for mb in micro_batches}
        rec = tputs[max(micro_batches)]
        predicted.put(RECOMMENDED, cell, [rec])
        predicted.put(BEST, cell, [max(tputs.values())])

    # ---- measurement -------------------------------------------------------
    lock = MeasurementLock("thrust1 calibration")
    lock.__enter__()
    peak = measure_peak_gemm(device=device, dtype=dtype)
    if progress:
        print(f"  peak GEMM {peak:.1f} TFLOP/s", flush=True)
    penalties: list[float] = []
    n_cells = measured.n_cells
    for i_cell, cell in enumerate(measured.cells(), 1):
        c = measured.coords(cell)
        s, ckpt = int(c["seq_len"]), bool(c["checkpointing"])
        if progress:
            print(f"  [{i_cell}/{n_cells}] seq={s} checkpointing={ckpt}", flush=True)
        per_mb: dict[int, list[float]] = {}
        for mb in micro_batches:
            try:
                per_mb[mb] = _measure_block(
                    hidden, ffn, heads, mb, s, ckpt, dt, device, replicates, iters
                )
            except Exception:
                torch.cuda.empty_cache()
                continue
        if not per_mb:
            measured.mark_missing(RECOMMENDED, cell, "no micro batch fitted in memory")
            measured.mark_missing(BEST, cell, "no micro batch fitted in memory")
            continue
        rec_mb = max(k for k in per_mb)
        measured.put(RECOMMENDED, cell, per_mb[rec_mb])
        best_mb = max(per_mb, key=lambda k: float(np.median(per_mb[k])))
        measured.put(BEST, cell, per_mb[best_mb])
        penalties.append(float(np.median(per_mb[rec_mb])))

    lock.__exit__(None, None, None)

    # ---- score the prediction against the measurement -----------------------
    pred_cmps = deterministic_comparisons(predicted, RECOMMENDED, BEST, mde=mde)
    meas_cmps = compare_cells(measured, RECOMMENDED, BEST, mde=mde, q=q_level,
                              seed=seed, paired=True)
    result = calibrate(measured, pred_cmps, meas_cmps, mde=mde)

    out = ThrustOneCalibration(
        predicted=predicted, measured=measured, result=result,
        peak_tflops=peak, assumed_mfu=fabric.achievable_mfu,
        device=device_description(device),
        scope={
            "axes_checked": ["seq_len", "checkpointing", "micro_batch"],
            "axes_not_checked": ["world_size", "shard_group", "tp_degree"],
            "hidden": hidden, "heads": heads, "ffn": ffn, "dtype": dtype,
        },
        uncalibrated=[
            f"Scale. Calibrated over micro batches {list(micro_batches)} and sequence "
            f"lengths {list(seq_lens)} at hidden={hidden}, which is what a development "
            "card can measure in a usable time. The registered Thrust I grid is wider "
            "in every direction, and a slope fitted here is not evidence about the "
            "corners this grid does not reach.",
            "Collective time. Bus bandwidth, the per-collective latency floor and the "
            "bandwidth-against-message-size curve are the model's other half, and none "
            "of them is measurable on a single device. Thrust I's loss regions are "
            "driven by exposed communication, so the parameters that matter most to "
            "its published conclusion remain unchecked by this study.",
            "Whether a real runtime achieves the overlap schedule the model assumes. "
            "That is the question Thrust I exists to ask and it needs a cluster.",
            "Scaling across world size. The registered grid spans 8 and 32 GPUs; this "
            "study touches one.",
            f"Model scale. Calibrated at hidden={hidden}, not the 7B-class "
            f"hidden={spec.hidden} the protocol registers. What transfers is the "
            "residual of a model of this form, not the parameter value.",
        ],
    )
    out.measured_mfu = _measured_mfu(measured, local, peak, micro_batches)
    if math.isfinite(out.measured_mfu) and out.measured_mfu > 1.0:
        # A model FLOP utilisation above one is arithmetically impossible; what
        # it actually reports is that the GEMM used as the ceiling is not a
        # ceiling. On this device the large fp16 GEMM lands on an emulated path
        # that smaller layer GEMMs avoid, so it measures lower than the work it
        # is supposed to bound. Reported rather than clipped: a clipped value
        # would look like a measurement.
        out.uncalibrated.insert(0, (
            f"The MFU reference is invalid on this device. A single large fp16 GEMM "
            f"measured {peak:.2f} TFLOP/s while the layer GEMMs it is supposed to "
            f"bound reached more, giving a nominal utilisation of "
            f"{out.measured_mfu:.2f} -- above one, which is impossible. The cause is "
            "the absent tensor cores: the large fp16 GEMM falls on an emulated path "
            "that the smaller layer shapes avoid. No MFU calibration is available "
            "here, and the figure below is retained only to show the contradiction."
        ))
    out.checkpoint_penalty = _checkpoint_penalty(measured)
    out.update = suggest_update(result, parameter="achievable_mfu",
                                current=fabric.achievable_mfu)
    return out


def _model_tokens_per_s(spec: ModelSpec, fabric: Fabric, mb: int, seq: int,
                        ckpt: bool) -> float:
    """One block's throughput as the Thrust I model computes it."""
    h, hf = spec.hidden, spec.ffn_hidden
    gemm_fwd = 2 * mb * seq * (4 * h * h + 3 * h * hf)
    attn_fwd = 4 * mb * spec.n_heads * seq * seq * (h // spec.n_heads)
    flops = 3.0 * (gemm_fwd + attn_fwd)
    if ckpt:
        flops *= 4.0 / 3.0
    ms = flops / (fabric.peak_tflops * 1e12 * fabric.achievable_mfu) * 1e3
    ms += fabric.kernel_launch_us * 1e-3 * 12
    return (mb * seq) / (ms * 1e-3) if ms > 0 else float("nan")


def _measure_block(hidden, ffn, heads, mb, seq, ckpt, dt, device, replicates, iters):
    """Tokens per second through one block, forward and backward."""
    import torch
    from torch.utils.checkpoint import checkpoint

    block = _block(hidden, ffn, heads, dt, device)
    x = torch.randn(mb, seq, hidden, device=device, dtype=dt, requires_grad=True)

    def step() -> None:
        out = checkpoint(block, x, use_reentrant=False) if ckpt else block(x)
        out.sum().backward()
        block.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    times, _ = paired_ab(step, step, replicates=replicates, iters=iters, device=device)
    out = [(mb * seq) / (t * 1e-3) for t in times if t > 0]
    torch.cuda.empty_cache()
    return out


def _measured_mfu(env: Envelope, spec: ModelSpec, peak: float,
                  micro_batches: Sequence[int]) -> float:
    if not math.isfinite(peak) or peak <= 0:
        return float("nan")
    best: list[float] = []
    for cell in env.cells():
        if not env.is_measured(BEST, cell):
            continue
        c = env.coords(cell)
        if bool(c["checkpointing"]):
            continue
        seq, mb = int(c["seq_len"]), max(micro_batches)
        h, hf = spec.hidden, spec.ffn_hidden
        flops = 3.0 * (2 * mb * seq * (4 * h * h + 3 * h * hf)
                       + 4 * mb * spec.n_heads * seq * seq * (h // spec.n_heads))
        tps = float(np.median(env.replicates_at(BEST, cell)))
        secs = (mb * seq) / tps if tps > 0 else float("nan")
        if math.isfinite(secs) and secs > 0:
            best.append(flops / secs / 1e12)
    return float(np.median(best)) / peak if best else float("nan")


def _checkpoint_penalty(env: Envelope) -> dict[str, Any]:
    """The measured cost of activation checkpointing, against the model's 4/3."""
    on, off = {}, {}
    for cell in env.cells():
        if not env.is_measured(BEST, cell):
            continue
        c = env.coords(cell)
        key = int(c["seq_len"])
        (on if bool(c["checkpointing"]) else off)[key] = float(
            np.median(env.replicates_at(BEST, cell))
        )
    ratios = [off[k] / on[k] for k in sorted(set(on) & set(off)) if on[k] > 0]
    if not ratios:
        return {}
    return {
        "assumed": 4.0 / 3.0,
        "measured_median": float(np.median(ratios)),
        "measured_min": float(np.min(ratios)),
        "measured_max": float(np.max(ratios)),
        "ratio": float(np.median(ratios)) / (4.0 / 3.0),
        "varies": bool(np.max(ratios) - np.min(ratios) > 0.15),
        "n": len(ratios),
        "per_seq_len": {str(k): off[k] / on[k] for k in sorted(set(on) & set(off))},
    }
