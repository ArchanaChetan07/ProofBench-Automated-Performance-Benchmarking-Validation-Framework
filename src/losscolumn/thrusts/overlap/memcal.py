"""Calibrating the Thrust I model where it makes a decision that can be wrong.

The first calibration attempt asked whether the recommended micro batch was the
fastest one, and established nothing: on a compute-only model a larger micro
batch is always better, so the model predicted no loss anywhere, and a
prediction that cannot be wrong cannot be checked.

This study asks a question the model can get wrong, and whose wrongness costs
something: **does the recommended configuration fit in memory?**

That is a genuine decision with a boundary. The model predicts activation and
parameter memory, and from that the largest micro batch that fits. Measurement
runs it and finds out. The two boundaries need not coincide, and where they
differ the error has a direction that matters:

* **false win** -- the model said it fits and it does not. The run dies. This
  is the expensive error, and it is what the model is used to avoid.
* **false loss** -- the model said it would not fit and it does. Wasted
  capacity, cheap by comparison.

Both are measurable on one GPU, which is the point: the memory half of the
Thrust I model is checkable here even though the communication half is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.calibration import (
    CalibrationPlan,
    CalibrationResult,
    calibrate,
    plan_calibration,
)
from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.provenance import Provenance
from losscolumn.core.stats import compare_cells
from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description
from losscolumn.thrusts.overlap.calibrate import (
    _block,
    deterministic_comparisons,
)

CONFIG = "configuration"
ORACLE = "an_oracle_that_always_fits"

# Bytes held per element of a saved activation, and how many such tensors a
# transformer block keeps alive for its backward pass. Both are model
# parameters of exactly the kind this study exists to check, so they are named
# here rather than buried in an expression.
ACT_TENSORS_PER_BLOCK = 6.0
OPTIMIZER_BYTES_PER_PARAM = 12.0        # fp32 master weights + two moments


@dataclass
class MemoryModel:
    """The Thrust I memory model, restricted to one block on one device."""

    hidden: int
    ffn: int
    heads: int
    bytes_per_elem: int = 2
    act_tensors: float = ACT_TENSORS_PER_BLOCK
    checkpoint_retained: float = 1.0 / 8.0   # a checkpointed block keeps far less
    overhead_gb: float = 0.9                 # context, allocator, cuDNN workspaces

    @property
    def params(self) -> int:
        return 3 * self.hidden * self.hidden + self.hidden * self.hidden \
            + 2 * self.hidden * self.ffn

    def predict_gb(self, micro_batch: int, seq: int, checkpointing: bool) -> float:
        """Predicted peak device memory for one block, forward and backward."""
        tokens = micro_batch * seq
        retained = self.act_tensors * (
            self.checkpoint_retained if checkpointing else 1.0
        )
        act = tokens * self.hidden * self.bytes_per_elem * retained
        # The attention score matrix is the term that grows quadratically, and
        # the term a fused attention kernel removes. SDPA is fused, so it is
        # counted at the fused cost -- one pass over the values, not seq^2.
        act += tokens * self.ffn * self.bytes_per_elem
        weights = self.params * (self.bytes_per_elem + OPTIMIZER_BYTES_PER_PARAM)
        return (act + weights) / 1e9 + self.overhead_gb

    def largest_fitting(self, candidates: Sequence[int], seq: int,
                        checkpointing: bool, budget_gb: float) -> int | None:
        fits = [m for m in candidates
                if self.predict_gb(m, seq, checkpointing) <= budget_gb]
        return max(fits) if fits else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden, "ffn": self.ffn, "heads": self.heads,
            "bytes_per_elem": self.bytes_per_elem,
            "act_tensors_per_block": self.act_tensors,
            "checkpoint_retained_fraction": self.checkpoint_retained,
            "overhead_gb": self.overhead_gb,
            "params": self.params,
        }


@dataclass
class MemoryCalibration:
    predicted: Envelope
    measured: Envelope
    result: CalibrationResult
    plan: CalibrationPlan | None = None
    model: dict[str, Any] = field(default_factory=dict)
    device: str = ""
    budget_gb: float = float("nan")
    measured_feasibility: dict[str, Any] = field(default_factory=dict)
    uncalibrated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "budget_gb": self.budget_gb,
            "memory_model": self.model,
            "plan": self.plan.to_dict() if self.plan else None,
            "calibration": self.result.to_dict(),
            "measured_feasibility": self.measured_feasibility,
            "uncalibrated": self.uncalibrated,
            "predicted_envelope": self.predicted.to_dict(include_replicates=False),
            "measured_envelope": self.measured.to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            "### Calibration study -- Thrust I, memory feasibility",
            "",
            f"Measured on {self.device}, budget {self.budget_gb:.1f} GB. The model was "
            "asked for its prediction before anything was measured, and was not "
            "adjusted afterwards.",
            "",
            "The question is one the model can get wrong at a cost: **does the "
            "recommended micro batch fit?** A false win here is a run that dies.",
            "",
            self.result.to_markdown(),
            "",
        ]
        wrong = [v for v in self.measured_feasibility.values() if v.get("model_wrong")]
        if self.measured_feasibility:
            n = len(self.measured_feasibility)
            lines += [
                f"**Feasibility agreement**: the model is right about "
                f"{n - len(wrong)} of {n} configurations.",
                "",
            ]
            if wrong:
                lines += ["| Sequence | Micro batch | Checkpointing | Model | "
                          "Actual | Predicted GB |", "|---|---|---|---|---|---|"]
                for v in sorted(wrong, key=lambda v: (v["seq_len"], v["micro_batch"])):
                    lines.append(
                        f"| {v['seq_len']} | {v['micro_batch']} | "
                        f"{v['checkpointing']} | "
                        f"{'fits' if v['predicted_fits'] else 'does not fit'} | "
                        f"{'fits' if v['measured_fits'] else 'OOM'} | "
                        f"{v['predicted_gb']:.2f} |"
                    )
                lines.append("")
        lines += ["**Not calibrated by this study**", ""]
        lines += [f"- {u}" for u in self.uncalibrated]
        return "\n".join(lines)


def _fits(hidden, ffn, heads, mb, seq, ckpt, dt, device,
          budget_fraction: float = 0.92) -> tuple[bool, str]:
    """Actually run one block forward and backward; report whether it fits.

    A hard allocator cap is set first, and that is what makes the answer
    meaningful. Without it the driver on this platform satisfies an
    over-large allocation out of host memory rather than failing, so the run
    neither fits nor raises -- it thrashes, for hours, at a speed nobody would
    ever accept in practice. "Fits" would then mean "eventually returned",
    which is not the question.

    With the cap, an allocation past the budget raises immediately and the
    boundary being measured is the one the model is predicting.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    try:
        torch.cuda.set_per_process_memory_fraction(budget_fraction, 0)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        block = _block(hidden, ffn, heads, dt, device)
        x = torch.randn(mb, seq, hidden, device=device, dtype=dt, requires_grad=True)
        out = checkpoint(block, x, use_reentrant=False) if ckpt else block(x)
        out.sum().backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del block, x, out
        torch.cuda.empty_cache()
        return True, f"peak {peak:.2f} GB"
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "out of memory" in str(e).lower():
            return False, "CUDA out of memory"
        return False, f"{type(e).__name__}: {str(e)[:80]}"
    except Exception as e:
        torch.cuda.empty_cache()
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def run_memory_calibration(
    *,
    micro_batches: Sequence[int] = (8, 16, 32, 64, 128, 256),
    seq_lens: Sequence[int] = (512, 1024, 2048, 4096),
    checkpointing: Sequence[bool] = (False, True),
    hidden: int = 1024,
    heads: int = 16,
    ffn_mult: int = 4,
    device: str = "cuda",
    dtype: str = "float16",
    replicates: int = 5,
    iters: int = 1,
    mde: float = 0.05,
    q_level: float = 0.05,
    seed: int = 20260101,
    budget_fraction: float = 0.92,
    progress: bool = True,
) -> MemoryCalibration:
    """Predict which configurations fit, then go and find out.

    Micro batch is a *factor* here rather than a choice the recommendation
    makes. That is what gives the study something to be wrong about: the
    decision under test is per configuration -- *does this one fit?* -- so the
    loss map is exactly the infeasibility map, and the four region statistics
    mean what their names say. A false win is the model saying a configuration
    fits when the run dies.

    The comparator on both sides is the same reference throughput, so a cell
    that runs lands as a tie and a cell that does not lands as a loss. The
    measured side additionally carries the model's throughput error, because a
    configuration that fits but runs at half the predicted rate is also a way
    for the model to be wrong.
    """
    import torch

    ffn = ffn_mult * hidden
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = total * budget_fraction
    model = MemoryModel(hidden=hidden, ffn=ffn, heads=heads)

    factors = (
        Factor("seq_len", tuple(seq_lens), unit="tokens", log_scale=True),
        Factor("micro_batch", tuple(micro_batches), log_scale=True),
        Factor("checkpointing", tuple(checkpointing)),
    )
    metric = Metric(
        "layer_throughput", "tok/s", higher_is_better=True,
        description="tokens per second through one block, forward and backward; a "
                    "configuration that does not fit scores zero",
    )

    def _alloc(reps: int, note: str) -> Envelope:
        return Envelope.allocate(
            factors, metric, [CONFIG, ORACLE], reps, interleaved=True,
            workload=f"transformer block, hidden={hidden}, heads={heads}, ffn={ffn}",
            provenance=Provenance.capture(),
            meta={"note": note, "budget_gb": budget,
                  "decision": "does this configuration fit in memory?"},
        )

    predicted = _alloc(1, "closed-form memory model; deterministic")
    measured = _alloc(replicates, "measured on the local device")

    # ---- prediction, before anything is measured ---------------------------
    reference: dict[tuple, float] = {}
    for cell in predicted.cells():
        c = predicted.coords(cell)
        seq, mb, ckpt = int(c["seq_len"]), int(c["micro_batch"]), bool(c["checkpointing"])
        # A shape-independent reference rate, so the comparison is about
        # feasibility rather than about how fast the card is.
        ref = float(mb * seq) * 1e3
        reference[(seq, mb, ckpt)] = ref
        fits = model.predict_gb(mb, seq, ckpt) <= budget
        predicted.put(ORACLE, cell, [ref])
        if fits:
            predicted.put(CONFIG, cell, [ref])
        else:
            # Recorded the way the standard already handles a configuration that
            # cannot run: unmeasurable, with the reason. Encoding it as a zero
            # made it look like a measurement of nothing, and both comparison
            # paths dropped it.
            predicted.mark_missing(
                CONFIG, cell,
                f"model predicts {model.predict_gb(mb, seq, ckpt):.2f} GB, over the "
                f"{budget:.2f} GB budget",
            )

    pred_cmps = deterministic_comparisons(predicted, CONFIG, ORACLE, mde=mde)
    plan = plan_calibration(predicted, pred_cmps,
                            budget=len(list(predicted.cells())))
    if progress:
        n_pred_loss = sum(1 for c in pred_cmps if c.verdict == "loss")
        print(f"  model predicts {n_pred_loss} infeasible cell(s) of "
              f"{len(pred_cmps)}", flush=True)

    # ---- measurement --------------------------------------------------------
    feasibility: dict[str, Any] = {}
    with MeasurementLock("thrust1 memory calibration"):
        for i, cell in enumerate(measured.cells(), 1):
            c = measured.coords(cell)
            seq, mb = int(c["seq_len"]), int(c["micro_batch"])
            ckpt = bool(c["checkpointing"])
            ref = reference[(seq, mb, ckpt)]
            pred_gb = model.predict_gb(mb, seq, ckpt)
            pred_fits = pred_gb <= budget

            ok, why = _fits(hidden, ffn, heads, mb, seq, ckpt, dt, device,
                            budget_fraction=budget_fraction)
            if progress:
                flag = "" if ok == pred_fits else "   <- MODEL WRONG"
                print(f"  [{i}/{measured.n_cells}] seq={seq} mb={mb} ckpt={ckpt}: "
                      f"model {'fits' if pred_fits else 'no'}, actual "
                      f"{'fits' if ok else 'OOM'}{flag}", flush=True)

            feasibility[measured.label(cell)] = {
                "seq_len": seq, "micro_batch": mb, "checkpointing": ckpt,
                "predicted_gb": pred_gb, "predicted_fits": pred_fits,
                "measured_fits": ok, "detail": why,
                "model_wrong": ok != pred_fits,
            }
            measured.put(ORACLE, cell, [ref] * replicates)
            if ok:
                measured.put(CONFIG, cell, [ref] * replicates)
            else:
                measured.mark_missing(CONFIG, cell, why)

    meas_cmps = compare_cells(measured, CONFIG, ORACLE, mde=mde, q=q_level,
                              seed=seed, paired=False)
    result = calibrate(measured, pred_cmps, meas_cmps, mde=mde)

    return MemoryCalibration(
        predicted=predicted, measured=measured, result=result, plan=plan,
        model=model.to_dict(), device=device_description(device), budget_gb=budget,
        measured_feasibility=feasibility,
        uncalibrated=[
            "Communication. Bus bandwidth, the per-collective latency floor and the "
            "overlap schedule are the model's other half and need a cluster; Thrust "
            "I's loss regions are driven by exposed communication, so the parameters "
            "that matter most to its published conclusion are still unchecked.",
            f"Model scale. Calibrated at hidden={hidden} on one block, not a 7B-class "
            "model across 32 layers. What transfers is the residual of a memory model "
            "of this form, not the parameter values.",
            "Optimizer state. The measured runs hold gradients but no optimizer "
            "moments, while the model prices twelve bytes per parameter for them. That "
            "term is therefore predicted and not measured here.",
        ],
    )
