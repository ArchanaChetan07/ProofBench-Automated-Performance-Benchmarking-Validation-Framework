"""The memory-model calibration and validation study, version 2.

The study that replaced the Thrust I memory model, and then graded the
replacement on cells it had never seen.

Version 1 was diagnosed on an eight-cell grid: region overlap 0.17, boundary
one level late, five false wins, zero false losses. That diagnosis grid is the
**calibration** grid here and can establish nothing about version 2, because
version 2 was designed with those failures in view. The verdict on version 2
comes from a **validation** grid that was selected before measurement, sealed,
and left untouched until the model was frozen.

The order is enforced rather than intended:

    seal the protocol
      -> preflight both grids from the model alone
      -> measure the calibration grid, fit whatever is fittable
      -> FREEZE
      -> measure the validation grid
      -> report

:class:`losscolumn.core.gridsplit.GridSplit` raises if a validation cell is
read before the freeze, if a calibration cell is offered as validation
evidence, or if a parameter is fitted after the freeze.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.feasibility import (
    DEFAULT_DANGER_WEIGHT,
    FeasibilityCell,
    FeasibilityMatrix,
    FeasibilityObservation,
)
from losscolumn.core.gridsplit import GridSplit, PreflightReport, preflight, select_boundary_grid
from losscolumn.core.memory_model import GB, BlockMemoryModel, BlockShape, ScalingMap
from losscolumn.core.provenance import Provenance as ProvCapture
from losscolumn.core.provenance import content_hash, utcnow
from losscolumn.thrusts.overlap.probe import (
    ProbeConfig,
    install_budget,
    probe_feasibility,
    release_budget,
)

STUDY_VERSION = "thrust1-memory-v2"

# The local block the study measures. Fixed here, registered in the protocol,
# and mapped explicitly to the simulated architecture by ScalingMap.
LOCAL_HIDDEN = 1024
LOCAL_HEADS = 16
LOCAL_FFN = 4096


def _shape(mb: int, seq: int, ckpt: bool, *, bytes_per_elem: int = 2) -> BlockShape:
    return BlockShape(
        hidden=LOCAL_HIDDEN, heads=LOCAL_HEADS, ffn=LOCAL_FFN,
        micro_batch=mb, seq_len=seq, n_layers=1,
        bytes_per_elem=bytes_per_elem, checkpointing=ckpt,
    )


# The eight cells version 1 was diagnosed on. Fixed as the calibration grid so
# it cannot be reused as evidence about version 2.
DIAGNOSIS_GRID: tuple[tuple[int, int, bool], ...] = tuple(
    (mb, seq, ck)
    for seq in (2048, 4096)
    for mb in (64, 128)
    for ck in (False, True)
)


def candidate_cells(
    micro_batches: Sequence[int] = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256),
    seq_lens: Sequence[int] = (512, 1024, 1536, 2048, 3072, 4096),
    checkpointing: Sequence[bool] = (False, True),
) -> list[tuple[int, int, bool]]:
    return [(mb, s, c) for s in seq_lens for mb in micro_batches for c in checkpointing]


@dataclass
class MemoryStudy:
    """The whole study: protocol, preflight, calibration, freeze, validation."""

    model: BlockMemoryModel
    split: GridSplit
    scaling: ScalingMap
    probe_config: ProbeConfig
    budget_bytes: int
    danger_weight: float = DEFAULT_DANGER_WEIGHT
    preflight_calibration: PreflightReport | None = None
    preflight_validation: PreflightReport | None = None
    calibration: FeasibilityMatrix | None = None
    validation: FeasibilityMatrix | None = None
    fitted: list[str] = field(default_factory=list)
    device: str = ""
    sealed_at: str = field(default_factory=utcnow)
    notes: list[str] = field(default_factory=list)

    def protocol(self) -> dict[str, Any]:
        """Everything registered before execution."""
        doc = {
            "kind": "memory-model-protocol",
            "study": STUDY_VERSION,
            "sealed_at": self.sealed_at,
            "memory_model_version": self.model.version,
            "model_parameters": self.model.params.to_dict(),
            "precision_bytes_per_element": 2,
            "device_memory_budget_bytes": self.budget_bytes,
            "device_memory_budget_gb": self.budget_bytes / GB,
            "allocator_mechanism": self.probe_config.to_dict(),
            "checkpointing_policy": (
                "activation checkpointing affects the saved-activation term only. "
                "Weights, gradients, optimizer state and the live per-block workspace "
                "are untouched, because a recomputed block still materialises its "
                "full workspace. Applying one multiplier to the whole expression is "
                "the structural error that made version 1 unusable."
            ),
            "grid_split": self.split.to_dict(),
            "scaling": self.scaling.to_dict(),
            "metric_definitions": {
                "false_win": "model predicts feasible, measurement says infeasible",
                "false_loss": "model predicts infeasible, measurement says feasible",
                "infeasible_iou": "intersection over union of the predicted and "
                                  "measured infeasible cell sets; undefined, not 1.0, "
                                  "when both are empty",
                "false_win_rate": "false wins as a share of the cells the model cleared",
                "false_loss_rate": "false losses as a share of the cells it rejected",
                "dangerous_error_score": f"({self.danger_weight:g} * false_wins + "
                                         "false_losses) / comparable cells",
                "comparable": "both sides gave a definite FEASIBLE or INFEASIBLE; "
                              "HOST_FALLBACK, INVALID and NOT_MEASURED are excluded "
                              "rather than rounded",
            },
            "false_win_severity": {
                "weight": self.danger_weight,
                "policy": (
                    "A false win kills a job launched on the model's advice; a false "
                    "loss leaves capacity unused. The weight is a policy choice, fixed "
                    "here before the data, and the raw counts are always reported "
                    "beside the weighted score."
                ),
            },
            "acceptance_criteria": {
                "primary": "zero false wins on the validation grid",
                "secondary": "false-loss rate below 0.5, so the model is conservative "
                             "without being useless",
                "stopping": "the validation grid is fixed and measured once; there is "
                            "no interim analysis and no cell is added after the seal",
            },
            "safety_margin_policy": (
                "The safety margin is registered at "
                f"{self.model.params.get('safety_margin_fraction'):.0%} and is NOT "
                "tuned against validation results. Tuning it there would be fitting "
                "on the data the model is graded by."
            ),
        }
        doc["seal_hash"] = content_hash(doc)
        return doc

    # ---- reporting -------------------------------------------------------

    def comparison_table(self, old: dict[str, Any] | None = None) -> str:
        """Version 1 against version 2, on the validation grid only."""
        if self.validation is None:
            return "_No validation result yet._"
        v = self.validation
        old = old or {
            "iou": 0.17, "boundary": "1 level late", "false_wins": "5/8",
            "false_losses": "0/8", "dangerous": (10 * 5 + 0) / 8,
        }
        return "\n".join([
            "| Metric | v1 (diagnosis grid) | v2 (validation grid) |",
            "|---|---|---|",
            f"| Infeasible-region IoU | {old['iou']} | {v.infeasible_iou:.2f} |",
            f"| Boundary error | {old['boundary']} | {_worst_shift(v)} |",
            f"| False wins | {old['false_wins']} | "
            f"**{v.n_false_win}/{v.n_comparable}** |",
            f"| False losses | {old['false_losses']} | "
            f"{v.n_false_loss}/{v.n_comparable} |",
            f"| Dangerous-error score | {old['dangerous']:.2f} | "
            f"{v.dangerous_error_score:.2f} |",
            f"| Accuracy | 38% | {v.accuracy:.0%} |",
            "",
            "The two columns are **not** measured on the same cells, and cannot be: "
            "the v1 numbers come from the grid that diagnosed it, which is this "
            "study's calibration grid and therefore inadmissible as evidence about "
            "v2. The v2 column is the untouched validation grid. What the comparison "
            "supports is a statement about each model on the hardest grid available "
            "to it, not a paired contest.",
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": STUDY_VERSION,
            "device": self.device,
            "protocol": self.protocol(),
            "preflight": {
                "calibration": self.preflight_calibration.to_dict()
                if self.preflight_calibration else None,
                "validation": self.preflight_validation.to_dict()
                if self.preflight_validation else None,
            },
            "calibration_result": self.calibration.to_dict() if self.calibration else None,
            "validation_result": self.validation.to_dict() if self.validation else None,
            "fitted_parameters": self.fitted,
            "model_after_freeze": self.model.to_dict(),
            "notes": self.notes,
            "provenance": ProvCapture.capture().to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            "### Thrust I memory model, version 2",
            "",
            f"Measured on {self.device}, budget "
            f"{self.budget_bytes / GB:.2f} GB, enforced by "
            f"`{self.probe_config.to_dict()['cap_mechanism']}`.",
            "",
            self.split.to_markdown(),
            "",
        ]
        if self.preflight_calibration:
            lines += [self.preflight_calibration.to_markdown(), ""]
        if self.preflight_validation:
            lines += [self.preflight_validation.to_markdown(), ""]
        if self.calibration:
            lines += ["#### Calibration grid (used to fit; not evidence)", "",
                      self.calibration.to_markdown(), ""]
        if self.validation:
            lines += ["#### Validation grid (untouched until the model was frozen)", "",
                      self.validation.to_markdown(), ""]
            lines += ["#### Old model against new", "", self.comparison_table(), ""]
        lines += ["#### Model parameters", "", self.model.params.to_markdown(), ""]
        lines += ["#### Scaling to the simulated architecture", "",
                  self.scaling.to_markdown(), ""]
        if self.notes:
            lines += ["#### Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _worst_shift(v: FeasibilityMatrix) -> str:
    """The largest boundary displacement over any axis, for the summary row."""
    import math

    bd = [b for b in v.boundary_report(["micro_batch", "seq_len", "checkpointing"])
          if b["n_fibers_compared"]]
    if not bd:
        return "no fiber has a boundary in both maps"
    worst = max(bd, key=lambda b: abs(b["mean_signed_shift"]))
    if math.isclose(worst["mean_signed_shift"], 0.0, abs_tol=1e-9):
        return f"**0 levels** on all {len(bd)} axes"
    return f"{worst['mean_signed_shift']:+.2f} level(s) on `{worst['axis']}`"


def _measure_cell(cell: tuple[int, int, bool], cfg: ProbeConfig,
                  dtype: Any) -> FeasibilityObservation:
    """Run one configuration under the enforced budget."""
    import torch

    from losscolumn.thrusts.overlap.calibrate import _block

    mb, seq, ckpt = cell

    def run() -> None:
        from torch.utils.checkpoint import checkpoint

        block = _block(LOCAL_HIDDEN, LOCAL_FFN, LOCAL_HEADS, dtype, "cuda")
        x = torch.randn(mb, seq, LOCAL_HIDDEN, device="cuda", dtype=dtype,
                        requires_grad=True)
        out = checkpoint(block, x, use_reentrant=False) if ckpt else block(x)
        out.sum().backward()
        del block, x, out

    return probe_feasibility(run, cfg)


def build_study(
    *,
    budget_fraction: float = 0.92,
    danger_weight: float = DEFAULT_DANGER_WEIGHT,
    n_boundary: int = 6,
    n_anchor: int = 2,
) -> MemoryStudy:
    """Seal the protocol: model, grids, budget and metrics, before measurement."""
    cfg = install_budget(budget_fraction)
    budget = cfg.budget_bytes
    model = BlockMemoryModel()

    calibration = list(DIAGNOSIS_GRID)
    validation, rationale = select_boundary_grid(
        candidate_cells(),
        boundary_distance=lambda c: model.boundary_distance(_shape(*c), budget),
        predict_fits=lambda c: model.fits(_shape(*c), budget),
        n_boundary=n_boundary,
        n_clear_feasible=n_anchor,
        n_clear_infeasible=n_anchor,
        exclude=calibration,
    )

    split = GridSplit(
        calibration=tuple(calibration),
        validation=tuple(validation),
        rationale={**{c: "the grid that diagnosed version 1" for c in calibration},
                   **rationale},
    )
    scaling = ScalingMap(
        local=_shape(1, 1, False),
        target_hidden=4096, target_heads=32, target_ffn=11008, target_layers=32,
    )
    study = MemoryStudy(
        model=model, split=split, scaling=scaling, probe_config=cfg,
        budget_bytes=budget, danger_weight=danger_weight,
    )

    study.preflight_calibration = preflight(
        calibration,
        predict_fits=lambda c: model.fits(_shape(*c), budget),
        boundary_distance=lambda c: model.boundary_distance(_shape(*c), budget),
        label="calibration grid",
        describe=lambda c: {"predicted_gb": model.predict_gb(_shape(*c))},
    )
    study.preflight_validation = preflight(
        validation,
        predict_fits=lambda c: model.fits(_shape(*c), budget),
        boundary_distance=lambda c: model.boundary_distance(_shape(*c), budget),
        label="validation grid",
        describe=lambda c: {"predicted_gb": model.predict_gb(_shape(*c))},
    )
    return study


def run_grid(study: MemoryStudy, cells: Sequence[tuple[int, int, bool]], *,
             purpose: str, dtype_name: str = "float16",
             progress: bool = True) -> FeasibilityMatrix:
    """Measure a grid, refusing any cell the split does not permit for this purpose."""
    import torch

    from losscolumn.thrusts.kernel.bench import MeasurementLock

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    out: list[FeasibilityCell] = []
    with MeasurementLock(f"thrust1 memory {purpose}"):
        for i, cell in enumerate(cells, 1):
            study.split.check_readable(cell, purpose=purpose)
            mb, seq, ckpt = cell
            shape = _shape(mb, seq, ckpt)
            predicted = FeasibilityObservation.predicted(
                study.model.fits(shape, study.budget_bytes),
                detail=study.model.explain(shape, study.budget_bytes),
                predicted_bytes=int(study.model.predict_bytes(shape)),
                budget_bytes=study.budget_bytes,
            )
            measured = _measure_cell(cell, study.probe_config, dtype)
            if progress:
                mark = ""
                if predicted.status.is_definite and measured.status.is_definite:
                    if predicted.is_feasible and not measured.is_feasible:
                        mark = "   <- FALSE WIN"
                    elif not predicted.is_feasible and measured.is_feasible:
                        mark = "   <- false loss"
                print(
                    f"  [{purpose} {i}/{len(cells)}] mb={mb} seq={seq} ckpt={ckpt}: "
                    f"model {predicted.status.value}, actual {measured.status.value}"
                    f"{mark}",
                    flush=True,
                )
            out.append(FeasibilityCell(
                key=cell, predicted=predicted, measured=measured,
                coords={"micro_batch": mb, "seq_len": seq, "checkpointing": ckpt},
            ))
    return FeasibilityMatrix(cells=out, danger_weight=study.danger_weight)


def run_study(*, budget_fraction: float = 0.92, progress: bool = True) -> MemoryStudy:
    """Seal, preflight, calibrate, freeze, validate -- in that order."""
    from losscolumn.thrusts.kernel.bench import device_description

    study = build_study(budget_fraction=budget_fraction)
    study.device = device_description("cuda")

    if progress:
        print(study.preflight_calibration.to_markdown(), flush=True)
        print(flush=True)
        print(study.preflight_validation.to_markdown(), flush=True)
        print(flush=True)

    try:
        study.calibration = run_grid(
            study, study.split.calibration, purpose="calibration", progress=progress
        )
        # Nothing is fitted here. Every version 2 parameter is REGISTERED from
        # the arithmetic of what a block allocates, and the calibration grid is
        # used to confirm the replacement addresses the diagnosed failures --
        # not to choose a value. Recorded explicitly so the absence of fitting
        # is a stated fact rather than an omission.
        study.notes.append(
            "No parameter was fitted. Version 2's parameters are all REGISTERED, "
            "derived from what a block allocates rather than from these "
            "measurements, so the calibration grid confirms the replacement "
            "without conferring any fit on it. That makes the validation result "
            "a stronger statement, not a weaker one: the model had no opportunity "
            "to absorb the calibration data."
        )
        study.split.freeze()
        study.validation = run_grid(
            study, study.split.validation, purpose="validation", progress=progress
        )
    finally:
        release_budget()
    return study
