"""Expanded validation of block-memory-v2. Validation only; nothing is fitted.

The milestone recorded four things the ten-cell study did not establish. Three
of them are reachable on this hardware and this study reaches them:

* **more sequence lengths** -- the original grid used four; this uses nine,
  including sizes that are not powers of two, because a model that only works
  on round numbers is fitting the grid.
* **another memory capacity** -- without another card. The allocator cap is a
  budget, so lowering it makes the device behave as a smaller one. Three
  budgets are swept, which tests the thing the milestone flagged: the model was
  only ever checked against a single 7.9 GB ceiling.
* **more near-boundary cells** -- selected by the model's own boundary
  distance, so measurement lands where being slightly wrong changes the answer.
* **a further configuration axis** -- with a single block, checkpointing has
  exactly two states and there is no third to add. The axis that *is* untested
  is precision: the model predicts memory proportional to bytes per element and
  that term has never been checked. fp32 doubles every tensor term and leaves
  context alone, which is a sharp prediction.

**block-memory-v2 is frozen.** Its parameters are asserted by test and are not
touched here whatever this study finds. A result that disagreed would be a
finding about the model's range of validity, to be recorded and carried; it
would not be licence to retune, because a model retuned on the data that found
its limits has been fitted by them.
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
from losscolumn.core.gridsplit import preflight, select_boundary_grid
from losscolumn.core.memory_model import GB, BlockMemoryModel, BlockShape
from losscolumn.core.provenance import Provenance as ProvCapture
from losscolumn.core.provenance import content_hash, utcnow
from losscolumn.thrusts.overlap.memcal2 import (
    DIAGNOSIS_GRID,
    LOCAL_FFN,
    LOCAL_HEADS,
    LOCAL_HIDDEN,
)
from losscolumn.thrusts.overlap.probe import install_budget, probe_feasibility, release_budget

STUDY_VERSION = "thrust1-memory-v2-expanded-validation"

# Nine sequence lengths, four of them not powers of two.
SEQ_LENS = (512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
MICRO_BATCHES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
# Three effective capacities on one card. The cap is a real budget, so a lower
# fraction is a smaller device as far as the allocator is concerned.
BUDGET_FRACTIONS = (0.35, 0.60, 0.92)
DTYPES = ("float16", "float32")

# Cell key: (micro_batch, seq_len, checkpointing, dtype, budget_fraction).
# A five-part key, so it cannot collide with the three-part keys of the earlier
# studies even where the shape coincides -- the budget and precision are part
# of the question being asked.
Cell = tuple[int, int, bool, str, float]


def _shape(cell: Cell) -> BlockShape:
    mb, seq, ckpt, dtype, _ = cell
    return BlockShape(
        hidden=LOCAL_HIDDEN, heads=LOCAL_HEADS, ffn=LOCAL_FFN,
        micro_batch=mb, seq_len=seq, n_layers=1,
        bytes_per_elem=2 if dtype == "float16" else 4,
        checkpointing=ckpt,
    )


def _prior_shapes() -> set[tuple[int, int, bool]]:
    """Every (mb, seq, ckpt) the earlier studies already measured."""
    from losscolumn.thrusts.overlap.memcal2 import build_study

    prior = set(DIAGNOSIS_GRID)
    try:
        prior |= set(build_study().split.validation)
    except Exception:
        pass
    return {tuple(c) for c in prior}


@dataclass
class ExpandedValidation:
    model: BlockMemoryModel
    matrix: FeasibilityMatrix | None = None
    per_budget: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_dtype: dict[str, dict[str, Any]] = field(default_factory=dict)
    cells: list[Cell] = field(default_factory=list)
    preflights: dict[str, Any] = field(default_factory=dict)
    device: str = ""
    total_bytes: int = 0
    sealed_at: str = field(default_factory=utcnow)
    notes: list[str] = field(default_factory=list)

    def protocol(self) -> dict[str, Any]:
        doc = {
            "kind": "memory-validation-protocol",
            "study": STUDY_VERSION,
            "sealed_at": self.sealed_at,
            "validates": "block-memory-v2",
            "model_frozen": True,
            "fitting_permitted": False,
            "fitting_policy": (
                "Nothing is fitted. block-memory-v2 is frozen and its parameters are "
                "asserted by test. A disagreement found here is a finding about the "
                "model's range of validity, recorded and carried; it is not licence "
                "to retune, because a model retuned on the data that found its limits "
                "has been fitted by them."
            ),
            "axes": {
                "sequence_lengths": list(SEQ_LENS),
                "micro_batches": list(MICRO_BATCHES),
                "checkpointing": [False, True],
                "precision": list(DTYPES),
                "budget_fractions": list(BUDGET_FRACTIONS),
            },
            "capacity_method": (
                "torch.cuda.set_per_process_memory_fraction. The cap is a real "
                "allocator budget, so a lower fraction makes the device behave as a "
                "smaller one. This reaches the 'another memory capacity' axis without "
                "a second card, and does not reach 'another GPU' -- allocator "
                "behaviour, context cost and fragmentation are properties of this "
                "device and driver."
            ),
            "disjointness": (
                "every cell carries its precision and budget in the key, so no cell "
                "here repeats a question the calibration or first validation grid "
                "already answered; shapes that coincide at fp16 and 0.92 are excluded "
                "explicitly as well"
            ),
            "metrics": ["false_win", "false_loss", "infeasible_iou",
                        "boundary_displacement", "dangerous_error_score"],
            "acceptance": "zero false wins, as at v0.3.0-memory-v2",
        }
        doc["seal_hash"] = content_hash(doc)
        return doc

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": STUDY_VERSION,
            "device": self.device,
            "total_bytes": self.total_bytes,
            "protocol": self.protocol(),
            "preflight": self.preflights,
            "result": self.matrix.to_dict() if self.matrix else None,
            "per_budget": self.per_budget,
            "per_dtype": self.per_dtype,
            "model": self.model.to_dict(),
            "notes": self.notes,
            "provenance": ProvCapture.capture().to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            "### block-memory-v2, expanded validation",
            "",
            f"Measured on {self.device}. **The model is frozen**: nothing here is "
            "fitted, and a disagreement would be recorded rather than corrected.",
            "",
            f"Nine sequence lengths, {len(MICRO_BATCHES)} micro batches, two "
            f"precisions and three effective capacities "
            f"({', '.join(f'{f:.0%}' for f in BUDGET_FRACTIONS)} of "
            f"{self.total_bytes / GB:.1f} GB).",
            "",
        ]
        if self.matrix:
            lines += [self.matrix.to_markdown(), ""]
        if self.per_budget:
            lines += ["#### By effective capacity", "",
                      "| Budget | cells | false wins | false losses | accuracy |",
                      "|---|---|---|---|---|"]
            for k, v in sorted(self.per_budget.items(), key=lambda kv: float(kv[0])):
                lines.append(
                    f"| {float(k):.0%} ({v['budget_gb']:.1f} GB) | {v['n']} | "
                    f"**{v['false_win']}** | {v['false_loss']} | {v['accuracy']:.0%} |"
                )
            lines.append("")
        if self.per_dtype:
            lines += ["#### By precision", "",
                      "| dtype | cells | false wins | false losses | accuracy |",
                      "|---|---|---|---|---|"]
            for k, v in sorted(self.per_dtype.items()):
                lines.append(
                    f"| `{k}` | {v['n']} | **{v['false_win']}** | {v['false_loss']} | "
                    f"{v['accuracy']:.0%} |"
                )
            lines.append("")
        if self.notes:
            lines += ["#### Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def select_cells(model: BlockMemoryModel, total_bytes: int, *,
                 per_budget_boundary: int = 6, per_budget_anchor: int = 2
                 ) -> list[Cell]:
    """Boundary-focused cells at each capacity and precision.

    Selected per (budget, precision) rather than pooled, because the boundary
    moves with both: a cell that straddles it at 7.9 GB in fp16 is deep inside
    the infeasible region at 2.8 GB in fp32.
    """
    prior = _prior_shapes()
    out: list[Cell] = []
    for frac in BUDGET_FRACTIONS:
        budget = total_bytes * frac
        for dtype in DTYPES:
            candidates = [
                (mb, seq, ckpt, dtype, frac)
                for seq in SEQ_LENS for mb in MICRO_BATCHES for ckpt in (False, True)
                # Exclude anything the earlier studies already answered.
                if not (dtype == "float16" and abs(frac - 0.92) < 1e-9
                        and (mb, seq, ckpt) in prior)
            ]
            chosen, _ = select_boundary_grid(
                candidates,
                boundary_distance=lambda c, b=budget: model.boundary_distance(_shape(c), b),
                predict_fits=lambda c, b=budget: model.fits(_shape(c), b),
                n_boundary=per_budget_boundary,
                n_clear_feasible=per_budget_anchor,
                n_clear_infeasible=per_budget_anchor,
            )
            out.extend(chosen)
    return out


def run_expanded_validation(*, progress: bool = True,
                            per_budget_boundary: int = 6) -> ExpandedValidation:
    import torch

    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description
    from losscolumn.thrusts.overlap.calibrate import _block

    model = BlockMemoryModel()
    total = int(torch.cuda.get_device_properties(0).total_memory)
    study = ExpandedValidation(model=model, device=device_description("cuda"),
                               total_bytes=total)
    study.cells = select_cells(model, total,
                               per_budget_boundary=per_budget_boundary)

    for frac in BUDGET_FRACTIONS:
        sub = [c for c in study.cells if abs(c[4] - frac) < 1e-9]
        budget = total * frac
        rep = preflight(
            sub,
            predict_fits=lambda c, b=budget: model.fits(_shape(c), b),
            boundary_distance=lambda c, b=budget: model.boundary_distance(_shape(c), b),
            label=f"budget {frac:.0%}",
        )
        study.preflights[f"{frac}"] = rep.to_dict()
        if progress:
            print(rep.to_markdown(), flush=True)
            print(flush=True)

    results: list[FeasibilityCell] = []
    with MeasurementLock("thrust1 memory expanded validation"):
        for frac in BUDGET_FRACTIONS:
            cfg = install_budget(frac)
            budget = cfg.budget_bytes
            sub = [c for c in study.cells if abs(c[4] - frac) < 1e-9]
            for i, cell in enumerate(sub, 1):
                mb, seq, ckpt, dtype, _ = cell
                dt = torch.float16 if dtype == "float16" else torch.float32
                shape = _shape(cell)
                predicted = FeasibilityObservation.predicted(
                    model.fits(shape, budget),
                    detail=model.explain(shape, budget),
                    predicted_bytes=int(model.predict_bytes(shape)),
                    budget_bytes=budget,
                )

                def run(mb=mb, seq=seq, ckpt=ckpt, dt=dt) -> None:
                    from torch.utils.checkpoint import checkpoint

                    block = _block(LOCAL_HIDDEN, LOCAL_FFN, LOCAL_HEADS, dt, "cuda")
                    x = torch.randn(mb, seq, LOCAL_HIDDEN, device="cuda", dtype=dt,
                                    requires_grad=True)
                    out = checkpoint(block, x, use_reentrant=False) if ckpt else block(x)
                    out.sum().backward()
                    del block, x, out

                measured = probe_feasibility(run, cfg)
                if progress:
                    mark = ""
                    if predicted.status.is_definite and measured.status.is_definite:
                        if predicted.is_feasible and not measured.is_feasible:
                            mark = "   <- FALSE WIN"
                        elif not predicted.is_feasible and measured.is_feasible:
                            mark = "   <- false loss"
                    print(f"  [{frac:.0%} {i}/{len(sub)}] mb={mb} seq={seq} "
                          f"ckpt={ckpt} {dtype}: model {predicted.status.value}, "
                          f"actual {measured.status.value}{mark}", flush=True)
                results.append(FeasibilityCell(
                    key=cell, predicted=predicted, measured=measured,
                    coords={"micro_batch": mb, "seq_len": seq, "checkpointing": ckpt,
                            "dtype": dtype, "budget_fraction": frac},
                ))
            release_budget()

    study.matrix = FeasibilityMatrix(cells=results,
                                     danger_weight=DEFAULT_DANGER_WEIGHT)
    study.per_budget = _slice(results, "budget_fraction", total)
    study.per_dtype = _slice(results, "dtype", total)
    study.notes.append(
        "block-memory-v2 was not modified. Its parameters are the values validated "
        "at v0.3.0-memory-v2 and are asserted by test."
    )
    return study


def _slice(cells: Sequence[FeasibilityCell], axis: str,
           total_bytes: int) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[FeasibilityCell]] = {}
    for c in cells:
        groups.setdefault(str(c.coords.get(axis)), []).append(c)
    out: dict[str, dict[str, Any]] = {}
    for k, group in groups.items():
        m = FeasibilityMatrix(cells=group)
        entry = {
            "n": m.n_comparable, "false_win": m.n_false_win,
            "false_loss": m.n_false_loss, "accuracy": m.accuracy,
            "infeasible_iou": m.infeasible_iou,
        }
        if axis == "budget_fraction":
            entry["budget_gb"] = total_bytes * float(k) / GB
        out[k] = entry
    return out
