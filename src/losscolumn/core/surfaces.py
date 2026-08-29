"""Synthetic performance surfaces with known ground truth.

The loss-region extractor and the cliff detector are the two pieces of this
package that turn measurements into claims. Everything downstream inherits
their errors, so they need a regression suite built on surfaces whose right
answer is known by construction rather than by inspection.

Each generator returns the envelope *and* the ground truth: which cells are
genuinely losses, and which adjacent-level transitions are genuinely cliffs. A
test then scores the detector against that, so "the detector still works" is a
measurement rather than an opinion.

These are also what the calibration study compares a model's predicted loss map
against, which is why they live in the package rather than in ``tests/``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.envelope import Cell, Envelope, Factor, Metric


@dataclass
class GroundTruth:
    """A synthetic surface and the answer a detector should reach on it."""

    envelope: Envelope
    name: str
    description: str
    axis: str = "x"
    true_cliffs: set[tuple[Cell, Cell]] = field(default_factory=set)
    true_loss_cells: set[Cell] = field(default_factory=set)
    expected_n_cliffs_per_fiber: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_fibers(self) -> int:
        return len(list(self.envelope.fibers(self.axis)))

    def score_cliffs(self, found: Sequence[Any]) -> dict[str, float]:
        """Precision and recall of a cliff report against the truth."""
        got = {(tuple(c.from_cell), tuple(c.to_cell)) for c in found}
        truth = {(tuple(a), tuple(b)) for a, b in self.true_cliffs}
        tp = len(got & truth)
        return {
            "true_positives": tp,
            "false_positives": len(got - truth),
            "false_negatives": len(truth - got),
            "precision": tp / len(got) if got else (1.0 if not truth else 0.0),
            "recall": tp / len(truth) if truth else (1.0 if not got else 0.0),
        }


def _base_envelope(
    x_levels: Sequence[Any],
    n_fibers: int,
    replicates: int,
    system: str = "sys",
    higher_is_better: bool = True,
) -> Envelope:
    return Envelope.allocate(
        (
            Factor("x", tuple(x_levels)),
            Factor("fiber", tuple(f"f{i}" for i in range(n_fibers)), ordered=False),
        ),
        Metric("throughput", "tok/s", higher_is_better=higher_is_better),
        [system],
        replicates,
        interleaved=True,
    )


def _fill(env: Envelope, system: str, values: Callable[[int], float], *,
          noise: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for cell in env.cells():
        base = values(cell[0])
        env.put(system, cell, base * (1 + rng.normal(0, noise, env.replicates)))


# --------------------------------------------------------------------------
# Test A -- a perfectly smooth decline
# --------------------------------------------------------------------------


def smooth_decline(*, ratio: float = 0.70, n_levels: int = 6, n_fibers: int = 3,
                   replicates: int = 11, noise: float = 0.01,
                   seed: int = 0) -> GroundTruth:
    """A surface that loses a constant fraction at every step.

    Steep -- 30% per level compounds to a 5x fall across the axis -- and
    containing no discontinuity at all. A detector that reports a cliff here
    has confused a slope for a surprise, which is the single most common way
    this kind of analysis goes wrong.
    """
    env = _base_envelope(range(n_levels), n_fibers, replicates)
    _fill(env, "sys", lambda i: 1000.0 * ratio**i, noise=noise, seed=seed)
    return GroundTruth(
        envelope=env,
        name="A-smooth-decline",
        description=f"constant {1 - ratio:.0%} degradation per level; no discontinuity",
        meta={"per_step_ratio": ratio},
    )


# --------------------------------------------------------------------------
# Test B -- exactly one abrupt transition
# --------------------------------------------------------------------------


def single_cliff(*, at: int = 3, drop: float = 0.45, n_levels: int = 6,
                 n_fibers: int = 3, replicates: int = 11, noise: float = 0.01,
                 seed: int = 1) -> GroundTruth:
    """Flat, then one abrupt fall between levels ``at-1`` and ``at``."""
    env = _base_envelope(range(n_levels), n_fibers, replicates)
    _fill(env, "sys", lambda i: 1000.0 * (drop if i >= at else 1.0),
          noise=noise, seed=seed)
    truth = set()
    for _fixed, base in env.fibers("x"):
        cells = env.fiber_cells("x", base)
        truth.add((cells[at - 1], cells[at]))
    return GroundTruth(
        envelope=env, name="B-single-cliff",
        description=f"one {1 - drop:.0%} fall at x={at}, flat elsewhere",
        true_cliffs=truth, expected_n_cliffs_per_fiber=1,
        meta={"at": at, "drop": drop},
    )


# --------------------------------------------------------------------------
# Test C -- several independent cliffs
# --------------------------------------------------------------------------


def multiple_cliffs(*, positions: Sequence[int] = (2, 5), drops: Sequence[float] = (0.55, 0.5),
                    n_levels: int = 8, n_fibers: int = 3, replicates: int = 11,
                    noise: float = 0.01, seed: int = 2) -> GroundTruth:
    """Two separated falls on the same axis; each must be found on its own."""
    env = _base_envelope(range(n_levels), n_fibers, replicates)

    def value(i: int) -> float:
        v = 1000.0
        for pos, d in zip(positions, drops, strict=True):
            if i >= pos:
                v *= d
        return v

    _fill(env, "sys", value, noise=noise, seed=seed)
    truth = set()
    for _fixed, base in env.fibers("x"):
        cells = env.fiber_cells("x", base)
        for pos in positions:
            truth.add((cells[pos - 1], cells[pos]))
    return GroundTruth(
        envelope=env, name="C-multiple-cliffs",
        description=f"{len(positions)} independent falls at x={list(positions)}",
        true_cliffs=truth, expected_n_cliffs_per_fiber=len(positions),
        meta={"positions": list(positions), "drops": list(drops)},
    )


# --------------------------------------------------------------------------
# Test D -- a noisy but smooth surface
# --------------------------------------------------------------------------


def noisy_smooth(*, noise: float = 0.06, n_levels: int = 8, n_fibers: int = 4,
                 replicates: int = 11, seed: int = 3) -> GroundTruth:
    """Flat in truth, heavily noisy in measurement. Nothing here is a cliff.

    The noise level is deliberately large -- 6% per replicate, which is worse
    than a well-controlled GPU benchmark -- because the failure mode being
    guarded against is a detector whose threshold collapses when the step
    distribution has little spread, and then reads ordinary noise as structure.
    """
    env = _base_envelope(range(n_levels), n_fibers, replicates)
    _fill(env, "sys", lambda i: 1000.0 * (1 - 0.02 * i), noise=noise, seed=seed)
    return GroundTruth(
        envelope=env, name="D-noisy-smooth",
        description=f"gentle 2%/level decline under {noise:.0%} measurement noise",
        meta={"noise": noise},
    )


# --------------------------------------------------------------------------
# Test E -- one catastrophic cliff alongside a moderate one
# --------------------------------------------------------------------------


def dominant_cliff(*, catastrophic_at: int = 2, catastrophic_drop: float = 0.05,
                   moderate_at: int = 5, moderate_drop: float = 0.6,
                   n_levels: int = 8, n_fibers: int = 3, replicates: int = 11,
                   noise: float = 0.01, seed: int = 4) -> GroundTruth:
    """A 95% collapse and a 40% fall on the same axis.

    The point of this case: a robust threshold estimated from the step
    distribution is pulled far out by the catastrophic step, and a detector
    that includes the candidate in its own baseline will then miss the moderate
    cliff entirely. Detecting only the dramatic failure and missing the one a
    practitioner is more likely to hit is worse than detecting neither, because
    it looks like a complete answer.
    """
    env = _base_envelope(range(n_levels), n_fibers, replicates)

    def value(i: int) -> float:
        v = 1000.0
        if i >= catastrophic_at:
            v *= catastrophic_drop
        if i >= moderate_at:
            v *= moderate_drop
        return v

    _fill(env, "sys", value, noise=noise, seed=seed)
    truth = set()
    for _fixed, base in env.fibers("x"):
        cells = env.fiber_cells("x", base)
        truth.add((cells[catastrophic_at - 1], cells[catastrophic_at]))
        truth.add((cells[moderate_at - 1], cells[moderate_at]))
    return GroundTruth(
        envelope=env, name="E-dominant-cliff",
        description=(
            f"a {1 - catastrophic_drop:.0%} collapse at x={catastrophic_at} and a "
            f"{1 - moderate_drop:.0%} fall at x={moderate_at}"
        ),
        true_cliffs=truth, expected_n_cliffs_per_fiber=2,
        meta={"catastrophic_drop": catastrophic_drop, "moderate_drop": moderate_drop},
    )


# --------------------------------------------------------------------------
# loss-region ground truth
# --------------------------------------------------------------------------


def planted_region(
    *,
    planted: dict[str, list[Any]] | None = None,
    regression: float = 0.30,
    replicates: int = 11,
    noise: float = 0.01,
    seed: int = 5,
    factors: Sequence[Factor] | None = None,
) -> GroundTruth:
    """Two systems identical except inside a known box.

    Replicates share a common shock, which is what interleaved A/B measurement
    actually produces and what the paired statistics downstream assume.
    """
    rng = np.random.default_rng(seed)
    factors = tuple(factors or (
        Factor("batch", (1, 2, 4, 8)),
        Factor("seq_len", (128, 512, 2048)),
        Factor("dtype", ("fp16", "bf16"), ordered=False),
    ))
    env = Envelope.allocate(
        factors,
        Metric("throughput", "tok/s", higher_is_better=True),
        ["method", "baseline"],
        replicates,
        interleaved=True,
    )
    truth: set[Cell] = set()
    for cell in env.cells():
        coords = env.coords(cell)
        base = 1000.0 * (1 + 0.1 * cell[0])
        inside = planted is not None and all(coords[k] in v for k, v in planted.items())
        if inside:
            truth.add(cell)
        shock = rng.normal(0, noise, replicates)
        jitter = rng.normal(0, noise / 3, replicates)
        env.put("baseline", cell, base * (1 + shock + jitter))
        env.put("method", cell,
                base * ((1 - regression) if inside else 1.0) * (1 + shock + jitter))
    return GroundTruth(
        envelope=env, name="planted-region",
        description=f"a {regression:.0%} regression confined to {planted}",
        true_loss_cells=truth,
        meta={"planted": planted, "regression": regression},
    )


def region_iou(found_cells: set[Cell], truth: set[Cell]) -> float:
    """Intersection over union of a detected region set and the truth."""
    if not found_cells and not truth:
        return 1.0
    union = found_cells | truth
    return len(found_cells & truth) / len(union) if union else 1.0


ALL_SURFACES: tuple[Callable[..., GroundTruth], ...] = (
    smooth_decline,
    single_cliff,
    multiple_cliffs,
    noisy_smooth,
    dominant_cliff,
)


def describe_all() -> str:
    lines = ["| Surface | Description | Expected cliffs per fiber |",
             "|---------|-------------|---------------------------|"]
    for fn in ALL_SURFACES:
        gt = fn()
        lines.append(
            f"| `{gt.name}` | {gt.description} | {gt.expected_n_cliffs_per_fiber} |"
        )
    return "\n".join(lines)
