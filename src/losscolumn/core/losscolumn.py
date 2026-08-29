"""Loss-column extraction: from a measurement surface to a statement a hostile
reader will accept.

The proposal's central requirement is that every claim ships an explicit map of
where the method underperforms. Making that requirement enforceable means
answering a concrete question: given a few hundred cells, each with a verdict,
what is the *smallest set of human-readable statements* that covers every cell
where the method lost?

Dumping the losing cells is not an answer -- nobody reads 137 rows, and a list
of cells hides the structure ("it loses whenever seq_len >= 8192 in fp16"). The
answer used here is greedy maximal-box covering over the factor lattice, close
in spirit to bump hunting (PRIM) and to rule induction, with three properties
the standard needs:

1. **Complete.** Every loss cell lands in at least one region. A singleton box
   is always admissible, so the loop always terminates with full coverage.
2. **Pure.** A region may absorb ``inconclusive`` cells -- the boundary of a
   loss region is genuinely fuzzy -- but never a ``win`` or a ``tie``. A region
   containing a win is not a description of a loss.
3. **Ordered by what a reader cares about.** Regions are emitted worst-first by
   severity x coverage, not by discovery order.

The regions are the deliverable. The severity numbers attached to them are what
makes the deliverable checkable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.envelope import Cell, Envelope, Factor
from losscolumn.core.stats import CellComparison, design_can_reject, verdict_counts

_CODES = {"loss": 0, "win": 1, "tie": 2, "inconclusive": 3, "missing": 4}


def pct(x: float) -> str:
    """Format a regression, keeping "cannot run here" distinct from "very slow".

    An infinite regression means the configuration failed outright -- OOM, an
    unsupported shape, a crash. Printing that as "+inf%" invites a reader to
    treat it as a large number on the same scale as the others, when it is a
    categorically different result and usually the most important row in the
    table.
    """
    if not math.isfinite(x):
        return "cannot run"
    return f"{x:+.1f}%"


@dataclass
class LossRegion:
    """One axis-aligned region of the envelope where the method underperforms."""

    bounds: dict[str, list[Any]]         # factor -> the levels included
    n_cells: int
    n_loss: int
    n_inconclusive: int
    frac_of_envelope: float
    weighted_frac: float
    median_regression_pct: float
    worst_regression_pct: float
    worst_cell: dict[str, Any]
    max_q: float
    min_effect_pct: float
    contains_missing: int = 0
    missing_reasons: list[str] = field(default_factory=list)
    attribution: str | None = None       # filled by the thrust that owns it
    label: str = ""

    @property
    def purity(self) -> float:
        return self.n_loss / self.n_cells if self.n_cells else 0.0

    def describe(self, *, all_levels: dict[str, tuple] | None = None) -> str:
        """Human-readable condition, omitting axes that span their full range."""
        parts: list[str] = []
        for name, levels in self.bounds.items():
            if all_levels and len(levels) == len(all_levels.get(name, ())):
                continue
            if len(levels) == 1:
                parts.append(f"{name}={levels[0]}")
            elif all(isinstance(x, (int, float)) for x in levels):
                parts.append(f"{levels[0]} <= {name} <= {levels[-1]}")
            else:
                parts.append(f"{name} in {{{', '.join(str(x) for x in levels)}}}")
        return " and ".join(parts) if parts else "the entire envelope"

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["purity"] = self.purity
        return d


@dataclass
class LossColumn:
    """The mandatory loss column of a conforming claim."""

    metric: str
    unit: str
    method: str
    baseline: str
    mde_pct: float
    q_level: float
    regions: list[LossRegion] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    n_cells: int = 0
    loss_fraction: float = 0.0
    weighted_loss_fraction: float = 0.0
    inconclusive_fraction: float = 0.0
    win_fraction: float = 0.0
    n_unrunnable: int = 0
    envelope_digest: str | None = None
    factor_levels: dict[str, list[Any]] = field(default_factory=dict)
    design_check: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ---- headline numbers a reader can check ------------------------------

    @property
    def is_empty(self) -> bool:
        return not self.regions

    @property
    def underpowered(self) -> bool:
        return bool(self.design_check) and not self.design_check.get("can_reject", True)

    @property
    def worst_regression_pct(self) -> float:
        return max((r.worst_regression_pct for r in self.regions), default=0.0)

    @property
    def worst_finite_regression_pct(self) -> float:
        vals = [
            r.worst_regression_pct for r in self.regions if math.isfinite(r.worst_regression_pct)
        ]
        return max(vals, default=0.0)

    def summary_sentence(self) -> str:
        if self.is_empty:
            if self.underpowered:
                return (
                    "No loss region could be resolved, but this sweep is arithmetically "
                    "incapable of resolving one: "
                    f"{self.design_check.get('message', '')} An empty loss column here is "
                    "a property of the design, not evidence about the method."
                )
            if self.inconclusive_fraction > 0.25:
                return (
                    f"No loss region was resolved, but {self.inconclusive_fraction:.0%} of the "
                    f"envelope is inconclusive at the pre-registered MDE of {self.mde_pct:.1f}%; "
                    "absence of a loss column here is a statement about power, not about "
                    "the method."
                )
            return (
                f"No region of the swept envelope shows a regression of at least "
                f"{self.mde_pct:.1f}% at q<={self.q_level:g}."
            )
        tail = (
            f"worst measurable case {pct(self.worst_finite_regression_pct)}"
            if self.worst_finite_regression_pct
            else "every loss is an outright failure to run"
        )
        unrun = (
            f", and cannot run at all in {self.n_unrunnable} cell(s)"
            if self.n_unrunnable
            else ""
        )
        return (
            f"{self.method} loses to {self.baseline} on {self.loss_fraction:.0%} of the swept "
            f"envelope ({self.counts.get('loss', 0)}/{self.n_cells} cells), across "
            f"{len(self.regions)} region(s); {tail} on {self.metric}{unrun}."
        )

    def to_markdown(self) -> str:
        """The loss column itself, rendered as the table that ships with a claim."""
        lines = [
            f"### Loss column -- {self.method} vs {self.baseline} ({self.metric}, {self.unit})",
            "",
            self.summary_sentence(),
            "",
            "| # | Where it loses | Cells | Median | Worst | max q | Attributed to |",
            "|---|----------------|-------|--------|-------|-------|---------------|",
        ]
        levels = {k: tuple(v) for k, v in self.factor_levels.items()}
        for i, r in enumerate(self.regions, 1):
            q = "n/a" if not math.isfinite(r.max_q) else f"{r.max_q:.3g}"
            lines.append(
                f"| {i} | {r.describe(all_levels=levels)} | {r.n_loss}/{r.n_cells} "
                f"({r.frac_of_envelope:.0%} of envelope) | {pct(r.median_regression_pct)} | "
                f"{pct(r.worst_regression_pct)} | {q} | "
                f"{r.attribution or '_not yet attributed_'} |"
            )
        if not self.regions:
            lines.append("| - | _no loss region resolved_ | - | - | - | - | - |")
        lines += [
            "",
            f"- Envelope: {self.n_cells} cells. "
            f"Losses {self.counts.get('loss', 0)}, wins {self.counts.get('win', 0)}, "
            f"ties {self.counts.get('tie', 0)}, inconclusive {self.counts.get('inconclusive', 0)}, "
            f"missing {self.counts.get('missing', 0)}.",
            f"- Decision rule: regression >= {self.mde_pct:.1f}% (pre-registered MDE) "
            f"and Benjamini-Hochberg q <= {self.q_level:g}.",
        ]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "regions"}
        d["regions"] = [r.to_dict() for r in self.regions]
        d["summary"] = self.summary_sentence()
        return d


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def _levels_key(f: Factor, idxs: Iterable[int]) -> list[Any]:
    return [f.levels[i] for i in sorted(idxs)]


def _box_cells(box: list[list[int]]) -> list[Cell]:
    import itertools

    return list(itertools.product(*box))


def _contiguous(idxs: Sequence[int]) -> bool:
    s = sorted(idxs)
    return s == list(range(s[0], s[0] + len(s)))


def extract_loss_column(
    env: Envelope,
    comparisons: Sequence[CellComparison],
    *,
    method: str,
    baseline: str,
    mde: float = 0.05,
    q_level: float = 0.05,
    min_purity: float = 0.75,
    min_slab_purity: float = 0.5,
    weights: np.ndarray | None = None,
    max_regions: int = 12,
) -> LossColumn:
    """Cover every loss cell with a small set of readable regions.

    ``min_purity`` is the floor on loss-cells / cells for the whole region;
    ``min_slab_purity`` is the floor for each slab as it is added, which stops a
    large pure core from dragging in a mostly-inconclusive fringe. Both defaults
    are conservative: they produce more, tighter regions rather than one
    sweeping claim, because an overstated loss region is as much a reporting
    failure as an omitted one.
    """
    shape = env.grid_shape
    by_cell = {tuple(c.cell): c for c in comparisons}
    codes = np.full(shape, _CODES["missing"], dtype=np.int8)
    effect = np.full(shape, np.nan, dtype=float)
    qval = np.full(shape, np.nan, dtype=float)
    for c in comparisons:
        codes[tuple(c.cell)] = _CODES.get(c.verdict, 4)
        effect[tuple(c.cell)] = c.effect
        qval[tuple(c.cell)] = c.q_worse

    loss = codes == _CODES["loss"]
    absorbable = loss | (codes == _CODES["inconclusive"]) | (codes == _CODES["missing"])
    n_cells = int(np.prod(shape))

    if weights is None:
        weights = np.ones(shape, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else weights

    covered = np.zeros(shape, dtype=bool)
    regions: list[LossRegion] = []

    # Severity ordering: seed from the worst uncovered loss cell so the first
    # region emitted is the one a reader most needs to see.
    while True:
        cand = loss & ~covered
        if not cand.any() or len(regions) >= max_regions:
            break
        sev = np.where(cand, np.nan_to_num(effect, nan=-np.inf), -np.inf)
        seed = np.unravel_index(int(np.argmax(sev)), shape)
        box = [[int(i)] for i in seed]
        box = _grow(box, env, loss, absorbable, covered, min_purity, min_slab_purity)
        cells = _box_cells(box)
        for c in cells:
            if loss[c]:
                covered[c] = True
        regions.append(_summarize_region(env, box, cells, by_cell, weights, n_cells))

    regions = _prune_redundant(regions, env, loss)
    leftover = int(np.sum(loss & ~covered))
    counts = verdict_counts(comparisons)
    lc = LossColumn(
        metric=env.metric.name,
        unit=env.metric.unit,
        method=method,
        baseline=baseline,
        mde_pct=mde * 100.0,
        q_level=q_level,
        regions=sorted(
            regions, key=lambda r: (-r.worst_regression_pct * max(r.frac_of_envelope, 1e-9))
        ),
        counts=counts,
        n_cells=n_cells,
        loss_fraction=counts.get("loss", 0) / n_cells if n_cells else 0.0,
        weighted_loss_fraction=float(weights[loss].sum()),
        inconclusive_fraction=counts.get("inconclusive", 0) / n_cells if n_cells else 0.0,
        win_fraction=counts.get("win", 0) / n_cells if n_cells else 0.0,
        n_unrunnable=sum(
            1 for c in comparisons if c.verdict == "loss" and not math.isfinite(c.effect)
        ),
        envelope_digest=env.digest(),
        factor_levels={f.name: list(f.levels) for f in env.factors},
    )

    if leftover:
        lc.notes.append(
            f"{leftover} loss cell(s) were not covered because the region cap "
            f"(max_regions={max_regions}) was reached; the column is INCOMPLETE and "
            "must not be published in this state."
        )
    n_missing = counts.get("missing", 0)
    if n_missing:
        lc.notes.append(
            f"{n_missing} cell(s) could not be measured for either system and are "
            "excluded from the comparison rather than counted as wins."
        )
    if counts.get("inconclusive", 0) / max(n_cells, 1) > 0.2:
        lc.notes.append(
            f"{lc.inconclusive_fraction:.0%} of the envelope is inconclusive at r="
            f"{env.replicates}: the sweep resolves differences of roughly "
            f"{_median_resolution(comparisons):.1f}% and cannot speak to smaller ones."
        )
    if not env.interleaved:
        lc.notes.append(
            "Measurements were not interleaved; comparisons fall back to unpaired "
            "statistics and are correspondingly less powerful."
        )

    # The design check has to run whether or not losses were found, and it is
    # most important precisely when none were: an empty loss column from an
    # underpowered sweep is indistinguishable, on the page, from an empty one
    # earned by a method that genuinely never loses.
    n_live = sum(1 for c in comparisons if math.isfinite(c.p_worse))
    power = design_can_reject(env.replicates, n_live, q_level)
    lc.design_check = power
    if not power["can_reject"]:
        big = [
            c
            for c in comparisons
            if c.verdict != "loss" and math.isfinite(c.effect) and c.effect >= math.log1p(mde)
        ]
        lc.notes.append("UNDERPOWERED: " + power["message"])
        if big:
            worst = max(big, key=lambda c: c.effect)
            lc.notes.append(
                f"{len(big)} cell(s) show a regression above the MDE but could not be "
                f"declared, the largest being {worst.pct:+.1f}% at "
                f"{', '.join(f'{k}={v}' for k, v in worst.coords.items())}. "
                "This loss column understates the losses."
            )
    return lc


def _region_loss_cells(env: Envelope, region: LossRegion, loss: np.ndarray) -> set[Cell]:
    return {
        c
        for c in env.cells()
        if loss[c] and all(env.coords(c)[k] in v for k, v in region.bounds.items())
    }


def _prune_redundant(
    regions: list[LossRegion], env: Envelope, loss: np.ndarray
) -> list[LossRegion]:
    """Drop regions whose loss cells are already covered by the others.

    Greedy growth can produce a later box that overlaps an earlier one so
    heavily it contributes nothing new. Keeping it would inflate the apparent
    number of distinct failure modes, which is a subtler form of the same
    dishonesty the standard targets -- in the other direction.
    """
    sets = [_region_loss_cells(env, r, loss) for r in regions]
    order = sorted(range(len(regions)), key=lambda i: -len(sets[i]))
    keep: list[int] = []
    covered: set[Cell] = set()
    for i in order:
        if sets[i] - covered:
            keep.append(i)
            covered |= sets[i]
    return [regions[i] for i in sorted(keep)]


def _median_resolution(cmps: Sequence[CellComparison]) -> float:
    r = [c.resolution for c in cmps if math.isfinite(c.resolution)]
    return (math.exp(float(np.median(r))) - 1) * 100 if r else float("nan")


def _grow(
    box: list[list[int]],
    env: Envelope,
    loss: np.ndarray,
    absorbable: np.ndarray,
    covered: np.ndarray,
    min_purity: float,
    min_slab_purity: float,
) -> list[list[int]]:
    """Greedily extend the box one slab at a time, best gain first."""
    shape = env.grid_shape
    while True:
        best: tuple[float, int, list[list[int]]] | None = None
        for ax, f in enumerate(env.factors):
            cur = box[ax]
            candidates: list[int] = []
            if f.ordered:
                lo, hi = min(cur), max(cur)
                if lo - 1 >= 0:
                    candidates.append(lo - 1)
                if hi + 1 < shape[ax]:
                    candidates.append(hi + 1)
            else:
                candidates = [i for i in range(shape[ax]) if i not in cur]

            for new_i in candidates:
                slab = [list(b) for b in box]
                slab[ax] = [new_i]
                slab_cells = _box_cells(slab)
                if not all(absorbable[c] for c in slab_cells):
                    continue  # a win or a tie in the slab: stop here
                slab_loss = sum(1 for c in slab_cells if loss[c])
                if slab_loss / len(slab_cells) < min_slab_purity:
                    continue
                trial = [list(b) for b in box]
                trial[ax] = sorted(cur + [new_i])
                t_cells = _box_cells(trial)
                purity = sum(1 for c in t_cells if loss[c]) / len(t_cells)
                if purity < min_purity:
                    continue
                gain = sum(1 for c in slab_cells if loss[c] and not covered[c])
                score = gain + 0.25 * slab_loss  # break ties toward denser slabs
                if gain == 0 and slab_loss == 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, ax, trial)
        if best is None:
            return box
        box = best[2]


def _summarize_region(
    env: Envelope,
    box: list[list[int]],
    cells: list[Cell],
    by_cell: dict[Cell, CellComparison],
    weights: np.ndarray,
    n_cells: int,
) -> LossRegion:
    losses = [by_cell[c] for c in cells if by_cell[c].verdict == "loss"]
    incs = [c for c in cells if by_cell[c].verdict == "inconclusive"]
    miss = [by_cell[c] for c in cells if by_cell[c].verdict == "missing" or by_cell[c].reason]
    finite = [c for c in losses if math.isfinite(c.effect)]
    pcts = [c.pct for c in finite]
    worst = max(finite, key=lambda c: c.effect) if finite else (losses[0] if losses else None)
    hard = [c for c in losses if not math.isfinite(c.effect)]
    qs = [c.q_worse for c in losses if math.isfinite(c.q_worse)]

    return LossRegion(
        bounds={f.name: _levels_key(f, box[i]) for i, f in enumerate(env.factors)},
        n_cells=len(cells),
        n_loss=len(losses),
        n_inconclusive=len(incs),
        frac_of_envelope=len(cells) / n_cells if n_cells else 0.0,
        weighted_frac=float(sum(weights[c] for c in cells)),
        median_regression_pct=float(np.median(pcts)) if pcts else float("inf"),
        worst_regression_pct=float(np.max(pcts)) if pcts else float("inf"),
        worst_cell=dict(worst.coords) if worst else {},
        max_q=float(np.max(qs)) if qs else float("nan"),
        min_effect_pct=float(np.min(pcts)) if pcts else float("inf"),
        contains_missing=len(hard) + len([m for m in miss if m.verdict == "missing"]),
        missing_reasons=sorted({m.reason for m in miss if m.reason}),
        label=", ".join(
            f"{f.name}={_levels_key(f, box[i])}" for i, f in enumerate(env.factors)
        ),
    )
