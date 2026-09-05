"""Communication cost models, chosen from data rather than from convenience.

The straight line ``t = alpha + n/beta`` fit PCIe at R^2 = 1.000 and gloo
all-reduce at 0.984, and failed gloo all-gather at 0.715. A fit that explains
71% of the variance is not a parameter set; it is a statement that the model
family is wrong for that transport. The honest response is to find out what
family *is* right, not to widen the gate.

Three families, in increasing order of what they can express:

``LinearModel``     ``t = alpha + n/beta``. One latency, one bandwidth. Correct
                    when a transport does one thing across the whole size
                    range.
``PiecewiseModel``  the same form on either side of a breakpoint, with the
                    breakpoint fitted. Correct when a transport switches
                    algorithm -- which collectives routinely do, small messages
                    going one way and large messages another.
``RegimeModel``     an arbitrary number of segments, each with its own latency
                    and bandwidth, and the segmentation chosen by search.

More parameters always fit better, so the family is not chosen by fit quality.
It is chosen by **held-out error** on points the fit never saw, and reported
alongside AIC so that a reader can see the complexity penalty. A piecewise
model that wins on training error and loses on held-out error is rejected, and
the report says which.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Sample",
    "CostModel",
    "LinearModel",
    "PiecewiseModel",
    "RegimeModel",
    "fit_linear",
    "fit_piecewise",
    "fit_regime",
    "select_model",
    "ModelSelection",
]


@dataclass(frozen=True)
class Sample:
    """One timed collective."""

    nbytes: int
    seconds: float
    world: int = 2
    kind: str = "all_reduce"
    replicate: int = 0

    @property
    def effective_bytes(self) -> float:
        """Bytes over the wire under the ring cost model."""
        if self.world <= 1:
            return float(self.nbytes)
        factor = 2.0 if self.kind == "all_reduce" else 1.0
        return factor * (self.world - 1) / self.world * self.nbytes


# --------------------------------------------------------------------------
# model families
# --------------------------------------------------------------------------


@dataclass
class CostModel:
    """Base: predicts seconds from effective bytes."""

    family: str = "base"
    n_params: int = 0

    def predict(self, nbytes: float) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict_many(self, xs: Sequence[float]) -> np.ndarray:
        return np.array([self.predict(x) for x in xs], dtype=float)

    def residuals(self, samples: Sequence[Sample]) -> np.ndarray:
        y = np.array([s.seconds for s in samples], dtype=float)
        p = self.predict_many([s.effective_bytes for s in samples])
        return y - p

    def relative_residuals(self, samples: Sequence[Sample]) -> np.ndarray:
        """Residuals as a fraction of the measured time.

        The scale on which structure has to be chosen. Over a domain spanning
        four orders of magnitude an absolute residual is a statement about the
        largest points and nothing else.
        """
        y = np.array([s.seconds for s in samples], dtype=float)
        p = self.predict_many([s.effective_bytes for s in samples])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(y > 0, (y - p) / y, 0.0)
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    def r_squared(self, samples: Sequence[Sample]) -> float:
        y = np.array([s.seconds for s in samples], dtype=float)
        if y.size < 2 or float(np.var(y)) == 0.0:
            return float("nan")
        ss_res = float(np.sum(self.residuals(samples) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    def median_rel_error(self, samples: Sequence[Sample]) -> float:
        y = np.array([s.seconds for s in samples], dtype=float)
        p = self.predict_many([s.effective_bytes for s in samples])
        ok = y > 0
        if not ok.any():
            return float("nan")
        return float(np.median(np.abs(y[ok] - p[ok]) / y[ok]))

    def max_rel_error(self, samples: Sequence[Sample]) -> float:
        y = np.array([s.seconds for s in samples], dtype=float)
        p = self.predict_many([s.effective_bytes for s in samples])
        ok = y > 0
        if not ok.any():
            return float("nan")
        return float(np.max(np.abs(y[ok] - p[ok]) / y[ok]))

    def relative_aic(self, samples: Sequence[Sample]) -> float:
        """AIC on the relative-residual scale.

        The one that governs structural decisions. An absolute-residual AIC over
        a domain spanning four orders of magnitude is a statement about the
        largest points, so it will buy a segment that helps them and refuse one
        that helps the small end -- while the segment fits themselves are
        weighted to do the opposite.
        """
        n = len(samples)
        if n <= self.n_params + 1:
            return float("inf")
        rss = float(np.sum(self.relative_residuals(samples) ** 2))
        if rss <= 0:
            return -float("inf")
        return n * math.log(rss / n) + 2 * self.n_params

    def aic(self, samples: Sequence[Sample]) -> float:
        """Akaike information criterion on absolute residuals.

        Reported for continuity with earlier artifacts. Structural decisions use
        `relative_aic`; see its docstring for why they must.
        """
        n = len(samples)
        if n <= self.n_params + 1:
            return float("inf")
        rss = float(np.sum(self.residuals(samples) ** 2))
        if rss <= 0:
            return -float("inf")
        return n * math.log(rss / n) + 2 * self.n_params

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "n_params": self.n_params}


@dataclass
class LinearModel(CostModel):
    alpha_s: float = 0.0
    beta_bytes_per_s: float = 1.0

    def __post_init__(self) -> None:
        self.family, self.n_params = "linear", 2

    def predict(self, nbytes: float) -> float:
        return self.alpha_s + nbytes / self.beta_bytes_per_s

    @property
    def alpha_us(self) -> float:
        return self.alpha_s * 1e6

    @property
    def beta_gbs(self) -> float:
        return self.beta_bytes_per_s / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "alpha_us": self.alpha_us,
                "beta_gbs": self.beta_gbs}


@dataclass
class PiecewiseModel(CostModel):
    """Two linear regimes meeting at a fitted breakpoint."""

    breakpoint_bytes: float = 0.0
    low: LinearModel = field(default_factory=LinearModel)
    high: LinearModel = field(default_factory=LinearModel)

    def __post_init__(self) -> None:
        self.family, self.n_params = "piecewise", 5

    def predict(self, nbytes: float) -> float:
        m = self.low if nbytes < self.breakpoint_bytes else self.high
        return m.predict(nbytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "breakpoint_bytes": self.breakpoint_bytes,
            "breakpoint_kib": self.breakpoint_bytes / 1024,
            "low": self.low.to_dict(), "high": self.high.to_dict(),
        }


@dataclass
class RegimeModel(CostModel):
    """An ordered set of size regimes, each with its own latency and bandwidth."""

    edges: tuple[float, ...] = ()
    segments: tuple[LinearModel, ...] = ()

    def __post_init__(self) -> None:
        self.family = "regime"
        self.n_params = 2 * len(self.segments) + len(self.edges)

    def predict(self, nbytes: float) -> float:
        i = 0
        while i < len(self.edges) and nbytes >= self.edges[i]:
            i += 1
        return self.segments[min(i, len(self.segments) - 1)].predict(nbytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "edges_bytes": list(self.edges),
            "edges_kib": [e / 1024 for e in self.edges],
            "segments": [s.to_dict() for s in self.segments],
        }


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _ols(samples: Sequence[Sample], *, loss: str = "weighted") -> LinearModel | None:
    """Fit ``t = alpha + n/beta``, weighted for proportional error.

    Unweighted least squares minimises absolute residuals, so on a log-spaced
    size sweep it is dominated entirely by the largest messages: a 32 MiB point
    taking 100 ms contributes ten thousand times the leverage of a 1 KiB point
    taking 1 ms, and the small end is fitted only incidentally.

    That is the same failure as judging the fit by R^2, and it has the same
    consequence -- a model that is 45% wrong on every small message and looks
    excellent. Timing error is multiplicative, not additive, so each point is
    weighted by ``1/t^2`` and every message size gets comparable influence.
    """
    if len(samples) < 2:
        return None
    x = np.array([s.effective_bytes for s in samples], dtype=float)
    y = np.array([s.seconds for s in samples], dtype=float)
    if np.ptp(x) <= 0 or not (y > 0).all():
        return None
    if loss == "weighted":
        slope, intercept = np.polyfit(x, y, 1, w=1.0 / y)
    elif loss in ("log", "huber"):
        fitted = _fit_nonlinear(x, y, loss=loss)
        if fitted is None:
            return None
        intercept, slope = fitted
    else:
        raise ValueError(f"unknown loss {loss!r}")
    if slope <= 0:
        return None
    return LinearModel(alpha_s=max(intercept, 0.0), beta_bytes_per_s=1.0 / slope)


def _fit_nonlinear(x: np.ndarray, y: np.ndarray, *, loss: str
                   ) -> tuple[float, float] | None:
    """Fit alpha and the slope under a log-relative or robust loss.

    ``log``   minimises squared error in log time, which is the exact form of
              "get every message size equally right" rather than the 1/t
              weighting's approximation to it.
    ``huber`` uses a soft-L1 loss on relative residuals, so a single
              pathological point -- a scheduling hiccup on one repeat -- moves
              the fit far less than it would under least squares.

    Both need an iterative solver. Where SciPy is unavailable the weighted
    linear fit is returned instead, and the caller records which was used.
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        # SciPy is optional. Without it the log and robust estimators are simply
        # unavailable, which the selection reports rather than silently
        # substituting the weighted fit under their name.
        return None

    slope0, inter0 = np.polyfit(x, y, 1, w=1.0 / y)
    if slope0 <= 0:
        return None

    def resid(theta: np.ndarray) -> np.ndarray:
        a, b = theta
        pred = np.maximum(a + b * x, 1e-12)
        if loss == "log":
            return np.log(pred) - np.log(y)
        return (pred - y) / y

    try:
        out = least_squares(
            resid, x0=np.array([max(inter0, 1e-9), slope0]),
            loss="soft_l1" if loss == "huber" else "linear",
            bounds=(np.array([0.0, 1e-18]), np.array([np.inf, np.inf])),
            max_nfev=2000,
        )
    except Exception:
        return None
    if not out.success:
        return None
    return float(out.x[0]), float(out.x[1])


def fit_linear(samples: Sequence[Sample], *, loss: str = "weighted"
               ) -> LinearModel | None:
    return _ols(samples, loss=loss)


def fit_piecewise(samples: Sequence[Sample], *, min_per_side: int = 3,
                  loss: str = "weighted") -> PiecewiseModel | None:
    """Fit two regimes, choosing the breakpoint by exhaustive search.

    The breakpoint is a real physical quantity -- the size at which the
    collective changes algorithm -- so it is fitted rather than assumed, and
    reported so it can be checked against what the library documents.
    """
    ordered = sorted(samples, key=lambda s: s.effective_bytes)
    if len(ordered) < 2 * min_per_side:
        return None
    best: tuple[float, PiecewiseModel] | None = None
    for i in range(min_per_side, len(ordered) - min_per_side + 1):
        lo, hi = ordered[:i], ordered[i:]
        ml, mh = _ols(lo, loss=loss), _ols(hi, loss=loss)
        if ml is None or mh is None:
            continue
        bp = (ordered[i - 1].effective_bytes + ordered[i].effective_bytes) / 2
        model = PiecewiseModel(breakpoint_bytes=bp, low=ml, high=mh)
        # Relative, not absolute: see relative_residuals. Choosing the
        # breakpoint by absolute RSS places it wherever the largest messages
        # want it, which is the leverage problem the 1/t weighting exists to
        # remove from the segment fits.
        rss = float(np.sum(model.relative_residuals(ordered) ** 2))
        if best is None or rss < best[0]:
            best = (rss, model)
    return best[1] if best else None


def fit_regime(samples: Sequence[Sample], *, max_segments: int = 4,
               min_per_side: int = 3, loss: str = "weighted") -> RegimeModel | None:
    """Segment the size range into up to ``max_segments`` linear regimes.

    Greedy top-down: repeatedly split the segment whose residual is worst,
    keeping the split only if it lowers AIC. That matters because a transport
    can have more than one transition -- a latency-to-bandwidth crossover at the
    small end and a cache-to-DRAM crossover higher up -- and the two-segment
    model this replaced could express only one of them, so it placed its single
    breakpoint between them and fitted neither side of the middle.

    The measured signature of that case is a *non-monotonic* effective
    bandwidth: it rises out of the latency floor, peaks where the buffer still
    fits in cache, and falls again once it does not. No two-segment model can
    reproduce a peak.
    """
    ordered = sorted(samples, key=lambda s: s.effective_bytes)
    base = _ols(ordered, loss=loss)
    if base is None:
        return None

    # Each entry is (slice_of_ordered, fitted_model).
    segs: list[tuple[list[Sample], LinearModel]] = [(list(ordered), base)]

    def rss_of(pieces: list[tuple[list[Sample], LinearModel]]) -> float:
        # Relative residuals, for the same reason fit_piecewise uses them: the
        # decision of where to split must not be made by the largest points
        # alone.
        return float(sum(np.sum(m.relative_residuals(g) ** 2) for g, m in pieces))

    def build(pieces: list[tuple[list[Sample], LinearModel]]) -> RegimeModel:
        edges = tuple(
            (pieces[i][0][-1].effective_bytes + pieces[i + 1][0][0].effective_bytes) / 2
            for i in range(len(pieces) - 1)
        )
        return RegimeModel(edges=edges, segments=tuple(m for _, m in pieces))

    current = build(segs)
    while len(segs) < max_segments:
        best: tuple[float, int, list[tuple[list[Sample], LinearModel]]] | None = None
        for i, (group, _) in enumerate(segs):
            if len(group) < 2 * min_per_side:
                continue
            for k in range(min_per_side, len(group) - min_per_side + 1):
                lo, hi = group[:k], group[k:]
                ml, mh = _ols(lo, loss=loss), _ols(hi, loss=loss)
                if ml is None or mh is None:
                    continue
                trial = segs[:i] + [(lo, ml), (hi, mh)] + segs[i + 1:]
                r = rss_of(trial)
                if best is None or r < best[0]:
                    best = (r, i, trial)
        if best is None:
            break
        candidate = build(best[2])
        if candidate.relative_aic(ordered) >= current.relative_aic(ordered):
            break
        segs, current = best[2], candidate
    return current


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


@dataclass
class ModelSelection:
    """Which family the data chose, and what the alternatives scored."""

    chosen: CostModel | None = None
    chosen_family: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    transport: str = ""
    kind: str = ""
    world: int = 0
    n_fit: int = 0
    n_heldout: int = 0
    reason: str = ""
    # Median run-to-run variation for this group. A model cannot predict better
    # than the measurement repeats, so a held-out error at this level is the
    # instrument's, not the model's -- and no better family would fix it.
    noise_floor: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport, "kind": self.kind, "world": self.world,
            "noise_floor": self.noise_floor,
            "chosen_family": self.chosen_family,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "n_fit": self.n_fit, "n_heldout": self.n_heldout,
            "candidates": self.candidates, "reason": self.reason,
        }

    def to_markdown(self) -> str:
        lines = [
            f"**{self.transport} / {self.kind} / world {self.world}** &mdash; "
            f"chosen family: `{self.chosen_family or 'none'}`"
            + (f" &nbsp;&middot;&nbsp; measurement noise floor "
               f"{self.noise_floor:.1%}" if self.noise_floor == self.noise_floor else ""),
            "",
            "| Family | params | fit R² | held-out median | **worst tier** | AIC |",
            "|---|---|---|---|---|---|",
        ]
        ranked = sorted(
            self.candidates,
            key=lambda c: c.get("heldout_worst_tier_err", float("inf"))
            if c.get("heldout_worst_tier_err") == c.get("heldout_worst_tier_err")
            else float("inf"),
        )
        for c in ranked[:8]:
            mark = " **<-**" if c["family"] == self.chosen_family else ""
            worst = c.get("heldout_worst_tier_err", float("nan"))
            lines.append(
                f"| `{c['family']}`{mark} | {c['n_params']} | {c['fit_r2']:.3f} | "
                f"{c['heldout_median_rel_err']:.1%} | {worst:.1%} | "
                f"{c['aic']:.1f} |"
            )
        if len(ranked) > 8:
            lines.append(f"| _{len(ranked) - 8} further combinations_ | | | | | |")
        lines += ["", self.reason]
        return "\n".join(lines)




def cross_validate(
    samples: Sequence[Sample],
    build: Any,
    *,
    k: int = 4,
    tier_fn: Any = None,
) -> dict[str, Any]:
    """K-fold cross-validation over message sizes, stratified by regime.

    A single nested hold-out leaves about one size per regime, and a
    worst-regime score computed on one point is mostly noise -- pessimistic
    when that point is unlucky, optimistic when it is not. Cross-validation
    uses every calibration size as held-out exactly once, so each regime gets
    as many out-of-sample predictions as it has sizes.

    Folds are assigned by round-robin over size *within* each regime, so no
    fold can be missing a regime entirely, which would make its worst-regime
    score incomparable with the others.
    """
    sizes = sorted({s.effective_bytes for s in samples})
    if len(sizes) < 2 * k:
        k = max(2, len(sizes) // 2)
    if len(sizes) < 4:
        return {"ok": False, "reason": "too few distinct sizes to cross-validate"}

    by_tier: dict[str, list[float]] = {}
    for x in sizes:
        raw = next((s.nbytes for s in samples if s.effective_bytes == x), int(x))
        by_tier.setdefault(str(tier_fn(raw)) if tier_fn else "all", []).append(x)

    fold_of: dict[float, int] = {}
    for group in by_tier.values():
        for i, x in enumerate(sorted(group)):
            fold_of[x] = i % k

    errs: list[float] = []
    per_tier: dict[str, list[float]] = {}
    n_fitted = 0
    for f in range(k):
        train = [s for s in samples if fold_of.get(s.effective_bytes, 0) != f]
        test = [s for s in samples if fold_of.get(s.effective_bytes, 0) == f]
        if not train or not test:
            continue
        try:
            model = build(train)
        except Exception:
            model = None
        if model is None:
            continue
        n_fitted += 1
        for t in test:
            if t.seconds <= 0:
                continue
            e = abs(model.predict(t.effective_bytes) - t.seconds) / t.seconds
            errs.append(e)
            tier = str(tier_fn(t.nbytes)) if tier_fn else "all"
            per_tier.setdefault(tier, []).append(e)

    if not errs or n_fitted < 2:
        return {"ok": False, "reason": "the family could not be fitted on enough folds"}
    tier_medians = {t: float(np.median(v)) for t, v in per_tier.items()}
    return {
        "ok": True,
        "k": k,
        "n_folds_fitted": n_fitted,
        "n_predictions": len(errs),
        "median_rel_err": float(np.median(errs)),
        "max_rel_err": float(np.max(errs)),
        "per_tier_err": tier_medians,
        "worst_tier_err": max(tier_medians.values()) if tier_medians else float("nan"),
        "all_tier_err": dict(tier_medians),
        "n_per_tier": {t: len(v) for t, v in per_tier.items()},
    }


def _per_tier_error(model: CostModel, samples: Sequence[Sample],
                    tier_fn: Any) -> dict[str, float]:
    """Median relative error within each size regime."""
    if tier_fn is None or not samples:
        return {}
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(str(tier_fn(s.nbytes)), []).append(s)
    return {t: model.median_rel_error(g) for t, g in groups.items() if g}


def _worst_gradable(per_tier: dict, ungradable: Sequence[str]) -> float:
    """Worst error over the tiers that can actually be graded.

    Falls back to the plain worst when every tier is ungradable: a family is not
    promoted on the strength of having no evidence against it.
    """
    usable = {t: v for t, v in (per_tier or {}).items()
              if t not in set(ungradable) and v == v}
    if usable:
        return float(max(usable.values()))
    vals = [v for v in (per_tier or {}).values() if v == v]
    return float(max(vals)) if vals else float("nan")


def select_model(
    fit_samples: Sequence[Sample],
    heldout_samples: Sequence[Sample],
    *,
    transport: str = "",
    kind: str = "",
    world: int = 0,
    noise_floor: float = float("nan"),
    max_heldout_median_err: float = 0.15,
    tier_fn: Any = None,
    ungradable_tiers: Sequence[str] = (),
    losses: Sequence[str] = ("weighted", "log", "huber"),
) -> ModelSelection:
    """Choose the family by held-out error, not by fit quality.

    A more flexible family always fits its own data better; the only question
    that discriminates is how it does on points it never saw. AIC is reported
    beside it so the complexity penalty is visible, but the decision is the
    held-out error, because that is the quantity a user of the model actually
    experiences.
    """
    sel = ModelSelection(transport=transport, kind=kind, world=world,
                         noise_floor=noise_floor,
                         n_fit=len(fit_samples), n_heldout=len(heldout_samples))
    # Each family is tried under each loss. A family that only wins under one
    # estimator is telling you about the estimator, and the report shows both.
    builders: list[tuple[str, Any]] = []
    for lname in losses:
        builders += [
            (f"linear/{lname}", lambda s, ln=lname: fit_linear(s, loss=ln)),
            (f"piecewise/{lname}", lambda s, ln=lname: fit_piecewise(s, loss=ln)),
            (f"regime/{lname}", lambda s, ln=lname: fit_regime(s, loss=ln)),
        ]
    best: tuple[float, CostModel, str] | None = None
    # Every point the selector is allowed to see, used through cross-validation
    # rather than a single nested split.
    pool = list(fit_samples) + list(heldout_samples)
    for name, build in builders:
        cv = cross_validate(pool, build, tier_fn=tier_fn)
        try:
            model = build(fit_samples)
        except Exception:
            model = None
        if model is None:
            # Every candidate carries the same keys whether or not it fitted.
            # A ragged shape here meant consumers had to special-case the
            # failure path, and one did not -- which only showed up in a fresh
            # environment without SciPy, where the log and robust estimators
            # take exactly this branch.
            sel.candidates.append({
                "family": name, "n_params": 0, "fit_r2": float("nan"),
                "cv_ok": False,
                "cv_note": "could not be fitted on these points",
                "heldout_median_rel_err": float("nan"),
                "heldout_max_rel_err": float("nan"),
                "heldout_per_tier_err": {},
                "heldout_worst_tier_err": float("nan"),
                "cv_n_predictions": 0, "cv_n_per_tier": {},
                "aic": float("inf"), "model": None,
                "note": "could not be fitted on these points",
            })
            continue
        entry = {
            "family": name,
            "n_params": model.n_params,
            "fit_r2": model.r_squared(pool),
            "cv_ok": cv.get("ok", False),
            "cv_note": cv.get("reason", ""),
            "heldout_median_rel_err": cv.get("median_rel_err", float("nan")),
            "heldout_max_rel_err": cv.get("max_rel_err", float("nan")),
            "heldout_per_tier_err": cv.get("per_tier_err", {}),
            "heldout_worst_tier_err": cv.get("worst_tier_err", float("nan")),
            "heldout_worst_gradable_tier_err": _worst_gradable(
                cv.get("per_tier_err", {}), ungradable_tiers),
            "ungradable_tiers": list(ungradable_tiers),
            "cv_n_predictions": cv.get("n_predictions", 0),
            "cv_n_per_tier": cv.get("n_per_tier", {}),
            "aic": model.aic(pool),
            "relative_aic": model.relative_aic(pool),
            "model": model.to_dict(),
        }
        sel.candidates.append(entry)
        # Selection is on the WORST tier, not the pooled median. Pooling lets a
        # family win by being excellent on the tiers with the most points while
        # being useless on one regime -- which is exactly how a model that was
        # 35% wrong in the medium tier came to be accepted.
        # Scored on the worst GRADABLE tier. Still the worst, not the pooled
        # median: pooling lets a family win by being excellent where the points
        # are and useless in one regime. But a tier whose measurement cannot
        # resolve the gate has no evidence to offer about any model, so it does
        # not get to reject one. It remains uncovered in the coverage matrix
        # either way, and a group is covered only when every tier is.
        score = (entry["heldout_worst_gradable_tier_err"]
                 if entry["heldout_per_tier_err"]
                 else entry["heldout_median_rel_err"])
        if math.isfinite(score) and (best is None or score < best[0]):
            best = (score, model, name)

    if best is None:
        sel.reason = "no family could be fitted on these points"
        return sel

    score, model, name = best
    if score > max_heldout_median_err:
        sel.chosen = None
        sel.chosen_family = ""
        noise_limited = (
            math.isfinite(noise_floor) and score <= noise_floor * 1.5
        )
        sel.reason = (
            f"The best family, `{name}`, still misses its worst-fitting regime by "
            f"{score:.0%} (gate: {max_heldout_median_err:.0%}). No family here "
            f"describes this transport, so none is promoted; all remain diagnostic. "
            "Widening the gate to admit one would be choosing the answer."
            + (f" Scored over the gradable regimes only; "
               f"{', '.join(ungradable_tiers)} could not be measured precisely "
               "enough to offer evidence about any model."
               if ungradable_tiers else "")
        )
        if noise_limited:
            sel.reason += (
                f" Note that run-to-run variation on this group is {noise_floor:.1%}, "
                "so the residual is close to the measurement noise: a better model "
                "family would not fix it, and a cleaner measurement might. That is a "
                "statement about the instrument, not about the model."
            )
        return sel

    sel.chosen, sel.chosen_family = model, name
    # The parsimony tiebreak must compare the SAME metric the score uses.
    # It used to filter on the pooled median while `score` was the worst
    # regime, which is apples to oranges: a simpler family with a good median
    # and a bad worst regime displaced a better model, reintroducing exactly
    # the failure the worst-regime criterion exists to prevent. It really did:
    # all_gather/world4 selected linear at 14.9% worst-regime error when a
    # regime model scoring 11.2% was available and under the gate.
    def _score_of(c: dict[str, Any]) -> float:
        v = (c.get("heldout_worst_tier_err") if c.get("heldout_per_tier_err")
             else c.get("heldout_median_rel_err"))
        return v if v is not None and v == v else float("inf")

    simpler = [c for c in sel.candidates
               if c["n_params"] < model.n_params and math.isfinite(_score_of(c))]
    close = [c for c in simpler if _score_of(c) <= score * 1.1]
    if close:
        pick = min(close, key=lambda c: c["n_params"])
        sel.reason = (
            f"`{pick['family']}` is within 10% of `{name}` on worst-regime held-out "
            f"error ({_score_of(pick):.1%} against {score:.1%}) with fewer "
            "parameters, so the simpler family is preferred."
        )
        for c in sel.candidates:
            if c["family"] == pick["family"]:
                rebuilt = dict(builders)[pick["family"]](pool)
                sel.chosen, sel.chosen_family = rebuilt, pick["family"]
        return sel

    sel.reason = (
        f"`{name}` wins on worst-tier held-out error ({score:.1%}) against "
        + ", ".join(
            f"`{c['family']}` {c.get('heldout_worst_tier_err', float('nan')):.1%}"
            for c in sorted(
                sel.candidates,
                key=lambda c: c.get("heldout_worst_tier_err", float("inf")))[:3]
            if c["family"] != name
        )
        + ". The decision is held-out error, not fit quality: a more flexible family "
          "always fits its own data better."
    )
    return sel
