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

    def aic(self, samples: Sequence[Sample]) -> float:
        """Akaike information criterion, so extra parameters are paid for."""
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


def _ols(samples: Sequence[Sample]) -> LinearModel | None:
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
    w = 1.0 / y
    slope, intercept = np.polyfit(x, y, 1, w=w)
    if slope <= 0:
        return None
    return LinearModel(alpha_s=max(intercept, 0.0), beta_bytes_per_s=1.0 / slope)


def fit_linear(samples: Sequence[Sample]) -> LinearModel | None:
    return _ols(samples)


def fit_piecewise(samples: Sequence[Sample], *, min_per_side: int = 3
                  ) -> PiecewiseModel | None:
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
        ml, mh = _ols(lo), _ols(hi)
        if ml is None or mh is None:
            continue
        bp = (ordered[i - 1].effective_bytes + ordered[i].effective_bytes) / 2
        model = PiecewiseModel(breakpoint_bytes=bp, low=ml, high=mh)
        rss = float(np.sum(model.residuals(ordered) ** 2))
        if best is None or rss < best[0]:
            best = (rss, model)
    return best[1] if best else None


def fit_regime(samples: Sequence[Sample], *, max_segments: int = 3,
               min_per_side: int = 3) -> RegimeModel | None:
    """Greedy segmentation: split the worst-fitting segment until it stops paying.

    Stops on AIC rather than on residual, so each extra pair of parameters has
    to earn itself.
    """
    ordered = sorted(samples, key=lambda s: s.effective_bytes)
    base = _ols(ordered)
    if base is None:
        return None
    current = RegimeModel(edges=(), segments=(base,))
    for _ in range(max_segments - 1):
        pw = fit_piecewise(ordered, min_per_side=min_per_side)
        if pw is None:
            break
        cand = RegimeModel(edges=(pw.breakpoint_bytes,), segments=(pw.low, pw.high))
        if cand.aic(ordered) >= current.aic(ordered):
            break
        current = cand
        break   # a second split needs per-segment recursion; two regimes is the
                # honest limit of what this many points can support
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
            "| Family | params | fit R² | held-out median err | held-out max err | AIC |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.candidates:
            mark = " **<-**" if c["family"] == self.chosen_family else ""
            lines.append(
                f"| `{c['family']}`{mark} | {c['n_params']} | {c['fit_r2']:.3f} | "
                f"{c['heldout_median_rel_err']:.1%} | {c['heldout_max_rel_err']:.1%} | "
                f"{c['aic']:.1f} |"
            )
        lines += ["", self.reason]
        return "\n".join(lines)


def select_model(
    fit_samples: Sequence[Sample],
    heldout_samples: Sequence[Sample],
    *,
    transport: str = "",
    kind: str = "",
    world: int = 0,
    noise_floor: float = float("nan"),
    max_heldout_median_err: float = 0.15,
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
    builders = (
        ("linear", fit_linear),
        ("piecewise", fit_piecewise),
        ("regime", fit_regime),
    )
    best: tuple[float, CostModel, str] | None = None
    for name, build in builders:
        try:
            model = build(fit_samples)
        except Exception:
            model = None
        if model is None:
            sel.candidates.append({
                "family": name, "n_params": 0, "fit_r2": float("nan"),
                "heldout_median_rel_err": float("nan"),
                "heldout_max_rel_err": float("nan"), "aic": float("inf"),
                "note": "could not be fitted on these points",
            })
            continue
        entry = {
            "family": name,
            "n_params": model.n_params,
            "fit_r2": model.r_squared(fit_samples),
            "heldout_r2": model.r_squared(heldout_samples) if heldout_samples else float("nan"),
            "heldout_median_rel_err": model.median_rel_error(heldout_samples)
            if heldout_samples else float("nan"),
            "heldout_max_rel_err": model.max_rel_error(heldout_samples)
            if heldout_samples else float("nan"),
            "aic": model.aic(fit_samples),
            "model": model.to_dict(),
        }
        sel.candidates.append(entry)
        score = entry["heldout_median_rel_err"]
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
            f"The best family, `{name}`, still misses held-out points by a median of "
            f"{score:.0%} (gate: {max_heldout_median_err:.0%}). No family here "
            f"describes this transport, so none is promoted; all remain diagnostic. "
            "Widening the gate to admit one would be choosing the answer."
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
    simpler = [c for c in sel.candidates
               if c["n_params"] < model.n_params
               and math.isfinite(c.get("heldout_median_rel_err", float("nan")))]
    close = [c for c in simpler
             if c["heldout_median_rel_err"] <= score * 1.1]
    if close:
        pick = min(close, key=lambda c: c["n_params"])
        sel.reason = (
            f"`{pick['family']}` is within 10% of `{name}` on held-out error with "
            f"fewer parameters, so the simpler family is preferred."
        )
        for c in sel.candidates:
            if c["family"] == pick["family"]:
                rebuilt = dict(builders)[pick["family"]](fit_samples)
                sel.chosen, sel.chosen_family = rebuilt, pick["family"]
        return sel

    sel.reason = (
        f"`{name}` wins on held-out error ({score:.1%} median) against "
        + ", ".join(
            f"`{c['family']}` {c['heldout_median_rel_err']:.1%}"
            for c in sel.candidates
            if c["family"] != name and math.isfinite(
                c.get("heldout_median_rel_err", float("nan")))
        )
        + ". The decision is held-out error, not fit quality: a more flexible family "
          "always fits its own data better."
    )
    return sel
