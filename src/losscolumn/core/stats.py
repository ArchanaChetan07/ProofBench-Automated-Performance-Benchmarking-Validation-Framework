"""Paired comparison statistics, implemented in numpy so core has no scipy dep.

Everything here works in **log space**. Ratios of timings and throughputs are
multiplicative and right-skewed; differences of logs are approximately
symmetric, compose across factors, and give an effect measure that reads the
same whether you framed the metric as latency or as throughput.

Orientation convention, used without exception in this package:

    effect > 0  means THE METHOD IS WORSE THAN THE BASELINE.

The four-way verdict per cell is the part that matters for the standard. A cell
is not simply win-or-lose:

``loss``          significantly worse, by at least the pre-registered MDE.
``win``           significantly better, by at least the MDE.
``tie``           statistically equivalent within +/- MDE (a TOST result, not
                  an unrejected null -- "we found no difference" is a claim
                  that has to be earned).
``inconclusive``  the data cannot distinguish the three above. Reported as its
                  own category and counted, because an envelope that is 40%
                  inconclusive is a different object from one that is 5%.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from losscolumn.core.envelope import Cell, Envelope

Verdict = Literal["loss", "win", "tie", "inconclusive", "missing"]

# --------------------------------------------------------------------------
# distributions (kept local so `losscolumn.core` needs numpy only)
# --------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Absolute error < 1.15e-9 over (0, 1), which is far inside anything that
    matters for a bootstrap bias correction.
    """
    if not 0.0 < p < 1.0:
        return -math.inf if p <= 0.0 else math.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-12) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper-tail probability of Student's t."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def signflip_p(d: np.ndarray, *, alternative: str = "greater", n_mc: int = 20000,
               rng: np.random.Generator | None = None) -> float:
    """One-sided paired sign-flip permutation test on differences ``d``.

    Exact by enumeration when 2**n <= 32768, Monte Carlo otherwise. The null is
    that the sign of each paired difference is exchangeable, which is the right
    null for interleaved A/B timing: it assumes symmetry, not normality, and it
    is exact at the replicate counts benchmark work actually uses (r = 5..11).

    Monte Carlo p-values include the observed statistic in numerator and
    denominator, so they are never zero -- a p of 0 from resampling is a
    reporting error, not a finding.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan")
    if np.allclose(d, 0.0):
        return 1.0
    obs = float(np.mean(d))
    if alternative == "less":
        obs, d = -obs, -d

    if n <= 15:
        signs = 1 - 2 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
        stats = (signs * d).mean(axis=1)
        return float(np.mean(stats >= obs - 1e-15))

    rng = rng or np.random.default_rng(0)
    signs = rng.integers(0, 2, size=(n_mc, n)) * 2 - 1
    stats = (signs * d).mean(axis=1)
    return float((np.sum(stats >= obs - 1e-15) + 1) / (n_mc + 1))


def welch_p(a: np.ndarray, b: np.ndarray, *, alternative: str = "greater") -> tuple[float, float]:
    """Welch one-sided p for ``mean(a) - mean(b)``, plus the effective df."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan"), 0.0
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = math.sqrt(va + vb)
    if se == 0:
        return (1.0, float(na + nb - 2))
    t = (a.mean() - b.mean()) / se
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    p = t_sf(t, df) if alternative == "greater" else t_sf(-t, df)
    return float(p), float(df)


def bca_ci(
    sample: np.ndarray,
    stat: Any = np.mean,
    *,
    alpha: float = 0.05,
    n_boot: int = 4000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval.

    Percentile intervals are visibly wrong at r <= 8 replicates, which is the
    regime every GPU sweep lives in; BCa costs one extra jackknife pass and
    removes most of that error.
    """
    x = np.asarray(sample, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    theta = float(stat(x))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(x[i]) for i in idx], dtype=float)

    frac_less = float(np.mean(boots < theta))
    z0 = _norm_ppf(min(max(frac_less, 1.0 / (2 * n_boot)), 1 - 1.0 / (2 * n_boot)))

    jack = np.array([stat(np.delete(x, i)) for i in range(n)], dtype=float)
    jbar = jack.mean()
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * (float(np.sum((jbar - jack) ** 2)) ** 1.5)
    a = num / den if den != 0 else 0.0

    def endpoint(z_alpha: float) -> float:
        num_ = z0 + z_alpha
        adj = z0 + num_ / (1 - a * num_) if (1 - a * num_) != 0 else z0 + num_
        return float(np.clip(_norm_cdf(adj), 0.0, 1.0))

    lo_q = endpoint(_norm_ppf(alpha / 2))
    hi_q = endpoint(_norm_ppf(1 - alpha / 2))
    return (float(np.quantile(boots, lo_q)), float(np.quantile(boots, hi_q)))


def signflip_p_floor(replicates: int) -> float:
    """Smallest p-value an exact paired sign-flip test can ever produce.

    With ``r`` paired replicates there are ``2**r`` sign assignments, and the
    observed data is one of them, so no result -- however extreme -- can score
    below ``2**-r``. This is a hard floor, not an approximation.
    """
    return 2.0 ** -max(replicates, 1)


def min_replicates_for_fdr(n_tests: int, q: float = 0.05) -> int:
    """Replicates needed before any cell *can* be declared a loss.

    Benjamini-Hochberg rejects the most extreme of ``m`` tests only if its
    p-value is at most ``q/m``. Combined with the sign-flip floor above, a sweep
    with too few replicates is arithmetically incapable of reporting a loss no
    matter how large the regression is -- it will report an empty loss column
    and look like a clean bill of health.

    This is the single most dangerous failure mode of the whole standard: a
    conforming, well-formatted, entirely empty loss column produced by an
    underpowered design. Every analysis path checks it.
    """
    if n_tests <= 0:
        return 1
    return int(math.ceil(math.log2(n_tests / q)))


def design_can_reject(replicates: int, n_tests: int, q: float = 0.05,
                      exact_limit: int = 15) -> dict[str, Any]:
    """Whether this design could ever produce a loss, and what it would take."""
    need = min_replicates_for_fdr(n_tests, q)
    floor = signflip_p_floor(replicates) if replicates <= exact_limit else 1.0 / 20001
    threshold = q / max(n_tests, 1)
    ok = floor <= threshold
    return {
        "replicates": replicates,
        "n_tests": n_tests,
        "q": q,
        "p_floor": floor,
        "bh_threshold_for_rank_1": threshold,
        "can_reject": bool(ok),
        "min_replicates": need,
        "message": (
            ""
            if ok
            else (
                f"design cannot reject: with r={replicates} the smallest attainable "
                f"p-value is {floor:.2g}, but Benjamini-Hochberg over {n_tests} cells "
                f"requires p <= {threshold:.2g} for the most extreme cell. "
                f"At least r={need} replicates are needed before ANY loss can be "
                f"declared, regardless of effect size."
            )
        ),
    }


def bh_fdr(pvals: Sequence[float], q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up. Returns ``(rejected, qvalues)``.

    A sweep is hundreds of simultaneous tests. Without FDR control, an envelope
    of pure noise produces a confident-looking loss region at any alpha, which
    would make the loss column itself unfalsifiable.
    """
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out_q = np.full(p.shape, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    if not ok.any():
        return rej, out_q
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    qv = np.empty(m)
    qv[order] = adj
    out_q[ok] = qv
    rej[ok] = qv <= q
    return rej, out_q


# --------------------------------------------------------------------------
# per-cell comparison
# --------------------------------------------------------------------------


@dataclass
class CellComparison:
    """The full evidential state of one cell of the envelope."""

    cell: Cell
    coords: dict[str, Any]
    effect: float = float("nan")          # log-ratio, >0 = method worse
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    p_worse: float = float("nan")
    p_better: float = float("nan")
    q_worse: float = float("nan")
    q_better: float = float("nan")
    p_equiv: float = float("nan")         # TOST
    verdict: Verdict = "inconclusive"
    n_method: int = 0
    n_baseline: int = 0
    method_point: float = float("nan")
    baseline_point: float = float("nan")
    paired: bool = False
    reason: str | None = None             # populated for `missing`
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pct(self) -> float:
        """Regression as a percentage; positive means worse."""
        return (math.exp(self.effect) - 1.0) * 100.0 if math.isfinite(self.effect) else float("nan")

    @property
    def resolution(self) -> float:
        """Half-width of the interval: the smallest effect this cell can see."""
        if not (math.isfinite(self.ci_lo) and math.isfinite(self.ci_hi)):
            return float("inf")
        return (self.ci_hi - self.ci_lo) / 2.0

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["cell"] = list(self.cell)
        d["pct"] = self.pct
        d["resolution"] = self.resolution
        return d


def _oriented_logs(env: Envelope, system: str, cell: Cell) -> np.ndarray:
    v = env.replicates_at(system, cell)
    v = v[v > 0]
    return np.log(v)


def compare_cells(
    env: Envelope,
    method: str,
    baseline: str,
    *,
    mde: float = 0.05,
    q: float = 0.05,
    alpha: float = 0.05,
    n_boot: int = 4000,
    paired: bool | None = None,
    seed: int = 0,
    treat_missing_as_loss: bool = True,
) -> list[CellComparison]:
    """Compare two systems across an envelope, with FDR control over cells.

    ``mde`` is the minimum effect worth calling, as a fraction (0.05 = 5%). It
    must come from the pre-registration, not from the data: choosing it after
    seeing results is the exact move the standard exists to block, and
    :mod:`losscolumn.core.prereg` binds it to the sealed protocol.
    """
    if paired is None:
        paired = env.interleaved
    rng = np.random.default_rng(seed)
    sign = -1.0 if env.metric.higher_is_better else 1.0
    lm = math.log1p(mde)
    out: list[CellComparison] = []

    for cell in env.cells():
        c = CellComparison(cell=cell, coords=env.coords(cell), paired=bool(paired))
        a = _oriented_logs(env, method, cell)
        b = _oriented_logs(env, baseline, cell)
        c.n_method, c.n_baseline = int(a.size), int(b.size)

        if a.size == 0 or b.size == 0:
            reason = env.missing.get(method, {}).get(cell) or env.missing.get(baseline, {}).get(cell)
            c.reason = reason or "no measurements"
            # A cell the method cannot run at all is a loss, and the harshest
            # kind. It is only excluded when the baseline also failed there.
            if treat_missing_as_loss and a.size == 0 and b.size > 0:
                c.verdict = "loss"
                c.effect = float("inf")
            else:
                c.verdict = "missing"
            out.append(c)
            continue

        c.method_point = float(np.exp(np.median(a)))
        c.baseline_point = float(np.exp(np.median(b)))

        if paired and a.size == b.size:
            d = sign * (a - b)
            c.effect = float(np.mean(d))
            c.ci_lo, c.ci_hi = bca_ci(d, np.mean, alpha=alpha, n_boot=n_boot, rng=rng)
            c.p_worse = signflip_p(d, alternative="greater", rng=rng)
            c.p_better = signflip_p(d, alternative="less", rng=rng)
            # TOST. Equivalence requires rejecting BOTH one-sided nulls:
            #   H0_upper: effect >= +mde   rejected when (d - mde) is < 0
            #   H0_lower: effect <= -mde   rejected when (d + mde) is > 0
            # The alternatives point inward, toward zero; pointing them outward
            # is a subtle inversion that makes equivalence essentially
            # unprovable, so the direction is spelled out here.
            p_upper = signflip_p(d - lm, alternative="less", rng=rng)
            p_lower = signflip_p(d + lm, alternative="greater", rng=rng)
            c.p_equiv = float(max(p_upper, p_lower))
        else:
            c.effect = float(sign * (np.mean(a) - np.mean(b)))
            diff_boot = _unpaired_boot(a, b, sign, n_boot, rng)
            c.ci_lo, c.ci_hi = (
                float(np.quantile(diff_boot, alpha / 2)),
                float(np.quantile(diff_boot, 1 - alpha / 2)),
            )
            aa, bb = (a, b) if sign > 0 else (-a, -b)
            c.p_worse, _ = welch_p(aa, bb, alternative="greater")
            c.p_better, _ = welch_p(aa, bb, alternative="less")
            p_upper, _ = welch_p(aa - lm, bb, alternative="less")
            p_lower, _ = welch_p(aa + lm, bb, alternative="greater")
            c.p_equiv = float(max(p_upper, p_lower))
        out.append(c)

    live = [c for c in out if math.isfinite(c.p_worse)]
    rej_w, qw = bh_fdr([c.p_worse for c in live], q=q)
    rej_b, qb = bh_fdr([c.p_better for c in live], q=q)
    for c, w, b_, rw, rb in zip(live, qw, qb, rej_w, rej_b, strict=False):
        c.q_worse, c.q_better = float(w), float(b_)
        c.verdict = _classify(c, rw, rb, mde=lm, alpha=alpha)
    return out


def _unpaired_boot(a: np.ndarray, b: np.ndarray, sign: float, n_boot: int,
                   rng: np.random.Generator) -> np.ndarray:
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    return sign * (a[ia].mean(axis=1) - b[ib].mean(axis=1))


def _classify(c: CellComparison, rej_worse: bool, rej_better: bool, *, mde: float,
              alpha: float) -> Verdict:
    """Turn evidence into one of the five verdicts.

    Significance alone is not enough in either direction: a 0.4% regression
    that is significant at r=11 is not a loss anyone should act on, and the
    standard would lose its meaning if the loss column filled up with them.
    Significance AND a practically relevant effect are both required.
    """
    if rej_worse and c.effect >= mde:
        return "loss"
    if rej_better and c.effect <= -mde:
        return "win"
    # Equivalence has to be demonstrated, not assumed from a non-significant test.
    if math.isfinite(c.p_equiv) and c.p_equiv <= alpha and abs(c.effect) < mde:
        return "tie"
    return "inconclusive"


def verdict_counts(cmps: Sequence[CellComparison]) -> dict[str, int]:
    out: dict[str, int] = {"loss": 0, "win": 0, "tie": 0, "inconclusive": 0, "missing": 0}
    for c in cmps:
        out[c.verdict] = out.get(c.verdict, 0) + 1
    return out
