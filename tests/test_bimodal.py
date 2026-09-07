"""Detecting a configuration that has two behaviours rather than one.

The medium regime resisted every remedy: more repeats, more segments, a better
loss, a corrected noise gate. It resisted them because the obstacle was not
noise or flexibility. Between 512 KiB and 1 MiB the gloo all_reduce at world 2
takes one of two paths, and at 1 MiB the median steps by a factor of eight.

A single-valued cost model cannot describe that, and the tests below pin both
the detection and the reason the usual remedies do not apply.
"""

from __future__ import annotations

import numpy as np
import pytest

from losscolumn.core.bimodal import (
    MIN_SAMPLES,
    detect_bimodal,
    find_threshold_band,
)


class _Rec:
    def __init__(self, nbytes, timings, valid=True):
        self.nbytes, self.timings_s, self.valid = nbytes, list(timings), valid


def _two_modes(lo, hi, n_lo, n_hi, jitter=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return ([lo * (1 + jitter * rng.standard_normal()) for _ in range(n_lo)]
            + [hi * (1 + jitter * rng.standard_normal()) for _ in range(n_hi)])


# ---------------------------------------------------------------- detection


def test_two_clear_modes_are_detected():
    ms = detect_bimodal(_two_modes(1e-3, 4e-3, 8, 13), nbytes=1 << 19)
    assert ms.bimodal
    assert ms.separation == pytest.approx(4.0, rel=0.1)
    assert ms.minority_share == pytest.approx(8 / 21, rel=0.1)


def test_a_wide_unimodal_spread_is_not_called_bimodal():
    """The conservative case that matters: timing noise is wide and skewed.

    Calling every noisy cell bimodal would make the label useless, and would
    excuse real modelling failures as unmodellable.
    """
    rng = np.random.default_rng(7)
    t = np.abs(rng.lognormal(mean=np.log(1e-3), sigma=0.45, size=40))
    assert not detect_bimodal(t).bimodal


def test_a_long_tail_is_not_a_second_mode():
    """Three slow stragglers out of thirty are a tail, which timings have."""
    t = [1e-3] * 27 + [6e-3, 6.2e-3, 6.4e-3]
    ms = detect_bimodal(t)
    assert not ms.bimodal
    assert "stragglers" in ms.reason or "gap" in ms.reason


def test_a_small_separation_is_one_distribution():
    ms = detect_bimodal(_two_modes(1e-3, 1.3e-3, 10, 11))
    assert not ms.bimodal
    assert "1.8" in ms.reason or "wide distribution" in ms.reason


def test_too_few_timings_to_judge():
    ms = detect_bimodal([1e-3, 4e-3, 1e-3])
    assert not ms.bimodal
    assert str(MIN_SAMPLES) in ms.reason


def test_identical_timings_do_not_crash():
    assert not detect_bimodal([1e-3] * 20).bimodal


# ------------------------------------------------------- group-level signals


def _monotone(sizes, alpha=1e-5, beta=1e9, n=21):
    return [_Rec(s, [alpha + s / beta] * n) for s in sizes]


def test_a_clean_surface_shows_no_threshold():
    sizes = [1 << k for k in range(12, 24)]
    rep = find_threshold_band(_monotone(sizes), group="clean")
    assert not rep.has_threshold
    assert rep.band_lo == 0


def test_a_step_is_detected_and_the_largest_one_wins():
    """A small early jump must not mask the threshold that matters."""
    recs = _monotone([1 << k for k in range(12, 24)])
    for r in recs:
        if r.nbytes >= (1 << 20):
            r.timings_s = [t * 8.0 for t in r.timings_s]
    rep = find_threshold_band(recs, group="stepped")
    assert rep.has_threshold
    assert rep.step_at == (1 << 20)
    assert rep.step_factor > 3.0


def test_an_inversion_is_detected():
    """A larger message measured faster cannot happen on a monotone surface."""
    recs = _monotone([1 << k for k in range(12, 20)])
    # Enough of a drop to be outside the tolerance: halving the largest point
    # still leaves it above its neighbour, because the surface is rising.
    recs[-1].timings_s = [t / 4.0 for t in recs[-1].timings_s]
    rep = find_threshold_band(recs, group="inverted")
    assert rep.inversions
    assert rep.has_threshold


def test_inversions_survive_averaging_when_bimodality_does_not():
    """Why the inversion signal is the one to rely on.

    As each timing averages more calls the two modes blend into a mixture mean:
    the point looks unimodal and behaves exactly as badly. The ordering of the
    medians is unaffected, so it still reports the problem.
    """
    # A bistable point whose timings have been averaged into mixture means.
    blended = _monotone([1 << k for k in range(12, 20)])
    for r in blended:
        if r.nbytes == (1 << 17):
            r.timings_s = [t * 6.0 for t in r.timings_s]   # mixture, unimodal
    per_point = [detect_bimodal(r.timings_s) for r in blended]
    assert not any(m.bimodal for m in per_point), (
        "averaging has hidden the two modes, as it must")
    rep = find_threshold_band(blended, group="blended")
    assert rep.has_threshold, "the ordering still gives it away"


def test_the_band_spans_every_flagged_size():
    recs = _monotone([1 << k for k in range(12, 24)])
    for r in recs:
        if r.nbytes >= (1 << 20):
            r.timings_s = [t * 8.0 for t in r.timings_s]
    rep = find_threshold_band(recs, group="g")
    assert rep.band_lo <= (1 << 20) <= rep.band_hi


def test_the_report_says_why_the_usual_remedies_do_not_apply():
    recs = _monotone([1 << k for k in range(12, 24)])
    for r in recs:
        if r.nbytes >= (1 << 20):
            r.timings_s = [t * 8.0 for t in r.timings_s]
    rep = find_threshold_band(recs, group="g")
    joined = " ".join(rep.notes)
    assert "single-valued" in joined
    assert "repeats" in joined and "segments" in joined


def test_no_usable_records_is_reported_not_crashed():
    rep = find_threshold_band([_Rec(1024, [1e-3, 2e-3])], group="g")
    assert not rep.has_threshold
    assert rep.notes


# ------------------------------------------------------------- the status


def test_bimodal_is_a_distinct_status_from_too_noisy():
    """They call for opposite conclusions about whether to measure more."""
    from losscolumn.core.coverage import RegimeStatus

    assert RegimeStatus.BIMODAL.value == "bimodal"
    assert RegimeStatus.BIMODAL is not RegimeStatus.TOO_NOISY


def test_a_bimodal_regime_is_not_covered():
    from losscolumn.core.coverage import RegimeCoverage, RegimeStatus

    rc = RegimeCoverage(regime="medium", status=RegimeStatus.BIMODAL)
    assert not rc.covered


# ------------------------------------------------------ the debt it produces


def _cell(status, regime="medium", err=0.30, cv=0.28, detail=""):
    class _R:
        def __init__(self):
            self.regime, self.heldout_err, self.noise_cv = regime, err, cv
            self.n_points, self.n_validation_points = 20, 8
            self.covered = False
            self.detail = detail
            self.status = type("S", (), {"value": status})()
    return _R()


class _Group:
    def __init__(self, key, regimes):
        self.key, self.regimes = key, {r.regime: r for r in regimes}

    def uncovered(self):
        return list(self.regimes.values())


class _Cov:
    def __init__(self, groups):
        self.groups = {g.key: g for g in groups}
        self.n_required_cells = sum(len(g.regimes) for g in groups)
        self.n_covered_cells = 0


def test_a_bimodal_cell_is_not_filed_as_noise_limited():
    """Filing it as noise sends someone to buy repeats that cannot help."""
    from losscolumn.core.debt import DebtKind, assess_debt

    cov = _Cov([_Group("g/w2", [_cell("bimodal", detail="threshold at 1 MiB")])])
    d = assess_debt(cov, {}, cv_from_n=21)
    assert d.items[0].kind is DebtKind.BIMODAL
    assert "1 MiB" in d.items[0].rationale


def test_the_bimodal_remedy_says_it_is_a_decision_not_work():
    from losscolumn.core.debt import DebtKind

    r = DebtKind.BIMODAL.remedy
    assert "decision" in r
    assert "branch" in r or "excluding" in r


def test_no_experiment_is_proposed_when_only_bimodal_cells_remain():
    """The honest answer to 'what next' is sometimes 'no measurement helps'."""
    from losscolumn.core.debt import assess_debt

    cov = _Cov([_Group("g/w2", [_cell("bimodal")])])
    nxt = assess_debt(cov, {}, cv_from_n=21).next_experiment()
    assert nxt.startswith("None")
    assert "single-valued" in nxt


def test_a_mixed_debt_still_proposes_the_discriminating_experiment():
    """Bimodal cells must not silence advice about the ones that can move."""
    from losscolumn.core.debt import assess_debt

    cov = _Cov([_Group("g/w2", [
        _cell("bimodal"),
        _cell("error_too_high", regime="small", err=0.22, cv=0.06),
    ])])
    nxt = assess_debt(cov, {}, cv_from_n=21).next_experiment()
    assert not nxt.startswith("None")
