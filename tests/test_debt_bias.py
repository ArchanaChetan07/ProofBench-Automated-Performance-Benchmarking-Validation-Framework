"""The two-sample CV correction, and what it moves.

Six of sixteen uncovered cells changed kind under this correction, all of them
from model-limited to noise-limited. The two kinds have opposite remedies -- one
sends you to build a model family, the other says no model can help -- so an
estimator that biases the classification one way is not a detail.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from losscolumn.core.debt import CV_N2_BIAS, DebtKind, debias_cv


def test_two_sample_cv_understates_dispersion():
    """The bias is a property of the estimator, reproducible from any sample.

    Drawn here rather than asserted: a CV computed from two points divides by
    two rather than one, and a two-point range is a poor sample of a
    distribution's spread even after that.
    """
    rng = np.random.default_rng(4)
    truth = 0.25
    full, pairs = [], []
    for _ in range(400):
        x = rng.normal(1.0, truth, size=7)
        full.append(x.std() / x.mean())
        pairs += [x[[i, j]].std() / x[[i, j]].mean()
                  for i, j in itertools.combinations(range(7), 2)]
    ratio = float(np.median(pairs) / np.median(full))
    assert ratio == pytest.approx(CV_N2_BIAS, abs=0.08), (
        f"the registered bias {CV_N2_BIAS} should reproduce on synthetic data")
    assert ratio < 1.0, "a two-sample CV must read cleaner than the truth"


def test_debias_leaves_well_sampled_estimates_alone():
    for n in (5, 7, 11, 30):
        assert debias_cv(0.20, n) == 0.20


def test_debias_inflates_a_two_sample_estimate():
    assert debias_cv(0.058, 2) == pytest.approx(0.058 / CV_N2_BIAS, rel=1e-9)
    assert debias_cv(0.058, 2) > 0.10


def test_debias_passes_nan_through():
    assert math.isnan(debias_cv(float("nan"), 2))


class _Regime:
    def __init__(self, regime, err, cv, n=20, nval=8):
        self.regime, self.heldout_err, self.noise_cv = regime, err, cv
        self.n_points, self.n_validation_points = n, nval
        self.covered = False
        self.status = type("S", (), {"value": "uncovered"})()


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


def test_the_correction_changes_a_cells_kind():
    """A cell that reads model-limited uncorrected and noise-limited corrected.

    Held-out error 19.7% against a recorded 10.5% CV looks like a clean
    measurement the model cannot fit. The same CV, corrected for having come
    from two samples, is 19.3% -- and the residual is then the instrument's.
    """
    from losscolumn.core.debt import assess_debt

    cov = _Cov([_Group("all_gather/world2", [_Regime("medium", 0.197, 0.105)])])

    naive = assess_debt(cov, {})
    assert naive.items[0].kind is DebtKind.MODEL_LIMITED

    fixed = assess_debt(cov, {}, cv_from_n=2)
    assert fixed.items[0].kind is DebtKind.NOISE_LIMITED
    assert fixed.items[0].noise_cv == pytest.approx(0.105 / CV_N2_BIAS, rel=1e-6)


def test_the_correction_does_not_rescue_every_cell():
    """A genuine modelling gap survives it: 20.2% error against 7.4% corrected."""
    from losscolumn.core.debt import assess_debt

    cov = _Cov([_Group("all_gather/world2", [_Regime("small", 0.202, 0.0403)])])
    fixed = assess_debt(cov, {}, cv_from_n=2)
    assert fixed.items[0].kind is DebtKind.MODEL_LIMITED, (
        "the correction must not turn every debt into noise")


def test_the_correction_is_recorded_in_the_ledger():
    """A reader must be able to see that a correction was applied."""
    from losscolumn.core.debt import assess_debt

    cov = _Cov([_Group("g", [_Regime("medium", 0.197, 0.105)])])
    assert any("corrected" in n for n in assess_debt(cov, {}, cv_from_n=2).notes)
    assert not assess_debt(cov, {}).notes


def test_noise_and_model_limited_have_opposite_remedies():
    """Why the misclassification mattered."""
    assert "No modelling work will help" in DebtKind.NOISE_LIMITED.remedy
    assert "More repeats will not help" in DebtKind.MODEL_LIMITED.remedy
