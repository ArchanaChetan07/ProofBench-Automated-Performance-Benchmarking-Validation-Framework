"""Noise estimates, and why a two-sample CV cannot be corrected into one.

This replaces a suite that verified a correction factor. The factor was
withdrawn after a pre-registered re-measurement falsified it; what is tested now
is the refusal that took its place, and the finding that made the correction
impossible rather than merely mis-estimated.

The measured facts pinned here come from one session of the 21-repeat
recampaign, so nothing below depends on pooling across sessions.
"""

from __future__ import annotations

import math

import pytest

from losscolumn.core.debt import (
    MIN_TRUSTWORTHY_REPEATS,
    DebtKind,
    UntrustworthyNoiseEstimate,
    assess_debt,
)


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


def _cov():
    return _Cov([_Group("all_gather/world2", [_Regime("medium", 0.197, 0.105)])])


# --------------------------------------------------------------- the refusal


def test_a_two_repeat_noise_estimate_is_refused():
    """Not corrected. Refused.

    A CV from two adjacent repeats measures how much one call differs from its
    neighbour. A coverage gate asks how much the call varies over a run. Those
    are different quantities and no factor converts between them, which is what
    the withdrawn correction assumed.
    """
    with pytest.raises(UntrustworthyNoiseEstimate, match="below the"):
        assess_debt(_cov(), {}, cv_from_n=2)


def test_the_refusal_names_the_preserved_ledger():
    """So a reader can check the failure it prevents rather than take it on trust."""
    with pytest.raises(UntrustworthyNoiseEstimate) as e:
        assess_debt(_cov(), {}, cv_from_n=2)
    assert "2026-08-31-two-sample-cv-correction" in str(e.value)


def test_an_untrustworthy_estimate_can_be_admitted_but_must_be_declared():
    d = assess_debt(_cov(), {}, cv_from_n=2, allow_untrustworthy=True)
    assert any("below the trustworthy threshold" in n for n in d.notes)


def test_a_well_sampled_estimate_passes_without_ceremony():
    d = assess_debt(_cov(), {}, cv_from_n=21)
    assert d.items
    assert not any("trustworthy threshold" in n for n in d.notes)


def test_no_correction_is_applied_to_a_recorded_cv():
    """The classification must see the number that was actually measured."""
    d = assess_debt(_cov(), {}, cv_from_n=21)
    assert d.items[0].noise_cv == pytest.approx(0.105)


def test_the_threshold_sits_above_two():
    assert MIN_TRUSTWORTHY_REPEATS > 2


def test_silence_about_the_repeat_count_is_not_a_claim_about_it():
    d = assess_debt(_cov(), {})
    assert d.items and not d.notes


# ------------------------------------- what made the correction impossible


def test_measured_averaging_exponent_is_not_one_half():
    """SE of a median of n falls as n^-0.31 here, not n^-0.5.

    Measured from 5520 disjoint block pairs inside one session of the 21-repeat
    recampaign. Correlated noise, not white: halving an error costs nine times
    the repeats rather than four, and a model assuming 1/sqrt(n) promises a
    precision that more measurement will not deliver.
    """
    from losscolumn.core.modellability import MEASURED_AVERAGING_EXPONENT

    assert -0.5 < MEASURED_AVERAGING_EXPONENT < -0.2


def test_halving_the_error_costs_more_than_four_times_the_repeats():
    from losscolumn.core.modellability import MEASURED_AVERAGING_EXPONENT

    n_to_halve = 2 ** (1.0 / -MEASURED_AVERAGING_EXPONENT)
    assert math.isfinite(n_to_halve)
    assert n_to_halve > 4.0, "white noise halves at 4x; correlated noise costs more"


def test_the_withdrawn_constant_is_kept_only_for_reference():
    """It must remain readable, so the withdrawal can be read against it."""
    from losscolumn.core import debt

    assert hasattr(debt, "CV_N2_BIAS_WITHDRAWN")
    assert not hasattr(debt, "CV_N2_BIAS"), "the live name must not come back"
    assert not hasattr(debt, "debias_cv"), "nor the function that applied it"


def test_noise_and_model_limited_still_have_opposite_remedies():
    """Why misclassifying between them was worth withdrawing a correction over."""
    assert "No modelling work will help" in DebtKind.NOISE_LIMITED.remedy
    assert "More repeats will not help" in DebtKind.MODEL_LIMITED.remedy
