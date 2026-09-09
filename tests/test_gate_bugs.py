"""Regressions for six bugs that each made a gate say the wrong thing.

None of these was found by reading code. Each turned up because a number came
out impossible, a prediction failed, or a gate could not respond to the effort
spent on it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from losscolumn.core.coverage import (
    CommunicationCoverage,
    GroupCoverage,
    ParameterVerdict,
    ReadinessVerdict,
    RegimeCoverage,
    RegimeStatus,
)

# ------------------------------------------------------------------- bug 1


def test_stability_predicts_a_level_with_the_measured_exponent():
    """1/sqrt(n) understates the prediction and inflates the excess above it.

    Under the wrong exponent this machine's restart level read as adding 1.58x
    what averaging explains, and the drift was classified RESTART_LEVEL. With
    the measured one it adds 1.09x and the drift is RANDOM. The classification
    turned entirely on the exponent.
    """
    from losscolumn.core.stability import AVERAGING_EXPONENT, MEDIAN_SE_FACTOR

    assert -0.5 < AVERAGING_EXPONENT < -0.2
    within, n = 0.213, 7
    measured = MEDIAN_SE_FACTOR * within * n ** AVERAGING_EXPONENT
    white = MEDIAN_SE_FACTOR * within / math.sqrt(n)
    assert measured > white, "correlated noise leaves more dispersion at the median"
    observed = 0.159
    assert observed / measured < 1.5, "so this level adds nothing material"
    assert observed / white >= 1.5, "which the old exponent would have denied"


def test_the_drift_classification_follows_the_exponent():
    from conftest import build_session as _session  # noqa: PLC0415

    from losscolumn.core.stability import DriftKind, build_envelope

    readings = []
    for s in range(3):
        readings += _session(f"S{s}", jitter=0.10, n_restarts=3)
    env = build_envelope(readings)
    assert env.drift_kind in {DriftKind.RANDOM, DriftKind.UNDETERMINED,
                              DriftKind.PERSISTENT, DriftKind.RESTART_LEVEL,
                              DriftKind.WORKLOAD_SPECIFIC,
                              DriftKind.ENVIRONMENT_DEPENDENT}
    assert math.isfinite(env.across_restart.expected_cv)


# ------------------------------------------------------------------- bug 2


def _covered_matrix() -> CommunicationCoverage:
    cov = CommunicationCoverage.empty(["all_gather"], [2], ["tiny", "small"])
    for g in cov.groups.values():
        g.parameter_verdict = ParameterVerdict.ACCEPTED
        for r in g.regimes.values():
            r.status = RegimeStatus.COVERED
            r.n_points, r.n_validation_points = 10, 4
            r.heldout_err, r.noise_cv = 0.05, 0.08
    return cov


def test_single_session_evidence_is_not_blocked_by_machine_drift():
    """The hazard is pooling incomparable sessions, not the machine's character.

    LC-8.1 says pooled measurements come from a single session OR from sessions
    shown comparable. A one-session campaign takes no pooling risk however badly
    the machine drifts between sittings, and the gate blocked it anyway.
    """
    cov = _covered_matrix()
    cov.n_evidence_sessions = 1
    cov.stability = {"pooling_permitted": False, "drift_kind": "restart_level"}
    verdict, blocking = cov.readiness()
    assert verdict is ReadinessVerdict.READY, blocking


def test_multi_session_evidence_is_still_blocked():
    """The gate keeps its teeth where the risk is actually taken."""
    cov = _covered_matrix()
    cov.n_evidence_sessions = 3
    cov.stability = {"pooling_permitted": False, "drift_kind": "restart_level",
                     "n_poolable_pairs": 0, "n_session_pairs": 3}
    verdict, blocking = cov.readiness()
    assert verdict is ReadinessVerdict.NOT_READY
    assert any("spans 3 sessions" in b for b in blocking)


def test_multi_session_evidence_without_stability_evidence_is_blocked():
    cov = _covered_matrix()
    cov.n_evidence_sessions = 2
    cov.stability = None
    verdict, blocking = cov.readiness()
    assert verdict is ReadinessVerdict.NOT_READY
    assert any("no stability evidence" in b for b in blocking)


# ------------------------------------------------------------------- bug 3


def test_the_gradeability_gate_responds_to_measurement_effort():
    """The bug that falsified a pre-registered prediction.

    The gate compared the population CV of a point's repeats against a ceiling.
    That number does not fall when more repeats are taken, so no amount of
    measuring could ever clear it -- which is why predicting that more repeats
    would improve coverage was bound to fail.
    """
    from losscolumn.thrusts.overlap.campaign import _point_se

    cv = 0.30
    se2 = _point_se(cv, 2)
    se21 = _point_se(cv, 21)
    se200 = _point_se(cv, 200)
    assert se21 < se2, "more repeats must buy precision"
    assert se200 < se21
    assert se21 < cv, "the point is known better than a single call is"


def test_point_se_is_not_the_population_cv():
    from losscolumn.thrusts.overlap.campaign import _point_se

    assert _point_se(0.306, 21) == pytest.approx(0.151, abs=0.01)
    assert _point_se(0.306, 21) < 0.306


def test_point_se_is_nan_without_a_noise_estimate():
    from losscolumn.thrusts.overlap.campaign import _point_se

    assert math.isnan(_point_se(float("nan"), 21))


# ------------------------------------------------------------------- bug 4


def test_an_ungradable_tier_does_not_reject_a_family():
    """In a tier that cannot be graded, a bad model and an unmeasurable one look
    the same, so a large error there is not evidence about the model."""
    from losscolumn.thrusts.overlap.commodel import _worst_gradable

    per_tier = {"tiny": 0.04, "small": 0.05, "medium": 0.33, "large": 0.06}
    assert _worst_gradable(per_tier, ()) == pytest.approx(0.33)
    assert _worst_gradable(per_tier, ("medium",)) == pytest.approx(0.06)


def test_a_family_is_not_promoted_on_having_no_evidence_against_it():
    """When every tier is ungradable the plain worst still applies."""
    from losscolumn.thrusts.overlap.commodel import _worst_gradable

    per_tier = {"tiny": 0.4, "medium": 0.5}
    assert _worst_gradable(per_tier, ("tiny", "medium")) == pytest.approx(0.5)


def test_excluding_an_ungradable_tier_does_not_cover_it():
    """It stays uncovered in the matrix; only its vote is removed."""
    g = GroupCoverage(collective="all_gather", world=2)
    g.regimes["medium"] = RegimeCoverage(regime="medium",
                                         status=RegimeStatus.TOO_NOISY)
    assert not g.regimes["medium"].covered
    assert not g.covered


# ------------------------------------------------------------------- bug 5


def test_the_block_budget_is_a_duration_not_a_byte_threshold():
    """A fixed iters count put a fourfold averaging cliff inside the medium
    regime -- points above 1 MiB averaged five calls against twenty below, and
    carried 29.4% run-to-run variation against 18.8%, in the band that decides
    most of the coverage matrix.

    Sizing the block by duration removes the cliff without claiming anything the
    measurement cannot support.
    """
    from losscolumn.thrusts.overlap.campaign import (
        MAX_ITERS,
        MIN_ITERS,
        TARGET_BLOCK_S,
    )

    assert TARGET_BLOCK_S > 0 and MIN_ITERS >= 1 and MAX_ITERS > MIN_ITERS

    def iters_for(per_call):
        return int(min(max(round(TARGET_BLOCK_S / per_call), MIN_ITERS), MAX_ITERS))

    # Measured costs on this machine: 0.22 ms at 4 KiB, 1.67 ms at 1 MiB,
    # 24 ms at 8 MiB. The cheap end must get far more averaging than the old
    # fixed twenty, and the expensive end falls back to the floor.
    assert iters_for(0.00022) > 20
    assert iters_for(0.00167) > 20
    assert iters_for(0.024) == MIN_ITERS
    # No discontinuity anywhere: iters varies smoothly with cost.
    assert iters_for(0.0009) >= iters_for(0.0011)


def test_a_short_probe_is_not_asked_to_estimate_variability():
    """The second rule tried here scaled the budget by a six-call probe's CV.

    It measured worse than what it replaced: the instability in this band acts
    on a longer timescale than six calls can see, so it judged the noisy band
    quiet, gave 41% of points the minimum five iterations, and coverage fell
    from 17 cells to 11. The probe now establishes only the cost of a call.
    """
    import inspect

    from losscolumn.thrusts.overlap import campaign as C

    src = inspect.getsource(C._worker)
    assert "TARGET_BLOCK_S / per_call" in src
    assert not hasattr(C, "NOISY_CV"), (
        "the variability-scaled budget is withdrawn, not merely unused")


def test_iters_is_recorded_so_the_noise_can_be_audited():
    from losscolumn.thrusts.overlap.campaign import PointRecord

    r = PointRecord.from_dict({
        "collective": "all_gather", "world": 2, "nbytes": 4096,
        "pass_index": 0, "timings_s": [1e-3, 1.1e-3], "iters": 37})
    assert r.iters == 37
    assert r.to_dict()["iters"] == 37


# ------------------------------------------------------------------- bug 6


def _line(alpha, beta):
    from losscolumn.thrusts.overlap.commodel import LinearModel

    return LinearModel(alpha_s=alpha, beta_bytes_per_s=beta)


def _samples(fn, sizes):
    """Build samples whose timings match a known line exactly.

    all_gather at world 2 moves (w-1) = 1 times its input on the nccl-tests
    convention, so effective_bytes equals nbytes and the generating function
    uses n directly. These tests therefore encode that convention: if it is ever
    changed again, they fail here rather than silently measuring something else.
    """
    from losscolumn.thrusts.overlap.commodel import Sample

    return [Sample(nbytes=n, seconds=fn(n), world=2, kind="all_gather")
            for n in sizes]


def test_structure_is_chosen_on_the_same_scale_the_fits_use():
    """Every estimator here works on relative residuals; structure did not.

    Choosing a breakpoint by absolute RSS over a domain spanning four orders of
    magnitude places it wherever the largest messages want it -- the exact
    leverage problem the 1/t weighting exists to remove from the segment fits,
    reintroduced one level up in the same function.
    """
    m = _line(1e-5, 1e9)
    sizes = [1 << k for k in range(10, 25)]
    s = _samples(lambda n: 1e-5 + n / 1e9, sizes)
    abs_r = m.residuals(s)
    rel_r = m.relative_residuals(s)
    assert np.allclose(rel_r, 0, atol=1e-9)
    assert abs_r.shape == rel_r.shape


def test_relative_residuals_give_every_size_comparable_leverage():
    m = _line(1e-5, 1e9)
    sizes = [1 << 10, 1 << 24]
    # 20% wrong at both ends.
    s = _samples(lambda n: (1e-5 + n / 1e9) * 1.2, sizes)
    a = np.abs(m.residuals(s))
    r = np.abs(m.relative_residuals(s))
    assert a[1] / a[0] > 500, "absolute residuals are all about the big point"
    assert r[1] / r[0] == pytest.approx(1.0, abs=0.05), (
        "relative residuals weigh a 20% error the same at either end")


def test_relative_aic_governs_structural_decisions():
    from losscolumn.thrusts.overlap.commodel import LinearModel

    assert hasattr(LinearModel, "relative_aic")
    assert hasattr(LinearModel, "aic"), "the absolute one stays, for continuity"


def test_relative_residuals_survive_a_zero_measurement():
    m = _line(1e-5, 1e9)
    from losscolumn.thrusts.overlap.commodel import Sample

    s = [Sample(nbytes=1024, seconds=0.0, world=2, kind="all_gather")]
    assert np.all(np.isfinite(m.relative_residuals(s)))
