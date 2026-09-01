"""The noise budget, and what it says a gate costs.

The load-bearing idea is that measurement variance splits into a part that
averages down with repeats and a part that does not. Getting that split wrong in
either direction is expensive: understate the floor and you promise a precision
the machine cannot deliver, overstate it and you abandon a surface that was
reachable all along.
"""

from __future__ import annotations

import math

import pytest

from losscolumn.core.modellability import (
    MEDIAN_SE_FACTOR,
    NoiseBudget,
    assess_modellability,
    decompose_noise,
    plan_replicates,
)


def _budget(within=0.213, restart=0.159, n_rep=7, n_launch=3) -> NoiseBudget:
    return NoiseBudget(within_run_cv=within, across_restart_cv=restart,
                       n_repeats=n_rep, n_launches=n_launch)


# ------------------------------------------------------------------ the split


def test_expected_restart_cv_is_what_averaging_alone_predicts():
    b = _budget()
    assert b.expected_restart_cv == pytest.approx(
        MEDIAN_SE_FACTOR * 0.213 / math.sqrt(7), rel=1e-9)
    assert b.expected_restart_cv < b.within_run_cv, (
        "a median of seven must disperse less than the seven")


def test_launch_component_subtracts_variances_not_coefficients():
    """The arithmetic that decides whether a gate is reachable.

    Subtracting the CVs directly would give 15.9% - 10.1% = 5.8% and understate
    the floor by half, which is the difference between promising a 10% gate and
    correctly refusing it.
    """
    b = _budget()
    naive = b.across_restart_cv - b.expected_restart_cv
    assert b.launch_cv == pytest.approx(
        math.sqrt(0.159 ** 2 - b.expected_restart_cv ** 2), rel=1e-9)
    assert b.launch_cv > naive * 1.5, "variances, not coefficients"


def test_launch_component_is_zero_when_restarts_add_nothing():
    """Never imaginary. A level quieter than predicted contributes nothing."""
    b = _budget(within=0.20, restart=0.02)
    assert b.launch_cv == 0.0
    assert b.floor == 0.0


def test_repeats_cannot_go_below_the_launch_floor():
    b = _budget()
    assert b.point_noise(100_000, 1) == pytest.approx(b.floor, abs=1e-3)
    assert b.point_noise(1_000_000, 1) >= b.floor


def test_more_launches_do_lower_the_floor():
    b = _budget()
    assert b.point_noise(50, 4) < b.point_noise(50, 1)


def test_decompose_reads_a_measured_envelope():
    env = {"within_run": {"cv": 0.213, "n_aggregated": 7},
           "across_restart": {"cv": 0.159, "n_aggregated": 3}}
    b = decompose_noise(env)
    assert b.within_run_cv == 0.213 and b.n_repeats == 7
    assert b.launch_cv == pytest.approx(_budget().launch_cv, rel=1e-9)


# -------------------------------------------------------------------- costing


def test_a_target_above_the_floor_is_bought_with_repeats_alone():
    p = plan_replicates(_budget(), 0.15)
    assert p.reachable_by_repeats_alone and p.launches_needed == 1
    assert p.repeats_needed > 1
    assert _budget().point_noise(p.repeats_needed, 1) <= 0.15


def test_the_plan_is_the_cheapest_one_that_works():
    b = _budget()
    p = plan_replicates(b, 0.15)
    assert b.point_noise(p.repeats_needed - 1, 1) > 0.15, (
        "one repeat fewer must miss the target")


def test_a_target_below_the_floor_needs_more_launches():
    b = _budget()
    p = plan_replicates(b, 0.10)
    assert p.target < b.floor
    assert not p.reachable_by_repeats_alone
    assert p.launches_needed > 1
    assert "does not average down inside one process" in p.note


def test_an_impossible_target_is_reported_as_such():
    p = plan_replicates(_budget(), 0.001, max_launches=4)
    assert not p.reachable_at_all


def test_no_noise_estimate_yields_no_plan():
    p = plan_replicates(NoiseBudget(), 0.15)
    assert not p.reachable_at_all and "no noise estimate" in p.note


# ---------------------------------------------------------------- the verdict


ENV = {"within_run": {"cv": 0.213, "n_aggregated": 7},
       "across_restart": {"cv": 0.159, "n_aggregated": 3}}


def _debt(items):
    return {"items": [
        {"group": g, "regime": r, "kind": k, "heldout_err": e, "noise_cv": c}
        for g, r, k, e, c in items]}


def test_a_clean_cell_with_a_large_residual_is_model_work():
    m = assess_modellability(
        ENV, _debt([("g/w2", "small", "model_limited", 0.202, 0.074)]))
    assert m.cells[0].verdict == "MODEL_WORK"


def test_a_noisy_cell_is_costed_rather_than_declared_unreachable():
    """The bug this guards: `cv >= gate` is not the same as out of reach.

    A cell with 29.5% run-to-run variation reads as hopeless against a 15% gate
    until you notice that repeats shrink it as 1/sqrt(n). It is expensive, not
    impossible, and the two call for different decisions.
    """
    m = assess_modellability(
        ENV, _debt([("g/w2", "medium", "noise_limited", 0.353, 0.295)]))
    c = m.cells[0]
    assert c.verdict != "UNREACHABLE"
    assert c.repeats_needed > 10, "a 29.5% cell needs real replication"
    assert c.launches_needed == 1, "and the floor does not force extra launches"


def test_a_cell_below_the_floor_is_genuinely_unreachable():
    """Where the floor defeats it, no replicate count helps."""
    m = assess_modellability(
        ENV, _debt([("g/w2", "medium", "noise_limited", 0.40, 0.35)]),
        gate=0.05, max_launches=2)
    assert m.cells[0].verdict == "UNREACHABLE"


def test_the_floor_alone_can_settle_the_whole_question():
    """A gate under the launch floor is unreachable regardless of any cell."""
    m = assess_modellability(ENV, _debt([]), gate=0.05)
    assert m.verdict == "NOT_MODELLABLE_AT_THIS_GATE"
    assert "converge on that floor and stop" in m.summary


def test_a_reachable_surface_is_reported_as_such():
    m = assess_modellability(
        ENV, _debt([("a/w2", "tiny", "noise_limited", 0.09, 0.10),
                    ("b/w3", "small", "noise_limited", 0.04, 0.14)]))
    assert m.verdict == "MODELLABLE"
    assert all(c.verdict == "BUYABLE" for c in m.cells)


def test_verdict_separates_measurement_work_from_model_work():
    m = assess_modellability(
        ENV, _debt([("a/w2", "tiny", "noise_limited", 0.09, 0.10),
                    ("b/w2", "small", "model_limited", 0.202, 0.074)]))
    assert m.verdict == "MODELLABLE_WITH_WORK"
    assert m.counts == {"BUYABLE": 1, "MODEL_WORK": 1}


def test_an_empty_debt_is_not_a_crash():
    m = assess_modellability({}, {"items": []})
    assert m.cells == []


# ------------------------------------------------- the A100 protocol's stages


def test_a100_stage_numbers_are_contiguous_and_ordered():
    """Inserting the stability sentinel renumbered everything after it.

    A protocol whose stages are referred to by number, and whose numbers drift
    when one is inserted, will eventually say 'stage 9 seals the prediction'
    about a stage that does something else.
    """
    from losscolumn.thrusts.overlap.a100_protocol import STAGES

    assert [s.order for s in STAGES] == list(range(1, len(STAGES) + 1))
    assert len({s.name for s in STAGES}) == len(STAGES)


def test_the_stability_sentinel_precedes_the_calibration_it_sizes():
    """It exists to set the replicate count, so it cannot run afterwards."""
    from losscolumn.thrusts.overlap.a100_protocol import STAGES

    by = {s.name: s.order for s in STAGES}
    assert by["machine stability sentinel"] < by["communication calibration"]
    assert by["machine stability sentinel"] < by["parameter quality gates"]


def test_prediction_is_sealed_before_measurement():
    from losscolumn.thrusts.overlap.a100_protocol import STAGES

    by = {s.name: s.order for s in STAGES}
    assert by["prediction lock"] < by["independent 8-GPU measurement"]
    assert by["Thrust I prediction generation"] < by["prediction lock"]


def test_every_numeric_stage_reference_names_a_real_stage():
    """Guards the stale cross-reference a renumbering leaves behind."""
    import re

    from losscolumn.thrusts.overlap.a100_protocol import STAGES

    valid = {s.order for s in STAGES}
    src = open("src/losscolumn/thrusts/overlap/a100_protocol.py",
               encoding="utf-8").read()
    for n in re.findall(r"stages? (\d+)", src, flags=re.IGNORECASE):
        assert int(n) in valid, f"reference to stage {n}, which does not exist"


def test_the_budget_names_what_an_overrun_may_not_buy():
    """An overrun must cost scope, never the gate."""
    from losscolumn.thrusts.overlap.a100_protocol import A100Protocol

    d = A100Protocol(local_readiness="not_ready", local_campaign_seal="x",
                     standard_version="LC-1.2").to_dict()
    b = d["budget"]
    assert b["hard_ceiling_hours"] >= b["estimated_hours"]
    assert any("replicates" in x for x in b["what_overrun_must_not_buy"])
    assert any("gate" in x for x in b["what_overrun_must_not_buy"])
