"""Adversarial tests for the stability machinery.

Each scenario is a session pair constructed so that exactly one thing is wrong
with it, and the test asserts the machinery notices that thing and not some
other thing. Constructed rather than measured on purpose: a real drifting
machine gives you one example and no control over which defect it exhibits.

The last group pins the world-4 failure that motivated all of this.
"""

from __future__ import annotations

import math

import pytest

from losscolumn.core.pooling import (
    PoolingViolation,
    SessionLedger,
    guard_pool,
)
from losscolumn.core.stability import (
    ComparabilityVerdict,
    DriftKind,
    SentinelReading,
    build_envelope,
    compare_sessions,
    null_from_replicates,
)

# A plausible base surface: time grows with size, all_reduce costs about double.
BASE = {("all_gather", 1 << 16): 1.0e-4, ("all_gather", 1 << 20): 1.2e-3,
        ("all_gather", 1 << 23): 9.0e-3, ("all_reduce", 1 << 16): 1.9e-4,
        ("all_reduce", 1 << 20): 2.3e-3, ("all_reduce", 1 << 23): 1.8e-2}


# Defined in conftest so every module can reach it however pytest is invoked.
# A cross-module import here (`from tests.test_stability import _session`) only
# resolves when the repository root is on sys.path, which is true from the root
# and false from an installed copy -- a rented machine found that on its first run.
from conftest import PROBES, build_session as _session  # noqa: E402, I001

CRITERION = 1.20


# -------------------------------------------------------------------- stable


def test_stable_sessions_are_comparable():
    a, b = _session("A", jitter=0.02), _session("B", jitter=0.02)
    cmp = compare_sessions(a, b, criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.COMPARABLE
    assert cmp.poolable
    assert cmp.observed_ratio < CRITERION
    assert cmp.shape_divergence < CRITERION


def test_stable_sessions_pool_without_complaint():
    led = SessionLedger.from_readings(
        _session("A", jitter=0.02) + _session("B", jitter=0.02), criterion=CRITERION)
    kept, dec = guard_pool(_session("A") + _session("B"), led)
    assert dec.allowed and len(kept) == 36


# ------------------------------------------------------------------- shifted


def test_uniformly_shifted_session_is_incomparable_on_level():
    """Everything 1.45x slower. The surface kept its shape and still cannot pool.

    Attributed to the level gate rather than the magnitude one, which is the
    more precise reading: every probe moved, by the same factor. A pure level
    shift leaves the shape intact, which is exactly why a shape-only check would
    have missed it.
    """
    a, b = _session("A", jitter=0.02), _session("B", scale=1.45, jitter=0.02)
    cmp = compare_sessions(a, b, criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert cmp.failed_gate == "level"
    assert cmp.level_shift == pytest.approx(1.45, rel=0.05)
    assert cmp.observed_ratio == pytest.approx(1.45, rel=0.05)
    assert cmp.shape_divergence == pytest.approx(1.0, abs=0.02)


def test_shifted_session_is_refused_by_the_guard():
    led = SessionLedger.from_readings(
        _session("A", jitter=0.02) + _session("B", scale=1.45, jitter=0.02),
        criterion=CRITERION)
    with pytest.raises(PoolingViolation, match="not poolable"):
        guard_pool(_session("A") + _session("B", scale=1.45), led)


def test_guard_reports_rather_than_raises_when_not_strict():
    led = SessionLedger.from_readings(
        _session("A") + _session("B", scale=1.45), criterion=CRITERION)
    kept, dec = guard_pool(_session("A") + _session("B", scale=1.45), led, strict=False)
    assert kept == [] and not dec.allowed and "worst is A against B" in dec.reason


# ------------------------------------------------------- regime-order change


def test_regime_order_change_is_caught_even_within_the_magnitude_gate():
    """Small messages get slower, large ones faster, each by less than the gate.

    This is the case a magnitude-only check cannot see. No probe moves 1.20x, so
    the worst-ratio test passes, yet the surface has been turned over: what was
    the cheapest probe relative to the others is now dearer.
    """
    def deform(kind, nb):
        return 1.14 if nb <= (1 << 16) else (0.88 if nb >= (1 << 23) else 1.0)

    a, b = _session("A", jitter=0.01), _session("B", scale=deform, jitter=0.01)
    cmp = compare_sessions(a, b, criterion=CRITERION)
    assert cmp.observed_ratio < CRITERION, "magnitude gate alone would have passed this"
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert cmp.failed_gate == "shape"
    assert cmp.shape_divergence == pytest.approx(1.14 / 0.88, rel=0.05)


# ---------------------------------------------- same median, different shape


def test_same_median_different_distribution_is_incomparable():
    """Identical pooled median, opposite shapes. The classic pooling trap.

    Half the probes 1.3x slower, half 1/1.3x faster: any statistic computed on
    the pooled set agrees between the sessions, and the surfaces disagree
    everywhere.
    """
    def swap(kind, nb):
        return 1.30 if kind == "all_gather" else 1.0 / 1.30

    a, b = _session("A", jitter=0.01), _session("B", scale=swap, jitter=0.01)

    import numpy as np
    ma = np.median([r.median_s for r in a])
    mb = np.median([r.median_s for r in b])
    assert abs(ma - mb) / ma < 0.35, "constructed so the pooled medians are close"

    cmp = compare_sessions(a, b, criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert cmp.shape_divergence == pytest.approx(1.30 * 1.30, rel=0.05)


# ------------------------------------------------------------------- outlier


def test_one_outlier_probe_makes_sessions_incomparable():
    """Five probes agree perfectly and one is 2.4x out. The worst probe decides."""
    def one_bad(kind, nb):
        return 2.4 if (kind == "all_reduce" and nb == (1 << 20)) else 1.0

    cmp = compare_sessions(_session("A", jitter=0.01),
                           _session("B", scale=one_bad, jitter=0.01),
                           criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert "all_reduce" in cmp.reason
    # The median probe ratio is 1.0; a median-based rule would have pooled these.
    import numpy as np
    assert np.median(list(cmp.per_probe_ratio.values())) == pytest.approx(1.0, abs=0.05)


def test_outlier_does_not_make_the_measurement_invalid():
    """LC-8.4: valid-but-incomparable is its own state, not a broken measurement."""
    def one_bad(kind, nb):
        return 2.4 if nb == (1 << 20) else 1.0

    b = _session("B", scale=one_bad)
    assert all(r.valid for r in b), "an unusual timing is not an invalid one"
    cmp = compare_sessions(_session("A"), b, criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert cmp.verdict is not ComparabilityVerdict.INVALID


# ------------------------------------------------------------------- invalid


def test_no_valid_readings_is_invalid_not_incomparable():
    a = _session("A")
    dead = _session("B")
    for r in dead:
        r.valid = False
        r.invalid_reason = "rank crashed"
    cmp = compare_sessions(a, dead, criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.INVALID
    assert not cmp.poolable


def test_too_few_shared_probes_is_invalid():
    """Sessions that measured different things cannot be shown comparable."""
    a = _session("A", probes=PROBES[:2])
    b = _session("B", probes=PROBES[4:])
    cmp = compare_sessions(a, b, criterion=CRITERION)
    assert cmp.verdict is ComparabilityVerdict.INVALID
    assert "shared probe" in cmp.reason


def test_session_without_sentinel_evidence_is_invalid_not_comparable():
    led = SessionLedger.from_readings(_session("A"), criterion=CRITERION)
    cmp = led.comparison("A", "Z")
    assert cmp.verdict is ComparabilityVerdict.INVALID
    assert "absence of evidence is not comparability" in cmp.reason


def test_pooling_without_any_ledger_is_refused():
    with pytest.raises(PoolingViolation, match="no stability evidence"):
        guard_pool(_session("A") + _session("B"), None)


def test_single_session_needs_no_ledger():
    kept, dec = guard_pool(_session("A"), None)
    assert dec.allowed and len(kept) == 18
    assert "single session" in dec.reason


# ------------------------------------------------------ non-transitive pooling


def test_comparability_is_not_assumed_transitive():
    """A~B and B~C does not give A~C, and the ledger must check every pair."""
    a = _session("A", jitter=0.01)
    b = _session("B", scale=1.15, jitter=0.01)
    c = _session("C", scale=1.15 * 1.15, jitter=0.01)
    led = SessionLedger.from_readings(a + b + c, criterion=CRITERION)
    assert led.comparison("A", "B").poolable
    assert led.comparison("B", "C").poolable
    assert not led.comparison("A", "C").poolable
    assert not led.poolable_set(["A", "B", "C"])


# ------------------------------------------------------------------ envelope


def test_levels_are_compared_against_what_averaging_predicts():
    """A level whose dispersion is pure noise from below sits near 1.0x excess.

    Guards the bug this replaced: comparing the raw CV of repeats against the
    raw CV of medians-of-repeats made every level look quieter than the one
    below, so a drifting machine read as a stable one.
    """
    env = build_envelope(_session("A", jitter=0.10, n_restarts=4)
                         + _session("B", jitter=0.10, n_restarts=4))
    assert env.across_restart.expected_cv < env.within_run.cv, (
        "a median of n repeats must be predicted to disperse less than the repeats")
    assert math.isfinite(env.restart_over_within)


def test_uniform_shift_between_sessions_refuses_pooling():
    env = build_envelope(_session("A", jitter=0.02, n_restarts=3)
                         + _session("B", scale=1.6, jitter=0.02, n_restarts=3))
    assert env.max_session_drift_observed == pytest.approx(1.6, rel=0.1)
    assert not env.pooling_permitted


def test_workload_specificity_is_not_claimed_from_two_sessions():
    """With one degree of freedom the per-probe ratio is chance, not structure."""
    env = build_envelope(_session("A", jitter=0.05) + _session("B", jitter=0.05))
    assert env.n_sessions == 2
    assert not math.isfinite(env.workload_spread)
    assert env.drift_kind is not DriftKind.WORKLOAD_SPECIFIC
    assert any("one degree of freedom" in n for n in env.notes)


def test_empty_readings_do_not_crash_the_envelope():
    env = build_envelope([])
    assert env.n_readings == 0 and env.drift_kind is DriftKind.UNDETERMINED
    assert not env.pooling_permitted


def test_state_is_recorded_per_reading_not_per_session():
    """LC-8.6. A state read once cannot describe a machine that changed during."""
    rs = _session("A", n_restarts=3)
    assert len({r.state.restart_index for r in rs}) == 3


# ------------------------------------------- regression: the world-4 failure


def test_world4_regression_pooling_two_drifted_sessions_is_refused():
    """The exact failure: a campaign session and a probe session 1.42x apart.

    The medium probe pooled its own 32 new points with the campaign's records
    without checking that the two sessions described the same machine. The
    pooled fit's medium-regime error rose from 13.5% to 32.5% -- adding data
    made it worse, which cannot happen on one surface.
    """
    campaign = _session("campaign", jitter=0.05)
    probe = _session("probe", scale=1.0 / 1.42, jitter=0.05)
    led = SessionLedger.from_readings(campaign + probe, criterion=1.20)

    cmp = led.comparison("campaign", "probe")
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert cmp.observed_ratio == pytest.approx(1.42, rel=0.08)

    with pytest.raises(PoolingViolation):
        guard_pool(campaign + probe, led)


def test_world4_regression_drift_range_spans_the_criterion():
    """The observed drift ran 1.2x to 2.7x: partly inside a naive tolerance.

    The lower end is why an eyeballed 'looks about the same' passed at the time.
    A registered criterion has to reject the 1.2x end too, or it only catches
    the drift that was already obvious.
    """
    a = _session("A", jitter=0.02)
    for factor, expect in ((1.2, False), (2.7, False), (1.02, True)):
        cmp = compare_sessions(a, _session("B", scale=factor, jitter=0.02),
                               criterion=1.15)
        assert cmp.poolable is expect, f"{factor}x should be poolable={expect}"


def test_world4_regression_a_valid_probe_is_not_discarded_as_broken():
    """The probe's data was good. Only its pooling was wrong.

    Marking it invalid would have destroyed the evidence that the machine
    drifted, which turned out to be the more important finding.
    """
    probe = _session("probe", scale=1.0 / 1.42)
    assert all(r.valid for r in probe)
    kept, dec = guard_pool(probe, None)
    assert dec.allowed and len(kept) == len(probe), (
        "usable on its own, within its own session")


# --------------------------------------------------- lightweight import path


def test_stability_machinery_imports_without_heavy_dependencies():
    """LC-1.2 must be checkable by someone who only wants to grade an artifact.

    Standard verification stays installable with numpy alone; measurement is
    what gets to need torch and psutil. A stability rule that could only be
    checked on a machine able to run the benchmark would not be a standard.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        for m in ("torch", "psutil", "scipy", "matplotlib", "triton"):
            sys.modules[m] = None
        import losscolumn.spec.registry
        import losscolumn.spec.schema
        import losscolumn.validate
        from losscolumn.core.pooling import SessionLedger, guard_pool
        from losscolumn.core.stability import build_envelope, compare_sessions
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout


# ------------------------------------------- multiplicity and the internal null


def _noisy_session(sid: str, seed: int, n_probes: int = 200, repeats: int = 2,
                   scale: float = 1.0, noise: float = 0.30):
    """Many probes, few repeats, heavy noise, drawn from one distribution.

    Both sessions come from the *same* generating process and differ only in
    seed, so any apparent difference between them is noise by construction. A
    seeded generator rather than a smooth function: a sine has different
    structure within a probe than between sessions, which is precisely the
    thing the null is supposed to hold constant.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_probes):
        base = 1e-4 * (1.0 + i)
        ts = [float(base * scale * (1.0 + noise * rng.standard_normal()))
              for _ in range(repeats)]
        out.append(SentinelReading(
            session_id=sid, restart_index=0, collective="all_gather", world=3,
            nbytes=1024 + i, size_class="x", timings_s=[abs(t) for t in ts]))
    return out


def test_worst_of_many_probes_exceeds_a_per_probe_criterion_under_pure_noise():
    """The bug this guards: a 6-probe criterion applied to a 276-probe maximum.

    A threshold of the form 1 + 3 sigma bounds ONE comparison. Across hundreds
    of probes some point exceeds it almost surely, so an uncorrected worst-probe
    test calls a perfectly stable machine incomparable.
    """
    a, b = _noisy_session("A", seed=11), _noisy_session("B", seed=22)
    naive = compare_sessions(a, b, criterion=1.20)
    assert naive.observed_ratio > 1.20, (
        "pure noise across 200 probes should breach a per-probe threshold")
    assert naive.verdict is ComparabilityVerdict.INCOMPARABLE


def test_internal_null_admits_what_is_only_noise():
    """The same pair, judged against the null the data itself supplies."""
    a, b = _noisy_session("A", seed=11), _noisy_session("B", seed=22)
    corrected = compare_sessions(a, b, criterion=1.20, use_internal_null=True)
    assert corrected.effective_criterion > 1.20, "the null must widen the threshold"
    assert corrected.verdict is ComparabilityVerdict.COMPARABLE


def test_internal_null_still_rejects_a_real_shift():
    """Widening for multiplicity must not blunt the test against real drift.

    It does blunt the *worst-probe* gate, which is why the level gate exists: a
    maximum over 200 noisy probes must clear a wide floor and so sees only gross
    drift, while the median over the same probes moves with the surface and has
    a standard error that shrinks as probes are added.
    """
    a = _noisy_session("A", seed=11)
    b = _noisy_session("B", seed=22, scale=3.0)
    cmp = compare_sessions(a, b, criterion=1.20, use_internal_null=True)
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE
    assert cmp.failed_gate == "level"
    assert cmp.level_shift == pytest.approx(3.0, rel=0.15)


def test_the_level_gate_is_what_catches_a_shift_the_maximum_misses():
    """Explicitly: the worst-probe gate alone would have passed this pair."""
    a = _noisy_session("A", seed=11)
    b = _noisy_session("B", seed=22, scale=3.0)
    cmp = compare_sessions(a, b, criterion=1.20, use_internal_null=True)
    assert cmp.observed_ratio <= cmp.effective_criterion, (
        "the maximum is inside the noise floor -- it cannot see this")
    assert cmp.verdict is ComparabilityVerdict.INCOMPARABLE


def test_level_gate_ignores_a_single_bad_probe():
    """And conversely: one outlier does not move a median over 200 probes."""
    a = _noisy_session("A", seed=11)
    b = _noisy_session("B", seed=22)
    b[7].timings_s = [t * 6.0 for t in b[7].timings_s]
    cmp = compare_sessions(a, b, criterion=1.20, use_internal_null=True)
    assert cmp.level_shift == pytest.approx(1.0, abs=0.15)


def test_null_from_replicates_needs_enough_repeats():
    """One timing per probe cannot be split, so no null can be derived from it."""
    rs = _noisy_session("A", seed=11, repeats=1)
    w, sh, n = null_from_replicates(rs)
    assert math.isnan(w) and math.isnan(sh) and n < 2


def test_sentinel_scale_comparison_is_unaffected_by_the_correction():
    """With six probes and seven repeats the registered criterion still governs."""
    a, b = _session("A", jitter=0.02), _session("B", scale=1.45, jitter=0.02)
    plain = compare_sessions(a, b, criterion=CRITERION)
    nulled = compare_sessions(a, b, criterion=CRITERION, use_internal_null=True)
    assert plain.verdict is nulled.verdict is ComparabilityVerdict.INCOMPARABLE


def test_workload_specificity_requires_significance_not_a_large_ratio():
    """Guards the classifier bug: chance alone produces spreads of six or seven.

    Six per-probe CVs each estimated from four session medians have enormous
    spread under a null where every probe behaves identically, so a fixed ratio
    threshold reports the sample size rather than the machine.
    """
    readings = []
    for s in range(4):
        readings += _session(f"S{s}", jitter=0.12, n_restarts=3)
    env = build_envelope(readings)
    if math.isfinite(env.workload_spread) and env.workload_spread >= 2.0:
        assert math.isfinite(env.workload_spread_p), (
            "a large spread must be tested, not asserted")
        if env.workload_spread_p >= 0.05:
            assert env.drift_kind is not DriftKind.WORKLOAD_SPECIFIC


def test_restart_level_drift_is_a_named_outcome():
    """Variation that enters at the launch and not with elapsed time.

    Distinct from PERSISTENT, and the distinction changes the remedy: 'measure
    it all in one sitting' does nothing about a difference that arrives with
    each new process.
    """
    assert DriftKind.RESTART_LEVEL.value == "restart_level"
    assert "process launch" in DriftKind.RESTART_LEVEL.remedy


# ----------------------------------------- a verdict states what it covers


def test_a_stability_verdict_names_the_worlds_it_certifies():
    """SENTINEL_WORLD was 3 and the driver never overrode it, so every
    stability verdict this project produced described a three-rank group --
    while licensing campaigns that ran at 2, 3, 4 and 8.

    Stability is not world-independent. More ranks means more contention, more
    memory pressure, and on a split node a transport that crosses the
    interconnect. A machine steady at three ranks can be anything at eight.
    """
    from losscolumn.thrusts.overlap.sentinel import sentinel_protocol

    p = sentinel_protocol(n_sessions=2, n_restarts=2, repeats=7, gap_s=300,
                          worlds=(2, 3, 4))
    assert p["worlds_certified"] == [2, 3, 4]
    assert "no others" in p["scope_rule"]


def test_a_single_world_protocol_still_states_its_scope():
    from losscolumn.thrusts.overlap.sentinel import sentinel_protocol

    p = sentinel_protocol(n_sessions=2, n_restarts=2, repeats=7, gap_s=300)
    assert p["worlds_certified"] == [p["world"]]


def test_uncertified_worlds_are_named_not_hidden():
    from losscolumn.thrusts.overlap.sentinel import (
        certified_worlds,
        uncertified_worlds,
    )

    rs = _session("S0", world=2) + _session("S0", world=3)
    assert certified_worlds(rs) == {2, 3}
    assert uncertified_worlds(rs, [2, 3, 4, 8]) == [4, 8]


def test_nothing_is_missing_when_the_sentinel_covered_it():
    from losscolumn.thrusts.overlap.sentinel import uncertified_worlds

    rs = _session("S0", world=2) + _session("S0", world=4)
    assert uncertified_worlds(rs, [2, 4]) == []


def test_invalid_readings_do_not_certify_a_world():
    """A world that was attempted and failed is not a world that was checked."""
    from losscolumn.thrusts.overlap.sentinel import uncertified_worlds

    rs = _session("S0", world=2) + _session("S0", world=8)
    for r in rs:
        if r.world == 8:
            r.valid = False
    assert uncertified_worlds(rs, [2, 8]) == [8]
