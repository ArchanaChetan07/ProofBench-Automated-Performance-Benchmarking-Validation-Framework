"""Communication model families, and the gate that keeps bad fits out.

The finding these tests protect: a straight line scored R^2 = 0.984 on gloo
all-reduce and was accepted, then failed at 29% median error once the sweep
reached small messages. R^2 on a log-spaced sweep is dominated by the large
points, so a model can be badly wrong on every small message and still score
0.98. Held-out relative error is the gate that catches it.
"""

from __future__ import annotations

import math

import pytest

from losscolumn.core.parameters import Provenance, RejectedParameterError
from losscolumn.thrusts.overlap.commodel import (
    LinearModel,
    PiecewiseModel,
    Sample,
    fit_linear,
    fit_piecewise,
    select_model,
)


def _linear_samples(alpha_s=1e-4, beta=1e9, sizes=None, noise=0.0, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    sizes = sizes or [1 << k for k in range(10, 25)]
    out = []
    for n in sizes:
        t = alpha_s + n / beta
        if noise:
            t *= 1 + rng.normal(0, noise)
        out.append(Sample(nbytes=n, seconds=t, world=1, kind="copy"))
    return out


def _two_regime_samples(bp=1 << 18, seed=1):
    """Cheap below the breakpoint, expensive above -- an algorithm switch."""
    out = []
    for k in range(10, 25):
        n = 1 << k
        t = 5e-5 + n / 4e9 if n < bp else 2e-4 + n / 1e9
        out.append(Sample(nbytes=n, seconds=t, world=1, kind="copy"))
    _ = seed
    return out


class TestLinearFamily:
    def test_recovers_planted_latency_and_bandwidth(self):
        m = fit_linear(_linear_samples(alpha_s=2e-4, beta=5e9))
        assert m is not None
        assert m.alpha_us == pytest.approx(200, rel=0.05)
        assert m.beta_gbs == pytest.approx(5.0, rel=0.05)

    def test_refuses_a_negative_slope(self):
        bad = [Sample(nbytes=n, seconds=1.0 - n * 1e-9) for n in (1 << 20, 1 << 22)]
        assert fit_linear(bad) is None

    def test_high_r_squared_coexists_with_large_relative_error(self):
        """The trap that made a 0.984 fit look acceptable.

        R^2 measures explained variance, which on a log-spaced sweep is almost
        entirely the largest points. A line fitted to a two-regime transport
        can therefore score above 0.9 while being wrong by a fifth on the
        typical point. That is why the gate is relative error, not R^2.
        """
        s = _two_regime_samples()
        lin = fit_linear(s)
        assert lin is not None
        assert lin.r_squared(s) > 0.9
        assert lin.median_rel_error(s) > 0.15
        assert lin.max_rel_error(s) > 0.5

    def test_the_weighted_fit_spreads_error_instead_of_dumping_it_on_small_messages(self):
        """Unweighted least squares is dominated by the largest messages.

        Each point now carries comparable influence, so a misfit shows up as
        error spread across the range rather than concentrated where nobody
        was looking.
        """
        s = _two_regime_samples()
        lin = fit_linear(s)
        small = [x for x in s if x.nbytes <= (1 << 13)]
        large = [x for x in s if x.nbytes >= (1 << 22)]
        assert lin.median_rel_error(small) < 3 * lin.median_rel_error(large)


class TestPiecewiseFamily:
    def test_finds_a_planted_breakpoint(self):
        m = fit_piecewise(_two_regime_samples(bp=1 << 18))
        assert m is not None
        assert (1 << 15) <= m.breakpoint_bytes <= (1 << 21)

    def test_beats_linear_on_a_two_regime_transport(self):
        s = _two_regime_samples()
        lin, pw = fit_linear(s), fit_piecewise(s)
        assert pw.median_rel_error(s) < lin.median_rel_error(s)

    def test_needs_enough_points_on_each_side(self):
        assert fit_piecewise(_linear_samples(sizes=[1 << 10, 1 << 11])) is None

    def test_predict_switches_at_the_breakpoint(self):
        m = PiecewiseModel(
            breakpoint_bytes=1000,
            low=LinearModel(alpha_s=1.0, beta_bytes_per_s=1e12),
            high=LinearModel(alpha_s=5.0, beta_bytes_per_s=1e12),
        )
        assert m.predict(999) == pytest.approx(1.0, abs=1e-6)
        assert m.predict(1001) == pytest.approx(5.0, abs=1e-6)


class TestFamilySelection:
    def test_a_truly_linear_transport_chooses_linear(self):
        """The simpler family wins when the flexible one buys nothing."""
        s = _linear_samples(noise=0.01, seed=3)
        sel = select_model(s[::2], s[1::2], transport="t", kind="k")
        # The family name now carries the estimator that fitted it, because a
        # family that only wins under one loss is telling you about the loss.
        assert sel.chosen_family.startswith("linear"), sel.chosen_family

    def test_a_two_regime_transport_chooses_piecewise(self):
        s = _two_regime_samples()
        sel = select_model(s[::2], s[1::2], transport="t", kind="k")
        assert not sel.chosen_family.startswith("linear"), sel.chosen_family

    def test_selection_uses_held_out_error_not_fit_quality(self):
        """A family that only fits its own data must not win."""
        s = _two_regime_samples()
        sel = select_model(s[::2], s[1::2], transport="t", kind="k")
        chosen = next(c for c in sel.candidates if c["family"] == sel.chosen_family)
        for c in sel.candidates:
            if math.isfinite(c["heldout_median_rel_err"]):
                assert chosen["heldout_median_rel_err"] <= (
                    c["heldout_median_rel_err"] * 1.1 + 1e-12
                )

    def test_nothing_is_promoted_when_no_family_clears_the_gate(self):
        import numpy as np

        rng = np.random.default_rng(7)
        s = [Sample(nbytes=1 << k, seconds=float(abs(rng.normal(1e-3, 9e-4))))
             for k in range(10, 25)]
        sel = select_model(s[::2], s[1::2], transport="t", kind="k",
                           max_heldout_median_err=0.15)
        assert sel.chosen is None
        assert "choosing the answer" in sel.reason

    def test_a_noise_limited_residual_is_named_as_such(self):
        """A model cannot predict better than the measurement repeats."""
        import numpy as np

        rng = np.random.default_rng(11)
        s = [Sample(nbytes=1 << k,
                    seconds=(1e-4 + (1 << k) / 1e9) * (1 + rng.normal(0, 0.18)))
             for k in range(10, 25)]
        sel = select_model(s[::2], s[1::2], transport="t", kind="k",
                           noise_floor=0.18, max_heldout_median_err=0.05)
        assert sel.chosen is None
        assert "measurement noise" in sel.reason

    def test_all_candidates_are_reported_even_when_none_is_chosen(self):
        """A rejected fit stays visible as a diagnostic."""
        import numpy as np

        rng = np.random.default_rng(5)
        s = [Sample(nbytes=1 << k, seconds=float(abs(rng.normal(1e-3, 9e-4))))
             for k in range(10, 25)]
        sel = select_model(s[::2], s[1::2], max_heldout_median_err=0.15)
        # Three estimators, and under each: linear once, plus the two segmented
        # families on each of the two structure scales. `linear` has no
        # structure to choose so it is not doubled. Fifteen in all, every one
        # reported even though none is promoted.
        assert len(sel.candidates) == 15
        assert all("family" in c for c in sel.candidates)
        scales = {c["family"].endswith("/abs") for c in sel.candidates}
        assert scales == {True, False}, (
            "both structure scales must be offered; hard-coding either one was "
            "the bug this replaced")


class TestRejectedParametersStayOut:
    """Phase 6: no rejected fit may enter the simulator."""

    def test_a_diagnostic_parameter_cannot_be_read_as_a_model_input(self):
        from losscolumn.core.parameters import Parameter, ParameterSet

        s = ParameterSet("comm")
        s.add(Parameter("gloo/all_reduce/alpha_us", float("nan"),
                        Provenance.DIAGNOSTIC, source="selection",
                        note="no family cleared the gate"))
        with pytest.raises(RejectedParameterError):
            s.get("gloo/all_reduce/alpha_us")

    def test_fabric_falls_back_rather_than_using_a_rejected_value(self):
        from losscolumn.core.parameters import Parameter, ParameterSet

        s = ParameterSet("comm")
        s.add(Parameter("beta_gbs", 0.14, Provenance.REJECTED, source="fit",
                        note="held-out error 46%"))
        assert s.get_or("beta_gbs", 235.0) == 235.0

    def test_the_study_promotes_only_what_passes(self):
        """Checked against the real artifact when it is present."""
        import json
        from pathlib import Path

        art = (Path(__file__).resolve().parents[1] / "artifacts"
               / "calibration-thrust1-communication.json")
        if not art.exists():
            pytest.skip("communication study not run here")
        d = json.loads(art.read_text(encoding="utf-8"))
        for p in d["parameters"]["parameters"]:
            q = p.get("quality") or {}
            err = q.get("heldout_median_rel_err")
            if p["provenance"] == "accepted" and err is not None and math.isfinite(err):
                assert err <= d["protocol"]["quality_gate"][
                    "max_heldout_median_relative_error"
                ], p["name"]
            if not p["usable"]:
                assert p["note"], f"{p['name']} is unusable with no reason recorded"


# --------------------------------------------------------------------------
# adversarial synthetic surfaces, with known right answers
# --------------------------------------------------------------------------


def _surface(fn, k_lo=10, k_hi=25, noise=0.0, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for k in range(k_lo, k_hi):
        n = 1 << k
        t = fn(n)
        if noise:
            t *= 1 + rng.normal(0, noise)
        out.append(Sample(nbytes=n, seconds=t, world=1, kind="copy"))
    return out


def _tier(nbytes: int) -> str:
    if nbytes < (1 << 14):
        return "tiny"
    if nbytes < (1 << 18):
        return "small"
    if nbytes < (1 << 22):
        return "medium"
    return "large"


class TestAdversarialSurfaces:
    """Surfaces whose right answer is known, including the ones that fooled us."""

    def test_A_pure_linear_chooses_linear(self):
        s = _surface(lambda n: 1e-4 + n / 1e9, noise=0.01, seed=1)
        sel = select_model(s[::2], s[1::2], tier_fn=_tier)
        assert sel.chosen_family.startswith("linear")

    def test_B_two_regimes_chooses_at_least_piecewise(self):
        s = _surface(lambda n: 5e-5 + n / 4e9 if n < (1 << 18) else 2e-4 + n / 1e9)
        sel = select_model(s[::2], s[1::2], tier_fn=_tier)
        assert not sel.chosen_family.startswith("linear")

    def test_C_non_monotonic_bandwidth_needs_three_regimes(self):
        """The real shape: bandwidth peaks in the middle and falls again.

        A buffer that fits in cache goes faster than one that does not, so
        effective bandwidth rises out of the latency floor, peaks, then drops.
        No two-segment model can reproduce a peak, and this is the case the
        measured gloo data turned out to be.
        """
        def f(n):
            if n < (1 << 15):
                return 5e-5 + n / 0.01e9
            if n < (1 << 21):
                return 1e-4 + n / 0.35e9
            return 3e-4 + n / 0.24e9

        s = _surface(f)
        from losscolumn.thrusts.overlap.commodel import fit_regime

        two = fit_regime(s, max_segments=2)
        three = fit_regime(s, max_segments=3)
        assert len(three.segments) == 3
        assert three.median_rel_error(s) < two.median_rel_error(s) / 5

    def test_C_aic_refuses_a_fourth_segment_it_does_not_need(self):
        def f(n):
            if n < (1 << 15):
                return 5e-5 + n / 0.01e9
            if n < (1 << 21):
                return 1e-4 + n / 0.35e9
            return 3e-4 + n / 0.24e9

        from losscolumn.thrusts.overlap.commodel import fit_regime

        assert len(fit_regime(_surface(f), max_segments=5).segments) == 3

    def test_D_pure_noise_promotes_nothing(self):
        import numpy as np

        rng = np.random.default_rng(4)
        s = [Sample(nbytes=1 << k, seconds=float(abs(rng.normal(1e-3, 9e-4))))
             for k in range(10, 25)]
        sel = select_model(s[::2], s[1::2], tier_fn=_tier)
        assert sel.chosen is None

    def test_E_one_bad_regime_cannot_hide_behind_three_good_ones(self):
        """The pooling failure, as a surface.

        Three regimes fitted well and one badly gives a good pooled median and
        a terrible worst regime. Selection must see the worst.
        """
        def f(n):
            return (1e-4 + n / 1e9) * (6.0 if (1 << 18) <= n < (1 << 22) else 1.0)

        s = _surface(f)
        sel = select_model(s[::2], s[1::2], tier_fn=_tier,
                           max_heldout_median_err=0.15)
        best = min(
            (c for c in sel.candidates if c["heldout_per_tier_err"]),
            key=lambda c: c["heldout_worst_tier_err"],
        )
        assert best["heldout_worst_tier_err"] > best["heldout_median_rel_err"]

    def test_F_a_single_outlier_does_not_capture_the_robust_fit(self):
        """Huber exists for the scheduling hiccup on one repeat."""
        from losscolumn.thrusts.overlap.commodel import fit_linear

        s = _surface(lambda n: 1e-4 + n / 1e9)
        s = list(s)
        s[7] = Sample(nbytes=s[7].nbytes, seconds=s[7].seconds * 12, world=1,
                      kind="copy")
        w = fit_linear(s, loss="weighted")
        h = fit_linear(s, loss="huber")
        if h is None:
            import pytest as _p

            _p.skip("SciPy unavailable; robust loss falls back")
        clean = [x for i, x in enumerate(s) if i != 7]
        assert h.median_rel_error(clean) < w.median_rel_error(clean)


class TestCrossValidation:
    def test_every_size_is_held_out_exactly_once(self):
        from losscolumn.thrusts.overlap.commodel import cross_validate, fit_linear

        s = _surface(lambda n: 1e-4 + n / 1e9, noise=0.02, seed=2)
        cv = cross_validate(s, fit_linear, k=4, tier_fn=_tier)
        assert cv["ok"]
        assert cv["n_predictions"] == len(s)

    def test_folds_are_stratified_so_no_regime_is_missing(self):
        from losscolumn.thrusts.overlap.commodel import cross_validate, fit_linear

        s = _surface(lambda n: 1e-4 + n / 1e9, noise=0.02, seed=3)
        cv = cross_validate(s, fit_linear, k=4, tier_fn=_tier)
        assert set(cv["per_tier_err"]) == {"tiny", "small", "medium", "large"}
        assert all(v >= 1 for v in cv["n_per_tier"].values())

    def test_it_refuses_when_there_are_too_few_sizes(self):
        from losscolumn.thrusts.overlap.commodel import cross_validate, fit_linear

        s = _surface(lambda n: 1e-4 + n / 1e9, k_lo=10, k_hi=13)
        assert not cross_validate(s, fit_linear, k=4)["ok"]
