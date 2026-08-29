"""Statistical core: correctness against known answers, not against itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from losscolumn.core.stats import (
    bca_ci,
    bh_fdr,
    design_can_reject,
    min_replicates_for_fdr,
    signflip_p,
    signflip_p_floor,
    t_sf,
    welch_p,
)


class TestStudentT:
    """Checked against published critical values, not against scipy."""

    @pytest.mark.parametrize(
        "t,df,expected",
        [
            (0.0, 10, 0.5),
            (2.228, 10, 0.025),      # two-sided 0.05 critical value at df=10
            (1.812, 10, 0.05),
            (2.086, 20, 0.025),
            (1.960, 10_000, 0.025),  # normal limit
        ],
    )
    def test_upper_tail(self, t, df, expected):
        assert t_sf(t, df) == pytest.approx(expected, abs=5e-4)

    def test_symmetry(self):
        for t in (0.3, 1.0, 2.5, 4.0):
            assert t_sf(t, 7) + t_sf(-t, 7) == pytest.approx(1.0, abs=1e-9)


class TestSignFlip:
    def test_exact_floor_is_two_to_the_minus_n(self):
        """The most extreme possible result still cannot beat 2**-n."""
        for n in (5, 8, 11):
            d = np.ones(n)
            assert signflip_p(d) == pytest.approx(2.0**-n, rel=1e-12)
            assert signflip_p_floor(n) == pytest.approx(2.0**-n)

    def test_null_is_not_significant(self):
        rng = np.random.default_rng(0)
        ps = [signflip_p(rng.normal(0, 1, 10)) for _ in range(200)]
        # A valid test is uniform-ish under the null; at minimum it must not
        # fire far more often than its nominal rate.
        assert np.mean(np.asarray(ps) < 0.05) < 0.12

    def test_direction(self):
        d = np.array([0.3, 0.25, 0.4, 0.2, 0.35, 0.28, 0.31])
        assert signflip_p(d, alternative="greater") < 0.02
        assert signflip_p(d, alternative="less") > 0.9

    def test_monte_carlo_p_is_never_zero(self):
        d = np.full(20, 5.0)
        p = signflip_p(d, n_mc=500)
        assert 0 < p <= 1


A_SAMPLE = np.array([27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1])
B_SAMPLE = np.array([27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0])


class TestWelch:
    def test_known_case(self):
        """Values verified against scipy.stats.ttest_ind(equal_var=False)."""
        p, df = welch_p(A_SAMPLE, B_SAMPLE, alternative="less")
        assert p == pytest.approx(0.0293333239, abs=1e-9)
        assert df == pytest.approx(11.093011, abs=1e-5)

    def test_matches_scipy_across_a_grid(self):
        """The whole point of hand-rolling the t-distribution is that core stays
        numpy-only. That is only defensible if it agrees with the reference
        implementation where the reference is available."""
        scipy_stats = pytest.importorskip("scipy.stats")
        for t in (-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0):
            for df in (1, 3, 10, 50, 1000):
                assert t_sf(t, df) == pytest.approx(
                    float(scipy_stats.t.sf(t, df)), abs=1e-10
                )
        got = welch_p(A_SAMPLE, B_SAMPLE, alternative="less")
        ref = scipy_stats.ttest_ind(A_SAMPLE, B_SAMPLE, equal_var=False, alternative="less")
        assert got[0] == pytest.approx(float(ref.pvalue), abs=1e-10)


class TestBCa:
    def test_covers_the_truth(self):
        """Nominal 95% intervals should cover close to 95% of the time."""
        rng = np.random.default_rng(3)
        covered = 0
        trials = 120
        for i in range(trials):
            x = rng.normal(1.0, 0.25, 12)
            lo, hi = bca_ci(x, np.mean, n_boot=800, rng=np.random.default_rng(i))
            covered += lo <= 1.0 <= hi
        assert 0.86 <= covered / trials <= 1.0

    def test_degenerate_input(self):
        lo, hi = bca_ci(np.array([1.0]))
        assert math.isnan(lo) and math.isnan(hi)


class TestBenjaminiHochberg:
    def test_controls_fdr_under_the_null(self):
        rng = np.random.default_rng(1)
        false_positives = 0
        rounds = 200
        for _ in range(rounds):
            p = rng.uniform(size=60)
            rej, _ = bh_fdr(p, q=0.05)
            false_positives += rej.any()
        assert false_positives / rounds < 0.12

    def test_finds_real_effects(self):
        p = np.concatenate([np.full(10, 1e-6), np.random.default_rng(0).uniform(size=50)])
        rej, q = bh_fdr(p, q=0.05)
        assert rej[:10].all()
        assert q[0] <= 0.05

    def test_monotone_qvalues(self):
        p = np.sort(np.random.default_rng(2).uniform(size=40))
        _, q = bh_fdr(p)
        assert np.all(np.diff(q) >= -1e-12)

    def test_handles_nan(self):
        rej, q = bh_fdr([0.001, float("nan"), 0.5])
        assert rej[0] and not rej[2]
        assert math.isnan(q[1])


class TestDesignPower:
    """The check that stops an underpowered sweep reporting a clean bill of health."""

    def test_five_replicates_cannot_reject_over_a_large_grid(self):
        d = design_can_reject(5, 48, 0.05)
        assert not d["can_reject"]
        assert d["min_replicates"] == 10
        assert "cannot reject" in d["message"]

    def test_eleven_replicates_can(self):
        assert design_can_reject(11, 48, 0.05)["can_reject"]

    def test_requirement_grows_with_grid_size(self):
        assert min_replicates_for_fdr(10) < min_replicates_for_fdr(1000)

    def test_boundary_is_exact(self):
        """r is sufficient exactly when 2**-r <= q/m."""
        m, q = 48, 0.05
        r = min_replicates_for_fdr(m, q)
        assert 2.0**-r <= q / m
        assert 2.0 ** -(r - 1) > q / m
