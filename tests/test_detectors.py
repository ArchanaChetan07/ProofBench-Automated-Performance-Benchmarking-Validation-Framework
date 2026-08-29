"""Regression suite for the two detectors, on surfaces with known ground truth.

The loss-region extractor and the cliff detector decide what every claim this
package emits actually says. A detector that drifts silently would move every
result downstream of it, so each is scored here against surfaces whose right
answer is fixed by construction.

The five cliff surfaces are the cases where this kind of detector characteristically
fails: mistaking a slope for a discontinuity, hallucinating structure in noise,
and being blinded to a moderate cliff by a catastrophic one.
"""

from __future__ import annotations

import pytest

from losscolumn.core.cliffs import detect_cliffs
from losscolumn.core.losscolumn import extract_loss_column
from losscolumn.core.stats import compare_cells
from losscolumn.core.surfaces import (
    dominant_cliff,
    multiple_cliffs,
    noisy_smooth,
    planted_region,
    region_iou,
    single_cliff,
    smooth_decline,
)


def _cliffs(gt, **kw):
    return detect_cliffs(gt.envelope, "sys", seed=0, **kw)


# --------------------------------------------------------------------------
# A -- smooth decline
# --------------------------------------------------------------------------


class TestASmoothDecline:
    def test_no_cliff_on_a_steep_but_smooth_surface(self):
        gt = smooth_decline(ratio=0.70)
        rep = _cliffs(gt)
        assert rep.n_cliffs == 0, [c.describe() for c in rep.worst(5)]

    @pytest.mark.parametrize("ratio", [0.9, 0.8, 0.7, 0.6, 0.5])
    def test_holds_at_every_slope(self, ratio):
        """Even a 50%-per-level decline is a slope, not a discontinuity."""
        rep = _cliffs(smooth_decline(ratio=ratio, seed=int(ratio * 100)))
        assert rep.n_cliffs == 0, f"ratio={ratio}: {[c.describe() for c in rep.worst(3)]}"

    def test_exposure_is_still_reported(self):
        """No cliff does not mean no risk: the per-step exposure is still real."""
        gt = smooth_decline(ratio=0.7)
        rep = _cliffs(gt)
        assert max(rep.exposure.values()) > 30.0


# --------------------------------------------------------------------------
# B -- one abrupt transition
# --------------------------------------------------------------------------


class TestBSingleCliff:
    def test_finds_exactly_the_planted_cliff(self):
        gt = single_cliff(at=3, drop=0.45)
        rep = _cliffs(gt)
        score = gt.score_cliffs(rep.cliffs)
        assert score["recall"] == 1.0, score
        assert score["false_positives"] == 0, [c.describe() for c in rep.cliffs]

    def test_one_per_fiber(self):
        gt = single_cliff()
        assert _cliffs(gt).n_cliffs == gt.n_fibers

    def test_locates_the_right_transition(self):
        gt = single_cliff(at=4, drop=0.4)
        for c in _cliffs(gt).cliffs:
            assert (c.from_level, c.to_level) == (3, 4)

    def test_reports_the_right_magnitude(self):
        gt = single_cliff(at=3, drop=0.5)      # halving -> +100% degradation
        for c in _cliffs(gt).cliffs:
            assert 90 < c.degradation_pct < 110
            assert c.ci_lo_pct < c.degradation_pct < c.ci_hi_pct

    @pytest.mark.parametrize("drop", [0.5, 0.6, 0.7])
    def test_detects_across_magnitudes(self, drop):
        gt = single_cliff(drop=drop, seed=int(drop * 50))
        assert gt.score_cliffs(_cliffs(gt).cliffs)["recall"] == 1.0

    def test_a_drop_below_the_floor_is_not_a_cliff(self):
        """A 5% step is real but not a discontinuity worth acting on."""
        gt = single_cliff(drop=0.95, noise=0.005)
        assert _cliffs(gt, min_drop=0.15).n_cliffs == 0


# --------------------------------------------------------------------------
# C -- several independent cliffs
# --------------------------------------------------------------------------


class TestCMultipleCliffs:
    def test_finds_both(self):
        gt = multiple_cliffs(positions=(2, 5), drops=(0.55, 0.5))
        score = gt.score_cliffs(_cliffs(gt).cliffs)
        assert score["recall"] == 1.0, score
        assert score["false_positives"] == 0, score

    def test_each_is_reported_separately(self):
        gt = multiple_cliffs(positions=(2, 5))
        rep = _cliffs(gt)
        transitions = {(c.from_level, c.to_level) for c in rep.cliffs}
        assert transitions == {(1, 2), (4, 5)}

    def test_three_cliffs(self):
        gt = multiple_cliffs(positions=(2, 4, 6), drops=(0.55, 0.55, 0.55),
                             n_levels=8, seed=22)
        score = gt.score_cliffs(_cliffs(gt).cliffs)
        assert score["recall"] == 1.0, score

    def test_count_scales_with_fibers(self):
        gt = multiple_cliffs(positions=(2, 5), n_fibers=5, seed=23)
        assert _cliffs(gt).n_cliffs == 2 * gt.n_fibers


# --------------------------------------------------------------------------
# D -- noise must not become structure
# --------------------------------------------------------------------------


class TestDNoisySmooth:
    def test_no_hallucinated_cliffs(self):
        rep = _cliffs(noisy_smooth(noise=0.06))
        assert rep.n_cliffs == 0, [c.describe() for c in rep.worst(5)]

    @pytest.mark.parametrize("seed", range(8))
    def test_holds_across_seeds(self, seed):
        """Run repeatedly: a threshold that is only usually right is not right."""
        rep = _cliffs(noisy_smooth(noise=0.06, seed=seed))
        assert rep.n_cliffs == 0, f"seed={seed}: {[c.describe() for c in rep.worst(3)]}"

    @pytest.mark.parametrize("noise", [0.02, 0.05, 0.08, 0.12])
    def test_holds_across_noise_levels(self, noise):
        rep = _cliffs(noisy_smooth(noise=noise, seed=int(noise * 300)))
        assert rep.n_cliffs == 0, f"noise={noise}: {[c.describe() for c in rep.worst(3)]}"

    def test_a_flat_surface_with_no_spread_at_all(self):
        """The degenerate case: near-zero variance in the step distribution.

        This is where a scale estimated from the steps collapses toward zero and
        every step becomes an outlier. The floor on the scale exists for this.
        """
        rep = _cliffs(noisy_smooth(noise=0.0005, seed=99))
        assert rep.n_cliffs == 0


# --------------------------------------------------------------------------
# E -- a catastrophic cliff must not mask a moderate one
# --------------------------------------------------------------------------


class TestEDominantCliff:
    def test_both_are_found(self):
        gt = dominant_cliff(catastrophic_drop=0.05, moderate_drop=0.6)
        score = gt.score_cliffs(_cliffs(gt).cliffs)
        assert score["recall"] == 1.0, (
            "the catastrophic step contaminated the baseline and hid the moderate one: "
            f"{score}"
        )

    def test_the_moderate_cliff_is_reported_on_its_own_terms(self):
        gt = dominant_cliff(catastrophic_drop=0.05, moderate_drop=0.6)
        by_transition = {
            (c.from_level, c.to_level): c for c in _cliffs(gt).cliffs
        }
        moderate = by_transition[(4, 5)]
        assert 55 < moderate.degradation_pct < 80      # 1/0.6 - 1 = +67%
        catastrophic = by_transition[(1, 2)]
        assert catastrophic.degradation_pct > 1500     # 1/0.05 - 1 = +1900%

    @pytest.mark.parametrize("catastrophic_drop", [0.02, 0.05, 0.1])
    def test_holds_as_the_dominant_cliff_grows(self, catastrophic_drop):
        gt = dominant_cliff(catastrophic_drop=catastrophic_drop, moderate_drop=0.6,
                            seed=int(catastrophic_drop * 400))
        assert gt.score_cliffs(_cliffs(gt).cliffs)["recall"] == 1.0

    def test_an_unrunnable_level_does_not_mask_a_real_cliff(self):
        """The steepest possible cliff is a level that does not run at all."""
        gt = dominant_cliff(catastrophic_drop=0.05, moderate_drop=0.6, seed=7)
        env = gt.envelope
        for cell in env.cells():
            if env.coords(cell)["x"] == 7:
                env.mark_missing("sys", cell, "OOM")
        rep = detect_cliffs(env, "sys", seed=0)
        transitions = {(c.from_level, c.to_level) for c in rep.cliffs}
        assert (4, 5) in transitions, "the moderate cliff was lost"
        assert (6, 7) in transitions, "the unrunnable level was not reported"


# --------------------------------------------------------------------------
# the loss-region extractor
# --------------------------------------------------------------------------


def _extract(gt, mde=0.05):
    cmps = compare_cells(gt.envelope, "method", "baseline", mde=mde, q=0.05, seed=0)
    lc = extract_loss_column(gt.envelope, cmps, method="method", baseline="baseline",
                             mde=mde, q_level=0.05)
    found = {
        tuple(c.cell) for c in cmps if c.verdict == "loss"
    }
    return cmps, lc, found


class TestLossRegionDetector:
    def test_recovers_a_planted_box_exactly(self):
        gt = planted_region(planted={"seq_len": [2048], "dtype": ["fp16"]})
        _, lc, found = _extract(gt)
        assert region_iou(found, gt.true_loss_cells) == 1.0
        assert len(lc.regions) == 1

    def test_no_region_on_an_unperturbed_surface(self):
        gt = planted_region(planted=None)
        _, lc, found = _extract(gt)
        assert not found and lc.is_empty

    @pytest.mark.parametrize("regression", [0.15, 0.30, 0.60])
    def test_recovers_across_effect_sizes(self, regression):
        gt = planted_region(planted={"seq_len": [2048]}, regression=regression,
                            seed=int(regression * 100))
        _, _, found = _extract(gt)
        assert region_iou(found, gt.true_loss_cells) == 1.0

    @pytest.mark.parametrize("seed", range(6))
    def test_stable_across_seeds(self, seed):
        gt = planted_region(planted={"dtype": ["fp16"]}, regression=0.25, seed=seed)
        _, _, found = _extract(gt)
        assert region_iou(found, gt.true_loss_cells) >= 0.95, f"seed={seed}"

    def test_a_disjoint_pair_of_regions(self):
        gt = planted_region(planted={"seq_len": [128]}, regression=0.30, seed=31)
        env = gt.envelope
        for b in (1, 2, 4, 8):
            cell = env.cell_of(batch=b, seq_len=2048, dtype="bf16")
            env.put("method", cell, env.replicates_at("baseline", cell) * 0.65)
            gt.true_loss_cells.add(cell)
        _, lc, found = _extract(gt)
        assert region_iou(found, gt.true_loss_cells) == 1.0
        assert len(lc.regions) >= 2

    def test_noise_alone_produces_no_region(self):
        """The false-positive rate matters as much as the recall."""
        for seed in range(8):
            gt = planted_region(planted=None, noise=0.03, seed=100 + seed)
            _, lc, found = _extract(gt)
            assert not found, f"seed={seed} invented {len(found)} losing cell(s)"
            assert lc.is_empty


class TestEquivalenceIsEstablished:
    """The regression test for an inverted TOST.

    An equivalence test whose one-sided alternatives point outward instead of
    inward makes ties essentially unprovable, so every genuinely-equal cell
    lands as inconclusive. That bug was real in this package; this is the test
    that would have caught it, and LC-Q6 is the validator rule that catches its
    signature in a finished artifact.
    """

    def test_identical_systems_are_ties_not_inconclusive(self):
        gt = planted_region(planted=None, regression=0.0)
        cmps, _, _ = _extract(gt)
        verdicts = [c.verdict for c in cmps]
        assert verdicts.count("tie") >= 0.9 * len(verdicts), (
            f"only {verdicts.count('tie')}/{len(verdicts)} cells established as "
            f"equivalent; inconclusive={verdicts.count('inconclusive')}"
        )

    def test_a_subthreshold_difference_is_a_tie(self):
        gt = planted_region(planted={"dtype": ["fp16"]}, regression=0.02, noise=0.002,
                            seed=41)
        cmps, lc, _ = _extract(gt, mde=0.05)
        assert lc.counts["loss"] == 0
        assert lc.counts["tie"] > 0

    def test_equivalence_is_not_claimed_when_power_is_absent(self):
        """Symmetric check: with too few replicates, ties must not be asserted."""
        gt = planted_region(planted=None, replicates=3, seed=42)
        cmps, lc, _ = _extract(gt)
        assert lc.underpowered


class TestInapplicableCells:
    """A hole in the lattice is not a failed measurement.

    The distinction is load-bearing in both directions. Counting inapplicable
    corners as failures makes a fully-measured sweep read as half finished,
    which is how a genuine coverage warning gets ignored; dropping them
    silently makes the artifact claim a full factorial it never ran.
    """

    def _env_with_hole(self):
        gt = planted_region(planted={"seq_len": [2048]}, regression=0.30, seed=77)
        env = gt.envelope
        holes = [c for c in env.cells() if env.coords(c)["dtype"] == "bf16"
                 and env.coords(c)["seq_len"] == 128]
        for c in holes:
            env.mark_inapplicable(c, "not a configuration")
        return gt, env, holes

    def test_coverage_ignores_inapplicable_cells(self):
        _, env, holes = self._env_with_hole()
        assert all(v == 1.0 for v in env.coverage().values()), env.coverage()
        assert len(env.applicable_cells()) == env.n_cells - len(holes)

    def test_loss_column_counts_them_apart_from_missing(self):
        gt, env, holes = self._env_with_hole()
        cmps = compare_cells(env, "method", "baseline", mde=0.05, q=0.05, seed=0)
        lc = extract_loss_column(env, cmps, method="method", baseline="baseline",
                                 mde=0.05, q_level=0.05)
        assert lc.n_inapplicable == len(holes)
        assert lc.counts["inapplicable"] == len(holes)
        assert lc.counts["missing"] == 0, "an inapplicable cell is not a failure"
        assert lc.n_cells == env.n_cells - len(holes)

    def test_fractions_use_the_applicable_denominator(self):
        gt, env, holes = self._env_with_hole()
        cmps = compare_cells(env, "method", "baseline", mde=0.05, q=0.05, seed=0)
        lc = extract_loss_column(env, cmps, method="method", baseline="baseline",
                                 mde=0.05, q_level=0.05)
        assert lc.loss_fraction == lc.counts["loss"] / lc.n_cells

    def test_a_real_failure_still_counts_as_missing(self):
        """The symmetric check: an OOM must not be laundered into a hole."""
        gt, env, _ = self._env_with_hole()
        oom = next(c for c in env.cells()
                   if env.coords(c)["seq_len"] == 2048 and env.coords(c)["batch"] == 8)
        env.mark_missing("method", oom, "CUDA out of memory")
        cmps = compare_cells(env, "method", "baseline", mde=0.05, q=0.05, seed=0)
        lc = extract_loss_column(env, cmps, method="method", baseline="baseline",
                                 mde=0.05, q_level=0.05)
        assert env.coverage()["method"] < 1.0
        assert lc.counts["inapplicable"] > 0
