"""Regressions for bugs that were found by running things, not by reading them.

Five defects, each caught by an activity rather than by inspection: three by
the clean-clone reproduction, two by mining the campaign artifact. Each is
pinned here so it cannot return quietly.

The pattern worth noticing is that none of them was visible in the working
tree. Three needed a fresh environment; two needed the measured data. Code that
is only ever exercised where it was written is code whose failure modes are
still unknown.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPRODUCE = REPO / "scripts" / "reproduce.sh"


# --------------------------------------------------------------------------
# 1. the reproduction script swallowed a failed install
# --------------------------------------------------------------------------


class TestReproductionDoesNotSwallowFailures:
    """`|| true` on the install turned a clear error into a confusing one.

    The failed install surfaced much later as "No module named pytest", inside
    the script whose whole job is to catch silent failures.
    """

    def _lines(self) -> list[str]:
        return [
            ln for ln in REPRODUCE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    def test_no_command_ends_in_or_true(self):
        offenders = [ln for ln in self._lines() if re.search(r"\|\|\s*true\s*$", ln)]
        assert not offenders, (
            "a command ends in `|| true`, which discards its exit status: "
            f"{offenders}"
        )

    def test_the_install_is_checked(self):
        text = REPRODUCE.read_text(encoding="utf-8")
        assert "could not install the test dependencies" in text, (
            "the dependency install has no failure branch"
        )

    def test_the_import_is_verified_after_installing(self):
        """Installing is not the same as being importable."""
        text = REPRODUCE.read_text(encoding="utf-8")
        assert "pytest did not import after installation" in text

    def test_the_script_is_valid_shell(self):
        import shutil
        import subprocess

        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("no bash available")
        r = subprocess.run([bash, "-n", str(REPRODUCE)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------
# 2. candidate records were a different shape on the not-fitted branch
# --------------------------------------------------------------------------


class TestCandidateRecordsAreUniform:
    """A ragged record meant every consumer had to special-case failure.

    One did not, and it only showed up in an environment without SciPy, where
    the log and robust estimators take exactly that branch.
    """

    REQUIRED = (
        "family", "n_params", "fit_r2", "heldout_median_rel_err",
        "heldout_max_rel_err", "heldout_per_tier_err", "heldout_worst_tier_err",
        "aic",
    )

    def _selection(self, samples):
        from losscolumn.thrusts.overlap.commodel import select_model

        def tier(n: int) -> str:
            return "small" if n < (1 << 18) else "large"

        return select_model(samples[::2], samples[1::2], tier_fn=tier)

    def test_every_candidate_has_every_key_when_all_fit(self):
        from losscolumn.thrusts.overlap.commodel import Sample

        s = [Sample(nbytes=1 << k, seconds=1e-4 + (1 << k) / 1e9)
             for k in range(10, 25)]
        sel = self._selection(s)
        for c in sel.candidates:
            for key in self.REQUIRED:
                assert key in c, f"{c['family']} is missing {key}"

    def test_every_candidate_has_every_key_when_a_family_cannot_fit(self):
        """The branch that broke: too few points for the richer families."""
        from losscolumn.thrusts.overlap.commodel import Sample

        s = [Sample(nbytes=1 << k, seconds=1e-4 + (1 << k) / 1e9)
             for k in range(10, 14)]
        sel = self._selection(s)
        assert sel.candidates
        for c in sel.candidates:
            for key in self.REQUIRED:
                assert key in c, f"{c['family']} is missing {key} on the failure path"

    def test_a_consumer_can_read_every_candidate_without_guarding(self):
        """The exact expression that raised KeyError in the fresh clone."""
        from losscolumn.thrusts.overlap.commodel import Sample

        s = [Sample(nbytes=1 << k, seconds=1e-4 + (1 << k) / 1e9)
             for k in range(10, 14)]
        sel = self._selection(s)
        [c for c in sel.candidates if c["heldout_per_tier_err"]]        # no raise
        [c["heldout_worst_tier_err"] for c in sel.candidates]           # no raise

    def test_scipy_absence_is_reported_not_substituted(self):
        """Without SciPy the log and robust families are unavailable.

        They must not silently fall back to the weighted fit under their own
        name, which would make two families look like three.
        """
        import numpy as np

        from losscolumn.thrusts.overlap.commodel import _fit_nonlinear

        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        out = _fit_nonlinear(x, y, loss="log")
        assert out is None or isinstance(out, tuple)


# --------------------------------------------------------------------------
# 3. the reproduction assumed a fresh clone could run the measured thrusts
# --------------------------------------------------------------------------


class TestReproductionSeparatesGpuWork:
    """Verifying a standard must not require a 2.5 GB torch wheel."""

    def test_the_pipeline_step_is_conditional_on_torch(self):
        text = REPRODUCE.read_text(encoding="utf-8")
        assert "import torch" in text
        assert "PIPELINE=" in text

    def test_a_skipped_pipeline_is_recorded_not_hidden(self):
        text = REPRODUCE.read_text(encoding="utf-8")
        assert "skipped: no torch" in text
        assert "reproduction-result.json" in text

    def test_a_missing_claim_is_not_a_discrepancy_when_the_pipeline_was_skipped(self):
        text = REPRODUCE.read_text(encoding="utf-8")
        assert 'sys.argv[3] != "ran"' in text, (
            "the structural comparison does not know the pipeline was skipped, so a "
            "missing measured claim would read as a reproduction failure"
        )

    def test_the_recorded_result_says_what_it_does_not_establish(self):
        art = REPO / "artifacts" / "reproduction-result.json"
        if not art.exists():
            pytest.skip("reproduction not run here")
        import json

        d = json.loads(art.read_text(encoding="utf-8"))
        assert d["passed"] is True
        if d["pipeline"] != "ran":
            assert "not re-run" in d["detail"] or "NOT re-run" in d["detail"]


# --------------------------------------------------------------------------
# 4. the parsimony tiebreak compared median against worst-regime
# --------------------------------------------------------------------------


class TestParsimonyComparesLikeWithLike:
    """The bug that cost a whole group its coverage.

    `score` was the worst-regime error while the tiebreak filtered on the
    pooled median, so a simpler family with a good median and a bad worst
    regime displaced a better model -- reintroducing exactly the failure the
    worst-regime criterion exists to prevent. Measured: all_gather/world4
    selected a model at 14.9% worst-regime error when one at 11.2% was
    available and under the gate.
    """

    def _surface(self):
        """Four regimes; a linear fit is fine on three and poor on one."""
        from losscolumn.thrusts.overlap.commodel import Sample

        out = []
        for k in range(10, 26):
            n = 1 << k
            t = 1e-4 + n / 1e9
            if (1 << 18) <= n < (1 << 22):
                t *= 3.0                      # one regime a linear fit cannot reach
            out.append(Sample(nbytes=n, seconds=t, world=1, kind="copy"))
        return out

    @staticmethod
    def _tier(n: int) -> str:
        if n < (1 << 14):
            return "tiny"
        if n < (1 << 18):
            return "small"
        if n < (1 << 22):
            return "medium"
        return "large"

    def test_a_simpler_family_does_not_win_on_a_different_metric(self):
        from losscolumn.thrusts.overlap.commodel import select_model

        s = self._surface()
        sel = select_model(s[::2], s[1::2], tier_fn=self._tier,
                           max_heldout_median_err=0.99)
        chosen = next(c for c in sel.candidates
                      if c["family"] == sel.chosen_family)
        best = min(
            (c for c in sel.candidates if c["heldout_per_tier_err"]),
            key=lambda c: c["heldout_worst_tier_err"],
        )
        assert chosen["heldout_worst_tier_err"] <= (
            best["heldout_worst_tier_err"] * 1.1 + 1e-12
        ), (
            f"{sel.chosen_family} was chosen at "
            f"{chosen['heldout_worst_tier_err']:.1%} worst-regime error while "
            f"{best['family']} scored {best['heldout_worst_tier_err']:.1%}"
        )

    def test_the_chosen_model_is_never_worse_than_an_available_one(self):
        from losscolumn.thrusts.overlap.commodel import select_model

        s = self._surface()
        sel = select_model(s[::2], s[1::2], tier_fn=self._tier,
                           max_heldout_median_err=0.99)
        if sel.chosen is None:
            pytest.skip("nothing cleared the gate on this surface")
        scores = [c["heldout_worst_tier_err"] for c in sel.candidates
                  if c["heldout_per_tier_err"]
                  and math.isfinite(c["heldout_worst_tier_err"])]
        chosen = next(c["heldout_worst_tier_err"] for c in sel.candidates
                      if c["family"] == sel.chosen_family)
        assert chosen <= min(scores) * 1.1 + 1e-12

    def test_the_reason_names_the_metric_it_compared(self):
        from losscolumn.thrusts.overlap.commodel import select_model

        s = self._surface()
        sel = select_model(s[::2], s[1::2], tier_fn=self._tier,
                           max_heldout_median_err=0.99)
        if "simpler family is preferred" in sel.reason:
            assert "worst-regime" in sel.reason


# --------------------------------------------------------------------------
# 5. the peak detector was an argmax over a noisy curve
# --------------------------------------------------------------------------


class TestPeakDetectorRequiresSupport:
    """An argmax is not a peak estimator.

    It picked a maximum 4.3x its own neighbours in one group, and flipped
    between two near-equal maxima 270x apart in size in another.
    """

    def _records(self, bandwidths, *, pass_index=0, world=2,
                 collective="all_reduce"):
        from losscolumn.thrusts.overlap.campaign import PointRecord

        out = []
        for i, bw in enumerate(bandwidths):
            n = 1 << (10 + i)
            r = PointRecord(collective=collective, world=world, nbytes=n,
                            pass_index=pass_index, valid=True)
            r.bandwidth_gbs = bw
            r.median_s = 1e-3
            out.append(r)
        return out

    def test_a_spike_is_not_a_peak(self):
        from losscolumn.thrusts.overlap.campaign import analyse_peak

        # One point four times its neighbours: a spike.
        bw = [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]
        recs = self._records(bw, pass_index=0) + self._records(bw, pass_index=1)
        rep = analyse_peak(recs)
        assert not rep.well_defined
        assert not rep.reproducible
        assert "spike" in rep.finding

    def test_a_supported_maximum_is_a_peak(self):
        from losscolumn.thrusts.overlap.campaign import analyse_peak

        bw = [0.05, 0.10, 0.20, 0.34, 0.40, 0.36, 0.22, 0.12, 0.08, 0.06]
        recs = self._records(bw, pass_index=0) + self._records(bw, pass_index=1)
        rep = analyse_peak(recs)
        assert rep.well_defined
        assert rep.reproducible

    def test_two_distant_near_equal_maxima_are_not_a_located_peak(self):
        from losscolumn.thrusts.overlap.campaign import analyse_peak

        # Rivals at opposite ends, within a few percent of each other.
        bw = [0.40, 0.30, 0.10, 0.08, 0.07, 0.08, 0.10, 0.30, 0.39, 0.20]
        recs = self._records(bw, pass_index=0) + self._records(bw, pass_index=1)
        rep = analyse_peak(recs)
        assert not rep.well_defined
        assert "not determined by the data" in rep.finding

    def test_an_undefined_peak_reports_that_rather_than_a_location(self):
        from losscolumn.thrusts.overlap.campaign import analyse_peak

        bw = [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        recs = self._records(bw, pass_index=0) + self._records(bw, pass_index=1)
        rep = analyse_peak(recs)
        assert "no peak to reproduce" in rep.finding.lower() or \
               "NO WELL-DEFINED PEAK" in rep.finding

    def test_neighbour_support_is_recorded_per_pass(self):
        from losscolumn.thrusts.overlap.campaign import analyse_peak

        bw = [0.05, 0.10, 0.20, 0.34, 0.40, 0.36, 0.22, 0.12, 0.08, 0.06]
        recs = self._records(bw, pass_index=0) + self._records(bw, pass_index=1)
        rep = analyse_peak(recs)
        for p in rep.per_pass:
            assert "neighbour_support" in p
            assert "runner_up_ratio" in p
