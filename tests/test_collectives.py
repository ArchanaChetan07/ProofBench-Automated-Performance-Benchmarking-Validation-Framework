"""How many bytes a collective moves, checked against a figure from outside.

The formula was written out by hand in three places and two of them were wrong
in the same way for the whole life of the project: all_gather understated by a
factor of the world size. It survived because the number was only ever compared
with itself. These tests compare it with the nccl-tests convention, which is
the thing anyone reading a bandwidth figure will assume.
"""

from __future__ import annotations

import pytest

from losscolumn.core.collectives import effective_bytes

MiB = 1 << 20


class TestAgainstNcclTests:
    """`busbw` as nccl-tests defines it, expressed from the per-rank input.

    all_reduce: output is the same size as input, so 2(w-1)/w times it.
    all_gather: output is w times the input and the convention is stated
    against the output, so (w-1) times the input -- NOT (w-1)/w, which is the
    error this file exists to prevent returning.
    """

    @pytest.mark.parametrize("world,expected", [
        (2, 1.0), (3, 4 / 3), (4, 1.5), (8, 1.75),
    ])
    def test_all_reduce(self, world, expected):
        assert effective_bytes("all_reduce", world, MiB) == pytest.approx(
            expected * MiB)

    @pytest.mark.parametrize("world,expected", [
        (2, 1.0), (3, 2.0), (4, 3.0), (8, 7.0),
    ])
    def test_all_gather(self, world, expected):
        assert effective_bytes("all_gather", world, MiB) == pytest.approx(
            expected * MiB)

    def test_all_gather_is_not_the_all_reduce_shape(self):
        """The specific confusion, pinned.

        (w-1)/w is bounded above by 1 and (w-1) is not, so reading one as the
        other understates by exactly the world size. It is how a PCIe 4.0 x16
        link came to be reported at 0.5 GB/s.
        """
        for w in (2, 3, 4, 8):
            wrong = (w - 1) / w * MiB
            assert effective_bytes("all_gather", w, MiB) == pytest.approx(
                wrong * w)


class TestProperties:
    def test_all_gather_grows_without_bound_in_world_size(self):
        """A larger group moves strictly more data per call, unboundedly."""
        vals = [effective_bytes("all_gather", w, MiB) for w in (2, 4, 8, 16)]
        assert vals == sorted(vals)
        assert vals[-1] / vals[0] == pytest.approx(15.0)

    def test_all_reduce_saturates_below_twice_the_buffer(self):
        """2(w-1)/w approaches 2 and never reaches it, however large the group."""
        for w in (2, 8, 64, 1024):
            assert effective_bytes("all_reduce", w, MiB) < 2 * MiB
        assert effective_bytes("all_reduce", 1024, MiB) / MiB > 1.99

    def test_a_single_rank_moves_its_buffer_once(self):
        for kind in ("all_reduce", "all_gather"):
            assert effective_bytes(kind, 1, MiB) == MiB

    def test_an_unknown_collective_is_counted_once(self):
        assert effective_bytes("reduce_scatter", 4, MiB) == MiB


class TestEverySiteAgrees:
    """Three call sites, one definition. Two of them used to disagree."""

    def test_sample_and_sentinel_match_the_shared_definition(self):
        from losscolumn.core.stability import SentinelReading
        from losscolumn.thrusts.overlap.commodel import Sample

        for world in (2, 3, 4, 8):
            for kind in ("all_reduce", "all_gather"):
                want = effective_bytes(kind, world, MiB)
                s = Sample(nbytes=MiB, seconds=1e-3, world=world, kind=kind)
                assert s.effective_bytes == pytest.approx(want), (kind, world)

                r = SentinelReading(
                    session_id="s", restart_index=0, collective=kind,
                    world=world, nbytes=MiB, size_class="m", timings_s=[1e-3])
                assert r.bandwidth_gbs == pytest.approx(want / 1e-3 / 1e9), (
                    kind, world)

    def test_the_campaign_finaliser_matches_too(self):
        """The path a real record takes, rather than a reconstruction of it."""
        from losscolumn.thrusts.overlap.campaign import _finalise

        for world in (2, 4, 8):
            for kind in ("all_reduce", "all_gather"):
                rec = _finalise({
                    "collective": kind, "world": world, "nbytes": MiB,
                    "pass_index": 0, "timings_s": [1e-3, 1.1e-3],
                    "n_failures": 0, "invalid_reason": "",
                })
                assert rec.effective_bytes == pytest.approx(
                    effective_bytes(kind, world, MiB)), (kind, world)


class TestPartialAcceptance:
    """A model graded on a corner of the surface is not a usable model.

    The gate is scored over the regimes able to grade it, so that an
    unmeasurable part of the surface cannot veto a family that fits the
    measurable part. Applied where MOST regimes are ungradable, that rule
    produces a clean pass on very little evidence: all_reduce/world4 on the
    rented box fits `large` at 1.1% with the other three regimes bimodal, and
    its worst-regime error across the whole surface is 94.5%.
    """

    def test_partial_is_not_usable(self):
        from losscolumn.core.coverage import ParameterVerdict

        assert not ParameterVerdict.PARTIAL.usable
        assert ParameterVerdict.ACCEPTED.usable

    def test_partial_is_distinct_from_diagnostic(self):
        """Nothing failed. The fit is good where it was tested."""
        from losscolumn.core.coverage import ParameterVerdict

        assert ParameterVerdict.PARTIAL is not ParameterVerdict.DIAGNOSTIC
        assert ParameterVerdict.PARTIAL is not ParameterVerdict.REJECTED

    def test_the_threshold_is_more_than_one_regime(self):
        from losscolumn.thrusts.overlap.campaign import MIN_GRADABLE_REGIMES

        assert MIN_GRADABLE_REGIMES >= 2, (
            "one gradable regime is a corner, not a transport")

    def test_every_verdict_states_whether_it_may_be_consumed(self):
        from losscolumn.core.coverage import ParameterVerdict

        usable = {v for v in ParameterVerdict if v.usable}
        assert usable == {ParameterVerdict.ACCEPTED}, (
            "exactly one verdict licenses use, and it is the strict one")
