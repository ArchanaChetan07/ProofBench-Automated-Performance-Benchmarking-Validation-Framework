"""Interval algebra and overlap attribution.

Overlap efficiency is the headline metric of Thrust I, and it is computed
entirely from these two layers. Each test below checks a case where the naive
implementation -- summing kernel durations -- gives a different, wrong answer.
"""

from __future__ import annotations

import pytest

from losscolumn.core.intervals import IntervalSet, coverage_profile
from losscolumn.thrusts.overlap.attrib import Span, StepTrace, classify


class TestIntervalSet:
    def test_merges_overlapping_and_touching(self):
        s = IntervalSet.from_spans([(0, 5), (3, 8), (8, 10), (20, 22)])
        assert s.to_list() == [[0, 10], [20, 22]]
        assert s.measure() == 12

    def test_drops_empty_intervals(self):
        assert IntervalSet.from_spans([(1, 1), (2, 2)]).to_list() == []

    def test_union_is_idempotent(self):
        a = IntervalSet.from_spans([(0, 10), (20, 30)])
        assert (a | a).to_list() == a.to_list()

    def test_difference(self):
        a = IntervalSet.from_spans([(0, 10), (20, 30)])
        b = IntervalSet.from_spans([(5, 25)])
        assert (a - b).to_list() == [[0, 5], [25, 30]]
        assert (a - b).measure() == 10

    def test_difference_with_a_hole_inside(self):
        a = IntervalSet.from_spans([(0, 10)])
        b = IntervalSet.from_spans([(3, 4), (6, 7)])
        assert (a - b).to_list() == [[0, 3], [4, 6], [7, 10]]

    def test_intersection(self):
        a = IntervalSet.from_spans([(0, 10), (20, 30)])
        b = IntervalSet.from_spans([(5, 25)])
        assert (a & b).to_list() == [[5, 10], [20, 25]]

    def test_difference_with_disjoint_set_is_identity(self):
        a = IntervalSet.from_spans([(0, 10)])
        b = IntervalSet.from_spans([(50, 60)])
        assert (a - b).to_list() == a.to_list()

    def test_full_cover_leaves_nothing(self):
        a = IntervalSet.from_spans([(2, 8)])
        b = IntervalSet.from_spans([(0, 10)])
        assert (a - b).to_list() == []

    def test_gaps_are_the_bubbles(self):
        s = IntervalSet.from_spans([(0, 2), (5, 7), (9, 10)])
        assert s.gaps().to_list() == [[2, 5], [7, 9]]
        assert s.span() == 10
        assert s.measure() == 5

    def test_inclusion_exclusion_identity(self):
        a = IntervalSet.from_spans([(0, 7), (10, 15)])
        b = IntervalSet.from_spans([(3, 12), (14, 20)])
        assert (a | b).measure() == pytest.approx(
            a.measure() + b.measure() - (a & b).measure()
        )

    def test_coverage_profile_finds_uncovered_time(self):
        prof = coverage_profile(
            {"compute": IntervalSet.from_spans([(0, 3)]),
             "comm": IntervalSet.from_spans([(2, 5)])},
            0, 8,
        )
        idle = [p for p in prof if not p["live"]]
        assert idle and idle[0]["start"] == 5 and idle[0]["end"] == 8


class TestClassification:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("ncclDevKernel_AllGather_RING", "comm"),
            ("nccl_reduce_scatter", "comm"),
            ("ampere_bf16_gemm_128x128", "compute"),
            ("void at::native::vectorized_elementwise_kernel", "compute"),
            ("Memcpy DtoD", "memory"),
            ("some_unknown_thing", "other"),
        ],
    )
    def test_names(self, name, expected):
        assert classify(name) == expected

    def test_comm_wins_over_compute_when_both_match(self):
        # A fused name containing both markers must be communication: getting
        # this backwards inflates overlap efficiency toward 1.
        assert classify("nccl_allreduce_with_gemm_fusion") == "comm"


class TestAttribution:
    def test_fully_hidden_communication(self):
        t = StepTrace(
            spans=[
                Span("gemm", 0, 10, category="compute"),
                Span("nccl_ag", 2, 6, category="comm"),
            ],
            step_start=0, step_end=10,
        )
        m = t.attribute()
        assert m.overlap_efficiency == pytest.approx(1.0)
        assert m.exposed_ms == 0.0
        assert not m.comm_bound

    def test_fully_exposed_communication(self):
        t = StepTrace(
            spans=[
                Span("gemm", 0, 5, category="compute"),
                Span("nccl_ag", 5, 10, category="comm"),
            ],
            step_start=0, step_end=10,
        )
        m = t.attribute()
        assert m.overlap_efficiency == pytest.approx(0.0)
        assert m.exposed_ms == pytest.approx(5.0)
        assert m.exposed_frac_of_step == pytest.approx(0.5)
        assert m.comm_bound

    def test_partial_overlap(self):
        t = StepTrace(
            spans=[
                Span("gemm", 0, 6, category="compute"),
                Span("nccl_ag", 4, 10, category="comm"),
            ],
            step_start=0, step_end=10,
        )
        m = t.attribute()
        assert m.exposed_ms == pytest.approx(4.0)
        assert m.overlap_efficiency == pytest.approx(1 - 4 / 6)

    def test_concurrent_collectives_are_not_double_counted(self):
        """Two collectives running at once on different streams occupy one
        interval of wall-clock, not two. Summing durations would report 10 ms
        of communication where the timeline holds 6."""
        t = StepTrace(
            spans=[
                Span("gemm", 0, 2, category="compute"),
                Span("nccl_a", 2, 7, stream=2, category="comm"),
                Span("nccl_b", 3, 8, stream=3, category="comm"),
            ],
            step_start=0, step_end=8,
        )
        m = t.attribute()
        assert m.comm_ms == pytest.approx(6.0)      # union of [2,7] and [3,8]
        assert sum(s.dur for s in t.spans if s.category == "comm") == 10.0

    def test_efficiency_is_bounded(self):
        """However pathological the trace, the metric stays in [0, 1]."""
        t = StepTrace(
            spans=[
                Span("gemm", 0, 1, category="compute"),
                Span("nccl_a", 0, 9, stream=2, category="comm"),
                Span("nccl_b", 1, 9, stream=3, category="comm"),
                Span("nccl_c", 2, 9, stream=4, category="comm"),
            ],
            step_start=0, step_end=9,
        )
        m = t.attribute()
        assert 0.0 <= m.overlap_efficiency <= 1.0

    def test_idle_time_is_reported_not_absorbed(self):
        t = StepTrace(
            spans=[
                Span("gemm", 0, 2, category="compute"),
                Span("nccl", 6, 8, category="comm"),
            ],
            step_start=0, step_end=10,
        )
        m = t.attribute()
        assert m.idle_ms == pytest.approx(6.0)
        assert m.idle_frac == pytest.approx(0.6)

    def test_no_communication_gives_nan_not_one(self):
        """An engine with no collectives has undefined overlap efficiency.
        Reporting 1.0 would let a single-GPU run look perfectly overlapped."""
        import math

        t = StepTrace(spans=[Span("gemm", 0, 5, category="compute")],
                      step_start=0, step_end=5)
        assert math.isnan(t.attribute().overlap_efficiency)

    def test_unclassified_kernels_are_surfaced(self):
        t = StepTrace(
            spans=[Span("mystery_kernel", 0, 3), Span("gemm", 3, 5, category="compute")],
            step_start=0, step_end=5,
        )
        m = t.attribute()
        assert "mystery_kernel" in m.unclassified_kernels
        assert m.unclassified_ms == pytest.approx(3.0)


class TestChromeTraceIngest:
    def test_microseconds_are_converted(self):
        t = StepTrace.from_records(
            [{"name": "ncclAllGather", "ts": 1000, "dur": 2000}]
        )
        assert t.spans[0].start == pytest.approx(1.0)
        assert t.spans[0].end == pytest.approx(3.0)
