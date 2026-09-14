"""A model that answers any question it is asked.

block-memory-v2 was fitted on a T1000: 8.6 GB, hidden 1024, heads 16, ffn 4096,
micro-batch 64-128, sequence 2048-4096. Every term is linear in tokens, hidden
or parameter count, so it returns a well-formed number for any shape at all.

This was unreachable locally. The development machine cannot HOLD a shape
outside that box, so every question it was ever asked was in range. The first
out-of-range question arrives on rented hardware, and the answer looks exactly
like the validated ones.
"""

from __future__ import annotations

import pytest

from losscolumn.core.memory_model import (
    MAX_EXCURSION,
    V2_DOMAIN,
    BlockMemoryModel,
    BlockShape,
    ExtrapolationError,
)


def _fitted(**kw):
    """A shape inside the region the parameters were fitted over."""
    base = dict(hidden=1024, heads=16, ffn=4096, micro_batch=64, seq_len=2048)
    base.update(kw)
    return BlockShape(**base)


def _rental():
    """What a 96 GB card gets asked about: a real model, not a local block."""
    return BlockShape(hidden=8192, heads=64, ffn=28672, micro_batch=8,
                      seq_len=32768)


class TestTheDomainIsRecorded:
    def test_it_matches_the_calibration_artifact(self):
        """Read from the artifact, not chosen to be convenient."""
        assert V2_DOMAIN.micro_batch == (64, 128)
        assert V2_DOMAIN.seq_len == (2048, 4096)
        assert V2_DOMAIN.hidden == (1024, 1024)
        assert "T1000" in V2_DOMAIN.device

    def test_the_frozen_parameters_are_untouched(self):
        """Recording where a model was fitted is not changing it."""
        p = BlockMemoryModel().params
        assert p.get("activation_tensors_per_block") == 4.0
        assert p.get("safety_margin_fraction") == 0.10
        assert p.get("checkpoint_retained_fraction") == 0.25


class TestInsideTheBox:
    def test_a_fitted_shape_is_an_interpolation(self):
        assert BlockMemoryModel().is_interpolation(_fitted())

    def test_and_reports_no_excursion(self):
        assert BlockMemoryModel().excursions(_fitted()) == {}

    def test_strict_permits_it(self):
        assert BlockMemoryModel().predict_gb(_fitted(), strict=True) > 0


class TestOutsideTheBox:
    def test_a_rental_shape_is_far_outside_on_several_axes(self):
        e = BlockMemoryModel().excursions(_rental())
        assert set(e) >= {"hidden", "seq_len", "micro_batch"}
        assert e["hidden"] == pytest.approx(8.0)
        assert e["seq_len"] == pytest.approx(8.0)

    def test_it_still_returns_a_plausible_number(self):
        """The whole danger: nothing about the value says it is unchecked."""
        gb = BlockMemoryModel().predict_gb(_rental())
        assert 10 < gb < 500, "a well-formed, entirely unvalidated answer"

    def test_strict_refuses_and_says_how_far(self):
        with pytest.raises(ExtrapolationError, match="outside the range"):
            BlockMemoryModel().predict_gb(_rental(), strict=True)

    def test_the_refusal_names_the_axis_and_the_hardware(self):
        with pytest.raises(ExtrapolationError) as e:
            BlockMemoryModel().predict_gb(_rental(), strict=True)
        msg = str(e.value)
        assert "hidden" in msg and "T1000" in msg
        assert "re-calibrate" in msg.lower()

    def test_below_the_range_counts_too(self):
        """Extrapolation is not only upward.

        A micro-batch of 8 is as far outside a 64-128 fit as 1024 would be, and
        the small-batch end is where a real serving workload lives.
        """
        e = BlockMemoryModel().excursions(_fitted(micro_batch=8))
        assert e.get("micro_batch") == pytest.approx(8.0)


class TestTheBoundary:
    def test_just_inside_the_strict_limit_is_allowed(self):
        shape = _fitted(seq_len=int(4096 * MAX_EXCURSION))
        assert BlockMemoryModel().predict_gb(shape, strict=True) > 0

    def test_just_outside_is_not(self):
        shape = _fitted(seq_len=int(4096 * MAX_EXCURSION) + 4096)
        with pytest.raises(ExtrapolationError):
            BlockMemoryModel().predict_gb(shape, strict=True)

    def test_a_mild_excursion_is_flagged_but_not_refused(self):
        """Flagging and refusing are different thresholds on purpose.

        Twice outside a fitted range is worth recording and usually worth
        acting on; refusing it would make the model useless at the edges it
        was built to explore.
        """
        m, shape = BlockMemoryModel(), _fitted(seq_len=8192)
        assert m.excursions(shape)["seq_len"] == pytest.approx(2.0)
        assert m.predict_gb(shape, strict=True) > 0
