"""The replacement memory model, and the protocol that keeps its grading honest.

Version 1 failed structurally: it applied one multiplier to the whole
activation expression, so checkpointing scaled terms it does not physically
touch. Measurement found five false wins in eight cells. These tests pin the
properties that failure motivated, so a future edit cannot quietly reintroduce
the same shape of error.
"""

from __future__ import annotations

import math

import pytest

from losscolumn.core.gridsplit import GridSplit, SplitViolation, preflight, select_boundary_grid
from losscolumn.core.memory_model import (
    GB,
    BlockMemoryModel,
    BlockShape,
    ScalingMap,
    default_parameters,
)
from losscolumn.core.parameters import Parameter, Provenance

BUDGET = 8.0 * GB


def shape(mb=32, seq=1024, ckpt=False, **kw) -> BlockShape:
    base = dict(hidden=1024, heads=16, ffn=4096, micro_batch=mb, seq_len=seq,
                checkpointing=ckpt)
    base.update(kw)
    return BlockShape(**base)


class TestComponentTerms:
    def test_every_term_is_reported_separately(self):
        t = BlockMemoryModel().terms(shape())
        d = t.to_dict()
        for k in ("weights", "gradients", "saved_activations", "attention_workspace",
                  "ffn_workspace", "autograd_overhead", "allocator_reserve", "context"):
            assert k in d, k

    def test_terms_sum_to_the_total(self):
        t = BlockMemoryModel().terms(shape())
        assert t.total == pytest.approx(t.subtotal + t.allocator_reserve)
        assert t.subtotal == pytest.approx(t.token_dependent + t.token_independent)

    def test_token_dependent_terms_scale_with_tokens(self):
        m = BlockMemoryModel()
        a, b = m.terms(shape(mb=32)), m.terms(shape(mb=64))
        assert b.token_dependent == pytest.approx(2 * a.token_dependent)

    def test_weights_do_not_scale_with_tokens(self):
        m = BlockMemoryModel()
        assert m.terms(shape(mb=64)).weights == pytest.approx(
            m.terms(shape(mb=8)).weights
        )

    def test_sequence_length_and_batch_are_interchangeable_for_activations(self):
        """Both enter only through the token count, which is the honest claim."""
        m = BlockMemoryModel()
        a = m.terms(shape(mb=64, seq=1024)).token_dependent
        b = m.terms(shape(mb=32, seq=2048)).token_dependent
        assert a == pytest.approx(b)

    def test_ffn_expansion_moves_only_the_ffn_term(self):
        m = BlockMemoryModel()
        a, b = m.terms(shape(ffn=4096)), m.terms(shape(ffn=8192))
        assert b.ffn_workspace == pytest.approx(2 * a.ffn_workspace)
        assert b.attention_workspace == pytest.approx(a.attention_workspace)
        assert b.saved_activations == pytest.approx(a.saved_activations)

    def test_layer_count_scales_saved_activations_but_not_workspace(self):
        """Workspace is live inside one block; saved activations accumulate."""
        m = BlockMemoryModel()
        a, b = m.terms(shape(n_layers=1)), m.terms(shape(n_layers=4))
        assert b.saved_activations == pytest.approx(4 * a.saved_activations)
        assert b.attention_workspace == pytest.approx(a.attention_workspace)
        assert b.ffn_workspace == pytest.approx(a.ffn_workspace)


class TestPrecision:
    def test_halving_the_element_size_halves_the_tensor_terms(self):
        m = BlockMemoryModel()
        fp16 = m.terms(shape(bytes_per_elem=2))
        fp32 = m.terms(shape(bytes_per_elem=4))
        assert fp32.saved_activations == pytest.approx(2 * fp16.saved_activations)
        assert fp32.attention_workspace == pytest.approx(2 * fp16.attention_workspace)
        assert fp32.weights == pytest.approx(2 * fp16.weights)

    def test_context_is_independent_of_precision(self):
        m = BlockMemoryModel()
        assert m.terms(shape(bytes_per_elem=4)).context == pytest.approx(
            m.terms(shape(bytes_per_elem=2)).context
        )


class TestCheckpointingIsSurgical:
    """The structural defect of version 1, pinned so it cannot come back."""

    def test_checkpointing_reduces_saved_activations(self):
        m = BlockMemoryModel()
        on, off = m.terms(shape(ckpt=True)), m.terms(shape(ckpt=False))
        assert on.saved_activations < off.saved_activations

    @pytest.mark.parametrize(
        "term",
        ["weights", "gradients", "optimizer_state", "attention_workspace",
         "ffn_workspace", "autograd_overhead", "context"],
    )
    def test_checkpointing_touches_nothing_else(self, term):
        """A recomputed block still materialises its full workspace."""
        m = BlockMemoryModel()
        on, off = m.terms(shape(ckpt=True)), m.terms(shape(ckpt=False))
        assert getattr(on, term) == pytest.approx(getattr(off, term)), term

    def test_checkpointing_is_not_a_global_multiplier(self):
        """The version 1 error: one ratio applied to everything.

        If checkpointing were a global multiplier, the ratio of totals would
        equal the ratio of every individual term. It must not.
        """
        m = BlockMemoryModel()
        on, off = m.terms(shape(ckpt=True)), m.terms(shape(ckpt=False))
        total_ratio = on.total / off.total
        ws_ratio = on.ffn_workspace / off.ffn_workspace
        assert ws_ratio == pytest.approx(1.0)
        assert total_ratio < 1.0
        assert not math.isclose(total_ratio, ws_ratio, rel_tol=1e-6)

    def test_checkpointing_saves_less_at_large_workspace(self):
        """Its benefit shrinks as workspace dominates -- which a multiplier cannot express."""
        m = BlockMemoryModel()
        small = m.terms(shape(ffn=1024, ckpt=True)).total / m.terms(shape(ffn=1024)).total
        large = m.terms(shape(ffn=16384, ckpt=True)).total / m.terms(shape(ffn=16384)).total
        assert large > small


class TestSafetyMargin:
    def test_required_exceeds_estimated_by_the_registered_fraction(self):
        m = BlockMemoryModel()
        s = shape()
        frac = m.params.get("safety_margin_fraction")
        assert m.required_bytes(s) == pytest.approx(m.predict_bytes(s) * (1 + frac))

    def test_margin_is_applied_at_every_shape(self):
        m = BlockMemoryModel()
        for mb in (8, 64, 256):
            s = shape(mb=mb)
            assert m.required_bytes(s) > m.predict_bytes(s)

    def test_margin_makes_the_decision_conservative(self):
        """A cell just inside the raw estimate is refused once the margin applies."""
        m = BlockMemoryModel()
        s = shape(mb=64, seq=2048)
        budget = m.predict_bytes(s) * 1.02      # fits raw, not with a 10% margin
        assert not m.fits(s, budget)

    def test_margin_is_registered_not_fitted(self):
        """The structural guarantee, plus the recorded policy behind it."""
        p = default_parameters().params["safety_margin_fraction"]
        assert p.provenance is Provenance.REGISTERED
        assert p.provenance is not Provenance.FITTED
        recorded = f"{p.source} {p.note}".lower()
        assert "not tuned" in recorded or "fixed in the sealed protocol" in recorded


class TestBoundaryGeometry:
    def test_boundary_distance_is_zero_on_the_boundary(self):
        m = BlockMemoryModel()
        s = shape()
        assert m.boundary_distance(s, m.required_bytes(s)) == pytest.approx(0.0)

    def test_headroom_is_negative_when_infeasible(self):
        m = BlockMemoryModel()
        s = shape(mb=256, seq=4096)
        assert m.headroom(s, 1.0 * GB) < 0
        assert not m.fits(s, 1.0 * GB)


class TestScalingIsExplicit:
    def test_the_map_names_what_transfers_and_what_does_not(self):
        sm = ScalingMap(local=shape(mb=1, seq=1), target_hidden=4096,
                        target_heads=32, target_ffn=11008, target_layers=32)
        d = sm.to_dict()
        assert d["parameters_claimed_to_transfer"]
        assert d["parameters_re_registered_per_target"]
        assert set(d["parameters_claimed_to_transfer"]).isdisjoint(
            d["parameters_re_registered_per_target"]
        )

    def test_context_bytes_is_not_claimed_to_transfer(self):
        """It depends on the driver and the card, so it must be re-registered."""
        sm = ScalingMap(local=shape(), target_hidden=4096, target_heads=32,
                        target_ffn=11008, target_layers=32)
        assert "context_bytes" in sm.re_registered
        assert "context_bytes" not in sm.transfers

    def test_target_shape_carries_the_target_architecture(self):
        sm = ScalingMap(local=shape(), target_hidden=4096, target_heads=32,
                        target_ffn=11008, target_layers=32)
        t = sm.to_target(micro_batch=4, seq_len=2048, checkpointing=True)
        assert (t.hidden, t.n_layers, t.checkpointing) == (4096, 32, True)
        assert sm.width_ratio == pytest.approx(4.0)


class TestProtocolIsolation:
    def test_a_cell_cannot_be_in_both_grids(self):
        with pytest.raises(SplitViolation, match="both the calibration"):
            GridSplit(calibration=((1, 2, False),), validation=((1, 2, False),))

    def test_validation_cells_are_unreadable_before_freezing(self):
        s = GridSplit(calibration=((1, 1, False),), validation=((2, 2, False),))
        with pytest.raises(SplitViolation, match="not frozen"):
            s.check_readable((2, 2, False), purpose="validation")

    def test_validation_cells_become_readable_after_freezing(self):
        s = GridSplit(calibration=((1, 1, False),), validation=((2, 2, False),)).freeze()
        s.check_readable((2, 2, False), purpose="validation")   # no raise

    def test_calibration_cells_cannot_be_offered_as_validation(self):
        s = GridSplit(calibration=((1, 1, False),), validation=((2, 2, False),)).freeze()
        with pytest.raises(SplitViolation, match="already informed the model"):
            s.check_readable((1, 1, False), purpose="validation")

    def test_validation_cells_cannot_be_used_for_calibration(self):
        s = GridSplit(calibration=((1, 1, False),), validation=((2, 2, False),))
        with pytest.raises(SplitViolation, match="circular"):
            s.check_readable((2, 2, False), purpose="calibration")

    def test_fitting_after_freezing_raises(self):
        s = GridSplit(calibration=((1, 1, False),), validation=((2, 2, False),))
        s.record_fit("alpha")
        s.freeze()
        with pytest.raises(SplitViolation, match="after the model was frozen"):
            s.record_fit("beta")

    def test_the_split_is_hashed(self):
        a = GridSplit(calibration=((1, 1, False),), validation=((2, 2, False),))
        b = GridSplit(calibration=((1, 1, False),), validation=((3, 3, False),))
        assert a.digest() != b.digest()
        assert a.digest() == GridSplit(
            calibration=((1, 1, False),), validation=((2, 2, False),)
        ).digest()

    def test_an_empty_validation_grid_is_refused(self):
        with pytest.raises(SplitViolation, match="nothing to validate"):
            GridSplit(calibration=((1, 1, False),), validation=())


class TestPreflight:
    def _fns(self, budget=BUDGET):
        m = BlockMemoryModel()
        return (
            lambda c: m.fits(shape(mb=c[0], seq=c[1], ckpt=c[2]), budget),
            lambda c: m.boundary_distance(shape(mb=c[0], seq=c[1], ckpt=c[2]), budget),
        )

    def test_reports_both_sides_of_the_prediction(self):
        fits, dist = self._fns()
        cells = [(8, 512, False), (256, 4096, False)]
        r = preflight(cells, fits, dist, label="t")
        assert r.n_predicted_feasible == 1
        assert r.n_predicted_infeasible == 1
        assert r.can_classify

    def test_an_all_feasible_grid_is_flagged_but_not_refused_wholesale(self):
        """A false win there would still be a finding."""
        fits, dist = self._fns()
        r = preflight([(8, 512, False), (8, 1024, False)], fits, dist, label="t")
        assert r.n_predicted_infeasible == 0
        assert not r.can_classify
        assert any("confusion matrix" in x for x in r.reasons)

    def test_an_empty_grid_is_not_informative(self):
        fits, dist = self._fns()
        assert not preflight([], fits, dist, label="t").informative

    def test_boundary_selection_prefers_cells_near_the_boundary(self):
        m = BlockMemoryModel()
        cands = [(mb, seq, False) for seq in (512, 1024, 2048, 4096)
                 for mb in (8, 16, 32, 64, 128, 256)]
        chosen, why = select_boundary_grid(
            cands,
            boundary_distance=lambda c: m.boundary_distance(
                shape(mb=c[0], seq=c[1], ckpt=c[2]), BUDGET),
            predict_fits=lambda c: m.fits(shape(mb=c[0], seq=c[1], ckpt=c[2]), BUDGET),
            n_boundary=4, n_clear_feasible=1, n_clear_infeasible=1,
        )
        assert len(chosen) >= 4
        near = [c for c in chosen if "boundary" in why[c]]
        far = [c for c in chosen if "anchor" in why[c]]
        assert near and far
        worst_near = max(
            m.boundary_distance(shape(mb=c[0], seq=c[1], ckpt=c[2]), BUDGET) for c in near
        )
        best_far = min(
            m.boundary_distance(shape(mb=c[0], seq=c[1], ckpt=c[2]), BUDGET) for c in far
        )
        assert worst_near <= best_far

    def test_selection_excludes_the_calibration_cells(self):
        m = BlockMemoryModel()
        cands = [(mb, 2048, False) for mb in (8, 16, 32, 64, 128)]
        banned = [(64, 2048, False)]
        chosen, _ = select_boundary_grid(
            cands,
            boundary_distance=lambda c: m.boundary_distance(
                shape(mb=c[0], seq=c[1], ckpt=c[2]), BUDGET),
            predict_fits=lambda c: m.fits(shape(mb=c[0], seq=c[1], ckpt=c[2]), BUDGET),
            n_boundary=3, exclude=banned,
        )
        assert (64, 2048, False) not in chosen


class TestModelSerialisation:
    def test_roundtrip_preserves_parameters_and_provenance(self):
        m = BlockMemoryModel()
        back = BlockMemoryModel.from_dict(m.to_dict())
        assert back.version == m.version
        assert back.predict_bytes(shape()) == pytest.approx(m.predict_bytes(shape()))

    def test_a_rejected_parameter_breaks_the_model_loudly(self):
        """Not silently: a model running on a rejected value must fail."""
        from losscolumn.core.parameters import RejectedParameterError

        m = BlockMemoryModel()
        m.params.add(
            Parameter("allocator_reserve_fraction", 0.12, Provenance.REJECTED,
                      source="fit", note="R^2 below gate")
        )
        with pytest.raises(RejectedParameterError):
            m.predict_bytes(shape())
