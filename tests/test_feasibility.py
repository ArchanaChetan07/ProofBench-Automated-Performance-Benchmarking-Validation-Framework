"""Feasibility must be a state, and must never quietly disappear.

The defect these tests exist to prevent was real and cost two calibration runs:
infeasible cells were encoded as a throughput of zero, the comparison paths
skipped zero-valued cells, and four falsifiable predictions vanished from the
map. The study then reported that it could establish nothing.

Every test here is a specific way that could happen again.
"""

from __future__ import annotations

import pytest

from losscolumn.core.feasibility import (
    Feasibility,
    FeasibilityCell,
    FeasibilityMatrix,
    FeasibilityObservation,
)
from losscolumn.core.parameters import (
    Parameter,
    ParameterSet,
    Provenance,
    RejectedParameterError,
)


def _obs(status: Feasibility, throughput: float | None = None) -> FeasibilityObservation:
    return FeasibilityObservation(status=status, throughput=throughput, method="test")


def _cell(key, predicted: Feasibility, measured: Feasibility) -> FeasibilityCell:
    return FeasibilityCell(key=key, predicted=_obs(predicted), measured=_obs(measured))


class TestOOMIsNotZero:
    def test_infeasible_cannot_carry_a_throughput(self):
        """The forbidden encoding is impossible to construct."""
        with pytest.raises(ValueError, match="did not run has no throughput"):
            FeasibilityObservation(status=Feasibility.INFEASIBLE, throughput=0.0)

    @pytest.mark.parametrize(
        "status", [Feasibility.INFEASIBLE, Feasibility.INVALID, Feasibility.NOT_MEASURED]
    )
    def test_no_did_not_run_state_carries_a_number(self, status):
        with pytest.raises(ValueError):
            FeasibilityObservation(status=status, throughput=123.0)

    def test_infeasible_throughput_is_none_not_zero(self):
        o = _obs(Feasibility.INFEASIBLE)
        assert o.throughput is None
        assert o.throughput != 0.0

    def test_a_feasible_run_measuring_zero_is_still_feasible(self):
        """Zero is a legitimate measurement, and must not be read as failure."""
        o = FeasibilityObservation(status=Feasibility.FEASIBLE, throughput=0.0)
        assert o.is_feasible
        assert o.usable_for_timing
        assert o.throughput == 0.0

    def test_a_model_prediction_carries_no_throughput(self):
        """A model did not run anything."""
        assert FeasibilityObservation.predicted(True).throughput is None
        assert FeasibilityObservation.predicted(False).throughput is None


class TestInfeasibleCellsSurvive:
    def test_infeasible_cells_are_counted(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.FEASIBLE),
            _cell("b", Feasibility.FEASIBLE, Feasibility.INFEASIBLE),
            _cell("c", Feasibility.INFEASIBLE, Feasibility.INFEASIBLE),
        ])
        assert m.n_comparable == 3
        assert m.n_measured_infeasible == 2
        assert m.n_predicted_infeasible == 1

    def test_an_all_infeasible_grid_does_not_vanish(self):
        """The case that broke: every cell infeasible must still be a grid."""
        m = FeasibilityMatrix(cells=[
            _cell(i, Feasibility.FEASIBLE, Feasibility.INFEASIBLE) for i in range(5)
        ])
        assert m.n_comparable == 5
        assert m.n_false_win == 5
        assert m.accuracy == 0.0
        assert not m.safe

    def test_cells_appear_in_the_serialised_form(self):
        m = FeasibilityMatrix(cells=[
            _cell("x", Feasibility.FEASIBLE, Feasibility.INFEASIBLE)
        ])
        d = m.to_dict()
        assert len(d["cells"]) == 1
        assert d["cells"][0]["outcome"] == "false_win"


class TestErrorDirection:
    def test_false_win_is_model_says_yes_hardware_says_no(self):
        c = _cell("k", Feasibility.FEASIBLE, Feasibility.INFEASIBLE)
        assert c.false_win and not c.false_loss
        assert c.outcome == "false_win"

    def test_false_loss_is_model_says_no_hardware_says_yes(self):
        c = _cell("k", Feasibility.INFEASIBLE, Feasibility.FEASIBLE)
        assert c.false_loss and not c.false_win
        assert c.outcome == "false_loss"

    def test_the_two_are_never_collapsed(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.INFEASIBLE),
            _cell("b", Feasibility.INFEASIBLE, Feasibility.FEASIBLE),
        ])
        assert m.n_false_win == 1 and m.n_false_loss == 1
        assert m.accuracy == 0.0
        # The weighted score must separate them even though accuracy cannot.
        assert m.dangerous_error_score == pytest.approx((10 * 1 + 1) / 2)

    def test_raw_counts_are_reported_beside_the_weighted_score(self):
        d = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.INFEASIBLE)
        ]).to_dict()
        assert d["counts"]["false_win"] == 1
        assert d["danger_weight"] == 10.0
        assert "policy" in d["danger_weight_policy"].lower() or d["danger_weight_policy"]

    def test_safe_requires_zero_false_wins(self):
        assert FeasibilityMatrix(cells=[
            _cell("a", Feasibility.INFEASIBLE, Feasibility.FEASIBLE)
        ]).safe
        assert not FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.INFEASIBLE)
        ]).safe


class TestIndefiniteStatesAreExcluded:
    @pytest.mark.parametrize(
        "status",
        [Feasibility.HOST_FALLBACK, Feasibility.INVALID, Feasibility.NOT_MEASURED],
    )
    def test_indefinite_measurements_do_not_enter_metrics(self, status):
        """Excluded, not rounded to the nearest verdict."""
        m = FeasibilityMatrix(cells=[
            _cell("good", Feasibility.FEASIBLE, Feasibility.FEASIBLE),
            _cell("odd", Feasibility.FEASIBLE, status),
        ])
        assert m.n_comparable == 1
        assert len(m.excluded) == 1
        assert m.accuracy == 1.0

    def test_host_fallback_is_not_feasible(self):
        """Completing is not the same as fitting."""
        assert not Feasibility.HOST_FALLBACK.is_feasible
        assert Feasibility.HOST_FALLBACK.ran        # it did complete
        assert not Feasibility.HOST_FALLBACK.is_definite

    def test_excluded_cells_are_reported_by_reason(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.HOST_FALLBACK),
            _cell("b", Feasibility.FEASIBLE, Feasibility.INVALID),
        ])
        by = m.excluded_by_status()
        assert by["measured:host_fallback"] == 1
        assert by["measured:invalid"] == 1


class TestEmptyVersusEmptyIsRejected:
    def test_iou_of_two_empty_sets_is_undefined_not_one(self):
        """Two empty sets agreeing is not agreement."""
        import math

        m = FeasibilityMatrix(cells=[
            _cell(i, Feasibility.FEASIBLE, Feasibility.FEASIBLE) for i in range(4)
        ])
        assert math.isnan(m.infeasible_iou)

    def test_no_comparable_cells_reports_not_computed(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.NOT_MEASURED)
        ])
        assert "NOT COMPUTED" in m.verdict()

    def test_non_empty_disagreement_produces_a_real_iou(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.INFEASIBLE, Feasibility.INFEASIBLE),
            _cell("b", Feasibility.INFEASIBLE, Feasibility.FEASIBLE),
            _cell("c", Feasibility.FEASIBLE, Feasibility.INFEASIBLE),
        ])
        # predicted {a,b}, measured {a,c} -> intersection {a}, union {a,b,c}
        assert m.infeasible_iou == pytest.approx(1 / 3)


class TestClassificationWithoutRegression:
    """Every metric here is computed without any continuous quantity."""

    def test_all_classification_metrics_available_with_no_numbers_at_all(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.FEASIBLE),
            _cell("b", Feasibility.FEASIBLE, Feasibility.INFEASIBLE),
            _cell("c", Feasibility.INFEASIBLE, Feasibility.INFEASIBLE),
            _cell("d", Feasibility.INFEASIBLE, Feasibility.FEASIBLE),
        ])
        assert all(o.throughput is None for c in m.cells for o in (c.predicted, c.measured))
        assert m.accuracy == pytest.approx(0.5)
        assert m.n_false_win == 1 and m.n_false_loss == 1
        assert m.infeasible_iou == pytest.approx(1 / 3)
        assert m.confusion()["predicted_feasible"]["measured_infeasible"] == 1
        assert m.false_win_area == pytest.approx(0.25)


class TestParameterProvenance:
    def test_rejected_parameter_cannot_be_used(self):
        p = Parameter("beta", 0.14, Provenance.FITTED, source="gloo").rejected(
            "R^2 = 0.715, below the 0.90 gate"
        )
        with pytest.raises(RejectedParameterError, match="must not be used"):
            p.get()

    def test_rejected_parameter_remains_inspectable(self):
        """The gloo all-gather principle: visible as a diagnostic, never used."""
        p = Parameter("beta", 0.14, Provenance.FITTED, source="gloo").rejected("bad fit")
        assert p.value == 0.14
        assert not p.usable
        assert "bad fit" in p.to_dict()["note"]

    def test_diagnostic_is_also_unusable(self):
        p = Parameter("x", 1.0, Provenance.DIAGNOSTIC, source="probe")
        with pytest.raises(RejectedParameterError):
            p.get()

    def test_a_parameter_without_a_source_is_refused(self):
        with pytest.raises(ValueError, match="no source"):
            Parameter("x", 1.0, Provenance.REGISTERED)

    def test_set_get_raises_for_rejected_and_get_or_falls_back(self):
        s = ParameterSet("m")
        s.add(Parameter("a", 5.0, Provenance.REGISTERED, source="protocol"))
        s.add(Parameter("b", 9.0, Provenance.REJECTED, source="fit", note="bad"))
        assert s.get("a") == 5.0
        with pytest.raises(RejectedParameterError):
            s.get("b")
        assert s.get_or("b", 1.0) == 1.0
        assert s.get_or("a", 1.0) == 5.0

    def test_provenance_survives_a_roundtrip(self):
        s = ParameterSet("m")
        s.add(Parameter("a", 5.0, Provenance.ACCEPTED, source="fit",
                        quality={"r2": 0.99}))
        s.add(Parameter("b", 9.0, Provenance.REJECTED, source="fit", note="bad"))
        back = ParameterSet.from_dict(s.to_dict())
        assert back.params["a"].provenance is Provenance.ACCEPTED
        assert back.params["b"].provenance is Provenance.REJECTED
        assert back.params["a"].quality["r2"] == 0.99
        with pytest.raises(RejectedParameterError):
            back.get("b")

    def test_evidence_backed_excludes_registered_values(self):
        s = ParameterSet("m")
        s.add(Parameter("reg", 1.0, Provenance.REGISTERED, source="protocol"))
        s.add(Parameter("meas", 2.0, Provenance.MEASURED, source="microbenchmark"))
        assert [p.name for p in s.evidence_backed] == ["meas"]


class TestBoundaryDisplacement:
    def _grid(self, pred_flip: int, meas_flip: int) -> FeasibilityMatrix:
        """One fiber along `mb`; each map turns infeasible at its own index."""
        levels = [8, 16, 32, 64, 128]
        cells = []
        for i, mb in enumerate(levels):
            cells.append(FeasibilityCell(
                key=(mb,),
                predicted=_obs(Feasibility.FEASIBLE if i < pred_flip
                               else Feasibility.INFEASIBLE),
                measured=_obs(Feasibility.FEASIBLE if i < meas_flip
                              else Feasibility.INFEASIBLE),
                coords={"micro_batch": mb},
            ))
        return FeasibilityMatrix(cells=cells)

    def test_agreeing_boundaries_displace_by_zero(self):
        b = self._grid(3, 3).boundary_displacement("micro_batch")
        assert b["mean_signed_shift"] == 0.0
        assert b["n_fibers_compared"] == 1

    def test_a_model_that_holds_on_too_long_shifts_positive(self):
        """Positive is the direction that produces false wins."""
        m = self._grid(pred_flip=4, meas_flip=3)
        assert m.boundary_displacement("micro_batch")["mean_signed_shift"] == 1.0
        assert m.n_false_win == 1

    def test_a_model_that_gives_up_early_shifts_negative(self):
        m = self._grid(pred_flip=2, meas_flip=3)
        assert m.boundary_displacement("micro_batch")["mean_signed_shift"] == -1.0
        assert m.n_false_loss == 1

    def test_a_fiber_with_a_boundary_on_only_one_side_is_counted_apart(self):
        m = self._grid(pred_flip=5, meas_flip=3)     # predicted never flips
        b = m.boundary_displacement("micro_batch")
        assert b["n_fibers_compared"] == 0
        assert b["n_fibers_measured_boundary_only"] == 1

    def test_displacement_is_reported_per_axis_not_pooled(self):
        m = self._grid(4, 3)
        rep = m.boundary_report(["micro_batch", "seq_len"])
        assert [r["axis"] for r in rep] == ["micro_batch", "seq_len"]


class TestMatrixRoundTrip:
    def test_a_published_matrix_rebuilds_from_its_artifact(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.INFEASIBLE),
            _cell("b", Feasibility.INFEASIBLE, Feasibility.INFEASIBLE),
            _cell("c", Feasibility.FEASIBLE, Feasibility.HOST_FALLBACK),
        ])
        back = FeasibilityMatrix.from_dict(m.to_dict())
        assert back.n_false_win == m.n_false_win == 1
        assert back.n_comparable == m.n_comparable == 2
        assert len(back.excluded) == 1

    def test_indefinite_states_survive_the_roundtrip(self):
        m = FeasibilityMatrix(cells=[
            _cell("a", Feasibility.FEASIBLE, Feasibility.HOST_FALLBACK)
        ])
        back = FeasibilityMatrix.from_dict(m.to_dict())
        assert back.cells[0].measured.status is Feasibility.HOST_FALLBACK
