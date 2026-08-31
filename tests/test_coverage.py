"""Coverage, acceptance and readiness are three things, and must stay three.

The state this guards against is one the project actually reached: one group of
six with an accepted model, and nothing in the code distinguishing that from a
calibrated subsystem. Acting on the confusion means renting eight A100s to
debug rather than to measure.

Every test here is a way the three could collapse back into one.
"""

from __future__ import annotations

import pytest

from losscolumn.core.coverage import (
    CommunicationCoverage,
    CoverageVerdict,
    ParameterVerdict,
    ReadinessVerdict,
    RegimeStatus,
)

COLLECTIVES = ("all_reduce", "all_gather")
WORLDS = (2, 3, 4)
REGIMES = ("tiny", "small", "medium", "large")


def _empty() -> CommunicationCoverage:
    return CommunicationCoverage.empty(COLLECTIVES, WORLDS, REGIMES)


def _cover_group(cov: CommunicationCoverage, key: str, *, err: float = 0.05,
                 noise: float = 0.05) -> None:
    g = cov.groups[key]
    g.parameter_verdict = ParameterVerdict.ACCEPTED
    g.model_family, g.estimator = "piecewise", "weighted"
    for r in g.regimes.values():
        r.status = RegimeStatus.COVERED
        r.n_points = 6
        r.n_validation_points = 3
        r.heldout_err = err
        r.noise_cv = noise


def _cover_all(cov: CommunicationCoverage) -> None:
    for key in cov.groups:
        _cover_group(cov, key)


class TestTheThreeConceptsAreDistinct:
    def test_an_accepted_model_is_not_coverage(self):
        """Parameter validity says nothing about which regimes are spoken for."""
        cov = _empty()
        g = cov.groups["all_gather/world3"]
        g.parameter_verdict = ParameterVerdict.ACCEPTED
        assert g.parameter_verdict.usable
        assert not g.covered
        assert g.verdict is CoverageVerdict.NOT_MEASURED

    def test_coverage_of_one_group_is_not_readiness(self):
        """G: one accepted group out of six must not make the subsystem ready."""
        cov = _empty()
        _cover_group(cov, "all_gather/world3")
        assert len(cov.accepted_groups) == 1
        assert len(cov.covered_groups) == 1
        verdict, blocking = cov.readiness()
        assert verdict is ReadinessVerdict.NOT_READY
        assert blocking
        assert any("required groups" in b for b in blocking)

    def test_readiness_needs_every_group(self):
        cov = _empty()
        for key in list(cov.groups)[:-1]:
            _cover_group(cov, key)
        verdict, blocking = cov.readiness()
        assert verdict is ReadinessVerdict.NOT_READY
        assert any("all_gather" in b or "all_reduce" in b for b in blocking)

    def test_readiness_is_reachable_when_everything_is_covered(self):
        """The gate must be satisfiable, or it is not a gate."""
        cov = _empty()
        _cover_all(cov)
        verdict, blocking = cov.readiness()
        assert verdict is ReadinessVerdict.READY
        assert blocking == []


class TestARegimeIsNotCoveredWhen:
    @pytest.mark.parametrize(
        "status",
        [RegimeStatus.NOT_MEASURED, RegimeStatus.NOT_VALIDATED,
         RegimeStatus.EXTRAPOLATED, RegimeStatus.MODEL_REJECTED,
         RegimeStatus.ERROR_TOO_HIGH, RegimeStatus.TOO_NOISY,
         RegimeStatus.INSUFFICIENT_POINTS],
    )
    def test_any_non_covered_status_blocks_the_group(self, status):
        cov = _empty()
        _cover_all(cov)
        cov.groups["all_reduce/world2"].regimes["medium"].status = status
        assert not cov.groups["all_reduce/world2"].covered
        assert cov.readiness()[0] is ReadinessVerdict.NOT_READY

    def test_a_regime_with_no_validation_point_blocks_readiness(self):
        """F: a regime nobody validated is not validated."""
        cov = _empty()
        _cover_all(cov)
        cov.groups["all_gather/world4"].regimes["tiny"].n_validation_points = 0
        verdict, blocking = cov.readiness()
        assert verdict is ReadinessVerdict.NOT_READY
        assert any("held-out validation" in b for b in blocking)

    def test_extrapolation_is_not_coverage(self):
        cov = _empty()
        _cover_all(cov)
        cov.groups["all_reduce/world4"].regimes["large"].status = (
            RegimeStatus.EXTRAPOLATED
        )
        assert cov.readiness()[0] is ReadinessVerdict.NOT_READY


class TestNoArithmeticTradesGroupsOffAgainstEachOther:
    def test_high_coverage_fraction_still_fails(self):
        """23 of 24 cells covered is still not ready."""
        cov = _empty()
        _cover_all(cov)
        cov.groups["all_reduce/world3"].regimes["small"].status = (
            RegimeStatus.ERROR_TOO_HIGH
        )
        assert cov.coverage_fraction > 0.95
        assert cov.readiness()[0] is ReadinessVerdict.NOT_READY

    def test_the_report_shows_both_numbers_separately(self):
        cov = _empty()
        _cover_group(cov, "all_gather/world3")
        d = cov.to_dict()
        assert d["parameter_validity"]["n_accepted"] == 1
        assert d["model_coverage"]["n_covered_groups"] == 1
        assert d["model_coverage"]["n_covered_cells"] == 4
        assert d["subsystem_readiness"]["ready"] is False
        # The three live under three different keys, so a reader cannot mistake
        # one for another by glancing at the wrong number.
        assert set(d) >= {"parameter_validity", "model_coverage",
                          "subsystem_readiness"}

    def test_markdown_names_the_blocking_gates(self):
        cov = _empty()
        _cover_group(cov, "all_gather/world3")
        md = cov.to_markdown()
        assert "NOT READY" in md
        assert "Blocking gates" in md
        assert "neither implies readiness" in md


class TestRejectedModelsCannotBecomeCoverage:
    def test_a_diagnostic_verdict_is_not_usable(self):
        """H: a rejected model must stay impossible to consume."""
        assert not ParameterVerdict.DIAGNOSTIC.usable
        assert not ParameterVerdict.REJECTED.usable
        assert ParameterVerdict.ACCEPTED.usable

    def test_a_group_with_a_rejected_model_is_not_covered(self):
        cov = _empty()
        _cover_all(cov)
        g = cov.groups["all_reduce/world2"]
        g.parameter_verdict = ParameterVerdict.DIAGNOSTIC
        for r in g.regimes.values():
            r.status = RegimeStatus.MODEL_REJECTED
        assert not g.covered
        verdict, blocking = cov.readiness()
        assert verdict is ReadinessVerdict.NOT_READY
        assert any("no accepted model" in b for b in blocking)

    def test_covered_regimes_under_a_rejected_model_still_block(self):
        """The pathological case: cells marked covered, model not accepted."""
        cov = _empty()
        _cover_all(cov)
        cov.groups["all_reduce/world2"].parameter_verdict = ParameterVerdict.REJECTED
        verdict, blocking = cov.readiness()
        assert verdict is ReadinessVerdict.NOT_READY
        assert any("no accepted model" in b for b in blocking)
