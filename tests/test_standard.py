"""The standard itself: does the validator catch what it claims to catch?

Every rule is tested by handing the validator a claim that violates it and
asserting the rule fires. A conformance checker that has never been shown a
non-conforming document is a rubber stamp.
"""

from __future__ import annotations

import pytest

from losscolumn.core.prereg import PreRegistration, seal, verify
from losscolumn.validate import validate_claim
from losscolumn.version import STANDARD_VERSION


def minimal_claim() -> dict:
    """A document that passes every rule; each test breaks exactly one thing."""
    return {
        "kind": "claim",
        "standard_version": STANDARD_VERSION,
        "id": "test-claim",
        "title": "A test claim",
        "thrust": "III",
        "method": "mine",
        "baseline": "theirs",
        "evidence_class": "measured",
        "evidence_note": "",
        "loss_column_placement": "main",
        "headline": {
            "statement": "Faster in some places and slower in others.",
            "best_pct": 12.0,
            "worst_pct": -30.0,
            "median_pct": 2.0,
            "conditions_best": {"seq_len": 2048},
            "conditions_worst": {"seq_len": 128},
        },
        "loss_column": {
            "n_cells": 24,
            "counts": {"loss": 4, "win": 6, "tie": 12, "inconclusive": 2, "missing": 0},
            "inconclusive_fraction": 0.08,
            "mde_pct": 5.0,
            "q_level": 0.05,
            "design_check": {"can_reject": True, "min_replicates": 10},
            "regions": [
                {
                    "label": "seq_len=[128]",
                    "worst_regression_pct": 30.0,
                    "median_regression_pct": 22.0,
                    "attribution": "launch bound at short sequences",
                    "missing_reasons": [],
                }
            ],
            "notes": [],
        },
        "envelope": {
            "replicates": 11,
            "interleaved": True,
            "coverage": {"mine": 1.0, "theirs": 1.0},
            "factors": [
                {"name": "seq_len", "levels": [128, 512, 2048]},
                {"name": "head_dim", "levels": [64, 128]},
                {"name": "batch", "levels": [1, 4]},
                {"name": "dtype", "levels": ["fp16", "bf16"]},
            ],
        },
        "reproduction": {
            "command": "losscolumn run thrust3",
            "repo": "https://example.invalid/repo",
            "commit": "abc123def456",
            "dirty": False,
            "container_image": "ghcr.io/example/img:1.0",
            "hardware": "1x H100",
            "estimated_runtime_min": 30,
        },
        "prereg": {
            "sealed_at": "2026-01-01T00:00:00Z",
            "seal_hash": "sha256:deadbeef",
            "normative_hash": "sha256:deadbeef",
            "anchor": "git:v1",
            "deviations": [],
        },
        "prereg_verification": {"ok": True, "findings": []},
        "limitations": ["forward pass only"],
        "supporting": {},
    }


def rules(claim: dict) -> dict[str, bool]:
    return {f.rule: f.passed for f in validate_claim(claim).findings}


class TestBaseline:
    def test_the_minimal_claim_conforms(self):
        rep = validate_claim(minimal_claim())
        assert rep.ok, [f.message for f in rep.failures]
        assert rep.grade == "conforming"


class TestLC1LossColumn:
    def test_missing_loss_column_is_fatal(self):
        c = minimal_claim()
        del c["loss_column"]
        assert not validate_claim(c).ok
        assert not rules(c)["LC-1.1"]

    def test_losses_without_regions(self):
        c = minimal_claim()
        c["loss_column"]["regions"] = []
        assert not rules(c)["LC-1.2"]

    def test_self_declared_incomplete_column(self):
        c = minimal_claim()
        c["loss_column"]["notes"] = ["the column is INCOMPLETE and must not be published"]
        assert not rules(c)["LC-1.3"]

    def test_loss_column_in_the_appendix(self):
        c = minimal_claim()
        c["loss_column_placement"] = "appendix"
        assert not rules(c)["LC-1.4"]
        assert not validate_claim(c).ok

    def test_unattributed_region_warns(self):
        c = minimal_claim()
        c["loss_column"]["regions"][0]["attribution"] = ""
        assert not rules(c)["LC-1.5"]
        assert validate_claim(c).ok, "unattributed is a warning, not a failure"

    def test_undetectable_design_is_fatal(self):
        """The most important rule: an empty column from a powerless sweep."""
        c = minimal_claim()
        c["loss_column"]["design_check"] = {
            "can_reject": False,
            "message": "design cannot reject: with r=5 the smallest attainable p is 0.031",
        }
        assert not rules(c)["LC-1.8"]
        assert not validate_claim(c).ok


class TestLC2Parity:
    def test_tuned_comparison_without_a_certificate(self):
        c = minimal_claim()
        c["thrust"] = "II"
        assert not rules(c)["LC-2.1"]

    def test_untuned_baseline_is_the_target_failure(self):
        c = minimal_claim()
        c["thrust"] = "II"
        c["parity"] = {
            "ok": True,
            "ledgers": [
                {"system": "mine", "n_trials": 40, "operator": "x", "still_improving": False},
                {"system": "other", "n_trials": 40, "operator": "x", "still_improving": False},
            ],
            "violations": [],
        }
        assert not rules(c)["LC-2.7"], "the baseline has no ledger"

    def test_derived_baseline_inherits_ledgers(self):
        c = minimal_claim()
        c["thrust"] = "II"
        c["baseline"] = "best_alternative"
        c["supporting"] = {"baseline_derived_from": ["other", "third"]}
        c["parity"] = {
            "ok": True,
            "ledgers": [
                {"system": "mine", "n_trials": 40, "operator": "x", "still_improving": False},
                {"system": "other", "n_trials": 40, "operator": "x", "still_improving": False},
                {"system": "third", "n_trials": 40, "operator": "x", "still_improving": False},
            ],
            "violations": [],
        }
        assert rules(c)["LC-2.7"]

    def test_derived_baseline_with_an_untuned_constituent(self):
        c = minimal_claim()
        c["thrust"] = "II"
        c["baseline"] = "best_alternative"
        c["supporting"] = {"baseline_derived_from": ["other", "never_tuned"]}
        c["parity"] = {
            "ok": True,
            "ledgers": [
                {"system": "mine", "n_trials": 40, "operator": "x", "still_improving": False},
                {"system": "other", "n_trials": 40, "operator": "x", "still_improving": False},
            ],
            "violations": [],
        }
        assert not rules(c)["LC-2.7"]

    def test_unequal_trial_counts_warn(self):
        c = minimal_claim()
        c["thrust"] = "II"
        c["baseline"] = "other"
        c["parity"] = {
            "ok": True,
            "ledgers": [
                {"system": "mine", "n_trials": 80, "operator": "x", "still_improving": False},
                {"system": "other", "n_trials": 40, "operator": "x", "still_improving": False},
            ],
            "violations": [],
        }
        assert not rules(c)["LC-2.4"]


class TestLC3Envelope:
    def test_single_factor_is_fatal(self):
        c = minimal_claim()
        c["envelope"]["factors"] = [{"name": "seq_len", "levels": [1, 2, 3]}]
        c["loss_column"]["n_cells"] = 3
        assert not rules(c)["LC-3.1"]

    def test_too_few_cells(self):
        c = minimal_claim()
        c["loss_column"]["n_cells"] = 4
        assert not rules(c)["LC-3.2"]

    def test_headline_without_a_range(self):
        c = minimal_claim()
        del c["headline"]["worst_pct"]
        assert not rules(c)["LC-3.3"]

    def test_degenerate_range_warns(self):
        c = minimal_claim()
        c["headline"]["best_pct"] = c["headline"]["worst_pct"] = 5.0
        assert not rules(c)["LC-3.4"]

    @pytest.mark.parametrize("phrase", ["up to 2x faster", "as much as 40% better"])
    def test_unbounded_superlatives_warn(self, phrase):
        c = minimal_claim()
        c["headline"]["statement"] = phrase
        assert not rules(c)["LC-3.6"]


class TestLC4PreRegistration:
    def test_absent_protocol_is_fatal(self):
        c = minimal_claim()
        c["prereg"] = None
        assert not rules(c)["LC-4.1"]

    def test_unsealed_protocol(self):
        c = minimal_claim()
        c["prereg"]["sealed_at"] = None
        assert not rules(c)["LC-4.2"]

    def test_protocol_edited_after_sealing(self):
        c = minimal_claim()
        c["prereg"]["normative_hash"] = "sha256:something_else"
        assert not rules(c)["LC-4.3"]
        assert not validate_claim(c).ok

    def test_fatal_verification_finding_propagates(self):
        c = minimal_claim()
        c["prereg_verification"] = {
            "ok": False,
            "findings": [{"code": "PRE-004", "severity": "fatal",
                          "message": "data predates the seal"}],
        }
        assert not rules(c)["LC-4.5"]


class TestLC5Reproduction:
    def test_no_command(self):
        c = minimal_claim()
        c["reproduction"]["command"] = ""
        assert not rules(c)["LC-5.1"]

    def test_shell_pipeline_warns(self):
        c = minimal_claim()
        c["reproduction"]["command"] = "python a.py && python b.py"
        assert not rules(c)["LC-5.2"]

    def test_unpinned_commit(self):
        c = minimal_claim()
        c["reproduction"]["commit"] = None
        assert not rules(c)["LC-5.3"]

    def test_dirty_tree_is_fatal(self):
        c = minimal_claim()
        c["reproduction"]["dirty"] = True
        assert not rules(c)["LC-5.4"]

    def test_undocumented_hardware(self):
        c = minimal_claim()
        c["reproduction"]["hardware"] = ""
        assert not rules(c)["LC-5.6"]


class TestEvidenceClass:
    def test_bad_value_is_fatal(self):
        c = minimal_claim()
        c["evidence_class"] = "probably fine"
        assert not rules(c)["LC-0.2"]

    def test_simulated_must_explain_itself(self):
        c = minimal_claim()
        c["evidence_class"] = "simulated"
        c["evidence_note"] = ""
        assert not rules(c)["LC-0.3"]

    def test_simulated_with_a_note_is_acceptable(self):
        c = minimal_claim()
        c["evidence_class"] = "simulated"
        c["evidence_note"] = "produced by the calibratable model, not measured"
        assert validate_claim(c).ok


class TestPreRegistrationMechanics:
    def _proto(self) -> PreRegistration:
        return PreRegistration(
            title="t",
            hypotheses={"H0": "nothing"},
            metrics=[{"name": "x", "unit": "u", "higher_is_better": True}],
            factors=[{"name": "a", "levels": [1, 2]}],
            systems=["m", "b"],
            mde=0.05,
        )

    def test_sealing_is_stable_and_detects_edits(self):
        p = seal(self._proto())
        assert verify(p).ok
        p.mde = 0.20                       # a normative field
        assert not verify(p).ok

    def test_editing_prose_does_not_break_the_seal(self):
        p = seal(self._proto())
        p.rationale = "a much longer explanation added later"
        assert verify(p).ok, "non-normative fields are outside the hash by design"

    def test_resealing_is_refused(self):
        p = seal(self._proto())
        with pytest.raises(RuntimeError, match="already sealed"):
            seal(p)

    def test_data_predating_the_seal_is_fatal(self):
        p = seal(self._proto())
        res = verify(p, collected_at="2020-01-01T00:00:00Z")
        assert not res.ok
        assert any(f["code"] == "PRE-004" for f in res.findings)

    def test_undeclared_parameter_change_is_fatal(self):
        p = seal(self._proto())
        res = verify(p, analysis={"mde": 0.30})
        assert not res.ok
        assert any("UNDECLARED" in f["message"] for f in res.findings)

    def test_declared_deviation_downgrades_to_a_warning(self):
        p = seal(self._proto())
        p.declare_deviation("mde", 0.30, "pilot data showed the registered MDE was optimistic")
        res = verify(p, analysis={"mde": 0.30})
        assert res.ok
        assert any("declared deviation" in f["message"] for f in res.findings)

    def test_dropping_registered_levels_is_fatal(self):
        p = seal(self._proto())
        res = verify(p, analysis={"factors": [{"name": "a", "levels": [1]}]})
        assert not res.ok
        assert any(f["code"] == "PRE-009" for f in res.findings)

    def test_unsealed_protocol_cannot_be_verified(self):
        assert not verify(self._proto()).ok
