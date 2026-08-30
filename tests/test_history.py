"""Withdrawn artifacts must stay withdrawn, and stay unedited.

Preservation only means something if two properties hold. A preserved artifact
must be immutable -- otherwise it is a second draft, not a record of what was
published. And it must never be counted, graded or cited as a live claim --
otherwise withdrawing it accomplished nothing.

Both are checked here against the repository's own history, not only against
fixtures, because the repository's own history is what the guarantees are for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from losscolumn.history import (
    REASONS,
    Incident,
    Withdrawal,
    load,
    load_incidents,
    record,
    record_incident,
    summarise,
    verify,
)
from losscolumn.report.index import build_index
from losscolumn.report.index import summarise as index_summary

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
HAS_HISTORY = (ARTIFACTS / "history").exists()
needs_history = pytest.mark.skipif(not HAS_HISTORY, reason="no history recorded yet")


class TestWithdrawalRecord:
    def test_reason_must_be_recognised(self):
        """A withdrawal filed under a vague reason is not actionable later."""
        with pytest.raises(ValueError, match="not a recognised withdrawal reason"):
            Withdrawal(artifact_id="x", title="t", reason="it was bad")

    @pytest.mark.parametrize("reason", sorted(REASONS))
    def test_every_recognised_reason_is_accepted(self, reason):
        assert Withdrawal(artifact_id="x", title="t", reason=reason).reason == reason

    def test_roundtrip(self):
        w = Withdrawal(artifact_id="a", title="T", reason="vacuous-result",
                       what_was_wrong="w", lesson="l")
        assert Withdrawal.from_dict(w.to_dict()).to_dict() == w.to_dict()

    def test_markdown_states_it_is_not_a_claim(self):
        w = Withdrawal(artifact_id="a", title="T", reason="unfair-comparison")
        assert "Nothing here is a live claim" in w.to_markdown()


class TestImmutability:
    def test_editing_a_preserved_file_is_detected(self, tmp_path):
        """The check that makes preservation mean anything."""
        w = Withdrawal(artifact_id="a", title="T", reason="vacuous-result")
        d = record(tmp_path, "case", w, {"result.json": '{"answer": 1}'})
        assert verify(tmp_path) == []

        (d / "result.json").write_text('{"answer": 2}', encoding="utf-8")
        problems = verify(tmp_path)
        assert problems and "content changed since withdrawal" in problems[0]

    def test_deleting_a_preserved_file_is_detected(self, tmp_path):
        w = Withdrawal(artifact_id="a", title="T", reason="vacuous-result")
        d = record(tmp_path, "case", w, {"result.json": "{}"})
        (d / "result.json").unlink()
        assert any("missing" in p for p in verify(tmp_path))

    def test_manifest_is_rebuilt_from_the_directories(self, tmp_path):
        for i, reason in enumerate(("vacuous-result", "unfair-comparison")):
            record(tmp_path, f"case{i}",
                   Withdrawal(artifact_id=f"a{i}", title=f"T{i}", reason=reason),
                   {"x.json": "{}"})
        man = json.loads(
            (tmp_path / "history" / "withdrawals.json").read_text(encoding="utf-8")
        )
        assert man["n_withdrawn"] == 2
        assert {e["reason"] for e in man["withdrawals"]} == {
            "vacuous-result", "unfair-comparison"
        }


class TestIncidents:
    def test_an_incident_preserves_nothing(self, tmp_path):
        """No artifact was published, so none should be citable."""
        inc = Incident(incident_id="i1", title="T", occurred_at="2026-01-01T00:00:00Z")
        p = record_incident(tmp_path, "i1", inc)
        assert p.exists()
        assert not list((tmp_path / "history" / "incidents").glob("*.claim.json"))
        assert "none should be citable" in p.read_text(encoding="utf-8")

    def test_incidents_load_back(self, tmp_path):
        record_incident(tmp_path, "i1", Incident(
            incident_id="i1", title="T", occurred_at="2026-01-01T00:00:00Z"))
        assert [i.incident_id for i in load_incidents(tmp_path)] == ["i1"]


@needs_history
class TestThisRepositoryHistory:
    """The guarantees, checked against the real preserved artifacts."""

    def test_preserved_files_are_intact(self):
        assert verify(ARTIFACTS) == []

    def test_both_withdrawals_are_present(self):
        reasons = {w.reason for _, w in load(ARTIFACTS)}
        assert "unfair-comparison" in reasons, "the unfair-baseline run is not preserved"
        assert "vacuous-result" in reasons, "the empty-set calibration is not preserved"

    def test_the_unfair_baseline_still_reports_zero_losses(self):
        """The preserved artifact keeps the wrong answer, on purpose.

        Correcting it in place would destroy the only direct evidence that the
        error occurred and was caught.
        """
        d = next(d for d, w in load(ARTIFACTS) if w.reason == "unfair-comparison")
        claim = json.loads(
            (d / "thrust3b-triton-attention.claim.json").read_text(encoding="utf-8")
        )
        assert claim["loss_column"]["counts"]["loss"] == 0

    def test_the_vacuous_calibration_still_reports_its_vacuous_verdict(self):
        d = next(d for d, w in load(ARTIFACTS) if w.reason == "vacuous-result")
        rep = json.loads(
            (d / "calibration-thrust1-compute.json").read_text(encoding="utf-8")
        )
        assert "USABLE FOR RANKING" in rep["calibration"]["verdict"]

    def test_every_withdrawal_says_what_superseded_it(self):
        for _, w in load(ARTIFACTS):
            assert w.superseded_by, f"{w.artifact_id} has no successor recorded"
            assert w.what_was_wrong.strip()
            assert w.lesson.strip()

    def test_the_contended_gpu_incident_is_recorded(self):
        ids = {i.incident_id for i in load_incidents(ARTIFACTS)}
        assert "2026-08-29-contended-gpu" in ids

    def test_summary_reports_history_intact(self):
        s = summarise(ARTIFACTS)
        assert s["verified"] is True
        assert s["n_withdrawn"] >= 2


@needs_history
class TestWithdrawnAreNotLiveClaims:
    """Withdrawing an artifact has to actually remove it from the record."""

    def test_withdrawn_claims_are_not_counted(self):
        """A withdrawn claim shares its id with the artifact that replaced it.

        So identity is the wrong thing to test on, and testing it that way was
        this file's own first bug. What must hold is positional: every claim the
        index counts lives at the top level, and none is drawn from history/.
        """
        preserved = list((ARTIFACTS / "history").rglob("*.claim.json"))
        assert preserved, "no preserved claim to test against"

        live = list(ARTIFACTS.glob("*.claim.json"))
        assert index_summary(ARTIFACTS)["n_claims"] == len(live)
        assert len(live) < len(live) + len(preserved)

    def test_the_index_separates_them(self):
        md = build_index(ARTIFACTS)
        assert "WITHDRAWN AND DISCARDED" in md
        assert "nothing here carries a conformance grade" in md
