"""LC-1.0 is frozen. These tests are the lock.

A standard that can drift is not a standard. Rule identifiers and severities
carry meaning into other people's artifacts: a claim graded `conforming` last
year must mean the same thing this year, and a rule that silently changes from
warning to fatal retroactively re-grades every artifact that cited it.

So the freeze is enforced mechanically. Changing a rule id or a severity fails
``test_registry_digest_is_frozen``, and the only way past that test is to
deliberately update the digest -- which is a visible, reviewable act rather
than an edit that slips through.
"""

from __future__ import annotations

import pytest

from losscolumn.spec import registry, schema
from losscolumn.validate import adversarial as adv
from losscolumn.validate import validate_claim
from losscolumn.version import STANDARD_VERSION


class TestFrozenRegistry:
    def test_standard_version(self):
        assert STANDARD_VERSION == "LC-1.0"

    def test_registry_digest_is_frozen(self):
        """If this fails, a rule id or severity changed.

        That is a breaking change to the standard. Bump the version, update
        FROZEN_DIGEST deliberately, and say so in the changelog -- do not
        simply re-pin the hash to make the test green.
        """
        assert registry.compute_digest() == registry.FROZEN_DIGEST, (
            "the frozen rule table changed; this is a standard version bump, "
            "not a code edit"
        )

    def test_rule_count(self):
        assert registry.N_RULES == 47
        assert registry.N_FATAL == 28

    @pytest.mark.parametrize(
        "rule_id,severity",
        [
            # Spot-checks of the severities that carry the most weight, written
            # out literally so a change is visible in the diff of this file too.
            ("LC-1.1", registry.FATAL),
            ("LC-1.4", registry.FATAL),
            ("LC-1.5", registry.WARNING),
            ("LC-1.8", registry.FATAL),
            ("LC-1.9", registry.FATAL),
            ("LC-2.1", registry.FATAL),
            ("LC-2.7", registry.FATAL),
            ("LC-2.8", registry.FATAL),
            ("LC-3.3", registry.FATAL),
            ("LC-3.7", registry.FATAL),
            ("LC-4.3", registry.FATAL),
            ("LC-4.5", registry.FATAL),
            ("LC-5.4", registry.FATAL),
            ("LC-Q1", registry.WARNING),
            ("LC-Q6", registry.FATAL),
        ],
    )
    def test_severity_is_pinned(self, rule_id, severity):
        assert registry.get(rule_id).severity == severity

    def test_every_rule_has_a_requirement_and_a_summary(self):
        for r in registry.RULES.values():
            assert r.requirement and r.summary, r.id
            assert r.severity in (registry.FATAL, registry.WARNING, registry.INFO)

    def test_ids_are_unique_and_well_formed(self):
        import re

        for rid in registry.RULES:
            assert re.fullmatch(r"LC-(\d+\.\d+|Q\d+)", rid), rid
        assert len(set(registry.RULES)) == registry.N_RULES

    def test_unregistered_rule_cannot_be_emitted(self):
        """The validator may not invent a grade the standard does not define."""
        with pytest.raises(KeyError, match="not a registered rule"):
            registry.get("LC-9.9")


class TestFrozenSchemas:
    def test_schema_digest_is_frozen(self):
        assert schema.schema_digest() == schema.FROZEN_SCHEMA_DIGEST, (
            "a frozen document schema changed; this is a standard version bump"
        )

    def test_the_four_document_shapes_are_present(self):
        assert set(schema.SCHEMAS) == {
            "envelope", "loss_column", "claim", "preregistration"
        }

    def test_conforming_claim_validates(self):
        assert schema.validate_claim_document(adv.conforming_claim()) == []

    def test_schema_rejects_a_missing_required_field(self):
        c = adv.conforming_claim()
        del c["loss_column"]
        errs = schema.validate_claim_document(c)
        assert any("loss_column" in e for e in errs)

    def test_schema_rejects_a_bad_evidence_class(self):
        c = adv.conforming_claim()
        c["evidence_class"] = "vibes"
        assert any("evidence_class" in e for e in schema.validate_claim_document(c))

    def test_schema_requires_the_power_check(self):
        """LC-1.8 is unenforceable if design_check may be absent."""
        c = adv.conforming_claim()
        del c["loss_column"]["design_check"]
        assert any("design_check" in e for e in schema.validate_claim_document(c))

    def test_schema_rejects_a_negative_replicate_count(self):
        c = adv.conforming_claim()
        c["envelope"]["replicates"] = 0
        from losscolumn.spec.schema import ENVELOPE_SCHEMA, validate_document

        assert validate_document(c["envelope"], ENVELOPE_SCHEMA)


class TestBaselinesConform:
    @pytest.mark.parametrize(
        "builder",
        [adv.conforming_claim, adv.conforming_tuned_claim, adv.conforming_simulated_claim],
        ids=["measured", "tuned", "simulated"],
    )
    def test_baseline_is_clean(self, builder):
        rep = validate_claim(builder())
        assert rep.grade == "conforming", [f.message for f in rep.failures + rep.warnings]


class TestAdversarialCorpus:
    """Every defect must be caught by the rule that claims to catch it."""

    @pytest.mark.parametrize("defect", adv.CORPUS, ids=[d.name for d in adv.CORPUS])
    def test_defect_is_caught(self, defect):
        rep = validate_claim(defect.build())
        failed = {f.rule for f in rep.findings
                  if not f.passed and not f.detail.get("not_applicable")}
        missing = set(defect.expect_fail) - failed
        assert not missing, (
            f"{defect.name}: expected {sorted(defect.expect_fail)} to fail, "
            f"but {sorted(missing)} passed. Actually failing: {sorted(failed)}"
        )

    @pytest.mark.parametrize(
        "defect",
        [d for d in adv.CORPUS if d.severity == registry.FATAL],
        ids=[d.name for d in adv.CORPUS if d.severity == registry.FATAL],
    )
    def test_fatal_defect_is_non_conforming(self, defect):
        assert validate_claim(defect.build()).grade == "non-conforming", defect.name

    def test_the_nine_named_defects_are_all_fatal(self):
        """The freeze request named nine defects and required all to be fatal."""
        named = {
            "insufficient-replicates", "bh-impossible-power", "inverted-tost-bounds",
            "mde-differs-from-seal", "simulated-unlabeled",
            "tuned-baseline-called-default", "unsupported-headline", "dirty-tree",
            "unsupported-loss-region",
        }
        present = {d.name for d in adv.CORPUS}
        assert named <= present, f"missing from the corpus: {sorted(named - present)}"
        for d in adv.CORPUS:
            if d.name in named:
                assert validate_claim(d.build()).grade == "non-conforming", d.name

    def test_corpus_runner_agrees(self):
        results = adv.run_corpus()
        uncaught = [r.defect for r in results if not r.caught]
        assert not uncaught, uncaught


class TestCoverage:
    """The target: every rule has one passing test and at least one failing test."""

    def test_every_rule_has_a_passing_case(self):
        cov = adv.coverage()
        missing = [k for k, v in cov.items() if not v["passing_case"]]
        assert not missing, f"no passing case for {missing}"

    def test_every_rule_has_a_failing_case(self):
        cov = adv.coverage()
        missing = [
            k for k, v in cov.items()
            if not v["failing_case"] and k not in adv.UNFAILABLE
        ]
        assert not missing, f"no failing case for {missing}"

    def test_unfailable_rules_are_justified(self):
        """A rule that cannot fail must say why, not merely be absent."""
        for rid, reason in adv.UNFAILABLE.items():
            assert rid in registry.RULES
            assert reason.strip()

    def test_coverage_matrix_renders(self):
        md = adv.coverage_markdown()
        assert "Validator coverage matrix" in md
        for rid in registry.RULES:
            assert f"`{rid}`" in md
