"""Withdrawn artifacts, preserved rather than deleted.

Three results produced during this project were wrong, and all three were wrong
in ways the project exists to talk about. Deleting them would remove the only
direct evidence that the discipline does what it claims -- and would leave the
repository looking as though nothing had ever gone wrong, which is exactly the
impression a loss column is supposed to prevent.

So they are preserved, unedited, beside a record of why they were withdrawn.
Two rules make that meaningful:

**A withdrawn artifact is never edited.** Its content is what it was when it
was published, including the parts that were wrong. Correcting it in place
would destroy the thing that makes it evidence. :func:`verify` re-hashes every
preserved file against the digest recorded at withdrawal, so a silent edit is
detectable.

**A withdrawal is a separate document.** The reason, the superseding artifact,
and what the error taught sit in a ``withdrawal.json`` next to the artifact,
never inside it. This is how retraction works in publishing, and for the same
reason: the record of the error and the error itself are different documents
with different authors and different dates.

Nothing here is a claim. The index renders withdrawn artifacts in their own
section, and they carry no conformance grade, because grading a withdrawn
artifact would invite it to be cited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from losscolumn.core.provenance import file_hash, utcnow

HISTORY_DIRNAME = "history"
MANIFEST = "withdrawals.json"

# Why an artifact left the record. Kept as a closed set so a withdrawal cannot
# be filed under a vague reason nobody can act on.
REASONS = {
    "unfair-comparison": "the comparison was not apples to apples",
    "vacuous-result": "the result could not have come out any other way",
    "invalid-measurement": "the measurement itself was not valid",
    "superseded-protocol": "the registered protocol was replaced",
    "contended-measurement": "the measurement shared hardware with another job",
}


@dataclass
class Withdrawal:
    """The record of why an artifact was taken out of the record."""

    artifact_id: str
    title: str
    reason: str
    withdrawn_at: str = field(default_factory=utcnow)
    original_commit: str = ""
    preserved: dict[str, str] = field(default_factory=dict)   # filename -> digest
    superseded_by: str | None = None
    what_was_wrong: str = ""
    how_it_was_found: str = ""
    what_changed: str = ""
    lesson: str = ""

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(
                f"{self.reason!r} is not a recognised withdrawal reason; expected one "
                f"of {sorted(REASONS)}. A withdrawal filed under a vague reason is not "
                "actionable by anyone reading it later."
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "withdrawal"
        d["reason_description"] = REASONS[self.reason]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Withdrawal:
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def to_markdown(self) -> str:
        lines = [
            f"# Withdrawn: {self.title}",
            "",
            f"**Reason** &mdash; {self.reason} ({REASONS[self.reason]})  ",
            f"**Withdrawn** {self.withdrawn_at}  ",
            f"**Originally published at** `{self.original_commit or 'unrecorded'}`  ",
        ]
        if self.superseded_by:
            lines.append(f"**Superseded by** `{self.superseded_by}`")
        lines += [
            "",
            "> The files in this directory are preserved exactly as published, "
            "including the parts that were wrong. They are not corrected in place, "
            "because correcting them would destroy the evidence that the error "
            "happened and was caught. Nothing here is a live claim.",
            "",
            "## What was wrong", "", self.what_was_wrong, "",
            "## How it was found", "", self.how_it_was_found, "",
            "## What changed", "", self.what_changed, "",
            "## What it taught", "", self.lesson, "",
            "## Preserved files", "",
            "| File | Digest at withdrawal |", "|---|---|",
        ]
        for name, dig in sorted(self.preserved.items()):
            lines.append(f"| `{name}` | `{dig[:23]}...` |")
        return "\n".join(lines)


def history_dir(artifacts: str | Path) -> Path:
    return Path(artifacts) / HISTORY_DIRNAME


def record(
    artifacts: str | Path,
    slug: str,
    withdrawal: Withdrawal,
    files: dict[str, bytes | str],
) -> Path:
    """Preserve an artifact's files and file its withdrawal beside them."""
    d = history_dir(artifacts) / slug
    d.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, content in files.items():
        p = d / name
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
        digests[name] = file_hash(p)
    withdrawal.preserved = digests
    (d / "withdrawal.json").write_text(
        json.dumps(withdrawal.to_dict(), indent=2), encoding="utf-8"
    )
    (d / "NOTE.md").write_text(withdrawal.to_markdown(), encoding="utf-8")
    _rebuild_manifest(artifacts)
    return d


def load(artifacts: str | Path) -> list[tuple[Path, Withdrawal]]:
    out: list[tuple[Path, Withdrawal]] = []
    root = history_dir(artifacts)
    if not root.exists():
        return out
    for wf in sorted(root.glob("*/withdrawal.json")):
        try:
            out.append((wf.parent, Withdrawal.from_dict(
                json.loads(wf.read_text(encoding="utf-8"))
            )))
        except Exception:
            continue
    return out


def verify(artifacts: str | Path) -> list[str]:
    """Re-hash every preserved file, withdrawn and milestone alike.

    This is the check that makes preservation mean something. A preserved
    artifact anyone can quietly tidy up is not a record; it is a second draft.
    """
    problems: list[str] = []
    for d, w in load(artifacts):
        for name, expected in w.preserved.items():
            p = d / name
            if not p.exists():
                problems.append(f"{d.name}/{name}: preserved file is missing")
                continue
            actual = file_hash(p)
            if actual != expected:
                problems.append(
                    f"{d.name}/{name}: content changed since withdrawal "
                    f"(recorded {expected[:19]}..., found {actual[:19]}...). A "
                    "withdrawn artifact must not be edited."
                )
    problems.extend(verify_milestones(artifacts))
    return problems


def _rebuild_manifest(artifacts: str | Path) -> Path:
    root = history_dir(artifacts)
    root.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "slug": d.name,
            "artifact_id": w.artifact_id,
            "title": w.title,
            "reason": w.reason,
            "withdrawn_at": w.withdrawn_at,
            "superseded_by": w.superseded_by,
            "original_commit": w.original_commit,
            "n_files": len(w.preserved),
        }
        for d, w in load(artifacts)
    ]
    p = root / MANIFEST
    p.write_text(
        json.dumps(
            {"kind": "withdrawal-manifest", "generated_at": utcnow(),
             "n_withdrawn": len(entries), "withdrawals": entries},
            indent=2,
        ),
        encoding="utf-8",
    )
    return p


def summarise(artifacts: str | Path) -> dict[str, Any]:
    items = load(artifacts)
    return {
        "n_withdrawn": len(items),
        "n_incidents": len(load_incidents(artifacts)),
        "n_milestones": len(load_milestones(artifacts)),
        "by_reason": {
            r: sum(1 for _, w in items if w.reason == r)
            for r in sorted({w.reason for _, w in items})
        },
        "verified": not verify(artifacts),
    }


# --------------------------------------------------------------------------
# incidents
# --------------------------------------------------------------------------


@dataclass
class Incident:
    """A run that was discarded before it produced anything worth withdrawing.

    Distinct from a withdrawal, and the distinction is not bookkeeping. A
    withdrawal preserves a published artifact that turned out to be wrong; an
    incident records a run whose *conditions* were invalid, where the right
    response was to discard the numbers and re-measure. There is nothing to
    preserve, because nothing should be citable -- but the fact that it
    happened, and what changed as a result, belongs in the reproducibility
    record. Otherwise the repository reads as though every measurement taken
    was a measurement kept.
    """

    incident_id: str
    title: str
    occurred_at: str
    what_happened: str = ""
    why_the_data_was_unusable: str = ""
    detection: str = ""
    disposition: str = ""
    safeguard: str = ""
    artifacts_published: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "incident"
        return d

    def to_markdown(self) -> str:
        return "\n".join([
            f"# Incident: {self.title}",
            "",
            f"**Occurred** {self.occurred_at}  ",
            f"**Artifacts published** {'yes' if self.artifacts_published else 'none'}",
            "",
            "> No artifact is preserved here, because none should be citable. What is "
            "recorded is that the run happened, why its numbers were discarded, and "
            "what changed so it cannot happen silently again.",
            "",
            "## What happened", "", self.what_happened, "",
            "## Why the data was unusable", "", self.why_the_data_was_unusable, "",
            "## How it was detected", "", self.detection, "",
            "## Disposition", "", self.disposition, "",
            "## Safeguard added", "", self.safeguard, "",
        ])


def record_incident(artifacts: str | Path, slug: str, incident: Incident) -> Path:
    d = history_dir(artifacts) / "incidents"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.json").write_text(
        json.dumps(incident.to_dict(), indent=2), encoding="utf-8"
    )
    (d / f"{slug}.md").write_text(incident.to_markdown(), encoding="utf-8")
    return d / f"{slug}.md"


def load_incidents(artifacts: str | Path) -> list[Incident]:
    d = history_dir(artifacts) / "incidents"
    out: list[Incident] = []
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            allowed = set(Incident.__dataclass_fields__)
            out.append(Incident(**{k: v for k, v in raw.items() if k in allowed}))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# milestones
# --------------------------------------------------------------------------


@dataclass
class Milestone:
    """A result that stands, preserved so it cannot drift.

    The mirror of a withdrawal. A withdrawal preserves something that turned
    out to be wrong; a milestone preserves something that turned out to be
    right, for the same reason: a result nobody can quietly edit is the only
    kind that can still be cited a year later.

    It also carries what the result does NOT establish. A validation on ten
    cells of one card is a real result and a narrow one, and the narrowness has
    to travel with it -- otherwise "zero false wins" detaches from the grid it
    was measured on and becomes a claim about the model in general.
    """

    milestone_id: str
    title: str
    tag: str = ""
    commit: str = ""
    recorded_at: str = field(default_factory=utcnow)
    headline: dict[str, Any] = field(default_factory=dict)
    preserved: dict[str, str] = field(default_factory=dict)
    establishes: list[str] = field(default_factory=list)
    does_not_establish: list[str] = field(default_factory=list)
    frozen_parameters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "milestone"
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# Milestone: {self.title}",
            "",
            f"**Tag** `{self.tag}` &middot; **commit** `{self.commit}` &middot; "
            f"recorded {self.recorded_at}",
            "",
            "> Preserved unedited. The files here are hash-locked and re-checked by "
            "`losscolumn history verify`; a milestone nobody can quietly edit is the "
            "only kind that can still be cited later.",
            "",
        ]
        if self.headline:
            lines += ["## Result", "", "| Metric | Value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in self.headline.items()]
            lines.append("")
        if self.establishes:
            lines += ["## What this establishes", ""]
            lines += [f"- {x}" for x in self.establishes] + [""]
        if self.does_not_establish:
            lines += ["## What it does NOT establish", "",
                      "Carried with the result on purpose: a narrow validation whose "
                      "narrowness gets separated from it becomes a claim it never "
                      "supported.", ""]
            lines += [f"- {x}" for x in self.does_not_establish] + [""]
        if self.frozen_parameters:
            lines += ["## Frozen parameters", "",
                      "Asserted by test. Any change is a new model version, not an "
                      "edit to this one.", "",
                      "| Parameter | Value |", "|---|---|"]
            lines += [f"| `{k}` | {v:g} |" for k, v in sorted(self.frozen_parameters.items())]
            lines.append("")
        if self.preserved:
            lines += ["## Preserved files", "", "| File | Digest |", "|---|---|"]
            lines += [f"| `{n}` | `{d[:23]}...` |" for n, d in sorted(self.preserved.items())]
        return "\n".join(lines)


def record_milestone(artifacts: str | Path, slug: str, milestone: Milestone,
                     files: dict[str, bytes | str]) -> Path:
    d = history_dir(artifacts) / "milestones" / slug
    d.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, content in files.items():
        q = d / name
        if isinstance(content, bytes):
            q.write_bytes(content)
        else:
            q.write_text(content, encoding="utf-8")
        digests[name] = file_hash(q)
    milestone.preserved = digests
    (d / "milestone.json").write_text(
        json.dumps(milestone.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (d / "NOTE.md").write_text(milestone.to_markdown(), encoding="utf-8")
    return d


def load_milestones(artifacts: str | Path) -> list[tuple[Path, Milestone]]:
    root = history_dir(artifacts) / "milestones"
    out: list[tuple[Path, Milestone]] = []
    if not root.exists():
        return out
    for f in sorted(root.glob("*/milestone.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            allowed = set(Milestone.__dataclass_fields__)
            out.append((f.parent, Milestone(**{k: v for k, v in raw.items()
                                               if k in allowed})))
        except Exception:
            continue
    return out


def verify_milestones(artifacts: str | Path) -> list[str]:
    problems: list[str] = []
    for d, m in load_milestones(artifacts):
        for name, expected in m.preserved.items():
            q = d / name
            if not q.exists():
                problems.append(f"milestone {d.name}/{name}: file is missing")
                continue
            actual = file_hash(q)
            if actual != expected:
                problems.append(
                    f"milestone {d.name}/{name}: content changed since it was recorded "
                    f"(recorded {expected[:19]}..., found {actual[:19]}...)"
                )
    return problems
