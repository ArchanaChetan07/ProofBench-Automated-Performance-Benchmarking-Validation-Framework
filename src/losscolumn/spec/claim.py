"""The data model of a conforming performance claim (standard LC-1.0).

A claim is the unit the standard grades. It bundles the assertion, the surface
it was measured on, the loss column, the parity certificate, the sealed
protocol, and the reproduction recipe -- in one content-addressed object, so
that "the paper says X" and "the artifact shows X" cannot drift apart.

Nothing here is specific to attention kernels or serving engines. Thrusts I,
II and III all emit this type, which is what makes the standard portable to
work that has nothing to do with this proposal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from losscolumn.core.envelope import Envelope
from losscolumn.core.losscolumn import LossColumn
from losscolumn.core.parity import ParityCertificate
from losscolumn.core.prereg import PreRegistration
from losscolumn.core.provenance import Provenance, content_hash, utcnow
from losscolumn.version import STANDARD_VERSION


@dataclass
class Reproduction:
    """How a third party re-runs this exact measurement without asking anyone."""

    command: str
    repo: str | None = None
    commit: str | None = None
    dirty: bool | None = None
    container_image: str | None = None
    hardware: str = ""
    estimated_runtime_min: float | None = None
    estimated_cost_usd: float | None = None
    data_inputs: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Headline:
    """The sentence a reader will quote, forced to carry its own envelope.

    A headline may not be a bare scalar. ``best`` and ``worst`` are both
    required, which is the smallest possible structural defence against
    "up to 2x faster".
    """

    statement: str
    best_pct: float
    worst_pct: float
    median_pct: float
    conditions_best: dict[str, Any] = field(default_factory=dict)
    conditions_worst: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return (
            f"{self.statement} Median {self.median_pct:+.1f}% across the swept envelope; "
            f"best {self.best_pct:+.1f}% at {_fmt(self.conditions_best)}; "
            f"worst {self.worst_pct:+.1f}% at {_fmt(self.conditions_worst)}."
        )


def _fmt(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "n/a"


@dataclass
class Claim:
    """One falsifiable performance claim, with everything needed to falsify it."""

    id: str
    title: str
    thrust: str                                  # "I" | "II" | "III" | free text
    method: str
    baseline: str
    headline: Headline
    loss_column: LossColumn
    reproduction: Reproduction
    standard_version: str = STANDARD_VERSION
    created_at: str = field(default_factory=utcnow)
    # "measured" | "simulated" | "mixed". A model-generated number and a
    # measured one are both useful and are not the same kind of evidence.
    # Making this a required, validated, prominently rendered field is the
    # cheapest possible defence against the two being confused later, by a
    # reader or by the author six months on.
    evidence_class: str = "measured"
    evidence_note: str = ""
    envelope: Envelope | None = None
    envelope_ref: str | None = None              # path, when the envelope ships separately
    prereg: PreRegistration | None = None
    prereg_verification: dict[str, Any] | None = None
    parity: ParityCertificate | None = None
    provenance: Provenance | None = None
    loss_column_placement: str = "main"          # "main" | "appendix"
    attributions: dict[str, str] = field(default_factory=dict)
    supporting: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)

    # ---- serialisation ----------------------------------------------------

    def to_dict(self, *, include_envelope: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "claim",
            "standard_version": self.standard_version,
            "id": self.id,
            "title": self.title,
            "thrust": self.thrust,
            "method": self.method,
            "baseline": self.baseline,
            "created_at": self.created_at,
            "evidence_class": self.evidence_class,
            "evidence_note": self.evidence_note,
            "authors": self.authors,
            "headline": self.headline.to_dict(),
            "headline_rendered": self.headline.render(),
            "loss_column": self.loss_column.to_dict(),
            "loss_column_placement": self.loss_column_placement,
            "reproduction": self.reproduction.to_dict(),
            "attributions": self.attributions,
            "limitations": self.limitations,
            "supporting": self.supporting,
            "envelope_ref": self.envelope_ref,
            "prereg": self.prereg.to_dict() if self.prereg else None,
            "prereg_verification": self.prereg_verification,
            "parity": self.parity.to_dict() if self.parity else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
        if include_envelope and self.envelope is not None:
            d["envelope"] = self.envelope.to_dict()
        d["digest"] = content_hash({k: v for k, v in d.items() if k != "digest"})
        return d

    def save(self, path: str | Path, *, include_envelope: bool = True) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(include_envelope=include_envelope), indent=2, default=str),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Claims are loaded as plain dicts.

        The validator grades documents, not objects: an artifact produced by
        someone else's tooling must be gradable, and requiring it to
        round-trip through this package's classes would defeat the point of
        publishing a standard.
        """
        return json.loads(Path(path).read_text(encoding="utf-8"))
