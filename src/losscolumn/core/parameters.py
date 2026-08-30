"""Model parameters that carry where they came from.

A number in a model is not just a number. ``achievable_mfu = 0.48`` quoted from
a datasheet, fitted from a microbenchmark, or guessed, all read identically at
the point of use -- and only one of them is evidence.

This module attaches provenance to the value and makes the distinction
enforceable rather than documentary: a parameter whose fit was **rejected**
raises when read as an active model input. It can still be inspected, printed
and reported, because a rejected fit is a useful diagnostic; it simply cannot
silently become a model parameter.

The principle comes from a real case in this repository. The gloo all-gather
alpha-beta fit came back at R^2 = 0.715, which means a straight line does not
describe that transport. Its alpha and beta are still reported -- in italics,
marked rejected -- and they are not used. This module generalises that
behaviour so it does not depend on anyone remembering to do it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Provenance", "Parameter", "ParameterSet", "RejectedParameterError"]


class Provenance(str, Enum):
    """Where a parameter's value came from, and whether it may be used."""

    REGISTERED = "registered"
    """Fixed in a sealed protocol before data existed. Usable."""

    MEASURED = "measured"
    """Read directly off hardware by a microbenchmark. Usable."""

    FITTED = "fitted"
    """Estimated from data by a fitting procedure that has not yet been graded.
    Usable, but a fit that is later rejected must be re-marked."""

    ACCEPTED = "accepted"
    """Fitted and passed its quality gate. Usable."""

    REJECTED = "rejected"
    """Fitted and failed its quality gate. NOT usable. Reading it as an active
    model input raises."""

    DIAGNOSTIC = "diagnostic"
    """Recorded to inform a human, never to feed a model. NOT usable."""

    SIMULATED = "simulated"
    """Produced by another model rather than by measurement. Usable only in a
    claim whose evidence class already says ``simulated``."""

    @property
    def usable(self) -> bool:
        return self not in (Provenance.REJECTED, Provenance.DIAGNOSTIC)

    @property
    def is_evidence(self) -> bool:
        """Whether this value rests on a measurement of the target system."""
        return self in (Provenance.MEASURED, Provenance.ACCEPTED)


class RejectedParameterError(RuntimeError):
    """Raised when a rejected or diagnostic parameter is read as a model input."""


@dataclass(frozen=True)
class Parameter:
    """One model parameter, its value, and the account of where it came from."""

    name: str
    value: float
    provenance: Provenance
    unit: str = ""
    source: str = ""
    note: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    superseded_value: float | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError(
                f"parameter {self.name!r} has no source. A value whose origin is not "
                "recorded cannot be graded later, and every parameter in this project "
                "has to be gradeable."
            )

    @property
    def usable(self) -> bool:
        return self.provenance.usable

    def get(self) -> float:
        """The value, for use in a model. Raises if the parameter is not usable."""
        if not self.usable:
            raise RejectedParameterError(
                f"{self.name} is marked {self.provenance.value} and must not be used as "
                f"a model input. {self.note or 'No reason recorded.'} It remains "
                "available for inspection via .value, which is deliberately a "
                "different call."
            )
        return self.value

    def rejected(self, reason: str) -> Parameter:
        """A copy marked rejected, keeping the value visible for the record."""
        return Parameter(
            name=self.name, value=self.value, provenance=Provenance.REJECTED,
            unit=self.unit, source=self.source, note=reason, quality=self.quality,
            superseded_value=self.superseded_value,
        )

    def accepted(self, note: str = "") -> Parameter:
        return Parameter(
            name=self.name, value=self.value, provenance=Provenance.ACCEPTED,
            unit=self.unit, source=self.source, note=note or self.note,
            quality=self.quality, superseded_value=self.superseded_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "provenance": self.provenance.value,
            "usable": self.usable,
            "is_evidence": self.provenance.is_evidence,
            "unit": self.unit,
            "source": self.source,
            "note": self.note,
            "quality": self.quality,
            "superseded_value": self.superseded_value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Parameter:
        return cls(
            name=d["name"], value=float(d["value"]),
            provenance=Provenance(d.get("provenance", "registered")),
            unit=d.get("unit", ""), source=d.get("source", "unrecorded"),
            note=d.get("note", ""), quality=d.get("quality", {}) or {},
            superseded_value=d.get("superseded_value"),
        )

    def describe(self) -> str:
        mark = "" if self.usable else "  [NOT USED]"
        return (
            f"{self.name} = {self.value:g}{' ' + self.unit if self.unit else ''} "
            f"({self.provenance.value}; {self.source}){mark}"
        )


@dataclass
class ParameterSet:
    """A named collection of parameters, with the unusable ones kept visible."""

    name: str
    params: dict[str, Parameter] = field(default_factory=dict)

    def add(self, p: Parameter) -> ParameterSet:
        self.params[p.name] = p
        return self

    def __contains__(self, name: str) -> bool:
        return name in self.params

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.params.values())

    def __len__(self) -> int:
        return len(self.params)

    def get(self, name: str) -> float:
        """Value for use in a model. Raises for a rejected or missing parameter."""
        if name not in self.params:
            raise KeyError(
                f"{self.name} has no parameter {name!r}; it holds "
                f"{sorted(self.params)}"
            )
        return self.params[name].get()

    def get_or(self, name: str, default: float) -> float:
        """Value if usable, otherwise the default. Never raises.

        For call sites that can proceed without a calibrated value, and must
        then say so. A rejected parameter falls back to the default rather than
        being used, which is the whole point.
        """
        p = self.params.get(name)
        return p.value if (p is not None and p.usable) else default

    @property
    def usable(self) -> list[Parameter]:
        return [p for p in self if p.usable]

    @property
    def unusable(self) -> list[Parameter]:
        return [p for p in self if not p.usable]

    @property
    def evidence_backed(self) -> list[Parameter]:
        return [p for p in self if p.provenance.is_evidence]

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self:
            out[p.provenance.value] = out.get(p.provenance.value, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary(),
            "n_usable": len(self.usable),
            "n_unusable": len(self.unusable),
            "n_evidence_backed": len(self.evidence_backed),
            "parameters": [p.to_dict() for p in sorted(self, key=lambda x: x.name)],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParameterSet:
        s = cls(name=d.get("name", "parameters"))
        for pd in d.get("parameters", []):
            s.add(Parameter.from_dict(pd))
        return s

    def to_markdown(self) -> str:
        lines = [
            "| Parameter | Value | Provenance | Source |",
            "|---|---|---|---|",
        ]
        for p in sorted(self, key=lambda x: x.name):
            mark = "" if p.usable else " **not used**"
            lines.append(
                f"| `{p.name}` | {p.value:g} {p.unit} | {p.provenance.value}{mark} | "
                f"{p.source} |"
            )
        unusable = self.unusable
        if unusable:
            lines += ["", "Parameters not used by the model, and why:", ""]
            lines += [f"- `{p.name}`: {p.note}" for p in unusable]
        return "\n".join(lines)
