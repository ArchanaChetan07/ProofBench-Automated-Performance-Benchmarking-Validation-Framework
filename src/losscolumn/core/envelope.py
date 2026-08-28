"""Measurement envelopes: the unit of evidence in the standard.

Requirement LC-3 says results are surfaces, not points. An ``Envelope`` is that
surface: a full factorial grid over the operating factors, holding replicate
measurements for every system under comparison in every cell.

The design commitments worth naming:

* **Replicates are kept, never pre-averaged.** Every downstream statistic --
  paired tests, bootstrap intervals, cliff reproducibility -- needs the spread,
  and an artifact that ships only means cannot be re-analysed by a skeptic.
* **Cells may be missing.** Real sweeps OOM. A missing cell is recorded as
  missing with a reason, and is reported; it is never silently dropped, because
  "the method OOMs here" is exactly the kind of loss the standard exists to
  surface.
* **Measurement order is recorded.** Paired statistics are only valid if the
  systems were measured interleaved rather than in blocks, and the envelope
  carries the flag that lets the validator check the claim.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from losscolumn.core.provenance import Provenance, content_hash

Level = int | float | str
Cell = tuple[int, ...]


@dataclass(frozen=True)
class Factor:
    """One axis of the operating envelope."""

    name: str
    levels: tuple[Level, ...]
    ordered: bool = True
    unit: str | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        if len(self.levels) == 0:
            raise ValueError(f"factor {self.name!r} has no levels")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"factor {self.name!r} has duplicate levels")

    def index_of(self, level: Level) -> int:
        return self.levels.index(level)

    def numeric(self) -> np.ndarray | None:
        """Level positions as floats, or ``None`` for categorical factors."""
        if not self.ordered:
            return None
        try:
            return np.asarray([float(x) for x in self.levels], dtype=float)
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "levels": list(self.levels),
            "ordered": self.ordered,
            "unit": self.unit,
            "log_scale": self.log_scale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Factor:
        return cls(
            name=d["name"],
            levels=tuple(d["levels"]),
            ordered=d.get("ordered", True),
            unit=d.get("unit"),
            log_scale=d.get("log_scale", False),
        )


@dataclass(frozen=True)
class Metric:
    """What is being measured, and which direction is better."""

    name: str
    unit: str
    higher_is_better: bool
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Metric:
        return cls(
            name=d["name"],
            unit=d["unit"],
            higher_is_better=bool(d["higher_is_better"]),
            description=d.get("description"),
        )


@dataclass
class Envelope:
    """A factorial grid of replicate measurements for two or more systems.

    ``data[system]`` has shape ``grid_shape + (replicates,)`` and holds NaN for
    absent replicates. ``missing[system]`` maps a cell to the reason it could
    not be measured at all.
    """

    factors: tuple[Factor, ...]
    metric: Metric
    data: dict[str, np.ndarray] = field(default_factory=dict)
    missing: dict[str, dict[Cell, str]] = field(default_factory=dict)
    replicates: int = 0
    interleaved: bool = False
    workload: str | None = None
    provenance: Provenance | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- construction -----------------------------------------------------

    @classmethod
    def allocate(
        cls,
        factors: Sequence[Factor],
        metric: Metric,
        systems: Sequence[str],
        replicates: int,
        **kw: Any,
    ) -> Envelope:
        factors = tuple(factors)
        shape = tuple(len(f.levels) for f in factors) + (replicates,)
        env = cls(
            factors=factors,
            metric=metric,
            data={s: np.full(shape, np.nan, dtype=float) for s in systems},
            missing={s: {} for s in systems},
            replicates=replicates,
            **kw,
        )
        return env

    # ---- shape and addressing --------------------------------------------

    @property
    def systems(self) -> tuple[str, ...]:
        return tuple(self.data.keys())

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(len(f.levels) for f in self.factors)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.grid_shape)) if self.factors else 0

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.factors)

    def axis(self, name: str) -> int:
        return self.factor_names.index(name)

    def factor(self, name: str) -> Factor:
        return self.factors[self.axis(name)]

    def cells(self) -> Iterator[Cell]:
        return itertools.product(*(range(n) for n in self.grid_shape))

    def coords(self, cell: Cell) -> dict[str, Level]:
        return {f.name: f.levels[i] for f, i in zip(self.factors, cell)}

    def cell_of(self, **coords: Level) -> Cell:
        return tuple(f.index_of(coords[f.name]) for f in self.factors)

    def label(self, cell: Cell, sep: str = ", ") -> str:
        return sep.join(f"{f.name}={f.levels[i]}" for f, i in zip(self.factors, cell))

    # ---- writing ----------------------------------------------------------

    def put(self, system: str, cell: Cell, values: Sequence[float]) -> None:
        arr = self.data[system]
        v = np.asarray(values, dtype=float)
        if v.size > self.replicates:
            raise ValueError(
                f"{v.size} replicates supplied for a grid allocated with {self.replicates}"
            )
        arr[cell + (slice(0, v.size),)] = v

    def mark_missing(self, system: str, cell: Cell, reason: str) -> None:
        """Record that a cell could not be measured, and why.

        The reason string is surfaced verbatim in the loss column: an OOM at
        ``batch=64, seq=32768`` is a loss even though no number exists for it.
        """
        self.missing.setdefault(system, {})[tuple(cell)] = reason
        self.data[system][tuple(cell)] = np.nan

    # ---- reading ----------------------------------------------------------

    def replicates_at(self, system: str, cell: Cell) -> np.ndarray:
        v = self.data[system][tuple(cell)]
        return v[~np.isnan(v)]

    def summary(self, system: str, stat: str = "median") -> np.ndarray:
        """Per-cell point estimate; NaN where the cell has no data."""
        arr = self.data[system]
        with np.errstate(invalid="ignore"):
            if stat == "median":
                out = np.nanmedian(arr, axis=-1)
            elif stat == "mean":
                out = np.nanmean(arr, axis=-1)
            elif stat == "min":
                out = np.nanmin(arr, axis=-1)
            elif stat == "max":
                out = np.nanmax(arr, axis=-1)
            else:
                raise ValueError(f"unknown stat {stat!r}")
        return np.asarray(out, dtype=float)

    def is_measured(self, system: str, cell: Cell) -> bool:
        return bool(self.replicates_at(system, cell).size)

    def coverage(self) -> dict[str, float]:
        """Fraction of cells with at least one replicate, per system."""
        return {
            s: float(np.mean([self.is_measured(s, c) for c in self.cells()]))
            for s in self.systems
        }

    def noise(self, system: str) -> dict[str, float]:
        """Robust within-cell dispersion, as a run-quality signal.

        A claim whose measurement noise is comparable to its claimed effect is
        not a claim, and the validator uses this to say so.
        """
        cvs: list[float] = []
        for c in self.cells():
            v = self.replicates_at(system, c)
            if v.size >= 2 and np.median(v) > 0:
                mad = float(np.median(np.abs(v - np.median(v))))
                cvs.append(1.4826 * mad / float(np.median(v)))
        if not cvs:
            return {"median_cv": float("nan"), "p95_cv": float("nan"), "n": 0}
        return {
            "median_cv": float(np.median(cvs)),
            "p95_cv": float(np.percentile(cvs, 95)),
            "n": len(cvs),
        }

    # ---- slicing ----------------------------------------------------------

    def slice(self, **fixed: Level) -> Envelope:
        """Sub-envelope with some factors pinned to single levels."""
        keep = [f for f in self.factors if f.name not in fixed]
        idx: list[Any] = []
        for f in self.factors:
            idx.append(f.index_of(fixed[f.name]) if f.name in fixed else slice(None))
        new_data = {s: a[tuple(idx)] for s, a in self.data.items()}
        return Envelope(
            factors=tuple(keep),
            metric=self.metric,
            data=new_data,
            missing={s: {} for s in self.systems},
            replicates=self.replicates,
            interleaved=self.interleaved,
            workload=self.workload,
            provenance=self.provenance,
            meta={**self.meta, "sliced": fixed},
        )

    def fibers(self, axis_name: str) -> Iterator[tuple[dict[str, Level], Cell]]:
        """Iterate 1-D fibers along ``axis_name``.

        Yields ``(fixed_coords, base_cell)`` where ``base_cell`` has index 0 on
        the swept axis. Used by cliff detection, which is a per-fiber question.
        """
        ax = self.axis(axis_name)
        other = [i for i in range(len(self.factors)) if i != ax]
        for combo in itertools.product(*(range(self.grid_shape[i]) for i in other)):
            cell = [0] * len(self.factors)
            for i, v in zip(other, combo):
                cell[i] = v
            fixed = {self.factors[i].name: self.factors[i].levels[v] for i, v in zip(other, combo)}
            yield fixed, tuple(cell)

    def fiber_cells(self, axis_name: str, base: Cell) -> list[Cell]:
        ax = self.axis(axis_name)
        out = []
        for k in range(self.grid_shape[ax]):
            c = list(base)
            c[ax] = k
            out.append(tuple(c))
        return out

    # ---- serialisation ----------------------------------------------------

    def to_dict(self, *, include_replicates: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "envelope",
            "factors": [f.to_dict() for f in self.factors],
            "metric": self.metric.to_dict(),
            "systems": list(self.systems),
            "replicates": self.replicates,
            "interleaved": self.interleaved,
            "workload": self.workload,
            "meta": self.meta,
            "coverage": self.coverage(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
        if include_replicates:
            d["data"] = {
                s: [[None if np.isnan(x) else float(x) for x in row] for row in a.reshape(-1, self.replicates)]
                for s, a in self.data.items()
            }
        else:
            d["summary"] = {s: self.summary(s).ravel().tolist() for s in self.systems}
        d["missing"] = {
            s: [{"cell": list(c), "reason": r} for c, r in sorted(m.items())]
            for s, m in self.missing.items()
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Envelope:
        factors = tuple(Factor.from_dict(x) for x in d["factors"])
        metric = Metric.from_dict(d["metric"])
        replicates = int(d["replicates"])
        shape = tuple(len(f.levels) for f in factors) + (replicates,)
        data = {}
        for s, rows in d["data"].items():
            flat = np.array(
                [[np.nan if x is None else float(x) for x in row] for row in rows], dtype=float
            )
            data[s] = flat.reshape(shape)
        missing = {
            s: {tuple(e["cell"]): e["reason"] for e in entries}
            for s, entries in d.get("missing", {}).items()
        }
        prov = d.get("provenance")
        return cls(
            factors=factors,
            metric=metric,
            data=data,
            missing=missing,
            replicates=replicates,
            interleaved=bool(d.get("interleaved", False)),
            workload=d.get("workload"),
            provenance=Provenance.from_dict(prov) if prov else None,
            meta=d.get("meta", {}),
        )

    def digest(self) -> str:
        """Content hash of the measurements, for binding results to a claim."""
        return content_hash(self.to_dict(include_replicates=True))

    def __repr__(self) -> str:  # pragma: no cover - display only
        dims = " x ".join(f"{f.name}[{len(f.levels)}]" for f in self.factors)
        return (
            f"<Envelope {self.metric.name} ({self.metric.unit}) "
            f"{dims} systems={list(self.systems)} r={self.replicates}>"
        )
