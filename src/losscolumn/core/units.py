"""Metric direction and unit bookkeeping.

Every quantitative claim in the standard carries a direction.  Getting this
wrong silently inverts a loss column into a win column, so direction is a
required field everywhere rather than an inferred convention.
"""

from __future__ import annotations

from dataclasses import dataclass

# Units the standard knows how to reason about.  Anything else is accepted but
# marked ``opaque``: it can be compared, but not converted or aggregated.
KNOWN_UNITS: dict[str, str] = {
    "tok/s": "throughput",
    "req/s": "throughput",
    "samples/s": "throughput",
    "TFLOP/s": "throughput",
    "ms": "time",
    "us": "time",
    "s": "time",
    "GB": "memory",
    "MiB": "memory",
    "GB/s": "bandwidth",
    "fraction": "dimensionless",
    "ratio": "dimensionless",
    "count": "dimensionless",
}

_TIME_TO_MS = {"us": 1e-3, "ms": 1.0, "s": 1e3}


@dataclass(frozen=True)
class Direction:
    """Which way is better for a metric."""

    higher_is_better: bool

    def regression_sign(self) -> float:
        """Multiplier that turns ``log(method) - log(baseline)`` into a regression.

        Positive output always means *the method is worse*.
        """
        return -1.0 if self.higher_is_better else 1.0


def unit_kind(unit: str) -> str:
    return KNOWN_UNITS.get(unit, "opaque")


def to_ms(value: float, unit: str) -> float:
    if unit not in _TIME_TO_MS:
        raise ValueError(f"{unit!r} is not a time unit; cannot convert to ms")
    return value * _TIME_TO_MS[unit]


def fmt_pct(x: float, digits: int = 1) -> str:
    """Format a log-ratio effect as a human-facing percentage."""
    import math

    pct = (math.exp(x) - 1.0) * 100.0
    return f"{pct:+.{digits}f}%"


def fmt_speedup(x: float, digits: int = 2) -> str:
    import math

    return f"{math.exp(x):.{digits}f}x"
