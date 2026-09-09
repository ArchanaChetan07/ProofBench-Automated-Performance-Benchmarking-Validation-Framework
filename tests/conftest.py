"""Shared test fixtures.

`_session` lives here rather than in one test module that another imports.
`from tests.test_stability import ...` resolves only when the repository root
happens to be on sys.path, which it is when pytest is invoked from the root and
is not when the package is installed and the suite run from elsewhere. The
clean-clone check runs from the root, so it never caught this; a rented machine
running the same suite from an installed copy did, on the first try.

pytest puts a conftest's directory on sys.path itself, so anything here is
importable from every test module however the suite is invoked.
"""

from __future__ import annotations

import math

import pytest

from losscolumn.core.stability import SentinelReading, SessionState

PROBES: list[tuple[str, int]] = [
    ("all_gather", 1 << 16), ("all_gather", 1 << 20), ("all_gather", 1 << 23),
    ("all_reduce", 1 << 16), ("all_reduce", 1 << 20), ("all_reduce", 1 << 23),
]

# A plausible base surface: time grows with size, all_reduce costs about double.
BASE: dict[tuple[str, int], float] = {
    ("all_gather", 1 << 16): 1.0e-4, ("all_gather", 1 << 20): 1.2e-3,
    ("all_gather", 1 << 23): 9.0e-3, ("all_reduce", 1 << 16): 1.9e-4,
    ("all_reduce", 1 << 20): 2.3e-3, ("all_reduce", 1 << 23): 1.8e-2,
}


def build_session(sid: str, *, scale=None, jitter: float = 0.0,
                  n_restarts: int = 3, repeats: int = 7, world: int = 3,
                  probes=None, state: SessionState | None = None
                  ) -> list[SentinelReading]:
    """A synthetic session. `scale` may be a float or a per-probe callable.

    The jitter is a deterministic function of the indices rather than an RNG
    draw, so a failure reproduces exactly instead of only usually.
    """
    out: list[SentinelReading] = []
    probes = probes if probes is not None else PROBES
    for r in range(n_restarts):
        for i, (kind, nb) in enumerate(probes):
            base = BASE[(kind, nb)]
            f = scale(kind, nb) if callable(scale) else (
                1.0 if scale is None else scale)
            ts = [base * f * (1.0 + jitter * math.sin(
                3.7 * (r + 1) + 1.9 * i + 0.6 * j)) for j in range(repeats)]
            out.append(SentinelReading(
                session_id=sid, restart_index=r, collective=kind, world=world,
                nbytes=nb, size_class="x", timings_s=ts,
                state=state or SessionState(session_id=sid, restart_index=r),
            ))
    return out


@pytest.fixture
def session_builder():
    return build_session
