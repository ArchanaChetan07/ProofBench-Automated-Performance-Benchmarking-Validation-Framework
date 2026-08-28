"""The engine adapter contract.

Every engine in the audit -- real or simulated -- is reached through this one
interface. The point is not abstraction for its own sake: it is that the
tuning loop, the measurement loop and the parity ledger must be *identical
code* across engines. If vLLM were tuned by one script and SGLang by another,
tuning-budget parity would be a claim about two scripts rather than a property
of the experiment.

An adapter supplies four things and nothing else:

``search_space``  the knobs it may be tuned over, with their candidate values
``defaults``      the library's own defaults, measured as a first-class result
``serve``         run one configuration against one workload, return a measurement
``describe``      version and provenance of the engine actually under test
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from losscolumn.core.parity import SearchSpace
from losscolumn.thrusts.engines.workloads import Workload


@dataclass
class ServeResult:
    """One measurement of one configuration on one workload."""

    ttft_ms_p50: float
    ttft_ms_p99: float
    tpot_ms_p50: float
    tpot_ms_p99: float
    e2e_ms_p50: float
    e2e_ms_p99: float
    output_throughput: float          # output tok/s
    total_throughput: float           # prompt + output tok/s
    request_throughput: float         # req/s
    # Normalised latency: end-to-end time divided by the number of output
    # tokens the request asked for, percentiled ACROSS REQUESTS. Dividing a p99
    # end-to-end figure by a median output length mixes two different requests
    # and produces a number that describes neither of them.
    norm_latency_ms_p50: float = float("nan")
    norm_latency_ms_p99: float = float("nan")
    goodput: float = 0.0              # req/s meeting the workload's SLO
    n_ok: int = 0
    n_failed: int = 0
    peak_kv_utilisation: float = 0.0
    prefix_hit_rate: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.n_ok > 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["ok"] = self.ok
        return d


class EngineAdapter(Protocol):
    name: str
    evidence_class: str          # "measured" | "simulated"

    def describe(self) -> dict[str, Any]: ...

    def search_space(self) -> SearchSpace: ...

    def defaults(self) -> dict[str, Any]: ...

    def serve(self, config: dict[str, Any], workload: Workload, *,
              concurrency: int, seed: int) -> ServeResult: ...


@dataclass
class EngineUnavailable(Exception):
    """Raised when a real engine is not installed on this machine.

    Carries the reason so the artifact can state which engines were reachable
    rather than quietly reporting a comparison over whichever subset happened
    to import.
    """

    engine: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.engine} unavailable: {self.reason}"
