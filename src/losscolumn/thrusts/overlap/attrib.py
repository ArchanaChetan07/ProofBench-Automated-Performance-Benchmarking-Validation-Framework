"""Attribution of a training step into compute, communication, exposed and idle.

The metric the proposal defines -- "overlap efficiency: the fraction of
collective time hidden behind compute" -- is only well defined if you first say
what "collective time" means on a timeline where kernels on several streams run
concurrently. Summing NCCL kernel durations double-counts concurrent
collectives; subtracting summed compute from summed comm can go negative. The
computation below is set-theoretic and therefore always in [0, 1]:

    C = union of compute kernel intervals
    M = union of communication kernel intervals
    exposed = M \\ C
    overlap_efficiency = 1 - |exposed| / |M|

Two derived quantities matter as much as the headline:

``exposed_frac_of_step``  exposed comm as a share of step wall-clock. This is
                          the number that actually costs money; a step can be
                          95% overlapped and still lose a fifth of its time to
                          the 5% that is not.
``idle_frac``             wall-clock covered by neither. Large idle means the
                          bottleneck is not the network at all -- launch
                          overhead, the data loader, a straggler -- and a
                          sharding change will not fix it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.intervals import IntervalSet

# Substring rules for classifying kernel names. Deliberately explicit and
# ordered: an unclassified kernel is reported, never silently bucketed, because
# a misclassified kernel moves the headline number.
COMM_MARKERS = (
    "nccl",
    "ncclDevKernel",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "all_reduce",
    "allreduce",
    "broadcast",
    "alltoall",
    "all_to_all",
    "send",
    "recv",
    "c10d",
)
COMPUTE_MARKERS = (
    "gemm",
    "sgemm",
    "cutlass",
    "wgrad",
    "dgrad",
    "conv",
    "attention",
    "flash",
    "softmax",
    "layer_norm",
    "layernorm",
    "rms_norm",
    "elementwise",
    "vectorized_elementwise",
    "reduce_kernel",
    "at::native",
    "triton_",
    "ampere_",
    "sm80_",
    "sm90_",
    "cublas",
    "cudnn",
)
MEMORY_MARKERS = ("memcpy", "memset", "copy_device_to_device", "d2d", "h2d", "d2h")


@dataclass
class Span:
    """One kernel or range on the timeline, in milliseconds."""

    name: str
    start: float
    end: float
    stream: int | str | None = None
    category: str | None = None       # compute | comm | memory | other
    bytes: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.end - self.start


def classify(name: str) -> str:
    n = name.lower()
    for m in COMM_MARKERS:
        if m.lower() in n:
            return "comm"
    for m in MEMORY_MARKERS:
        if m in n:
            return "memory"
    for m in COMPUTE_MARKERS:
        if m.lower() in n:
            return "compute"
    return "other"


@dataclass
class OverlapMetrics:
    """The attribution result for one step."""

    step_ms: float
    compute_ms: float
    comm_ms: float
    exposed_ms: float
    idle_ms: float
    memory_ms: float = 0.0
    unclassified_ms: float = 0.0
    n_comm_kernels: int = 0
    n_compute_kernels: int = 0
    unclassified_kernels: list[str] = field(default_factory=list)

    @property
    def overlap_efficiency(self) -> float:
        """Fraction of communication hidden behind compute. NaN if no comm."""
        if self.comm_ms <= 0:
            return float("nan")
        return 1.0 - self.exposed_ms / self.comm_ms

    @property
    def exposed_frac_of_step(self) -> float:
        return self.exposed_ms / self.step_ms if self.step_ms > 0 else float("nan")

    @property
    def compute_frac_of_step(self) -> float:
        return self.compute_ms / self.step_ms if self.step_ms > 0 else float("nan")

    @property
    def idle_frac(self) -> float:
        return self.idle_ms / self.step_ms if self.step_ms > 0 else float("nan")

    @property
    def comm_bound(self) -> bool:
        """Whether removing all exposed comm would materially shorten the step."""
        return self.exposed_frac_of_step > 0.05

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.update(
            overlap_efficiency=self.overlap_efficiency,
            exposed_frac_of_step=self.exposed_frac_of_step,
            compute_frac_of_step=self.compute_frac_of_step,
            idle_frac=self.idle_frac,
            comm_bound=self.comm_bound,
        )
        return d


@dataclass
class StepTrace:
    """All spans belonging to one optimizer step, on one rank."""

    spans: list[Span]
    step_start: float | None = None
    step_end: float | None = None
    rank: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], **kw: Any) -> StepTrace:
        """Build from generic ``{name, ts, dur}`` records (Chrome trace units: us)."""
        spans = []
        for r in records:
            start = float(r.get("ts", r.get("start", 0.0)))
            dur = float(r.get("dur", r.get("duration", 0.0)))
            if r.get("_units", "us") == "us":
                start, dur = start / 1000.0, dur / 1000.0
            spans.append(
                Span(
                    name=str(r.get("name", "")),
                    start=start,
                    end=start + dur,
                    stream=r.get("stream", r.get("tid")),
                    category=r.get("category"),
                    bytes=r.get("bytes"),
                )
            )
        return cls(spans=spans, **kw)

    # ---- attribution ------------------------------------------------------

    def bucket(self, category: str) -> IntervalSet:
        return IntervalSet.from_spans(
            (s.start, s.end) for s in self.spans if (s.category or classify(s.name)) == category
        )

    def bounds(self) -> tuple[float, float]:
        if self.step_start is not None and self.step_end is not None:
            return self.step_start, self.step_end
        if not self.spans:
            return 0.0, 0.0
        return min(s.start for s in self.spans), max(s.end for s in self.spans)

    def attribute(self) -> OverlapMetrics:
        lo, hi = self.bounds()
        compute = self.bucket("compute").clip(lo, hi)
        comm = self.bucket("comm").clip(lo, hi)
        memory = self.bucket("memory").clip(lo, hi)
        other = self.bucket("other").clip(lo, hi)
        exposed = comm.difference(compute)
        busy = compute.union(comm).union(memory).union(other)
        idle = IntervalSet(((lo, hi),)).difference(busy)
        unc = [s.name for s in self.spans if (s.category or classify(s.name)) == "other"]
        return OverlapMetrics(
            step_ms=hi - lo,
            compute_ms=compute.measure(),
            comm_ms=comm.measure(),
            exposed_ms=exposed.measure(),
            idle_ms=idle.measure(),
            memory_ms=memory.measure(),
            unclassified_ms=other.measure(),
            n_comm_kernels=sum(1 for s in self.spans if (s.category or classify(s.name)) == "comm"),
            n_compute_kernels=sum(
                1 for s in self.spans if (s.category or classify(s.name)) == "compute"
            ),
            unclassified_kernels=sorted(set(unc))[:20],
        )

    def exposed_spans(self) -> IntervalSet:
        lo, hi = self.bounds()
        return self.bucket("comm").clip(lo, hi).difference(self.bucket("compute").clip(lo, hi))

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.bounds()
        return {
            "rank": self.rank,
            "config": self.config,
            "step_start": lo,
            "step_end": hi,
            "n_spans": len(self.spans),
            "metrics": self.attribute().to_dict(),
        }


def aggregate(metrics: Sequence[OverlapMetrics]) -> dict[str, Any]:
    """Median across steps, with the spread that a single-step figure would hide."""
    import numpy as np

    if not metrics:
        return {}

    def q(f) -> dict[str, float]:
        v = np.array([f(m) for m in metrics], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
        return {
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
        }

    return {
        "n_steps": len(metrics),
        "step_ms": q(lambda m: m.step_ms),
        "overlap_efficiency": q(lambda m: m.overlap_efficiency),
        "exposed_frac_of_step": q(lambda m: m.exposed_frac_of_step),
        "idle_frac": q(lambda m: m.idle_frac),
    }
