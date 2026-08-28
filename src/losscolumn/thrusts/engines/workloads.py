"""Workload shapes for the engine audit.

Four shapes, chosen because they place load on different parts of a serving
stack and because published comparisons almost always report only the first
one. An engine tuned for aggregate throughput on short chat traffic can be a
poor choice for long-prefill retrieval traffic, and a single-workload benchmark
cannot show that.

Each workload is a *trace*, not a distribution parameter set: the same request
sequence, with the same arrival times and the same prompts, is replayed against
every engine. Sampling fresh requests per engine would leave a difference in
the input as a confound in the output, and the trace digest in the parity
certificate is what proves the inputs were identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from losscolumn.core.provenance import content_hash


@dataclass
class Request:
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    prefix_group: int = -1     # requests sharing a group share a system prompt

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Workload:
    """A replayable request trace."""

    name: str
    description: str
    requests: list[Request] = field(default_factory=list)
    shared_prefix_tokens: int = 0
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None

    @property
    def n_requests(self) -> int:
        return len(self.requests)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.requests)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.requests)

    @property
    def prefix_reuse_fraction(self) -> float:
        """Share of prompt tokens that a prefix cache could serve from a hit.

        This is the quantity RadixAttention monetises, and it is a property of
        the *workload*, not of the engine. Publishing an engine comparison on a
        trace with 0% or 90% reuse and not saying which is the single easiest
        way to make either engine look dominant.
        """
        if not self.requests or not self.shared_prefix_tokens:
            return 0.0
        seen: set[int] = set()
        reused = 0
        for r in self.requests:
            if r.prefix_group >= 0:
                if r.prefix_group in seen:
                    reused += min(self.shared_prefix_tokens, r.prompt_tokens)
                seen.add(r.prefix_group)
        return reused / max(self.total_prompt_tokens, 1)

    def digest(self) -> str:
        return content_hash(
            {
                "name": self.name,
                "shared_prefix_tokens": self.shared_prefix_tokens,
                "requests": [r.to_dict() for r in self.requests],
            }
        )

    def summary(self) -> dict[str, Any]:
        pl = [r.prompt_tokens for r in self.requests] or [0]
        ol = [r.output_tokens for r in self.requests] or [0]
        span = max((r.arrival_s for r in self.requests), default=0.0)
        return {
            "name": self.name,
            "description": self.description,
            "n_requests": self.n_requests,
            "prompt_tokens_median": int(np.median(pl)),
            "prompt_tokens_p95": int(np.percentile(pl, 95)),
            "output_tokens_median": int(np.median(ol)),
            "arrival_span_s": span,
            "request_rate_qps": self.n_requests / span if span else float("inf"),
            "prefix_reuse_fraction": self.prefix_reuse_fraction,
            "digest": self.digest(),
        }


def _trace(
    name: str,
    description: str,
    n: int,
    rate_qps: float,
    prompt_lo: int,
    prompt_hi: int,
    out_lo: int,
    out_hi: int,
    *,
    seed: int,
    prefix_groups: int = 0,
    shared_prefix_tokens: int = 0,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
) -> Workload:
    rng = np.random.default_rng(seed)
    # Poisson arrivals: bursty enough to exercise the scheduler, which a
    # fixed-rate generator never does.
    gaps = rng.exponential(1.0 / rate_qps, size=n)
    arrivals = np.cumsum(gaps)
    # Log-uniform lengths: real prompt distributions are heavy tailed, and a
    # uniform draw understates the tail that actually breaks schedulers.
    prompts = np.exp(rng.uniform(np.log(prompt_lo), np.log(prompt_hi), size=n)).astype(int)
    outs = np.exp(rng.uniform(np.log(out_lo), np.log(out_hi), size=n)).astype(int)
    groups = (
        rng.integers(0, prefix_groups, size=n) if prefix_groups else np.full(n, -1)
    )
    return Workload(
        name=name,
        description=description,
        requests=[
            Request(float(a), int(p), int(o), int(g))
            for a, p, o, g in zip(arrivals, prompts, outs, groups)
        ],
        shared_prefix_tokens=shared_prefix_tokens,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
    )


def standard_workloads(n: int = 400, seed: int = 20260101) -> dict[str, Workload]:
    """The four shapes the audit reports. All four, always, in every table."""
    return {
        "chat": _trace(
            "chat",
            "Short interactive turns; the shape most published comparisons use.",
            n, 8.0, 64, 1024, 64, 512, seed=seed, slo_ttft_ms=500, slo_tpot_ms=50,
        ),
        "rag": _trace(
            "rag",
            "Long retrieval-augmented prefill with a shared system prompt and short answers.",
            n, 4.0, 2048, 16384, 32, 256, seed=seed + 1,
            prefix_groups=8, shared_prefix_tokens=1024,
            slo_ttft_ms=2000, slo_tpot_ms=50,
        ),
        "summarize": _trace(
            "summarize",
            "Long input, long output; decode dominated with a heavy prefill.",
            n, 2.0, 4096, 32768, 512, 2048, seed=seed + 2,
            slo_ttft_ms=5000, slo_tpot_ms=80,
        ),
        "agentic": _trace(
            "agentic",
            "Many short calls reusing one long tool-definition prefix; high concurrency.",
            n * 2, 32.0, 512, 4096, 16, 128, seed=seed + 3,
            prefix_groups=3, shared_prefix_tokens=2048,
            slo_ttft_ms=800, slo_tpot_ms=40,
        ),
    }


def workload_digest(workloads: Sequence[Workload]) -> str:
    return content_hash(sorted(w.digest() for w in workloads))
