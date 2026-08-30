"""Measuring feasibility so the answer means what it says.

"Does this fit?" is only a question with an answer if the runtime is made to
answer it. On this platform, left alone, it does not: an over-large allocation
is served from host memory rather than refused, so the run neither fits nor
fails -- it thrashes, for hours. Taken at face value that reads as *feasible*,
and "fits" comes to mean "eventually returned".

So the budget is enforced rather than hoped for, and every determination
records how it was reached:

1. **A hard allocator cap** is installed with
   ``torch.cuda.set_per_process_memory_fraction``. Past the cap the caching
   allocator raises instead of growing. This is the primary mechanism, and if
   it cannot be installed the measurement is ``INVALID`` rather than assumed.
2. **The peak is checked against the cap** after a successful run. An
   allocation that completed while exceeding the budget did not come from the
   device budget, and is recorded as ``HOST_FALLBACK``.
3. **Elapsed time is recorded but never decides.** A run can be slow for many
   reasons and thrashing is only one; using duration as the test would make
   feasibility depend on how busy the machine was. It is kept as corroboration
   and reported.
4. **A failure that is not a memory failure is ``INVALID``.** A shape error is
   not evidence about memory.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from losscolumn.core.feasibility import Feasibility, FeasibilityObservation

# A completed run whose peak exceeded this share of the cap is suspicious even
# if the allocator did not complain: the cap should have refused it. Set just
# above 1.0 so ordinary rounding in the reported peak does not trip it.
OVER_BUDGET_TOLERANCE = 1.02

# Corroborating only. A run slower than this multiple of the median feasible
# run of similar size is flagged in the record; it never changes the verdict.
SLOWNESS_FLAG = 20.0


@dataclass
class ProbeConfig:
    """The runtime configuration a feasibility measurement was taken under."""

    budget_fraction: float = 0.92
    device_index: int = 0
    cap_installed: bool = False
    total_bytes: int = 0
    allocator_backend: str = ""
    env: dict[str, str] | None = None

    @property
    def budget_bytes(self) -> int:
        return int(self.total_bytes * self.budget_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_fraction": self.budget_fraction,
            "budget_bytes": self.budget_bytes,
            "total_bytes": self.total_bytes,
            "device_index": self.device_index,
            "cap_installed": self.cap_installed,
            "cap_mechanism": "torch.cuda.set_per_process_memory_fraction",
            "allocator_backend": self.allocator_backend,
            "env": self.env or {},
        }


def install_budget(budget_fraction: float = 0.92, device: int = 0) -> ProbeConfig:
    """Install the hard device-memory cap, and record whether it took.

    Returns a config with ``cap_installed`` false if the cap could not be set.
    Callers must treat that as a reason to refuse measurement, not as a reason
    to proceed without one: without the cap the platform answers a different
    question than the one being asked.
    """
    import os

    cfg = ProbeConfig(budget_fraction=budget_fraction, device_index=device)
    try:
        import torch

        if not torch.cuda.is_available():
            return cfg
        cfg.total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        cfg.allocator_backend = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "default caching allocator"
        )
        cfg.env = {
            k: v for k, v in os.environ.items()
            if k.startswith(("PYTORCH_CUDA_ALLOC", "CUDA_"))
        }
        torch.cuda.set_per_process_memory_fraction(budget_fraction, device)
        cfg.cap_installed = True
    except Exception:
        cfg.cap_installed = False
    return cfg


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def probe_feasibility(
    run: Callable[[], Any],
    cfg: ProbeConfig,
    *,
    measure_throughput: Callable[[], float] | None = None,
) -> FeasibilityObservation:
    """Run one configuration and classify what happened.

    ``run`` performs the work once and is expected to raise on OOM.
    ``measure_throughput`` is called only after the configuration is known to
    be device-feasible, so a timing run is never spent on something that does
    not fit.
    """
    import torch

    if not cfg.cap_installed:
        return FeasibilityObservation(
            status=Feasibility.INVALID,
            method="none",
            detail=(
                "the device-memory cap could not be installed, so an over-large "
                "allocation would be served from host memory rather than refused. "
                "Feasibility is not measurable in that state and is not guessed at."
            ),
            budget_bytes=cfg.budget_bytes,
            runtime_config=cfg.to_dict(),
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cfg.device_index)
    t0 = time.perf_counter()
    try:
        run()
        torch.cuda.synchronize()
    except RuntimeError as e:
        elapsed = time.perf_counter() - t0
        torch.cuda.empty_cache()
        if _is_oom(e):
            return FeasibilityObservation(
                status=Feasibility.INFEASIBLE,
                method="allocator-cap",
                detail=f"allocator refused inside the budget: {str(e)[:160]}",
                budget_bytes=cfg.budget_bytes,
                elapsed_s=elapsed,
                runtime_config=cfg.to_dict(),
            )
        return FeasibilityObservation(
            status=Feasibility.INVALID,
            method="allocator-cap",
            detail=(
                f"the run failed for a reason that is not memory: "
                f"{type(e).__name__}: {str(e)[:140]}. A shape or kernel error is not "
                "evidence about memory."
            ),
            budget_bytes=cfg.budget_bytes,
            elapsed_s=elapsed,
            runtime_config=cfg.to_dict(),
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        torch.cuda.empty_cache()
        return FeasibilityObservation(
            status=Feasibility.INVALID, method="allocator-cap",
            detail=f"{type(e).__name__}: {str(e)[:160]}",
            budget_bytes=cfg.budget_bytes, elapsed_s=elapsed,
            runtime_config=cfg.to_dict(),
        )

    elapsed = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated(cfg.device_index))
    torch.cuda.empty_cache()

    # The run completed. Whether it completed *on the device* is a separate
    # question, and the peak answers it: a completed run whose peak is past the
    # cap did not obtain that memory from the budget.
    if peak > cfg.budget_bytes * OVER_BUDGET_TOLERANCE:
        return FeasibilityObservation(
            status=Feasibility.HOST_FALLBACK,
            method="allocator-cap+peak-check",
            detail=(
                f"the run completed but peaked at {peak / 1e9:.2f} GB against a "
                f"{cfg.budget_bytes / 1e9:.2f} GB budget. The cap should have refused "
                "it, so the memory did not come from the device budget. Completing is "
                "not the same as fitting."
            ),
            peak_bytes=peak, budget_bytes=cfg.budget_bytes, elapsed_s=elapsed,
            runtime_config=cfg.to_dict(),
        )

    throughput: float | None = None
    if measure_throughput is not None:
        try:
            throughput = float(measure_throughput())
        except Exception:
            throughput = None

    return FeasibilityObservation(
        status=Feasibility.FEASIBLE,
        throughput=throughput,
        peak_bytes=peak,
        budget_bytes=cfg.budget_bytes,
        method="allocator-cap+peak-check",
        detail=f"completed with a peak of {peak / 1e9:.2f} GB inside the budget",
        elapsed_s=elapsed,
        runtime_config=cfg.to_dict(),
    )


def release_budget(device: int = 0) -> None:
    """Remove the cap. Left installed, it would silently constrain later work."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(1.0, device)
    except Exception:
        pass
