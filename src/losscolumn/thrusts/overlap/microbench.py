"""Communication microbenchmarks, and the alpha-beta fit they feed.

The Thrust I model prices every collective as

    t(n) = alpha * log2(world) + factor * (world-1)/world * n / beta

with ``alpha`` a per-collective latency floor and ``beta`` a bus bandwidth.
Those two numbers drive the model's loss regions, and until now both were
quoted from an A100 datasheet rather than measured. This module measures them.

**What can honestly be measured here, and what cannot.** The registered
protocol is about NVLink and InfiniBand on an eight-GPU node. A single-GPU
workstation has neither. What it does have is real transports with the same
cost structure -- a latency floor, a bandwidth slope, and a message size where
one gives way to the other:

``pcie_h2d`` / ``pcie_d2h``   host-to-device copies over PCIe. A real fabric,
                              and the one an inter-node collective traverses on
                              its way to the NIC.
``gloo_*``                    real collectives, over shared memory between
                              processes, with real ring algorithms.

Fitting those calibrates the *functional form* and the *fitting procedure*, and
produces alpha and beta for transports that exist on this machine. It does not
produce NVLink or InfiniBand parameters, and the report says so rather than
letting a fitted number stand in for one. That distinction is the whole reason
this file separates ``transport`` from ``fabric``: a fit is always a fit *of a
named transport*, and substituting one for another is the error it exists to
prevent.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Message sizes spanning the latency-bound and bandwidth-bound regimes. The
# small end matters most: that is where alpha is identifiable, and where a
# gradient bucket for a small layer actually lands.
DEFAULT_SIZES = (
    4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024,
    1 << 20, 4 << 20, 16 << 20, 64 << 20,
)

# Ring-collective cost factors, the same ones the model uses.
RING_FACTOR = {"all_gather": 1.0, "reduce_scatter": 1.0, "all_reduce": 2.0,
               "copy": 1.0}

# Why a gloo run failed, keyed by collective, so the suite can report the actual
# cause rather than "unavailable" -- which tells a reader nothing about whether
# it is worth trying to fix.
_LAST_GLOO_ERROR: dict[str, str] = {}


@dataclass
class Point:
    """One measured transfer."""

    nbytes: int
    seconds: float
    world: int = 1
    kind: str = "copy"

    @property
    def effective_bytes(self) -> float:
        """Bytes actually crossing the wire, per the ring cost model."""
        if self.world <= 1:
            return float(self.nbytes)
        steps = (self.world - 1) / self.world
        return RING_FACTOR.get(self.kind, 1.0) * steps * self.nbytes

    @property
    def gbs(self) -> float:
        return self.effective_bytes / self.seconds / 1e9 if self.seconds > 0 else 0.0


@dataclass
class AlphaBetaFit:
    """A fitted latency-bandwidth model for one transport and collective."""

    transport: str
    kind: str
    world: int
    alpha_us: float = float("nan")
    beta_gbs: float = float("nan")
    r_squared: float = float("nan")
    n_points: int = 0
    residual_pct: float = float("nan")
    n_half_bytes: float = float("nan")
    points: list[dict[str, float]] = field(default_factory=list)
    note: str = ""

    # Below this, the straight line is not describing the data and the fitted
    # alpha and beta are two numbers with no physical content. Reporting them
    # anyway is the same failure as reporting a vacuous calibration statistic:
    # they look like parameters and get used as parameters.
    MIN_R2 = 0.90

    @property
    def numerically_ok(self) -> bool:
        return (
            math.isfinite(self.alpha_us) and math.isfinite(self.beta_gbs)
            and self.beta_gbs > 0 and self.n_points >= 3
        )

    @property
    def ok(self) -> bool:
        """Whether this fit describes its data well enough to be used."""
        return (
            self.numerically_ok
            and math.isfinite(self.r_squared)
            and self.r_squared >= self.MIN_R2
        )

    def predict_us(self, nbytes: float) -> float:
        return self.alpha_us + nbytes / (self.beta_gbs * 1e9) * 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport, "kind": self.kind, "world": self.world,
            "alpha_us": self.alpha_us, "beta_gbs": self.beta_gbs,
            "r_squared": self.r_squared, "n_points": self.n_points,
            "residual_pct": self.residual_pct,
            "n_half_bytes": self.n_half_bytes,
            "points": self.points, "note": self.note,
            "fit_accepted": self.ok,
            "min_r_squared": self.MIN_R2,
            "usable_for_fabric": False,
        }

    def describe(self) -> str:
        if not self.ok:
            return f"{self.transport}/{self.kind}: fit failed ({self.note})"
        return (
            f"{self.transport}/{self.kind} (world={self.world}): "
            f"alpha={self.alpha_us:.1f} us, beta={self.beta_gbs:.2f} GB/s, "
            f"R^2={self.r_squared:.3f}, median residual {self.residual_pct:.1f}%, "
            f"n_half={self.n_half_bytes / 1024:.0f} KiB"
        )


def fit_alpha_beta(points: Sequence[Point], *, transport: str, kind: str,
                   world: int = 1) -> AlphaBetaFit:
    """Least squares for ``t = alpha + bytes / beta``.

    Fitted on time against effective bytes, not on bandwidth against size. The
    difference matters: bandwidth is a ratio whose small-message points carry
    enormous relative error, and a fit weighted by them recovers a latency that
    is mostly noise. Time is linear in the parameters and the residuals are
    homoscedastic enough for ordinary least squares.
    """
    fit = AlphaBetaFit(transport=transport, kind=kind, world=world,
                       n_points=len(points))
    usable = [p for p in points if p.seconds > 0 and p.nbytes > 0]
    if len(usable) < 3:
        fit.note = f"only {len(usable)} usable point(s); need at least 3"
        return fit

    x = np.array([p.effective_bytes for p in usable], dtype=float)
    y = np.array([p.seconds for p in usable], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    if slope <= 0:
        fit.note = (
            "the fitted slope is not positive, so no bandwidth can be recovered: "
            "the measurements are dominated by overhead across the whole size range"
        )
        return fit

    fit.alpha_us = max(intercept, 0.0) * 1e6
    fit.beta_gbs = 1.0 / slope / 1e9
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    fit.residual_pct = float(np.median(np.abs(y - pred) / y) * 100)
    # The message size at which latency and bandwidth contribute equally. This
    # is the number a practitioner can act on: below it, fusing buckets helps;
    # above it, only bandwidth does.
    fit.n_half_bytes = fit.alpha_us * 1e-6 * fit.beta_gbs * 1e9
    fit.points = [
        {"nbytes": p.nbytes, "effective_bytes": p.effective_bytes,
         "seconds": p.seconds, "gbs": p.gbs}
        for p in usable
    ]
    if math.isfinite(fit.r_squared) and fit.r_squared < AlphaBetaFit.MIN_R2:
        fit.note = (
            f"a latency-plus-bandwidth line explains only {fit.r_squared:.0%} of the "
            f"variance here (need {AlphaBetaFit.MIN_R2:.0%}), so the fitted alpha and "
            "beta are not parameters of this transport -- the model form does not "
            "describe it. Most likely the transport changes algorithm across the size "
            "range, which a single line cannot represent."
        )
    return fit


# --------------------------------------------------------------------------
# transports
# --------------------------------------------------------------------------


def measure_pcie(
    *, sizes: Sequence[int] = DEFAULT_SIZES, direction: str = "h2d",
    replicates: int = 7, iters: int = 20, device: str = "cuda",
) -> list[Point]:
    """Host-to-device or device-to-host copies over PCIe.

    Pinned host memory, so the copy is a DMA and not a staged one. Timed with
    CUDA events around a synchronous copy: the asynchronous path would measure
    the launch and not the transfer.
    """
    import torch

    if not torch.cuda.is_available():
        return []
    out: list[Point] = []
    for n in sizes:
        host = torch.empty(n, dtype=torch.uint8, device="cpu", pin_memory=True)
        dev = torch.empty(n, dtype=torch.uint8, device=device)
        src, dst = (host, dev) if direction == "h2d" else (dev, host)

        def once(dst=dst, src=src) -> None:
            dst.copy_(src, non_blocking=False)

        for _ in range(5):
            once()
        torch.cuda.synchronize()
        best: list[float] = []
        for _ in range(replicates):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                once()
            end.record()
            torch.cuda.synchronize()
            best.append(start.elapsed_time(end) * 1e-3 / iters)
        out.append(Point(nbytes=n, seconds=float(np.median(best)), world=1,
                         kind="copy"))
        del host, dev
        torch.cuda.empty_cache()
    return out


def _gloo_worker(rank: int, world: int, sizes: list[int], kind: str, q: Any) -> None:
    """One rank of a gloo collective benchmark. Runs in its own process."""
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29677")
    # This PyTorch build has no libuv, and the TCP store now requests it by
    # default. Without this the rendezvous raises before a single byte moves.
    os.environ.setdefault("USE_LIBUV", "0")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
        return
    try:
        results = []
        for n in sizes:
            elems = max(n // 4, 1)
            buf = torch.ones(elems, dtype=torch.float32)
            out = [torch.empty_like(buf) for _ in range(world)]

            def once(buf=buf, out=out) -> None:
                if kind == "all_reduce":
                    dist.all_reduce(buf)
                else:
                    dist.all_gather(out, buf)

            for _ in range(3):
                once()
            dist.barrier()
            import time as _t

            iters = 20 if n <= (1 << 20) else 5
            t0 = _t.perf_counter()
            for _ in range(iters):
                once()
            dist.barrier()
            el = (_t.perf_counter() - t0) / iters
            results.append((elems * 4, el))
        if rank == 0:
            q.put(results)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        dist.destroy_process_group()


def measure_gloo(
    *, sizes: Sequence[int] = DEFAULT_SIZES[:6], kind: str = "all_reduce",
    world: int = 2, timeout_s: int = 180,
) -> list[Point]:
    """Real ring collectives between processes, over shared memory.

    Not a GPU fabric. What it is: a genuine collective, with a genuine ring
    algorithm and a genuine latency floor, whose alpha-beta structure can be
    fitted by exactly the procedure that will be pointed at nccl-tests output
    on the cluster. Calibrating the procedure is worth doing before there is
    hardware to point it at.
    """
    import multiprocessing as mp
    import sys

    try:
        import torch.distributed as dist

        if not dist.is_available():
            return []
    except Exception:
        return []

    # The spawn start method re-imports the parent's __main__ in each child. Run
    # from `python -c`, or from anything else without an importable __main__,
    # the children never get going and the parent waits forever on the queue.
    # Refusing up front turns a hang into a recorded "not measurable here".
    main = sys.modules.get("__main__")
    if mp.get_start_method(allow_none=True) != "fork" and not getattr(main, "__file__", None):
        return []

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    procs = [
        ctx.Process(target=_gloo_worker, args=(r, world, list(sizes), kind, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results: Any = []
    error: str | None = None
    try:
        got = q.get(timeout=timeout_s)
        if isinstance(got, dict) and "error" in got:
            error = got["error"]
        else:
            results = got
    except Exception as e:
        error = f"no result from the worker processes: {type(e).__name__}: {e}"
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
    if error:
        _LAST_GLOO_ERROR[kind] = error
    return [Point(nbytes=n, seconds=s, world=world, kind=kind) for n, s in results]


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------


@dataclass
class MicrobenchSuite:
    """Every transport this machine could measure, fitted."""

    fits: list[AlphaBetaFit] = field(default_factory=list)
    device: str = ""
    unavailable: dict[str, str] = field(default_factory=dict)

    def by_name(self, transport: str, kind: str) -> AlphaBetaFit | None:
        for f in self.fits:
            if f.transport == transport and f.kind == kind:
                return f
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "fits": [f.to_dict() for f in self.fits],
            "unavailable": self.unavailable,
            "usable_for_registered_fabric": False,
            "why": (
                "The registered protocol is about NVLink and InfiniBand on an "
                "eight-GPU node. Neither is present here, so none of these fits is a "
                "value for the fabric the protocol registers. What they calibrate is "
                "the functional form and the fitting procedure."
            ),
        }

    def to_markdown(self) -> str:
        lines = [
            "### Communication microbenchmarks",
            "",
            f"Measured on {self.device}.",
            "",
            "| Transport | Collective | alpha (us) | beta (GB/s) | R^2 | n_half |",
            "|---|---|---|---|---|---|",
        ]
        for f in self.fits:
            if not f.numerically_ok:
                lines.append(f"| `{f.transport}` | {f.kind} | — | — | — | {f.note} |")
                continue
            mark = "" if f.ok else " **rejected**"
            alpha = f"{f.alpha_us:.1f}" if f.ok else f"_{f.alpha_us:.1f}_"
            beta = f"{f.beta_gbs:.2f}" if f.ok else f"_{f.beta_gbs:.2f}_"
            lines.append(
                f"| `{f.transport}` | {f.kind}{mark} | {alpha} | {beta} | "
                f"{f.r_squared:.3f} | {f.n_half_bytes / 1024:.0f} KiB |"
            )
        rejected = [f for f in self.fits if f.numerically_ok and not f.ok]
        if rejected:
            lines += ["", "**Rejected fits** &mdash; the numbers are shown in italics "
                      "so they are not mistaken for parameters.", ""]
            lines += [f"- `{f.transport}`/{f.kind}: {f.note}" for f in rejected]
        lines += [
            "",
            "`n_half` is the message size at which the latency term and the bandwidth "
            "term contribute equally. Below it, fusing gradient buckets is what helps; "
            "above it, only bandwidth is.",
            "",
            "**None of these is a value for the registered fabric.** The protocol is "
            "about NVLink and InfiniBand on an eight-GPU node, and neither exists on "
            "this machine. What is calibrated here is the *functional form* -- that a "
            "latency floor plus a bandwidth slope describes these transports, and how "
            "well -- and the fitting procedure that will be pointed at `nccl-tests` "
            "output when there is a cluster to point it at.",
        ]
        if self.unavailable:
            lines += ["", "**Not measurable here**", ""]
            lines += [f"- `{k}`: {v}" for k, v in self.unavailable.items()]
        return "\n".join(lines)


def run_microbenchmarks(*, device: str = "cuda", quick: bool = False) -> MicrobenchSuite:
    """Measure and fit every transport available on this machine."""
    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description

    sizes = DEFAULT_SIZES[:5] if quick else DEFAULT_SIZES
    suite = MicrobenchSuite(device=device_description(device))

    with MeasurementLock("communication microbenchmarks"):
        for direction in ("h2d", "d2h"):
            pts = measure_pcie(sizes=sizes, direction=direction, device=device)
            if pts:
                suite.fits.append(
                    fit_alpha_beta(pts, transport=f"pcie_{direction}", kind="copy")
                )
            else:
                suite.unavailable[f"pcie_{direction}"] = "no CUDA device"

        for kind in ("all_reduce", "all_gather"):
            pts = measure_gloo(sizes=sizes[:6], kind=kind, world=2)
            if pts:
                suite.fits.append(
                    fit_alpha_beta(pts, transport="gloo_shm", kind=kind, world=2)
                )
            else:
                suite.unavailable[f"gloo_{kind}"] = _LAST_GLOO_ERROR.get(
                    kind,
                    "torch.distributed gloo backend unavailable, or the spawn start "
                    "method has no importable __main__ to re-import in the children",
                )

    suite.unavailable["nvlink"] = (
        "requires at least two GPUs on one node; this machine has one"
    )
    suite.unavailable["infiniband"] = (
        "requires a multi-node allocation"
    )
    return suite
