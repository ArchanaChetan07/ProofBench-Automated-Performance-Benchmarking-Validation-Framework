"""GPU timing with the hygiene that makes paired statistics valid.

Most benchmark harnesses get four things wrong, and each of them moves results
by more than the effects being reported:

**Wall-clock instead of CUDA events.** ``time.perf_counter`` around an async
launch measures the launch, not the kernel. Events are recorded on the stream
and measure the device.

**A warm L2.** Re-running one kernel on one buffer leaves the inputs resident,
which flatters small shapes and disappears the moment the kernel runs inside a
real model. The cache is flushed between every timed iteration.

**Blocked A/B.** Timing all of A then all of B confounds any drift -- clock
throttling, another tenant, a background process -- with the comparison. Every
replicate here alternates, so drift hits both arms equally and the pairing in
:mod:`losscolumn.core.stats` is legitimate.

**Mean of iterations.** Timing distributions on a GPU are right-skewed with a
long tail of interrupts. The median of per-iteration times is the statistic,
and the replicate-level spread is kept rather than collapsed.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False


@dataclass
class TimingResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    iters: int
    samples: list[float] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class L2Flusher:
    """Evicts L2 between timed iterations by touching a buffer larger than it."""

    def __init__(self, device: str = "cuda", mib: int | None = None) -> None:
        self.enabled = HAS_TORCH and torch.cuda.is_available() and str(device).startswith("cuda")
        if not self.enabled:
            self.buf = None
            return
        if mib is None:
            props = torch.cuda.get_device_properties(0)
            l2 = getattr(props, "L2_cache_size", 0) or 40 * 1024 * 1024
            mib = max(int(2 * l2 / (1024 * 1024)), 64)
        self.buf = torch.empty(int(mib) * 1024 * 1024 // 4, dtype=torch.int32, device=device)

    def flush(self) -> None:
        if self.enabled and self.buf is not None:
            self.buf.zero_()


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 8,
    iters: int = 25,
    flusher: L2Flusher | None = None,
    device: str = "cuda",
) -> TimingResult:
    """Median device time of ``fn`` in milliseconds."""
    if not HAS_TORCH:
        return TimingResult(float("nan"), float("nan"), float("nan"), 0, error="torch missing")
    import numpy as np

    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    try:
        for _ in range(warmup):
            fn()
        if use_cuda:
            torch.cuda.synchronize()
    except Exception as e:
        return TimingResult(float("nan"), float("nan"), float("nan"), 0, error=f"{type(e).__name__}: {e}")

    samples: list[float] = []
    try:
        if use_cuda:
            start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                if flusher:
                    flusher.flush()
                start[i].record()
                fn()
                end[i].record()
            torch.cuda.synchronize()
            samples = [float(s.elapsed_time(e)) for s, e in zip(start, end, strict=False)]
        else:
            import time

            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1e3)
    except Exception as e:
        return TimingResult(float("nan"), float("nan"), float("nan"), 0, error=f"{type(e).__name__}: {e}")

    a = np.asarray(samples, dtype=float)
    return TimingResult(
        median_ms=float(np.median(a)),
        p10_ms=float(np.percentile(a, 10)),
        p90_ms=float(np.percentile(a, 90)),
        iters=iters,
        samples=samples,
    )


def paired_ab(
    fn_a: Callable[[], Any],
    fn_b: Callable[[], Any],
    *,
    replicates: int = 11,
    warmup: int = 8,
    iters: int = 15,
    device: str = "cuda",
) -> tuple[list[float], list[float]]:
    """Interleaved A/B timing. Returns ``(a_times_ms, b_times_ms)``, paired by index.

    The alternation is the whole point: replicate ``i`` of A and replicate ``i``
    of B are separated by milliseconds, not minutes, so anything that drifts on
    a slower timescale cancels in the paired difference.
    """
    flusher = L2Flusher(device=device)
    a_out: list[float] = []
    b_out: list[float] = []
    for _ in range(replicates):
        ra = time_callable(fn_a, warmup=warmup, iters=iters, flusher=flusher, device=device)
        rb = time_callable(fn_b, warmup=warmup, iters=iters, flusher=flusher, device=device)
        a_out.append(ra.median_ms)
        b_out.append(rb.median_ms)
        warmup = 1  # only the first replicate needs a full warm-up
    return a_out, b_out


def free_memory() -> None:
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def device_description(device: str = "cuda") -> str:
    if not (HAS_TORCH and torch.cuda.is_available()):
        return "cpu"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} (sm_{p.major}{p.minor}, {p.total_memory / 1e9:.0f} GB)"


def clock_stability(device: str = "cuda", samples: int = 5) -> dict[str, Any]:
    """Report SM clock spread during the run.

    A benchmark taken while the card is throttling is not comparable to one
    taken cold, and the difference routinely exceeds the effects people
    publish. The number is recorded so a reader can discount accordingly; the
    harness does not pretend to control it.
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        row = out.stdout.strip().splitlines()[0].split(",")
        return {
            "available": True,
            "sm_clock_mhz": float(row[0]),
            "temperature_c": float(row[1]),
            "power_w": float(row[2]) if row[2].strip() not in ("", "N/A") else None,
        }
    except Exception:
        return {"available": False}


# --------------------------------------------------------------------------
# measurement exclusivity
# --------------------------------------------------------------------------

_LOCK_ENV = "LC_MEASUREMENT_LOCK"


def _lock_path() -> Path:
    import os
    import tempfile

    override = os.environ.get(_LOCK_ENV)
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "losscolumn-measurement.lock"


def _pid_alive(pid: int) -> bool:
    import os

    if pid == os.getpid():
        return True
    try:
        if os.name == "nt":
            import subprocess

            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def competing_gpu_memory() -> list[dict[str, Any]]:
    """Other processes holding *substantial* GPU memory, where that is knowable.

    Deliberately narrow. On Windows under WDDM ``nvidia-smi
    --query-compute-apps`` lists every graphics context on the machine -- the
    desktop compositor, every browser tab -- and reports ``N/A`` for their
    memory, so treating that list as "competing compute jobs" produces dozens
    of false positives and the guard gets switched off, which is worse than not
    having one. Only processes with a reported allocation above the threshold
    are returned; where memory is unknowable this reports nothing and the lock
    below does the real work.
    """
    import os
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    me = os.getpid()
    procs: list[dict[str, Any]] = []
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == me:
            continue
        try:
            mb = float(parts[1])
        except ValueError:
            continue  # [N/A] under WDDM: unknowable, not zero
        if mb >= 512:
            procs.append({"pid": int(parts[0]), "used_memory_mb": mb})
    return procs


class MeasurementLock:
    """An exclusive lock held for the duration of a timing run.

    Two benchmark processes sharing one device do not each get half the
    machine in a way that averages out. They interleave at the scheduler,
    contend for L2 and memory bandwidth, and hold each other's clocks down, so
    every timing either produces is wrong by an amount that depends on what the
    other one happened to be doing. Nothing downstream can detect it: the
    replicates stay self-consistent and the intervals stay tight.

    This exists because it happened here. A calibration run was launched while
    a sweep was still going; both had to be discarded and re-run.

    A lock file rather than a GPU query, because the hazard is specifically
    *another measurement*, and that is knowable exactly, on every platform,
    without depending on what nvidia-smi can see.
    """

    def __init__(self, purpose: str = "measurement", *, strict: bool = True) -> None:
        self.purpose = purpose
        self.strict = strict
        self.path = _lock_path()
        self.acquired = False
        self.state: dict[str, Any] = {}

    def __enter__(self) -> MeasurementLock:
        import json
        import os

        holder = None
        if self.path.exists():
            try:
                holder = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                holder = None
            if holder and _pid_alive(int(holder.get("pid", -1))) \
                    and int(holder.get("pid", -1)) != os.getpid():
                msg = (
                    f"another losscolumn measurement is running: pid {holder.get('pid')} "
                    f"({holder.get('purpose')}, started {holder.get('started')}). "
                    "Timings taken now would be contended and not comparable to timings "
                    "taken alone. Wait for it, or set strict=False to record the "
                    "condition and measure anyway."
                )
                if self.strict:
                    raise RuntimeError(msg)
                self.state["contended_by"] = holder
            else:
                # A stale lock from a killed run is not a reason to refuse.
                self.path.unlink(missing_ok=True)

        self.path.write_text(
            json.dumps({
                "pid": os.getpid(),
                "purpose": self.purpose,
                "started": _utcnow(),
            }),
            encoding="utf-8",
        )
        self.acquired = True
        self.state.update({
            "exclusive": "contended_by" not in self.state,
            "lock": str(self.path),
            "competing_gpu_memory": competing_gpu_memory(),
        })
        return self

    def __exit__(self, *exc: Any) -> None:
        import json
        import os

        if not self.acquired:
            return
        try:
            held = json.loads(self.path.read_text(encoding="utf-8"))
            if int(held.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
        except Exception:
            pass


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
