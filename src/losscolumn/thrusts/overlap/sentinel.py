"""The stability sentinel: a small fixed probe, run over and over.

The campaign measures 46 sizes to learn a surface. The sentinel measures six
points to learn whether the machine is the same machine it was an hour ago.
Those are different jobs and want different instruments -- a stability check
that takes as long as the experiment cannot be run often enough to be useful.

Six probes: two collectives crossed with small, medium and large, at one world
size, with the medium regime deliberately included because that is where the
world-4 contradiction appeared.

The three levels are produced by *how* the sentinel is run rather than by what
it measures, which is what makes them nested and therefore comparable:

* repeats inside one process, one initialised process group -> within run
* several process launches back to back                     -> across restart
* groups of launches separated in time                      -> across session

Every reading carries the machine state that produced it, so that a drift can
be tested against something rather than only described.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from typing import Any

from losscolumn.core.stability import SentinelReading, SessionState

__all__ = ["SENTINEL_SIZES", "SENTINEL_COLLECTIVES", "SENTINEL_WORLD",
           "SENTINEL_WORLDS", "capture_state", "run_sentinel",
           "sentinel_protocol", "certified_worlds", "uncertified_worlds"]

# One size per regime, at the geometric centre of each so that a nearby
# breakpoint cannot dominate the reading.
SENTINEL_SIZES: tuple[tuple[str, int], ...] = (
    ("small", 1 << 16),    # 64 KiB
    ("medium", 1 << 20),   # 1 MiB
    ("large", 1 << 23),    # 8 MiB
)
SENTINEL_COLLECTIVES: tuple[str, ...] = ("all_gather", "all_reduce")
SENTINEL_WORLD: int = 3
"""Default when only one world is asked for. Not a certification scope."""

SENTINEL_WORLDS: tuple[int, ...] = (2, 3, 4)
"""World sizes a stability run certifies unless told otherwise.

A sentinel that measures one world certifies one world. Contention, memory
pressure and -- on a split node -- which interconnect the ranks land across all
change with the rank count, so a verdict from three ranks says nothing about
eight. Drivers should pass the worlds the campaign will actually use.
"""


def capture_state(session_id: str = "", restart_index: int = 0) -> SessionState:
    """Snapshot the machine. Missing fields stay NaN rather than zero.

    A zero here would be indistinguishable from a real measurement of zero and
    would pull every correlation toward it, so absence is recorded as absence.
    """
    from losscolumn.core.provenance import utcnow

    st = SessionState(captured_at=utcnow(), session_id=session_id,
                      restart_index=restart_index, process_id=os.getpid())
    try:
        import psutil

        st.n_cpu = psutil.cpu_count() or 0
        st.cpu_percent = float(psutil.cpu_percent(interval=0.15))
        st.free_ram_gb = float(psutil.virtual_memory().available) / 1e9
        try:
            st.load_avg_1m = float(psutil.getloadavg()[0])
        except (AttributeError, OSError):
            pass
    except ImportError:
        pass

    fields = ("name,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,"
              "memory.used,power.draw")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False)
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]

            def num(i: int) -> float:
                try:
                    return float(parts[i])
                except (ValueError, IndexError):
                    return float("nan")

            st.gpu_name = parts[0] if parts else ""
            st.gpu_clock_mhz = num(1)
            st.gpu_mem_clock_mhz = num(2)
            st.gpu_temperature_c = num(3)
            st.gpu_utilisation_pct = num(4)
            st.gpu_memory_used_mb = num(5)
            st.gpu_power_w = num(6)
    except (OSError, subprocess.SubprocessError):
        pass
    return st


def _worker(rank: int, world: int, probes: list[tuple[str, str, int]],
            repeats: int, iters_small: int, q: Any) -> None:
    """One rank of one sentinel launch."""
    import time

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("LC_SENTINEL_PORT", "29811"))
    os.environ.setdefault("USE_LIBUV", "0")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
        return
    try:
        rows: list[dict[str, Any]] = []
        for size_class, kind, nbytes in probes:
            elems = max(nbytes // 4, 1)
            timings: list[float] = []
            reason = ""
            try:
                buf = torch.ones(elems, dtype=torch.float32)
                out = [torch.empty_like(buf) for _ in range(world)]

                def once(buf=buf, out=out, kind=kind) -> None:
                    if kind == "all_reduce":
                        dist.all_reduce(buf)
                    else:
                        dist.all_gather(out, buf)

                for _ in range(3):
                    once()
                dist.barrier()
                iters = iters_small if nbytes <= (1 << 20) else max(iters_small // 4, 3)
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        once()
                    dist.barrier()
                    timings.append((time.perf_counter() - t0) / iters)
            except Exception as e:
                reason = f"{type(e).__name__}: {str(e)[:80]}"
            if rank == 0:
                rows.append({
                    "size_class": size_class, "collective": kind,
                    "nbytes": elems * 4, "timings_s": timings,
                    "invalid_reason": reason,
                })
        if rank == 0:
            q.put(rows)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        dist.destroy_process_group()


def run_sentinel(*, session_id: str, restart_index: int,
                 world: int = SENTINEL_WORLD, repeats: int = 7,
                 iters_small: int = 20, port_offset: int = 0,
                 timeout_s: int = 900,
                 sizes: Sequence[tuple[str, int]] = SENTINEL_SIZES,
                 collectives: Sequence[str] = SENTINEL_COLLECTIVES,
                 ) -> tuple[list[SentinelReading], dict[str, str]]:
    """One sentinel launch: a fresh process group, every probe, `repeats` each.

    State is captured immediately before and after the launch and both are kept,
    because a state read only at the start cannot describe a machine that
    changed during the measurement.
    """
    import multiprocessing as mp
    import sys

    errors: dict[str, str] = {}
    main = sys.modules.get("__main__")
    if mp.get_start_method(allow_none=True) != "fork" and not getattr(main, "__file__", None):
        return [], {"spawn": "no importable __main__ for the spawn start method"}

    probes = [(sc, kind, n) for kind in collectives for sc, n in sizes]
    before = capture_state(session_id, restart_index)

    ctx = mp.get_context("spawn")
    os.environ["LC_SENTINEL_PORT"] = str(29811 + port_offset % 400)
    q: Any = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(r, world, probes, repeats, iters_small, q))
             for r in range(world)]
    for p in procs:
        p.start()
    try:
        got = q.get(timeout=timeout_s)
    except Exception as e:
        got = {"error": f"no result: {type(e).__name__}: {e}"}
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    after = capture_state(session_id, restart_index)
    if isinstance(got, dict):
        return [], {f"{session_id}/r{restart_index}": str(got.get("error", "unknown"))}

    # The state attached to a reading is the mean of before and after, so a
    # machine that heated up during the launch is described as having done so
    # rather than as having been cool throughout.
    state = _mean_state(before, after)
    out: list[SentinelReading] = []
    for row in got:
        r = SentinelReading(
            session_id=session_id, restart_index=restart_index,
            collective=row["collective"], world=world, nbytes=row["nbytes"],
            size_class=row["size_class"], timings_s=list(row["timings_s"]),
            state=state,
        )
        if not r.timings_s:
            r.valid = False
            r.invalid_reason = row.get("invalid_reason") or "no timing completed"
        elif not (r.median_s > 0):
            r.valid = False
            r.invalid_reason = "median timing is not positive"
        out.append(r)
    return out, errors


def _mean_state(a: SessionState, b: SessionState) -> SessionState:
    import math

    st = SessionState(**{**a.__dict__})
    for k, va in a.__dict__.items():
        vb = getattr(b, k)
        if isinstance(va, float) and isinstance(vb, float):
            if math.isfinite(va) and math.isfinite(vb):
                setattr(st, k, (va + vb) / 2.0)
            elif math.isfinite(vb):
                setattr(st, k, vb)
    return st


def certified_worlds(readings) -> set:
    """World sizes a set of sentinel readings actually covers."""
    return {r.world for r in readings if getattr(r, "valid", True)}


def uncertified_worlds(readings, needed) -> list:
    """Worlds a campaign will use that the sentinel never measured.

    Returned rather than raised because the honest response depends on the
    caller: a campaign may legitimately proceed and record the gap, but it may
    not claim the stability verdict covers it.
    """
    have = certified_worlds(readings)
    return sorted(w for w in needed if w not in have)


def sentinel_protocol(*, n_sessions: int, n_restarts: int, repeats: int,
                      gap_s: float, world: int = SENTINEL_WORLD,
                      worlds: Sequence[int] = (),
                      criterion_source: str = "derived from across-restart CV",
                      ) -> dict[str, Any]:
    """The pre-registered design, sealed before any of it runs.

    The comparability criterion is registered as a *rule for deriving* the
    threshold rather than as a number, because the number has to come from this
    machine's own repeatability. Inventing it in advance would be arbitrary;
    choosing it afterwards would let it be set to admit whichever sessions one
    happened to want to pool.
    """
    from losscolumn.core.provenance import content_hash

    body = {
        "name": "machine-stability-sentinel-v1",
        "question": (
            "Is this machine's communication timing stable enough that two "
            "measurement sessions can be compared at all?"
        ),
        "probes": [{"size_class": sc, "nbytes": n, "collective": k}
                   for k in SENTINEL_COLLECTIVES for sc, n in SENTINEL_SIZES],
        "world": world,
        "worlds_certified": sorted(worlds) if worlds else [world],
        "scope_rule": (
            "A stability verdict covers the world sizes measured here and no "
            "others. Contention and, on a split node, which interconnect the "
            "ranks span both change with rank count, so a campaign running at "
            "an uncertified world must record that gap rather than inherit "
            "this verdict."
        ),
        "levels": {
            "within_run": f"{repeats} repeats inside one process group",
            "across_restart": f"{n_restarts} process launches per session",
            "across_session": f"{n_sessions} sessions separated by >= {gap_s:.0f}s",
        },
        "comparability_criterion": {
            "form": "max(1 + 3 * across_restart_CV, 1.05)",
            "source": criterion_source,
            "applied_to": "the worst shared probe, not the pooled median",
            "rationale": (
                "A threshold tighter than the machine's own restart-level "
                "repeatability would reject honest pairs; one looser than three "
                "of its standard deviations would admit real state changes."
            ),
        },
        "decision_rules": {
            "RANDOM": "across_session CV < 1.5x across_restart CV, with no probe-"
                      "specific or state-correlated structure",
            "PERSISTENT": "across_session CV >= 1.5x across_restart CV, uniformly",
            "WORKLOAD_SPECIFIC": "session-level CV differs >= 2x between probes",
            "ENVIRONMENT_DEPENDENT": "|r| >= 0.7 between bandwidth and a captured "
                                     "state variable",
        },
        "committed_before_measurement": True,
        "pooling_rule": (
            "Sessions judged INCOMPARABLE are never pooled. A fit that spans "
            "them is invalid regardless of how good it looks."
        ),
    }
    body["seal_hash"] = content_hash(body)
    return body
