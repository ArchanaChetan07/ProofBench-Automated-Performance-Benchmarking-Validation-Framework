"""Does sizing the timed block by duration actually reduce the noise?

The fixed rule averaged 20 calls below 1 MiB and 5 above it, and medium points
above that boundary carried 29.4% run-to-run variation against 18.8% below.
This measures the same sizes both ways, back to back in one process, so the
comparison is not confounded by anything the machine does between launches.
"""
import json
import sys
import time
from pathlib import Path

SIZES = [1 << 18, 393216, 1 << 19, 786432, 1 << 20, 1572864, 1 << 21, 3145728,
         1 << 22, 1 << 23]
REPEATS = 15


def _worker(rank, world, q):
    import os

    import numpy as np
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29931")
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    rows = []
    try:
        for nbytes in SIZES:
            elems = max(nbytes // 4, 1)
            buf = torch.ones(elems, dtype=torch.float32)
            out = [torch.empty_like(buf) for _ in range(world)]

            def once(out=out, buf=buf):
                dist.all_gather(out, buf)

            for _ in range(3):
                once()
            dist.barrier()

            # Old rule.
            fixed = 20 if nbytes <= (1 << 20) else 5
            t_fixed = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                for _ in range(fixed):
                    once()
                dist.barrier()
                t_fixed.append((time.perf_counter() - t0) / fixed)

            # New rule: block budget scales with the point's own variability.
            probe = []
            for _ in range(6):
                tp = time.perf_counter()
                once()
                probe.append(time.perf_counter() - tp)
            pa = np.array(probe, dtype=float)
            per_call = max(float(np.median(pa)), 1e-9)
            probe_cv = float(pa.std() / pa.mean()) if pa.mean() > 0 else 0.0
            frac = min(probe_cv / 0.30, 1.0)
            budget = 0.002 + (0.030 - 0.002) * frac
            adaptive = int(min(max(round(budget / per_call), 5), 200))
            t_adapt = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                for _ in range(adaptive):
                    once()
                dist.barrier()
                t_adapt.append((time.perf_counter() - t0) / adaptive)

            if rank == 0:
                f = np.array(t_fixed)
                a = np.array(t_adapt)
                rows.append({
                    "nbytes": elems * 4,
                    "fixed_iters": fixed, "adaptive_iters": adaptive,
                    "cv_fixed": float(f.std() / f.mean()),
                    "cv_adaptive": float(a.std() / a.mean()),
                    "median_fixed": float(np.median(f)),
                    "median_adaptive": float(np.median(a)),
                })
        if rank == 0:
            q.put(rows)
    finally:
        dist.destroy_process_group()


def main() -> int:
    import multiprocessing as mp

    import numpy as np

    from losscolumn.thrusts.kernel.bench import MeasurementLock

    world = 3
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with MeasurementLock("iters calibration"):
        ps = [ctx.Process(target=_worker, args=(r, world, q)) for r in range(world)]
        for p in ps:
            p.start()
        rows = q.get(timeout=1800)
        for p in ps:
            p.join(timeout=30)

    print(f"{'bytes':>10s} {'iters':>12s} {'cv fixed':>9s} {'cv adaptive':>12s} "
          f"{'change':>8s}  {'median agree':>12s}")
    for r in rows:
        drift = r["median_adaptive"] / r["median_fixed"]
        print(f"{r['nbytes']:10d} {r['fixed_iters']:5d}->{r['adaptive_iters']:<5d} "
              f"{r['cv_fixed']:8.1%} {r['cv_adaptive']:11.1%} "
              f"{r['cv_adaptive'] / r['cv_fixed']:7.2f}x  {drift:11.3f}x")
    f = np.median([r["cv_fixed"] for r in rows])
    a = np.median([r["cv_adaptive"] for r in rows])
    print(f"\nmedian CV: fixed {f:.1%} -> adaptive {a:.1%}  ({a / f:.2f}x)")
    big = [r for r in rows if r["nbytes"] > (1 << 20)]
    if big:
        bf = np.median([r["cv_fixed"] for r in big])
        ba = np.median([r["cv_adaptive"] for r in big])
        print(f"above 1 MiB (where the old rule dropped to 5 iters): "
              f"{bf:.1%} -> {ba:.1%}  ({ba / bf:.2f}x)")
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/iters-calibration.json").write_text(
        json.dumps({"repeats": REPEATS, "world": world, "rows": rows}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
