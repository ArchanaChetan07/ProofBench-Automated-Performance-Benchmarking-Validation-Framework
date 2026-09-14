"""Communication campaign over a real GPU fabric.

Written for a rented box, and deliberately not a copy of the local campaign with
the backend swapped. Two things about a multi-GPU host make the local design
wrong here, and both were found by running stage 2 before anything else:

**The node is not one fabric.** On the machine this was written against, GPUs
0-3 and 4-7 sit on different NUMA domains, `nvidia-smi topo -m` reporting SYS
between the groups and NODE within them. A world-8 collective therefore crosses
a different interconnect from a world-4 one confined to a quad, and fitting a
single cost model across both is exactly the error the local campaign's
inter_node flag exists to prevent.

**The links are not identical.** Two of the eight GPUs negotiated PCIe x8 where
the rest have x16. Which ranks a collective lands on changes what it measures,
so the device set is chosen explicitly and recorded rather than defaulting to
range(world).

So the campaign runs as several declared fabrics, each internally homogeneous,
each named in the artifact. A number that does not say which fabric produced it
cannot be compared with one that does.
"""
import json
import sys
import time
from pathlib import Path

REPEATS = 21


def topology() -> dict:
    """What the machine actually is, read from it rather than assumed."""
    import subprocess

    import torch

    out = {"n_gpus": torch.cuda.device_count(), "gpus": [], "p2p": {}}
    for i in range(out["n_gpus"]):
        p = torch.cuda.get_device_properties(i)
        q = subprocess.run(
            ["nvidia-smi", "-i", str(i), "--query-gpu=pcie.link.gen.current,"
             "pcie.link.width.current", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False)
        gen, width = (q.stdout.strip().split(", ") + ["", ""])[:2]
        out["gpus"].append({
            "index": i, "name": p.name,
            "memory_gb": round(p.total_memory / 1e9, 1),
            "pcie_gen": gen, "pcie_width": width,
        })
    n = out["n_gpus"]
    out["p2p"] = {
        f"{i}-{j}": bool(torch.cuda.can_device_access_peer(i, j))
        for i in range(n) for j in range(n) if i != j
    }
    try:
        t = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                           text=True, check=False)
        out["topo_matrix"] = t.stdout
    except OSError:
        out["topo_matrix"] = ""
    out["links"] = _parse_topo(out["topo_matrix"], out["n_gpus"])
    out["domains"] = _domains(out["links"], out["n_gpus"])
    return out


def _parse_topo(matrix: str, n: int) -> dict:
    """Pairwise link class from `nvidia-smi topo -m`.

    The labels are ordered: NV# is an NVLink of that many lanes, PIX/PXB/PHB a
    PCIe path of decreasing directness, NODE within one NUMA node, SYS across
    the interconnect between them. What matters here is only that SYS is a
    different fabric from everything else, and NVLink from PCIe.
    """
    links: dict[str, str] = {}
    for line in matrix.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("GPU"):
            continue
        try:
            i = int(parts[0][3:])
        except ValueError:
            continue
        for j, tok in enumerate(parts[1:1 + n]):
            if i != j and tok not in ("X", ""):
                links[f"{i}-{j}"] = tok
    return links


def _class_of(tok: str) -> str:
    if tok.startswith("NV"):
        return "nvlink"
    if tok == "SYS":
        return "cross_numa"
    return "pcie"


def _domains(links: dict, n: int) -> list[list[int]]:
    """Connected components under 'not cross-NUMA'.

    A set of devices is one fabric only if every pair inside it avoids the
    slowest class present. Union-find over the non-SYS pairs gives exactly that,
    and reports the whole node as one domain when nothing is SYS -- which is
    what an NVSwitch box should look like.
    """
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for key, tok in links.items():
        i, j = (int(x) for x in key.split("-"))
        if _class_of(tok) != "cross_numa":
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in groups.values()), key=lambda g: -len(g))


def fabrics(topo: dict) -> list[dict]:
    """Group the devices into sets that are plausibly one fabric.

    Homogeneous PCIe width first, because a x8 member changes what every
    collective touching it measures. Beyond that the split is declared from the
    topology matrix rather than inferred, and a set that mixes NUMA domains is
    labelled as such instead of being quietly averaged with one that does not.
    """
    gpus = topo["gpus"]
    width_of = {g["index"]: g["pcie_width"] for g in gpus}
    all_ids = [g["index"] for g in gpus]
    domains = topo.get("domains") or [all_ids]

    # Three sets, because on a split node no single one answers everything and
    # pretending otherwise is what produced a "homogeneous" fabric whose world-4
    # collective crossed the NUMA boundary at 1.9 GB/s while a set admitted as
    # mixed stayed inside a quad at 4.5.
    out = []

    # Strictest: one connectivity domain AND one link width. The cleanest
    # parameters, and on a split node it may be too small for the larger worlds.
    best: list[int] = []
    for dom in domains:
        widths: dict[str, list[int]] = {}
        for i in dom:
            widths.setdefault(width_of[i], []).append(i)
        cand = max(widths.values(), key=len) if widths else []
        if len(cand) > len(best):
            best = cand
    if len(best) >= 2:
        out.append({
            "name": "uniform",
            "device_ids": sorted(best),
            "note": (f"{len(best)} devices sharing one connectivity domain and "
                     f"one PCIe width ({width_of[best[0]]}x)"),
        })

    # One domain, whatever the widths. Usually the largest set that is still a
    # single fabric, which is what the larger world sizes need.
    dom = max(domains, key=len)
    if len(dom) > len(best) and len(dom) >= 2:
        mixed = sorted({width_of[i] for i in dom})
        out.append({
            "name": "one_domain",
            "device_ids": sorted(dom),
            "note": (f"the largest single connectivity domain ({len(dom)} "
                     "devices)" + (f", spanning PCIe widths "
                                   f"{', '.join(w + 'x' for w in mixed)}"
                                   if len(mixed) > 1 else "")),
        })

    # Everything, which on a split node is deliberately more than one fabric.
    if len(all_ids) > len(dom):
        out.append({
            "name": "whole_node",
            "device_ids": all_ids,
            "note": (f"all {len(all_ids)} devices across {len(domains)} "
                     "connectivity domains; a collective here crosses the "
                     "interconnect between them and is a different fabric from "
                     "the sets above, reported separately rather than pooled"),
        })

    return out or [{"name": "all", "device_ids": all_ids,
                    "note": "every device"}]


def main() -> int:

    from losscolumn.core.provenance import content_hash, utcnow
    from losscolumn.thrusts.kernel.bench import MeasurementLock
    from losscolumn.thrusts.overlap.campaign import (
        REQUIRED_COLLECTIVES,
        Campaign,
        analyse,
        build_grid,
        sweep_pass,
    )
    from losscolumn.version import STANDARD_VERSION

    quick = "--quick" in sys.argv
    art = Path("artifacts")
    art.mkdir(exist_ok=True)

    topo = topology()
    fabs = fabrics(topo)
    print(f"{topo['n_gpus']} GPUs: {topo['gpus'][0]['name']}")
    for f in fabs:
        print(f"  fabric {f['name']}: devices {f['device_ids']} -- {f['note']}")

    grid = build_grid(backbone_per_decade=3 if quick else 6,
                      oversample_per_region=2 if quick else 5)
    repeats = 3 if quick else REPEATS

    seal = {
        "name": "gpu-communication-campaign-v1",
        "question": "What are this fabric's alpha-beta parameters, and is the "
                    "surface modellable at the registered gate?",
        "topology": topo,
        "fabrics": fabs,
        "grid_sizes": len(grid),
        "repeats_per_point_per_pass": repeats,
        "passes": 2,
        "backend": "nccl",
        "declared_deviations": [
            "world sizes are re-derived from the measured topology, as stage 2 "
            "of the sealed protocol permits and requires to be recorded",
            "each fabric is fitted separately; no model spans two of them",
        ],
        "committed_before_measurement": True,
    }
    seal["seal_hash"] = content_hash(seal)
    (art / "prereg").mkdir(exist_ok=True)
    (art / "prereg" / "gpu-campaign-v1.protocol.json").write_text(
        json.dumps(seal, indent=2, default=str), encoding="utf-8")
    print(f"sealed {seal['seal_hash'][:23]} before measurement", flush=True)

    results = {}
    t0 = time.time()
    with MeasurementLock("gpu communication campaign"):
        for fab in fabs:
            ids = fab["device_ids"]
            worlds = [w for w in (2, 3, 4, 8) if w <= len(ids)]
            if quick:
                worlds = worlds[:2]
            print(f"\n=== fabric {fab['name']}: worlds {worlds} ===", flush=True)
            camp = Campaign(grid=grid, n_passes=2, repeats=repeats,
                            device=topo["gpus"][0]["name"],
                            backend="nccl", fabric=fab["note"])
            for p in range(2):
                recs, errs = sweep_pass(
                    sizes=grid, worlds=worlds, collectives=REQUIRED_COLLECTIVES,
                    repeats=repeats, pass_index=p, backend="nccl",
                    device_ids=ids)
                camp.records.extend(recs)
                camp.errors.update(errs)
                print(f"  pass {p}: {len(recs)} records, {len(errs)} failure(s), "
                      f"{time.time() - t0:.0f}s", flush=True)
            if camp.records:
                analyse(camp)
            results[fab["name"]] = {"fabric": fab, "campaign": camp.to_dict()}

            # Checkpoint after every fabric. The first version of this wrote
            # once, at the end, which is a poor bargain on a machine billed by
            # the hour: a run stopped two thirds through left nothing on disk
            # and the completed fabrics existed only in memory. A partial
            # artifact that names what it contains is worth more than a
            # complete one that was never written.
            (art / "gpu-campaign.json").write_text(json.dumps({
                "kind": "gpu-campaign-report", "standard_version": STANDARD_VERSION,
                "generated_at": utcnow(), "protocol": seal, "topology": topo,
                "results": results, "elapsed_s": time.time() - t0,
                "n_evidence_sessions": 1, "session_id": "gpu-campaign-v1",
                "complete": len(results) == len(fabs),
                "fabrics_measured": sorted(results),
                "fabrics_planned": [f["name"] for f in fabs],
            }, indent=2, default=str), encoding="utf-8")
            print(f"  checkpointed after {fab['name']} "
                  f"({len(results)} of {len(fabs)} fabrics)", flush=True)
            cov = camp.coverage.to_dict() if camp.coverage else {}
            if cov:
                print(f"  covered {cov['model_coverage']['n_covered_cells']} of "
                      f"{cov['n_required_cells']}", flush=True)

    payload = {
        "kind": "gpu-campaign-report", "standard_version": STANDARD_VERSION,
        "generated_at": utcnow(), "protocol": seal, "topology": topo,
        "results": results, "elapsed_s": time.time() - t0,
        "n_evidence_sessions": 1, "session_id": "gpu-campaign-v1",
    }
    (art / "gpu-campaign.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote artifacts/gpu-campaign.json ({time.time() - t0:.0f}s)")
    for name, r in results.items():
        cov = (r["campaign"].get("coverage") or {})
        mc = (cov.get("model_coverage") or {})
        bw = [x["bandwidth_gbs"] for x in r["campaign"]["records"]
              if x.get("valid") and x.get("bandwidth_gbs") == x.get("bandwidth_gbs")]
        print(f"  {name}: covered {mc.get('n_covered_cells', 0)} of "
              f"{cov.get('n_required_cells', 0)}, peak "
              f"{max(bw) if bw else float('nan'):.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
