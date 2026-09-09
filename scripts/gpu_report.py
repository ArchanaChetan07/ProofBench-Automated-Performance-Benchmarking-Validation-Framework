"""Read the GPU campaign, and ask what the rental was for.

Three questions, in order of what they cost to answer wrongly.

**Does the surface have algorithm-selection thresholds?** The local gloo
campaign found them in world 2 and world 3 and none in world 4, and could not
say whether that was a property of collectives or of one contended desktop. A
different transport on different hardware answers it.

**Does the methodology transfer?** Every gate, detector and classifier was
written against one machine. Running them unchanged on another is the only way
to find the ones that encoded that machine rather than the problem.

**What are this fabric's parameters?** Last, and deliberately so. A number from
a surface whose structure is not understood is a number with no error bar.
"""
import json
import sys
from pathlib import Path


def main() -> int:
    import numpy as np

    from losscolumn.core.bimodal import find_threshold_band
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.campaign import PointRecord
    from losscolumn.version import STANDARD_VERSION

    art = Path("artifacts")
    src = art / "gpu-campaign.json"
    if not src.exists():
        print("no artifacts/gpu-campaign.json", file=sys.stderr)
        return 1
    d = json.loads(src.read_text(encoding="utf-8"))

    L = ["# The fabric, measured", "",
         f"Protocol `{d['protocol']['name']}` sealed "
         f"`{d['protocol']['seal_hash'][:23]}` before measurement. "
         f"{d['elapsed_s'] / 60:.0f} minutes, one session, "
         f"{d['protocol']['repeats_per_point_per_pass']} repeats per point per "
         "pass over nccl.", ""]

    topo = d["topology"]
    g0 = topo["gpus"][0]
    widths = sorted({g["pcie_width"] for g in topo["gpus"]})
    n_p2p = sum(1 for v in topo["p2p"].values() if v)
    L += ["## What the machine turned out to be", "",
          f"{topo['n_gpus']}x {g0['name']}, {g0['memory_gb']:.0f} GB each.",
          "",
          f"- PCIe generation **{g0['pcie_gen']}**, not the 5 advertised",
          f"- link widths present: {', '.join(w + 'x' for w in widths)}"
          + (" -- not one fabric" if len(widths) > 1 else ""),
          f"- peer access enabled on {n_p2p} of "
          f"{topo['n_gpus'] * (topo['n_gpus'] - 1)} pairs, so nccl moves data "
          "directly rather than staging it through the host",
          "",
          "Stage 2 is the reason any of that is known before the numbers rather "
          "than after them.", ""]

    summary = {}
    for name, r in d["results"].items():
        camp = r["campaign"]
        recs = [PointRecord.from_dict(x) for x in camp["records"]]
        valid = [x for x in recs if x.valid]
        cov = camp.get("coverage") or {}
        mc = cov.get("model_coverage") or {}

        L += [f"## Fabric `{name}`", "",
              f"Devices {r['fabric']['device_ids']} -- {r['fabric']['note']}.", ""]

        groups = sorted({f"{x.collective}/world{x.world}" for x in valid})
        L += ["| group | peak GB/s | median CV | thresholds | uncovered |",
              "|---|---|---|---|---|"]
        thresholds_found = 0
        for key in groups:
            kind, w = key.split("/world")
            sub = [x for x in valid if x.collective == kind and x.world == int(w)]
            bw = [x.bandwidth_gbs for x in sub
                  if x.bandwidth_gbs == x.bandwidth_gbs]
            cvs = [x.cv for x in sub if x.cv == x.cv]
            rep = find_threshold_band(sub, group=key)
            thresholds_found += len(rep.steps)
            gc = (cov.get("groups") or {}).get(key, {})
            bad = [f"{k}" for k, v in (gc.get("regimes") or {}).items()
                   if not v.get("covered")]
            steps = ", ".join(f"{f:.1f}x@{b}" for _, b, f in rep.steps) or "none"
            L.append(f"| {key} | {max(bw) if bw else float('nan'):.1f} | "
                     f"{np.median(cvs) if cvs else float('nan'):.1%} | {steps} | "
                     f"{', '.join(bad) if bad else '—'} |")
        L += ["", f"**Covered {mc.get('n_covered_cells', 0)} of "
              f"{cov.get('n_required_cells', 0)} cells; "
              f"{len(((cov.get('parameter_validity') or {}).get('accepted')) or [])} "
              f"of {cov.get('n_required_groups', 0)} groups have an accepted "
              f"model.**", ""]
        summary[name] = {
            "covered": mc.get("n_covered_cells", 0),
            "required": cov.get("n_required_cells", 0),
            "n_steps": thresholds_found,
            "peak_gbs": max((x.bandwidth_gbs for x in valid
                             if x.bandwidth_gbs == x.bandwidth_gbs), default=float("nan")),
            "median_cv": float(np.median([x.cv for x in valid if x.cv == x.cv]))
            if valid else float("nan"),
        }

    # The comparison the rental exists to make.
    L += ["## What this says about the local finding", ""]
    loc = art / "campaign-v2.json"
    if loc.exists():
        lc = json.loads(loc.read_text(encoding="utf-8"))
        lcov = lc["coverage"]
        lrecs = [PointRecord.from_dict(x) for x in lc["records"]]
        lsteps = 0
        per_world = {}
        for key in sorted({f"{x.collective}/world{x.world}" for x in lrecs if x.valid}):
            kind, w = key.split("/world")
            sub = [x for x in lrecs if x.valid and x.collective == kind
                   and x.world == int(w)]
            n = len(find_threshold_band(sub, group=key).steps)
            lsteps += n
            per_world.setdefault(int(w), 0)
            per_world[int(w)] += n
        L += [
            "| | local (gloo, shared memory) | rented (nccl, PCIe) |",
            "|---|---|---|",
            f"| covered | {lcov['model_coverage']['n_covered_cells']} of "
            f"{lcov['n_required_cells']} | "
            + " / ".join(f"{v['covered']} of {v['required']}"
                         for v in summary.values()) + " |",
            f"| algorithm steps detected | {lsteps} | "
            + " / ".join(str(v["n_steps"]) for v in summary.values()) + " |",
            f"| median CV | "
            f"{np.median([x.cv for x in lrecs if x.valid and x.cv == x.cv]):.1%} | "
            + " / ".join(f"{v['median_cv']:.1%}" for v in summary.values()) + " |",
            "",
            "Locally the steps sat in world 2 and world 3 and world 4 was clean "
            f"(steps per world: {dict(sorted(per_world.items()))}). Whether that "
            "was a property of collectives or of one contended desktop was not "
            "answerable there.", "",
        ]
    else:
        L += ["The local campaign artifact is not present here, so no comparison "
              "is drawn.", ""]

    md = "\n".join(L)
    print(md)
    payload = {"kind": "gpu-fabric-report", "standard_version": STANDARD_VERSION,
               "generated_at": utcnow(), "summary": summary,
               "topology": topo, "protocol_seal": d["protocol"]["seal_hash"]}
    (art / "gpu-fabric-report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (art / "gpu-fabric-report.md").write_text(md, encoding="utf-8")
    (art / "gpu-fabric-report.html").write_text(
        render_page(title="The fabric, measured",
                    subtitle="An eight-GPU node, characterised before it is trusted",
                    body=md_to_html(md)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
