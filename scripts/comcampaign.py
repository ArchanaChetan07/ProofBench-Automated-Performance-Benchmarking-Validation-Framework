"""Run the communication coverage and readiness campaign."""
import json
import sys
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description
    from losscolumn.thrusts.overlap.campaign import (
        REQUIRED_COLLECTIVES,
        REQUIRED_WORLDS,
        Campaign,
        analyse,
        build_grid,
        sweep_pass,
    )
    from losscolumn.version import STANDARD_VERSION

    quick = "--quick" in sys.argv
    grid = build_grid(backbone_per_decade=3 if quick else 6,
                      oversample_per_region=3 if quick else 5)
    worlds = (2, 3) if quick else REQUIRED_WORLDS
    n_passes = 2
    repeats = 2

    camp = Campaign(grid=grid, n_passes=n_passes, repeats=repeats,
                    device=device_description("cuda"))
    proto = camp.protocol()
    out = Path("artifacts")
    (out / "prereg").mkdir(parents=True, exist_ok=True)
    (out / "prereg" / "comm-campaign-v1.protocol.json").write_text(
        json.dumps(proto, indent=2, default=str), encoding="utf-8")
    print(f"protocol sealed {proto['seal_hash'][:23]}")
    print(f"grid: {len(grid)} sizes, {grid[0]} B .. {grid[-1] / 1e6:.1f} MB")
    print(f"{len(REQUIRED_COLLECTIVES)} collectives x {len(worlds)} worlds x "
          f"{n_passes} passes x {repeats} repeats", flush=True)

    with MeasurementLock("communication campaign"):
        for p in range(n_passes):
            print(f"\n=== pass {p} ===", flush=True)
            recs, errs = sweep_pass(sizes=grid, worlds=worlds,
                                    collectives=REQUIRED_COLLECTIVES,
                                    repeats=repeats, pass_index=p)
            camp.records.extend(recs)
            camp.errors.update(errs)
            print(f"  {len(recs)} records, {len(errs)} failure(s)", flush=True)

    analyse(camp)

    md = _report(camp)
    print()
    print(md)

    stem = "campaign-thrust1-communication"
    payload = camp.to_dict()
    payload.update({"kind": "campaign-report", "standard_version": STANDARD_VERSION,
                    "generated_at": utcnow(),
                    "command": "python scripts/comcampaign.py"})
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    verdict, _ = camp.coverage.readiness()
    (out / f"{stem}.html").write_text(
        render_page(title="Communication readiness campaign",
                    subtitle="Coverage, not just acceptance.",
                    body=md_to_html(md),
                    meta={"Device": camp.device, "Readiness": verdict.value,
                          "Seal": proto["seal_hash"][7:23]}),
        encoding="utf-8")
    print(f"\nwritten {out / stem}.json")
    return 0


def _report(camp) -> str:
    lines = ["### Communication coverage and readiness campaign", "",
             f"Measured on {camp.device}. "
             f"{camp.protocol()['n_grid_points']} message sizes, "
             f"{camp.n_passes} independent passes, {camp.repeats} repeats per point.",
             "", "#### Peak reproducibility", "",
             "| Group | pass 0 peak | pass 1 peak | magnitude ratio | ordering | reproducible |",
             "|---|---|---|---|---|---|"]
    for k, pk in sorted(camp.peaks.items()):
        if len(pk.per_pass) < 2:
            lines.append(f"| {k} | — | — | — | — | insufficient passes |")
            continue
        a, b = pk.per_pass[0], pk.per_pass[1]
        lines.append(
            f"| {k} | {a['peak_bytes'] / 1024:.0f} KiB ({a['peak_regime']}) | "
            f"{b['peak_bytes'] / 1024:.0f} KiB ({b['peak_regime']}) | "
            f"{pk.peak_magnitude_ratio:.2f}x | "
            f"{'stable' if pk.ordering_stable else 'CHANGED'} | "
            f"{'yes' if pk.reproducible else '**no**'} |")
    lines.append("")
    for k, pk in sorted(camp.peaks.items()):
        if pk.finding:
            lines.append(f"- **{k}**: {pk.finding}.")
    lines += ["", "#### Coverage matrix", "", camp.coverage.to_markdown()]
    if camp.errors:
        lines += ["", "#### Measurement failures", ""]
        lines += [f"- `{k}`: {v}" for k, v in camp.errors.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
