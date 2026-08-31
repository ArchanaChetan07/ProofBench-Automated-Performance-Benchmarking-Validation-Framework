"""Re-analyse the communication campaign from its stored records.

No re-measurement. The artifact carries every timing, so the diagnosis, family
selection, fit, validation and coverage are a pure function of it -- which is
also a check on the artifact: anything the analysis needs and the artifact
lacks fails here.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.campaign import Campaign, PointRecord, analyse

    art = Path("artifacts") / "campaign-thrust1-communication.json"
    d = json.loads(art.read_text(encoding="utf-8"))
    camp = Campaign(
        grid=tuple(d["protocol"]["dense_grid"]),
        records=[PointRecord.from_dict(r) for r in d["records"]],
        device=d["device"], errors=d.get("errors", {}) or {},
        n_passes=d["protocol"]["passes"],
        repeats=d["protocol"]["repeats_per_point_per_pass"],
    )
    analyse(camp)

    from scripts.comcampaign import _report  # noqa: PLC0415
    md = _report(camp)
    print(md)

    payload = camp.to_dict()
    payload.update({k: d[k] for k in ("kind", "standard_version", "generated_at",
                                      "command") if k in d})
    art.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (art.parent / "campaign-thrust1-communication.md").write_text(md, encoding="utf-8")
    verdict, _ = camp.coverage.readiness()
    (art.parent / "campaign-thrust1-communication.html").write_text(
        render_page(title="Communication readiness campaign",
                    subtitle="Coverage, not just acceptance.",
                    body=md_to_html(md),
                    meta={"Device": camp.device, "Readiness": verdict.value}),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
