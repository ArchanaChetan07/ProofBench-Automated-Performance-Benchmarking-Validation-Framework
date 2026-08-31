"""Re-analyse the communication study from its stored samples.

No re-measurement: the artifact carries every timed sample, so the diagnosis,
family selection, fit, freeze and validation are a pure function of it.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.comcal import CommStudy, run_analysis
    from losscolumn.thrusts.overlap.commodel import Sample

    art = Path("artifacts") / "calibration-thrust1-communication.json"
    d = json.loads(art.read_text(encoding="utf-8"))
    samples = [Sample(nbytes=s["nbytes"], seconds=s["seconds"], world=s["world"],
                      kind=s["kind"], replicate=s["replicate"])
               for s in d["samples"]]
    study = CommStudy(samples=samples, device=d["device"],
                      errors=d.get("errors", {}) or {})
    run_analysis(study)
    md = study.to_markdown()
    print(md)

    payload = study.to_dict()
    payload.update({k: d[k] for k in ("kind", "standard_version", "calibrates",
                                      "generated_at", "command") if k in d})
    art.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (art.parent / "calibration-thrust1-communication.md").write_text(md, encoding="utf-8")
    (art.parent / "calibration-thrust1-communication.html").write_text(
        render_page(
            title="Thrust I communication model",
            subtitle="Why one collective fits a straight line and the other does not.",
            body=md_to_html(md),
            meta={"Device": study.device, "Samples": str(len(study.samples)),
                  "Accepted": str(len(study.params.usable))},
        ), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
