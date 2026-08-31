"""Run the Thrust I communication calibration and validation study.

A real file with a __main__ guard: the collectives are measured in spawned
worker processes, and spawn re-imports the parent's __main__ in each child.
"""
import json
import sys
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.comcal import run_study
    from losscolumn.version import STANDARD_VERSION

    quick = "--quick" in sys.argv
    study = run_study(
        world_sizes=(2, 3) if quick else (2, 3, 4),
        replicates=2 if quick else 3,
    )
    md = study.to_markdown()
    print()
    print(md)

    out = Path("artifacts")
    (out / "prereg").mkdir(parents=True, exist_ok=True)
    proto = study.protocol()
    (out / "prereg" / "comm-model-v1.protocol.json").write_text(
        json.dumps(proto, indent=2, default=str), encoding="utf-8")

    stem = "calibration-thrust1-communication"
    payload = study.to_dict()
    payload.update({
        "kind": "calibration-report", "standard_version": STANDARD_VERSION,
        "calibrates": "thrust1-communication", "generated_at": utcnow(),
        "command": "python scripts/comcal.py",
    })
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    (out / f"{stem}.html").write_text(
        render_page(
            title="Thrust I communication model",
            subtitle="Why one collective fits a straight line and the other does not.",
            body=md_to_html(md),
            meta={"Device": study.device,
                  "Samples": str(len(study.samples)),
                  "Seal": proto["seal_hash"][7:23]},
        ),
        encoding="utf-8",
    )
    print(f"\nwritten {out / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
