"""Run the version 2 memory-model study: preflight, calibrate, freeze, validate."""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.memcal2 import run_study
    from losscolumn.version import STANDARD_VERSION

    study = run_study(progress=True)
    md = study.to_markdown()
    print()
    print(md)

    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)

    proto = study.protocol()
    (out / "prereg" / "memory-model-v2.protocol.json").parent.mkdir(
        parents=True, exist_ok=True)
    (out / "prereg" / "memory-model-v2.protocol.json").write_text(
        json.dumps(proto, indent=2, default=str), encoding="utf-8")

    stem = "calibration-thrust1-memory-v2"
    payload = study.to_dict()
    payload.update({
        "kind": "calibration-report", "standard_version": STANDARD_VERSION,
        "calibrates": "thrust1", "generated_at": utcnow(),
        "command": "python scripts/memcal2.py",
    })
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    v = study.validation
    (out / f"{stem}.html").write_text(
        render_page(
            title="Thrust I memory model v2",
            subtitle="Replaced, then graded on a validation grid it had never seen.",
            body=md_to_html(md),
            meta={
                "Device": study.device,
                "False wins": f"{v.n_false_win}/{v.n_comparable}" if v else "-",
                "Seal": proto["seal_hash"][7:23],
            },
        ),
        encoding="utf-8",
    )
    print(f"\nwritten {out / stem}.json")
    print(f"written {out / 'prereg' / 'memory-model-v2.protocol.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
