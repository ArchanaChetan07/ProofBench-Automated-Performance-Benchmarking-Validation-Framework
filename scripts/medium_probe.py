"""Run the targeted medium-regime probe."""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.medium_probe import run_probe
    from losscolumn.version import STANDARD_VERSION

    probe = run_probe()
    md = probe.to_markdown()
    print()
    print(md)

    out = Path("artifacts")
    (out / "prereg").mkdir(parents=True, exist_ok=True)
    proto = probe.protocol()
    (out / "prereg" / "medium-probe-v1.protocol.json").write_text(
        json.dumps(proto, indent=2, default=str), encoding="utf-8")

    stem = "probe-thrust1-medium-regime"
    payload = probe.to_dict()
    payload.update({"kind": "probe-report", "standard_version": STANDARD_VERSION,
                    "generated_at": utcnow(),
                    "command": "python scripts/medium_probe.py"})
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    (out / f"{stem}.html").write_text(
        render_page(title="Medium-regime probe",
                    subtitle="Model-limited, or evidence-thin?",
                    body=md_to_html(md),
                    meta={"Device": probe.device,
                          "Seal": proto["seal_hash"][7:23]}),
        encoding="utf-8")
    print(f"\nwritten {out / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
