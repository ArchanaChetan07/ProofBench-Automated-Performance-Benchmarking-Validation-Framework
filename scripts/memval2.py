"""Expanded validation of the frozen block-memory-v2 model."""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.memval2 import run_expanded_validation
    from losscolumn.version import STANDARD_VERSION

    study = run_expanded_validation()
    md = study.to_markdown()
    print()
    print(md)

    out = Path("artifacts")
    (out / "prereg").mkdir(parents=True, exist_ok=True)
    proto = study.protocol()
    (out / "prereg" / "memory-v2-expanded-validation.protocol.json").write_text(
        json.dumps(proto, indent=2, default=str), encoding="utf-8")

    stem = "validation-thrust1-memory-v2-expanded"
    payload = study.to_dict()
    payload.update({"kind": "validation-report", "standard_version": STANDARD_VERSION,
                    "generated_at": utcnow(), "command": "python scripts/memval2.py"})
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    m = study.matrix
    (out / f"{stem}.html").write_text(
        render_page(
            title="block-memory-v2 expanded validation",
            subtitle="Nine sequence lengths, two precisions, three effective capacities.",
            body=md_to_html(md),
            meta={"Device": study.device,
                  "False wins": f"{m.n_false_win}/{m.n_comparable}" if m else "-",
                  "Seal": proto["seal_hash"][7:23]},
        ), encoding="utf-8")
    print(f"\nwritten {out / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
