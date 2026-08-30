"""Entry point for the Thrust I memory-feasibility calibration."""
import json
import sys
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import Provenance, utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.memcal import run_memory_calibration
    from losscolumn.version import STANDARD_VERSION

    quick = "--quick" in sys.argv
    cal = run_memory_calibration(
        # The focused grid straddles the model's own predicted feasibility
        # boundary, chosen by plan_calibration rather than by sweeping
        # everything: the cells far inside either region cost measurement time
        # and settle nothing.
        # Straddles the model's own predicted feasibility boundary, chosen by
        # plan_calibration rather than by sweeping everything: cells far inside
        # either region cost measurement time and settle nothing.
        #
        # The upper corner is excluded deliberately. At seq=4096 with a micro
        # batch of 256 the allocator falls back to host memory rather than
        # raising, and the process thrashes for hours instead of reporting an
        # honest out-of-memory. A cell that cannot produce a clean answer is
        # not a cheap cell to include.
        seq_lens=(2048, 4096) if quick else (512, 1024, 2048, 4096),
        micro_batches=(64, 128) if quick else (8, 16, 32, 64, 128, 256),
    )
    md = cal.to_markdown()
    print()
    print(md)

    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    stem = "calibration-thrust1-memory"
    payload = cal.to_dict()
    payload.update({
        "kind": "calibration-report", "standard_version": STANDARD_VERSION,
        "calibrates": "thrust1", "generated_at": utcnow(),
        "command": "python scripts/memcal.py",
        "provenance": Provenance.capture().to_dict(),
    })
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    (out / f"{stem}.html").write_text(
        render_page(
            title="Calibration -- Thrust I memory feasibility",
            subtitle="Does the recommended configuration fit? The model predicts, the "
                     "hardware answers.",
            body=md_to_html(md),
            meta={"Device": cal.device,
                  "Verdict": cal.result.verdict()[:90],
                  "False wins": str(cal.result.n_false_win)},
        ),
        encoding="utf-8",
    )
    print(f"\nwritten {out / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
