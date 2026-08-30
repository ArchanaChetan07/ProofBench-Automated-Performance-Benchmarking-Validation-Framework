"""Regenerate the v2 study report from its stored artifact.

No re-measurement: the artifact carries every cell's predicted and measured
state, so a report is a pure function of it. That is also a check on the
artifact -- anything the report needs and the artifact lacks would fail here.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.feasibility import FeasibilityMatrix
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page

    art = Path("artifacts") / "calibration-thrust1-memory-v2.json"
    d = json.loads(art.read_text(encoding="utf-8"))
    cal = FeasibilityMatrix.from_dict(d["calibration_result"])
    val = FeasibilityMatrix.from_dict(d["validation_result"])
    d["calibration_result"] = cal.to_dict()
    d["validation_result"] = val.to_dict()

    old = {"iou": 0.17, "boundary": "1 level late", "false_wins": "5/8",
           "false_losses": "0/8", "dangerous": 6.25}
    bd = [b for b in val.boundary_report(["micro_batch", "seq_len", "checkpointing"])
          if b["n_fibers_compared"]]
    worst = max((abs(b["mean_signed_shift"]) for b in bd), default=float("nan"))
    shift = f"**0 levels** on all {len(bd)} axes" if worst == 0 else f"{worst:.2f} levels"

    md = "\n".join([
        "### Thrust I memory model, version 2",
        "",
        f"Measured on {d['device']}. Protocol seal "
        f"`{d['protocol']['seal_hash'][7:23]}...`, grid split "
        f"`{d['protocol']['grid_split']['digest'][7:23]}...`.",
        "",
        "#### Calibration grid (the grid that diagnosed v1; not evidence about v2)",
        "", cal.to_markdown(), "",
        "#### Validation grid (untouched until the model was frozen)",
        "", val.to_markdown(), "",
        "#### Old model against new",
        "",
        "| Metric | v1 (diagnosis grid) | v2 (validation grid) |",
        "|---|---|---|",
        f"| Infeasible-region IoU | {old['iou']} | {val.infeasible_iou:.2f} |",
        f"| Boundary error | {old['boundary']} | {shift} |",
        f"| False wins | {old['false_wins']} | **{val.n_false_win}/{val.n_comparable}** |",
        f"| False losses | {old['false_losses']} | {val.n_false_loss}/{val.n_comparable} |",
        f"| Dangerous-error score (w=10) | {old['dangerous']:.2f} | "
        f"{val.dangerous_error_score:.2f} |",
        f"| Accuracy | 38% | {val.accuracy:.0%} |",
        "",
        "The two columns are **not** measured on the same cells and cannot be: the v1 "
        "numbers come from the grid that diagnosed it, which is this study's "
        "calibration grid and therefore inadmissible as evidence about v2. Each column "
        "is that model's result on the hardest grid available to it, not a paired "
        "contest.",
        "",
        "#### Notes",
        "",
    ] + [f"- {n}" for n in d.get("notes", [])])

    art.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    (art.parent / "calibration-thrust1-memory-v2.md").write_text(md, encoding="utf-8")
    (art.parent / "calibration-thrust1-memory-v2.html").write_text(
        render_page(
            title="Thrust I memory model v2",
            subtitle="Replaced, then graded on a validation grid it had never seen.",
            body=md_to_html(md),
            meta={"Device": d["device"],
                  "False wins": f"{val.n_false_win}/{val.n_comparable}",
                  "Seal": d["protocol"]["seal_hash"][7:23]},
        ),
        encoding="utf-8",
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
