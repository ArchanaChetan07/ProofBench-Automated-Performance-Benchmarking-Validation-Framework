"""The campaign re-run with six corrected gates and a corrected harness.

One variable is deliberately held fixed: the replicate count stays at 21, the
same as the run this is compared against. If coverage moves, it moves because
the instrument and the gates were wrong, not because more effort was spent.

The predictions are sealed before the measurement. The previous round's were
too, and two of the three were falsified -- which is the reason several of the
things being tested here exist.
"""
import json
import sys
import time
from pathlib import Path

REPEATS = 21


def predictions(baseline: dict) -> dict:
    return {
        "name": "campaign-v2-corrected-gates-v1",
        "question": (
            "With the gradeability gate judging the precision of the point "
            "rather than the variability of a call, the block budget following "
            "the noise instead of the message size, and structure selected on "
            "the same scale the fits use -- does the surface become gradable?"
        ),
        "design": {
            "repeats_per_point_per_pass": REPEATS,
            "held_fixed": "the replicate count, so the comparison isolates the "
                          "harness and gate corrections",
            "sessions": 1,
        },
        "corrections_under_test": [
            "the timed block is sized by the point's own variability rather "
            "than by a 1 MiB threshold, which had put a 4x averaging cliff "
            "inside the medium regime",
            "the gradeability gate judges the standard error of the point the "
            "model is fitted to, not the population CV of its repeats, which "
            "could never respond to measurement effort",
            "a regime that cannot be graded no longer votes to reject a family",
            "breakpoints and segment splits are chosen on relative residuals, "
            "the scale all three estimators already use",
        ],
        "predictions": [
            {
                "id": "P1-medium-noise-falls",
                "claim": (
                    "The median recorded CV in the medium regime falls by at "
                    "least a quarter, because points above 1 MiB there averaged "
                    "5 calls per timing and will now average considerably more."
                ),
                "baseline": baseline["medium_cv"],
                "predicted_below": baseline["medium_cv"] * 0.75,
                "falsified_if": (
                    "medium-regime CV is unchanged, which would mean the cliff "
                    "was a coincidence of size rather than of averaging and the "
                    "harness correction addresses nothing"
                ),
            },
            {
                "id": "P2-noise-cliff-closes",
                "claim": (
                    "The gap between medium points below and above 1 MiB "
                    f"(currently {baseline['cliff']:.2f}x) falls below 1.25x, "
                    "because nothing in the new rule keys on that boundary."
                ),
                "baseline": baseline["cliff"],
                "predicted_below": 1.25,
                "falsified_if": (
                    "the gap persists, which would mean something other than "
                    "the iteration count makes messages above 1 MiB noisier"
                ),
            },
            {
                "id": "P3-coverage-improves",
                "claim": (
                    f"Coverage exceeds {baseline['n_covered_cells']} of "
                    f"{baseline['n_required_cells']} cells."
                ),
                "baseline": baseline["n_covered_cells"],
                "falsified_if": (
                    "coverage does not improve, in which case the gates were "
                    "not what was blocking it and the remaining obstacle is the "
                    "surface itself"
                ),
            },
        ],
        "committed_before_measurement": True,
        "not_predicted": (
            "Nothing here predicts 24 of 24. The corrections are to gates that "
            "were measuring the wrong quantity; whether what remains is "
            "reachable at all is a separate question, and a result showing the "
            "surface is simply too variable in places would be an answer to it."
        ),
    }


def main() -> int:
    import numpy as np

    from losscolumn.core.provenance import content_hash, utcnow
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

    art = Path("artifacts")
    old = json.loads(
        (art / "recampaign-thrust1-communication.json").read_text(encoding="utf-8"))
    R = [r for r in old["records"] if r["valid"] and r["cv"] == r["cv"]]
    MB = 1 << 20
    med_lo = [r["cv"] for r in R if (1 << 18) <= r["nbytes"] < MB]
    med_hi = [r["cv"] for r in R if MB <= r["nbytes"] < (1 << 22)]
    med_all = [r["cv"] for r in R if (1 << 18) <= r["nbytes"] < (1 << 22)]
    baseline = {
        "medium_cv": float(np.median(med_all)),
        "cliff": float(np.median(med_hi) / np.median(med_lo)),
        "n_covered_cells": old["coverage"]["model_coverage"]["n_covered_cells"],
        "n_required_cells": old["coverage"]["n_required_cells"],
    }

    quick = "--quick" in sys.argv
    grid = build_grid(backbone_per_decade=3 if quick else 6,
                      oversample_per_region=3 if quick else 5)
    worlds = (2, 3) if quick else REQUIRED_WORLDS
    repeats = 5 if quick else REPEATS

    pred = predictions(baseline)
    pred["seal_hash"] = content_hash(pred)
    (art / "prereg").mkdir(parents=True, exist_ok=True)
    (art / "prereg" / "campaign-v2.protocol.json").write_text(
        json.dumps(pred, indent=2, default=str), encoding="utf-8")
    print(f"predictions sealed {pred['seal_hash'][:23]} BEFORE measurement")
    print(f"baseline: medium CV {baseline['medium_cv']:.1%}, "
          f"cliff {baseline['cliff']:.2f}x, coverage {baseline['n_covered_cells']}")
    print(f"grid {len(grid)} sizes x {len(worlds)} worlds x 2 passes x "
          f"{repeats} repeats", flush=True)

    camp = Campaign(grid=grid, n_passes=2, repeats=repeats,
                    device=device_description("cuda"))
    t0 = time.time()
    with MeasurementLock("campaign v2, corrected harness"):
        for p in range(2):
            print(f"\n=== pass {p} ===", flush=True)
            recs, errs = sweep_pass(sizes=grid, worlds=worlds,
                                    collectives=REQUIRED_COLLECTIVES,
                                    repeats=repeats, pass_index=p)
            camp.records.extend(recs)
            camp.errors.update(errs)
            print(f"  {len(recs)} records, {len(errs)} failure(s), "
                  f"{time.time() - t0:.0f}s", flush=True)
    analyse(camp)

    payload = camp.to_dict()
    payload.update({
        "kind": "campaign-report", "standard_version": STANDARD_VERSION,
        "generated_at": utcnow(), "predictions": pred, "baseline": baseline,
        "session_id": "campaign-v2", "elapsed_s": time.time() - t0,
        "n_evidence_sessions": 1,
    })
    (art / "campaign-v2.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = _verdict(payload, pred, baseline, comparable=not quick)
    print()
    print(md)
    (art / "campaign-v2.md").write_text(md, encoding="utf-8")
    (art / "campaign-v2.html").write_text(
        render_page(title="Campaign v2",
                    subtitle="Corrected gates, corrected harness, same replicate count",
                    body=md_to_html(md)), encoding="utf-8")
    return 0


def _verdict(payload, pred, baseline, *, comparable=True) -> str:
    import numpy as np

    R = [r for r in payload["records"] if r["valid"] and r["cv"] == r["cv"]]
    MB = 1 << 20
    lo = [r["cv"] for r in R if (1 << 18) <= r["nbytes"] < MB]
    hi = [r["cv"] for r in R if MB <= r["nbytes"] < (1 << 22)]
    allm = [r["cv"] for r in R if (1 << 18) <= r["nbytes"] < (1 << 22)]
    med = float(np.median(allm)) if allm else float("nan")
    cliff = float(np.median(hi) / np.median(lo)) if lo and hi else float("nan")
    cov = payload["coverage"]
    n_cov = cov["model_coverage"]["n_covered_cells"]

    p1 = med < baseline["medium_cv"] * 0.75
    p2 = cliff < 1.25
    p3 = comparable and n_cov > baseline["n_covered_cells"]

    iters = [r.get("iters", 0) for r in R if r.get("iters")]
    L = ["# Campaign v2", "",
         f"Predictions sealed `{pred['seal_hash'][:23]}` before the run. One "
         f"session, {payload['elapsed_s'] / 60:.0f} minutes, {REPEATS} repeats "
         "per point per pass -- the same replicate count as the run it is "
         "compared against, so any movement is the corrections and not the "
         "effort.", "",
         "| Prediction | Baseline | Predicted | Measured | Result |",
         "|---|---|---|---|---|",
         f"| P1 medium-regime CV falls a quarter | {baseline['medium_cv']:.1%} | "
         f"< {baseline['medium_cv'] * 0.75:.1%} | {med:.1%} | "
         f"{'**HELD**' if p1 else '**FALSIFIED**'} |",
         f"| P2 the 1 MiB noise cliff closes | {baseline['cliff']:.2f}x | "
         f"< 1.25x | {cliff:.2f}x | {'**HELD**' if p2 else '**FALSIFIED**'} |",
         f"| P3 coverage improves | {baseline['n_covered_cells']} | "
         f"> {baseline['n_covered_cells']} | {n_cov} | "
         + ("not evaluated |" if not comparable else
            f"{'**HELD**' if p3 else '**FALSIFIED**'} |"), "",
         f"Coverage **{n_cov} of {cov['n_required_cells']}** cells; "
         f"**{len(cov['parameter_validity']['accepted'])} of "
         f"{cov['n_required_groups']}** groups have an accepted model.", ""]
    if iters:
        L += [f"Timed blocks averaged {min(iters)}-{max(iters)} calls "
              f"(median {int(np.median(iters))}), sized by a fixed duration so "
              "that averaging follows the cost of a call rather than a byte "
              "threshold. Points where one call already exceeds the budget take "
              "the floor of five.", ""]
    if not p1:
        L += ["**P1 is falsified.** The medium regime is no less variable for "
              "being averaged harder, so its variability is not the averaging. "
              "Something about that size band on this machine is intrinsically "
              "unstable and the harness correction does not reach it.", ""]
    if not p2:
        L += ["**P2 is falsified.** The gap across 1 MiB survives a rule that "
              "does not know where 1 MiB is, so it is a property of the "
              "transport rather than of the measurement.", ""]
    if comparable and not p3:
        L += ["**P3 is falsified.** The gates were not what was blocking "
              "coverage. What remains is the surface.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
