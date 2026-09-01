"""Re-run the communication campaign at the replicate count the analysis asks for.

One session, one process invocation, one measurement lock. The stability
sentinel established that this machine's variation enters at the process launch
and that no two of its sessions have been shown poolable, so an experiment that
will be compared against itself has to be gathered in one sitting.

The predictions below are sealed before the measurement runs. They are
falsifiable and one of them tests the correction that motivated the whole
exercise: if a two-sample CV really understates dispersion by 1.84x, then
measuring the same points with twenty-one repeats must push the *recorded* CV
up by about that factor. Nothing else in the analysis would explain it moving.
"""
import json
import sys
import time
from pathlib import Path

REPEATS = 21
"""Chosen from the modellability report: the noisiest uncovered cell
(all_reduce/world2, medium) needs 19 repeats to reach the 15% gate."""


def predictions(baseline: dict) -> dict:
    # The withdrawn constant, referenced by its withdrawn name. The prediction
    # this builds was sealed against the value 0.545 and must go on reproducing
    # that seal: a prediction rewritten after its own falsification is not one.
    from losscolumn.core.debt import CV_N2_BIAS_WITHDRAWN as CV_N2_BIAS

    return {
        "name": "recampaign-higher-replicates-v1",
        "question": (
            "Does the communication surface become gradable when it is measured "
            "at the replicate count the noise budget requires, rather than at the "
            "two the original campaign carried?"
        ),
        "design": {
            "repeats_per_point_per_pass": REPEATS,
            "was": baseline["repeats"],
            "sessions": 1,
            "rationale": (
                "The launch-level noise floor is 12.3% and the gate is 15%, so the "
                "gate is reachable inside one launch. Repeats are bought rather "
                "than launches because a launch pays transport setup for every "
                "probe it carries."
            ),
        },
        "predictions": [
            {
                "id": "P1-cv-rises",
                "claim": (
                    f"The median recorded per-record CV rises from "
                    f"{baseline['median_cv']:.3f} to about "
                    f"{baseline['median_cv'] / CV_N2_BIAS:.3f}, because a CV from "
                    f"two samples understates dispersion by {1 / CV_N2_BIAS:.2f}x."
                ),
                "falsified_if": (
                    "the median recorded CV stays within 20% of its old value, "
                    "which would mean the two-sample bias correction is wrong and "
                    "every cell it reclassified must be reclassified back"
                ),
                "predicted_value": baseline["median_cv"] / CV_N2_BIAS,
                "tolerance": 0.25,
            },
            {
                "id": "P2-coverage-improves",
                "claim": (
                    f"Coverage rises above {baseline['n_covered_cells']} of "
                    f"{baseline['n_required_cells']} cells, because thirteen of the "
                    "sixteen uncovered cells were classified as limited by "
                    "measurement rather than by the model."
                ),
                "falsified_if": (
                    "coverage does not improve, which would mean the residuals were "
                    "never the instrument's and the noise-limited classification is "
                    "wrong"
                ),
                "predicted_value": baseline["n_covered_cells"],
                "direction": "greater",
            },
            {
                "id": "P3-model-work-cells-persist",
                "claim": (
                    "all_gather/world2 small, all_reduce/world4 tiny and "
                    "all_reduce/world4 medium stay uncovered, because their "
                    "residuals are already well above their measurement noise and "
                    "more repeats cannot close a gap the instrument did not open."
                ),
                "cells": [["all_gather/world2", "small"],
                          ["all_reduce/world4", "tiny"],
                          ["all_reduce/world4", "medium"]],
                "falsified_if": (
                    "any of the three becomes covered, which would mean the "
                    "model-limited classification was itself a noise artefact"
                ),
            },
        ],
        "committed_before_measurement": True,
        "not_predicted": (
            "Nothing here predicts that six of six groups will pass. The point is "
            "to find out which of the three diagnoses -- noise, model, instrument "
            "-- was right about each cell, and a result in which coverage improves "
            "only a little is as informative as one in which it improves a lot."
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
        (art / "campaign-thrust1-communication.json").read_text(encoding="utf-8"))
    ocv = [r["cv"] for r in old["records"] if r["valid"] and r["cv"] == r["cv"]]
    baseline = {
        "repeats": old["protocol"].get("repeats_per_point_per_pass", 2),
        "median_cv": float(np.median(ocv)),
        "n_covered_cells": old["coverage"]["model_coverage"]["n_covered_cells"],
        "n_required_cells": old["coverage"]["n_required_cells"],
        "seal": old["protocol"]["seal_hash"],
    }

    quick = "--quick" in sys.argv
    grid = build_grid(backbone_per_decade=3 if quick else 6,
                      oversample_per_region=3 if quick else 5)
    worlds = (2, 3) if quick else REQUIRED_WORLDS
    repeats = 5 if quick else REPEATS

    pred = predictions(baseline)
    pred["seal_hash"] = content_hash(pred)
    (art / "prereg").mkdir(parents=True, exist_ok=True)
    (art / "prereg" / "recampaign-v1.protocol.json").write_text(
        json.dumps(pred, indent=2, default=str), encoding="utf-8")
    print(f"predictions sealed {pred['seal_hash'][:23]} BEFORE measurement")
    print(f"grid {len(grid)} sizes x {len(REQUIRED_COLLECTIVES)} collectives x "
          f"{len(worlds)} worlds x 2 passes x {repeats} repeats", flush=True)

    camp = Campaign(grid=grid, n_passes=2, repeats=repeats,
                    device=device_description("cuda"))
    t0 = time.time()
    with MeasurementLock("recampaign at higher replicates"):
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
        "session_id": "recampaign-2026-08-31", "elapsed_s": time.time() - t0,
        "comparable_to_baseline": not quick,
    })
    (art / "recampaign-thrust1-communication.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = _verdict(payload, pred, baseline, comparable=not quick)
    print()
    print(md)
    (art / "recampaign-thrust1-communication.md").write_text(md, encoding="utf-8")
    (art / "recampaign-thrust1-communication.html").write_text(
        render_page(title="Campaign at higher replicates",
                    subtitle="Measuring at the replicate count the noise budget asks for",
                    body=md_to_html(md)), encoding="utf-8")
    return 0


def _verdict(payload: dict, pred: dict, baseline: dict, *,
             comparable: bool = True) -> str:
    import numpy as np

    cv = [r["cv"] for r in payload["records"] if r["valid"] and r["cv"] == r["cv"]]
    med = float(np.median(cv)) if cv else float("nan")
    cov = payload["coverage"]
    n_cov = cov["model_coverage"]["n_covered_cells"]

    p1 = pred["predictions"][0]
    p1_hit = abs(med - p1["predicted_value"]) / p1["predicted_value"] <= p1["tolerance"]
    # P2 and P3 compare coverage against a baseline measured on the full grid and
    # all four worlds. A reduced smoke-test design cannot falsify them, and a
    # script that says otherwise is manufacturing a result out of its own
    # configuration.
    p2_hit = comparable and n_cov > baseline["n_covered_cells"]

    still = []
    for key, reg in pred["predictions"][2]["cells"]:
        g = cov["groups"].get(key, {})
        r = (g.get("regimes") or {}).get(reg, {})
        if not r.get("covered"):
            still.append(f"{key}/{reg}")
    p3_hit = comparable and len(still) == 3

    L = ["# Campaign at higher replicates", "",
         f"Predictions sealed `{pred['seal_hash'][:23]}` before the measurement ran. "
         f"One session, {payload['elapsed_s'] / 60:.0f} minutes, "
         f"{pred['design']['repeats_per_point_per_pass']} repeats per point per "
         f"pass against the original {baseline['repeats']}.", "",
         "| Prediction | Predicted | Measured | Result |", "|---|---|---|---|",
         f"| P1 recorded CV rises by the de-bias factor | "
         f"{p1['predicted_value']:.1%} | {med:.1%} | "
         f"{'**HELD**' if p1_hit else '**FALSIFIED**'} |",
         f"| P2 coverage improves | > {baseline['n_covered_cells']} | {n_cov} | "
         + ("not evaluated |" if not comparable else
            f"{'**HELD**' if p2_hit else '**FALSIFIED**'} |"),
         f"| P3 the three model-limited cells stay uncovered | 3 | {len(still)} | "
         + ("not evaluated |" if not comparable else
            f"{'**HELD**' if p3_hit else '**FALSIFIED**'} |"), "",
         f"Coverage {n_cov} of {cov['n_required_cells']} cells, from "
         f"{baseline['n_covered_cells']}.", ""]

    if p1_hit:
        L += [f"P1 is the load-bearing one. The recorded CV moved from "
              f"{baseline['median_cv']:.1%} to {med:.1%} on the same grid and the "
              "same machine, and the only thing that changed is how many samples "
              "each CV was computed from. That is the two-sample bias, measured "
              "directly rather than inferred, and it is what licensed "
              "reclassifying six cells from model-limited to noise-limited.", ""]
    else:
        L += [f"**P1 is falsified.** The recorded CV went to {med:.1%} where "
              f"{p1['predicted_value']:.1%} was predicted. The two-sample bias "
              "correction does not describe this data, and every cell reclassified "
              "under it has to be reclassified back. That correction is withdrawn.", ""]
    if not comparable:
        L += ["**Reduced design: P2 and P3 are not evaluated.** This run used a "
              "sparser grid and two of the four worlds, so its coverage is not "
              "comparable with a baseline measured on the full lattice. P1 is "
              "evaluated because it concerns the dispersion of individual records "
              "and does not depend on which points were visited.", ""]
    elif not p2_hit:
        L += ["**P2 is falsified.** Coverage did not improve despite the residuals "
              "having been attributed to the instrument. They were not the "
              "instrument's, and the noise-limited classification is wrong.", ""]
    if comparable and still and not p3_hit:
        L += [f"**P3 is falsified.** Only {len(still)} of the three model-limited "
              "cells stayed uncovered, so at least one of them was a noise artefact "
              "rather than a modelling gap.", ""]
    if still and comparable:
        L += ["Still uncovered among the three predicted: " + ", ".join(still), ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
