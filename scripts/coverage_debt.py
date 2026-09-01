"""Generate the coverage-debt report from the campaign artifact."""
import json
from pathlib import Path

# The 21-repeat re-measurement, not the original 2-repeat campaign. Its CVs are
# measured rather than corrected, which is the whole reason it was run: a
# two-repeat CV cannot be turned into a usable noise estimate by any factor.
_SOURCE = "recampaign-thrust1-communication.json"
_FALLBACK = "campaign-thrust1-communication.json"


def main() -> int:
    from losscolumn.core.coverage import (
        CommunicationCoverage,
        ParameterVerdict,
        RegimeStatus,
    )
    from losscolumn.core.debt import assess_debt
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page

    art = Path("artifacts")
    src = art / _SOURCE if (art / _SOURCE).exists() else art / _FALLBACK
    d = json.loads(src.read_text(encoding="utf-8"))
    c = d["coverage"]
    cov = CommunicationCoverage.empty(c["required_collectives"], c["required_worlds"],
                                      c["required_regimes"])
    for k, g in c["groups"].items():
        G = cov.groups[k]
        G.parameter_verdict = ParameterVerdict(g["parameter_verdict"])
        G.model_family = g["model_family"]
        G.worst_regime_err = g["worst_regime_err"]
        for rk, r in g["regimes"].items():
            R = G.regimes[rk]
            R.status = RegimeStatus(r["status"])
            R.n_points = r["n_points"]
            R.n_validation_points = r["n_validation_points"]
            R.heldout_err = r["heldout_err"]
            R.noise_cv = r["noise_cv"]

    n_rep = int(d["protocol"].get("repeats_per_point_per_pass", 0) or 0)
    debt = assess_debt(cov, d["model_selection"], cv_from_n=n_rep)
    debt.notes.append(
        f"Computed from `{src.name}`, whose CVs come from {n_rep} repeats per "
        "point in a single session and are used exactly as measured."
    )
    debt.notes.append(
        "This supersedes a ledger that read 9 model-limited and 7 noise-limited. "
        "That one corrected the original campaign's two-repeat CVs by a single "
        "factor of 1.84x and moved six cells to noise-limited on the strength of "
        "it. A pre-registered re-measurement falsified the correction: it "
        "predicted the recorded CV would rise to 10.7% and it rose to 19.0%. The "
        "ledger and the correction are preserved at "
        "artifacts/history/2026-08-31-two-sample-cv-correction."
    )
    debt.notes.append(
        "No correction replaces it, because none can. Within the re-measurement's "
        "own session the excess splits into 2.25x from the estimator and a "
        "further 1.35x from timescale: twenty-one repeats span more wall-clock "
        "than two adjacent ones and see slower variation. The quantity a "
        "correction is meant to recover therefore grows with the window it is "
        "measured over, so the factor depends on an arbitrary reference -- 1.84x "
        "against seven repeats, 2.25x against twenty-one."
    )
    debt.notes.append(
        "The practical consequence is that every coverage figure this project "
        "produced before this measurement rested on a noise estimate roughly "
        "three times too small. Median cell CV moved from 5.7% to 18.4% and the "
        "number of cells above the 20% ceiling from 0 to 10. Coverage fell from 8 "
        "cells to 6, which is a correction rather than a regression: the earlier "
        "figure was inflated by an instrument that under-reported its own "
        "variation."
    )
    md = "### Coverage debt\n\n" + debt.to_markdown()
    print(md)

    payload = debt.to_dict()
    payload.update({"kind": "coverage-debt", "generated_at": utcnow(),
                    "command": "python scripts/coverage_debt.py"})
    (art / "coverage-debt.json").write_text(json.dumps(payload, indent=2, default=str),
                                            encoding="utf-8")
    (art / "coverage-debt.md").write_text(md, encoding="utf-8")
    (art / "coverage-debt.html").write_text(
        render_page(title="Coverage debt",
                    subtitle="What is missing, why, and what would fix it.",
                    body=md_to_html(md),
                    meta={"Uncovered cells": str(len(debt.items)),
                          "Dominant": debt.dominant_kind.value}),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
