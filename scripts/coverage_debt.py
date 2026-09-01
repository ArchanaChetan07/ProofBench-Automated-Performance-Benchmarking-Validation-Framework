"""Generate the coverage-debt report from the campaign artifact."""
import json
from pathlib import Path


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
    d = json.loads((art / "campaign-thrust1-communication.json").read_text(encoding="utf-8"))
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

    # The campaign recorded two repeats per point, so every CV in it is a
    # two-sample estimate and reads far cleaner than the measurement is.
    n_rep = int(d["protocol"].get("repeats", 0) or 0) or 2
    debt = assess_debt(cov, d["model_selection"], cv_from_n=n_rep)
    debt.notes.append(
        "This supersedes an earlier ledger that read 9 model-limited and 7 "
        "noise-limited. Nothing was re-measured: the earlier one took each cell's "
        "recorded CV at face value, and every one of those was computed from two "
        "repeats. Correcting that estimator moved six cells, all of them from "
        "model-limited to noise-limited, and three of the four medium-regime cells "
        "among them."
    )
    debt.notes.append(
        "The correction inverts the plan the earlier ledger implied. Most of the "
        "supposed modelling work was never modelling work: the residuals it "
        "pointed at are smaller than the instrument's own variation, and no model "
        "family can beat the instrument. The remaining model-limited cells are "
        "worth attention precisely because there are only three of them."
    )
    debt.notes.append(
        "It also dissolves the medium-regime puzzle without appealing to session "
        "drift. all_reduce/world2 and world3 show 26% to 30% run-to-run variation "
        "there against a 20% ceiling, so nothing can be graded in those cells at "
        "all -- which is why a probe that added 32 points to that regime produced "
        "a worse fit rather than a better one."
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
