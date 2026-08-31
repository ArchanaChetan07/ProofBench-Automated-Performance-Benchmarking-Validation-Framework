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

    debt = assess_debt(cov, d["model_selection"])
    debt.notes.append(
        "The medium regime is model-limited in every all_reduce group. The targeted "
        "probe that would have settled whether that is evidence or structure came "
        "back INCONCLUSIVE: its two measurement sessions differed by 1.42x and could "
        "not be pooled. Re-measuring both grids in one session is the outstanding "
        "experiment."
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
