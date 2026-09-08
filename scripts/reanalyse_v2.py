"""Re-analyse the campaign from its stored records with the current code.

The records are the measurement; the coverage matrix is a pure function of them
and of the analysis, so when the analysis is corrected the matrix has to be
recomputed rather than left as it was when the campaign ran. Nothing is
re-measured and the records are untouched.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.artifacts import campaign_path
    from losscolumn.core.provenance import utcnow
    from losscolumn.thrusts.overlap.campaign import Campaign, PointRecord, analyse

    art = Path("artifacts")
    p = campaign_path(art)
    d = json.loads(p.read_text(encoding="utf-8"))
    camp = Campaign(
        grid=tuple(d["protocol"]["dense_grid"]),
        records=[PointRecord.from_dict(r) for r in d["records"]],
        device=d["device"], errors=d.get("errors", {}) or {},
        n_passes=d["protocol"]["passes"],
        repeats=d["protocol"]["repeats_per_point_per_pass"],
    )
    analyse(camp)
    out = camp.to_dict()
    # Keep everything the run recorded about itself; replace only what the
    # analysis produces.
    for k in ("kind", "standard_version", "generated_at", "predictions",
              "baseline", "session_id", "elapsed_s", "n_evidence_sessions"):
        if k in d:
            out[k] = d[k]
    out["reanalysed_at"] = utcnow()
    out["reanalysis_note"] = (
        "Coverage, selection and threshold detection recomputed from the stored "
        "records with the current analysis. The records are byte-identical to "
        "what the campaign measured; what changed is the code that reads them."
    )
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    cov = out["coverage"]
    print(f"{p.name}: covered {cov['model_coverage']['n_covered_cells']} of "
          f"{cov['n_required_cells']}, accepted "
          f"{len(cov['parameter_validity']['accepted'])} of "
          f"{cov['n_required_groups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
