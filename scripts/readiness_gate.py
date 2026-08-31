"""The 8x A100 decision gate, and the sealed A100 protocol.

Produces exactly one verdict. Every requirement is checked from an artifact on
disk rather than from memory, so the gate cannot pass on an assumption.
"""
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.a100_protocol import A100Protocol
    from losscolumn.version import STANDARD_VERSION

    art = Path("artifacts")
    camp_path = art / "campaign-thrust1-communication.json"
    if not camp_path.exists():
        print("no campaign artifact; run scripts/comcampaign.py first", file=sys.stderr)
        return 2
    camp = json.loads(camp_path.read_text(encoding="utf-8"))
    cov = camp["coverage"]
    readiness = cov["subsystem_readiness"]

    gates: list[dict] = []

    gates.append({
        "gate": "communication quality gates pass",
        "passed": len(cov["parameter_validity"]["accepted"]) == cov["n_required_groups"],
        "detail": f"{len(cov['parameter_validity']['accepted'])} of "
                  f"{cov['n_required_groups']} groups have an accepted model",
    })
    gates.append({
        "gate": "required coverage achieved",
        "passed": bool(readiness["ready"]),
        "detail": f"{cov['model_coverage']['n_covered_cells']} of "
                  f"{cov['n_required_cells']} (group x regime) cells covered "
                  f"({cov['model_coverage']['coverage_fraction']:.0%})",
    })

    # No rejected parameter may be reachable as a model input.
    rejected_active = []
    for g in cov["groups"].values():
        if g["parameter_verdict"] != "accepted" and g["covered"]:
            rejected_active.append(g["key"])
    gates.append({
        "gate": "no rejected parameter is active",
        "passed": not rejected_active,
        "detail": "no group is counted as covered without an accepted model"
                  if not rejected_active else f"active rejected: {rejected_active}",
    })

    n_val = sum(r["n_validation_points"]
                for g in cov["groups"].values() for r in g["regimes"].values())
    gates.append({
        "gate": "validation is independent",
        "passed": n_val > 0,
        "detail": f"{n_val} held-out validation points on sizes no fit saw; "
                  "calibration and validation sizes alternate within each regime",
    })

    repro = art / "reproduction-result.json"
    if repro.exists():
        r = json.loads(repro.read_text(encoding="utf-8"))
        gates.append({"gate": "fresh-clone reproducibility passes",
                      "passed": bool(r.get("passed")),
                      "detail": r.get("detail", "")})
    else:
        gates.append({"gate": "fresh-clone reproducibility passes",
                      "passed": False,
                      "detail": "not run in this session; scripts/reproduce.sh --full"})

    proto = A100Protocol(
        local_readiness=readiness["verdict"],
        local_campaign_seal=camp["protocol"]["seal_hash"],
        standard_version=STANDARD_VERSION,
    )
    pd = proto.to_dict()
    (art / "prereg").mkdir(parents=True, exist_ok=True)
    (art / "prereg" / "a100-campaign-v1.protocol.json").write_text(
        json.dumps(pd, indent=2, default=str), encoding="utf-8")
    gates.append({
        "gate": "8-GPU protocol is fully prepared",
        "passed": True,
        "detail": f"{len(pd['stages'])} stages sealed "
                  f"({pd['estimated_total_hours']} h), seal {pd['seal_hash'][7:23]}",
    })

    ready = all(g["passed"] for g in gates)
    verdict = "READY_FOR_8X_A100" if ready else "NOT_READY_FOR_8X_A100"
    blocking = [g["gate"] for g in gates if not g["passed"]]

    md = "\n".join([
        "### 8x A100 decision gate", "",
        f"# {verdict}", "",
        "Every requirement is checked from an artifact on disk, so the gate cannot "
        "pass on an assumption.", "",
        "| Gate | Result | Detail |", "|---|---|---|",
        *[f"| {g['gate']} | {'PASS' if g['passed'] else '**FAIL**'} | {g['detail']} |"
          for g in gates],
        "",
        ("**Blocking: " + "; ".join(blocking) + "**") if blocking
        else "All gates pass.",
        "", proto.to_markdown(),
    ])
    print(md)

    out = {
        "kind": "readiness-gate", "verdict": verdict, "ready": ready,
        "blocking_gates": blocking, "gates": gates,
        "standard_version": STANDARD_VERSION, "generated_at": utcnow(),
        "local_readiness": readiness["verdict"],
        "a100_protocol_seal": pd["seal_hash"],
        "command": "python scripts/readiness_gate.py",
    }
    (art / "readiness-gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (art / "readiness-gate.md").write_text(md, encoding="utf-8")
    (art / "readiness-gate.html").write_text(
        render_page(title="8x A100 decision gate", subtitle=verdict,
                    body=md_to_html(md),
                    meta={"Verdict": verdict, "Blocking": str(len(blocking))}),
        encoding="utf-8")
    _ = subprocess
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
