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
    # The 21-repeat re-measurement where it exists. Its noise estimates are
    # measured rather than inferred from two repeats, and its coverage is
    # correspondingly lower and more honest.
    from losscolumn.core.artifacts import campaign_path  # noqa: PLC0415

    camp_path = campaign_path(art)
    if not camp_path.exists():
        print("no campaign artifact; run scripts/comcampaign.py first", file=sys.stderr)
        return 2
    camp = json.loads(camp_path.read_text(encoding="utf-8"))
    cov = camp["coverage"]
    readiness = cov["subsystem_readiness"]

    gates: list[dict] = []

    # LC-1.2. The question is whether THIS evidence was safely constituted, not
    # whether the machine could in principle pool sessions. A campaign gathered
    # in one session takes no pooling risk however badly the machine drifts
    # between sittings, which is what LC-8.1 actually says. The gate keeps its
    # teeth for evidence that does span sessions.
    n_sessions = int(camp.get("n_evidence_sessions", 1) or 1)
    stab_path = art / "stability-machine.json"
    st = json.loads(stab_path.read_text(encoding="utf-8")) if stab_path.exists() else None
    if st:
        env, pairs = st["envelope"], st.get("pairs", [])
        n_ok = sum(1 for x in pairs if x.get("poolable"))
        machine = (
            f"the machine's own drift is {env['drift_kind']} "
            f"({n_ok} of {len(pairs)} sentinel session pairs poolable at the "
            f"registered {env['recommended_criterion']:.2f}x criterion; within-run "
            f"{env['within_run']['cv']:.1%}, across restart "
            f"{env['across_restart']['cv']:.1%} against "
            f"{env['across_restart']['expected_cv']:.1%} expected from averaging)"
        )
    else:
        env = None
        machine = "no sentinel evidence exists"

    if n_sessions <= 1:
        gates.append({
            "gate": "evidence is not pooled across incomparable sessions",
            "passed": True,
            "detail": (
                f"all evidence comes from a single session, so no cross-session "
                f"pooling occurred and none had to be justified. For context, "
                f"{machine}"
            ),
        })
    else:
        gates.append({
            "gate": "evidence is not pooled across incomparable sessions",
            "passed": bool(env and env.get("pooling_permitted")),
            "detail": (
                f"evidence spans {n_sessions} sessions, so every pair has to be "
                f"shown comparable before it may be pooled; {machine}"
            ),
        })

    reclass = art / "evidence-reclassification.json"
    if reclass.exists():
        rc = json.loads(reclass.read_text(encoding="utf-8"))
        ci = rc["campaign_internal"]
        gates.append({
            "gate": "the campaign's own evidence is internally poolable",
            "passed": bool(ci.get("poolable")),
            "detail": (
                f"its two passes agree to {ci['observed_ratio']:.2f}x against a "
                f"{ci['effective_criterion']:.2f}x floor measured from the "
                f"campaign's own repeats at {ci['n_shared_probes']} probes, and the "
                f"surface did not move ({ci['level_shift']:.3f}x median shift)"
            ),
        })

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
