"""Re-analyse the measured sentinel readings under the corrected classifier.

No new measurement. The readings are the readings; what changed is that the
workload-specificity test now has to clear a permutation null instead of a
fixed ratio, and that the taxonomy has a name for variation that enters at the
process launch.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.core.stability import (
        SentinelReading,
        build_envelope,
        compare_sessions,
    )
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.version import STANDARD_VERSION

    out = Path("artifacts")
    old = json.loads((out / "stability-machine.json").read_text(encoding="utf-8"))
    readings = [SentinelReading.from_dict(d) for d in old["readings"]]

    env = build_envelope(readings)
    sessions = sorted({r.session_id for r in readings})
    pairs = [compare_sessions([r for r in readings if r.session_id == a],
                              [r for r in readings if r.session_id == b],
                              criterion=env.recommended_criterion)
             for i, a in enumerate(sessions) for b in sessions[i + 1:]]

    md = _report(env, pairs, old)
    print(md)
    payload = dict(old)
    payload.update({
        "generated_at": utcnow(), "standard_version": STANDARD_VERSION,
        "envelope": env.to_dict(), "pairs": [p.to_dict() for p in pairs],
        # Idempotent: re-running must not overwrite the record of what the FIRST
        # analysis said with what the second one did.
        "supersedes": old.get("supersedes") or {
            "drift_kind": old["envelope"]["drift_kind"],
            "reason": "workload-specificity was claimed from a ratio that chance "
                      "reproduces 40% of the time",
            "withdrawal": "artifacts/history/2026-08-31-unsound-drift-classification",
        },
    })
    (out / "stability-machine.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out / "stability-machine.md").write_text(md, encoding="utf-8")
    (out / "stability-machine.html").write_text(
        render_page(title="Machine stability",
                    subtitle="Is this machine the same machine it was an hour ago?",
                    body=md_to_html(md)), encoding="utf-8")
    return 0


def _report(env, pairs, old) -> str:
    L = ["# Machine stability", "",
         f"Protocol `{old['protocol']['name']}` sealed "
         f"`{old['protocol']['seal_hash'][:23]}`, registered before measurement. "
         f"{env.n_readings} readings over {env.n_sessions} sessions and "
         f"{env.n_restarts} process launches.", "",
         "## Repeatability", "", env.to_markdown(), "",
         "## Session comparability", "",
         "Three gates, failing on different things: the whole surface moving "
         "(`level`), one probe moving (`magnitude`), the probes moving against "
         "each other (`shape`).", "",
         "| A | B | level | worst probe | shape | criterion | failed | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p.session_a} | {p.session_b} | {p.level_shift:.2f}x | "
                 f"{p.observed_ratio:.2f}x | {p.shape_divergence:.2f}x | "
                 f"{p.criterion:.2f}x | {p.failed_gate or '-'} | "
                 f"`{p.verdict.value.upper()}` |")
    n_ok = sum(1 for p in pairs if p.poolable)
    L += ["", f"**{n_ok} of {len(pairs)} session pairs are poolable.**", "",
          "### A widening that was available and was not taken", "",
          "The internal null used to correct the campaign's comparison is also "
          "computable here, from the sentinel's own repeats: it runs 1.17x to "
          "1.43x on magnitude and up to 1.90x on shape. Applying it would move "
          "S1-S2 to COMPARABLE and make this 1 of 6 rather than 0 of 6.", "",
          "It is not applied, and the reason is not a technical one. The sealed "
          "protocol registered a criterion of the form `max(1 + 3 * "
          "across_restart_CV, 1.05)` applied to the worst shared probe, and the "
          "sentinel is the design that criterion was registered for -- six probes, "
          "seven repeats, exactly as specified. Widening it now with a statistic "
          "the protocol does not name would be the post-hoc loosening LC-8.2 "
          "exists to forbid, and it would be done knowing which pair it admits.", "",
          "The campaign is a different case and not a special one: 276 probes at "
          "two repeats is not the design the criterion was registered for, so "
          "carrying the number across was a category error rather than a "
          "loosening. Correcting a threshold that never applied is not the same "
          "act as relaxing one that does.", "",
          "## What was corrected", "",
          f"The first analysis of these same readings called the drift "
          f"`{(old.get('supersedes') or {}).get('drift_kind', old['envelope']['drift_kind'])}` on a {env.workload_spread:.1f}x spread "
          "between the per-probe session variabilities. A permutation test on the "
          f"readings shows chance alone reproduces a spread that large "
          f"{env.workload_spread_p:.0%} of the time: each per-probe figure rests on "
          "four session medians, and the ratio between the largest and smallest of "
          "six such estimates is naturally enormous. The threshold was measuring "
          "the sample size, not the machine.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
