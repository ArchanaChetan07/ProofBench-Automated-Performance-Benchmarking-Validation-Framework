"""Run the machine stability protocol: sentinel x restarts x sessions."""
import json
import sys
import time
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.core.stability import build_envelope, compare_sessions
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description
    from losscolumn.thrusts.overlap.sentinel import run_sentinel, sentinel_protocol
    from losscolumn.version import STANDARD_VERSION

    quick = "--quick" in sys.argv
    n_sessions = 2 if quick else 4
    n_restarts = 2 if quick else 3
    repeats = 5 if quick else 7
    gap_s = 20.0 if quick else 300.0

    proto = sentinel_protocol(n_sessions=n_sessions, n_restarts=n_restarts,
                              repeats=repeats, gap_s=gap_s)
    out = Path("artifacts")
    (out / "prereg").mkdir(parents=True, exist_ok=True)
    (out / "prereg" / "stability-sentinel-v1.protocol.json").write_text(
        json.dumps(proto, indent=2, default=str), encoding="utf-8")
    print(f"protocol sealed {proto['seal_hash'][:23]}")
    print(f"{n_sessions} sessions x {n_restarts} restarts x {repeats} repeats "
          f"x {len(proto['probes'])} probes, gap {gap_s:.0f}s", flush=True)

    readings = []
    errors: dict[str, str] = {}
    t_start = time.time()
    with MeasurementLock("stability sentinel"):
        for s in range(n_sessions):
            if s:
                print(f"  ... idling {gap_s:.0f}s to separate sessions", flush=True)
                time.sleep(gap_s)
            sid = f"S{s}"
            for r in range(n_restarts):
                rs, errs = run_sentinel(session_id=sid, restart_index=r,
                                        repeats=repeats,
                                        port_offset=17 * s + 3 * r)
                readings.extend(rs)
                errors.update(errs)
                ok = sum(1 for x in rs if x.valid)
                print(f"  {sid}/r{r}: {ok}/{len(rs)} valid "
                      f"({time.time() - t_start:.0f}s elapsed)", flush=True)

    env = build_envelope(readings)
    sessions = sorted({r.session_id for r in readings})
    pairs = []
    for i, a in enumerate(sessions):
        for b in sessions[i + 1:]:
            cmp = compare_sessions([r for r in readings if r.session_id == a],
                                   [r for r in readings if r.session_id == b],
                                   criterion=env.recommended_criterion)
            pairs.append(cmp)

    md = _report(env, pairs, proto, errors)
    print()
    print(md)

    payload = {
        "kind": "stability-report", "standard_version": STANDARD_VERSION,
        "generated_at": utcnow(), "protocol": proto,
        "device": device_description("cuda"),
        "envelope": env.to_dict(),
        "pairs": [p.to_dict() for p in pairs],
        "readings": [r.to_dict() for r in readings],
        "errors": errors,
    }
    (out / "stability-machine.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out / "stability-machine.md").write_text(md, encoding="utf-8")
    (out / "stability-machine.html").write_text(
        render_page(title="Machine stability",
                    subtitle="Is this machine the same machine it was an hour ago?",
                    body=md_to_html(md)), encoding="utf-8")
    print(f"\nwrote {out / 'stability-machine.json'}")
    return 0


def _report(env, pairs, proto, errors) -> str:
    L = ["# Machine stability", "",
         f"Protocol `{proto['name']}` sealed `{proto['seal_hash'][:23]}`, "
         f"registered before any measurement.", "",
         "## Repeatability", "", env.to_markdown(), "",
         "## Session comparability", "",
         "| A | B | worst probe ratio | criterion | verdict |",
         "|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p.session_a} | {p.session_b} | {p.observed_ratio:.2f}x | "
                 f"{p.criterion:.2f}x | `{p.verdict.value.upper()}` |")
    n_ok = sum(1 for p in pairs if p.poolable)
    L += ["", f"{n_ok} of {len(pairs)} session pairs are poolable.", ""]
    if errors:
        L += ["## Failures", ""] + [f"- `{k}`: {v}" for k, v in errors.items()]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
