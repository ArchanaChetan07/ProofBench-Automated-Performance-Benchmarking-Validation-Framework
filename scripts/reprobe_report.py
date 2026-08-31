"""Re-report the medium probe with the session-comparability check active.

No re-measurement: the check is a pure function of the records already stored.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.thrusts.overlap.campaign import PointRecord
    from losscolumn.thrusts.overlap.medium_probe import (
        TARGET_COLLECTIVE,
        TARGET_WORLDS,
        MediumProbe,
        _medium_error,
        session_comparability,
    )

    art = Path("artifacts")
    prb = json.loads((art / "probe-thrust1-medium-regime.json").read_text(encoding="utf-8"))
    camp = json.loads((art / "campaign-thrust1-communication.json").read_text(encoding="utf-8"))

    probe = MediumProbe(
        grid=tuple(prb["protocol"]["grid"]), device=prb["device"],
        records=[PointRecord.from_dict(r) for r in prb["records"]],
        before=prb.get("before", {}), errors=prb.get("errors", {}) or {},
    )
    prior = [PointRecord.from_dict(r) for r in camp["records"]
             if r["collective"] == TARGET_COLLECTIVE and r["valid"]]
    fresh = [r for r in probe.records if r.valid]
    probe.comparability = session_comparability(prior, fresh)

    probe.verdict = (
        "INCONCLUSIVE -- THE SESSIONS ARE NOT COMPARABLE. "
        + probe.comparability.get("reason", "")
        + " The hypothesis is untested: this measurement cannot distinguish a "
        "model-limited surface from a drifting machine."
    )
    probe.notes = [
        "The probe's own method was unsound: it pooled records across two measurement "
        "sessions without checking they were comparable. The check now exists, and it "
        "fails, which is how this was found.",
        "The signature was in the result and not recognised at first: adding 32 points "
        "made all_reduce/world4's best achievable medium error RISE from 13.5% to "
        "32.5%. That is impossible on a consistent surface.",
        "The remedy is to measure the campaign grid and the denser medium grid in ONE "
        "session. Nothing here changes any campaign verdict.",
        "The first verdict is preserved unedited under artifacts/history.",
    ]
    for w in TARGET_WORLDS:
        key = f"{TARGET_COLLECTIVE}/world{w}"
        base = [r for r in fresh if r.world == w]
        samples = [s for r in base for s in [r.to_sample()] if s is not None]
        err, nseg, fam = _medium_error(samples, max_err=probe.max_err) if samples else (
            float("nan"), 0, "")
        probe.after[key] = {
            "n_medium": sum(1 for r in base if r.regime == "medium"),
            "medium_err": err, "n_segments": nseg, "family": fam,
            "note": "fresh session only; NOT pooled with the campaign",
        }

    md = probe.to_markdown()
    print(md)
    payload = probe.to_dict()
    payload.update({k: prb[k] for k in ("kind", "standard_version", "generated_at",
                                        "command") if k in prb})
    (art / "probe-thrust1-medium-regime.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (art / "probe-thrust1-medium-regime.md").write_text(md, encoding="utf-8")
    (art / "probe-thrust1-medium-regime.html").write_text(
        render_page(title="Medium-regime probe", subtitle="Inconclusive: sessions drifted.",
                    body=md_to_html(md), meta={"Device": probe.device,
                                               "Verdict": "INCONCLUSIVE"}),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
