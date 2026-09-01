"""Re-evaluate the existing communication evidence under the LC-1.2 stability rules.

The campaign's two passes ran back to back inside one process invocation and one
measurement lock, so they are *probably* one machine state. Probably is not a
standard. This tests them the same way the sentinel tests anything else, and
lets the answer come out either way -- the instruction not to invalidate the
campaign automatically is not an instruction to exonerate it automatically.
"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.provenance import utcnow
    from losscolumn.core.stability import (
        ComparabilityVerdict,
        SentinelReading,
        SessionState,
        compare_sessions,
    )
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.version import STANDARD_VERSION

    out = Path("artifacts")
    stab = json.loads((out / "stability-machine.json").read_text(encoding="utf-8"))
    criterion = float(stab["envelope"]["recommended_criterion"])
    drift_kind = stab["envelope"]["drift_kind"]
    camp = json.loads(
        (out / "campaign-thrust1-communication.json").read_text(encoding="utf-8"))

    # The campaign's passes become sessions for the purpose of this test. That is
    # the strongest form of the question: if two back-to-back passes disagree,
    # nothing measured further apart can be trusted.
    def as_readings(pass_index: int) -> list[SentinelReading]:
        rs = []
        for r in camp["records"]:
            if r["pass_index"] != pass_index or not r["valid"]:
                continue
            rs.append(SentinelReading(
                session_id=f"campaign-pass{pass_index}", restart_index=pass_index,
                collective=r["collective"], world=r["world"], nbytes=r["nbytes"],
                size_class=r.get("regime", ""), timings_s=list(r["timings_s"]),
                state=SessionState(session_id=f"campaign-pass{pass_index}"),
            ))
        return rs

    # The internal null matters here and not in the sentinel: 276 shared probes
    # at 2 repeats each, against the sentinel's 6 probes at 7. A criterion that
    # bounds one comparison of well-averaged medians says nothing useful about
    # the worst of 276 comparisons of medians-of-two.
    p0, p1 = as_readings(0), as_readings(1)
    within = compare_sessions(p0, p1, criterion=criterion, use_internal_null=True)

    # And the cross-session comparison that actually failed: campaign against the
    # medium probe, measured hours later.
    probe_path = out / "probe-thrust1-medium-regime.json"
    cross = None
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        prs = []
        for r in probe.get("records", []):
            if not r.get("valid"):
                continue
            prs.append(SentinelReading(
                session_id="medium-probe", restart_index=0,
                collective=r["collective"], world=r["world"], nbytes=r["nbytes"],
                size_class=r.get("regime", ""), timings_s=list(r.get("timings_s", [])),
                state=SessionState(session_id="medium-probe"),
            ))
        if prs:
            cross = compare_sessions(p0 + p1, prs, criterion=criterion,
                                     use_internal_null=True)

    md = _report(within, cross, criterion, drift_kind)
    print(md)

    payload = {
        "kind": "evidence-reclassification", "standard_version": STANDARD_VERSION,
        "generated_at": utcnow(), "criterion": criterion, "drift_kind": drift_kind,
        "campaign_internal": within.to_dict(),
        "campaign_vs_probe": cross.to_dict() if cross else None,
        "campaign_evidence_class": (
            "single-session" if within.verdict is ComparabilityVerdict.COMPARABLE
            else "cross-session, not poolable"),
    }
    (out / "evidence-reclassification.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out / "evidence-reclassification.md").write_text(md, encoding="utf-8")
    (out / "evidence-reclassification.html").write_text(
        render_page(title="Evidence reclassification",
                    subtitle="The existing communication evidence, re-read under LC-1.2",
                    body=md_to_html(md)), encoding="utf-8")
    return 0


def _report(within, cross, criterion, drift_kind) -> str:
    L = [
        "# Evidence reclassification under LC-1.2", "",
        f"Criterion **{criterion:.2f}x**, derived from this machine's own "
        f"restart-level repeatability. Drift classified `{drift_kind}`.", "",
        "## The campaign's two passes", "",
        "They ran back to back inside one process invocation and one measurement "
        "lock, which makes them *likely* to share a machine state. Tested rather "
        "than assumed:", "",
        f"- worst shared probe: **{within.observed_ratio:.2f}x**, against a noise "
        f"floor of {within.effective_criterion:.2f}x measured from the campaign's "
        "own repeats at this probe count",
        f"- shape divergence: **{within.shape_divergence:.2f}x**, against "
        f"{within.effective_shape_criterion:.2f}x",
        f"- shared probes: {within.n_shared_probes}",
        f"- verdict: **{within.verdict.value.upper()}**", "",
        within.reason, "",
    ]
    if within.poolable:
        L += [
            "**The campaign stands.** Its two passes differ by less than two "
            "repeats within a single launch do, so whatever separates them is the "
            "instrument and not the machine. Its pooled analysis was legitimate "
            "and nothing in it needs withdrawing.", "",
            "This is a result rather than a reprieve, and it was nearly the "
            "opposite one. Judged against the registered 1.48x criterion the "
            "passes read as clearly incomparable at 2.90x -- but that criterion "
            "bounds *one* comparison of medians-of-seven, and it was being applied "
            "to the worst of 276 comparisons of medians-of-two. Splitting the "
            "campaign's own repeats within a single pass, which is the same "
            "machine state by construction, reaches 4.91x. The passes are more "
            "alike than the instrument is with itself.", "",
        ]
    else:
        L += [
            "**The campaign's own passes are not poolable.** Every pooled figure "
            "derived from it is therefore in question, including the coverage "
            "matrix and the model selection.", "",
        ]
    if cross is not None:
        L += [
            "## The campaign against the medium probe", "",
            f"- worst shared probe: **{cross.observed_ratio:.2f}x**",
            f"- shape divergence: **{cross.shape_divergence:.2f}x**",
            f"- shared probes: {cross.n_shared_probes}",
            f"- verdict: **{cross.verdict.value.upper()}**", "",
            cross.reason, "",
            "This is the pooling that produced the impossible result. Under LC-8.1 "
            "it is now refused at the point where it would happen, rather than "
            "discovered afterwards from a number that could not be true.", "",
        ]
    L += [
        "## What this does and does not license", "",
        "| | |", "|---|---|",
        "| Campaign-internal comparisons | "
        + ("permitted" if within.poolable else "**refused**") + " |",
        "| Campaign pooled with any later session | **refused** until that session "
        "is shown comparable |",
        "| Local gloo parameters as A100 fabric parameters | **refused** "
        "unconditionally, and not for a stability reason: a CPU-side gloo "
        "transport on one host is not an NVLink or InfiniBand fabric, and no "
        "amount of stability makes it one |",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
