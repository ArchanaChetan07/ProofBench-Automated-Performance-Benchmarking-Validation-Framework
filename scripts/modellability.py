"""Answer the Phase 4 question: is the required surface modellable here?"""
import json
from pathlib import Path


def main() -> int:
    from losscolumn.core.modellability import assess_modellability
    from losscolumn.core.provenance import utcnow
    from losscolumn.pipeline import md_to_html
    from losscolumn.report.render import render_page
    from losscolumn.version import STANDARD_VERSION

    art = Path("artifacts")
    env = json.loads((art / "stability-machine.json").read_text(encoding="utf-8"))["envelope"]
    debt = json.loads((art / "coverage-debt.json").read_text(encoding="utf-8"))

    m = assess_modellability(env, debt)
    md = _report(m)
    print(md)

    payload = {"kind": "modellability", "standard_version": STANDARD_VERSION,
               "generated_at": utcnow(), **m.to_dict()}
    (art / "modellability.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (art / "modellability.md").write_text(md, encoding="utf-8")
    (art / "modellability.html").write_text(
        render_page(title="Is the surface modellable",
                    subtitle="What the instrument can resolve, and what it cannot",
                    body=md_to_html(md)), encoding="utf-8")
    return 0


def _report(m) -> str:
    b = m.budget
    L = [f"# {m.verdict}", "", m.summary, "", "## The noise budget", "",
         "Measurement variance arrives at two levels and they behave differently "
         "under more work. One averages down; the other does not.", "",
         "| Component | Value | Behaviour under more repeats |", "|---|---|---|",
         f"| within a run | {b.within_run_cv:.1%} | falls as 1/sqrt(n) |",
         f"| across restarts, observed | {b.across_restart_cv:.1%} | n/a |",
         f"| across restarts, predicted from repeats alone | "
         f"{b.expected_restart_cv:.1%} | n/a |",
         f"| **launch-level component** | **{b.launch_cv:.1%}** | **does not fall "
         "at all inside one process** |", "",
         f"Measured from {b.n_repeats} repeats per launch and {b.n_launches} "
         "launches per probe. The launch component is what is left of the "
         "restart-level dispersion once averaging is accounted for; variances "
         "subtract, not the coefficients.", "",
         f"**The floor is {b.floor:.1%}.** A campaign that adds repeats forever "
         "converges there and stops.", "",
         "## What a target precision costs", "",
         "| Target | Repeats | Launches | Reachable |", "|---|---|---|---|"]
    for k, p in m.plans.items():
        L.append(f"| {k} | {p.repeats_needed or '—'} | {p.launches_needed} | "
                 f"{'yes' if p.reachable_at_all else '**no**'} |")
    L += ["", "Repeats first because they are much the cheaper of the two: a launch "
          "pays process startup and transport setup once for every probe it "
          "carries.", ""]
    for k, p in m.plans.items():
        if p.note:
            L.append(f"- **{k}**: {p.note}")
    L += ["", "## Cell by cell", "",
          "| Group | Regime | held-out | noise | Verdict | Why |",
          "|---|---|---|---|---|---|"]
    order = {"UNREACHABLE": 0, "MODEL_WORK": 1, "BUYABLE": 2, "UNDETERMINED": 3}
    for c in sorted(m.cells, key=lambda x: (order.get(x.verdict, 9), x.group)):
        e = f"{c.heldout_err:.1%}" if c.heldout_err == c.heldout_err else "—"
        v = f"{c.noise_cv:.1%}" if c.noise_cv == c.noise_cv else "—"
        L.append(f"| {c.group} | {c.regime} | {e} | {v} | `{c.verdict}` | "
                 f"{c.reason} |")
    counts = m.counts
    L += ["", "## The determination", "",
          ", ".join(f"**{v}** {k.lower().replace('_', ' ')}"
                    for k, v in sorted(counts.items())) + ".", "",
          "The question was never whether six of six groups could be made to pass. "
          "It is whether the operating surface this subsystem needs is one that can "
          "be modelled from this machine, and the honest answer separates three "
          "things that a single coverage percentage hides: cells where more "
          "measurement would help, cells where a better model would help, and "
          "cells where neither would because the instrument cannot resolve the "
          "error being chased."]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
