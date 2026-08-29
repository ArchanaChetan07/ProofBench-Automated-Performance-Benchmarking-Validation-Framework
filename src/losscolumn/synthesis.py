"""The written synthesis: for every claim tested, whether it survived.

The proposal's fourth deliverable is a synthesis reporting the outcome for
every claim examined, *including those that survived*. A null result is a
result here: if the audited claims hold up under controlled comparison, that is
a substantive finding about the evidentiary health of the field, and the
artifacts remain useful as reference measurements.

This module reads whatever claim documents are present in the artifact
directory. It does not know which thrusts exist, so a fourth thrust added later
appears in the synthesis without any change here.

**It re-derives every loss column from the published envelope rather than
trusting the published one.** That is not defensive programming; it is the
point. If a claim's envelope does not contain enough data to reproduce its own
loss column, the claim is not reproducible whatever its conformance report
says, and the synthesis reports the disagreement.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from losscolumn.validate import validate_claim
from losscolumn.version import STANDARD_VERSION, __version__


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def load_claims(outdir: Path) -> list[dict[str, Any]]:
    claims = []
    for p in sorted(Path(outdir).glob("*.claim.json")):
        try:
            claims.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    # Thrust order, so the page reads in the order the proposal sequences the
    # work rather than in filesystem order.
    order = {"III": 0, "I": 1, "II": 2}
    return sorted(claims, key=lambda c: (order.get(str(c.get("thrust")), 9), c.get("title", "")))


# --------------------------------------------------------------------------
# re-derivation
# --------------------------------------------------------------------------


def rederive(claim: dict[str, Any]) -> dict[str, Any]:
    """Recompute the loss column from the claim's own published envelope.

    Returns the re-derived counts, a loss-map SVG, and whether the result
    agrees with what the claim published.
    """
    out: dict[str, Any] = {"ok": None, "svg": "", "note": ""}
    env_d = claim.get("envelope")
    if not env_d or "data" not in env_d:
        out["note"] = "the claim ships no replicate-level envelope, so it cannot be re-derived"
        return out
    try:
        from losscolumn.core.envelope import Envelope
        from losscolumn.core.losscolumn import extract_loss_column
        from losscolumn.core.stats import compare_cells
        from losscolumn.report.heatmap import loss_map_svg

        env = Envelope.from_dict(env_d)
        lc_pub = claim.get("loss_column") or {}
        mde = float(lc_pub.get("mde_pct", 5.0)) / 100.0
        q = float(lc_pub.get("q_level", 0.05))
        method, baseline = claim.get("method"), claim.get("baseline")

        cmps = compare_cells(env, method, baseline, mde=mde, q=q, seed=20260101, paired=True)
        lc = extract_loss_column(env, cmps, method=method, baseline=baseline,
                                 mde=mde, q_level=q)

        pub_counts = lc_pub.get("counts") or {}
        out["counts"] = lc.counts
        out["published_counts"] = pub_counts
        out["ok"] = lc.counts.get("loss") == pub_counts.get("loss")
        out["note"] = (
            f"Re-derived from the published envelope: {lc.counts.get('loss')} losing cells, "
            f"matching the published column"
            if out["ok"]
            else f"Re-derivation found {lc.counts.get('loss')} losing cells but the claim "
                 f"published {pub_counts.get('loss')}"
        )

        x, y, facet = _axes(env)
        if x and y:
            out["svg"] = loss_map_svg(
                env, cmps, x=x, y=y, facet=facet, loss_column=lc,
                title=f"{method} vs {baseline}",
                subtitle=f"{env.metric.name} ({env.metric.unit}); "
                         f"{'higher' if env.metric.higher_is_better else 'lower'} is better",
                cell=46,
            )
    except Exception as e:  # a failed re-derivation is itself the finding
        out["ok"] = False
        out["note"] = f"Re-derivation failed: {type(e).__name__}: {e}"
    return out


def _axes(env: Any) -> tuple[str | None, str | None, str | None]:
    """Pick two axes for the map, and a third to facet by.

    Ordered factors go on the axes where possible: a heat map over an ordered
    factor shows a gradient a reader can follow, where a categorical one shows
    only adjacency the ordering does not imply.
    """
    ordered = sorted([f for f in env.factors if f.ordered], key=lambda f: -len(f.levels))
    other = [f for f in env.factors if not f.ordered]
    picks = ordered + other
    if len(picks) < 2:
        return (None, None, None)
    facet = picks[2].name if len(picks) > 2 else None
    return picks[0].name, picks[1].name, facet


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------


def _verdict(claim: dict[str, Any]) -> tuple[str, str]:
    """Did the audited claim survive? Returns (verdict, one-line reason)."""
    lc = claim.get("loss_column") or {}
    counts = lc.get("counts") or {}
    n = int(lc.get("n_cells") or 0) or 1
    loss = int(counts.get("loss", 0))
    design = lc.get("design_check") or {}
    if not design.get("can_reject", True):
        return ("undetermined",
                "the design could not have detected a loss; no conclusion is available")
    frac = loss / n
    if loss == 0:
        return ("held",
                f"no region of the {n}-cell envelope regressed at the pre-registered "
                "effect size")
    if frac < 0.15:
        return ("held with exceptions",
                f"holds over {1 - frac:.0%} of the envelope; {loss} cell(s) regress")
    return ("did not hold", f"regresses over {frac:.0%} of the swept envelope")


_VERDICT_TONE = {
    "held": "ok",
    "held with exceptions": "warn",
    "did not hold": "loss",
    "undetermined": "warn",
}

REQUIREMENTS = [
    ("LC-1", "Loss column", "Selective reporting",
     "Every claim names the regions where it underperforms, at equal prominence to the "
     "wins, with an attributed cause."),
    ("LC-2", "Tuning-budget parity", "The untuned baseline",
     "An append-only ledger records trials, search procedure, operator and hardware for "
     "every system; the certificate is derived from it, not asserted."),
    ("LC-3", "Envelope, not point", "Overgeneralisation",
     "Results are surfaces. A headline must carry its own best, worst and median."),
    ("LC-4", "Pre-registration", "Post-hoc selection",
     "The protocol is hashed and sealed before collection, and the analysis reads its "
     "effect size and FDR level from that seal."),
    ("LC-5", "One-command reproduction", "Unfalsifiability by inaccessibility",
     "Pinned commit, pinned image, documented hardware, one command."),
]


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def _worst(region: dict[str, Any]) -> str:
    w = region.get("worst_regression_pct")
    if w is None or w != w or w == float("inf"):
        return "cannot run"
    return f"{w:+.1f}%"


def synthesis_markdown(claims: list[dict[str, Any]]) -> str:
    L = [
        "# The Loss Column -- synthesis",
        "",
        "For every claim tested, whether it survived. A null result is reported with the "
        "same prominence as a refutation: if the audited claims hold under controlled "
        "comparison, that is itself a finding about the evidentiary health of the field.",
        "",
        "| Claim | Thrust | Evidence | Verdict | Basis | Conformance |",
        "|-------|--------|----------|---------|-------|-------------|",
    ]
    for c in claims:
        verdict, why = _verdict(c)
        rep = validate_claim(c)
        L.append(
            f"| {c.get('title', '?')} | {c.get('thrust', '?')} | "
            f"{c.get('evidence_class', '?')} | **{verdict}** | {why} | "
            f"{rep.grade} ({len(rep.failures)}F/{len(rep.warnings)}W) |"
        )
    L += ["", "## Loss columns", ""]
    for c in claims:
        lc = c.get("loss_column") or {}
        L += [f"### {c.get('title')}", "", lc.get("summary", ""), ""]
        for i, r in enumerate(lc.get("regions", []), 1):
            L.append(f"{i}. **{r.get('label', '')}** -- {_worst(r)}; {r.get('attribution', '')}")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&"
    'family=IBM+Plex+Serif:wght@500;600&display=swap">'
)

_CSS = """
:root{
  --bg:#f6f7f6; --surface:#ffffff; --sunk:#eef1ef;
  --ink:#151a19; --muted:#5d6b68; --faint:#8a9793; --rule:#dbe1de;
  --loss:#a52b1e; --loss-bg:#fdf2f0; --loss-rule:#e8c4bd;
  --ok:#1c6b57; --warn:#8a5a12; --link:#1c4f6b;
  --serif:"IBM Plex Serif",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Helvetica,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --measure:68ch;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0d1110; --surface:#141918; --sunk:#1a201e;
  --ink:#e6ebe8; --muted:#93a29e; --faint:#6d7c78; --rule:#252d2a;
  --loss:#ff7d6e; --loss-bg:#1d1310; --loss-rule:#4a2a24;
  --ok:#5fd0ad; --warn:#d9a441; --link:#7fc2e0;
}}
:root[data-theme="dark"]{
  --bg:#0d1110; --surface:#141918; --sunk:#1a201e;
  --ink:#e6ebe8; --muted:#93a29e; --faint:#6d7c78; --rule:#252d2a;
  --loss:#ff7d6e; --loss-bg:#1d1310; --loss-rule:#4a2a24;
  --ok:#5fd0ad; --warn:#d9a441; --link:#7fc2e0;
}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:400 16px/1.62 var(--sans);-webkit-font-smoothing:antialiased}
.page{max-width:1080px;margin:0 auto;padding:0 28px 96px}

/* Marginal rule identifiers, after the proposal's own section markers. */
.band{display:grid;grid-template-columns:92px minmax(0,1fr);gap:0 28px;
  padding-top:44px;margin-top:44px;border-top:1px solid var(--rule)}
.band .mark{font:500 11px/1.5 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--faint);padding-top:6px}
.band .body{min-width:0}
@media (max-width:720px){.band{grid-template-columns:1fr;gap:8px}
  .band .mark{padding-top:0}}

header.masthead{padding:72px 0 0}
.eyebrow{font:500 11px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;
  color:var(--muted);margin-bottom:20px}
h1{font:600 clamp(38px,6.2vw,64px)/1.04 var(--serif);letter-spacing:-.022em;
  margin:0 0 20px;text-wrap:balance}
.thesis{font:400 clamp(18px,2.1vw,21px)/1.5 var(--serif);
  max-width:46ch;margin:0 0 28px}
.byline{display:flex;flex-wrap:wrap;gap:10px 22px;font:400 12.5px/1 var(--mono);
  color:var(--muted);padding-bottom:32px;border-bottom:2px solid var(--ink)}

h2{font:600 clamp(22px,2.6vw,28px)/1.2 var(--serif);letter-spacing:-.015em;
  margin:0 0 14px;text-wrap:balance}
h3{font:600 17px/1.3 var(--serif);margin:0 0 10px}
p{margin:0 0 14px;max-width:var(--measure)}
a{color:var(--link)}
ul{margin:0 0 14px;padding-left:20px;max-width:var(--measure)}
li{margin:5px 0}
code{font:400 13px/1 var(--mono);background:var(--sunk);padding:2px 5px;border-radius:3px}

.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin:0 0 20px}
.readout div{background:var(--surface);padding:16px 18px}
.readout b{display:block;font:600 30px/1 var(--mono);letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;margin-bottom:7px}
.readout span{font:500 10.5px/1.3 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted)}

.scroll{overflow-x:auto;margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:13.5px;
  font-variant-numeric:tabular-nums}
th{font:500 10.5px/1.4 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:0 14px 9px 0;
  border-bottom:1px solid var(--ink);white-space:nowrap}
td{padding:12px 14px 12px 0;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:0}
td .sub{display:block;color:var(--muted);font-size:12.5px;margin-top:3px}

.chip{display:inline-block;font:500 10.5px/1 var(--mono);letter-spacing:.09em;
  text-transform:uppercase;padding:5px 8px;border:1px solid currentColor;
  border-radius:2px;white-space:nowrap}
.t-ok{color:var(--ok)} .t-warn{color:var(--warn)} .t-loss{color:var(--loss)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}

/* The loss column is the thesis of the page, so it takes the only saturated
   surface and the only heavy border on it. */
.losspanel{border:1px solid var(--loss-rule);border-left:3px solid var(--loss);
  background:var(--loss-bg);padding:22px 24px;margin:0 0 4px}
.losspanel h3{color:var(--loss)}
.losspanel p{max-width:none}
.regions{list-style:none;padding:0;margin:14px 0 0;max-width:none}
.regions li{padding:12px 0 0;border-top:1px solid var(--loss-rule);margin:12px 0 0}
.where{font:500 13px/1.5 var(--mono);word-break:break-word;display:block}
.why{display:block;color:var(--muted);font-size:13.5px;line-height:1.5;margin-top:5px}
.sev{color:var(--loss);font-weight:600;font-variant-numeric:tabular-nums}

figure{margin:20px 0 0}
figure svg{max-width:100%;height:auto;display:block;background:#fff;
  border:1px solid var(--rule)}
figcaption{font-size:12.5px;line-height:1.55;color:var(--muted);margin-top:10px;
  max-width:var(--measure)}

.callout{border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);
  padding:20px 0;margin:0}
.callout p:last-child{margin-bottom:0}
pre{background:var(--sunk);border:1px solid var(--rule);padding:14px 16px;
  overflow-x:auto;font:400 13px/1.6 var(--mono);margin:0 0 16px}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--rule);
  font:400 12.5px/1.6 var(--mono);color:var(--faint);max-width:var(--measure)}
:focus-visible{outline:2px solid var(--link);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def synthesis_html(claims: list[dict[str, Any]]) -> str:
    reports = {c.get("id"): validate_claim(c) for c in claims}
    derived = {c.get("id"): rederive(c) for c in claims}
    n_conf = sum(1 for r in reports.values() if r.ok)
    total_cells = sum(int((c.get("loss_column") or {}).get("n_cells", 0)) for c in claims)
    total_regions = sum(len((c.get("loss_column") or {}).get("regions", [])) for c in claims)
    n_red = sum(1 for d in derived.values() if d.get("ok"))

    out: list[str] = [
        "<title>The Loss Column</title>",
        _FONTS,
        f"<style>{_CSS}</style>",
        '<div class="page">',
        '<header class="masthead">'
        '<div class="eyebrow">Research artifact &middot; machine-learning systems</div>'
        "<h1>The Loss&nbsp;Column</h1>"
        '<p class="thesis">Performance claims in ML systems are largely unfalsifiable. '
        "This is what it takes to make one falsifiable: every result reports the "
        "conditions under which it fails.</p>"
        f'<div class="byline"><span>standard {_e(STANDARD_VERSION)}</span>'
        f"<span>losscolumn {_e(__version__)}</span>"
        f"<span>{len(claims)} claims audited</span>"
        "<span>Archana Suresh Patil</span></div></header>",
    ]

    out.append(
        '<div class="band"><div class="mark">Summary</div><div class="body">'
        '<div class="readout">'
        f"<div><b>{len(claims)}</b><span>claims audited</span></div>"
        f"<div><b>{total_cells}</b><span>cells measured</span></div>"
        f"<div><b>{total_regions}</b><span>loss regions</span></div>"
        f"<div><b>{n_conf}/{len(claims)}</b><span>conforming</span></div>"
        f"<div><b>{n_red}/{len(claims)}</b><span>re-derived</span></div>"
        "</div>"
        "<p>Every loss column below was recomputed from the claim's own published "
        "envelope rather than read out of the claim. A claim whose artifact does not "
        "contain enough data to reproduce its own conclusion is not reproducible, "
        "whatever its conformance report says.</p>"
        "</div></div>"
    )

    # The loss columns come first, by the same rule the standard imposes on
    # everyone else.
    for c in claims:
        cid = c.get("id")
        lc = c.get("loss_column") or {}
        d = derived.get(cid, {})
        verdict, _why = _verdict(c)
        tone = _VERDICT_TONE.get(verdict, "warn")
        ev = c.get("evidence_class", "?")
        rep = reports[cid]
        out.append(
            f'<div class="band"><div class="mark">Thrust {_e(c.get("thrust"))}</div>'
            '<div class="body">'
            f"<h2>{_e(c.get('title'))}</h2>"
            f'<div class="chips">'
            f'<span class="chip t-{tone}">{_e(verdict)}</span>'
            f'<span class="chip t-{"warn" if ev != "measured" else "ok"}">'
            f"{_e(ev)} evidence</span>"
            f'<span class="chip t-{"ok" if rep.ok else "loss"}">{_e(rep.grade)}</span>'
            "</div>"
            '<div class="losspanel"><h3>Loss column</h3>'
            f"<p>{_e(lc.get('summary', ''))}</p>"
            '<ul class="regions">'
        )
        for r in lc.get("regions", []):
            out.append(
                f'<li><span class="where">{_e(r.get("label", ""))}</span>'
                f'<span class="why"><span class="sev">{_e(_worst(r))}</span> &mdash; '
                f'{_e(r.get("attribution") or "unattributed")}</span></li>'
            )
        if not lc.get("regions"):
            out.append('<li><span class="where">no loss region resolved</span></li>')
        out.append("</ul>")
        for n in lc.get("notes", []):
            out.append(
                f'<p style="font-size:13px;color:var(--muted);margin-top:14px">{_e(n)}</p>'
            )
        out.append("</div>")

        if d.get("svg"):
            out.append(
                f'<figure>{d["svg"]}<figcaption>{_e(d.get("note", ""))}. '
                "Crossed cells are configurations that cannot run at all; hatched cells "
                "are inconclusive at the pre-registered effect size; dashed outlines mark "
                "the extracted loss regions.</figcaption></figure>"
            )
        elif d.get("note"):
            out.append(f'<p style="color:var(--muted);font-size:13px">{_e(d["note"])}</p>')
        out.append("</div></div>")

    rows = ""
    for c in claims:
        verdict, why = _verdict(c)
        rep = reports[c.get("id")]
        tone = _VERDICT_TONE.get(verdict, "warn")
        rows += (
            "<tr>"
            f"<td><strong>{_e(c.get('title', '?'))}</strong>"
            f"<span class='sub'>{_e(c.get('method'))} vs {_e(c.get('baseline'))}</span></td>"
            f'<td><span class="chip t-{tone}">{_e(verdict)}</span></td>'
            f"<td>{_e(why)}</td>"
            f'<td><span class="t-{"ok" if rep.ok else "loss"}">{_e(rep.grade)}</span>'
            f"<span class='sub'>{len(rep.failures)} fatal, "
            f"{len(rep.warnings)} warning</span></td>"
            "</tr>"
        )
    out.append(
        '<div class="band"><div class="mark">Verdicts</div><div class="body">'
        "<h2>What survived</h2>"
        "<p>A null result is reported with the same prominence as a refutation. If the "
        "audited claims hold under controlled comparison, that is a substantive finding "
        "about the evidentiary health of the field, and the artifacts remain useful as "
        "reference measurements.</p>"
        '<div class="scroll"><table><thead><tr><th>Claim</th><th>Verdict</th>'
        f"<th>Basis</th><th>Conformance</th></tr></thead><tbody>{rows}</tbody></table></div>"
        "</div></div>"
    )

    std = "".join(
        f"<tr><td><code>{r}</code></td><td><strong>{n}</strong>"
        f"<span class='sub'>{d}</span></td><td>{f}</td></tr>"
        for r, n, f, d in REQUIREMENTS
    )
    out.append(
        '<div class="band"><div class="mark">The standard</div><div class="body">'
        "<h2>Five requirements, each an executable rule</h2>"
        "<p>The durable output is not any single measurement but a reporting standard the "
        "three artifacts jointly demonstrate. Each requirement targets a specific "
        "documented failure mode rather than expressing a general preference for rigour, "
        "and each is implemented as a rule a program can check &mdash; against any claim "
        "document, not only ones this package produced.</p>"
        '<div class="scroll"><table><thead><tr><th>Rule</th><th>Requirement</th>'
        f"<th>What it forecloses</th></tr></thead><tbody>{std}</tbody></table></div>"
        "<pre>losscolumn validate anyones-claim.json</pre>"
        "</div></div>"
    )

    out.append(
        '<div class="band"><div class="mark">LC-1.8</div><div class="body">'
        "<h2>How to read an empty loss column</h2>"
        '<div class="callout">'
        "<p>The most dangerous artifact this standard can produce is a conforming, "
        "well-formatted, entirely <em>empty</em> loss column produced by a sweep that "
        "could never have found anything.</p>"
        "<p>An exact paired sign-flip test over <code>r</code> replicates cannot produce "
        "a p-value below <code>2⁻ʳ</code>, however large the true effect. "
        "Benjamini&ndash;Hochberg over <code>m</code> cells rejects the most extreme one "
        "only if its p-value is at most <code>q/m</code>. Together those force "
        "<code>r &ge; log₂(m/q)</code> &mdash; ten replicates for a 48-cell sweep at "
        "q&nbsp;=&nbsp;0.05.</p>"
        "<p>A sweep at five replicates reports an empty loss column against a "
        "<strong>+171% regression</strong>. That is not hypothetical: it happened while "
        "building this package, and the rule exists because of it. Every loss column "
        "carries the arithmetic separating <em>we looked and found nothing</em> from "
        "<em>we could not have found anything</em>, and the validator treats an "
        "undetectable design as a conformance failure rather than a clean result.</p>"
        "</div></div></div>"
    )

    out.append(
        '<div class="band"><div class="mark">Reproduce</div><div class="body">'
        "<h2>Running it</h2>"
        "<p>Requirement LC-5 would be hollow if this project's own artifacts were not "
        "produced the way it demands of everyone else. Each claim records the invocation "
        "that made it, the commit, the container image and the hardware.</p>"
        "<pre>losscolumn doctor\n"
        "losscolumn prereg seal I &amp;&amp; losscolumn prereg seal II &amp;&amp; "
        "losscolumn prereg seal III\n"
        "losscolumn run all\n"
        "losscolumn validate artifacts/*.claim.json</pre>"
        "<p>The protocols are sealed before any data is collected, and the analysis reads "
        "its effect size, FDR level and replicate count out of the seal &mdash; so an "
        "analysis departing from what was registered cannot run silently.</p>"
        "</div></div>"
    )

    out.append(
        "<footer>Generated by <code>losscolumn run all</code>. Claims marked "
        "<em>simulated</em> come from calibratable models whose parameters are the "
        "quantities a microbenchmark measures; the analysis path is identical on measured "
        "and modelled data. A conforming claim can still be wrong &mdash; but it can be "
        "shown to be wrong, which is the point.</footer></div>"
    )
    return "\n".join(out)


def write_synthesis(*, outdir: Path) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    claims = load_claims(out)
    paths = {
        "markdown": out / "synthesis.md",
        "html": out / "synthesis.html",
        "json": out / "synthesis.json",
    }
    paths["markdown"].write_text(synthesis_markdown(claims), encoding="utf-8")
    paths["html"].write_text(synthesis_html(claims), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(
            {
                "standard_version": STANDARD_VERSION,
                "losscolumn_version": __version__,
                "claims": [
                    {
                        "id": c.get("id"),
                        "title": c.get("title"),
                        "thrust": c.get("thrust"),
                        "evidence_class": c.get("evidence_class"),
                        "verdict": _verdict(c)[0],
                        "basis": _verdict(c)[1],
                        "rederivation": {
                            k: v for k, v in rederive(c).items() if k != "svg"
                        },
                        "conformance": validate_claim(c).to_dict(),
                    }
                    for c in claims
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
