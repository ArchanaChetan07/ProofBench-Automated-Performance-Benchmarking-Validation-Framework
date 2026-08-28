"""The written synthesis: for every claim tested, whether it survived.

The proposal's fourth deliverable is a synthesis reporting the outcome for
every claim examined, *including those that survived*. A null result is a
result here: if the audited claims hold up under controlled comparison, that is
a substantive finding about the evidentiary health of the field, and the
artifacts remain useful as reference measurements.

This module reads whatever claim documents are present in the artifact
directory and writes the synthesis over them. It does not know which thrusts
exist, so a fourth thrust added later appears in the synthesis without any
change here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from losscolumn.pipeline import md_to_html
from losscolumn.report.render import _CSS
from losscolumn.validate import validate_claim
from losscolumn.version import STANDARD_VERSION, __version__


def _e(s: Any) -> str:
    import html

    return html.escape(str(s), quote=True)


def load_claims(outdir: Path) -> list[dict[str, Any]]:
    claims = []
    for p in sorted(Path(outdir).glob("*.claim.json")):
        try:
            claims.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return claims


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
        return ("survived",
                f"no region of the {n}-cell envelope showed a regression at the "
                f"pre-registered effect size")
    if frac < 0.15:
        return ("survived with exceptions",
                f"holds over {1 - frac:.0%} of the envelope; {loss} cell(s) regress")
    return ("did not survive",
            f"regresses over {frac:.0%} of the swept envelope")


_VERDICT_CLASS = {
    "survived": "pass",
    "survived with exceptions": "warn",
    "did not survive": "fail",
    "undetermined": "warn",
}


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
    L += ["", "## Loss columns, side by side", ""]
    for c in claims:
        lc = c.get("loss_column") or {}
        L += [
            f"### {c.get('title')}",
            "",
            lc.get("summary", ""),
            "",
        ]
        for i, r in enumerate(lc.get("regions", []), 1):
            worst = r.get("worst_regression_pct")
            worst_s = "cannot run" if worst in (None, float("inf")) or worst != worst or worst == float("inf") else f"{worst:+.1f}%"
            L.append(f"{i}. **{r.get('label', '')}** -- {worst_s}; {r.get('attribution', '')}")
        L.append("")
    return "\n".join(L)


def _standard_table() -> str:
    rows = [
        ("LC-1", "Loss column",
         "Selective reporting. Every claim names the regions where it underperforms, at "
         "equal prominence to the wins, with an attributed cause."),
        ("LC-2", "Tuning-budget parity",
         "The untuned baseline. An append-only ledger records trials, search procedure, "
         "operator and hardware for every system, and the certificate is derived from it."),
        ("LC-3", "Envelope, not point",
         "Overgeneralisation from one configuration. Results are surfaces; a headline "
         "must carry its own best, worst and median."),
        ("LC-4", "Pre-registration",
         "Post-hoc selection of the favourable comparison. The protocol is hashed and "
         "sealed before collection, and the analysis is verified against the seal."),
        ("LC-5", "One-command reproduction",
         "Unfalsifiability by inaccessibility. Pinned commit, pinned image, documented "
         "hardware, one command."),
    ]
    body = "".join(
        f"<tr><td><code>{r}</code></td><td><b>{n}</b></td><td>{d}</td></tr>"
        for r, n, d in rows
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Rule</th><th>Requirement</th>'
        f"<th>What it forecloses</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def synthesis_html(claims: list[dict[str, Any]]) -> str:
    n_conf = sum(1 for c in claims if validate_claim(c).ok)
    total_cells = sum(int((c.get("loss_column") or {}).get("n_cells", 0)) for c in claims)
    total_regions = sum(len((c.get("loss_column") or {}).get("regions", [])) for c in claims)

    O = [
        "<title>The Loss Column</title>",
        f"<style>{_CSS}</style>",
        '<div class="wrap">',
        '<header><div class="kicker">Research artifact &middot; machine-learning systems'
        "</div>"
        "<h1>The Loss Column</h1>"
        '<div class="headline">Establishing falsifiable performance claims in LLM training '
        "and inference systems, by making every result report the conditions under which "
        "it fails.</div>"
        f'<div class="meta"><span>standard <code>{_e(STANDARD_VERSION)}</code></span>'
        f"<span>losscolumn <code>{_e(__version__)}</code></span>"
        f"<span>{len(claims)} claim(s)</span></div></header>",
        '<div class="stat">'
        f"<div><b>{len(claims)}</b><span>claims audited</span></div>"
        f"<div><b>{total_cells}</b><span>cells measured</span></div>"
        f"<div><b>{total_regions}</b><span>loss regions</span></div>"
        f"<div><b>{n_conf}/{len(claims)}</b><span>conforming</span></div>"
        "</div>",
        "<h2>The reporting standard</h2>",
        "<p>The durable output of this project is not any single measurement but a "
        "reporting standard that the three artifacts jointly demonstrate. Each requirement "
        "targets a specific documented failure mode rather than expressing a general "
        "preference for rigour, and each is implemented as an executable conformance rule "
        "rather than as prose.</p>",
        _standard_table(),
        "<h2>Verdicts</h2>",
    ]

    rows = ""
    for c in claims:
        verdict, why = _verdict(c)
        rep = validate_claim(c)
        cls = _VERDICT_CLASS.get(verdict, "warn")
        ev = c.get("evidence_class", "?")
        ev_tag = (
            f'<span class="tag warn">{_e(ev)}</span>' if ev != "measured"
            else f'<span class="tag pass">{_e(ev)}</span>'
        )
        rows += (
            "<tr>"
            f"<td><b>{_e(c.get('title', '?'))}</b><br>"
            f"<span style='color:var(--muted)'>{_e(c.get('method'))} vs "
            f"{_e(c.get('baseline'))}</span></td>"
            f"<td>{_e(c.get('thrust'))}</td>"
            f"<td>{ev_tag}</td>"
            f'<td><span class="{cls}"><b>{_e(verdict)}</b></span></td>'
            f"<td>{_e(why)}</td>"
            f'<td><span class="{"pass" if rep.ok else "fail"}">{_e(rep.grade)}</span></td>'
            "</tr>"
        )
    O.append(
        '<div class="scroll"><table><thead><tr><th>Claim</th><th>Thrust</th>'
        "<th>Evidence</th><th>Verdict</th><th>Basis</th><th>Conformance</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )

    O.append("<h2>Loss columns</h2>")
    for c in claims:
        lc = c.get("loss_column") or {}
        O.append(f"<h3>{_e(c.get('title'))}</h3>")
        O.append(f'<div class="loss"><p>{_e(lc.get("summary", ""))}</p><ul>')
        for r in lc.get("regions", []):
            worst = r.get("worst_regression_pct")
            worst_s = (
                "cannot run"
                if worst is None or worst != worst or worst in (float("inf"),)
                else f"{worst:+.1f}%"
            )
            O.append(
                f"<li><b>{_e(r.get('label', ''))}</b> &mdash; worst {worst_s}; "
                f"{_e(r.get('attribution') or 'unattributed')}</li>"
            )
        if not lc.get("regions"):
            O.append("<li>no loss region resolved</li>")
        O.append("</ul>")
        for n in lc.get("notes", []):
            O.append(f"<p style='font-size:13px'>{_e(n)}</p>")
        O.append("</div>")

    O.append(
        "<h2>How to read a null result</h2>"
        "<p>Three of the verdicts above are distinct and are not interchangeable. "
        "<b>Survived</b> means the sweep had the power to detect a regression at the "
        "pre-registered effect size and found none. <b>Undetermined</b> means it did not "
        "have that power &mdash; an exact paired sign-flip test over <i>r</i> replicates "
        "cannot produce a p-value below 2<sup>&minus;r</sup>, so a sweep with too few "
        "replicates will report an empty loss column no matter how large the true "
        "regression is. Every loss column in this project carries the arithmetic that "
        "distinguishes the two, and the validator treats an undetectable design as a "
        "conformance failure rather than as a clean result.</p>"
    )
    O.append(
        "<footer>Generated by <code>losscolumn run all</code>. Every figure and every "
        "number on the linked artifact pages is regenerable from the pinned commit and "
        "container image recorded in each claim.</footer></div>"
    )
    return "\n".join(O)


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
