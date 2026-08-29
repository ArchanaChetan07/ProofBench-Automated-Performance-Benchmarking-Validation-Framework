"""Artifact rendering: a claim becomes a document a reader can attack.

Two outputs, from one claim object:

``render_markdown``  for the repository and the paper appendix.
``render_html``      a single self-contained file -- no CDN, no external CSS,
                     SVG inlined -- so the artifact survives being emailed,
                     archived, or opened offline in five years.

The ordering of sections is normative, not cosmetic. The loss column sits
immediately under the headline, above the wins, above the method. Requirement
LC-1 says "equal prominence"; in practice equal prominence means the reader
cannot reach the good news without passing the bad news first.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

from losscolumn.spec.claim import Claim
from losscolumn.validate.validator import ConformanceReport


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


@dataclass
class Figure:
    """An inline SVG with a caption that carries the interpretation."""

    svg: str
    caption: str = ""
    anchor: str = ""


def render_markdown(
    claim: Claim,
    conformance: ConformanceReport | None = None,
    *,
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    lc = claim.loss_column
    L: list[str] = []
    L += [
        f"# {claim.title}",
        "",
    ]
    if claim.evidence_class != "measured":
        L += [
            f"> **{claim.evidence_class.upper()} EVIDENCE.** {claim.evidence_note}",
            "",
        ]
    L += [
        f"> {claim.headline.render()}",
        "",
        f"`{claim.id}` -- thrust {claim.thrust} -- {claim.method} vs {claim.baseline} -- "
        f"standard {claim.standard_version}",
        "",
        "## Loss column",
        "",
        "_Where this result does not hold. Published above the wins, by requirement LC-1._",
        "",
        lc.to_markdown(),
        "",
    ]

    if claim.attributions:
        L += ["### Attributed causes", ""]
        L += [f"- **{k}** -- {v}" for k, v in claim.attributions.items()]
        L += [""]

    if claim.limitations:
        L += ["## Where this fails as a study", ""]
        L += [f"- {x}" for x in claim.limitations]
        L += [""]

    for name, body in extra_sections or []:
        L += [f"## {name}", "", body, ""]

    if claim.parity is not None:
        L += ["## Tuning-budget parity", "", claim.parity.to_markdown(), ""]

    if claim.prereg is not None:
        pv = claim.prereg_verification or {}
        L += [
            "## Pre-registration",
            "",
            f"- Sealed: `{claim.prereg.sealed_at}`",
            f"- Seal hash: `{claim.prereg.seal_hash}`",
            f"- Verification: **{'PASS' if pv.get('ok') else 'FAIL'}**"
            f" ({len(pv.get('findings', []))} finding(s))",
            "",
        ]
        for f in pv.get("findings", []):
            L.append(f"  - `{f['code']}` ({f['severity']}): {f['message']}")
        L += [""]

    r = claim.reproduction
    L += [
        "## Reproduction",
        "",
        "```bash",
        r.command,
        "```",
        "",
        f"- Repository: `{r.repo or '-'}` at commit `{r.commit or '-'}`"
        f"{' (**dirty tree**)' if r.dirty else ''}",
        f"- Image: `{r.container_image or 'not pinned'}`",
        f"- Hardware: {r.hardware or 'undocumented'}",
        f"- Estimated runtime: "
        f"{f'{r.estimated_runtime_min:.0f} min' if r.estimated_runtime_min else 'unstated'}"
        + (f" (~${r.estimated_cost_usd:.2f})" if r.estimated_cost_usd else ""),
        "",
    ]

    if conformance is not None:
        L += ["## Conformance", "", conformance.to_markdown(), ""]

    if claim.provenance is not None:
        p = claim.provenance
        L += [
            "## Provenance",
            "",
            f"- Captured: `{p.captured_at}` on `{p.platform}`, Python {p.python}",
            f"- Hardware fingerprint: `{p.hardware_fingerprint()}` ({p.describe_hardware()})",
            "- Packages: "
            + ", ".join(f"`{k}={v}`" for k, v in p.packages.items() if v),
            "",
        ]
    return "\n".join(L)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root{--bg:#ffffff;--fg:#1c1917;--muted:#57534e;--line:#e7e5e4;--panel:#fafaf9;
--loss:#b91c1c;--loss-bg:#fef2f2;--win:#1d4ed8;--warn:#b45309;--warn-bg:#fffbeb;
--ok:#047857;--code:#f5f5f4;}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--line:#292524;--panel:#1c1917;
--loss:#f87171;--loss-bg:#1f1315;--win:#93c5fd;--warn:#fbbf24;--warn-bg:#1c1710;
--ok:#6ee7b7;--code:#1c1917;}}
:root[data-theme="dark"]{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--line:#292524;
--panel:#1c1917;--loss:#f87171;--loss-bg:#1f1315;--win:#93c5fd;--warn:#fbbf24;
--warn-bg:#1c1710;--ok:#6ee7b7;--code:#1c1917;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:0 20px 80px;
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,sans-serif;}
.wrap{max-width:900px;margin:0 auto}
header{border-bottom:2px solid var(--fg);padding:44px 0 18px;margin-bottom:26px}
.kicker{font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;
text-transform:uppercase;color:var(--muted);margin-bottom:12px}
h1{font-size:30px;line-height:1.18;margin:0 0 12px;letter-spacing:-.015em}
h2{font-size:19px;margin:38px 0 12px;padding-top:14px;border-top:1px solid var(--line);
letter-spacing:-.01em}
h3{font-size:15px;margin:24px 0 8px}
.headline{font-size:17px;line-height:1.5;color:var(--fg);border-left:3px solid var(--fg);
padding:2px 0 2px 16px;margin:18px 0}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:12.5px;color:var(--muted);margin-top:14px}
.meta code{background:var(--code);padding:1px 5px;border-radius:3px;font-size:12px}
.banner{background:var(--warn-bg);border:1px solid var(--warn);border-radius:6px;
padding:12px 16px;margin:16px 0;font-size:13.5px;line-height:1.5}
.banner b{letter-spacing:.06em;font-size:11.5px}
.loss{background:var(--loss-bg);border:1px solid var(--loss);border-radius:6px;
padding:18px 20px;margin:8px 0 4px}
.loss h2{border:0;margin-top:0;padding-top:0;color:var(--loss)}
.tag{display:inline-block;font:600 10.5px/1 ui-monospace,monospace;letter-spacing:.08em;
text-transform:uppercase;padding:4px 8px;border-radius:4px;border:1px solid currentColor}
.pass{color:var(--ok)}.fail{color:var(--loss)}.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{font-weight:650;font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;
color:var(--muted)}
tr:hover td{background:var(--panel)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:12px 14px;border-radius:6px;overflow-x:auto;font-size:12.5px}
code{font-size:12.5px}
figure{margin:20px 0;padding:0}
figure svg{max-width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line);
border-radius:6px}
figcaption{font-size:12.5px;color:var(--muted);margin-top:8px;line-height:1.5}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
ul{padding-left:20px}li{margin:4px 0}
.stat{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0}
.stat div{flex:1 1 130px;border:1px solid var(--line);border-radius:6px;padding:10px 12px;
background:var(--panel)}
.stat b{display:block;font-size:21px;letter-spacing:-.02em}
.stat span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);font-size:12px;
color:var(--muted)}
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{_e(x)}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(
    claim: Claim,
    conformance: ConformanceReport | None = None,
    *,
    figures: list[Figure] | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
    embed_json: bool = True,
) -> str:
    lc = claim.loss_column
    out: list[str] = [f"<title>{_e(claim.title)}</title>", f"<style>{_CSS}</style>", '<div class="wrap">']

    banner = ""
    if claim.evidence_class != "measured":
        banner = (
            f'<div class="banner"><b>{_e(claim.evidence_class.upper())} EVIDENCE</b> '
            f"&mdash; {_e(claim.evidence_note)}</div>"
        )
    out.append(
        f'<header><div class="kicker">Thrust {_e(claim.thrust)} &middot; conforming claim '
        f'&middot; {_e(claim.standard_version)}</div>'
        f"<h1>{_e(claim.title)}</h1>"
        f"{banner}"
        f'<div class="headline">{_e(claim.headline.render())}</div>'
        f'<div class="meta"><span><code>{_e(claim.id)}</code></span>'
        f"<span>{_e(claim.method)} vs {_e(claim.baseline)}</span>"
        f"<span>{_e(claim.created_at)}</span>"
        + (
            f'<span class="tag {"pass" if conformance.ok else "fail"}">'
            f"{_e(conformance.grade)}</span>"
            if conformance
            else ""
        )
        + "</div></header>"
    )

    out.append(
        '<div class="stat">'
        f"<div><b>{lc.loss_fraction:.0%}</b><span>of envelope lost</span></div>"
        f"<div><b>{lc.worst_regression_pct:+.0f}%</b><span>worst regression</span></div>"
        f"<div><b>{len(lc.regions)}</b><span>loss regions</span></div>"
        f"<div><b>{lc.inconclusive_fraction:.0%}</b><span>inconclusive</span></div>"
        f"<div><b>{lc.n_cells}</b><span>cells measured</span></div>"
        "</div>"
    )

    # LC-1: the loss column comes first.
    rows = []
    levels = {k: tuple(v) for k, v in lc.factor_levels.items()}
    for i, r in enumerate(lc.regions, 1):
        rows.append(
            [
                f"<b>{i}</b>",
                _e(r.describe(all_levels=levels)),
                f"{r.n_loss}/{r.n_cells} ({r.frac_of_envelope:.0%})",
                f'<span style="color:var(--loss)">{r.median_regression_pct:+.1f}%</span>',
                f'<span style="color:var(--loss)"><b>{r.worst_regression_pct:+.1f}%</b></span>',
                f"{r.max_q:.3g}",
                _e(r.attribution or "not yet attributed"),
            ]
        )
    if not rows:
        rows = [["&ndash;", "<i>no loss region resolved</i>", "&ndash;", "&ndash;", "&ndash;",
                 "&ndash;", "&ndash;"]]
    out.append(
        '<div class="loss"><h2>Loss column</h2>'
        f"<p>{_e(lc.summary_sentence())}</p>"
        + _table(
            ["#", "Where it loses", "Cells", "Median", "Worst", "max q", "Attributed to"], rows
        )
        + "<ul>"
        + "".join(f"<li>{_e(n)}</li>" for n in lc.notes)
        + f"<li>Decision rule: regression &ge; {lc.mde_pct:.1f}% (pre-registered MDE) with "
        f"Benjamini&ndash;Hochberg q &le; {lc.q_level:g}. "
        f"Verdicts: {lc.counts.get('loss', 0)} loss, {lc.counts.get('win', 0)} win, "
        f"{lc.counts.get('tie', 0)} tie, {lc.counts.get('inconclusive', 0)} inconclusive, "
        f"{lc.counts.get('missing', 0)} unmeasurable.</li></ul></div>"
    )

    for fig in figures or []:
        out.append(
            f'<figure{f" id={fig.anchor}" if fig.anchor else ""}>{fig.svg}'
            f"<figcaption>{fig.caption}</figcaption></figure>"
        )

    if claim.attributions:
        out.append("<h2>Attributed causes</h2><ul>")
        out += [f"<li><b>{_e(k)}</b> &mdash; {_e(v)}</li>" for k, v in claim.attributions.items()]
        out.append("</ul>")

    for name, body in extra_sections or []:
        out.append(f"<h2>{_e(name)}</h2>{body}")

    if claim.limitations:
        out.append("<h2>Where this fails as a study</h2><ul>")
        out += [f"<li>{_e(x)}</li>" for x in claim.limitations]
        out.append("</ul>")

    if claim.parity is not None:
        p = claim.parity
        out.append(
            f'<h2>Tuning-budget parity <span class="tag {"pass" if p.ok else "fail"}">'
            f'{"parity held" if p.ok else "violated"}</span></h2>'
        )
        out.append(
            _table(
                ["System", "Trials", "ok", "Space", "Trials/dim", "Coverage", "Wall-clock",
                 "Budget binding?"],
                [
                    [
                        _e(led.system),
                        str(led.n_trials),
                        str(led.n_ok),
                        f"{led.space.n_dims}d / {led.space.cardinality}",
                        f"{led.trials_per_dim:.1f}",
                        f"{led.space_coverage:.1%}",
                        f"{led.wall_time_s / 60:.0f} min",
                        '<span style="color:var(--warn)">yes</span>'
                        if led.still_improving()
                        else "no",
                    ]
                    for led in p.ledgers
                ],
            )
        )
        if p.violations:
            out.append(
                _table(
                    ["Severity", "Code", "Finding"],
                    [
                        [
                            f'<span class="{"fail" if v.severity == "fatal" else "warn"}">'
                            f"{_e(v.severity)}</span>",
                            f"<code>{_e(v.code)}</code>",
                            _e(v.message),
                        ]
                        for v in p.violations
                    ],
                )
            )

    if claim.prereg is not None:
        pv = claim.prereg_verification or {}
        out.append(
            f'<h2>Pre-registration <span class="tag {"pass" if pv.get("ok") else "fail"}">'
            f'{"verified" if pv.get("ok") else "unverified"}</span></h2>'
            f"<ul><li>Sealed <code>{_e(claim.prereg.sealed_at)}</code></li>"
            f"<li>Seal hash <code>{_e(claim.prereg.seal_hash)}</code></li>"
            f"<li>Anchor <code>{_e(claim.prereg.anchor or 'none')}</code></li></ul>"
        )
        if pv.get("findings"):
            out.append(
                _table(
                    ["Code", "Severity", "Finding"],
                    [
                        [f"<code>{_e(f['code'])}</code>", _e(f["severity"]), _e(f["message"])]
                        for f in pv["findings"]
                    ],
                )
            )

    r = claim.reproduction
    out.append(
        "<h2>Reproduction</h2>"
        f"<pre>{_e(r.command)}</pre>"
        + _table(
            ["Field", "Value"],
            [
                ["Repository", f"<code>{_e(r.repo or '-')}</code>"],
                [
                    "Commit",
                    f"<code>{_e(r.commit or '-')}</code>"
                    + (' <span class="fail">dirty tree</span>' if r.dirty else ""),
                ],
                ["Image", f"<code>{_e(r.container_image or 'not pinned')}</code>"],
                ["Hardware", _e(r.hardware or "undocumented")],
                [
                    "Estimated cost",
                    f"{r.estimated_runtime_min:.0f} min"
                    if r.estimated_runtime_min
                    else "unstated",
                ],
            ],
        )
    )

    if conformance is not None:
        out.append(
            f'<h2>Conformance <span class="tag {"pass" if conformance.ok else "fail"}">'
            f"{_e(conformance.grade)}</span></h2>"
        )
        out.append(
            _table(
                ["Rule", "Requirement", "Result", "Finding"],
                [
                    [
                        f"<code>{_e(f.rule)}</code>",
                        _e(f.requirement),
                        '<span class="pass">pass</span>'
                        if f.passed
                        else (
                            '<span class="fail">FAIL</span>'
                            if f.severity == "fatal"
                            else '<span class="warn">warn</span>'
                        ),
                        _e(f.message),
                    ]
                    for f in conformance.findings
                ],
            )
        )

    if claim.provenance is not None:
        p = claim.provenance
        out.append(
            "<h2>Provenance</h2>"
            + _table(
                ["Field", "Value"],
                [
                    ["Captured", f"<code>{_e(p.captured_at)}</code>"],
                    ["Platform", _e(p.platform)],
                    ["Hardware", f"{_e(p.describe_hardware())}"],
                    ["Fingerprint", f"<code>{_e(p.hardware_fingerprint())}</code>"],
                    [
                        "Packages",
                        ", ".join(
                            f"<code>{_e(k)}={_e(v)}</code>" for k, v in p.packages.items() if v
                        ),
                    ],
                ],
            )
        )

    if embed_json:
        payload = json.dumps(claim.to_dict(include_envelope=False), indent=1, default=str)
        out.append(
            '<h2>Machine-readable claim</h2><p>The same document the validator grades. '
            "A reader who disagrees with the analysis can re-run it from this.</p>"
            f'<details><summary>claim.json ({len(payload) // 1024} KB)</summary>'
            f"<pre>{_e(payload[:120000])}</pre></details>"
        )

    out.append(
        f'<footer>Generated by <code>losscolumn</code> against standard '
        f"{_e(claim.standard_version)}. Every number on this page is reproducible from the "
        f"command above; if it is not, that is a bug and a conformance failure.</footer></div>"
    )
    return "\n".join(out)


def render_page(
    *,
    title: str,
    subtitle: str = "",
    body: str,
    meta: dict[str, str] | None = None,
    banner: str = "",
) -> str:
    """A standalone page for a document that is not a claim.

    A calibration study is evidence *about* a claim rather than a claim, and
    giving it the claim document type would invite it to be cited as a result
    of its own. It gets the same stylesheet and none of the conformance
    furniture.
    """
    out: list[str] = [f"<title>{_e(title)}</title>", f"<style>{_CSS}</style>",
                      '<div class="wrap">']
    if banner:
        out.append(f'<div class="banner">{banner}</div>')
    out.append(f"<h1>{_e(title)}</h1>")
    if subtitle:
        out.append(f'<p class="lede">{_e(subtitle)}</p>')
    if meta:
        cells = "".join(
            f"<span><b>{_e(k)}</b> <code>{_e(v)}</code></span>" for k, v in meta.items()
        )
        out.append(f'<div class="meta">{cells}</div>')
    out.append(body)
    out.append("</div>")
    return "\n".join(out)
