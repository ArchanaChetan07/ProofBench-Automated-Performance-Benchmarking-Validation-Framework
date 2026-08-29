"""Assembling thrust results into conforming claims.

Each thrust produces measurements; this module turns those into the object the
standard grades -- headline, loss column, parity certificate, sealed protocol,
reproduction recipe, provenance -- and then renders and validates it.

The headline is *derived from the loss column*, not written by hand. That is
the single most useful structural property in the whole package: it is not
possible to state a summary here that the measurements do not support, because
the summary is computed from them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from losscolumn.core.envelope import Envelope
from losscolumn.core.losscolumn import LossColumn
from losscolumn.core.prereg import PreRegistration, verify
from losscolumn.core.provenance import Provenance, git_state
from losscolumn.core.stats import CellComparison
from losscolumn.report.render import Figure, render_html, render_markdown
from losscolumn.spec.claim import Claim, Headline, Reproduction
from losscolumn.validate.validator import ConformanceReport, validate_claim


def derive_headline(
    statement: str,
    env: Envelope,
    comparisons: list[CellComparison],
    lc: LossColumn,
) -> Headline:
    """Build the headline from the measurements themselves.

    ``best`` is the method's largest advantage anywhere in the envelope and
    ``worst`` its largest disadvantage, each reported with the conditions that
    produce it. Because the numbers come from the same comparison objects that
    produced the loss column, a headline cannot drift away from its evidence,
    and requirement LC-3's ban on point estimates is satisfied by construction
    rather than by editing.
    """
    finite = [c for c in comparisons if math.isfinite(c.effect)]
    if not finite:
        return Headline(statement=statement, best_pct=0.0, worst_pct=0.0, median_pct=0.0)
    # Effects are oriented so positive means the method is worse; the headline
    # speaks in the reader's direction, so the sign is flipped once, here.
    best = min(finite, key=lambda c: c.effect)
    worst = max(finite, key=lambda c: c.effect)
    med = float(np.median([c.effect for c in finite]))
    unrunnable = [c for c in comparisons if c.verdict == "loss" and not math.isfinite(c.effect)]
    conditions_worst = dict(worst.coords)
    if unrunnable:
        conditions_worst = dict(unrunnable[0].coords)
    return Headline(
        statement=statement,
        best_pct=-best.pct,
        worst_pct=-worst.pct if not unrunnable else float("-inf"),
        median_pct=-(math.exp(med) - 1) * 100,
        conditions_best=dict(best.coords),
        conditions_worst=conditions_worst,
    )


def reproduction(
    command: str,
    *,
    hardware: str,
    runtime_min: float | None = None,
    cost_usd: float | None = None,
    image: str | None = None,
    repo_path: str | None = None,
) -> Reproduction:
    g = git_state(repo_path)
    return Reproduction(
        command=command,
        repo=g.get("remote") or "local checkout",
        commit=g.get("commit"),
        dirty=g.get("dirty"),
        container_image=image,
        hardware=hardware,
        estimated_runtime_min=runtime_min,
        estimated_cost_usd=cost_usd,
    )


def assemble(
    *,
    claim_id: str,
    title: str,
    thrust: str,
    method: str,
    baseline: str,
    statement: str,
    envelope: Envelope,
    comparisons: list[CellComparison],
    loss_column: LossColumn,
    prereg: PreRegistration | None,
    repro: Reproduction,
    evidence_class: str = "measured",
    evidence_note: str = "",
    parity: Any = None,
    limitations: list[str] | None = None,
    supporting: dict[str, Any] | None = None,
    attributions: dict[str, str] | None = None,
    provenance: Provenance | None = None,
) -> Claim:
    claim = Claim(
        id=claim_id,
        title=title,
        thrust=thrust,
        method=method,
        baseline=baseline,
        headline=derive_headline(statement, envelope, comparisons, loss_column),
        loss_column=loss_column,
        reproduction=repro,
        evidence_class=evidence_class,
        evidence_note=evidence_note,
        envelope=envelope,
        prereg=prereg,
        parity=parity,
        provenance=provenance or envelope.provenance or Provenance.capture(),
        attributions=attributions or {},
        supporting={**(supporting or {}), "noise": envelope.noise(method)},
        limitations=limitations or [],
        authors=prereg.authors if prereg else [],
    )
    if prereg is not None:
        claim.prereg_verification = verify(
            prereg,
            analysis={
                "mde": loss_column.mde_pct / 100.0,
                "q_level": loss_column.q_level,
                "alpha": prereg.alpha,
                "replicates": envelope.replicates,
                "interleaved": envelope.interleaved,
                "systems": list(envelope.systems),
                "factors": [
                    {"name": f.name, "levels": list(f.levels)} for f in envelope.factors
                ],
            },
            collected_at=envelope.provenance.captured_at if envelope.provenance else None,
        ).to_dict()
    return claim


def publish(
    claim: Claim,
    outdir: str | Path,
    *,
    figures: list[Figure] | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
    stem: str | None = None,
) -> dict[str, Path]:
    """Write claim.json, the markdown artifact and the standalone HTML page."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or claim.id
    report: ConformanceReport = validate_claim(claim.to_dict(include_envelope=True))

    paths = {
        "claim": claim.save(out / f"{stem}.claim.json"),
        "markdown": out / f"{stem}.md",
        "html": out / f"{stem}.html",
        "conformance": out / f"{stem}.conformance.json",
    }
    paths["markdown"].write_text(
        render_markdown(claim, report, extra_sections=extra_sections), encoding="utf-8"
    )
    paths["html"].write_text(
        render_html(claim, report, figures=figures, extra_sections=extra_sections),
        encoding="utf-8",
    )
    import json

    paths["conformance"].write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return paths


def md_to_html(md: str) -> str:
    """Minimal Markdown-to-HTML for the table and list fragments thrusts emit.

    Deliberately not a Markdown implementation: it handles the pipe tables,
    headings, bullets and inline code that the report modules actually
    generate, and nothing else. Pulling in a full parser to render text this
    package wrote itself would be a dependency bought for no benefit.
    """

    out: list[str] = []
    rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal rows
        if not rows:
            return
        head, body = rows[0], rows[2:] if len(rows) > 2 else []
        out.append('<div class="scroll"><table><thead><tr>')
        # extend, not `out +=`: augmented assignment inside a closure rebinds
        # the name as local and shadows the enclosing list.
        out.extend(f"<th>{c}</th>" for c in head)
        out.append("</tr></thead><tbody>")
        for r in body:
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        out.append("</tbody></table></div>")
        rows = []

    in_list = False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            rows.append([_inline(c.strip()) for c in s.strip("|").split("|")])
            continue
        flush_table()
        if s.startswith("#"):
            if in_list:
                out.append("</ul>")
                in_list = False
            level = min(len(s) - len(s.lstrip("#")), 6)
            out.append(f"<h{level + 1}>{_inline(s.lstrip('# ').strip())}</h{level + 1}>")
        elif s.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(s[2:])}</li>")
        elif not s.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{_inline(s)}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(s: str) -> str:
    import html as _html
    import re

    s = _html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"_([^_]+)_", r"<i>\1</i>", s)
    return s
