"""Loss maps as standalone SVG.

The Thrust III deliverable is "a complete performance map with losses marked".
That map has to survive being copied into a paper, a README, an issue thread
and a slide, so it is emitted as dependency-free SVG rather than as a
matplotlib figure: no fonts to embed, no rasterisation, no import of a plotting
stack on a login node, and the text stays selectable and searchable.

Encoding, chosen so the loss is the thing your eye lands on first:

* fill      -- signed effect on a diverging ramp, red for loss, blue for win
* hatch     -- inconclusive: the data does not resolve this cell either way
* cross     -- unmeasurable (OOM, unsupported shape): the harshest loss there is
* heavy outline -- the boundary of an extracted loss region, labelled
"""

from __future__ import annotations

import html
import math
from typing import Any, Sequence

from losscolumn.core.envelope import Envelope
from losscolumn.core.losscolumn import LossColumn
from losscolumn.core.stats import CellComparison

# Diverging ramp. Endpoints are dark enough for white text and far enough apart
# to stay distinguishable in greyscale print.
_LOSS = ((253, 231, 224), (178, 24, 43))
_WIN = ((225, 238, 246), (33, 102, 172))
_TIE = (241, 241, 240)
_NEUTRAL = (250, 250, 249)


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _fill_for(c: CellComparison, vmax: float) -> tuple[str, str]:
    """Return (fill, overlay-kind) for a cell."""
    if c.verdict == "missing":
        return "#e7e5e4", "cross"
    if c.verdict == "loss" and not math.isfinite(c.effect):
        return "#67000d", "cross"
    if c.verdict == "inconclusive":
        return "#%02x%02x%02x" % _NEUTRAL, "hatch"
    if c.verdict == "tie":
        return "#%02x%02x%02x" % _TIE, ""
    t = min(abs(c.effect) / vmax, 1.0) if vmax > 0 else 0.0
    return (_lerp(*_LOSS, t) if c.effect > 0 else _lerp(*_WIN, t)), ""


def _text_color(fill: str) -> str:
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#1c1917"


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def loss_map_svg(
    env: Envelope,
    comparisons: Sequence[CellComparison],
    *,
    x: str,
    y: str,
    facet: str | None = None,
    loss_column: LossColumn | None = None,
    title: str = "",
    subtitle: str = "",
    cell: int = 54,
    vmax: float | None = None,
    show_values: bool = True,
) -> str:
    """Render the envelope as one or more heatmap panels.

    ``x``/``y`` name the factors on the axes; ``facet`` optionally splits the
    remaining dimension into side-by-side panels. Any factors beyond that are
    collapsed by taking the worst (most adverse) cell, which is the only
    defensible collapse for a loss map -- averaging would hide exactly what the
    map exists to show, and the caption says so.
    """
    by_cell = {tuple(c.cell): c for c in comparisons}
    fx, fy = env.factor(x), env.factor(y)
    ax_x, ax_y = env.axis(x), env.axis(y)
    facets: list[tuple[str, dict[str, Any]]]
    if facet:
        ff = env.factor(facet)
        facets = [(f"{facet}={lv}", {facet: lv}) for lv in ff.levels]
    else:
        facets = [("", {})]

    finite = [abs(c.effect) for c in comparisons if math.isfinite(c.effect)]
    if vmax is None:
        vmax = max(sorted(finite)[int(0.95 * (len(finite) - 1))] if finite else 0.3, 0.05)

    pad_l, pad_t = 96, 74 if title else 34
    pad_b, pad_r, gap = 62, 24, 34
    pw = pad_l + cell * len(fx.levels) + pad_r
    ph = cell * len(fy.levels)
    W = len(facets) * pw + (len(facets) - 1) * gap
    H = pad_t + ph + pad_b

    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="ui-sans-serif, -apple-system, Segoe UI, Helvetica, sans-serif" '
        f'role="img" aria-label="{_esc(title or "loss map")}">'
    )
    o.append(
        '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="6" height="6" fill="#fafaf9"/>'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#a8a29e" stroke-width="2"/></pattern></defs>'
    )
    o.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    if title:
        o.append(
            f'<text x="0" y="22" font-size="16" font-weight="650" fill="#1c1917">{_esc(title)}</text>'
        )
    if subtitle:
        o.append(f'<text x="0" y="42" font-size="12" fill="#57534e">{_esc(subtitle)}</text>')

    for pi, (flabel, fixed) in enumerate(facets):
        ox = pi * (pw + gap)
        if flabel:
            o.append(
                f'<text x="{ox + pad_l}" y="{pad_t - 12}" font-size="12" font-weight="600" '
                f'fill="#292524">{_esc(flabel)}</text>'
            )
        for iy, lvy in enumerate(fy.levels):
            gy = pad_t + (len(fy.levels) - 1 - iy) * cell
            o.append(
                f'<text x="{ox + pad_l - 8}" y="{gy + cell / 2 + 4}" font-size="11" '
                f'text-anchor="end" fill="#44403c">{_esc(lvy)}</text>'
            )
            for ix, lvx in enumerate(fx.levels):
                gx = ox + pad_l + ix * cell
                c = _pick(env, by_cell, {x: lvx, y: lvy, **fixed}, ax_x, ax_y)
                if c is None:
                    continue
                fill, overlay = _fill_for(c, vmax)
                tip = (
                    f"{env.label(tuple(c.cell))}\n"
                    f"verdict: {c.verdict}\n"
                    f"effect: {c.pct:+.1f}% (95% CI "
                    f"{(math.exp(c.ci_lo) - 1) * 100:+.1f}..{(math.exp(c.ci_hi) - 1) * 100:+.1f})\n"
                    f"q={c.q_worse:.3g}  n={c.n_method}/{c.n_baseline}"
                    if math.isfinite(c.effect)
                    else f"{env.label(tuple(c.cell))}\nverdict: {c.verdict}\n{c.reason or ''}"
                )
                o.append(
                    f'<g><title>{_esc(tip)}</title>'
                    f'<rect x="{gx}" y="{gy}" width="{cell - 1}" height="{cell - 1}" '
                    f'fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
                )
                if overlay == "hatch":
                    o.append(
                        f'<rect x="{gx}" y="{gy}" width="{cell - 1}" height="{cell - 1}" '
                        f'fill="url(#hatch)" opacity="0.85"/>'
                    )
                if overlay == "cross":
                    p = 12
                    o.append(
                        f'<path d="M{gx + p} {gy + p}L{gx + cell - 1 - p} {gy + cell - 1 - p}'
                        f'M{gx + cell - 1 - p} {gy + p}L{gx + p} {gy + cell - 1 - p}" '
                        f'stroke="#7f1d1d" stroke-width="2.5" fill="none"/>'
                    )
                elif show_values and math.isfinite(c.effect) and c.verdict != "inconclusive":
                    o.append(
                        f'<text x="{gx + cell / 2}" y="{gy + cell / 2 + 4}" font-size="10.5" '
                        f'text-anchor="middle" fill="{_text_color(fill)}">{c.pct:+.0f}</text>'
                    )
                o.append("</g>")

        if loss_column is not None:
            o += _region_outlines(env, loss_column, fx, fy, fixed, ox + pad_l, pad_t, cell)

        for ix, lvx in enumerate(fx.levels):
            gx = ox + pad_l + ix * cell
            o.append(
                f'<text x="{gx + cell / 2}" y="{pad_t + ph + 16}" font-size="11" '
                f'text-anchor="middle" fill="#44403c">{_esc(lvx)}</text>'
            )
        o.append(
            f'<text x="{ox + pad_l + (cell * len(fx.levels)) / 2}" y="{pad_t + ph + 36}" '
            f'font-size="12" font-weight="600" text-anchor="middle" fill="#1c1917">{_esc(x)}</text>'
        )
        o.append(
            f'<text transform="translate({ox + 22},{pad_t + ph / 2}) rotate(-90)" font-size="12" '
            f'font-weight="600" text-anchor="middle" fill="#1c1917">{_esc(y)}</text>'
        )

    o.append(_legend(W, H, vmax))
    o.append("</svg>")
    return "".join(o)


def _pick(env, by_cell, coords, ax_x, ax_y):
    """Cell at ``coords``; when other factors remain, take the most adverse one."""
    cands = []
    for cell in env.cells():
        cc = env.coords(cell)
        if all(cc[k] == v for k, v in coords.items()):
            c = by_cell.get(cell)
            if c is not None:
                cands.append(c)
    if not cands:
        return None
    return max(cands, key=lambda c: (c.effect if math.isfinite(c.effect) else 1e9))


def _region_outlines(env, lc: LossColumn, fx, fy, fixed, x0, y0, cell) -> list[str]:
    out: list[str] = []
    for i, r in enumerate(lc.regions, 1):
        if any(fixed.get(k) is not None and fixed[k] not in r.bounds.get(k, [fixed[k]])
               for k in fixed):
            continue
        xs = [fx.levels.index(v) for v in r.bounds.get(fx.name, fx.levels) if v in fx.levels]
        ys = [fy.levels.index(v) for v in r.bounds.get(fy.name, fy.levels) if v in fy.levels]
        if not xs or not ys:
            continue
        gx = x0 + min(xs) * cell
        gw = (max(xs) - min(xs) + 1) * cell - 1
        gy = y0 + (len(fy.levels) - 1 - max(ys)) * cell
        gh = (max(ys) - min(ys) + 1) * cell - 1
        out.append(
            f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="none" '
            f'stroke="#7f1d1d" stroke-width="2.5" stroke-dasharray="5 3"/>'
            f'<circle cx="{gx + 12}" cy="{gy + 12}" r="9" fill="#7f1d1d"/>'
            f'<text x="{gx + 12}" y="{gy + 16}" font-size="11" font-weight="700" '
            f'text-anchor="middle" fill="#ffffff">{i}</text>'
        )
    return out


def _legend(W: int, H: int, vmax: float) -> str:
    y = H - 20
    o = [f'<g transform="translate(0,{y})" font-size="10.5" fill="#44403c">']
    x = 0
    for t in [1.0, 0.6, 0.3]:
        o.append(f'<rect x="{x}" y="-9" width="15" height="12" fill="{_lerp(*_WIN, t)}"/>')
        x += 15
    o.append(f'<rect x="{x}" y="-9" width="15" height="12" fill="#{"%02x%02x%02x" % _TIE}"/>')
    x += 15
    for t in [0.3, 0.6, 1.0]:
        o.append(f'<rect x="{x}" y="-9" width="15" height="12" fill="{_lerp(*_LOSS, t)}"/>')
        x += 15
    o.append(
        f'<text x="{x + 8}" y="0">win {(math.exp(vmax) - 1) * 100:.0f}% .. tie .. '
        f'loss {(math.exp(vmax) - 1) * 100:.0f}%</text>'
    )
    x += 210
    o.append(f'<rect x="{x}" y="-9" width="15" height="12" fill="url(#hatch)" stroke="#d6d3d1"/>')
    o.append(f'<text x="{x + 20}" y="0">inconclusive</text>')
    x += 108
    o.append(
        f'<path d="M{x + 3} -7L{x + 12} 1M{x + 12} -7L{x + 3} 1" stroke="#7f1d1d" '
        f'stroke-width="2" fill="none"/><text x="{x + 20}" y="0">unmeasurable</text>'
    )
    x += 118
    o.append(
        f'<rect x="{x}" y="-9" width="15" height="12" fill="none" stroke="#7f1d1d" '
        f'stroke-width="2" stroke-dasharray="4 2"/>'
        f'<text x="{x + 20}" y="0">extracted loss region</text>'
    )
    o.append("</g>")
    return "".join(o)


def frontier_svg(
    frontiers: dict[str, Any],
    *,
    title: str = "",
    width: int = 620,
    height: int = 380,
    xlabel: str = "p99 latency (ms)",
    ylabel: str = "output tok/s",
) -> str:
    """Latency-throughput frontiers for several engines on shared axes."""
    colors = ["#1d4ed8", "#b45309", "#047857", "#7c3aed", "#be123c"]
    pad_l, pad_b, pad_t, pad_r = 66, 52, 40 if title else 16, 132
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    pts = [(p.latency_ms, p.throughput) for f in frontiers.values() for p in f.points]
    if not pts:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs) * 0.9, max(xs) * 1.05
    y0, y1 = 0.0, max(ys) * 1.08

    def px(v: float) -> float:
        return pad_l + (math.log(max(v, 1e-9)) - math.log(x0)) / (math.log(x1) - math.log(x0)) * pw

    def py(v: float) -> float:
        return pad_t + ph - (v - y0) / (y1 - y0) * ph

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-sans-serif, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        o.append(f'<text x="0" y="20" font-size="14" font-weight="650" fill="#1c1917">{_esc(title)}</text>')

    for i in range(5):
        gy = pad_t + ph * i / 4
        val = y1 - (y1 - y0) * i / 4
        o.append(
            f'<line x1="{pad_l}" y1="{gy}" x2="{pad_l + pw}" y2="{gy}" stroke="#e7e5e4"/>'
            f'<text x="{pad_l - 8}" y="{gy + 4}" font-size="10" text-anchor="end" '
            f'fill="#57534e">{val:,.0f}</text>'
        )
    dec = math.floor(math.log10(x0))
    tick = 10**dec
    while tick <= x1:
        for m in (1, 2, 5):
            v = tick * m
            if x0 <= v <= x1:
                o.append(
                    f'<line x1="{px(v)}" y1="{pad_t}" x2="{px(v)}" y2="{pad_t + ph}" '
                    f'stroke="#f5f5f4"/>'
                    f'<text x="{px(v)}" y="{pad_t + ph + 16}" font-size="10" '
                    f'text-anchor="middle" fill="#57534e">{v:g}</text>'
                )
        tick *= 10

    for i, (name, f) in enumerate(frontiers.items()):
        col = colors[i % len(colors)]
        for p in f.points:
            o.append(
                f'<circle cx="{px(p.latency_ms):.1f}" cy="{py(p.throughput):.1f}" r="2.6" '
                f'fill="{col}" opacity="0.28"><title>{_esc(p.label or p.config)}</title></circle>'
            )
        fr = sorted(f.frontier("point"), key=lambda p: p.latency_ms)
        if fr:
            d = " ".join(
                f"{'M' if k == 0 else 'L'}{px(p.latency_ms):.1f} {py(p.throughput):.1f}"
                for k, p in enumerate(fr)
            )
            o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2"/>')
            for p in fr:
                o.append(
                    f'<circle cx="{px(p.latency_ms):.1f}" cy="{py(p.throughput):.1f}" r="4" '
                    f'fill="{col}" stroke="#fff" stroke-width="1.2">'
                    f'<title>{_esc(p.label)}: {p.latency_ms:.1f} ms, {p.throughput:,.0f}</title>'
                    f'</circle>'
                )
        dflt = f.tagged("default")
        if dflt:
            o.append(
                f'<rect x="{px(dflt.latency_ms) - 4.5:.1f}" y="{py(dflt.throughput) - 4.5:.1f}" '
                f'width="9" height="9" fill="#ffffff" stroke="{col}" stroke-width="2" '
                f'transform="rotate(45 {px(dflt.latency_ms):.1f} {py(dflt.throughput):.1f})">'
                f'<title>{_esc(name)} at library defaults</title></rect>'
            )
        o.append(
            f'<rect x="{pad_l + pw + 16}" y="{pad_t + 6 + i * 20}" width="10" height="10" fill="{col}"/>'
            f'<text x="{pad_l + pw + 32}" y="{pad_t + 15 + i * 20}" font-size="11" '
            f'fill="#1c1917">{_esc(name)}</text>'
        )

    o.append(
        f'<text x="{pad_l + pw + 16}" y="{pad_t + 12 + len(frontiers) * 20 + 10}" font-size="10" '
        f'fill="#57534e">diamond = library</text>'
        f'<text x="{pad_l + pw + 16}" y="{pad_t + 12 + len(frontiers) * 20 + 23}" font-size="10" '
        f'fill="#57534e">defaults</text>'
    )
    o.append(
        f'<text x="{pad_l + pw / 2}" y="{height - 8}" font-size="11.5" font-weight="600" '
        f'text-anchor="middle" fill="#1c1917">{_esc(xlabel)}</text>'
        f'<text transform="translate(16,{pad_t + ph / 2}) rotate(-90)" font-size="11.5" '
        f'font-weight="600" text-anchor="middle" fill="#1c1917">{_esc(ylabel)}</text>'
    )
    o.append("</svg>")
    return "".join(o)
