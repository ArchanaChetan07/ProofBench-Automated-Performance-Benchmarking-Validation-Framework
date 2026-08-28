"""Timeline strips for communication/computation attribution (Thrust I).

One training step, drawn as three lanes -- compute, communication, and the
exposed remainder -- so that "overlap efficiency = 0.62" is accompanied by the
picture that produced it. The exposed lane is the point of the figure: it is
the wall-clock the collective failed to hide, and it is what a sharding change
actually buys or costs.
"""

from __future__ import annotations

import html
from typing import Any

from losscolumn.core.intervals import IntervalSet


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def step_timeline_svg(
    compute: IntervalSet,
    comm: IntervalSet,
    *,
    step_start: float,
    step_end: float,
    title: str = "",
    width: int = 720,
    lane_h: int = 26,
    labels: dict[str, str] | None = None,
) -> str:
    """Draw one step. Times are in milliseconds relative to any origin."""
    exposed = comm.difference(compute)
    idle = IntervalSet(((step_start, step_end),)).difference(comm.union(compute))
    lanes = [
        ("compute", compute, "#1d4ed8"),
        ("communication", comm, "#b45309"),
        ("exposed comm", exposed, "#be123c"),
        ("idle", idle, "#a8a29e"),
    ]
    pad_l, pad_t, pad_b, pad_r = 118, 34 if title else 14, 40, 18
    plot_w = width - pad_l - pad_r
    height = pad_t + len(lanes) * (lane_h + 8) + pad_b
    span = max(step_end - step_start, 1e-9)

    def px(t: float) -> float:
        return pad_l + (t - step_start) / span * plot_w

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" font-family="ui-sans-serif, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        o.append(f'<text x="0" y="18" font-size="13" font-weight="650" fill="#1c1917">{_esc(title)}</text>')

    for i, (name, iset, col) in enumerate(lanes):
        y = pad_t + i * (lane_h + 8)
        frac = iset.measure() / span
        o.append(
            f'<text x="{pad_l - 10}" y="{y + lane_h / 2 + 4}" font-size="11" text-anchor="end" '
            f'fill="#292524">{_esc((labels or {}).get(name, name))}</text>'
            f'<rect x="{pad_l}" y="{y}" width="{plot_w}" height="{lane_h}" fill="#fafaf9" '
            f'stroke="#e7e5e4"/>'
        )
        for a, b in iset:
            x0, x1 = px(max(a, step_start)), px(min(b, step_end))
            if x1 > x0:
                o.append(
                    f'<rect x="{x0:.2f}" y="{y + 2}" width="{max(x1 - x0, 0.6):.2f}" '
                    f'height="{lane_h - 4}" fill="{col}" opacity="0.88">'
                    f'<title>{a:.3f}-{b:.3f} ms</title></rect>'
                )
        o.append(
            f'<text x="{pad_l + plot_w + 6}" y="{y + lane_h / 2 + 4}" font-size="10" '
            f'fill="#57534e">{frac:.0%}</text>'
        )

    for k in range(6):
        t = step_start + span * k / 5
        o.append(
            f'<text x="{px(t):.1f}" y="{height - 18}" font-size="10" text-anchor="middle" '
            f'fill="#57534e">{t - step_start:.1f}</text>'
        )
    o.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 4}" font-size="11" text-anchor="middle" '
        f'fill="#1c1917">ms into step</text>'
    )
    o.append("</svg>")
    return "".join(o)


def overlap_bar_svg(rows: list[dict[str, Any]], *, title: str = "", width: int = 660,
                    row_h: int = 24) -> str:
    """Overlap efficiency per configuration, worst-first, with exposed-time share.

    Sorted ascending so the configurations that fail to overlap are at the top:
    a sweep summary that buries the failures at the bottom of a chart is the
    same reporting failure the standard targets, in visual form.
    """
    rows = sorted(rows, key=lambda r: r.get("overlap_efficiency", 0.0))
    pad_l, pad_t, pad_r, pad_b = 220, 30 if title else 8, 92, 26
    plot_w = width - pad_l - pad_r
    height = pad_t + len(rows) * row_h + pad_b

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" font-family="ui-sans-serif, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        o.append(f'<text x="0" y="17" font-size="13" font-weight="650" fill="#1c1917">{_esc(title)}</text>')

    for i, r in enumerate(rows):
        y = pad_t + i * row_h
        eff = float(r.get("overlap_efficiency", 0.0))
        col = "#be123c" if eff < 0.5 else ("#b45309" if eff < 0.8 else "#047857")
        o.append(
            f'<text x="{pad_l - 10}" y="{y + row_h / 2 + 4}" font-size="10.5" text-anchor="end" '
            f'fill="#292524">{_esc(r.get("label", ""))}</text>'
            f'<rect x="{pad_l}" y="{y + 4}" width="{plot_w}" height="{row_h - 9}" fill="#f5f5f4"/>'
            f'<rect x="{pad_l}" y="{y + 4}" width="{max(plot_w * eff, 1):.1f}" '
            f'height="{row_h - 9}" fill="{col}"/>'
            f'<text x="{pad_l + plot_w + 8}" y="{y + row_h / 2 + 4}" font-size="10" '
            f'fill="#1c1917">{eff:.0%}</text>'
        )
        if r.get("exposed_frac_of_step") is not None:
            o.append(
                f'<text x="{pad_l + plot_w + 46}" y="{y + row_h / 2 + 4}" font-size="9.5" '
                f'fill="#78716c">{float(r["exposed_frac_of_step"]):.0%} exp</text>'
            )
    o.append(
        f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="#57534e">'
        f'overlap efficiency = 1 - exposed communication / total communication '
        f'(right column: exposed share of step)</text>'
    )
    o.append("</svg>")
    return "".join(o)
