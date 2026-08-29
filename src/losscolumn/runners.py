"""End-to-end runners: measurement in, published artifact out.

One function per thrust, plus a synthesis. Each returns the paths it wrote, and
each records in the claim the exact command that reproduces it -- so requirement
LC-5 is satisfied by the artifacts of this project and not merely asserted by
its documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from losscolumn.core.provenance import Provenance, utcnow
from losscolumn.pipeline import assemble, md_to_html, publish, reproduction
from losscolumn.report.heatmap import frontier_svg, loss_map_svg
from losscolumn.report.render import Figure, render_page
from losscolumn.report.timeline import overlap_bar_svg, step_timeline_svg
from losscolumn.version import STANDARD_VERSION

SIMULATED_NOTE = (
    "Produced by the package's calibratable model, not by measurement on the target "
    "hardware. The model's parameters -- achievable FLOP/s, bus bandwidth, collective "
    "latency, per-step scheduler overhead -- are the quantities a microbenchmark measures, "
    "and the funded run replaces them with measured values. Every analysis step downstream "
    "of the numbers is the same code that runs on measured data; this artifact exists to "
    "show that path works before GPU time is committed to it."
)


def _prereg(thrust: str, prereg_dir: Path):
    from losscolumn.cli.main import load_prereg

    return load_prereg(thrust, prereg_dir)


QUICK_NOTE = (
    "Reduced-grid smoke run. This artifact is deliberately NOT conforming: it drops "
    "registered factor levels, which the standard treats as the selection it exists to "
    "prevent. It is published so the failure mode is visible, and so the deviation "
    "machinery is exercised on a case where the deviation is real."
)


def _declare_quick(pre: Any, grid: dict[str, Any]) -> None:
    """Record a reduced grid as a declared deviation rather than a silent one.

    A smoke run is a legitimate thing to do and an illegitimate thing to
    publish quietly. Declaring it turns the protocol check from a fatal
    "levels were dropped" into a warning that names who dropped them and why,
    which is the distinction the deviation mechanism exists to draw.
    """
    if pre is None:
        return
    pre.declare_deviation(
        "factors",
        {k: list(v) for k, v in grid.items() if isinstance(v, (tuple, list))},
        "reduced grid for a smoke run; not a publishable measurement",
    )


def _analysis_params(pre: Any, **fallback: Any) -> dict[str, Any]:
    """Read the analysis parameters out of the sealed protocol.

    This is what makes LC-4 more than a filing requirement. The MDE, the FDR
    level and the replicate count are *inputs to the sweep*, taken from the
    seal, so an analysis that silently used a different effect size than the
    one registered is not merely detected after the fact -- it cannot happen,
    because there is only one place those numbers live.
    """
    if pre is None:
        return dict(fallback)
    return {
        "mde": pre.mde,
        "q_level": pre.q_level,
        "replicates": pre.replicates,
        **{k: v for k, v in fallback.items()
           if k not in ("mde", "q_level", "replicates")},
    }


# --------------------------------------------------------------------------
# Thrust I
# --------------------------------------------------------------------------


def run_thrust_one(*, outdir: Path, prereg_dir: Path, quick: bool = False) -> dict[str, Path]:
    from losscolumn.core.provenance import Provenance
    from losscolumn.thrusts.overlap.sweep import RECOMMENDED, run_overlap_sweep

    pre = _prereg("I", prereg_dir)
    grid: dict[str, Any] = _analysis_params(pre)
    if quick:
        grid.update(micro_batches=(1, 4), seq_lens=(512, 2048), world_sizes=(8, 32))
        _declare_quick(pre, grid)
    res = run_overlap_sweep(**grid)
    env, lc = res.throughput, res.loss_column
    assert lc is not None

    figures = [
        Figure(
            svg=loss_map_svg(
                env, res.comparisons, x="micro_batch", y="seq_len", facet="world_size",
                loss_column=lc,
                title="Where the recommended sharding configuration loses",
                subtitle=f"{RECOMMENDED} vs the best of the swept alternatives, "
                         f"cluster throughput. Cells collapse over checkpointing by "
                         f"taking the most adverse.",
            ),
            caption=(
                "Red is a region where following the standard recommendation costs "
                "throughput; crossed cells are configurations that cannot run at all. "
                "Collapsing over the checkpointing axis takes the <em>worst</em> cell "
                "rather than the mean: averaging would hide the failures this map exists "
                "to show."
            ),
        ),
        Figure(
            svg=overlap_bar_svg(
                res.per_config_summary(),
                title="Overlap efficiency by sharding configuration (worst first)",
            ),
            caption=(
                "Overlap efficiency is computed by interval algebra over the trace -- the "
                "union of communication intervals minus the union of compute intervals -- "
                "not by summing kernel durations, which double-counts concurrent "
                "collectives. The right-hand figure is exposed communication as a share "
                "of step wall-clock, which is what a sharding change actually buys."
            ),
        ),
    ]
    ex = res.exemplar_traces.get(RECOMMENDED)
    if ex is not None:
        lo, hi = ex.bounds()
        figures.append(
            Figure(
                svg=step_timeline_svg(
                    ex.bucket("compute"), ex.bucket("comm"),
                    step_start=lo, step_end=min(hi, lo + 0.25 * (hi - lo)),
                    title=f"First quarter of one step -- {RECOMMENDED}",
                ),
                caption=(
                    "The exposed lane is the point of the figure: wall-clock the collective "
                    "failed to hide. A step can be 95% overlapped and still lose a fifth of "
                    "its time to the 5% that is not."
                ),
            )
        )

    claim = assemble(
        claim_id="lc-thrust1-overlap-envelope",
        title="The overlap envelope: where the recommended sharding configuration breaks down",
        thrust="I",
        method=RECOMMENDED,
        baseline="best_of_swept",
        statement=(
            f"Across a {env.n_cells}-cell sharding envelope for a 7B-class model, the "
            f"commonly recommended FSDP configuration ({RECOMMENDED}) is at or near the "
            "best swept alternative over most of the space and collapses in identifiable "
            "corners of it."
        ),
        envelope=env,
        comparisons=res.comparisons,
        loss_column=lc,
        prereg=pre,
        repro=reproduction(
            "losscolumn run thrust1",
            hardware="8x A100 80GB (measured path) / calibratable model (this artifact)",
            runtime_min=120,
            cost_usd=1560,
            image="ghcr.io/archanachetan07/losscolumn:0.1.0",
        ),
        evidence_class=res.evidence_class,
        evidence_note=(
            (SIMULATED_NOTE if res.evidence_class == "simulated" else "")
            + (("  " + QUICK_NOTE) if quick else "")
        ).strip(),
        attributions={
            r.describe(): r.attribution or "unattributed" for r in lc.regions
        },
        supporting={
            "baseline_derived_from": [k for k in res.configs if k != RECOMMENDED],
            "cliffs": {k: v.to_dict() for k, v in res.cliffs.items()},
            "cliff_adjacency": res.adjacency,
            "per_config": res.per_config_summary(),
        },
        limitations=[
            "A single 8-GPU node cannot exhibit the inter-node bandwidth effects that "
            "dominate at cluster scale. The world_size=32 column is modelled; on the "
            "funded allocation it is out of budget, and the artifact states that here "
            "rather than in a footnote.",
            "The comparator is an oracle -- the best of four swept configurations at each "
            "cell -- which flatters the alternatives, since a practitioner would have to "
            "find that configuration themselves.",
            "Overlap efficiency is measured on one rank. A straggler on another rank "
            "appears here as compute time, not as exposed communication.",
        ],
        provenance=env.provenance or Provenance.capture(),
    )

    extra = [
        ("Discontinuities", md_to_html(res.cliffs[RECOMMENDED].to_markdown(10))),
        ("Cliff adjacency of named configurations", _adjacency_html(res.adjacency)),
    ]
    return publish(claim, outdir, figures=figures, extra_sections=extra,
                   stem="thrust1-overlap-envelope")


def _adjacency_html(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{r.get('name')}</td>"
        f"<td>{'yes' if r.get('cliff_adjacent') else 'no'}</td>"
        f"<td>{r.get('exposure_pct', 0):.1f}%</td>"
        f"<td>{'<br>'.join(r.get('adjacent_cliffs', [])) or '&ndash;'}</td>"
        "</tr>"
        for r in rows
    )
    return (
        "<p>RQ1 in tabular form. A configuration is dangerous when a single-level change "
        "in any direction costs far more than the configuration itself gains.</p>"
        '<div class="scroll"><table><thead><tr><th>Configuration</th>'
        "<th>Adjacent to a cliff</th><th>One-step exposure</th><th>The cliff</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# Thrust II
# --------------------------------------------------------------------------


def run_thrust_two(*, outdir: Path, prereg_dir: Path, quick: bool = False) -> dict[str, Path]:
    from losscolumn.thrusts.engines.audit import run_engine_audit
    from losscolumn.thrusts.engines.workloads import standard_workloads

    pre = _prereg("II", prereg_dir)
    params = _analysis_params(pre)
    if quick:
        _declare_quick(pre, {"workload": ["chat", "rag", "summarize", "agentic"]})
    wl = standard_workloads(n=120 if quick else 400)
    res = run_engine_audit(
        workloads=wl,
        n_trials=12 if quick else (pre.tuning_budget_trials if pre else 40),
        **params,
    )
    env, lc = res.envelope, res.loss_column
    assert lc is not None

    figures = [
        Figure(
            svg=loss_map_svg(
                env, res.comparisons, x="latency_budget_ms", y="workload", loss_column=lc,
                title="Where vLLM loses under tuning-budget parity",
                subtitle="Throughput attainable under a normalised p99 latency budget, "
                         "after every engine received an identical 40-trial search.",
            ),
            caption=(
                "The latency axis is normalised -- p99 end-to-end time per output token -- "
                "because absolute budgets are not comparable across workloads whose "
                "requests differ in length by an order of magnitude. Crossed cells are "
                "budgets no engine could meet at any measured concurrency."
            ),
        )
    ]
    for wname in list(wl)[: 2 if quick else 4]:
        fr = {e: res.frontier(e, wname) for e in ("vllm", "sglang", "trtllm")}
        fr = {k: v for k, v in fr.items() if v is not None}
        if fr:
            figures.append(
                Figure(
                    svg=frontier_svg(
                        fr, title=f"Latency-throughput frontier -- {wname}",
                        xlabel="p99 latency per output token (ms)",
                    ),
                    caption=(
                        f"Every configuration measured on the {wname} trace, with each "
                        "engine's Pareto frontier drawn through it. Diamonds mark library "
                        "defaults. An engine is not 'x times faster'; it dominates over "
                        "some range of latency budgets and loses over others, and which "
                        "range you are in is a property of your SLO."
                    ),
                )
            )

    claim = assemble(
        claim_id="lc-thrust2-equal-tuning-audit",
        title="The equal-tuning audit: engine differences that survive budget parity",
        thrust="II",
        method=res.method,
        baseline="best_alternative",
        statement=(
            "Three inference engines, each given an identical 40-trial search from the "
            "same procedure against the same traces on the same hardware, compared across "
            "their full latency-throughput frontiers on four workload shapes."
        ),
        envelope=env,
        comparisons=res.comparisons,
        loss_column=lc,
        prereg=pre,
        repro=reproduction(
            "losscolumn run thrust2",
            hardware="8x H100 (measured path) / calibratable model (this artifact)",
            runtime_min=80,
            cost_usd=2000,
            image="ghcr.io/archanachetan07/losscolumn:0.1.0",
        ),
        parity=res.parity,
        evidence_class=res.evidence_class,
        evidence_note=(
            (SIMULATED_NOTE if res.evidence_class == "simulated" else "")
            + (("  " + QUICK_NOTE) if quick else "")
        ).strip(),
        attributions={r.describe(): r.attribution or "unattributed" for r in lc.regions},
        supporting={
            "baseline_derived_from": sorted(
                {k.split("|")[0] for k in res.ledgers} - {res.method}
            ),
            "workloads": {k: w.summary() for k, w in res.workloads.items()},
            "default_to_tuned": res.default_gaps,
            "frontier_comparisons": {
                k: v.summary() for k, v in res.frontier_comparisons.items()
            },
        },
        limitations=[
            "Equal tuning budget is not equal tuning skill. One operator ran all three "
            "searches; familiarity is a residual confound the protocol documents and "
            "cannot remove.",
            "Engine releases move weekly. This is a snapshot against pinned commits and "
            "is designed to be re-run rather than cited indefinitely.",
            "Equal trial counts are nominal parity. The certificate reports where they "
            "were not effective parity -- unequal search-space coverage and a large "
            "wall-clock asymmetry from per-configuration engine builds.",
            "Prompts are synthetic sequences of the measured length. Length drives the "
            "systems behaviour under test, but real text would change cache behaviour "
            "for the prefix-caching engines.",
        ],
    )

    extra = [
        ("Default-to-tuned gap", md_to_html(res.default_gap_markdown())),
        ("Workload traces", _workload_html(res)),
        ("Where each engine wins", _crossover_html(res)),
    ]
    return publish(claim, outdir, figures=figures, extra_sections=extra,
                   stem="thrust2-equal-tuning-audit")


def _workload_html(res: Any) -> str:
    rows = "".join(
        "<tr>"
        f"<td><b>{k}</b><br><span style='color:var(--muted)'>{w.description}</span></td>"
        f"<td>{w.n_requests}</td>"
        f"<td>{int(np.median([r.prompt_tokens for r in w.requests]))}</td>"
        f"<td>{int(np.median([r.output_tokens for r in w.requests]))}</td>"
        f"<td>{w.prefix_reuse_fraction:.0%}</td>"
        f"<td><code>{w.digest()[7:19]}</code></td>"
        "</tr>"
        for k, w in res.workloads.items()
    )
    return (
        "<p>The same request trace is replayed against every engine; the digest is what "
        "proves the inputs were identical. Prefix reuse is a property of the workload, not "
        "of the engine, and it is the quantity a radix cache monetises &mdash; publishing an "
        "engine comparison without stating it is the easiest way to make either engine look "
        "dominant.</p>"
        '<div class="scroll"><table><thead><tr><th>Workload</th><th>Requests</th>'
        "<th>Median prompt</th><th>Median output</th><th>Prefix reuse</th><th>Digest</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _crossover_html(res: Any) -> str:
    rows = ""
    for key, fc in sorted(res.frontier_comparisons.items()):
        s = fc.summary()
        cross = s.get("crossover_ms")
        rows += (
            "<tr>"
            f"<td>{key.replace('|', ' vs ')}</td>"
            f"<td>{s['frac_budgets_method_wins']:.0%}</td>"
            f"<td>{s['max_win_pct']:+.0f}%</td>"
            f"<td>{s['max_loss_pct']:+.0f}%</td>"
            f"<td>{f'{cross:.1f} ms/token' if cross else 'no single crossover'}</td>"
            "</tr>"
        )
    return (
        "<p>Attainment-curve comparison, budget by budget. The crossover is the latency "
        "budget at which the ordering flips; where there is no single crossover, the two "
        "frontiers interleave and no ordering statement is available at all.</p>"
        '<div class="scroll"><table><thead><tr><th>Comparison</th>'
        "<th>Budgets vLLM wins</th><th>Best</th><th>Worst</th><th>Crossover</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# Thrust III
# --------------------------------------------------------------------------


def run_thrust_three(*, outdir: Path, prereg_dir: Path, quick: bool = False) -> dict[str, Path]:
    """Run the registered lattice for every available implementation.

    Each implementation gets its own claim. The portable one measures what the
    algorithm costs in eager dispatch; the fused one measures what the code
    generator achieves. Pooling them would produce a single number describing
    neither, so the protocol registers them as separate systems and they are
    published as separate artifacts.
    """
    from losscolumn.thrusts.kernel.sparse_sweep import IMPLEMENTATIONS, run_sparse_sweep, select

    pre = _prereg("III", prereg_dir)
    params = _analysis_params(pre)
    grid: dict[str, Any] = dict(params)
    if quick:
        grid.update(seq_lens=(512, 2048), batches=(1,), iters=5)
        _declare_quick(pre, grid)

    available: list[str] = []
    unavailable: dict[str, str] = {}
    for impl in IMPLEMENTATIONS:
        try:
            select(impl)
            available.append(impl)
        except Exception as e:
            unavailable[impl] = str(e)

    out: dict[str, Path] = {}
    for impl in available:
        res = run_sparse_sweep(implementation=impl, **grid)
        out.update(
            _publish_kernel_claim(res, pre=pre, outdir=outdir, quick=quick,
                                  unavailable=unavailable)
        )
    if not available:
        raise RuntimeError(f"no attention implementation is runnable here: {unavailable}")
    return out


_IMPL_BLURB = {
    "portable": (
        "the FlashAttention algorithm written out in eager PyTorch ops: correct "
        "everywhere, portable to any device, and not fast. Its loss map measures what "
        "per-tile dispatch costs"
    ),
    "triton": (
        "the same algorithm as a fused Triton kernel with real block skipping. Its "
        "loss map measures what the code generator achieves"
    ),
}


def _publish_kernel_claim(res, *, pre, outdir: Path, quick: bool,
                          unavailable: dict[str, str]) -> dict[str, Path]:
    from losscolumn.thrusts.kernel.sparse_sweep import BASELINE, METHOD

    env, lc = res.latency, res.loss_column
    assert lc is not None
    label = "A" if res.implementation == "portable" else "B"

    figures = [
        Figure(
            svg=loss_map_svg(
                env, res.comparisons, x="seq_len", y="pattern", facet="phase",
                loss_column=lc,
                title=f"Loss map {label} -- {res.implementation} vs the reference",
                subtitle=f"per-call latency on {res.device}; batch and density collapse "
                         f"to the most adverse cell",
            ),
            caption=(
                "Red marks a region where this implementation is slower than the "
                "reference computing the same attention pattern. Crossed cells are "
                "combinations that are not a configuration &mdash; dense below full "
                "density, or a sparse pattern at full density &mdash; recorded rather "
                "than dropped so the artifact does not claim a full factorial it did "
                "not run. Collapsing takes the <em>worst</em> cell, never the mean."
            ),
        ),
        Figure(
            svg=loss_map_svg(
                env, res.comparisons, x="density", y="batch", facet="phase",
                loss_column=lc,
                title=f"Loss map {label} -- by density and batch",
                subtitle="the same measurements, sliced along the sparsity axis",
            ),
            caption=(
                "Density is reported as the fraction of blocks actually visited, and "
                "throughput counts only the blocks computed. Counting dense-equivalent "
                "FLOPs against a sparse kernel's runtime turns the pattern's density "
                "into a speedup it did not earn."
            ),
        ),
    ]

    claim = assemble(
        claim_id=f"lc-thrust3{label.lower()}-{res.implementation}-attention",
        title=(
            f"Sparse attention, {res.implementation} implementation: "
            f"a loss map over prefill and decode"
        ),
        thrust="III",
        method=METHOD,
        baseline=BASELINE,
        statement=(
            f"The {res.implementation} implementation &mdash; {_IMPL_BLURB[res.implementation]} "
            f"&mdash; swept over the registered "
            f"(seq_len x batch x pattern x density x phase) lattice on {res.device}, "
            f"against the reference computing the same pattern."
        ),
        envelope=env,
        comparisons=res.comparisons,
        loss_column=lc,
        prereg=pre,
        repro=reproduction(
            "losscolumn run thrust3",
            hardware=res.device,
            runtime_min=25,
            cost_usd=0.0,
            image="ghcr.io/archanachetan07/losscolumn:0.1.0",
        ),
        evidence_class=res.evidence_class,
        evidence_note=(QUICK_NOTE if quick else ""),
        attributions={r.describe(): r.attribution or "unattributed" for r in lc.regions},
        supporting={
            "implementation": res.implementation,
            "device_limits": res.limits,
            "phase_summary": res.phase_summary,
            "correctness": res.correctness.to_dict(),
            "clocks": res.clocks,
            "layouts": res.layouts,
            "throughput_envelope": res.throughput.to_dict(include_replicates=False),
            "memory_envelope": res.memory.to_dict(include_replicates=False),
            "implementations_unavailable": unavailable,
            "baseline_tuning_policy": env.meta.get("baseline_tuning_policy"),
        },
        limitations=[
            f"Measured on {res.limits.get('name', 'this device')} "
            f"({res.limits.get('capability', '?')}), which predates the architecture "
            "FlashAttention-3 targets. FA-3's warp specialisation, TMA and ping-pong "
            "scheduling do not exist on this hardware for either arm to use, so this "
            "is not evidence about them.",
            "Forward pass only. The backward pass has a different arithmetic intensity "
            "and a different loss map.",
            "One head dimension (64) and one dtype (float16). Both are held fixed to "
            "keep the sparsity lattice tractable; widening either is a new "
            "registration, not an extension of this one.",
            "The block-sparse pattern is one seeded random draw per shape, not an "
            "average over draws. A different draw would give a different pattern with "
            "the same density, and the artifact records the seed rather than claiming "
            "generality over patterns.",
            "SM clocks are recorded but not controlled. A run taken while the card is "
            "throttling is not comparable to one taken cold.",
        ],
    )

    extra = [
        ("Speedup by phase", _phase_table(res)),
        ("Correctness suite", md_to_html(res.correctness.to_markdown(40))),
    ]
    return {
        f"{res.implementation}.{k}": v
        for k, v in publish(
            claim, outdir, figures=figures, extra_sections=extra,
            stem=f"thrust3{label.lower()}-{res.implementation}-attention",
        ).items()
    }


def _phase_table(res) -> str:
    rows = ""
    for phase, s in res.phase_summary.items():
        rows += (
            "<tr>"
            f"<td><b>{phase}</b><span class='sub'>{s['interpretation']}</span></td>"
            f"<td>{s['n_measured']}/{s['n_cells']}</td>"
            f"<td>{s['median_latency_ms']:.3f} ms</td>"
            f"<td>{s['median_reference_latency_ms']:.3f} ms</td>"
            f"<td>{s['median_speedup']:.2f}x</td>"
            f"<td>{s['worst_speedup']:.2f}x</td>"
            f"<td>{s['best_speedup']:.2f}x</td>"
            "</tr>"
        )
    return (
        "<p>Latency means something different in each phase, so it is reported per "
        "phase rather than pooled: at prefill it is the time-to-first-token "
        "contribution, at decode it is the inter-token latency. A speedup below 1.00x "
        "is a loss.</p>"
        '<div class="scroll"><table><thead><tr><th>Phase</th><th>Cells</th>'
        "<th>Median latency</th><th>Reference</th><th>Median speedup</th>"
        f"<th>Worst</th><th>Best</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def run_all(*, outdir: Path, prereg_dir: Path, quick: bool = False) -> dict[str, Path]:
    from losscolumn.synthesis import write_synthesis

    out: dict[str, Path] = {}
    for name, fn in (
        ("thrust3", run_thrust_three),
        ("thrust1", run_thrust_one),
        ("thrust2", run_thrust_two),
    ):
        # Thrust III leads, as the proposal sequences it: it is the cheapest to
        # iterate and it validates the measurement tooling before the expensive
        # multi-GPU work is committed.
        for k, v in fn(outdir=outdir, prereg_dir=prereg_dir, quick=quick).items():
            out[f"{name}.{k}"] = v
    for k, v in write_synthesis(outdir=outdir).items():
        out[f"synthesis.{k}"] = v
    return out


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def run_calibration(*, outdir: Path, prereg_dir: Path, quick: bool = False,
                    **_: Any) -> dict[str, Path]:
    """Check a simulated thrust's predicted loss map against measurement.

    This is the step that turns "here is a simulation" into "here is how far
    the simulation can be trusted, and where it cannot be checked at all". It
    publishes a calibration report rather than a claim: a calibration study is
    evidence *about* a claim, and giving it the same document type would invite
    it to be cited as though it were a result of its own.
    """
    from losscolumn.thrusts.overlap.calibrate import run_thrust_one_calibration

    outdir.mkdir(parents=True, exist_ok=True)
    cal = run_thrust_one_calibration(
        seq_lens=(512, 1024) if quick else (512, 1024, 2048),
        replicates=5 if quick else 11,
        iters=3 if quick else 5,
    )

    payload = cal.to_dict()
    payload["kind"] = "calibration-report"
    payload["standard_version"] = STANDARD_VERSION
    payload["calibrates"] = "thrust1"
    payload["generated_at"] = utcnow()
    payload["command"] = "losscolumn calibrate thrust1"
    payload["provenance"] = Provenance.capture().to_dict()

    stem = "calibration-thrust1-compute"
    jpath = outdir / f"{stem}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = cal.to_markdown()
    mpath = outdir / f"{stem}.md"
    mpath.write_text(md, encoding="utf-8")

    html = render_page(
        title="Calibration -- Thrust I compute model",
        subtitle=(
            "How far the simulated overlap envelope can be trusted, and which half "
            "of the model remains unchecked"
        ),
        body=md_to_html(md),
        meta={
            "Device": cal.device,
            "Verdict": cal.result.verdict(),
            "Cells compared": str(cal.result.n_compared),
            "False wins": str(cal.result.n_false_win),
        },
    )
    hpath = outdir / f"{stem}.html"
    hpath.write_text(html, encoding="utf-8")

    print(cal.to_markdown())
    return {"calibration": jpath, "markdown": mpath, "html": hpath}
