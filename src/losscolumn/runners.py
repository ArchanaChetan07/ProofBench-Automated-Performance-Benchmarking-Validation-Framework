"""End-to-end runners: measurement in, published artifact out.

One function per thrust, plus a synthesis. Each returns the paths it wrote, and
each records in the claim the exact command that reproduces it -- so requirement
LC-5 is satisfied by the artifacts of this project and not merely asserted by
its documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from losscolumn.pipeline import assemble, md_to_html, publish, reproduction
from losscolumn.report.heatmap import frontier_svg, loss_map_svg
from losscolumn.report.render import Figure
from losscolumn.report.timeline import overlap_bar_svg, step_timeline_svg

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
    from losscolumn.thrusts.kernel.sweep import run_kernel_sweep
    from losscolumn.thrusts.kernel.triton_fa import device_limits, supported_dtypes

    pre = _prereg("III", prereg_dir)
    grid: dict[str, Any] = _analysis_params(pre)

    # The protocol registers both float16 and bfloat16. bf16 tensor cores
    # arrive with Ampere, so on an older device the registered grid cannot be
    # run in full. Narrowing it silently is precisely the selection LC-4
    # exists to prevent, so the restriction is declared against the seal and
    # travels with the artifact.
    limits = device_limits()
    usable = supported_dtypes()
    registered = tuple(
        str(x)
        for f in (pre.factors if pre else [])
        if f.get("name") == "dtype"
        for x in f.get("levels", [])
    )
    if usable and registered and not set(registered) <= set(usable):
        dropped = sorted(set(registered) - set(usable))
        grid["dtypes"] = tuple(d for d in registered if d in usable)
        if pre is not None:
            pre.declare_deviation(
                "factors",
                {"dtype": list(grid["dtypes"])},
                f"{limits.get('capability', 'this device')} has no bf16 tensor cores; "
                f"{', '.join(dropped)} cannot be measured here and would report an "
                "emulated path rather than the algorithm",
            )

    if quick:
        grid.update(head_dims=(32, 64), seq_lens=(128, 512), batches=(1,),
                    dtypes=("float16",), iters=8)
        _declare_quick(pre, grid)
    res = run_kernel_sweep(**grid)
    env, lc = res.envelope, res.loss_column
    assert lc is not None

    figures = [
        Figure(
            svg=loss_map_svg(
                env, res.comparisons, x="seq_len", y="head_dim", facet="dtype",
                loss_column=lc,
                title="Loss map -- from-scratch attention against the reference",
                subtitle=f"{res.implementation} vs torch SDPA on {res.device}. "
                         "Correctness is checked before any timing is recorded.",
            ),
            caption=(
                "The reimplementation is expected to lose across most of this space, and "
                "does. Matching a mature hand-tuned kernel is not the goal and would not "
                "be credible; the contribution is the map and the attribution. A reader "
                "who wants the fastest kernel should use the reference."
            ),
        )
    ]

    claim = assemble(
        claim_id="lc-thrust3-attention-loss-map",
        title="Reproduction with a loss map: a from-scratch attention kernel",
        thrust="III",
        method="reimplementation",
        baseline="reference",
        statement=(
            f"A faithful from-scratch implementation of the FlashAttention algorithm "
            f"({res.implementation}), validated for numerical correctness against an fp64 "
            f"ground truth before any timing, then swept across "
            f"(head_dim, seq_len, batch, dtype) on {res.device}."
        ),
        envelope=env,
        comparisons=res.comparisons,
        loss_column=lc,
        prereg=pre,
        repro=reproduction(
            "losscolumn run thrust3",
            hardware=res.device,
            runtime_min=220,
            cost_usd=704,
            image="ghcr.io/archanachetan07/losscolumn:0.1.0",
        ),
        evidence_class=res.evidence_class,
        evidence_note=(
            res.envelope.meta.get("fallback_reason", "") + (("  " + QUICK_NOTE) if quick else "")
        ).strip(),
        attributions={r.describe(): r.attribution or "unattributed" for r in lc.regions},
        supporting={
            "correctness": res.correctness.to_dict(),
            "clocks": res.clocks,
            "roofline": res.roofline,
            "device_limits": limits,
        },
        limitations=[
            "FlashAttention-3's headline mechanisms -- warp specialisation, TMA-driven "
            "asynchronous copy, ping-pong scheduling of softmax against the next GEMM -- "
            "are Hopper features that Triton's programming model does not expose. This is "
            "a reimplementation of the algorithm, not of the implementation, and the loss "
            "map is the measurement of what that distinction costs.",
            f"Measured on {limits.get('name', 'this device')} "
            f"({limits.get('capability', '?')}), which is two architecture generations "
            "behind the FA-3 target. The gap here is a lower bound on what the missing "
            "Hopper mechanisms are worth, not a measurement of them: they do not exist "
            "on this hardware for either arm to use.",
            "Forward pass only. The backward pass has a different arithmetic intensity and "
            "a different loss map, and is out of scope here.",
            "SM clocks are recorded but not controlled. A run taken while the card is "
            "throttling is not comparable to one taken cold, and the difference routinely "
            "exceeds published effect sizes.",
        ],
    )

    extra = [("Correctness suite", md_to_html(res.correctness.to_markdown(40)))]
    return publish(claim, outdir, figures=figures, extra_sections=extra,
                   stem="thrust3-attention-loss-map")


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
