"""Thrust II -- the equal-tuning audit.

RQ2: how much of the measured difference between inference engines survives
when each receives an identical tuning budget?

The design, in the order it runs:

1. **Tune under parity.** Every engine gets the same number of trials, from the
   same search procedure, against the same workload traces, on the same
   hardware, run by the same operator. Every trial lands in an append-only
   ledger and the parity certificate is derived from those ledgers, not
   asserted.
2. **Measure twice.** Each engine is evaluated at its library defaults *and* at
   its tuned optimum. The default-to-tuned gap is a primary result: most
   deployments run near defaults, so an engine that is only fast after a
   40-trial search is making a different offer than one that is fast out of
   the box.
3. **Sweep the frontier.** Concurrency is swept to trace the full
   latency-throughput Pareto frontier per engine per workload, not a single
   operating point.
4. **Compare attainment.** Two frontiers are compared through their attainment
   curves -- best throughput achievable under each latency budget. The loss
   column falls out of where that ratio drops below one; it is derived, not
   editorially chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.losscolumn import LossColumn, extract_loss_column
from losscolumn.core.pareto import (
    FrontierComparison,
    OperatingPoint,
    ParetoFrontier,
    compare_frontiers,
)
from losscolumn.core.parity import ParityCertificate, TuningLedger, certify_grouped
from losscolumn.core.provenance import Provenance
from losscolumn.core.stats import CellComparison, compare_cells
from losscolumn.thrusts.engines.adapters.synthetic import SyntheticEngine, synthetic_engines
from losscolumn.thrusts.engines.search import SearchProcedure, engine_seed
from losscolumn.thrusts.engines.workloads import Workload, standard_workloads

DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass
class AuditResult:
    envelope: Envelope
    parity: ParityCertificate
    ledgers: dict[str, TuningLedger]
    frontiers: dict[tuple[str, str], ParetoFrontier] = field(default_factory=dict)
    comparisons: list[CellComparison] = field(default_factory=list)
    loss_column: LossColumn | None = None
    frontier_comparisons: dict[str, FrontierComparison] = field(default_factory=dict)
    default_gaps: dict[str, dict[str, Any]] = field(default_factory=dict)
    workloads: dict[str, Workload] = field(default_factory=dict)
    method: str = "vllm"
    baseline: str = "best_alternative"
    evidence_class: str = "simulated"

    def frontier(self, engine: str, workload: str) -> ParetoFrontier | None:
        return self.frontiers.get((engine, workload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "baseline": self.baseline,
            "evidence_class": self.evidence_class,
            "workloads": {k: w.summary() for k, w in self.workloads.items()},
            "parity": self.parity.to_dict(),
            "envelope": self.envelope.to_dict(),
            "loss_column": self.loss_column.to_dict() if self.loss_column else None,
            "frontiers": {
                f"{e}|{w}": f.to_dict() for (e, w), f in self.frontiers.items()
            },
            "frontier_comparisons": {
                k: v.to_dict() for k, v in self.frontier_comparisons.items()
            },
            "default_to_tuned": self.default_gaps,
        }

    def default_gap_markdown(self) -> str:
        lines = [
            "### Default-to-tuned gap",
            "",
            "How much of each engine's performance is locked behind a tuning search. "
            "Reported as a primary result because production deployments run near defaults.",
            "",
            "| Engine | Workload | Default tok/s | Tuned tok/s | Gain from tuning | "
            "p99 latency change |",
            "|--------|----------|---------------|-------------|------------------|"
            "--------------------|",
        ]
        for key, g in sorted(self.default_gaps.items()):
            if not g:
                continue
            e, w = key.split("|")
            lines.append(
                f"| {e} | {w} | {g['default']['throughput']:,.0f} | "
                f"{g['tuned']['throughput']:,.0f} | {g['throughput_gain_pct']:+.1f}% | "
                f"{g['latency_change_pct']:+.1f}% |"
            )
        return "\n".join(lines)


def _objective_factory(engine: SyntheticEngine, workload: Workload, concurrency: int, seed: int):
    """Goodput at a reference concurrency: throughput that actually meets the SLO.

    Raw throughput as the tuning objective rewards configurations that trade
    unbounded latency for batch size, which no operator would deploy. Goodput
    is pre-registered as the objective for every engine, so no engine is tuned
    against a target the others were not.
    """

    def objective(cfg: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        r = engine.serve(cfg, workload, concurrency=concurrency, seed=seed)
        if r.error:
            return float("-inf"), "oom" if "fit" in r.error else "error", {"error": r.error}
        return (
            r.goodput,
            "ok",
            {
                "output_throughput": r.output_throughput,
                "e2e_p99": r.e2e_ms_p99,
                "batch": r.meta.get("batch"),
                # Engines that must compile per configuration pay for it in
                # tuning wall-clock, and the certificate reports that.
                "wall_time_s": 12.0 + engine.profile.build_time_s,
            },
        )

    return objective


def normalized_latency_ms(result: Any, workload: Workload) -> float:
    """p99 end-to-end latency per output token.

    The SLO axis has to be comparable across workloads, and absolute
    end-to-end latency is not: a request emitting 1,073 tokens cannot finish
    inside two seconds at any batch size, on any engine, so an absolute budget
    grid would mark most of the summarize column unmeetable for everyone and
    say nothing about the engines. Normalised latency -- the standard
    serving-systems measure -- divides out the length the user asked for and
    leaves the part the engine controls.
    """
    return float(result.norm_latency_ms_p99)


def _frontier_for(
    engine: SyntheticEngine,
    workload: Workload,
    cfg: dict[str, Any],
    *,
    tag: str,
    concurrencies: Sequence[int],
    seed: int,
    reference_concurrency: int = 32,
) -> list[OperatingPoint]:
    pts: list[OperatingPoint] = []
    for c in concurrencies:
        r = engine.serve(cfg, workload, concurrency=c, seed=seed + c)
        if r.error or not np.isfinite(r.norm_latency_ms_p99):
            continue
        pts.append(
            OperatingPoint(
                config=dict(cfg),
                latency_ms=normalized_latency_ms(r, workload),
                throughput=r.output_throughput,
                label=f"{engine.name}/{tag}/c={c}",
                tag=tag if c == reference_concurrency else "",
                meta={
                    "concurrency": c,
                    "e2e_p99_ms": r.e2e_ms_p99,
                    "ttft_p99": r.ttft_ms_p99,
                    "tpot_p50": r.tpot_ms_p50,
                    "goodput": r.goodput,
                    "prefix_hit_rate": r.prefix_hit_rate,
                    "kv_limited": r.meta.get("kv_limited"),
                },
            )
        )
    return pts


def run_engine_audit(
    engines: dict[str, SyntheticEngine] | None = None,
    workloads: dict[str, Workload] | None = None,
    *,
    method: str = "vllm",
    n_trials: int = 40,
    replicates: int = 11,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    latency_budgets_ms: Sequence[float] = (10, 15, 25, 50, 100, 250),
    tuning_concurrency: int = 256,
    operator: str = "A. S. Patil",
    seed: int = 20260101,
    mde: float = 0.05,
    q_level: float = 0.05,
    hardware_fingerprint: str | None = None,
) -> AuditResult:
    engines = engines or synthetic_engines()
    workloads = workloads or standard_workloads()
    prov = Provenance.capture()
    hw = hardware_fingerprint or prov.hardware_fingerprint()
    procedure = SearchProcedure(n_trials=n_trials)

    # ---- stage 1: tune every engine under one budget, per workload -------
    # Tuning is per workload, with an identical budget for every engine on
    # every workload. Tuning once on one trace and reporting on four would
    # confound "this engine is worse here" with "this engine was tuned
    # somewhere else", and the transfer penalty is not what RQ2 asks about.
    ledgers: dict[str, TuningLedger] = {}
    tuned: dict[tuple[str, str], dict[str, Any]] = {}
    groups: dict[str, list[TuningLedger]] = {}
    for wname, w in workloads.items():
        for name, eng in engines.items():
            obj = _objective_factory(eng, w, tuning_concurrency, seed)
            led = procedure.run(
                eng.search_space(),
                obj,
                engine=name,
                seed=engine_seed(seed, f"{name}|{wname}"),
                operator=operator,
                hardware_fingerprint=hw,
                workload_digest=w.digest(),
                objective_name=f"goodput@concurrency={tuning_concurrency}",
                group=wname,
            )
            ledgers[f"{name}|{wname}"] = led
            groups.setdefault(wname, []).append(led)
            best = led.best()
            tuned[(name, wname)] = dict(best.config) if best else eng.defaults()

    parity = certify_grouped(groups)
    parity.provenance = prov

    # ---- stage 2 and 3: frontiers at defaults and at the optimum ---------
    frontiers: dict[tuple[str, str], ParetoFrontier] = {}
    default_gaps: dict[str, dict[str, Any]] = {}
    for name, eng in engines.items():
        for wname, w in workloads.items():
            pts = _frontier_for(eng, w, eng.defaults(), tag="default",
                                concurrencies=concurrencies, seed=seed)
            pts += _frontier_for(eng, w, tuned[(name, wname)], tag="tuned",
                                 concurrencies=concurrencies, seed=seed + 7)
            fr = ParetoFrontier(system=name, points=pts, latency_metric="e2e_p99_ms",
                                throughput_metric="output_tok_s")
            frontiers[(name, wname)] = fr
            default_gaps[f"{name}|{wname}"] = fr.default_to_tuned_gap() or {}

    # ---- stage 4: envelope of attainable throughput ----------------------
    others = [n for n in engines if n != method]
    systems = list(engines) + ["best_alternative"]
    env = Envelope.allocate(
        (
            Factor("workload", tuple(workloads), ordered=False),
            Factor("latency_budget_ms", tuple(latency_budgets_ms), log_scale=True, unit="ms"),
        ),
        Metric("attainable_throughput", "tok/s", higher_is_better=True,
               description="best output throughput reachable under the normalised "
                           "p99 latency budget (ms per output token)"),
        systems,
        replicates,
        interleaved=True,
        workload="four standard serving traces",
        provenance=prov,
        meta={
            "evidence_class": "simulated",
            "tuning_trials_per_engine": n_trials,
            "search_procedure": procedure.identifier,
            "tuned_configs": {f"{e}|{w}": c for (e, w), c in tuned.items()},
            "latency_axis": "p99 end-to-end latency per output token (ms/token)",
            "tuning_concurrency": tuning_concurrency,
        },
    )

    budgets = np.asarray(latency_budgets_ms, dtype=float)
    for wi, wname in enumerate(workloads):
        w = workloads[wname]
        per_engine_attain: dict[str, np.ndarray] = {}
        for name, eng in engines.items():
            # Replicates re-run the whole frontier with a fresh seed, so the
            # replicate captures run-to-run variation of the entire pipeline
            # rather than of one operating point.
            reps = []
            for r in range(replicates):
                pts = _frontier_for(eng, w, tuned[(name, wname)], tag="tuned",
                                    concurrencies=concurrencies, seed=seed + 1000 * r)
                fr = ParetoFrontier(system=name, points=pts)
                reps.append(fr.attainment(budgets))
            arr = np.vstack(reps)  # (replicates, budgets)
            per_engine_attain[name] = arr
            for bi in range(len(budgets)):
                col = arr[:, bi]
                cell = (wi, bi)
                if np.all(~np.isfinite(col)):
                    env.mark_missing(
                        name, cell,
                        f"{name} cannot meet a {budgets[bi]:.0f} ms/token normalised p99 "
                        f"budget on {wname} at any measured concurrency",
                    )
                else:
                    env.put(name, cell, np.nan_to_num(col, nan=0.0).tolist())

        for bi in range(len(budgets)):
            cell = (wi, bi)
            stack = [per_engine_attain[n][:, bi] for n in others]
            best = np.nanmax(np.vstack(stack), axis=0) if stack else None
            if best is None or np.all(~np.isfinite(best)):
                env.mark_missing("best_alternative", cell,
                                 "no alternative engine meets this budget either")
            else:
                env.put("best_alternative", cell, np.nan_to_num(best, nan=0.0).tolist())

    comparisons = compare_cells(env, method, "best_alternative", mde=mde, q=q_level,
                                seed=seed, paired=True)
    lc = extract_loss_column(env, comparisons, method=method, baseline="best_alternative",
                             mde=mde, q_level=q_level)

    result = AuditResult(
        envelope=env,
        parity=parity,
        ledgers=ledgers,
        frontiers=frontiers,
        comparisons=comparisons,
        loss_column=lc,
        default_gaps=default_gaps,
        workloads=workloads,
        method=method,
    )
    # Compare against every alternative separately rather than against one
    # "best" engine: which alternative wins depends on the latency budget, and
    # collapsing them first would hide exactly that.
    for wname in workloads:
        m = frontiers.get((method, wname))
        for o in others:
            alt = frontiers.get((o, wname))
            if m and alt:
                result.frontier_comparisons[f"{wname}|{o}"] = compare_frontiers(
                    m, alt, n_budgets=32
                )
    _attribute(result)
    return result


def _attribute(result: AuditResult) -> None:
    """Attach a mechanism to each loss region, from the frontier metadata."""
    if result.loss_column is None:
        return
    for region in result.loss_column.regions:
        wls = [str(x) for x in region.bounds.get("workload", [])]
        parts: list[str] = []
        if region.contains_missing and region.missing_reasons:
            parts.append(region.missing_reasons[0])
        winners: set[str] = set()
        hits: list[float] = []
        engine_names = sorted({k.split("|")[0] for k in result.ledgers})
        for w in wls:
            for name in engine_names:
                if name == result.method:
                    continue
                fr = result.frontier(name, w)
                if fr and fr.points:
                    winners.add(name)
                    hits += [p.meta.get("prefix_hit_rate", 0.0) for p in fr.points]
            wl = result.workloads.get(w)
            if wl and wl.prefix_reuse_fraction > 0.15:
                parts.append(
                    f"{w} has {wl.prefix_reuse_fraction:.0%} prefix reuse, which the "
                    f"alternative caches and {result.method} does not at its tuned settings"
                )
        kv = [
            p.meta.get("kv_limited")
            for w in wls
            for p in (result.frontier(result.method, w).points if result.frontier(result.method, w) else [])
        ]
        if kv and sum(bool(x) for x in kv) > len(kv) / 2:
            parts.append("KV capacity, not compute, bounds the batch here")
        if not parts:
            parts.append("no mechanism identified from the measurement")
        region.attribution = "; ".join(dict.fromkeys(parts))
