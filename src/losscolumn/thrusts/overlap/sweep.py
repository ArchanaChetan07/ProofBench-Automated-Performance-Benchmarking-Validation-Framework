"""Thrust I -- the overlap envelope.

RQ1: for a fixed model and hardware, where does communication-computation
overlap break down, and do commonly-recommended sharding configurations sit at
an optimum or adjacent to a cliff?

The sweep produces four things, and the fourth is the one the standard cares
about:

1. A throughput envelope over ``(micro_batch, seq_len, activation_checkpointing)``
   for each sharding configuration.
2. An overlap-efficiency envelope over the same grid, from interval attribution
   of the traces rather than from summed kernel durations.
3. A cliff report per configuration: adjacent-level transitions that collapse
   throughput, with the bootstrap support for each.
4. A loss column for **the recommended configuration**, measured against the
   best alternative in the swept space at each cell. This is the direct answer
   to RQ1: not "FSDP full-shard is good" but "here is the region where
   following the standard recommendation costs you 24%, and here is what beats
   it there".

The oracle comparator is stated as a limitation, not hidden: "best of the four
configurations we swept" is an upper bound on what a practitioner would find,
and it flatters the alternatives by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from losscolumn.core.cliffs import CliffReport, cliff_adjacency, detect_cliffs
from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.losscolumn import LossColumn, extract_loss_column
from losscolumn.core.provenance import Provenance
from losscolumn.core.stats import CellComparison, compare_cells
from losscolumn.thrusts.overlap.attrib import OverlapMetrics, StepTrace, aggregate
from losscolumn.thrusts.overlap.simulate import (
    Fabric,
    ModelSpec,
    ShardingConfig,
    StepSimulator,
)

RECOMMENDED = "full_shard/tp1"
ORACLE = "best_of_swept"


def _stable_seed(name: str, cell: tuple[int, ...]) -> int:
    """Process-independent per-(system, cell) seed offset."""
    import zlib

    return zlib.crc32(f"{name}|{','.join(map(str, cell))}".encode()) % 997


class OverlapBackend(Protocol):
    """Anything that can turn a sharding configuration into step traces."""

    name: str
    evidence_class: str  # "measured" | "simulated"

    def feasible(self, cfg: ShardingConfig) -> tuple[bool, str]: ...

    def measure(self, cfg: ShardingConfig, *, steps: int, seed: int) -> list[StepTrace]: ...


@dataclass
class SyntheticOverlapBackend:
    """Model-driven backend. Every artifact it produces is stamped simulated."""

    model: ModelSpec = field(default_factory=ModelSpec)
    fabric: Fabric = field(default_factory=Fabric)
    hbm_gb: float = 80.0
    name: str = "synthetic"
    evidence_class: str = "simulated"

    def _sim(self, seed: int) -> StepSimulator:
        return StepSimulator(self.model, self.fabric, seed=seed)

    def feasible(self, cfg: ShardingConfig) -> tuple[bool, str]:
        return self._sim(0).feasible(cfg, hbm_gb=self.hbm_gb)

    def measure(self, cfg: ShardingConfig, *, steps: int = 3, seed: int = 0) -> list[StepTrace]:
        return self._sim(seed).simulate(cfg, steps=steps)


@dataclass
class SweepResult:
    """Everything Thrust I produces for one sweep."""

    throughput: Envelope
    overlap: Envelope
    configs: dict[str, ShardingConfig]
    metrics: dict[tuple[str, tuple[int, ...]], list[OverlapMetrics]] = field(default_factory=dict)
    cliffs: dict[str, CliffReport] = field(default_factory=dict)
    comparisons: list[CellComparison] = field(default_factory=list)
    loss_column: LossColumn | None = None
    adjacency: list[dict[str, Any]] = field(default_factory=list)
    exemplar_traces: dict[str, StepTrace] = field(default_factory=dict)
    backend: str = "synthetic"
    evidence_class: str = "simulated"

    def per_config_summary(self) -> list[dict[str, Any]]:
        rows = []
        for name in self.configs:
            ms = [m for (s, _c), lst in self.metrics.items() if s == name for m in lst]
            if not ms:
                continue
            agg = aggregate(ms)
            rows.append(
                {
                    "label": name,
                    "overlap_efficiency": agg["overlap_efficiency"]["median"],
                    "exposed_frac_of_step": agg["exposed_frac_of_step"]["median"],
                    "step_ms": agg["step_ms"]["median"],
                    "n_steps": agg["n_steps"],
                }
            )
        return sorted(rows, key=lambda r: r["overlap_efficiency"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "evidence_class": self.evidence_class,
            "configs": {k: v.to_dict() for k, v in self.configs.items()},
            "throughput_envelope": self.throughput.to_dict(),
            "overlap_envelope": self.overlap.to_dict(),
            "per_config": self.per_config_summary(),
            "cliffs": {k: v.to_dict() for k, v in self.cliffs.items()},
            "loss_column": self.loss_column.to_dict() if self.loss_column else None,
            "cliff_adjacency": self.adjacency,
        }


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "full_shard/tp1": {"strategy": "full_shard", "tp_degree": 1},
    "hybrid_shard/tp1": {"strategy": "hybrid_shard", "tp_degree": 1},
    "full_shard/tp2": {"strategy": "full_shard", "tp_degree": 2},
    "full_shard/tp8": {"strategy": "full_shard", "tp_degree": 8},
}


def run_overlap_sweep(
    backend: OverlapBackend | None = None,
    *,
    micro_batches: Sequence[int] = (1, 2, 4, 8),
    seq_lens: Sequence[int] = (512, 2048, 8192),
    checkpointing: Sequence[bool] = (False, True),
    world_sizes: Sequence[int] = (8, 32),
    configs: dict[str, dict[str, Any]] | None = None,
    replicates: int = 11,
    steps_per_replicate: int = 3,
    seed: int = 20260101,
    mde: float = 0.05,
    q_level: float = 0.05,
) -> SweepResult:
    """Run the full Thrust I sweep with interleaved replicates.

    Replicates are interleaved across configurations rather than blocked: on a
    shared cluster, thermal drift and neighbour noise move on minute scales, so
    a blocked design confounds drift with the effect being measured. Paired
    statistics downstream depend on this, and the envelope records it.

    The default of 11 replicates is not a round number. An exact paired
    sign-flip test over r replicates cannot produce a p-value below 2**-r, and
    Benjamini-Hochberg over this grid needs p <= q/m for the most extreme cell;
    at the default grid size that requires r >= 10. Running the sweep at r = 5
    would produce an empty loss column no matter how badly the recommendation
    lost -- see ``design_can_reject``.
    """
    backend = backend or SyntheticOverlapBackend()
    cfg_specs = configs or DEFAULT_CONFIGS
    factors = (
        Factor("micro_batch", tuple(micro_batches)),
        Factor("seq_len", tuple(seq_lens), unit="tokens", log_scale=True),
        Factor("checkpointing", tuple(checkpointing), ordered=False),
        Factor("world_size", tuple(world_sizes), unit="GPUs"),
    )
    systems = list(cfg_specs) + [ORACLE]

    thr = Envelope.allocate(
        factors,
        Metric("cluster_throughput", "tok/s", higher_is_better=True,
               description="tokens advanced per second by the whole job"),
        systems,
        replicates,
        interleaved=True,
        workload="7B-class decoder pretraining step",
        provenance=Provenance.capture(),
        meta={"backend": backend.name, "evidence_class": backend.evidence_class,
              "world_sizes": list(world_sizes)},
    )
    ovl = Envelope.allocate(
        factors,
        Metric("overlap_efficiency", "fraction", higher_is_better=True,
               description="1 - exposed communication / total communication"),
        list(cfg_specs),
        replicates,
        interleaved=True,
        provenance=thr.provenance,
        meta=dict(thr.meta),
    )

    result = SweepResult(
        throughput=thr,
        overlap=ovl,
        configs={},
        backend=backend.name,
        evidence_class=backend.evidence_class,
    )

    for cell in thr.cells():
        coords = thr.coords(cell)
        per_system_thr: dict[str, list[float]] = {}
        for name, spec in cfg_specs.items():
            cfg = ShardingConfig(
                world_size=int(coords["world_size"]),
                micro_batch=int(coords["micro_batch"]),
                seq_len=int(coords["seq_len"]),
                activation_checkpointing=bool(coords["checkpointing"]),
                **spec,
            )
            result.configs.setdefault(name, cfg)
            ok, why = backend.feasible(cfg)
            if not ok:
                thr.mark_missing(name, cell, why)
                ovl.mark_missing(name, cell, why)
                continue

            tps: list[float] = []
            effs: list[float] = []
            mlist: list[OverlapMetrics] = []
            for r in range(replicates):
                # Seed varies per (cell, system, replicate) so replicates are
                # independent. Python's built-in hash() is salted per process,
                # which would make the whole sweep unreproducible across runs;
                # crc32 over the canonical label is stable forever.
                traces = backend.measure(
                    cfg,
                    steps=steps_per_replicate,
                    seed=seed + 1000 * r + _stable_seed(name, cell),
                )
                ms = [t.attribute() for t in traces]
                mlist += ms
                step = float(np.median([m.step_ms for m in ms]))
                tps.append(cfg.tokens_per_step / (step / 1000.0))
                eff = [m.overlap_efficiency for m in ms if np.isfinite(m.overlap_efficiency)]
                effs.append(float(np.median(eff)) if eff else float("nan"))
                if name not in result.exemplar_traces and traces:
                    result.exemplar_traces[name] = traces[0]
            thr.put(name, cell, tps)
            ovl.put(name, cell, [e for e in effs])
            result.metrics[(name, cell)] = mlist
            per_system_thr[name] = tps

        # The oracle: at each cell, the best alternative to the recommendation.
        alts = {k: v for k, v in per_system_thr.items() if k != RECOMMENDED and v}
        if alts:
            best = max(alts, key=lambda k: float(np.median(alts[k])))
            thr.put(ORACLE, cell, alts[best])
            thr.meta.setdefault("oracle_choice", {})[str(list(cell))] = best
        elif per_system_thr.get(RECOMMENDED):
            thr.put(ORACLE, cell, per_system_thr[RECOMMENDED])
            thr.meta.setdefault("oracle_choice", {})[str(list(cell))] = RECOMMENDED
        else:
            thr.mark_missing(ORACLE, cell, "no configuration in the swept space could run here")

    for name in cfg_specs:
        result.cliffs[name] = detect_cliffs(thr, name, min_drop=0.15, seed=seed)

    result.comparisons = compare_cells(
        thr, RECOMMENDED, ORACLE, mde=mde, q=q_level, seed=seed, paired=True
    )
    result.loss_column = extract_loss_column(
        thr,
        result.comparisons,
        method=RECOMMENDED,
        baseline=ORACLE,
        mde=mde,
        q_level=q_level,
    )
    _attribute(result)

    recs = [
        {"name": "textbook single-node recipe", "micro_batch": 4, "seq_len": 2048,
         "checkpointing": True, "world_size": 8},
        {"name": "long-context default", "micro_batch": 1, "seq_len": 8192,
         "checkpointing": True, "world_size": 8},
        {"name": "same recipe scaled to 4 nodes", "micro_batch": 4, "seq_len": 2048,
         "checkpointing": True, "world_size": 32},
        {"name": "small-batch fine-tune at scale", "micro_batch": 1, "seq_len": 512,
         "checkpointing": False, "world_size": 32},
    ]
    recs = [
        r
        for r in recs
        if r["micro_batch"] in micro_batches
        and r["seq_len"] in seq_lens
        and r["world_size"] in world_sizes
    ]
    result.adjacency = cliff_adjacency(thr, result.cliffs[RECOMMENDED], recs)
    return result


def _attribute(result: SweepResult) -> None:
    """Attach a mechanism to each loss region, from the overlap envelope.

    Requirement LC-1 asks for an attributed cause, not just a location. The
    attribution here is mechanical and therefore checkable: for each region,
    compare the recommendation's overlap efficiency and exposed-time share
    against the configuration that beat it.
    """
    if result.loss_column is None:
        return
    thr, ovl = result.throughput, result.overlap
    choice = thr.meta.get("oracle_choice", {})
    for region in result.loss_column.regions:
        cells = [c for c in thr.cells() if _in_region(thr, c, region)]
        winners = {choice.get(str(list(c))) for c in cells} - {None}
        eff_rec = [
            float(np.nanmedian(ovl.replicates_at(RECOMMENDED, c)))
            for c in cells
            if ovl.is_measured(RECOMMENDED, c)
        ]
        exposed = [
            float(np.nanmedian([m.exposed_frac_of_step for m in result.metrics[(RECOMMENDED, c)]]))
            for c in cells
            if (RECOMMENDED, c) in result.metrics
        ]
        parts: list[str] = []
        # An out-of-memory region is not an overlap story, and describing it
        # with overlap numbers borrowed from the cells that did run would be
        # actively misleading.
        if region.contains_missing >= max(region.n_loss, 1) / 2:
            parts.append("the configuration cannot run here at all")
            parts += sorted(region.missing_reasons)[:1]
        else:
            if eff_rec:
                parts.append(f"overlap efficiency {np.median(eff_rec):.2f}")
            if exposed:
                parts.append(f"{np.median(exposed):.1%} of the step is exposed communication")
        if winners:
            parts.append(f"beaten by {', '.join(sorted(w for w in winners if w))}")
        region.attribution = "; ".join(parts) or "unattributed"


def _in_region(env: Envelope, cell, region) -> bool:
    coords = env.coords(cell)
    return all(coords[k] in v for k, v in region.bounds.items())
