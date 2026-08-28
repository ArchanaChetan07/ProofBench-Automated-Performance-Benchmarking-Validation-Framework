"""The three pre-registered protocols, as code.

Keeping the protocols in the repository rather than in a document is what makes
requirement LC-4 operational: the analysis reads its MDE, its FDR level, its
replicate count and its factor grid *from the sealed protocol*, so an analysis
that departs from the registration cannot run silently -- it either fails
verification or records a declared deviation.

Sealing is a separate, deliberate act (``losscolumn prereg seal``), and the
seal hash covers only the normative fields, so prose can be corrected without
invalidating the commitment.
"""

from __future__ import annotations

from typing import Any

from losscolumn.core.prereg import PreRegistration

AUTHORS = ["Archana Suresh Patil"]

# The minimum effect worth calling, per thrust. These are the numbers the whole
# project turns on, so the reasoning is recorded next to them rather than in a
# commit message.
#
#   Thrust I    5%  -- below this, a sharding change is not worth a migration.
#   Thrust II   5%  -- roughly one instance in twenty on a serving fleet.
#   Thrust III 10%  -- a kernel within 10% of the reference is a substitute for
#                      most purposes; the interesting result is the region
#                      where it is not.
MDE = {"I": 0.05, "II": 0.05, "III": 0.10}


def thrust_one() -> PreRegistration:
    return PreRegistration(
        title="Thrust I -- the overlap envelope",
        hypotheses={
            "H0": "The commonly recommended FSDP configuration (full shard, no tensor "
                  "parallelism) is within the MDE of the best configuration in the swept "
                  "space at every point of the operating envelope.",
            "H1": "There exist regions of the envelope where the recommended configuration "
                  "is beaten by at least the MDE, and those regions are identifiable from "
                  "communication-computation overlap.",
        },
        primary_outcome=(
            "The loss column of the recommended configuration against the best swept "
            "alternative, on cluster training throughput."
        ),
        secondary_outcomes=[
            "Overlap efficiency, from interval attribution of profiler traces.",
            "Throughput discontinuities between adjacent factor levels, with bootstrap support.",
            "Cliff exposure of each named recommended configuration.",
        ],
        metrics=[
            {"name": "cluster_throughput", "unit": "tok/s", "higher_is_better": True},
            {"name": "overlap_efficiency", "unit": "fraction", "higher_is_better": True},
        ],
        factors=[
            {"name": "micro_batch", "levels": [1, 2, 4, 8]},
            {"name": "seq_len", "levels": [512, 2048, 8192], "unit": "tokens"},
            {"name": "checkpointing", "levels": [False, True]},
            {"name": "world_size", "levels": [8, 32], "unit": "GPUs"},
        ],
        systems=["full_shard/tp1", "hybrid_shard/tp1", "full_shard/tp2", "full_shard/tp8",
                 "best_of_swept"],
        mde=MDE["I"],
        replicates=11,
        interleaved=True,
        decision_rules=[
            "A cell is a loss when the paired regression is at least the MDE and its "
            "Benjamini-Hochberg q-value over all cells is at most 0.05.",
            "A cell is a tie only when two one-sided tests establish equivalence within "
            "the MDE; a non-significant difference is reported as inconclusive, not as a tie.",
            "A configuration that cannot run in a cell is recorded as a loss in that cell, "
            "with the failure reason.",
            "A discontinuity is reported only when the degradation exceeds 15% and sits at "
            "least three robust deviations outside the pooled step distribution on its axis, "
            "and its bootstrap interval clears 15%.",
        ],
        exclusion_rules=[
            "The first five steps of every run are discarded as warm-up, by count, before "
            "any statistic is computed.",
            "A replicate whose measured step time exceeds five times the median of its cell "
            "is retained and reported, not dropped; outliers are the phenomenon.",
        ],
        stopping_rule="Fixed full-factorial grid, fixed replicate count, no interim analysis.",
        rationale=(
            "The replicate count is set by the arithmetic, not by convention: an exact "
            "paired sign-flip test over r replicates cannot produce a p-value below 2**-r, "
            "and Benjamini-Hochberg over a 48-cell grid needs p <= 0.05/48 for the most "
            "extreme cell. That requires r >= 10. A sweep at r = 5 would report an empty "
            "loss column no matter how badly the recommendation lost."
        ),
        loss_column_policy=(
            "Every region meeting the MDE at the stated FDR level is published in the main "
            "artifact above the wins, with an attributed mechanism drawn from the overlap "
            "measurement rather than from narrative."
        ),
        authors=AUTHORS,
    )


def thrust_two() -> PreRegistration:
    return PreRegistration(
        title="Thrust II -- the equal-tuning audit",
        hypotheses={
            "H0": "Under identical tuning budgets, the measured differences between "
                  "inference engines are within the MDE across the latency-budget range.",
            "H1": "A measurable share of published engine-to-engine difference does not "
                  "survive tuning-budget parity, and the differences that do survive are "
                  "confined to identifiable workload and latency regions.",
        },
        primary_outcome=(
            "The loss column of vLLM against the best alternative engine, on throughput "
            "attainable under a normalised p99 latency budget, across four workload traces."
        ),
        secondary_outcomes=[
            "The default-to-tuned gap for every engine on every workload.",
            "Full latency-throughput Pareto frontiers per engine per workload.",
            "The parity certificate, including effective-parity asymmetries.",
        ],
        metrics=[
            {"name": "attainable_throughput", "unit": "tok/s", "higher_is_better": True},
            {"name": "normalised_p99_latency", "unit": "ms/token", "higher_is_better": False},
        ],
        factors=[
            {"name": "workload", "levels": ["chat", "rag", "summarize", "agentic"]},
            {"name": "latency_budget_ms", "levels": [10, 15, 25, 50, 100, 250],
             "unit": "ms per output token"},
        ],
        systems=["vllm", "sglang", "trtllm", "best_alternative"],
        mde=MDE["II"],
        replicates=11,
        interleaved=True,
        tuning_budget_trials=40,
        search_algorithm="random-explore+coordinate-refine@1.0(n=40,explore=0.6)",
        decision_rules=[
            "Every engine receives exactly 40 trials per workload from the same search "
            "procedure, with per-engine seeds derived from one master seed.",
            "The tuning objective is goodput at concurrency 256 -- throughput that meets "
            "the workload's stated SLO -- identically for every engine.",
            "Every engine is measured twice per workload: at library defaults and at its "
            "tuned optimum. Both are reported.",
            "Engines are compared through attainment curves, not at a single operating "
            "point; the loss column is derived from where the attainment ratio falls "
            "below one.",
            "A budget that is still improving at exhaustion is reported as binding, and "
            "the affected engine's result is labelled a lower bound.",
        ],
        exclusion_rules=[
            "A configuration that fails to start or runs out of memory is recorded with "
            "its failure and counted against its budget; it is not retried at a milder "
            "setting.",
            "The first 32 requests of every replay are warm-up and excluded from timing.",
        ],
        stopping_rule="Fixed trial budget per engine per workload; no adaptive allocation.",
        rationale=(
            "Tuning is per workload so that 'this engine is worse here' is not confounded "
            "with 'this engine was tuned somewhere else'. Equal trial counts are nominal "
            "parity only; trials per dimension, fraction of space covered and tuning "
            "wall-clock are reported because equal budgets can still favour the engine "
            "with the smaller search space or the cheaper trial."
        ),
        authors=AUTHORS,
    )


def thrust_three() -> PreRegistration:
    return PreRegistration(
        title="Thrust III -- reproduction with a loss map",
        hypotheses={
            "H0": "A faithful from-scratch implementation of the FlashAttention algorithm "
                  "matches the reference within the MDE across the swept shape space.",
            "H1": "It does not, and the regions where it loses are attributable to specific "
                  "implementation mechanisms the reimplementation does not express.",
        },
        primary_outcome=(
            "A complete performance map over (head_dim, seq_len, batch, dtype) with every "
            "losing region marked and an attributed cause for each."
        ),
        secondary_outcomes=[
            "A correctness suite passing across the full swept space, independent of timing.",
            "An account of which architectural details of the reference mattered.",
        ],
        metrics=[
            {"name": "attention_throughput", "unit": "TFLOP/s", "higher_is_better": True},
            {"name": "max_relative_error", "unit": "ratio", "higher_is_better": False},
        ],
        factors=[
            {"name": "head_dim", "levels": [32, 64, 128]},
            {"name": "seq_len", "levels": [128, 512, 2048], "unit": "tokens"},
            {"name": "batch", "levels": [1, 4]},
            {"name": "dtype", "levels": ["float16", "bfloat16"]},
        ],
        systems=["reimplementation", "reference"],
        mde=MDE["III"],
        replicates=11,
        interleaved=True,
        decision_rules=[
            "Correctness is evaluated before timing at every shape. A shape that fails "
            "correctness is recorded as unmeasurable for the reimplementation and enters "
            "the comparison as a loss; its timing is never reported.",
            "Correctness tolerances are derived from the arithmetic -- 4u for the storage "
            "format plus 8*sqrt(seq)*u for the accumulator -- and are fixed before any "
            "measurement. They are not adjusted to make a shape pass.",
            "Timing uses CUDA events, an L2 flush between iterations, and interleaved A/B "
            "replicates; the statistic is the median of per-iteration device times.",
            "The reimplementation is expected to lose over much of the space. The result "
            "is the map and the attribution, not the aggregate.",
        ],
        exclusion_rules=[
            "Shapes that exceed device memory are recorded as unmeasurable with the "
            "allocation failure, for both arms.",
            "No shape is excluded after seeing its timing.",
        ],
        stopping_rule="Fixed full-factorial grid; correctness gate is evaluated first.",
        rationale=(
            "The MDE is 10% rather than 5% because a kernel within 10% of the reference is "
            "a practical substitute; the finding of interest is the region where it is far "
            "outside that, and a tighter MDE would fill the loss column with regions no "
            "reader would act on. Matching a mature hand-tuned kernel is not the goal and "
            "would not be credible."
        ),
        loss_column_policy=(
            "The performance map is the artifact. Losing regions are marked on the map "
            "itself, not listed in an appendix, and the artifact states plainly that a "
            "reader wanting the fastest kernel should use the reference."
        ),
        authors=AUTHORS,
    )


PROTOCOLS: dict[str, Any] = {
    "I": thrust_one,
    "II": thrust_two,
    "III": thrust_three,
}


def build(thrust: str) -> PreRegistration:
    key = thrust.upper().replace("THRUST", "").strip()
    key = {"1": "I", "2": "II", "3": "III"}.get(key, key)
    if key not in PROTOCOLS:
        raise KeyError(f"unknown thrust {thrust!r}; expected one of {sorted(PROTOCOLS)}")
    return PROTOCOLS[key]()
