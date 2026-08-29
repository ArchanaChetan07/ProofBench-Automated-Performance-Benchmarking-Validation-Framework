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
    """Thrust III, revision 2: the sparse-attention lattice.

    Revision 1 registered a dense ``(head_dim, seq_len, batch, dtype)`` grid and
    was sealed and run. This revision supersedes it: the question RQ3 actually
    asks -- where does a from-scratch implementation lose, and why -- is not
    answerable on a dense grid alone, because the mechanisms that decide it
    (block skipping, query-block padding at decode) do not appear there. The
    v1 seal is archived and remains verifiable; this is a new registration,
    not an edit of it.
    """
    return PreRegistration(
        title="Thrust III -- sparse attention, prefill and decode, with a loss map",
        version="2",
        hypotheses={
            "H0": "Across the registered lattice, each from-scratch implementation is "
                  "within the MDE of the reference computing the same attention "
                  "pattern.",
            "H1": "It is not, and the regions where it loses are attributable to "
                  "identifiable mechanisms: block-skipping that fails to amortise at "
                  "low density, query-block padding at decode, and launch-bound "
                  "regimes at short sequence.",
        },
        primary_outcome=(
            "A loss column over (seq_len, batch, pattern, density, phase) on "
            "per-call attention latency, published separately for each "
            "implementation, with an attributed mechanism for every region."
        ),
        secondary_outcomes=[
            "Achieved throughput over the blocks actually computed.",
            "Peak device memory per call.",
            "Speedup relative to the reference, per phase.",
            "A correctness suite passing across the full lattice, independent of timing.",
        ],
        metrics=[
            {"name": "latency_ms", "unit": "ms", "higher_is_better": False,
             "description": "per-call attention latency; the time-to-first-token "
                            "contribution at prefill and the inter-token latency at "
                            "decode"},
            {"name": "attention_throughput", "unit": "TFLOP/s", "higher_is_better": True,
             "description": "FLOPs over the blocks actually computed, per second"},
            {"name": "peak_memory_mb", "unit": "MB", "higher_is_better": False},
            {"name": "max_relative_error", "unit": "ratio", "higher_is_better": False},
        ],
        factors=[
            {"name": "seq_len", "levels": [512, 1024, 2048, 4096], "unit": "tokens"},
            {"name": "batch", "levels": [1, 8]},
            {"name": "pattern", "levels": ["dense", "sliding_window", "block_sparse"]},
            {"name": "density", "levels": [0.125, 0.25, 1.0]},
            {"name": "phase", "levels": ["prefill", "decode"]},
        ],
        systems=["implementation", "reference"],
        mde=MDE["III"],
        q_level=0.05,
        alpha=0.05,
        replicates=11,
        interleaved=True,
        decision_rules=[
            "The two implementations -- portable eager ops and the fused Triton "
            "kernel -- are swept separately and published as separate claims. They "
            "answer different questions and are never pooled into one number.",
            "The comparator for a sparse pattern is the reference computing THAT "
            "pattern densely under a mask, never unmasked dense attention: the two "
            "compute different functions, and comparing against the wrong one turns "
            "the pattern's density into a spurious speedup.",
            "Correctness is evaluated against an fp64 ground truth before timing at "
            "every cell. A cell that fails is recorded as unmeasurable for that "
            "implementation and enters the comparison as a loss; its timing is never "
            "reported.",
            "Tolerances are derived from the arithmetic -- 4u for the storage format "
            "plus 8*sqrt(seq)*u for the accumulator -- fixed before measurement and "
            "never adjusted to make a cell pass.",
            "Throughput counts the FLOPs of the blocks actually computed, not the "
            "dense equivalent.",
            "Block size is fixed at 64x64 and is not autotuned: it is part of the "
            "sparsity format, and tuning over it would change the pattern being "
            "measured.",
            "Timing uses CUDA events, an L2 flush between iterations, and interleaved "
            "A/B replicates; the statistic is the median of per-iteration device "
            "times.",
            "A cell is a loss when the paired regression is at least the MDE and its "
            "Benjamini-Hochberg q-value over all measured cells is at most 0.05.",
        ],
        exclusion_rules=[
            "Combinations of pattern and density that are not a configuration -- "
            "dense below full density, and any sparse pattern at full density -- are "
            "recorded as inapplicable with that reason. They are not silently "
            "dropped: the artifact must not claim a full factorial it did not run.",
            "Shapes that exceed device memory are recorded as unmeasurable with the "
            "allocation failure, for both arms.",
            "No cell is excluded after its timing is seen.",
        ],
        stopping_rule=(
            "Fixed full-factorial lattice, fixed replicate count, no interim "
            "analysis and no adaptive allocation of replicates."
        ),
        tuning_budget_trials=0,
        search_algorithm="none: neither arm is tuned per cell",
        rationale=(
            "BASELINE TUNING POLICY. Neither arm receives per-cell tuning, and the "
            "policy is symmetric by construction rather than by intention. The "
            "reference is given its own best backend for each shape by torch's "
            "dispatcher, with no backend disabled and no manual restriction; the "
            "Triton implementation autotunes only its scheduling parameters (warps "
            "and pipeline stages) from a fixed configuration list, with block size "
            "held constant. Neither arm is hand-tuned per cell by the operator, so "
            "the effort asymmetry that tuning-budget parity exists to control is "
            "absent here -- which is why no parity certificate accompanies this "
            "thrust and LC-2 is recorded as not applicable.\n\n"
            "GPU BUDGET. The registered lattice is 144 cells, of which 80 are "
            "applicable configurations, swept twice (once per implementation) at 11 "
            "replicates of 10 timed iterations. On a single device that is roughly "
            "25 minutes of measurement per implementation, and the registration "
            "commits to running it complete rather than stopping when a result "
            "appears.\n\n"
            "REPLICATE COUNT. Eleven is set by the arithmetic, not by convention. An "
            "exact paired sign-flip test over r replicates cannot produce a p-value "
            "below 2**-r, and Benjamini-Hochberg over the applicable cells needs "
            "p <= 0.05/m for the most extreme cell; at m = 80 that requires r >= 11. "
            "A sweep at r = 5 would report an empty loss column however large the "
            "regression.\n\n"
            "MDE. Ten percent rather than five: an implementation within 10% of the "
            "reference is a practical substitute for most purposes, and the finding "
            "of interest is the region where it is far outside that. A tighter MDE "
            "would fill the loss column with regions no reader would act on."
        ),
        loss_column_policy=(
            "Each implementation's loss map is the artifact, and losing regions are "
            "marked on the map itself rather than listed in an appendix. Where an "
            "implementation is slower than the reference the artifact says so on its "
            "front page, and says that a reader wanting speed should use the "
            "reference."
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
