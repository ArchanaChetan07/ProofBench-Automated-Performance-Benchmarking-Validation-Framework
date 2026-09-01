"""The 8x A100 protocol: sealed here, executed there.

Written before the allocation exists, so the expensive time is spent measuring
rather than deciding what to measure. Every stage below has a stated purpose, a
stated abort condition, and a stated cost if it is skipped -- because the
failure mode this document exists to prevent is arriving on eight A100s and
discovering that the first hour goes on topology surprises and the second on a
broken launcher.

**Nothing measured locally is a parameter here.** The local campaign ran gloo
over shared memory on one host. It cannot produce NVLink or InfiniBand
numbers, and no value from it is copied into an A100 fabric. What it does
establish, if it passes, is *methodology readiness*: that the grid construction,
the estimator families, the selection rule, the coverage criteria and the
readiness gate all work, and can be pointed at real hardware unchanged.

The order matters and is not negotiable on the day. The prediction is generated
and locked **before** the eight-GPU measurement, because a prediction produced
after seeing the measurement is not a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.provenance import content_hash, utcnow

A100_PROTOCOL_VERSION = "thrust1-a100-v1"


@dataclass
class Stage:
    """One stage of the campaign, with the reason it exists."""

    order: int
    name: str
    purpose: str
    produces: str
    abort_if: str
    estimated_minutes: int
    cost_if_skipped: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order, "name": self.name, "purpose": self.purpose,
            "produces": self.produces, "abort_if": self.abort_if,
            "estimated_minutes": self.estimated_minutes,
            "cost_if_skipped": self.cost_if_skipped,
        }


STAGES: tuple[Stage, ...] = (
    Stage(
        1, "environment smoke test",
        "Prove the harness runs at all before anything expensive depends on it: "
        "import the package, run the test suite, confirm the measurement lock and "
        "the allocator cap behave as they do locally.",
        "a pass/fail record and the environment fingerprint",
        "the suite does not pass on the target machine",
        20,
        "an hour of allocation spent discovering a packaging problem",
    ),
    Stage(
        2, "topology verification",
        "Establish which GPUs are actually connected how. nvidia-smi topo -m, "
        "NVLink presence and width per pair, NUMA affinity, and whether the eight "
        "devices are one NVSwitch domain or two quads.",
        "a topology map recorded in the artifact",
        "the topology differs from the one the protocol assumes, in which case the "
        "grid of world sizes is re-derived and the change is recorded as a "
        "declared deviation before measurement",
        15,
        "fitting one fabric model across two fabrics, which is the error the "
        "local campaign's inter_node flag exists to prevent",
    ),
    Stage(
        3, "machine stability sentinel",
        "Measure the rented machine's own repeatability before anything is built "
        "on it: six probes at three nested levels -- repeats inside one process, "
        "several process launches, launches separated in time -- and derive the "
        "comparability criterion from the result rather than carrying one over "
        "from elsewhere. Registers the criterion before any session is compared "
        "against it, and produces the noise budget that says how many replicates "
        "the calibration stage must carry to reach its gate.",
        "a stability envelope, a registered comparability criterion, and the "
        "replicate count the calibration stage needs",
        "the launch-level noise floor exceeds the coverage gate, in which case the "
        "campaign cannot grade a model to the precision it requires and the "
        "allocation is released rather than spent proving it",
        25,
        "the failure this whole protocol was rewritten to prevent: a campaign "
        "measured at a replicate count nobody costed, whose residuals are then "
        "attributed to the model rather than to the instrument. Locally that cost "
        "a coverage matrix that read 9 cells as needing better models when 6 of "
        "them needed more repeats",
    ),
    Stage(
        4, "device and allocator verification",
        "Confirm memory capacity per device, that the allocator cap installs, and "
        "that an over-large allocation raises rather than falling back. This is "
        "the check that made local feasibility measurable at all.",
        "ProbeConfig with cap_installed true, per device",
        "the cap cannot be installed on any device",
        15,
        "feasibility measurements that mean 'eventually returned' rather than 'fits'",
    ),
    Stage(
        5, "fabric identification",
        "Determine, by measurement rather than by assumption, which transport each "
        "world size actually traverses: intra-node NVLink, intra-node PCIe "
        "fallback, or inter-node. A four-rank group with one rank per node runs at "
        "IB speed, not NVLink speed.",
        "a transport label per (world size, rank placement)",
        "a group's transport cannot be identified",
        30,
        "pricing an inter-node collective at NVLink bandwidth, which flatters every "
        "sharding recommendation that depends on it",
    ),
    Stage(
        6, "communication calibration",
        "Run the dense campaign grid, unchanged, against NCCL on the real fabric. "
        "Same sizes, same oversampled regions, same two passes, same repeats.",
        "raw PointRecords, per collective per world per pass",
        "peak reproducibility fails, which is reported rather than worked around",
        90,
        "",
    ),
    Stage(
        7, "parameter quality gates",
        "Family selection under every registered estimator, stratified k-fold "
        "cross-validation, worst-regime acceptance. Identical code to the local "
        "campaign; only the data differs.",
        "a ParameterVerdict per group and a coverage matrix",
        "no group is accepted, which ends the campaign honestly rather than "
        "loosening the gate",
        10,
        "",
    ),
    Stage(
        8, "model freeze",
        "Freeze the accepted communication parameters into a Fabric. Rejected and "
        "diagnostic parameters are recorded and remain unusable.",
        "a sealed Fabric with parameter provenance",
        "any rejected parameter is reachable as a model input",
        5,
        "",
    ),
    Stage(
        9, "Thrust I prediction generation",
        "Run the simulator with the frozen memory model and the frozen "
        "communication model, over the registered sharding grid, and extract the "
        "predicted loss regions.",
        "a predicted loss map and its regions",
        "the memory or communication model is not frozen",
        15,
        "",
    ),
    Stage(
        10, "prediction lock",
        "Seal the prediction: hash it, record the hash, and stop. This is the step "
        "that makes the next one a test rather than a fit.",
        "a sealed prediction hash",
        "the prediction is not sealed before measurement begins",
        5,
        "a comparison that cannot distinguish a good model from a well-chosen "
        "narrative, which is the whole failure this project is about",
    ),
    Stage(
        11, "independent 8-GPU measurement",
        "Measure the registered sharding grid under torchrun on all eight devices, "
        "with GPU exclusivity enforced and the registered replicate count.",
        "the measured envelope",
        "the measurement shares hardware with another job",
        180,
        "",
    ),
    Stage(
        12, "measured loss-region extraction",
        "Extract loss regions from the measurement using the same extractor, at "
        "the registered MDE and FDR level.",
        "the measured loss map",
        "the design check reports the sweep could not have declared a loss",
        10,
        "",
    ),
    Stage(
        13, "prediction against measurement",
        "Score the sealed prediction: region overlap, boundary displacement, false "
        "wins, false losses, dangerous-error score, per-regime error.",
        "the calibration report and the final Thrust I claim",
        "the prediction hash does not match what was sealed at stage 10",
        10,
        "",
    ),
)


@dataclass
class A100Protocol:
    """The sealed plan. Generated locally, executed on the allocation."""

    local_readiness: str = "not_ready"
    local_campaign_seal: str = ""
    memory_model_version: str = "block-memory-v2"
    memory_model_milestone: str = "v0.3.0-memory-v2"
    standard_version: str = ""
    sealed_at: str = field(default_factory=utcnow)
    notes: list[str] = field(default_factory=list)

    @property
    def total_minutes(self) -> int:
        return sum(s.estimated_minutes for s in STAGES)

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "kind": "a100-campaign-protocol",
            "protocol": A100_PROTOCOL_VERSION,
            "sealed_at": self.sealed_at,
            "status": "PREPARED, NOT EXECUTED",
            "standard_version": self.standard_version,
            "preconditions": {
                "local_communication_readiness": self.local_readiness,
                "local_campaign_seal": self.local_campaign_seal,
                "memory_model": self.memory_model_version,
                "memory_model_milestone": self.memory_model_milestone,
                "note": (
                    "The local readiness verdict gates whether this protocol should be "
                    "executed. NOT_READY does not make the protocol wrong; it makes "
                    "running it premature."
                ),
            },
            "stages": [s.to_dict() for s in STAGES],
            "estimated_total_minutes": self.total_minutes,
            "estimated_total_hours": round(self.total_minutes / 60, 1),
            "transferability": {
                "local_parameters_are_not_fabric_parameters": True,
                "statement": (
                    "No value measured locally is used here. The local campaign ran "
                    "gloo over shared memory on one host; it cannot produce NVLink or "
                    "InfiniBand parameters and none of its fits is copied into an A100 "
                    "Fabric. What transfers is the method: grid construction, "
                    "estimator families, selection rule, coverage criteria and the "
                    "readiness gate, all unchanged."
                ),
                "what_the_local_campaign_establishes": [
                    "methodology readiness",
                    "model-family readiness",
                    "coverage methodology",
                    "measurement protocol maturity",
                ],
                "what_it_does_not_establish": [
                    "NVLink latency or bandwidth",
                    "InfiniBand latency or bandwidth",
                    "any Thrust I communication parameter",
                    "that Thrust I communication is calibrated",
                ],
            },
            "ordering_is_binding": (
                "Stages 9 and 10 run before stage 11. A prediction produced after "
                "seeing the measurement is not a prediction, and no result from an "
                "out-of-order run is admissible."
            ),
            "session_rule": (
                "Stages 6 through 13 are one session. Nothing measured in them is "
                "pooled with anything measured outside them unless stage 3's "
                "criterion says the two are comparable, and stage 3 registers that "
                "criterion before any comparison is made against it. The local "
                "machine's variation was found to enter at the process launch "
                "rather than with elapsed time, so 'measure it all in one sitting' "
                "is not by itself sufficient: the rented machine gets its own "
                "sentinel because there is no reason to assume it behaves the same "
                "way."
            ),
            "restart_rule": (
                "A restart is not a resume. If any stage from 6 onward has to be "
                "re-run, everything downstream of it re-runs too, because a "
                "measurement taken after a restart belongs to a different launch "
                "and the launch is where this machine's variation was found to "
                "enter. Splicing a re-run stage into an otherwise finished campaign "
                "is the pooling LC-8.1 forbids, wearing a different name."
            ),
            "budget": {
                "estimated_hours": round(
                    sum(x.estimated_minutes for x in STAGES) / 60.0, 1),
                "hard_ceiling_hours": 10.0,
                "ceiling_rule": (
                    "At the ceiling the campaign stops and reports what it has, "
                    "whatever stage it reached. A partial campaign with a stated "
                    "stopping point is a result; an over-running one whose scope "
                    "was quietly trimmed to fit is not."
                ),
                "what_overrun_must_not_buy": [
                    "fewer replicates than stage 3's noise budget requires",
                    "a narrower message-size grid than the one registered",
                    "dropping a world size to save time",
                    "a widened coverage gate",
                ],
                "overrun_rationale": (
                    "Each of those trades a stated result for a nominally complete "
                    "one. The honest response to running out of allocation is a "
                    "campaign that covers less and says so, because the alternative "
                    "is a campaign whose gate was set by the clock."
                ),
            },
            "abort_policy": (
                "Each stage names its own abort condition. Aborting is a successful "
                "outcome of the stage: the allocation is expensive and a campaign that "
                "stops at stage 2 with a recorded topology surprise is worth more than "
                "one that proceeds on a wrong assumption."
            ),
            "notes": self.notes,
        }
        doc["seal_hash"] = content_hash(doc)
        return doc

    def to_markdown(self) -> str:
        d = self.to_dict()
        lines = [
            "### 8x A100 campaign protocol",
            "",
            f"**{d['status']}** &mdash; seal `{d['seal_hash'][7:23]}...`, "
            f"about {d['estimated_total_hours']} hours of allocation.",
            "",
            f"Local communication readiness at sealing: **{self.local_readiness}**. "
            "That verdict gates whether this should be executed; it does not make the "
            "protocol wrong, it makes running it premature.",
            "",
            "| # | Stage | Produces | Abort if | min |",
            "|---|---|---|---|---|",
        ]
        for s in STAGES:
            lines.append(
                f"| {s.order} | {s.name} | {s.produces} | {s.abort_if} | "
                f"{s.estimated_minutes} |"
            )
        lines += [
            "",
            "**Ordering is binding.** Stages 9 and 10 run before stage 11: a "
            "prediction produced after seeing the measurement is not a prediction.",
            "",
            "**No local parameter is used here.** The local campaign ran gloo over "
            "shared memory on one host. It cannot produce NVLink or InfiniBand "
            "numbers and none of its fits is copied into an A100 fabric. What "
            "transfers is the method.",
            "",
        ]
        return "\n".join(lines)
