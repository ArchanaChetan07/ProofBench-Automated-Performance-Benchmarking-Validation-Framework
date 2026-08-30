"""A component-level memory model for a transformer block. Version 2.

Version 1 is preserved unmodified in :mod:`losscolumn.thrusts.overlap.memcal`
and :mod:`losscolumn.thrusts.overlap.simulate`. It was wrong structurally, not
numerically, and measurement showed how: over eight configurations it agreed
with the hardware three times, and **every one of its errors was a false win**
-- it cleared configurations the card then refused. Region overlap 0.17,
boundary one level late, five false wins, zero false losses.

Retuning a constant cannot fix that shape of error. The defect was that v1
applied a single multiplier to the whole activation expression:

    act = tokens * hidden * bytes * 6
    if checkpointing:
        act *= 1/8            # <- one multiplier over everything

Checkpointing does not scale memory. It changes *which tensors are kept*. The
weights are unaffected. The optimizer state is unaffected. The workspace during
recompute is unaffected -- in fact one block's activations must still be live
while it is recomputed. Only the tensors saved *between* blocks for the
backward pass are dropped. A single multiplier cannot express that, and no
value of it can, which is why v1 was replaced rather than refitted.

Version 2 decomposes peak memory into terms that scale on different factors:

    weights            hidden, ffn                     -- not tokens
    gradients          hidden, ffn                     -- not tokens
    optimizer_state    hidden, ffn                     -- not tokens
    saved_activations  tokens, hidden, layers          -- CHECKPOINTING APPLIES
    attention_workspace tokens, heads, head_dim        -- one block live
    ffn_workspace      tokens, ffn                     -- one block live
    autograd_overhead  tokens, hidden
    allocator_reserve  a fraction of the rest, plus a fixed context cost

and applies checkpointing to exactly one of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from losscolumn.core.parameters import Parameter, ParameterSet, Provenance

MODEL_VERSION = "block-memory-v2"

GIB = 1024**3
GB = 1e9


# --------------------------------------------------------------------------
# the default parameter set
# --------------------------------------------------------------------------


def default_parameters() -> ParameterSet:
    """Parameters registered before any validation data was collected.

    Every value here is ``REGISTERED``: derived from the arithmetic of what a
    transformer block allocates, not fitted to the measurements this model is
    graded against. Fitting them on the calibration grid is permitted and marks
    them ``FITTED``; fitting them on the validation grid is not.
    """
    s = ParameterSet(name=MODEL_VERSION)
    s.add(Parameter(
        "activation_tensors_per_block", 4.0, Provenance.REGISTERED, unit="tensors",
        source="counted from the block's forward graph: the residual input, the "
               "attention output, the post-projection residual, and the FFN "
               "activation input",
        note="the tensors autograd must keep between blocks for the backward pass",
    ))
    s.add(Parameter(
        "attention_workspace_tensors", 4.0, Provenance.REGISTERED, unit="tensors",
        source="q, k, v and the fused attention output, live simultaneously inside "
               "one block",
    ))
    s.add(Parameter(
        "ffn_workspace_tensors", 2.0, Provenance.REGISTERED, unit="tensors",
        source="the up-projection output and its activation, live simultaneously",
    ))
    s.add(Parameter(
        "autograd_overhead_tensors", 2.0, Provenance.REGISTERED, unit="tensors",
        source="gradient buffers autograd materialises for the residual path",
    ))
    s.add(Parameter(
        "optimizer_bytes_per_param", 0.0, Provenance.REGISTERED, unit="bytes",
        source="zero for the local calibration harness, which holds gradients but "
               "runs no optimizer step",
        note="a training run with Adam would register 12: fp32 master weights plus "
             "two moments. It is a parameter precisely so the calibration harness "
             "and the simulated trainer do not have to share a number that is only "
             "true for one of them",
    ))
    s.add(Parameter(
        "gradient_bytes_per_param", 2.0, Provenance.REGISTERED, unit="bytes",
        source="one gradient per parameter at the model's own precision",
    ))
    s.add(Parameter(
        "checkpoint_retained_fraction", 1.0 / 4.0, Provenance.REGISTERED,
        unit="fraction",
        source="a checkpointed block keeps its input boundary tensor and discards "
               "the interior; one of four saved tensors survives",
        note="APPLIES ONLY to saved_activations. Weights, optimizer state and the "
             "live workspace are untouched by checkpointing, which is the "
             "structural error that made version 1 unusable",
    ))
    s.add(Parameter(
        "allocator_reserve_fraction", 0.12, Provenance.REGISTERED, unit="fraction",
        source="caching-allocator fragmentation headroom; a block-structured "
               "workload does not achieve perfect packing",
    ))
    s.add(Parameter(
        "context_bytes", 0.75 * GB, Provenance.REGISTERED, unit="bytes",
        source="CUDA context, kernel images and cuBLAS/cuDNN workspaces, "
               "independent of shape",
    ))
    s.add(Parameter(
        "safety_margin_fraction", 0.10, Provenance.REGISTERED, unit="fraction",
        source="uncertainty reserve, registered before validation and NOT tuned "
               "against it",
        note="A model used to decide what to launch must be conservative near the "
             "boundary, because its two errors are not equally bad. Tuning this "
             "against the validation grid would be fitting on the data the model "
             "is graded by, so its value is fixed in the sealed protocol",
    ))
    return s


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryTerms:
    """Peak memory, decomposed into terms that scale on different factors."""

    weights: float = 0.0
    gradients: float = 0.0
    optimizer_state: float = 0.0
    saved_activations: float = 0.0
    attention_workspace: float = 0.0
    ffn_workspace: float = 0.0
    autograd_overhead: float = 0.0
    allocator_reserve: float = 0.0
    context: float = 0.0

    @property
    def token_dependent(self) -> float:
        return (
            self.saved_activations + self.attention_workspace
            + self.ffn_workspace + self.autograd_overhead
        )

    @property
    def token_independent(self) -> float:
        return self.weights + self.gradients + self.optimizer_state + self.context

    @property
    def subtotal(self) -> float:
        return self.token_dependent + self.token_independent

    @property
    def total(self) -> float:
        return self.subtotal + self.allocator_reserve

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "gradients": self.gradients,
            "optimizer_state": self.optimizer_state,
            "saved_activations": self.saved_activations,
            "attention_workspace": self.attention_workspace,
            "ffn_workspace": self.ffn_workspace,
            "autograd_overhead": self.autograd_overhead,
            "allocator_reserve": self.allocator_reserve,
            "context": self.context,
            "token_dependent": self.token_dependent,
            "token_independent": self.token_independent,
            "total": self.total,
            "total_gb": self.total / GB,
        }

    def to_markdown(self) -> str:
        rows = [
            ("weights", self.weights), ("gradients", self.gradients),
            ("optimizer state", self.optimizer_state),
            ("saved activations", self.saved_activations),
            ("attention workspace", self.attention_workspace),
            ("FFN workspace", self.ffn_workspace),
            ("autograd overhead", self.autograd_overhead),
            ("context", self.context),
            ("allocator reserve", self.allocator_reserve),
        ]
        lines = ["| Term | GB | Share |", "|---|---|---|"]
        for name, v in rows:
            share = v / self.total if self.total else 0.0
            lines.append(f"| {name} | {v / GB:.3f} | {share:.0%} |")
        lines.append(f"| **total** | **{self.total / GB:.3f}** | |")
        return "\n".join(lines)


@dataclass(frozen=True)
class BlockShape:
    """The architecture whose memory is being predicted."""

    hidden: int
    heads: int
    ffn: int
    micro_batch: int
    seq_len: int
    n_layers: int = 1
    bytes_per_elem: int = 2
    checkpointing: bool = False

    @property
    def tokens(self) -> int:
        return self.micro_batch * self.seq_len

    @property
    def head_dim(self) -> int:
        return self.hidden // max(self.heads, 1)

    @property
    def params(self) -> int:
        """Parameters in one block: qkv, output projection, up and down."""
        return 3 * self.hidden * self.hidden + self.hidden * self.hidden \
            + 2 * self.hidden * self.ffn

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden, "heads": self.heads, "ffn": self.ffn,
            "micro_batch": self.micro_batch, "seq_len": self.seq_len,
            "n_layers": self.n_layers, "bytes_per_elem": self.bytes_per_elem,
            "checkpointing": self.checkpointing, "tokens": self.tokens,
            "head_dim": self.head_dim, "params": self.params,
        }


@dataclass
class BlockMemoryModel:
    """Version 2. Component terms, and checkpointing applied to one of them."""

    params: ParameterSet = field(default_factory=default_parameters)
    version: str = MODEL_VERSION

    # ---- the model -------------------------------------------------------

    def terms(self, shape: BlockShape) -> MemoryTerms:
        """Decompose predicted peak memory for one configuration."""
        p, b = self.params, float(shape.bytes_per_elem)
        tokens, hidden = float(shape.tokens), float(shape.hidden)
        layers = float(max(shape.n_layers, 1))

        weights = shape.params * b * layers
        gradients = shape.params * p.get("gradient_bytes_per_param") * layers
        optimizer = shape.params * p.get("optimizer_bytes_per_param") * layers

        # Saved activations: the tensors autograd keeps BETWEEN blocks. This is
        # the only term checkpointing touches, and it scales with the number of
        # layers because every layer keeps its own.
        saved = tokens * hidden * b * p.get("activation_tensors_per_block") * layers
        if shape.checkpointing:
            saved *= p.get("checkpoint_retained_fraction")

        # Workspace: live inside ONE block at a time, so it does not scale with
        # layer count and checkpointing does not reduce it. Under checkpointing
        # the recomputed block still materialises exactly this much -- which is
        # why a global multiplier under-predicted so badly at the boundary.
        attention = (
            tokens * float(shape.heads) * float(shape.head_dim) * b
            * p.get("attention_workspace_tensors")
        )
        ffn = tokens * float(shape.ffn) * b * p.get("ffn_workspace_tensors")
        autograd = tokens * hidden * b * p.get("autograd_overhead_tensors")

        context = p.get("context_bytes")
        subtotal = (
            weights + gradients + optimizer + saved + attention + ffn + autograd
            + context
        )
        reserve = subtotal * p.get("allocator_reserve_fraction")

        return MemoryTerms(
            weights=weights, gradients=gradients, optimizer_state=optimizer,
            saved_activations=saved, attention_workspace=attention,
            ffn_workspace=ffn, autograd_overhead=autograd,
            allocator_reserve=reserve, context=context,
        )

    def predict_bytes(self, shape: BlockShape) -> float:
        return self.terms(shape).total

    def predict_gb(self, shape: BlockShape) -> float:
        return self.predict_bytes(shape) / GB

    # ---- the decision ----------------------------------------------------

    def required_bytes(self, shape: BlockShape) -> float:
        """Predicted memory *plus* the registered safety margin.

        The margin is applied to the requirement rather than deducted from the
        budget so that both numbers stay visible: a reader can see the estimate
        and the reserve separately, and can recompute the decision under a
        different policy.
        """
        return self.predict_bytes(shape) * (
            1.0 + self.params.get("safety_margin_fraction")
        )

    def fits(self, shape: BlockShape, budget_bytes: float) -> bool:
        return self.required_bytes(shape) <= budget_bytes

    def headroom(self, shape: BlockShape, budget_bytes: float) -> float:
        """Fraction of the budget left over. Negative means predicted infeasible."""
        req = self.required_bytes(shape)
        return (budget_bytes - req) / budget_bytes if budget_bytes else float("nan")

    def boundary_distance(self, shape: BlockShape, budget_bytes: float) -> float:
        """|log| distance from the decision boundary. Zero is exactly on it.

        The quantity a boundary-focused grid sorts on: cells with a small value
        are where being slightly wrong changes the answer.
        """
        req = self.required_bytes(shape)
        if req <= 0 or budget_bytes <= 0:
            return float("inf")
        return abs(math.log(req / budget_bytes))

    def explain(self, shape: BlockShape, budget_bytes: float) -> str:
        t = self.terms(shape)
        req = self.required_bytes(shape)
        return (
            f"{t.total / GB:.2f} GB estimated + "
            f"{(req - t.total) / GB:.2f} GB margin = {req / GB:.2f} GB against a "
            f"{budget_bytes / GB:.2f} GB budget "
            f"({'fits' if req <= budget_bytes else 'does not fit'})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "parameters": self.params.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlockMemoryModel:
        return cls(params=ParameterSet.from_dict(d["parameters"]),
                   version=d.get("version", MODEL_VERSION))


# --------------------------------------------------------------------------
# scaling: local block -> simulated architecture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingMap:
    """The explicit correspondence between what is measured and what is modelled.

    Calibration runs on one small block on one card; the simulator models a
    7B-class model across many. The mapping between them is written down here
    rather than absorbed into a constant, because it is an assumption and
    assumptions have to be inspectable.

    What is claimed to transfer:

    * The **decomposition** -- that peak memory is the sum of these terms.
    * The **per-tensor counts** -- how many activations a block of this shape
      keeps, which is a property of the graph and not of its width.
    * The **allocator reserve fraction** -- a property of the allocator.

    What is *not* claimed to transfer, and is re-registered per target:

    * ``context_bytes``, which depends on the driver and the card.
    * ``optimizer_bytes_per_param``, which depends on the optimizer actually
      used -- zero for the calibration harness, twelve for Adam.
    * Anything sharded across ranks, which the local block cannot exercise at
      all.
    """

    local: BlockShape
    target_hidden: int
    target_heads: int
    target_ffn: int
    target_layers: int
    target_bytes_per_elem: int = 2
    shard_degree: int = 1
    transfers: tuple[str, ...] = (
        "activation_tensors_per_block",
        "attention_workspace_tensors",
        "ffn_workspace_tensors",
        "autograd_overhead_tensors",
        "checkpoint_retained_fraction",
        "allocator_reserve_fraction",
    )
    re_registered: tuple[str, ...] = (
        "context_bytes",
        "optimizer_bytes_per_param",
        "gradient_bytes_per_param",
    )

    def to_target(self, micro_batch: int, seq_len: int,
                  checkpointing: bool = False) -> BlockShape:
        return BlockShape(
            hidden=self.target_hidden, heads=self.target_heads,
            ffn=self.target_ffn, micro_batch=micro_batch, seq_len=seq_len,
            n_layers=self.target_layers,
            bytes_per_elem=self.target_bytes_per_elem,
            checkpointing=checkpointing,
        )

    @property
    def width_ratio(self) -> float:
        return self.target_hidden / max(self.local.hidden, 1)

    @property
    def depth_ratio(self) -> float:
        return self.target_layers / max(self.local.n_layers, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": self.local.to_dict(),
            "target": {
                "hidden": self.target_hidden, "heads": self.target_heads,
                "ffn": self.target_ffn, "layers": self.target_layers,
                "bytes_per_elem": self.target_bytes_per_elem,
                "shard_degree": self.shard_degree,
            },
            "width_ratio": self.width_ratio,
            "depth_ratio": self.depth_ratio,
            "parameters_claimed_to_transfer": list(self.transfers),
            "parameters_re_registered_per_target": list(self.re_registered),
            "caveat": (
                "Calibrating these parameters on the local block establishes them for "
                "block-shaped workloads on this allocator. It does not establish the "
                "re-registered parameters, and it exercises no sharding at all: the "
                "local block runs on one device."
            ),
        }

    def to_markdown(self) -> str:
        return "\n".join([
            "| | local (measured) | target (simulated) |",
            "|---|---|---|",
            f"| hidden | {self.local.hidden} | {self.target_hidden} |",
            f"| heads | {self.local.heads} | {self.target_heads} |",
            f"| FFN | {self.local.ffn} | {self.target_ffn} |",
            f"| layers | {self.local.n_layers} | {self.target_layers} |",
            f"| bytes/element | {self.local.bytes_per_elem} | "
            f"{self.target_bytes_per_elem} |",
            f"| shard degree | 1 | {self.shard_degree} |",
            "",
            f"Width ratio {self.width_ratio:.1f}x, depth ratio {self.depth_ratio:.0f}x.",
            "",
            "**Claimed to transfer**: "
            + ", ".join(f"`{t}`" for t in self.transfers)
            + " -- these are properties of the computation graph and of the "
              "allocator, not of the width.",
            "",
            "**Re-registered per target**: "
            + ", ".join(f"`{t}`" for t in self.re_registered)
            + " -- these depend on the device, the driver and the optimizer, and a "
              "value measured here would not be evidence about there.",
        ])
