"""Sparse attention patterns, as a block layout both implementations share.

A sparse attention kernel is only interesting if it actually skips work. That
means the pattern has to reach the kernel as *structure* -- which key blocks
each query block attends to -- rather than as a mask the kernel evaluates and
then multiplies by zero. A masked dense kernel and a block-sparse kernel
compute the same thing and have completely different loss maps, and conflating
them is the easiest way to publish a sparse-attention speedup that is really a
masking overhead.

The layout here is the standard compressed-row form:

``crow[i] .. crow[i+1]``   the slice of ``cols`` belonging to query block ``i``
``cols[j]``                a key-block index that query block ``i`` attends to

Both the portable PyTorch implementation and the Triton kernel consume exactly
this, so a comparison between them is about code generation and not about who
got an easier pattern.

Block size is fixed at 64x64 rather than autotuned. It is part of the sparsity
format, not a scheduling choice: a pattern expressed in 64-wide blocks is a
different pattern when reinterpreted in 128-wide blocks, and autotuning over it
would silently change the density being measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BLOCK_Q = 64
BLOCK_K = 64


@dataclass(frozen=True)
class BlockLayout:
    """A block-sparse attention pattern in compressed-row form."""

    crow: np.ndarray            # int32[n_q_blocks + 1]
    cols: np.ndarray            # int32[nnz]
    n_q_blocks: int
    n_k_blocks: int
    pattern: str
    requested_density: float
    block_q: int = BLOCK_Q
    block_k: int = BLOCK_K
    causal: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def nnz(self) -> int:
        return int(self.cols.size)

    @property
    def n_blocks_total(self) -> int:
        return self.n_q_blocks * self.n_k_blocks

    @property
    def realized_density(self) -> float:
        """Fraction of blocks actually kept.

        Reported separately from the requested density because rounding to
        whole blocks, the causal constraint and the always-kept diagonal all
        move it, and a paper that quotes the requested figure while measuring
        the realized one is off by an amount nobody can audit.
        """
        denom = self.n_blocks_total
        if self.causal:
            denom = sum(
                min(self.n_k_blocks, i + 1 + (self.n_k_blocks - self.n_q_blocks))
                for i in range(self.n_q_blocks)
            )
        return self.nnz / denom if denom else 1.0

    def dense_mask(self) -> np.ndarray:
        """Expand to a boolean [n_q_blocks, n_k_blocks] grid, for the reference."""
        m = np.zeros((self.n_q_blocks, self.n_k_blocks), dtype=bool)
        for i in range(self.n_q_blocks):
            for j in self.cols[self.crow[i]:self.crow[i + 1]]:
                m[i, int(j)] = True
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "requested_density": self.requested_density,
            "realized_density": self.realized_density,
            "n_q_blocks": self.n_q_blocks,
            "n_k_blocks": self.n_k_blocks,
            "nnz": self.nnz,
            "block": [self.block_q, self.block_k],
            "causal": self.causal,
            **self.meta,
        }


def _finish(rows: list[list[int]], *, pattern: str, density: float,
            n_k_blocks: int, causal: bool, **meta: Any) -> BlockLayout:
    crow = np.zeros(len(rows) + 1, dtype=np.int32)
    for i, r in enumerate(rows):
        crow[i + 1] = crow[i] + len(r)
    cols = np.array([c for r in rows for c in r], dtype=np.int32)
    return BlockLayout(
        crow=crow, cols=cols, n_q_blocks=len(rows), n_k_blocks=n_k_blocks,
        pattern=pattern, requested_density=density, causal=causal, meta=meta,
    )


def build_layout(
    seq_q: int,
    seq_k: int,
    *,
    pattern: str = "dense",
    density: float = 1.0,
    causal: bool = False,
    block_q: int = BLOCK_Q,
    block_k: int = BLOCK_K,
    seed: int = 0,
) -> BlockLayout:
    """Build the block layout for one (pattern, density, shape).

    Deterministic given its arguments: the random pattern is drawn from a
    seeded generator keyed on the shape, so the same cell of the sweep gets the
    same pattern on every replicate and in every implementation. A sparse
    benchmark that redraws its pattern per run is measuring the pattern
    distribution, not the kernel.
    """
    nq = max(math.ceil(seq_q / block_q), 1)
    nk = max(math.ceil(seq_k / block_k), 1)
    offset_blocks = nk - nq              # right-aligned causal mask

    def allowed(i: int) -> list[int]:
        hi = min(nk, i + offset_blocks + 1) if causal else nk
        return list(range(max(hi, 1)))

    if pattern == "dense":
        rows = [allowed(i) for i in range(nq)]
        return _finish(rows, pattern=pattern, density=1.0, n_k_blocks=nk, causal=causal)

    if pattern == "sliding_window":
        # A window of w key blocks ending at the query block. Density is
        # requested as a fraction of the key axis, floored at one block so the
        # diagonal always survives -- a query that attends to nothing is not a
        # sparsity pattern, it is a bug.
        w = max(int(round(density * nk)), 1)
        rows = []
        for i in range(nq):
            av = allowed(i)
            rows.append(av[-w:] if av else [0])
        return _finish(rows, pattern=pattern, density=density, n_k_blocks=nk,
                       causal=causal, window_blocks=w)

    if pattern == "block_sparse":
        # A seeded random subset, with the diagonal block always kept. Keeping
        # the diagonal is not cosmetic: a block-sparse pattern that can drop a
        # query's own neighbourhood produces outputs that are not comparable to
        # the dense result in any regime, and the correctness check would be
        # measuring a different function.
        rng = np.random.default_rng(seed + 1000 * nq + nk)
        rows = []
        for i in range(nq):
            av = allowed(i)
            if not av:
                rows.append([0])
                continue
            keep = max(int(round(density * len(av))), 1)
            diag = min(i + offset_blocks, nk - 1)
            chosen = {diag} if diag in av else {av[-1]}
            pool = [c for c in av if c not in chosen]
            if keep > len(chosen) and pool:
                extra = rng.choice(pool, size=min(keep - len(chosen), len(pool)),
                                   replace=False)
                chosen |= {int(x) for x in np.atleast_1d(extra)}
            rows.append(sorted(chosen))
        return _finish(rows, pattern=pattern, density=density, n_k_blocks=nk,
                       causal=causal, seed=seed)

    raise ValueError(
        f"unknown sparsity pattern {pattern!r}; expected one of "
        "dense, sliding_window, block_sparse"
    )


PATTERNS = ("dense", "sliding_window", "block_sparse")


def is_applicable(pattern: str, density: float) -> tuple[bool, str]:
    """Whether a (pattern, density) pair is a meaningful configuration.

    The factor lattice is a product space, and pattern x density is not a valid
    one: dense at 25% density is a contradiction and sliding-window at 100% is
    just dense under another name. Rather than quietly dropping the invalid
    corners -- which would leave the artifact claiming a full factorial it did
    not run -- they are recorded as inapplicable with this reason, and the
    envelope carries them as unmeasurable cells.
    """
    if pattern == "dense":
        if density != 1.0:
            return False, "dense attention has density 1.0 by definition"
        return True, ""
    if density >= 1.0:
        return False, f"{pattern} at density 1.0 is dense attention under another name"
    return True, ""


def attention_flops_sparse(
    batch: int, heads: int, head_dim: int, layout: BlockLayout, seq_q: int, seq_k: int
) -> float:
    """FLOPs for the blocks actually computed.

    Counting the dense-equivalent FLOPs and dividing by the sparse kernel's
    runtime is how a sparse kernel is made to look arbitrarily fast: the
    "speedup" is then just the density. This counts the work the kernel really
    does, so TFLOP/s stays a measure of efficiency and the density appears where
    it belongs -- as a separate, reported axis.
    """
    per_block = 2.0 * 2.0 * layout.block_q * layout.block_k * head_dim
    return batch * heads * layout.nnz * per_block


def dense_equivalent_flops(batch: int, heads: int, head_dim: int, seq_q: int,
                           seq_k: int, causal: bool = False) -> float:
    full = 2.0 * 2.0 * batch * heads * seq_q * seq_k * head_dim
    return full * (0.5 if causal else 1.0)
