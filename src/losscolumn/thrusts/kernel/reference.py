"""Reference attention implementations and the ground truth to check against.

Three implementations, for three different jobs:

``math_reference``   fp64, materialised softmax, no tricks. Slow and exact
                     enough to serve as ground truth. Everything else is
                     measured against this, never against another fast kernel.
``sdpa_reference``   ``torch.nn.functional.scaled_dot_product_attention`` --
                     the reference the reimplementation is trying to match, and
                     the thing a reader should use if they want speed.
``flash_torch``      the FlashAttention algorithm -- block-wise online softmax
                     with running rescale -- written in plain PyTorch ops.

The third is the load-bearing one for this project. Writing the algorithm out
in portable ops separates two questions that a single Triton kernel conflates:
*is the algorithm implemented correctly* and *is this particular code generator
producing fast code on this particular architecture*. The first is answerable
on any GPU, including the sm_75 card this was developed on; the second needs
the target hardware. Sequencing them this way is what lets Thrust III validate
its own correctness suite before renting anything.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False


def _require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError("this path requires torch; install losscolumn[torch]")


def math_reference(q, k, v, *, causal: bool = False, softmax_scale: float | None = None):
    """Exact attention in float64. Ground truth, not a baseline to beat.

    Computed in double precision on purpose: comparing a bf16 kernel against a
    bf16 reference measures agreement between two approximations, which is not
    the same as measuring correctness, and it hides errors that both share.
    """
    _require_torch()
    qd, kd, vd = (x.to(torch.float64) for x in (q, k, v))
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qd.shape[-1])
    scores = torch.matmul(qd, kd.transpose(-1, -2)) * scale
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        i = torch.arange(sq, device=scores.device).view(-1, 1)
        j = torch.arange(sk, device=scores.device).view(1, -1)
        scores = scores.masked_fill(j > i + (sk - sq), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, vd)


def sdpa_reference(q, k, v, *, causal: bool = False, softmax_scale: float | None = None):
    """The production reference: PyTorch's fused SDPA."""
    _require_torch()
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=softmax_scale
    )


def flash_torch(
    q,
    k,
    v,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    block_q: int = 128,
    block_k: int = 64,
    return_lse: bool = False,
):
    """FlashAttention forward, written out in portable PyTorch ops.

    This is the algorithm, not a wrapper: query blocks stream over key blocks,
    the running max and running sum are carried across blocks, and the
    accumulator is rescaled whenever the max moves. The full ``s x s`` score
    matrix is never materialised -- only ``block_q x block_k`` at a time -- so
    the memory behaviour matches the real kernel even though the arithmetic
    runs through eager ops.

    Shapes follow SDPA: ``(batch, heads, seq, head_dim)``.
    """
    _require_torch()
    b, h, sq, d = q.shape
    sk = k.shape[-2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    # Accumulate in fp32 regardless of input dtype. Accumulating an online
    # softmax in bf16 loses the running sum long before it loses the max, and
    # the failure is silent.
    acc = torch.zeros((b, h, sq, d), device=q.device, dtype=torch.float32)
    m_i = torch.full((b, h, sq), float("-inf"), device=q.device, dtype=torch.float32)
    l_i = torch.zeros((b, h, sq), device=q.device, dtype=torch.float32)
    offset = sk - sq  # right-aligned causal mask, as SDPA defines it

    for q0 in range(0, sq, block_q):
        q1 = min(q0 + block_q, sq)
        qb = q[:, :, q0:q1, :].to(torch.float32)
        acc_b = acc[:, :, q0:q1, :]
        m_b = m_i[:, :, q0:q1]
        l_b = l_i[:, :, q0:q1]
        k_end = min(q1 + offset, sk) if causal else sk

        for k0 in range(0, k_end, block_k):
            k1 = min(k0 + block_k, k_end)
            kb = k[:, :, k0:k1, :].to(torch.float32)
            vb = v[:, :, k0:k1, :].to(torch.float32)
            s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
            if causal:
                qi = torch.arange(q0, q1, device=q.device).view(-1, 1)
                kj = torch.arange(k0, k1, device=q.device).view(1, -1)
                s = s.masked_fill(kj > qi + offset, float("-inf"))
            m_new = torch.maximum(m_b, s.amax(dim=-1))
            # A fully masked block leaves m_new at -inf; exp(-inf - -inf) is
            # NaN, so the correction is clamped. This is the single most common
            # bug in hand-written flash kernels and it only shows up on causal
            # inputs whose first key block is entirely masked.
            m_safe = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
            corr = torch.exp(m_b - m_safe)
            corr = torch.where(torch.isneginf(m_b), torch.zeros_like(corr), corr)
            p = torch.exp(s - m_safe.unsqueeze(-1))
            p = torch.nan_to_num(p, nan=0.0)
            acc_b = acc_b * corr.unsqueeze(-1) + torch.matmul(p, vb)
            l_b = l_b * corr + p.sum(dim=-1)
            m_b = m_new

        acc[:, :, q0:q1, :] = acc_b
        m_i[:, :, q0:q1] = m_b
        l_i[:, :, q0:q1] = l_b

    out = acc / l_i.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    out = out.to(q.dtype)
    if return_lse:
        return out, (m_i + torch.log(l_i.clamp_min(torch.finfo(torch.float32).tiny)))
    return out


def error_budget(
    dtype: Any, seq_len: int, head_dim: int, *, fp32_accumulation: bool = True
) -> dict[str, float]:
    """Tolerances derived from the arithmetic, not from what happened to pass.

    The budget has to match the arithmetic the kernel actually performs. With
    fp32 accumulation -- which both the Triton kernel and the portable
    implementation use -- the error is dominated by rounding the inputs into
    the storage format and rounding the result back out, roughly ``2u`` each;
    the accumulation itself contributes only ``sqrt(n) * u_fp32``, three orders
    of magnitude smaller. The constant 4 covers softmax conditioning when the
    score distribution is peaked.

    This distinction is worth being precise about, because the loose
    alternative -- assuming the accumulator has the storage format's precision,
    giving ``u * sqrt(n)`` -- produces a tolerance about 150x the error a
    correct kernel actually shows, and would therefore accept a kernel that
    accumulates in fp16. That bug is real, common, and costs roughly
    ``sqrt(seq_len)`` in accuracy; a tolerance that cannot see it is decoration.

    Tolerances chosen by running the test and rounding up until it passes are
    not tolerances; they are a record of the bug you decided to keep.
    """
    _require_torch()
    u = {
        torch.float16: 2**-11,
        torch.bfloat16: 2**-8,
        torch.float32: 2**-24,
        torch.float64: 2**-53,
    }.get(dtype, 2**-8)
    u_acc = 2**-24 if fp32_accumulation else u
    rel = 4.0 * u + 8.0 * math.sqrt(seq_len) * u_acc
    return {
        "unit_roundoff": u,
        "accumulator_roundoff": u_acc,
        "rel_tol": rel,
        "abs_tol": rel * 4.0,   # outputs are O(1) after softmax; absolute floor
        "derivation": "4u_storage + 8*sqrt(seq)*u_accumulator",
    }


# --------------------------------------------------------------------------
# block-sparse attention
# --------------------------------------------------------------------------


def flash_torch_sparse(
    q,
    k,
    v,
    layout,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
):
    """FlashAttention over a block layout, in portable PyTorch ops.

    Same online-softmax algorithm as :func:`flash_torch`, but the inner loop
    visits only the key blocks the layout names. Query blocks whose row is
    empty are impossible by construction -- :mod:`sparsity` always keeps the
    diagonal -- so no row can end with a zero normaliser.

    The block size comes from the layout, not from a tuning parameter: it is
    part of the sparsity format.
    """
    _require_torch()
    b, h, sq, d = q.shape
    sk = k.shape[-2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    bq, bk = layout.block_q, layout.block_k

    acc = torch.zeros((b, h, sq, d), device=q.device, dtype=torch.float32)
    m_i = torch.full((b, h, sq), float("-inf"), device=q.device, dtype=torch.float32)
    l_i = torch.zeros((b, h, sq), device=q.device, dtype=torch.float32)
    offset = sk - sq

    crow = layout.crow
    cols = layout.cols
    for bi in range(layout.n_q_blocks):
        q0, q1 = bi * bq, min((bi + 1) * bq, sq)
        if q1 <= q0:
            continue
        qb = q[:, :, q0:q1, :].to(torch.float32)
        acc_b, m_b, l_b = acc[:, :, q0:q1, :], m_i[:, :, q0:q1], l_i[:, :, q0:q1]

        for idx in range(int(crow[bi]), int(crow[bi + 1])):
            kj = int(cols[idx])
            k0, k1 = kj * bk, min((kj + 1) * bk, sk)
            if k1 <= k0:
                continue
            kb_ = k[:, :, k0:k1, :].to(torch.float32)
            vb = v[:, :, k0:k1, :].to(torch.float32)
            s = torch.matmul(qb, kb_.transpose(-1, -2)) * scale
            if causal:
                qi = torch.arange(q0, q1, device=q.device).view(-1, 1)
                kjx = torch.arange(k0, k1, device=q.device).view(1, -1)
                s = s.masked_fill(kjx > qi + offset, float("-inf"))
            m_new = torch.maximum(m_b, s.amax(dim=-1))
            m_safe = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
            corr = torch.exp(m_b - m_safe)
            corr = torch.where(torch.isneginf(m_b), torch.zeros_like(corr), corr)
            p = torch.nan_to_num(torch.exp(s - m_safe.unsqueeze(-1)), nan=0.0)
            acc_b = acc_b * corr.unsqueeze(-1) + torch.matmul(p, vb)
            l_b = l_b * corr + p.sum(dim=-1)
            m_b = m_new

        acc[:, :, q0:q1, :] = acc_b
        m_i[:, :, q0:q1] = m_b
        l_i[:, :, q0:q1] = l_b

    out = (acc / l_i.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)).to(q.dtype)
    if return_lse:
        return out, m_i + torch.log(l_i.clamp_min(torch.finfo(torch.float32).tiny))
    return out


def sparse_reference(q, k, v, layout, *, causal: bool = False,
                     softmax_scale: float | None = None):
    """The reference for a sparse pattern: SDPA computing it densely under a mask.

    This is the honest comparator. It produces the same output as the sparse
    kernel and does the full dense work, so the difference between them is
    exactly what block-skipping buys -- which is the question. Comparing a
    sparse kernel against *unmasked* dense attention would instead be comparing
    two different functions.
    """
    _require_torch()
    b, h, sq, d = q.shape
    sk = k.shape[-2]
    mask = torch.zeros((sq, sk), device=q.device, dtype=torch.bool)
    bq, bk = layout.block_q, layout.block_k
    for bi in range(layout.n_q_blocks):
        q0, q1 = bi * bq, min((bi + 1) * bq, sq)
        for idx in range(int(layout.crow[bi]), int(layout.crow[bi + 1])):
            kj = int(layout.cols[idx])
            mask[q0:q1, kj * bk:min((kj + 1) * bk, sk)] = True
    if causal:
        i = torch.arange(sq, device=q.device).view(-1, 1)
        j = torch.arange(sk, device=q.device).view(1, -1)
        mask &= j <= i + (sk - sq)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask.view(1, 1, sq, sk), scale=softmax_scale
    )


def math_reference_sparse(q, k, v, layout, *, causal: bool = False,
                          softmax_scale: float | None = None):
    """fp64 ground truth for a sparse pattern."""
    _require_torch()
    qd, kd, vd = (x.to(torch.float64) for x in (q, k, v))
    sq, sk = qd.shape[-2], kd.shape[-2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qd.shape[-1])
    scores = torch.matmul(qd, kd.transpose(-1, -2)) * scale
    keep = torch.zeros((sq, sk), device=q.device, dtype=torch.bool)
    bq, bk = layout.block_q, layout.block_k
    for bi in range(layout.n_q_blocks):
        q0, q1 = bi * bq, min((bi + 1) * bq, sq)
        for idx in range(int(layout.crow[bi]), int(layout.crow[bi + 1])):
            kj = int(layout.cols[idx])
            keep[q0:q1, kj * bk:min((kj + 1) * bk, sk)] = True
    if causal:
        i = torch.arange(sq, device=q.device).view(-1, 1)
        j = torch.arange(sk, device=q.device).view(1, -1)
        keep &= j <= i + (sk - sq)
    scores = scores.masked_fill(~keep.view(1, 1, sq, sk), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vd)
