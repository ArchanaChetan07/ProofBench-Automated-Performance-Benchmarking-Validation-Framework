"""A from-scratch FlashAttention forward kernel in Triton.

Scope, stated plainly because it is the honest part of Thrust III: this is a
faithful implementation of the FlashAttention *algorithm* -- tiling, online
softmax with running rescale, no materialised score matrix, fp32 accumulation.
It is not a reimplementation of FlashAttention-3's *implementation*. FA-3's
headline mechanisms are warp specialisation with a producer/consumer split,
TMA-driven asynchronous copies, and ping-pong scheduling that overlaps softmax
with the next GEMM -- and those are Hopper features expressed through
architecture-specific asynchrony that Triton's programming model does not
expose. FP8 support likewise depends on per-block scaling machinery outside
this kernel.

So the expected result is a loss, over most of the space, and the contribution
is the map of *where* and the account of *why*. A reader who wants the fastest
kernel should use the reference; the artifact says so on its front page.

Where the ceiling comes from, concretely:

* no producer/consumer warp specialisation -- loads and math contend
* no TMA -- descriptor-based bulk copies become ordinary vectorised loads
* no software pipelining across the softmax -- the second GEMM waits
* Triton picks its own register allocation, so occupancy is not directly tunable
"""

from __future__ import annotations

import importlib.util
import math
import os
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Triton is imported lazily, never at module load. Two reasons, both learned
# the hard way: a broken or partial install can fault the interpreter during
# import rather than raising something catchable, and `attention_flops` below
# is pure arithmetic that the rest of the package needs on machines with no
# Triton at all. Nothing here touches Triton until a caller asks for the kernel.
_TRITON: Any = None
_TL: Any = None
_KERNEL: Any = None
_PROBE: tuple[bool, str] | None = None


def _import_triton() -> tuple[bool, str]:
    """Import Triton once, and remember how it went."""
    global _TRITON, _TL, _PROBE
    if _PROBE is not None:
        return _PROBE
    if os.environ.get("LC_DISABLE_TRITON"):
        _PROBE = (False, "disabled by LC_DISABLE_TRITON")
        return _PROBE
    if importlib.util.find_spec("triton") is None:
        _PROBE = (False, "triton is not installed (no official Windows wheel; "
                         "try triton-windows)")
        return _PROBE
    try:
        import triton
        import triton.language as tl
    except BaseException as e:  # a broken install can raise almost anything
        _PROBE = (False, f"triton is installed but unusable: {type(e).__name__}: {e}")
        return _PROBE
    _TRITON, _TL = triton, tl
    _PROBE = (True, "")
    return _PROBE


def triton_available() -> tuple[bool, str]:
    """Whether a usable Triton + CUDA pair is present, and why not if not."""
    ok, why = _import_triton()
    if not ok:
        return False, why
    if torch is None or not torch.cuda.is_available():
        return False, "no CUDA device visible to torch"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        return (
            False,
            f"compute capability {cap[0]}.{cap[1]} lacks bf16 tensor cores; the kernel "
            "targets sm_80 and above",
        )
    return True, ""


def _build_kernel() -> Any:
    """Compile-time construction of the Triton kernel, on first use only."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    ok, why = _import_triton()
    if not ok:
        raise RuntimeError(f"Triton path unavailable: {why}")
    triton, tl = _TRITON, _TL

    def _configs() -> list[Any]:
        out = []
        for bm in (64, 128):
            for bn in (32, 64, 128):
                for w, s in ((4, 3), (8, 3), (4, 4)):
                    out.append(
                        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
                    )
        return out

    @triton.autotune(configs=_configs(), key=["SEQ_Q", "SEQ_K", "HEAD_DIM", "IS_CAUSAL"])
    @triton.jit
    def _fwd_kernel(
        Q, K, V, Out, LSE,
        sqz, sqh, sqm, sqd,
        skz, skh, skn, skd,
        svz, svh, svn, svd,
        soz, soh, som, sod,
        slz, slh, slm,
        SEQ_Q, SEQ_K,
        softmax_scale,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program computes one BLOCK_M block of queries for one (batch, head)."""
        # The grid is (query blocks, batch * heads); batch and head are already
        # flattened by the caller, so a single index addresses both.
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q + pid_bh * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        m_mask = offs_m < SEQ_Q
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        # Running statistics of the online softmax, always in fp32.
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        offset = SEQ_K - SEQ_Q
        hi = tl.minimum(SEQ_K, (pid_m + 1) * BLOCK_M + offset) if IS_CAUSAL else SEQ_K

        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            n_idx = start_n + offs_n
            n_mask = n_idx < SEQ_K
            k_ptrs = K + pid_bh * skh + n_idx[:, None] * skn + offs_d[None, :] * skd
            v_ptrs = V + pid_bh * svh + n_idx[:, None] * svn + offs_d[None, :] * svd
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            if IS_CAUSAL:
                qk = tl.where(n_idx[None, :] <= offs_m[:, None] + offset, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # A block that is entirely masked leaves m_new at -inf. Guarding it
            # keeps exp(-inf - -inf) from producing NaN, which is the classic
            # causal-attention kernel bug and only fires on the first block.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            p = tl.exp(qk - m_safe[:, None])
            p = tl.where(qk == float("-inf"), 0.0, p)

            acc = acc * alpha[:, None]
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]
        o_ptrs = Out + pid_bh * soh + offs_m[:, None] * som + offs_d[None, :] * sod
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask[:, None])
        if LSE is not None:
            lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log(l_safe))
            tl.store(LSE + pid_bh * slh + offs_m * slm, lse, mask=m_mask)

    _KERNEL = _fwd_kernel
    return _KERNEL


def flash_attention_triton(q, k, v, *, causal: bool = False,
                           softmax_scale: float | None = None, return_lse: bool = False):
    """Run the Triton kernel. Inputs are ``(batch, heads, seq, head_dim)``."""
    ok, why = triton_available()
    if not ok:
        raise RuntimeError(f"Triton path unavailable: {why}")
    triton = _TRITON
    kernel = _build_kernel()
    assert q.dim() == 4, "expected (batch, heads, seq, head_dim)"
    b, h, sq, d = q.shape
    sk = k.shape[-2]
    if d not in (16, 32, 64, 128, 256):
        raise ValueError(f"head_dim {d} is not a supported power of two for this kernel")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    q, k, v = (x.contiguous() for x in (q, k, v))
    qf = q.view(b * h, sq, d)
    kf = k.contiguous().view(b * h, sk, d)
    vf = v.contiguous().view(b * h, sk, d)
    out = torch.empty_like(qf)
    lse = torch.empty((b * h, sq), device=q.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(sq, meta["BLOCK_M"]), b * h)  # noqa: E731
    kernel[grid](
        qf, kf, vf, out, lse,
        0, qf.stride(0), qf.stride(1), qf.stride(2),
        0, kf.stride(0), kf.stride(1), kf.stride(2),
        0, vf.stride(0), vf.stride(1), vf.stride(2),
        0, out.stride(0), out.stride(1), out.stride(2),
        0, lse.stride(0), lse.stride(1),
        sq, sk, scale,
        HEAD_DIM=d, IS_CAUSAL=causal,
    )
    o = out.view(b, h, sq, d)
    return (o, lse.view(b, h, sq)) if return_lse else o


def attention_flops(batch: int, heads: int, seq_q: int, seq_k: int, head_dim: int,
                    causal: bool = False) -> float:
    """FLOPs for one forward pass: the QK^T and PV GEMMs, 2 FLOPs per MAC.

    The causal case does slightly more than half the work, not exactly half:
    the diagonal blocks are computed in full and then masked. Reporting the
    exact-half figure inflates causal TFLOP/s by a few percent, which is the
    kind of small dishonesty that accumulates across a table.
    """
    full = 2.0 * 2.0 * batch * heads * seq_q * seq_k * head_dim
    return full * (0.5 if causal else 1.0)
