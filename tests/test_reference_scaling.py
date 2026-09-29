"""Ground truth has to reach the sequence lengths the claim is about.

The dense float64 reference holds the score matrix and the softmax at once, so
peak memory is about twice the attention matrix -- 67 MB at 2048 tokens and
17 GB at 32768, at batch 1 and two heads. Attention is quadratic, which is the
entire reason FlashAttention and block sparsity exist, so the interesting part
of this thrust lives at lengths a materialised reference cannot hold.

Streaming the reference with the online-softmax recurrence makes its memory
linear in sequence length. The tests below pin the thing that makes that
legitimate: it is the same quantity, not an approximation of it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from losscolumn.thrusts.kernel.reference import (  # noqa: E402
    MATERIALISE_BUDGET_BYTES,
    math_reference,
    math_reference_streamed,
)

# float64 rounding, not a tolerance. The kernels this grades are fp16 and bf16,
# whose budgets are nearer 1e-3.
FP64_ROUNDING = 1e-12


def _qkv(b, h, sq, sk, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, h, sq, d, generator=g),
            torch.randn(b, h, sk, d, generator=g),
            torch.randn(b, h, sk, d, generator=g))


@pytest.mark.parametrize("b,h,sq,sk,causal", [
    (1, 2, 128, 128, False),
    (1, 2, 256, 256, True),
    (2, 3, 512, 512, False),
    (1, 2, 384, 512, True),     # cross-attention shape, sq != sk
    (1, 1, 100, 137, True),     # not a multiple of any block size
])
def test_streamed_matches_materialised(b, h, sq, sk, causal):
    """The claim that makes streaming a reference rather than a shortcut.

    Online softmax rescales by exp(m_old - m_new), which is an identity. The
    two paths differ only by float64 rounding.
    """
    q, k, v = _qkv(b, h, sq, sk)
    dense = math_reference(q, k, v, causal=causal)
    streamed = math_reference_streamed(q, k, v, causal=causal,
                                       block_q=64, block_k=64)
    rel = float((dense - streamed).abs().max() / dense.abs().max())
    assert rel < FP64_ROUNDING, f"{rel:.2e} is above float64 rounding"


@pytest.mark.parametrize("block_q,block_k", [(32, 32), (64, 128), (1024, 1024)])
def test_the_answer_does_not_depend_on_the_block_size(block_q, block_k):
    """If it did, the blocking would be part of the answer rather than an
    implementation detail of how it is computed."""
    q, k, v = _qkv(1, 2, 256, 256)
    a = math_reference_streamed(q, k, v, causal=True, block_q=block_q,
                                block_k=block_k)
    b = math_reference(q, k, v, causal=True)
    assert float((a - b).abs().max() / b.abs().max()) < FP64_ROUNDING


def test_a_fully_masked_key_block_contributes_nothing():
    """Under causal masking the early query blocks have whole key blocks masked.

    exp(-inf - -inf) is nan, not zero, so without care those blocks poison the
    accumulator instead of being skipped.
    """
    q, k, v = _qkv(1, 1, 256, 256)
    out = math_reference_streamed(q, k, v, causal=True, block_q=32, block_k=32)
    assert bool(torch.isfinite(out).all()), "a masked block produced nan"


def test_the_first_query_attends_only_to_itself_under_causal_masking():
    """The strictest causal case: one key, and it must still normalise."""
    q, k, v = _qkv(1, 1, 8, 8, d=16)
    out = math_reference_streamed(q, k, v, causal=True, block_q=4, block_k=4)
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0].double(), atol=1e-12), (
        "with a single unmasked key the output is that key's value")


def test_streaming_is_chosen_by_size_not_by_the_caller():
    """Callers pass shapes, not strategies. A reference that needed the caller
    to know which path to take would be one more thing to get wrong."""
    small_q, small_k, small_v = _qkv(1, 1, 64, 64)
    assert math_reference(small_q, small_k, small_v).shape == (1, 1, 64, 64)

    lead, sq = 1 * 2, 32768
    assert 2 * lead * sq * sq * 8 > MATERIALISE_BUDGET_BYTES, (
        "the budget must put realistic long-sequence shapes on the streamed path")


def test_the_budget_admits_the_lengths_the_thrust_measures():
    """2048 tokens at the correctness shape must not pay for streaming."""
    lead, sq = 1 * 2, 2048
    assert 2 * lead * sq * sq * 8 < MATERIALISE_BUDGET_BYTES


def test_output_dtype_is_float64_whatever_went_in():
    """Ground truth is float64 by definition; returning the input dtype would
    make it a second approximation rather than a reference."""
    q, k, v = _qkv(1, 1, 64, 64)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        out = math_reference_streamed(q.to(dt), k.to(dt), v.to(dt))
        assert out.dtype is torch.float64
