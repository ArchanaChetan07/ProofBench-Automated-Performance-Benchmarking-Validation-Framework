"""The baseline must pay only for the work the comparison is about.

This file exists because the same defect has now appeared twice in the same
function, and both times it inflated the method's win.

The first: the reference routed *every* pattern through an explicit attention
mask, including dense, and an explicit mask precludes torch's fused backend.
Against that handicapped baseline the Triton kernel reported zero losses -- a
clean sweep that was an artifact of the comparator.

The second, found by adversarial verification of an automated audit: the mask
itself was built inside the timed callable, one CUDA launch per non-zero block,
on the baseline arm alone. Measured at 15% to 24% of baseline latency against a
registered minimum effect size of 10%. Worse than the magnitude, the cost scales
with the layout's occupancy -- the sweep's own independent variable -- so it
bent the speedup-against-density curve rather than shifting it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from losscolumn.thrusts.kernel.reference import (  # noqa: E402
    build_sparse_mask,
    sparse_reference,
)
from losscolumn.thrusts.kernel.sparsity import build_layout  # noqa: E402


def _qkv(b=1, h=2, s=256, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, h, s, d, generator=g) for _ in range(3))


@pytest.mark.parametrize("pattern,density,causal", [
    ("block_sparse", 0.25, True),
    ("block_sparse", 0.125, False),
    ("sliding_window", 0.25, True),
    ("sliding_window", 0.5, False),
])
def test_prebuilt_mask_gives_the_same_answer(pattern, density, causal):
    """The hoist must not change what the reference computes.

    If it did, the fix would have traded a timing bug for a correctness one.
    """
    q, k, v = _qkv()
    layout = build_layout(256, 256, pattern=pattern, density=density, causal=causal)
    mask = build_sparse_mask(q, k, layout, causal=causal)

    with_mask = sparse_reference(q, k, v, layout, causal=causal, mask=mask)
    without = sparse_reference(q, k, v, layout, causal=causal)
    assert torch.equal(with_mask, without), (
        "supplying the mask must be an optimisation, not a different function")


@pytest.mark.parametrize("pattern,density,causal", [
    ("block_sparse", 0.25, True),
    ("sliding_window", 0.125, True),
    ("block_sparse", 1.0, False),
])
def test_the_vectorised_mask_matches_the_block_loop(pattern, density, causal):
    """Pins the construction itself against the loop it replaced."""
    q, k, _ = _qkv(s=192)
    layout = build_layout(192, 192, pattern=pattern, density=density, causal=causal)

    sq, sk = 192, 192
    ref = torch.zeros((sq, sk), dtype=torch.bool)
    bq, bk = layout.block_q, layout.block_k
    for bi in range(layout.n_q_blocks):
        q0, q1 = bi * bq, min((bi + 1) * bq, sq)
        for idx in range(int(layout.crow[bi]), int(layout.crow[bi + 1])):
            kj = int(layout.cols[idx])
            ref[q0:q1, kj * bk:min((kj + 1) * bk, sk)] = True
    if causal:
        i = torch.arange(sq).view(-1, 1)
        j = torch.arange(sk).view(1, -1)
        ref &= j <= i + (sk - sq)

    assert torch.equal(build_sparse_mask(q, k, layout, causal=causal), ref)


def test_a_ragged_sequence_length_still_matches():
    """Sizes that are not a multiple of the block, where expansion overruns."""
    q, k, _ = _qkv(s=200)
    layout = build_layout(200, 200, pattern="block_sparse", density=0.25, causal=True)
    m = build_sparse_mask(q, k, layout, causal=True)
    assert m.shape == (200, 200)
    i = torch.arange(200).view(-1, 1)
    j = torch.arange(200).view(1, -1)
    assert not bool((m & (j > i)).any()), "causal masking must survive the crop"


def test_the_sweep_builds_the_mask_outside_the_timed_callable():
    """The bug was positional, so the test has to be too.

    A correct mask built in the wrong place is exactly what shipped.
    """
    import inspect

    from losscolumn.thrusts.kernel import sparse_sweep

    src = inspect.getsource(sparse_sweep.run_sparse_sweep)
    build_at = src.find("build_sparse_mask(")
    bind_at = src.find("run_reference = partial(sparse_reference")
    assert build_at != -1, "the sweep must build the mask itself"
    assert bind_at != -1
    assert build_at < bind_at, (
        "the mask must be built before it is bound into the timed callable")
    assert "mask=cell_mask" in src, (
        "the timed callable must receive the prebuilt mask, not rebuild one")


def test_the_docstring_only_claims_what_is_true():
    """It asserts the gap measures block-skipping. That holds only once the
    mask build is outside the timed region, so the claim and the fix have to
    stay together."""
    assert "only true when `mask` is supplied" in sparse_reference.__doc__
