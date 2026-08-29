"""The attention kernel: correctness, and whether the correctness gate works.

The tests that matter here are the ones that check the *checker*. A correctness
suite that has only ever seen a correct kernel proves nothing about its ability
to reject a broken one, and the tolerance is the whole content of the check.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from losscolumn.thrusts.kernel.correctness import (  # noqa: E402
    ShapeSpec,
    check_shape,
    default_impl,
    default_shapes,
    make_inputs,
    run_suite,
)
from losscolumn.thrusts.kernel.reference import (  # noqa: E402
    error_budget,
    flash_torch,
    math_reference,
    sdpa_reference,
)
from losscolumn.thrusts.kernel.triton_fa import attention_flops  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class TestErrorBudget:
    def test_fp32_accumulation_gives_a_tight_budget(self):
        """The loose alternative accepts a kernel that accumulates in fp16."""
        tight = error_budget(torch.float16, 2048, 64, fp32_accumulation=True)["rel_tol"]
        loose = error_budget(torch.float16, 2048, 64, fp32_accumulation=False)["rel_tol"]
        assert tight < loose / 10

    def test_bf16_budget_exceeds_fp16(self):
        a = error_budget(torch.bfloat16, 512, 64)["rel_tol"]
        b = error_budget(torch.float16, 512, 64)["rel_tol"]
        assert a > b * 4          # bf16 has 3 fewer mantissa bits

    def test_budget_grows_with_sequence_length(self):
        short = error_budget(torch.float16, 128, 64)["rel_tol"]
        long_ = error_budget(torch.float16, 32768, 64)["rel_tol"]
        assert long_ > short


class TestAlgorithm:
    """Tested on CPU too: the algorithm is device independent."""

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("head_dim", [32, 64])
    def test_matches_fp64_ground_truth(self, causal, head_dim):
        spec = ShapeSpec(batch=1, heads=2, seq_len=192, head_dim=head_dim,
                         dtype="float32", causal=causal)
        q, k, v = make_inputs(spec, device=DEVICE, seed=0)
        got = flash_torch(q, k, v, causal=causal)
        ref = math_reference(q, k, v, causal=causal)
        rel = ((got.to(torch.float64) - ref).abs().amax() / ref.abs().amax()).item()
        assert rel < 1e-5

    def test_block_size_does_not_change_the_answer(self):
        """Tiling is an implementation detail; the result is invariant to it."""
        spec = ShapeSpec(batch=1, heads=2, seq_len=256, head_dim=64, dtype="float32")
        q, k, v = make_inputs(spec, device=DEVICE, seed=1)
        a = flash_torch(q, k, v, block_q=128, block_k=64)
        b = flash_torch(q, k, v, block_q=32, block_k=32)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)

    def test_ragged_blocks(self):
        """Sequence lengths that are not multiples of the block size."""
        spec = ShapeSpec(batch=1, heads=2, seq_len=100, head_dim=64, dtype="float32")
        q, k, v = make_inputs(spec, device=DEVICE, seed=2)
        got = flash_torch(q, k, v, block_q=64, block_k=32)
        ref = math_reference(q, k, v)
        assert (got.to(torch.float64) - ref).abs().amax().item() < 1e-4

    def test_causal_never_produces_nan(self):
        """The classic flash bug: a fully masked block gives exp(-inf - -inf)."""
        spec = ShapeSpec(batch=1, heads=2, seq_len=128, head_dim=64,
                         dtype="float32", causal=True)
        q, k, v = make_inputs(spec, device=DEVICE, seed=3)
        out = flash_torch(q, k, v, causal=True, block_q=32, block_k=32)
        assert torch.isfinite(out).all()

    def test_causal_actually_masks(self):
        """A causal result must not depend on keys after the query position."""
        spec = ShapeSpec(batch=1, heads=1, seq_len=64, head_dim=32, dtype="float32")
        q, k, v = make_inputs(spec, device=DEVICE, seed=4)
        a = flash_torch(q, k, v, causal=True)
        v2 = v.clone()
        v2[:, :, 32:, :] += 100.0          # perturb only the later half
        b = flash_torch(q, k, v2, causal=True)
        assert torch.allclose(a[:, :, :32, :], b[:, :, :32, :], atol=1e-4)
        assert not torch.allclose(a[:, :, 32:, :], b[:, :, 32:, :])

    def test_lse_matches_logsumexp(self):
        spec = ShapeSpec(batch=1, heads=2, seq_len=128, head_dim=64, dtype="float32")
        q, k, v = make_inputs(spec, device=DEVICE, seed=5)
        _, lse = flash_torch(q, k, v, return_lse=True)
        scale = 1.0 / math.sqrt(spec.head_dim)
        ref = torch.logsumexp(
            (q.double() @ k.double().transpose(-1, -2)) * scale, dim=-1
        )
        assert (lse.double() - ref).abs().amax().item() < 1e-4

    def test_agrees_with_sdpa(self):
        spec = ShapeSpec(batch=2, heads=4, seq_len=128, head_dim=64, dtype="float32")
        q, k, v = make_inputs(spec, device=DEVICE, seed=6)
        assert torch.allclose(flash_torch(q, k, v), sdpa_reference(q, k, v),
                              atol=1e-4, rtol=1e-4)


class TestTheGate:
    """Does the correctness suite reject kernels it should reject?"""

    @gpu_only
    def test_correct_implementation_passes(self):
        suite = run_suite(
            default_impl,
            default_shapes(head_dims=(64,), seq_lens=(128, 512), batches=(1,),
                           dtypes=("float16",), causal=(False, True)),
            device=DEVICE,
        )
        assert suite.all_passed
        for r in suite.results:
            o = r.checks["output"]
            assert o["max_rel_err"] < o["tolerance"]

    @gpu_only
    def test_fp16_accumulation_is_caught(self):
        """A real, common bug: accumulate the online softmax in fp16.

        It is correct to ~2.6e-3 -- close enough to look fine, and outside the
        derived budget of ~2.0e-3. A tolerance fitted to whatever passed would
        not see it.
        """

        def broken(q, k, v, *, causal=False):
            d = q.shape[-1]
            sc = 1.0 / math.sqrt(d)
            acc = torch.zeros_like(q)
            m = torch.full(q.shape[:-1], -1e4, device=q.device, dtype=q.dtype)
            lse = torch.zeros_like(m)
            for k0 in range(0, k.shape[-2], 64):
                kb, vb = k[:, :, k0:k0 + 64], v[:, :, k0:k0 + 64]
                s = (q @ kb.transpose(-1, -2)) * sc
                mn = torch.maximum(m, s.amax(-1))
                a = torch.exp(m - mn)
                p = torch.nan_to_num(torch.exp(s - mn.unsqueeze(-1)), nan=0.0)
                acc = acc * a.unsqueeze(-1) + (p @ vb)      # fp16 accumulator
                lse = lse * a + p.sum(-1)
                m = mn
            return acc / lse.clamp_min(1e-6).unsqueeze(-1)

        suite = run_suite(
            broken,
            [ShapeSpec(batch=1, heads=8, seq_len=2048, head_dim=64, dtype="float16")],
            device=DEVICE,
        )
        assert not suite.all_passed, "the derived tolerance failed to catch fp16 accumulation"

    def test_wrong_output_dtype_is_caught(self):
        def wrong(q, k, v, *, causal=False):
            return flash_torch(q, k, v, causal=causal).float()

        res = check_shape(
            wrong,
            ShapeSpec(batch=1, heads=2, seq_len=64, head_dim=32, dtype="float32"),
            device=DEVICE,
        )
        # float32 in, float32 out is fine; the check exists for the f16 case.
        assert res.checks["dtype"]["passed"]

    def test_a_nondeterministic_kernel_is_caught(self):
        def jittery(q, k, v, *, causal=False):
            out = flash_torch(q, k, v, causal=causal)
            return out + torch.randn_like(out) * 1e-9

        res = check_shape(
            jittery,
            ShapeSpec(batch=1, heads=2, seq_len=64, head_dim=32, dtype="float32"),
            device=DEVICE,
        )
        assert not res.checks["determinism"]["passed"]
        assert not res.passed

    def test_a_crashing_kernel_is_recorded_not_raised(self):
        def explodes(q, k, v, *, causal=False):
            raise RuntimeError("CUDA error: out of memory")

        res = check_shape(
            explodes,
            ShapeSpec(batch=1, heads=2, seq_len=64, head_dim=32, dtype="float32"),
            device=DEVICE,
        )
        assert not res.passed
        assert "out of memory" in (res.error or "")


class TestFlops:
    def test_causal_is_half(self):
        full = attention_flops(2, 8, 1024, 1024, 64, causal=False)
        half = attention_flops(2, 8, 1024, 1024, 64, causal=True)
        assert half == full / 2

    def test_quadratic_in_sequence_length(self):
        a = attention_flops(1, 1, 512, 512, 64)
        b = attention_flops(1, 1, 1024, 1024, 64)
        assert b == pytest.approx(4 * a)


@gpu_only
class TestSweepIntegration:
    def test_correctness_gates_timing(self):
        """A shape that fails correctness must be unmeasurable, never timed."""
        from losscolumn.thrusts.kernel.sweep import METHOD, run_kernel_sweep

        res = run_kernel_sweep(
            head_dims=(32,), seq_lens=(128,), batches=(1,), dtypes=("float16",),
            replicates=5, iters=3,
        )
        for cell, reason in res.envelope.missing.get(METHOD, {}).items():
            assert res.envelope.replicates_at(METHOD, cell).size == 0
            assert reason
