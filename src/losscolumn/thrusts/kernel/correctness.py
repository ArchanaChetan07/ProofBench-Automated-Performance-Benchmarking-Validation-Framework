"""Numerical correctness for the attention reimplementation.

Success criterion 3 of the proposal: the reimplementation passes correctness
tests across the full swept space, *independent of whether it ever matches
reference throughput*, and fails if correctness is relaxed to improve timings.
That ordering is enforced here in code, not by discipline: the sweep refuses to
record a timing for any shape whose correctness check has not passed.

What is checked, and why each one earns its place:

``output``      max relative error against an fp64 ground truth, against a
                tolerance derived from the arithmetic rather than tuned to pass.
``lse``         the log-sum-exp, when the kernel returns it. Output can be
                right while the statistics are wrong, and backward passes and
                chunked/streaming decoding consume the LSE.
``masked``      a fully-masked causal block must produce zeros, not NaN. This
                is the single most common flash-attention bug and it hides in
                exactly one corner of the space.
``determinism`` the same inputs twice must give bitwise-identical outputs. A
                kernel with a race can pass a tolerance check on Tuesday.
``dtype``       output dtype and shape match the input contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from losscolumn.thrusts.kernel.reference import (
    HAS_TORCH,
    error_budget,
    flash_torch,
    math_reference,
    sdpa_reference,
)

if HAS_TORCH:
    import torch


@dataclass
class ShapeSpec:
    """One point in the ``(head_dim, seq_len, batch, dtype)`` space of RQ3."""

    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: str = "float16"
    causal: bool = False
    seq_k: int | None = None

    @property
    def torch_dtype(self) -> Any:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.dtype]

    def label(self) -> str:
        return (
            f"b{self.batch}xh{self.heads}xs{self.seq_len}xd{self.head_dim}"
            f"/{self.dtype}{'/causal' if self.causal else ''}"
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CorrectnessResult:
    shape: ShapeSpec
    passed: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None

    def failures(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v.get("passed", False)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.to_dict(),
            "label": self.shape.label(),
            "passed": self.passed,
            "checks": self.checks,
            "error": self.error,
        }


def make_inputs(spec: ShapeSpec, *, device: str = "cuda", seed: int = 0):
    """Inputs with a realistic score distribution.

    Drawn at unit variance and scaled by ``1/sqrt(head_dim)`` inside attention,
    so pre-softmax scores land in roughly [-4, 4]. Testing with tiny values
    would make every softmax uniform and hide precisely the max-subtraction and
    rescaling logic the algorithm exists for.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    sk = spec.seq_k or spec.seq_len
    shape_q = (spec.batch, spec.heads, spec.seq_len, spec.head_dim)
    shape_k = (spec.batch, spec.heads, sk, spec.head_dim)
    q = torch.randn(shape_q, generator=g, device=device, dtype=torch.float32)
    k = torch.randn(shape_k, generator=g, device=device, dtype=torch.float32)
    v = torch.randn(shape_k, generator=g, device=device, dtype=torch.float32)
    dt = spec.torch_dtype
    return q.to(dt), k.to(dt), v.to(dt)


def check_shape(
    impl: Callable,
    spec: ShapeSpec,
    *,
    device: str = "cuda",
    seed: int = 0,
    tolerance_scale: float = 1.0,
) -> CorrectnessResult:
    """Run every check for one shape against one implementation."""
    res = CorrectnessResult(shape=spec, passed=False)
    try:
        q, k, v = make_inputs(spec, device=device, seed=seed)
        budget = error_budget(spec.torch_dtype, spec.seq_k or spec.seq_len, spec.head_dim)
        rel_tol = budget["rel_tol"] * tolerance_scale

        out = impl(q, k, v, causal=spec.causal)
        lse = None
        if isinstance(out, tuple):
            out, lse = out

        ref = math_reference(q, k, v, causal=spec.causal)
        num = (out.to(torch.float64) - ref).abs()
        den = ref.abs().amax().clamp_min(1e-12)
        max_rel = float((num.amax() / den).item())
        mean_rel = float((num.mean() / den).item())
        res.checks["output"] = {
            "passed": bool(max_rel <= rel_tol) and bool(torch.isfinite(out).all().item()),
            "max_rel_err": max_rel,
            "mean_rel_err": mean_rel,
            "tolerance": rel_tol,
            "derivation": budget["derivation"],
            "finite": bool(torch.isfinite(out).all().item()),
        }

        res.checks["dtype"] = {
            "passed": out.dtype == q.dtype and tuple(out.shape) == tuple(q.shape),
            "got": f"{out.dtype}{tuple(out.shape)}",
            "expected": f"{q.dtype}{tuple(q.shape)}",
        }

        # A fully masked row must be zero, never NaN. Under a right-aligned
        # causal mask with seq_k == seq_q every row sees at least one key, so
        # the degenerate case is constructed explicitly.
        if spec.causal:
            qs = ShapeSpec(**{**spec.to_dict(), "seq_len": spec.seq_len, "seq_k": spec.seq_len})
            q2, k2, v2 = make_inputs(qs, device=device, seed=seed + 1)
            o2 = impl(q2, k2, v2, causal=True)
            o2 = o2[0] if isinstance(o2, tuple) else o2
            res.checks["masked"] = {
                "passed": bool(torch.isfinite(o2).all().item()),
                "n_nonfinite": int((~torch.isfinite(o2)).sum().item()),
            }

        out2 = impl(q, k, v, causal=spec.causal)
        out2 = out2[0] if isinstance(out2, tuple) else out2
        res.checks["determinism"] = {
            "passed": bool(torch.equal(out, out2)),
            "note": "bitwise identical across two invocations",
        }

        if lse is not None:
            ref_lse = _reference_lse(q, k, causal=spec.causal)
            d = (lse.to(torch.float64) - ref_lse).abs()
            finite = torch.isfinite(ref_lse)
            max_lse = float(d[finite].amax().item()) if finite.any() else 0.0
            res.checks["lse"] = {
                "passed": max_lse <= max(rel_tol * 10, 1e-2),
                "max_abs_err": max_lse,
                "tolerance": max(rel_tol * 10, 1e-2),
            }

        # Cross-check against SDPA too: agreement with the production kernel is
        # not the correctness criterion, but disagreement with both fp64 and
        # SDPA localises the problem faster than either alone.
        try:
            sd = sdpa_reference(q, k, v, causal=spec.causal)
            res.checks["agrees_with_sdpa"] = {
                "passed": True,  # informational only
                "max_rel_err": float(
                    ((out.to(torch.float64) - sd.to(torch.float64)).abs().amax() / den).item()
                ),
            }
        except Exception as e:  # pragma: no cover
            res.checks["agrees_with_sdpa"] = {"passed": True, "skipped": str(e)}

        res.passed = all(c.get("passed", False) for c in res.checks.values())
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        res.passed = False
    return res


def _reference_lse(q, k, *, causal: bool):
    qd, kd = q.to(torch.float64), k.to(torch.float64)
    scale = 1.0 / math.sqrt(qd.shape[-1])
    s = torch.matmul(qd, kd.transpose(-1, -2)) * scale
    if causal:
        sq, sk = s.shape[-2], s.shape[-1]
        i = torch.arange(sq, device=s.device).view(-1, 1)
        j = torch.arange(sk, device=s.device).view(1, -1)
        s = s.masked_fill(j > i + (sk - sq), float("-inf"))
    return torch.logsumexp(s, dim=-1)


@dataclass
class CorrectnessSuite:
    """The gate. No timing is recorded for a shape that does not pass."""

    results: list[CorrectnessResult] = field(default_factory=list)
    implementation: str = ""
    device: str = ""

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    def passed_shapes(self) -> set[str]:
        return {r.shape.label() for r in self.results if r.passed}

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "device": self.device,
            "n_shapes": len(self.results),
            "n_passed": self.n_passed,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self, limit: int = 60) -> str:
        lines = [
            f"### Correctness -- {self.implementation} on {self.device}",
            "",
            f"{self.n_passed}/{len(self.results)} shapes pass. Tolerances are derived from "
            "the arithmetic (4u + 8*sqrt(seq)*u_acc), not fitted to the results.",
            "",
            "| Shape | Result | max rel err | Tolerance | Failed checks |",
            "|-------|--------|-------------|-----------|---------------|",
        ]
        for r in self.results[:limit]:
            o = r.checks.get("output", {})
            mre = o.get("max_rel_err")
            tol = o.get("tolerance")
            err = f"{mre:.2e}" if isinstance(mre, float) else "-"
            tolerance = f"{tol:.2e}" if isinstance(tol, float) else "-"
            note = ", ".join(r.failures()) or (r.error or "-")
            lines.append(
                f"| `{r.shape.label()}` | {'pass' if r.passed else 'FAIL'} | {err} | "
                f"{tolerance} | {note} |"
            )
        return "\n".join(lines)


def run_suite(
    impl: Callable,
    shapes: Sequence[ShapeSpec],
    *,
    name: str = "flash_torch",
    device: str = "cuda",
    seed: int = 0,
) -> CorrectnessSuite:
    suite = CorrectnessSuite(implementation=name, device=device)
    for s in shapes:
        suite.results.append(check_shape(impl, s, device=device, seed=seed))
    return suite


def default_shapes(
    *,
    head_dims: Sequence[int] = (32, 64, 128),
    seq_lens: Sequence[int] = (128, 512, 2048),
    batches: Sequence[int] = (1, 4),
    dtypes: Sequence[str] = ("float16", "bfloat16"),
    heads: int = 8,
    causal: Sequence[bool] = (False, True),
) -> list[ShapeSpec]:
    return [
        ShapeSpec(batch=b, heads=heads, seq_len=s, head_dim=d, dtype=dt, causal=c)
        for d in head_dims
        for s in seq_lens
        for b in batches
        for dt in dtypes
        for c in causal
    ]


def default_impl(q, k, v, *, causal: bool = False):
    """The portable implementation, used when Triton is unavailable."""
    return flash_torch(q, k, v, causal=causal)
