"""The communication calibration study: diagnose, model, calibrate, validate.

The counterpart of the memory study, for the half of the Thrust I model that
was left uncalibrated. It follows the same order and the same rules:

    seal
      -> sweep sizes and world sizes, with replicates
      -> split into calibration and validation points
      -> choose the model family on held-out CALIBRATION points
      -> fit on calibration only
      -> FREEZE
      -> predict the validation points, then measure them
      -> grade, and promote only what passes the gate

The specific question this exists to answer: gloo all-gather failed the linear
model at R^2 = 0.715 while all-reduce passed at 0.984, on the same transport,
between the same processes. A model family that fits one collective and not
another is telling you something about the collectives, and finding out what is
the point of the diagnostic sweep.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.gridsplit import GridSplit
from losscolumn.core.parameters import Parameter, ParameterSet, Provenance
from losscolumn.core.provenance import Provenance as ProvCapture
from losscolumn.core.provenance import content_hash, utcnow
from losscolumn.thrusts.overlap.commodel import (
    CostModel,
    ModelSelection,
    Sample,
    select_model,
)

STUDY_VERSION = "thrust1-comm-v1"

# Tiny -> small -> medium -> large. Log-spaced with a dense small end, because
# that is where a latency floor is identifiable and where a gradient bucket for
# a narrow layer actually lands.
SIZE_TIERS: dict[str, tuple[int, ...]] = {
    "tiny": (1 << 10, 2 << 10, 4 << 10, 8 << 10),
    "small": (16 << 10, 32 << 10, 64 << 10, 128 << 10),
    "medium": (256 << 10, 512 << 10, 1 << 20, 2 << 20),
    "large": (4 << 20, 8 << 20, 16 << 20, 32 << 20),
}
ALL_SIZES: tuple[int, ...] = tuple(
    n for tier in ("tiny", "small", "medium", "large") for n in SIZE_TIERS[tier]
)


def tier_of(nbytes: int) -> str:
    for name, sizes in SIZE_TIERS.items():
        if nbytes in sizes:
            return name
    return "unknown"


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def _worker(rank: int, world: int, sizes: list[int], kinds: list[str],
            replicates: int, q: Any) -> None:
    """One rank. Times each (kind, size) `replicates` times, interleaved."""
    import time

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("LC_COMM_PORT", "29691"))
    os.environ.setdefault("USE_LIBUV", "0")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
        return
    try:
        rows = []
        for rep in range(replicates):
            for kind in kinds:
                for n in sizes:
                    elems = max(n // 4, 1)
                    buf = torch.ones(elems, dtype=torch.float32)
                    out = [torch.empty_like(buf) for _ in range(world)]

                    def once(buf=buf, out=out, kind=kind) -> None:
                        if kind == "all_reduce":
                            dist.all_reduce(buf)
                        else:
                            dist.all_gather(out, buf)

                    for _ in range(2):
                        once()
                    dist.barrier()
                    iters = 20 if n <= (1 << 20) else 5
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        once()
                    dist.barrier()
                    el = (time.perf_counter() - t0) / iters
                    if rank == 0:
                        rows.append((elems * 4, el, kind, rep))
        if rank == 0:
            q.put(rows)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        dist.destroy_process_group()


def sweep(
    *,
    world_sizes: Sequence[int] = (2, 3, 4),
    sizes: Sequence[int] = ALL_SIZES,
    kinds: Sequence[str] = ("all_reduce", "all_gather"),
    replicates: int = 3,
    timeout_s: int = 900,
) -> tuple[list[Sample], dict[str, str]]:
    """Measure every (world, kind, size) combination, several times each.

    Replicates are interleaved across sizes rather than repeated back to back,
    so anything that drifts on a slower timescale than one sweep cancels
    instead of loading onto whichever size happened to run last.
    """
    import multiprocessing as mp
    import sys

    out: list[Sample] = []
    errors: dict[str, str] = {}

    main = sys.modules.get("__main__")
    if mp.get_start_method(allow_none=True) != "fork" and not getattr(main, "__file__", None):
        errors["spawn"] = (
            "the spawn start method has no importable __main__ to re-import in the "
            "worker processes; run this from a script, not from `python -c`"
        )
        return out, errors

    ctx = mp.get_context("spawn")
    for world in world_sizes:
        os.environ["LC_COMM_PORT"] = str(29691 + world)
        q: Any = ctx.Queue()
        procs = [
            ctx.Process(target=_worker,
                        args=(r, world, list(sizes), list(kinds), replicates, q))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        try:
            got = q.get(timeout=timeout_s)
        except Exception as e:
            got = {"error": f"no result: {type(e).__name__}: {e}"}
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        if isinstance(got, dict):
            errors[f"world={world}"] = got.get("error", "unknown")
            continue
        for nbytes, secs, kind, rep in got:
            out.append(Sample(nbytes=nbytes, seconds=secs, world=world,
                              kind=kind, replicate=rep))
    return out, errors


# --------------------------------------------------------------------------
# diagnosis
# --------------------------------------------------------------------------


@dataclass
class Diagnosis:
    """Why a transport does or does not fit the linear model."""

    transport: str
    kind: str
    world: int
    n_samples: int = 0
    linear_r2: float = float("nan")
    per_tier_gbs: dict[str, float] = field(default_factory=dict)
    per_tier_rel_spread: dict[str, float] = field(default_factory=dict)
    bandwidth_ratio: float = float("nan")
    replicate_cv: float = float("nan")
    finding: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport, "kind": self.kind, "world": self.world,
            "n_samples": self.n_samples, "linear_r2": self.linear_r2,
            "per_tier_gbs": self.per_tier_gbs,
            "per_tier_rel_spread": self.per_tier_rel_spread,
            "bandwidth_ratio_large_over_tiny": self.bandwidth_ratio,
            "replicate_cv": self.replicate_cv,
            "finding": self.finding,
        }


def diagnose(samples: Sequence[Sample], *, transport: str = "gloo_shm") -> list[Diagnosis]:
    """Explain each (kind, world) group's behaviour across the size tiers.

    A single effective bandwidth per tier is the diagnostic that matters. If it
    is constant, one line describes the transport and the linear model is
    right. If it climbs steeply from tiny to large, the transport is
    latency-dominated at the small end and a single line is being asked to pass
    through two different physical regimes -- which is what a low R^2 on a
    log-spaced sweep usually means.
    """
    out: list[Diagnosis] = []
    groups: dict[tuple[str, int], list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.kind, s.world), []).append(s)

    for (kind, world), group in sorted(groups.items()):
        d = Diagnosis(transport=transport, kind=kind, world=world,
                      n_samples=len(group))
        from losscolumn.thrusts.overlap.commodel import fit_linear

        lin = fit_linear(group)
        d.linear_r2 = lin.r_squared(group) if lin else float("nan")

        for tier in ("tiny", "small", "medium", "large"):
            in_tier = [s for s in group if tier_of(s.nbytes) == tier]
            if not in_tier:
                continue
            bws = [s.effective_bytes / s.seconds / 1e9 for s in in_tier if s.seconds > 0]
            if bws:
                d.per_tier_gbs[tier] = float(np.median(bws))
                d.per_tier_rel_spread[tier] = (
                    float((max(bws) - min(bws)) / np.median(bws)) if np.median(bws) else float("nan")
                )

        if "tiny" in d.per_tier_gbs and "large" in d.per_tier_gbs:
            d.bandwidth_ratio = d.per_tier_gbs["large"] / d.per_tier_gbs["tiny"]

        # Run-to-run stability, so a bad fit is not blamed on the model when it
        # is really noise.
        by_key: dict[tuple[int, str], list[float]] = {}
        for s in group:
            by_key.setdefault((s.nbytes, s.kind), []).append(s.seconds)
        cvs = [float(np.std(v) / np.mean(v)) for v in by_key.values()
               if len(v) > 1 and np.mean(v) > 0]
        d.replicate_cv = float(np.median(cvs)) if cvs else float("nan")

        bits = []
        if math.isfinite(d.bandwidth_ratio):
            if d.bandwidth_ratio > 5:
                bits.append(
                    f"effective bandwidth rises {d.bandwidth_ratio:.0f}x from the tiny "
                    "tier to the large tier, so the transport is latency-dominated at "
                    "the small end and bandwidth-dominated at the large end. One "
                    "straight line has to pass through both, and cannot"
                )
            else:
                bits.append(
                    f"effective bandwidth is roughly constant across tiers "
                    f"({d.bandwidth_ratio:.1f}x tiny to large), so a single line is the "
                    "right shape"
                )
        if math.isfinite(d.replicate_cv):
            bits.append(
                f"run-to-run variation is {d.replicate_cv:.1%}, "
                + ("small enough that the misfit is structural rather than noise"
                   if d.replicate_cv < 0.15 else
                   "large enough to account for part of the misfit on its own")
            )
        d.finding = "; ".join(bits)
        out.append(d)
    return out


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------


@dataclass
class CommStudy:
    """Calibration and validation of the communication model."""

    samples: list[Sample] = field(default_factory=list)
    diagnoses: list[Diagnosis] = field(default_factory=list)
    selections: dict[str, ModelSelection] = field(default_factory=dict)
    params: ParameterSet = field(default_factory=lambda: ParameterSet("thrust1-comm"))
    split: GridSplit | None = None
    validation: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    device: str = ""
    sealed_at: str = field(default_factory=utcnow)
    max_heldout_err: float = 0.15
    notes: list[str] = field(default_factory=list)

    def protocol(self) -> dict[str, Any]:
        doc = {
            "kind": "communication-model-protocol",
            "study": STUDY_VERSION,
            "sealed_at": self.sealed_at,
            "transports": ["gloo_shm"],
            "collectives": ["all_reduce", "all_gather"],
            "world_sizes": sorted({s.world for s in self.samples}) or [2, 3, 4],
            "size_tiers": {k: list(v) for k, v in SIZE_TIERS.items()},
            "model_families_considered": ["linear", "piecewise", "regime"],
            "family_selection_rule": (
                "held-out error on calibration points the fit never saw. A more "
                "flexible family always fits its own data better, so fit quality "
                "cannot decide. AIC is reported beside it; the simpler family wins "
                "ties within 10%."
            ),
            "quality_gate": {
                "max_heldout_median_relative_error": self.max_heldout_err,
                "on_failure": "the parameters remain DIAGNOSTIC and are not promoted "
                              "into Fabric; the fit stays visible in the report",
            },
            "split_rule": (
                "message sizes are partitioned by index into calibration and "
                "validation before any fitting; the validation sizes are not read "
                "until the model is frozen"
            ),
            "known_limitations": [
                "gloo over shared memory between processes on one host. This is not "
                "NVLink and not InfiniBand, and no parameter fitted here is a value "
                "for the fabric the Thrust I protocol registers.",
                "One host, so no inter-node hop and no NIC is exercised.",
                "World sizes 2 to 4, against the 8 and 32 the protocol registers.",
            ],
        }
        doc["seal_hash"] = content_hash(doc)
        return doc

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": STUDY_VERSION,
            "device": self.device,
            "protocol": self.protocol(),
            "n_samples": len(self.samples),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "model_selection": {k: v.to_dict() for k, v in self.selections.items()},
            "parameters": self.params.to_dict(),
            "split": self.split.to_dict() if self.split else None,
            "validation": self.validation,
            "errors": self.errors,
            "notes": self.notes,
            "samples": [
                {"nbytes": s.nbytes, "seconds": s.seconds, "world": s.world,
                 "kind": s.kind, "replicate": s.replicate, "tier": tier_of(s.nbytes)}
                for s in self.samples
            ],
            "provenance": ProvCapture.capture().to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            "### Thrust I communication model",
            "",
            f"Measured on {self.device}. Protocol seal "
            f"`{self.protocol()['seal_hash'][7:23]}...`.",
            "",
            "#### Diagnosis: why one collective fits and the other does not",
            "",
            "| Collective | world | linear R² | tiny GB/s | large GB/s | ratio | run-to-run |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in self.diagnoses:
            lines.append(
                f"| `{d.kind}` | {d.world} | {d.linear_r2:.3f} | "
                f"{d.per_tier_gbs.get('tiny', float('nan')):.3f} | "
                f"{d.per_tier_gbs.get('large', float('nan')):.3f} | "
                f"{d.bandwidth_ratio:.1f}x | {d.replicate_cv:.1%} |"
            )
        lines.append("")
        for d in self.diagnoses:
            if d.finding:
                lines.append(f"- **{d.kind}, world {d.world}**: {d.finding}.")
        lines.append("")

        if self.selections:
            lines += ["#### Model family, chosen on held-out points", ""]
            for sel in self.selections.values():
                lines += [sel.to_markdown(), ""]

        if self.validation:
            lines += ["#### Validation on untouched message sizes", "",
                      "| Collective | median err | max err | R² | latency err | "
                      "bandwidth err | covered |", "|---|---|---|---|---|---|---|"]
            for k, v in sorted(self.validation.items()):
                lines.append(
                    f"| `{k}` | {v['median_rel_err']:.1%} | {v['max_rel_err']:.1%} | "
                    f"{v['r2']:.3f} | {v['latency_rel_err']:.1%} | "
                    f"{v['bandwidth_rel_err']:.1%} | {v['coverage']:.0%} |"
                )
            lines += ["", "Errors by regime:", ""]
            for k, v in sorted(self.validation.items()):
                per = ", ".join(f"{t} {e:.1%}" for t, e in v["per_tier_err"].items())
                lines.append(f"- `{k}`: {per}")
            lines.append("")

        lines += ["#### Parameters", "", self.params.to_markdown(), ""]
        if self.errors:
            lines += ["#### Not measurable here", ""]
            lines += [f"- `{k}`: {v}" for k, v in self.errors.items()] + [""]
        lines += ["#### Limitations", ""]
        lines += [f"- {x}" for x in self.protocol()["known_limitations"]]
        if self.notes:
            lines += ["", "#### Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _key(kind: str, world: int) -> str:
    return f"gloo_shm/{kind}/world{world}"


def run_study(
    *,
    world_sizes: Sequence[int] = (2, 3, 4),
    replicates: int = 3,
    max_heldout_err: float = 0.15,
    progress: bool = True,
) -> CommStudy:
    """Sweep, diagnose, split, select, fit, freeze, validate."""
    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description

    study = CommStudy(max_heldout_err=max_heldout_err)
    study.device = device_description("cuda")

    if progress:
        print(f"sweeping {len(ALL_SIZES)} sizes x {len(world_sizes)} world sizes x "
              f"2 collectives x {replicates} replicates", flush=True)
    with MeasurementLock("thrust1 communication sweep"):
        study.samples, study.errors = sweep(
            world_sizes=world_sizes, replicates=replicates
        )
    if progress:
        print(f"  {len(study.samples)} samples, {len(study.errors)} failure(s)",
              flush=True)
    if not study.samples:
        study.notes.append("no samples were collected; nothing can be fitted")
        return study

    return run_analysis(study, max_heldout_err=max_heldout_err)


def run_analysis(study: CommStudy, *, max_heldout_err: float | None = None) -> CommStudy:
    """Diagnose, split, select, fit, freeze and validate an existing sample set.

    Separated from measurement so the whole analysis can be re-run from a
    published artifact -- which is also a check on the artifact: anything the
    analysis needs and the artifact lacks fails here.
    """
    if max_heldout_err is not None:
        study.max_heldout_err = max_heldout_err
    study.diagnoses = diagnose(study.samples)

    # Split by size index, before any fitting. Every other size is validation,
    # so both sets span the whole range and neither is confined to one regime.
    sizes = sorted({s.nbytes for s in study.samples})
    cal_sizes = tuple(sizes[::2])
    val_sizes = tuple(s for s in sizes if s not in set(cal_sizes))
    study.split = GridSplit(
        calibration=tuple((n,) for n in cal_sizes),
        validation=tuple((n,) for n in val_sizes),
        rationale={(n,): "alternating sizes, so both sets span every regime"
                   for n in sizes},
    )

    groups: dict[tuple[str, int], list[Sample]] = {}
    for s in study.samples:
        groups.setdefault((s.kind, s.world), []).append(s)

    for (kind, world), group in sorted(groups.items()):
        key = _key(kind, world)
        cal = [s for s in group if s.nbytes in set(cal_sizes)]
        # Family selection uses a held-out slice of the CALIBRATION points only.
        inner = sorted({s.nbytes for s in cal})
        inner_fit = {n for n in inner[::2]}
        fit_pts = [s for s in cal if s.nbytes in inner_fit]
        held_pts = [s for s in cal if s.nbytes not in inner_fit]
        nf = next((d.replicate_cv for d in study.diagnoses
                   if d.kind == kind and d.world == world), float("nan"))
        sel = select_model(fit_pts, held_pts, transport="gloo_shm", kind=kind,
                           world=world, noise_floor=nf,
                           max_heldout_median_err=study.max_heldout_err)
        study.selections[key] = sel

        if sel.chosen is None:
            study.params.add(Parameter(
                f"{key}/alpha_us", float("nan"), Provenance.DIAGNOSTIC,
                unit="us", source=f"{STUDY_VERSION} family selection",
                note=sel.reason,
            ))
            continue

        # Refit the chosen family on ALL calibration points.
        from losscolumn.thrusts.overlap.commodel import (
            fit_linear,
            fit_piecewise,
            fit_regime,
        )

        builder = {"linear": fit_linear, "piecewise": fit_piecewise,
                   "regime": fit_regime}[sel.chosen_family]
        final = builder(cal)
        if final is None:
            continue
        study.split.record_fit(key)
        _register(study, key, final, sel, cal)

    study.split.freeze()

    # ---- validation, only now ------------------------------------------
    for (kind, world), group in sorted(groups.items()):
        key = _key(kind, world)
        val = [s for s in group if s.nbytes in set(val_sizes)]
        for s in val:
            study.split.check_readable((s.nbytes,), purpose="validation")
        model = _model_for(study, key)
        if model is None or not val:
            continue
        study.validation[key] = _grade(model, val)
    return study


def _register(study: CommStudy, key: str, model: CostModel,
              sel: ModelSelection, cal: Sequence[Sample]) -> None:
    """Record the fitted parameters, gated on held-out error."""
    heldout = next((c["heldout_median_rel_err"] for c in sel.candidates
                    if c["family"] == sel.chosen_family), float("nan"))
    passed = math.isfinite(heldout) and heldout <= study.max_heldout_err
    prov = Provenance.ACCEPTED if passed else Provenance.REJECTED
    note = ("" if passed else
            f"held-out median error {heldout:.1%} exceeds the "
            f"{study.max_heldout_err:.0%} gate, so this fit is diagnostic only and is "
            "not promoted into Fabric")
    quality = {"family": sel.chosen_family,
               "heldout_median_rel_err": heldout,
               "fit_r2": model.r_squared(cal)}
    d = model.to_dict()
    study.params.add(Parameter(
        f"{key}/family", float(model.n_params), prov, unit="params",
        source=f"{STUDY_VERSION} calibration", note=note or sel.chosen_family,
        quality=quality,
    ))
    if d.get("alpha_us") is not None:
        study.params.add(Parameter(f"{key}/alpha_us", d["alpha_us"], prov, unit="us",
                                   source=f"{STUDY_VERSION} calibration", note=note,
                                   quality=quality))
        study.params.add(Parameter(f"{key}/beta_gbs", d["beta_gbs"], prov, unit="GB/s",
                                   source=f"{STUDY_VERSION} calibration", note=note,
                                   quality=quality))
    if d.get("breakpoint_bytes"):
        study.params.add(Parameter(
            f"{key}/breakpoint_kib", d["breakpoint_kib"], prov, unit="KiB",
            source=f"{STUDY_VERSION} calibration",
            note=note or "the size at which the collective changes algorithm",
            quality=quality,
        ))
        for side in ("low", "high"):
            study.params.add(Parameter(
                f"{key}/{side}_alpha_us", d[side]["alpha_us"], prov, unit="us",
                source=f"{STUDY_VERSION} calibration", note=note, quality=quality))
            study.params.add(Parameter(
                f"{key}/{side}_beta_gbs", d[side]["beta_gbs"], prov, unit="GB/s",
                source=f"{STUDY_VERSION} calibration", note=note, quality=quality))
    study._models = getattr(study, "_models", {})
    study._models[key] = model


def _model_for(study: CommStudy, key: str) -> CostModel | None:
    return getattr(study, "_models", {}).get(key)


def _grade(model: CostModel, val: Sequence[Sample]) -> dict[str, Any]:
    """Prediction against measurement on untouched points, by regime."""
    y = np.array([s.seconds for s in val], dtype=float)
    p = model.predict_many([s.effective_bytes for s in val])
    rel = np.abs(y - p) / np.where(y > 0, y, np.nan)

    # Latency error: the smallest sizes, where alpha dominates.
    small = [s for s in val if tier_of(s.nbytes) in ("tiny", "small")]
    large = [s for s in val if tier_of(s.nbytes) in ("medium", "large")]

    def _err(pts: Sequence[Sample]) -> float:
        if not pts:
            return float("nan")
        yy = np.array([s.seconds for s in pts])
        pp = model.predict_many([s.effective_bytes for s in pts])
        return float(np.median(np.abs(yy - pp) / yy))

    per_tier = {}
    for tier in ("tiny", "small", "medium", "large"):
        pts = [s for s in val if tier_of(s.nbytes) == tier]
        if pts:
            per_tier[tier] = _err(pts)

    # Uncertainty coverage: the share of points whose measurement falls within
    # the model's own median error band. Reported so a reader can see whether
    # the error is evenly spread or concentrated.
    band = float(np.nanmedian(rel)) if np.isfinite(rel).any() else float("nan")
    covered = (
        float(np.mean(rel <= max(band, 1e-9) * 2)) if np.isfinite(rel).any()
        else float("nan")
    )

    return {
        "n_points": len(val),
        "median_rel_err": float(np.nanmedian(rel)),
        "max_rel_err": float(np.nanmax(rel)),
        "r2": model.r_squared(val),
        "latency_rel_err": _err(small),
        "bandwidth_rel_err": _err(large),
        "per_tier_err": per_tier,
        "coverage": covered,
    }
