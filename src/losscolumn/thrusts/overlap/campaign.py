"""The communication coverage and readiness campaign.

The previous study left one group of six with an accepted model, and nothing
in the code distinguished that from a calibrated subsystem. This campaign asks
the question that actually gates an eight-GPU rental:

    is enough of the required operating surface covered, by accepted models,
    with independent validation, to be worth measuring at scale?

Four things it does that the previous study did not:

**A dense grid that is not only powers of two.** Sixteen power-of-two sizes
cannot distinguish a monotonic two-regime surface from a non-monotonic one --
there are four points per regime and the transitions fall between them. The
backbone here is log-spaced with intermediate sizes, and it is *oversampled*
where the previous campaign found structure: the latency-to-bandwidth
transition, the suspected cache-scale peak, the decline past it, and the sizes
where the earlier breakpoints landed.

**Repeat passes.** The bandwidth peak is the campaign's central physical
claim, and a peak measured once is a peak that might be a scheduling artifact.
Each group is swept in two independent passes so peak location, magnitude and
regime ordering can be compared.

**Explicit invalidity.** A point that could not be measured is recorded as
invalid with a reason. It is never a zero, and never silently absent.

**Coverage separate from acceptance.** The readiness verdict is conjunctive
over every required group and regime, and there is no arithmetic that lets a
well-covered group compensate for a missing one.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.bimodal import find_threshold_band
from losscolumn.core.coverage import (
    CommunicationCoverage,
    ParameterVerdict,
    RegimeStatus,
)
from losscolumn.core.provenance import Provenance as ProvCapture
from losscolumn.core.provenance import content_hash, utcnow
from losscolumn.thrusts.overlap.commodel import Sample, select_model

CAMPAIGN_VERSION = "thrust1-comm-campaign-v1"

REQUIRED_COLLECTIVES = ("all_reduce", "all_gather")
REQUIRED_WORLDS = (2, 3, 4)
REQUIRED_REGIMES = ("tiny", "small", "medium", "large")

# Regime edges in bytes. Chosen from the previous campaign's measured structure
# rather than from round numbers: the latency floor runs out around 16 KiB, the
# bandwidth peak sits near 1-2 MiB, and the decline past it begins around 4 MiB.
REGIME_EDGES: tuple[tuple[str, int, int], ...] = (
    ("tiny", 1 << 10, 1 << 14),
    ("small", 1 << 14, 1 << 18),
    ("medium", 1 << 18, 1 << 22),
    ("large", 1 << 22, (32 << 20) + 1),
)


def regime_of(nbytes: int) -> str:
    for name, lo, hi in REGIME_EDGES:
        if lo <= nbytes < hi:
            return name
    return "large" if nbytes >= REGIME_EDGES[-1][1] else "tiny"


# Regions deliberately oversampled, with the reason each is suspected.
OVERSAMPLED: tuple[tuple[str, int, int, str], ...] = (
    ("latency_to_bandwidth", 1 << 13, 1 << 16,
     "where the latency floor gives way to a bandwidth slope"),
    ("cache_peak", 1 << 19, 1 << 22,
     "where effective bandwidth peaked in the previous campaign, consistent with "
     "the buffer still fitting in cache"),
    ("peak_decline", 1 << 22, 1 << 24,
     "where bandwidth fell again past the peak"),
    ("prior_breakpoints", 1 << 20, 3 << 21,
     "where the earlier two-segment fits placed their breakpoints"),
)


def build_grid(*, backbone_per_decade: int = 6,
               oversample_per_region: int = 5,
               lo: int = 1 << 10, hi: int = 32 << 20) -> tuple[int, ...]:
    """A log-spaced backbone plus oversampling, deduplicated and sorted.

    Not only powers of two. A grid on powers of two alone puts every sample at
    a round number and leaves the transitions between them unobserved, which is
    exactly where the structure this campaign is looking for lives.
    """
    sizes: set[int] = set()
    decades = math.log10(hi / lo)
    n_backbone = max(int(round(decades * backbone_per_decade)), 8)
    for x in np.logspace(math.log10(lo), math.log10(hi), n_backbone):
        sizes.add(_round_to_word(x))
    for _, a, b, _ in OVERSAMPLED:
        for x in np.logspace(math.log10(a), math.log10(min(b, hi)),
                             oversample_per_region):
            sizes.add(_round_to_word(x))
    return tuple(sorted(n for n in sizes if lo <= n <= hi))


def _round_to_word(x: float) -> int:
    """Round to a multiple of 4 bytes, since buffers are fp32 elements."""
    return max(int(round(x / 4)) * 4, 4)


# --------------------------------------------------------------------------
# measurement records
# --------------------------------------------------------------------------


@dataclass
class PointRecord:
    """Everything measured at one (collective, world, size), in one pass."""

    collective: str
    world: int
    nbytes: int
    pass_index: int
    valid: bool = True
    invalid_reason: str = ""
    timings_s: list[float] = field(default_factory=list)
    median_s: float = float("nan")
    iqr_s: float = float("nan")
    cv: float = float("nan")
    effective_bytes: float = 0.0
    bandwidth_gbs: float = float("nan")
    n_runs: int = 0
    n_failures: int = 0
    iters: int = 0
    """Calls averaged inside each timing. Recorded because it sets the noise."""

    @property
    def regime(self) -> str:
        return regime_of(self.nbytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collective": self.collective, "world": self.world,
            "nbytes": self.nbytes, "regime": self.regime,
            "pass_index": self.pass_index,
            "valid": self.valid, "invalid_reason": self.invalid_reason,
            "timings_s": self.timings_s, "iters": self.iters, "median_s": self.median_s,
            "iqr_s": self.iqr_s, "cv": self.cv,
            "effective_bytes": self.effective_bytes,
            "bandwidth_gbs": self.bandwidth_gbs,
            "n_runs": self.n_runs, "n_failures": self.n_failures,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PointRecord:
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def to_sample(self) -> Sample | None:
        if not self.valid or not math.isfinite(self.median_s) or self.median_s <= 0:
            return None
        return Sample(nbytes=self.nbytes, seconds=self.median_s, world=self.world,
                      kind=self.collective, replicate=self.pass_index)


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------


# Budget for one timed block: a fixed duration, so averaging follows the cost of
# a call rather than an arbitrary byte threshold.
#
# This is the third rule tried here and the first two are worth recording. The
# original `20 if n <= 1MiB else 5` put a fourfold averaging cliff inside the
# medium regime, which is the band that decides most of the coverage matrix.
#
# The second scaled the budget by each point's own measured variability, which
# sounds better and measured worse: the estimate came from six back-to-back
# calls, and the variability that matters here acts on a longer timescale than
# six calls can see. So it judged the noisy band quiet, gave 41% of all points
# the minimum five iterations, and produced data on which coverage fell from 17
# cells to 11. The idea was not merely mis-tuned -- a short probe cannot
# estimate a slow instability, and no tuning fixes that.
#
# A fixed duration makes no claim it cannot support. Every point gets the same
# averaging effort in time; where one call already exceeds the budget, it gets
# the floor and the artifact records how many it got.
TARGET_BLOCK_S = 0.060

MIN_ITERS = 5
MAX_ITERS = 200


def _worker(rank: int, world: int, sizes: list[int], kinds: list[str],
            repeats: int, pass_index: int, q: Any) -> None:
    """One rank of one pass. Records raw timings, not just a summary."""
    import time

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("LC_COMM_PORT", "29711"))
    os.environ.setdefault("USE_LIBUV", "0")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
        return
    try:
        rows: list[dict[str, Any]] = []
        for kind in kinds:
            for n in sizes:
                elems = max(n // 4, 1)
                timings: list[float] = []
                failures = 0
                reason = ""
                iters = 0   # bound before the try: the row records it either way
                try:
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
                    # Size the timed block by duration. The probe establishes
                    # only the cost of a call, which six calls can measure; it
                    # is not asked to estimate variability, which they cannot.
                    probe = []
                    for _ in range(6):
                        tp = time.perf_counter()
                        once()
                        probe.append(time.perf_counter() - tp)
                    per_call = max(float(np.median(np.array(probe, dtype=float))),
                                   1e-9)
                    iters = int(min(max(round(TARGET_BLOCK_S / per_call),
                                        MIN_ITERS), MAX_ITERS))
                    # Every rank must issue the SAME number of collectives or
                    # the group deadlocks on the difference. Each rank times
                    # itself, so each would otherwise pick its own count: rank 0
                    # decides and the rest follow.
                    shared = torch.tensor([iters], dtype=torch.int64)
                    dist.broadcast(shared, src=0)
                    iters = int(shared.item())
                    dist.barrier()
                    for _ in range(repeats):
                        try:
                            t0 = time.perf_counter()
                            for _ in range(iters):
                                once()
                            dist.barrier()
                            timings.append((time.perf_counter() - t0) / iters)
                        except Exception as e:
                            failures += 1
                            reason = f"{type(e).__name__}: {str(e)[:80]}"
                except Exception as e:
                    failures += 1
                    reason = f"{type(e).__name__}: {str(e)[:80]}"
                if rank == 0:
                    rows.append({
                        "collective": kind, "world": world, "nbytes": elems * 4,
                        "pass_index": pass_index, "timings_s": timings,
                        "n_failures": failures, "invalid_reason": reason,
                        "iters": iters,
                    })
        if rank == 0:
            q.put(rows)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        dist.destroy_process_group()


MEDIAN_SE_FACTOR = 1.2533
"""sqrt(pi/2): the standard error of a median relative to that of a mean."""

AVERAGING_EXPONENT = -0.309
"""Measured on this machine from 5520 disjoint block pairs in one session.

White noise would give -0.5. Using that here would understate the precision
that repeats actually buy and overstate it for large n; the measured value is
carried so the gate responds to measurement effort the way it really behaves.
"""


def _point_se(noise_cv: float, n_repeats: int) -> float:
    """Standard error of the median the model is fitted to.

    The quantity a gradeability gate must judge. The population CV of a point's
    repeats says how much the machine varies; it does not say how well the point
    is known, and it does not improve when the point is measured harder.
    """
    if not math.isfinite(noise_cv) or n_repeats < 1:
        return float("nan")
    return MEDIAN_SE_FACTOR * noise_cv * n_repeats ** AVERAGING_EXPONENT


def _finalise(raw: dict[str, Any]) -> PointRecord:
    """Turn one worker row into a record, with dispersion and validity."""
    rec = PointRecord(
        collective=raw["collective"], world=raw["world"], nbytes=raw["nbytes"],
        pass_index=raw["pass_index"], timings_s=list(raw.get("timings_s", [])),
        n_failures=int(raw.get("n_failures", 0)),
        iters=int(raw.get("iters", 0)),
    )
    rec.n_runs = len(rec.timings_s)
    if not rec.timings_s:
        rec.valid = False
        rec.invalid_reason = raw.get("invalid_reason") or "no timing completed"
        return rec
    t = np.array(rec.timings_s, dtype=float)
    rec.median_s = float(np.median(t))
    rec.iqr_s = float(np.percentile(t, 75) - np.percentile(t, 25))
    rec.cv = float(np.std(t) / np.mean(t)) if np.mean(t) > 0 else float("nan")
    factor = 2.0 if rec.collective == "all_reduce" else 1.0
    steps = (rec.world - 1) / rec.world if rec.world > 1 else 1.0
    rec.effective_bytes = factor * steps * rec.nbytes
    if rec.median_s > 0:
        rec.bandwidth_gbs = rec.effective_bytes / rec.median_s / 1e9
    if rec.median_s <= 0 or not math.isfinite(rec.median_s):
        rec.valid = False
        rec.invalid_reason = "median timing is not positive"
    return rec


def sweep_pass(
    *, sizes: Sequence[int], worlds: Sequence[int],
    collectives: Sequence[str], repeats: int, pass_index: int,
    timeout_s: int = 3600,
) -> tuple[list[PointRecord], dict[str, str]]:
    import multiprocessing as mp
    import sys

    out: list[PointRecord] = []
    errors: dict[str, str] = {}
    main = sys.modules.get("__main__")
    if mp.get_start_method(allow_none=True) != "fork" and not getattr(main, "__file__", None):
        errors["spawn"] = "no importable __main__ for the spawn start method"
        return out, errors

    ctx = mp.get_context("spawn")
    for world in worlds:
        os.environ["LC_COMM_PORT"] = str(29711 + world + 10 * pass_index)
        q: Any = ctx.Queue()
        procs = [
            ctx.Process(target=_worker,
                        args=(r, world, list(sizes), list(collectives), repeats,
                              pass_index, q))
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
            errors[f"pass{pass_index}/world{world}"] = got.get("error", "unknown")
            continue
        out.extend(_finalise(r) for r in got)
    return out, errors


# --------------------------------------------------------------------------
# peak reproducibility
# --------------------------------------------------------------------------


@dataclass
class PeakReport:
    """Whether the bandwidth peak is a property of the hardware or of one run."""

    collective: str
    world: int
    per_pass: list[dict[str, Any]] = field(default_factory=list)
    reproducible: bool = False
    peak_location_ratio: float = float("nan")
    peak_magnitude_ratio: float = float("nan")
    ordering_stable: bool = False
    well_defined: bool = False
    finding: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "collective": self.collective, "world": self.world,
            "per_pass": self.per_pass, "reproducible": self.reproducible,
            "peak_location_ratio": self.peak_location_ratio,
            "peak_magnitude_ratio": self.peak_magnitude_ratio,
            "ordering_stable": self.ordering_stable,
            "well_defined": self.well_defined,
            "finding": self.finding,
        }


# A maximum whose neighbours are slower than this fraction of it is a spike.
MIN_NEIGHBOUR_SUPPORT = 0.60
# A size far from the maximum that reaches this fraction of it makes the
# location undetermined, however sharp the maximum itself looks.
MAX_RUNNER_UP = 0.90
# "Far" means this many grid steps away.
RUNNER_UP_DISTANCE = 3


def _locate_peak(group: Sequence[PointRecord]
                 ) -> tuple[PointRecord, float, float]:
    """The maximum, how well its neighbours support it, and its rival.

    Returns the argmax together with two numbers that say whether calling it a
    peak is justified: the ratio of its neighbours' bandwidth to its own, and
    the ratio of the best distant competitor to its own.
    """
    ordered = sorted(group, key=lambda r: r.nbytes)
    bw = [r.bandwidth_gbs for r in ordered]
    i = max(range(len(ordered)), key=lambda k: bw[k])
    peak = ordered[i]
    nb = [bw[j] for j in (i - 1, i + 1) if 0 <= j < len(bw)]
    support = float(np.median(nb) / bw[i]) if nb and bw[i] > 0 else 0.0
    far = [bw[j] for j in range(len(bw)) if abs(j - i) > RUNNER_UP_DISTANCE]
    runner_up = float(max(far) / bw[i]) if far and bw[i] > 0 else 0.0
    return peak, support, runner_up


def analyse_peak(records: Sequence[PointRecord]) -> PeakReport:
    """Locate the bandwidth peak in each pass and compare them."""
    valid = [r for r in records if r.valid and math.isfinite(r.bandwidth_gbs)]
    if not valid:
        return PeakReport(collective="", world=0, finding="no valid points")
    rep = PeakReport(collective=valid[0].collective, world=valid[0].world)

    by_pass: dict[int, list[PointRecord]] = {}
    for r in valid:
        by_pass.setdefault(r.pass_index, []).append(r)

    for idx, group in sorted(by_pass.items()):
        peak, support, runner_up = _locate_peak(group)
        order = [
            float(np.median([x.bandwidth_gbs for x in group if x.regime == reg]))
            if any(x.regime == reg for x in group) else float("nan")
            for reg in REQUIRED_REGIMES
        ]
        rep.per_pass.append({
            "pass": idx, "peak_bytes": peak.nbytes,
            "peak_regime": peak.regime,
            "peak_bandwidth_gbs": peak.bandwidth_gbs,
            "neighbour_support": support,
            "runner_up_ratio": runner_up,
            "regime_bandwidths": dict(zip(REQUIRED_REGIMES, order, strict=False)),
            "n_points": len(group),
        })

    if len(rep.per_pass) < 2:
        rep.finding = "only one pass; reproducibility not testable"
        return rep

    a, b = rep.per_pass[0], rep.per_pass[1]
    rep.peak_location_ratio = (
        max(a["peak_bytes"], b["peak_bytes"]) / max(min(a["peak_bytes"], b["peak_bytes"]), 1)
    )
    rep.peak_magnitude_ratio = (
        max(a["peak_bandwidth_gbs"], b["peak_bandwidth_gbs"])
        / max(min(a["peak_bandwidth_gbs"], b["peak_bandwidth_gbs"]), 1e-12)
    )
    ord_a = [a["regime_bandwidths"][r] for r in REQUIRED_REGIMES]
    ord_b = [b["regime_bandwidths"][r] for r in REQUIRED_REGIMES]

    def rank(xs: list[float]) -> list[int]:
        finite = [(i, x) for i, x in enumerate(xs) if math.isfinite(x)]
        return [i for i, _ in sorted(finite, key=lambda t: -t[1])]

    rep.ordering_stable = rank(ord_a) == rank(ord_b)
    # A maximum is only a peak if its neighbours are fast too, and only
    # well located if nothing far away is nearly as fast. An argmax over a
    # noisy curve satisfies neither and is not a peak estimator: it picked a
    # 4x spike in one group and flipped between two near-equal maxima 270x
    # apart in size in another.
    rep.well_defined = all(
        p["neighbour_support"] >= MIN_NEIGHBOUR_SUPPORT
        and p["runner_up_ratio"] <= MAX_RUNNER_UP
        for p in rep.per_pass
    )
    # Within one regime step and within 25% in magnitude counts as the same
    # peak. Tighter than that would call ordinary run-to-run variation a
    # different peak; looser would call two different peaks the same one.
    rep.reproducible = (
        rep.well_defined
        and a["peak_regime"] == b["peak_regime"]
        and rep.peak_magnitude_ratio <= 1.25
        and rep.ordering_stable
    )
    if not rep.well_defined:
        weak = [p for p in rep.per_pass
                if p["neighbour_support"] < MIN_NEIGHBOUR_SUPPORT]
        flat = [p for p in rep.per_pass if p["runner_up_ratio"] > MAX_RUNNER_UP]
        bits = []
        if weak:
            bits.append(
                f"the maximum is {1 / max(weak[0]['neighbour_support'], 1e-9):.1f}x its "
                "own neighbours, which is a spike rather than a peak"
            )
        if flat:
            bits.append(
                "a size far from the maximum is within "
                f"{(1 - MAX_RUNNER_UP) * 100:.0f}% of it, so the location is not "
                "determined by the data"
            )
        rep.finding = (
            "NO WELL-DEFINED PEAK: " + "; ".join(bits)
            + ". Reproducibility is not the question here -- there is no peak to "
              "reproduce, and reporting a location would be reporting an argmax"
        )
        return rep
    if rep.reproducible:
        rep.finding = (
            f"the peak lands in the {a['peak_regime']} regime in both passes, within "
            f"{rep.peak_magnitude_ratio:.2f}x in magnitude and with the same regime "
            "ordering: a property of the transport, not of one run"
        )
    else:
        why = []
        if a["peak_regime"] != b["peak_regime"]:
            why.append(f"the peak moved from {a['peak_regime']} to {b['peak_regime']}")
        if rep.peak_magnitude_ratio > 1.25:
            why.append(f"its magnitude differs by {rep.peak_magnitude_ratio:.2f}x")
        if not rep.ordering_stable:
            why.append("the regime ordering changed between passes")
        rep.finding = (
            "NOT reproducible: " + "; ".join(why)
            + ". The peak must not be reported as a physical property on this evidence"
        )
    return rep


# --------------------------------------------------------------------------
# the campaign
# --------------------------------------------------------------------------


@dataclass
class Campaign:
    grid: tuple[int, ...]
    records: list[PointRecord] = field(default_factory=list)
    peaks: dict[str, PeakReport] = field(default_factory=dict)
    selections: dict[str, Any] = field(default_factory=dict)
    selections_note: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    coverage: CommunicationCoverage | None = None
    errors: dict[str, str] = field(default_factory=dict)
    device: str = ""
    n_passes: int = 2
    repeats: int = 3
    max_worst_regime_err: float = 0.15
    max_noise_cv: float = 0.20
    min_points_per_regime: int = 3
    sealed_at: str = field(default_factory=utcnow)
    notes: list[str] = field(default_factory=list)

    def protocol(self) -> dict[str, Any]:
        doc = {
            "kind": "communication-campaign-protocol",
            "campaign": CAMPAIGN_VERSION,
            "sealed_at": self.sealed_at,
            "collectives": list(REQUIRED_COLLECTIVES),
            "world_sizes": list(REQUIRED_WORLDS),
            "transport": "gloo over shared memory, one host",
            "message_size_range_bytes": [self.grid[0], self.grid[-1]],
            "dense_grid": list(self.grid),
            "n_grid_points": len(self.grid),
            "grid_construction": (
                "log-spaced backbone rounded to 4-byte words, NOT restricted to powers "
                "of two, plus oversampling in four named regions"
            ),
            "oversampled_regions": [
                {"name": n, "lo": a, "hi": b, "why": w} for n, a, b, w in OVERSAMPLED
            ],
            "regime_definitions": [
                {"name": n, "lo": a, "hi": b} for n, a, b in REGIME_EDGES
            ],
            "passes": self.n_passes,
            "repeats_per_point_per_pass": self.repeats,
            "pass_purpose": (
                "two independent passes so the bandwidth peak's location, magnitude "
                "and regime ordering can be compared; a peak seen once is not a "
                "property of the transport"
            ),
            "estimator_families": ["weighted", "log", "huber"],
            "model_families": ["linear", "piecewise", "regime"],
            "segmentation": "greedy top-down, stopping on AIC",
            "cross_validation": (
                "stratified k-fold over message size, folds assigned round-robin "
                "within each regime so no fold can be missing a regime"
            ),
            "acceptance": {
                "criterion": "worst-regime cross-validated relative error",
                "max_worst_regime_err": self.max_worst_regime_err,
                "why_not_pooled": (
                    "a pooled median lets one useless regime hide behind three good "
                    "ones, which is how a model 35% wrong in the medium tier was "
                    "previously accepted"
                ),
                "max_noise_cv": self.max_noise_cv,
                "min_points_per_regime": self.min_points_per_regime,
            },
            "coverage_requirements": {
                "required_groups": len(REQUIRED_COLLECTIVES) * len(REQUIRED_WORLDS),
                "required_regimes_per_group": list(REQUIRED_REGIMES),
                "a_regime_is_covered_when": [
                    "it was measured with at least min_points_per_regime valid points",
                    "the group's model was ACCEPTED",
                    "held-out validation graded the model in that regime",
                    "the regime's cross-validated error is within the threshold",
                    "measurement noise in that regime is within max_noise_cv",
                ],
            },
            "subsystem_readiness": (
                "conjunctive: every required group covered, every required regime "
                "covered, accepted models only, held-out validation present in every "
                "cell. There is no arithmetic that trades a covered group against a "
                "missing one."
            ),
            "transferability": (
                "NOTHING measured here is a value for NVLink or InfiniBand. This is "
                "gloo over shared memory on one host. What the campaign can establish "
                "is methodology readiness -- that the grid, the estimators, the "
                "selection rule and the coverage criteria work -- not fabric "
                "parameters for an A100 node."
            ),
        }
        doc["seal_hash"] = content_hash(doc)
        return doc

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": CAMPAIGN_VERSION,
            "device": self.device,
            "protocol": self.protocol(),
            "n_records": len(self.records),
            "n_valid": sum(1 for r in self.records if r.valid),
            "n_invalid": sum(1 for r in self.records if not r.valid),
            "records": [r.to_dict() for r in self.records],
            "peaks": {k: v.to_dict() for k, v in self.peaks.items()},
            "model_selection": self.selections,
            "selection_notes": self.selections_note,
            "thresholds": self.thresholds,
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "errors": self.errors,
            "notes": self.notes,
            "provenance": ProvCapture.capture().to_dict(),
        }


# --------------------------------------------------------------------------
# analysis: fit on calibration sizes only, then grade coverage
# --------------------------------------------------------------------------


def _split_sizes(sizes: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Alternate sizes within each regime, so both halves span every regime.

    Splitting the whole sorted list alternately would still work, but doing it
    per regime guarantees the property that matters: no regime can end up
    entirely on one side, which would make it impossible to validate there.
    """
    cal: list[int] = []
    val: list[int] = []
    by_regime: dict[str, list[int]] = {}
    for n in sorted(sizes):
        by_regime.setdefault(regime_of(n), []).append(n)
    for group in by_regime.values():
        for i, n in enumerate(group):
            (cal if i % 2 == 0 else val).append(n)
    return tuple(sorted(cal)), tuple(sorted(val))


def analyse(campaign: Campaign) -> Campaign:
    """Fit on calibration sizes, validate on the rest, then grade coverage."""
    from losscolumn.thrusts.overlap.commodel import (
        fit_linear,
        fit_piecewise,
        fit_regime,
    )

    cov = CommunicationCoverage.empty(
        REQUIRED_COLLECTIVES, REQUIRED_WORLDS, REQUIRED_REGIMES
    )
    cal_sizes, val_sizes = _split_sizes(campaign.grid)
    cov.notes.append(
        f"{len(cal_sizes)} calibration sizes and {len(val_sizes)} validation sizes, "
        "alternating within each regime so neither half is missing a regime."
    )

    by_group: dict[tuple[str, int], list[PointRecord]] = {}
    for r in campaign.records:
        by_group.setdefault((r.collective, r.world), []).append(r)

    for (kind, world), recs in sorted(by_group.items()):
        key = f"{kind}/world{world}"
        g = cov.groups.get(key)
        if g is None:
            continue

        campaign.peaks[key] = analyse_peak(recs)

        cal = [s for r in recs if r.nbytes in set(cal_sizes)
               for s in [r.to_sample()] if s is not None]
        val = [s for r in recs if r.nbytes in set(val_sizes)
               for s in [r.to_sample()] if s is not None]

        noise_by_regime: dict[str, float] = {}
        for reg in REQUIRED_REGIMES:
            cvs = [r.cv for r in recs
                   if r.regime == reg and r.valid and math.isfinite(r.cv)]
            noise_by_regime[reg] = float(np.median(cvs)) if cvs else float("nan")
        noise_floor = float(np.nanmedian(list(noise_by_regime.values()))) \
            if noise_by_regime else float("nan")

        if len(cal) < 8:
            g.detail = f"only {len(cal)} valid calibration points; nothing fitted"
            for reg in REQUIRED_REGIMES:
                g.regimes[reg].status = RegimeStatus.INSUFFICIENT_POINTS
                g.regimes[reg].detail = g.detail
            continue

        # Where an algorithm-selection threshold sits inside a regime, that
        # regime has two behaviours rather than one and no single-valued model
        # can be right about it. Detected before fitting so the fit is not asked
        # to pass through it.
        threshold = find_threshold_band(recs, group=key)
        campaign.thresholds[key] = threshold.to_dict()
        bimodal_regimes = threshold.flagged_regimes(regime_of)

        # A regime whose points carry a standard error above the gradeability
        # ceiling cannot distinguish a bad model from an unmeasurable one, so it
        # is excluded from the selection score. It stays uncovered regardless.
        ungradable = tuple(
            reg for reg in REQUIRED_REGIMES
            if math.isfinite(_point_se(noise_by_regime.get(reg, float("nan")),
                                       campaign.repeats))
            and _point_se(noise_by_regime.get(reg, float("nan")),
                          campaign.repeats) > campaign.max_noise_cv
        ) + bimodal_regimes
        # Points at a flagged size are excluded from the FIT, not only from the
        # score. Their medians are not stable targets -- a bistable point's
        # median is a mixture proportion, and a point beside a step is whichever
        # side it landed on -- so fitting through them drags the curve toward a
        # value the transport does not produce, and it drags it everywhere, not
        # just locally. That was costing all_reduce/world2 20% error in tiny and
        # small, regimes that sit below every threshold it has.
        #
        # This cannot manufacture coverage: a flagged regime stays uncovered
        # whatever the fit does. It only stops the unmodellable part of the
        # surface from spoiling the part that is modellable.
        # Only bistable sizes are excluded. A point beside a step has a
        # perfectly good median; it is the segmented family's job to fit it.
        flagged = set(threshold.ungradable_sizes)
        cal_fit = [s for s in cal if s.nbytes not in flagged] or cal
        campaign.selections_note[key] = (
            f"{len(cal) - len(cal_fit)} of {len(cal)} calibration point(s) "
            "excluded from the fit as having two behaviours at one size"
        ) if len(cal_fit) < len(cal) else ""
        sel = select_model(cal_fit, [], transport="gloo_shm", kind=kind, world=world,
                           noise_floor=noise_floor, tier_fn=regime_of,
                           ungradable_tiers=ungradable,
                           max_heldout_median_err=campaign.max_worst_regime_err)
        campaign.selections[key] = sel.to_dict()

        if sel.chosen is None:
            g.parameter_verdict = ParameterVerdict.DIAGNOSTIC
            g.detail = sel.reason
            for reg in REQUIRED_REGIMES:
                g.regimes[reg].status = RegimeStatus.MODEL_REJECTED
                g.regimes[reg].noise_cv = noise_by_regime.get(reg, float("nan"))
                g.regimes[reg].n_repeats = campaign.repeats
                g.regimes[reg].point_se = _point_se(
                    g.regimes[reg].noise_cv, campaign.repeats)
                g.regimes[reg].n_points = sum(
                    1 for r in recs if r.regime == reg and r.valid)
                g.regimes[reg].n_validation_points = sum(
                    1 for s in val if regime_of(s.nbytes) == reg)
                g.regimes[reg].detail = "the group's model was not accepted"
            continue

        # "piecewise/huber/abs" -> family, loss, and the scale its structure was
        # selected on. The refit has to use the same scale, or the model graded
        # in validation is not the model the selection chose.
        parts = sel.chosen_family.split("/")
        fam = parts[0]
        est = parts[1] if len(parts) > 1 else "weighted"
        scale = "absolute" if len(parts) > 2 and parts[2] == "abs" else "relative"
        builder = {"linear": fit_linear, "piecewise": fit_piecewise,
                   "regime": fit_regime}[fam]
        # Refit on the same points the selection saw, for the same reason.
        final = (builder(cal_fit, loss=est or "weighted") if fam == "linear"
                 else builder(cal_fit, loss=est or "weighted", scale=scale))
        if final is None:
            g.parameter_verdict = ParameterVerdict.DIAGNOSTIC
            g.detail = "the chosen family could not be refitted on all calibration data"
            continue

        g.parameter_verdict = ParameterVerdict.ACCEPTED
        g.model_family, g.estimator = fam, est or "weighted"
        g.n_segments = len(getattr(final, "segments", ())) or (
            2 if fam == "piecewise" else 1)
        chosen_entry = next(
            (c for c in sel.candidates if c["family"] == sel.chosen_family), {}
        )
        g.worst_regime_err = chosen_entry.get("heldout_worst_tier_err", float("nan"))

        # Independent validation, per regime, on sizes the fit never saw.
        for reg in REQUIRED_REGIMES:
            rc = g.regimes[reg]
            rc.n_points = sum(1 for r in recs if r.regime == reg and r.valid)
            in_reg = [s for s in val if regime_of(s.nbytes) == reg]
            rc.n_validation_points = len(in_reg)
            rc.noise_cv = noise_by_regime.get(reg, float("nan"))
            rc.n_repeats = campaign.repeats
            rc.point_se = _point_se(rc.noise_cv, campaign.repeats)

            if rc.n_points < campaign.min_points_per_regime:
                rc.status = RegimeStatus.INSUFFICIENT_POINTS
                rc.detail = (f"{rc.n_points} valid point(s), below the "
                             f"{campaign.min_points_per_regime} the protocol requires")
                continue
            if not in_reg:
                rc.status = RegimeStatus.NOT_VALIDATED
                rc.detail = ("no held-out point falls in this regime, so nothing "
                             "independent grades the model here")
                continue
            rc.heldout_err = final.median_rel_error(in_reg)
            if reg in bimodal_regimes:
                rc.status = RegimeStatus.BIMODAL
                rc.detail = (
                    "an algorithm-selection threshold sits in this regime "
                    f"(band {threshold.band_lo}-{threshold.band_hi} bytes"
                    + (f", median step {threshold.step_factor:.1f}x at "
                       f"{threshold.step_at}B" if threshold.step_at else "")
                    + "). The transport has two behaviours here and no "
                      "single-valued cost model can be right about both"
                )
                continue
            if math.isfinite(rc.point_se) and rc.point_se > campaign.max_noise_cv:
                rc.status = RegimeStatus.TOO_NOISY
                rc.detail = (f"the points here carry a standard error of "
                             f"{rc.point_se:.1%} (from {rc.noise_cv:.1%} run-to-run "
                             f"variation over {rc.n_repeats} repeats), which exceeds "
                             f"the {campaign.max_noise_cv:.0%} the protocol allows, so a "
                             "model cannot be graded here")
                continue
            if not math.isfinite(rc.heldout_err) or \
                    rc.heldout_err > campaign.max_worst_regime_err:
                rc.status = RegimeStatus.ERROR_TOO_HIGH
                rc.detail = (f"held-out error {rc.heldout_err:.1%} exceeds the "
                             f"{campaign.max_worst_regime_err:.0%} threshold")
                continue
            rc.status = RegimeStatus.COVERED
            rc.detail = f"held-out error {rc.heldout_err:.1%} over {len(in_reg)} point(s)"

    campaign.coverage = cov
    return campaign
