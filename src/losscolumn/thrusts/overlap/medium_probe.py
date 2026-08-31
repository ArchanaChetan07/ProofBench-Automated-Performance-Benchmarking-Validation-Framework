"""One targeted experiment, to settle one hypothesis.

The coverage debt says the medium regime is *model-limited* in every all_reduce
group: the measurement there is clean and the best available family still
misses by 22% to 35%. The cleanest case is all_reduce at world 4, where
run-to-run variation is 6.1% and the residual is 22.3%. That gap is not the
instrument's.

**Hypothesis.** The medium regime contains structure the current fit cannot
resolve because there are too few points in it to justify the segments. With 16
medium points across the whole campaign grid, AIC refuses a third or fourth
segment that the surface may actually need.

**What would discriminate it.** Densify the medium regime, and only the medium
regime, for all_reduce only. Then re-fit with the same families and the same
gate.

    if a richer segmentation now clears the gate
        -> the families were adequate and the evidence was thin
    if the error stays where it is
        -> the surface is not modellable at this granularity, and that is the
           finding rather than a reason to keep adding segments

Both outcomes are informative, which is what makes the experiment worth its
time. Nothing else is measured: the other regimes are already classified, and
re-measuring them would cost time without changing a verdict.

This is a NEW protocol, not an extension of the campaign's. Adding points to a
sealed grid after seeing its results and then reporting one number would be
choosing the grid to suit the answer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.provenance import Provenance as ProvCapture
from losscolumn.core.provenance import content_hash, utcnow
from losscolumn.thrusts.overlap.campaign import (
    REGIME_EDGES,
    PointRecord,
    regime_of,
    sweep_pass,
)
from losscolumn.thrusts.overlap.commodel import Sample, select_model

PROBE_VERSION = "thrust1-comm-medium-probe-v1"

# The medium regime, and only it.
MEDIUM_LO = next(lo for n, lo, _ in REGIME_EDGES if n == "medium")
MEDIUM_HI = next(hi for n, _, hi in REGIME_EDGES if n == "medium")

TARGET_COLLECTIVE = "all_reduce"
TARGET_WORLDS = (2, 3, 4)


def medium_grid(*, n_points: int = 16) -> tuple[int, ...]:
    """Log-spaced sizes inside the medium regime, none of them a power of two.

    Deliberately offset from powers of two: the campaign grid already sampled
    those, and repeating them would add repeats rather than resolution. What
    this experiment needs is finer spacing, not more visits to the same sizes.
    """
    xs = np.logspace(math.log10(MEDIUM_LO * 1.05), math.log10(MEDIUM_HI * 0.95),
                     n_points)
    return tuple(sorted({max(int(round(x / 4)) * 4, 4) for x in xs}))



# Two measurement sessions on the same machine are not automatically
# comparable. Beyond this ratio between their median bandwidths over a shared
# size range, pooling them describes the drift rather than the transport.
MAX_SESSION_DRIFT = 1.15


def session_comparability(prior: Sequence[PointRecord],
                          fresh: Sequence[PointRecord]) -> dict[str, Any]:
    """Are two sessions measuring the same machine state?

    This check exists because its absence produced a wrong answer. The probe
    pooled its own records with the campaign's, hours apart, and the pooled
    surface was one no model could fit -- so the best achievable error rose
    when points were added, which is impossible for a consistent surface and is
    the signature of pooling two different worlds.

    Compared over the overlapping size range rather than at shared sizes,
    because the probe deliberately samples between the campaign's sizes and the
    two grids share none.
    """
    a = [r for r in prior if r.valid and math.isfinite(r.bandwidth_gbs)]
    b = [r for r in fresh if r.valid and math.isfinite(r.bandwidth_gbs)]
    if not a or not b:
        return {"comparable": False, "reason": "one session has no valid points"}
    lo = max(min(r.nbytes for r in a), min(r.nbytes for r in b))
    hi = min(max(r.nbytes for r in a), max(r.nbytes for r in b))
    xa = [r.bandwidth_gbs for r in a if lo <= r.nbytes <= hi]
    xb = [r.bandwidth_gbs for r in b if lo <= r.nbytes <= hi]
    if not xa or not xb:
        return {"comparable": False, "reason": "the size ranges do not overlap"}
    ma, mb = float(np.median(xa)), float(np.median(xb))
    ratio = max(ma, mb) / max(min(ma, mb), 1e-12)
    ok = ratio <= MAX_SESSION_DRIFT
    return {
        "comparable": ok,
        "prior_median_gbs": ma,
        "fresh_median_gbs": mb,
        "drift_ratio": ratio,
        "threshold": MAX_SESSION_DRIFT,
        "overlap_range_bytes": [lo, hi],
        "reason": "" if ok else (
            f"the two sessions differ by {ratio:.2f}x in median bandwidth over the "
            f"same size range ({ma:.3f} against {mb:.3f} GB/s), beyond the "
            f"{MAX_SESSION_DRIFT:.2f}x this protocol allows. Pooling them would "
            "describe the drift rather than the transport, and adding points would "
            "make the best achievable fit worse -- which is impossible for a "
            "consistent surface."
        ),
    }


@dataclass
class MediumProbe:
    """The targeted experiment and the verdict it reaches."""

    grid: tuple[int, ...]
    records: list[PointRecord] = field(default_factory=list)
    before: dict[str, dict[str, Any]] = field(default_factory=dict)
    after: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdict: str = ""
    device: str = ""
    errors: dict[str, str] = field(default_factory=dict)
    max_err: float = 0.15
    comparability: dict[str, Any] = field(default_factory=dict)
    sealed_at: str = field(default_factory=utcnow)
    notes: list[str] = field(default_factory=list)

    def protocol(self) -> dict[str, Any]:
        doc = {
            "kind": "targeted-probe-protocol",
            "probe": PROBE_VERSION,
            "sealed_at": self.sealed_at,
            "hypothesis": (
                "The medium regime is model-limited because there are too few "
                "points in it to justify the segments the surface needs. With 16 "
                "medium points on the campaign grid, AIC refuses a segmentation "
                "that may be real."
            ),
            "discriminates_between": {
                "evidence_was_thin": "a richer segmentation now clears the gate",
                "surface_is_not_modellable":
                    "the error stays where it is, at this granularity",
            },
            "why_this_and_nothing_else": (
                "Every other regime is already classified as noise-limited or "
                "covered. Re-measuring them would cost time without changing a "
                "verdict, and this experiment changes one either way."
            ),
            "collective": TARGET_COLLECTIVE,
            "worlds": list(TARGET_WORLDS),
            "regime": "medium",
            "size_range_bytes": [MEDIUM_LO, MEDIUM_HI],
            "grid": list(self.grid),
            "n_points": len(self.grid),
            "grid_offset_from_powers_of_two": (
                "the campaign already sampled the powers of two in this regime; "
                "repeating them would add repeats rather than resolution"
            ),
            "acceptance": {
                "criterion": "worst-regime cross-validated relative error, unchanged",
                "max_err": self.max_err,
                "note": "the gate is the campaign's. It is not relaxed for this probe.",
            },
            "is_new_protocol_not_an_extension": (
                "Adding points to a sealed grid after seeing its results, and then "
                "reporting one number, would be choosing the grid to suit the answer. "
                "This is registered separately and its result is reported separately."
            ),
        }
        doc["seal_hash"] = content_hash(doc)
        return doc

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": PROBE_VERSION,
            "device": self.device,
            "protocol": self.protocol(),
            "n_records": len(self.records),
            "records": [r.to_dict() for r in self.records],
            "before": self.before,
            "after": self.after,
            "verdict": self.verdict,
            "session_comparability": self.comparability,
            "errors": self.errors,
            "notes": self.notes,
            "provenance": ProvCapture.capture().to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            "### Targeted probe: is the medium regime model-limited or evidence-thin?",
            "",
            f"Measured on {self.device}. {len(self.grid)} additional medium-regime "
            f"sizes, {TARGET_COLLECTIVE} only, worlds {list(TARGET_WORLDS)}. "
            "Nothing else was measured.",
            "",
            f"**{self.verdict}**",
            "",
        ]
        if self.comparability and not self.comparability.get("comparable", True):
            lines += [
                "| | prior session | this session | ratio | allowed |",
                "|---|---|---|---|---|",
                f"| median bandwidth over the shared range | "
                f"{self.comparability.get('prior_median_gbs', float('nan')):.3f} | "
                f"{self.comparability.get('fresh_median_gbs', float('nan')):.3f} | "
                f"**{self.comparability.get('drift_ratio', float('nan')):.2f}x** | "
                f"{self.comparability.get('threshold', float('nan')):.2f}x |",
                "",
            ]
        lines += [
            "| Group | medium points before | after | error before | error after | "
            "segments after |",
            "|---|---|---|---|---|---|",
        ]
        for key in sorted(self.after):
            b, a = self.before.get(key, {}), self.after[key]
            lines.append(
                f"| {key} | {b.get('n_medium', 0)} | {a.get('n_medium', 0)} | "
                f"{b.get('medium_err', float('nan')):.1%} | "
                f"{a.get('medium_err', float('nan')):.1%} | "
                f"{a.get('n_segments', 0)} |"
            )
        lines.append("")
        if self.notes:
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _medium_error(samples: Sequence[Sample], *, max_err: float
                  ) -> tuple[float, int, str]:
    """Best achievable medium-regime error over all families, and its shape."""
    sel = select_model(samples, [], transport="gloo_shm",
                       kind=TARGET_COLLECTIVE, tier_fn=regime_of,
                       max_heldout_median_err=max_err)
    best, n_seg, fam = float("nan"), 0, ""
    for c in sel.candidates:
        e = (c.get("heldout_per_tier_err") or {}).get("medium")
        if e is not None and e == e and (best != best or e < best):
            best = e
            m = c.get("model") or {}
            n_seg = len(m.get("segments", [])) or (2 if "breakpoint_bytes" in m else 1)
            fam = c["family"]
    return best, n_seg, fam


def run_probe(*, n_points: int = 16, passes: int = 2, repeats: int = 3,
              max_err: float = 0.15, progress: bool = True) -> MediumProbe:
    """Measure the denser medium grid, then re-fit with the same families."""
    import json
    from pathlib import Path

    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description

    grid = medium_grid(n_points=n_points)
    probe = MediumProbe(grid=grid, device=device_description("cuda"),
                        max_err=max_err)

    # The campaign's own records, so "before" is its result rather than a
    # re-derivation that might differ.
    camp_path = Path("artifacts") / "campaign-thrust1-communication.json"
    prior: list[PointRecord] = []
    if camp_path.exists():
        d = json.loads(camp_path.read_text(encoding="utf-8"))
        prior = [PointRecord.from_dict(r) for r in d["records"]
                 if r["collective"] == TARGET_COLLECTIVE and r["valid"]]

    for w in TARGET_WORLDS:
        base = [r for r in prior if r.world == w]
        samples = [s for r in base for s in [r.to_sample()] if s is not None]
        err, nseg, fam = _medium_error(samples, max_err=max_err) if samples else (
            float("nan"), 0, "")
        probe.before[f"{TARGET_COLLECTIVE}/world{w}"] = {
            "n_medium": sum(1 for r in base if r.regime == "medium"),
            "medium_err": err, "n_segments": nseg, "family": fam,
        }

    with MeasurementLock("medium-regime probe"):
        for p in range(passes):
            if progress:
                print(f"=== probe pass {p} ===", flush=True)
            recs, errs = sweep_pass(sizes=grid, worlds=TARGET_WORLDS,
                                    collectives=(TARGET_COLLECTIVE,),
                                    repeats=repeats, pass_index=p)
            probe.records.extend(recs)
            probe.errors.update(errs)
            if progress:
                print(f"  {len(recs)} records, {len(errs)} failure(s)", flush=True)

    # Before pooling anything, check the two sessions describe the same machine.
    fresh = [r for r in probe.records if r.valid]
    probe.comparability = session_comparability(prior, fresh)
    if not probe.comparability.get("comparable", False):
        probe.verdict = (
            "INCONCLUSIVE -- THE SESSIONS ARE NOT COMPARABLE. "
            + probe.comparability.get("reason", "")
            + " The hypothesis is untested: this measurement cannot distinguish a "
            "model-limited surface from a drifting machine, and reporting either "
            "answer from it would be reporting the drift."
        )
        probe.notes.append(
            "The probe's own method was unsound: it pooled records across two "
            "measurement sessions without checking they were comparable. The check "
            "now exists, and it fails, which is how this was found."
        )
        probe.notes.append(
            "The remedy is to re-measure the campaign grid and the denser medium "
            "grid in ONE session, so the comparison is within a single machine "
            "state. Nothing here changes any campaign verdict."
        )
        for w in TARGET_WORLDS:
            key = f"{TARGET_COLLECTIVE}/world{w}"
            base = [r for r in fresh if r.world == w]
            samples = [s for r in base for s in [r.to_sample()] if s is not None]
            err, nseg, fam = _medium_error(samples, max_err=max_err) if samples else (
                float("nan"), 0, "")
            probe.after[key] = {
                "n_medium": sum(1 for r in base if r.regime == "medium"),
                "medium_err": err, "n_segments": nseg, "family": fam,
                "note": "fresh session only; NOT pooled with the campaign",
            }
        return probe

    improved = 0
    for w in TARGET_WORLDS:
        key = f"{TARGET_COLLECTIVE}/world{w}"
        combined = [r for r in prior if r.world == w] + [
            r for r in probe.records if r.world == w and r.valid
        ]
        samples = [s for r in combined for s in [r.to_sample()] if s is not None]
        err, nseg, fam = _medium_error(samples, max_err=max_err)
        probe.after[key] = {
            "n_medium": sum(1 for r in combined if r.regime == "medium"),
            "medium_err": err, "n_segments": nseg, "family": fam,
        }
        b = probe.before.get(key, {}).get("medium_err", float("nan"))
        if err == err and err <= max_err and not (b == b and b <= max_err):
            improved += 1

    n = len(TARGET_WORLDS)
    if improved == n:
        probe.verdict = (
            f"EVIDENCE WAS THIN: with a denser medium grid, all {n} groups now clear "
            f"the {max_err:.0%} gate in that regime. The families were adequate; the "
            "campaign grid simply had too few medium points to justify the "
            "segmentation the surface needs."
        )
    elif improved:
        probe.verdict = (
            f"PARTIAL: {improved} of {n} groups clear the gate with a denser medium "
            "grid. Evidence was part of the problem and not all of it."
        )
    else:
        probe.verdict = (
            f"SURFACE IS NOT MODELLABLE AT THIS GRANULARITY: none of the {n} groups "
            f"clears the {max_err:.0%} gate in the medium regime even with "
            f"{len(grid)} additional sizes there. The gap is not evidence and it is "
            "not noise. Adding segments beyond this point would be fitting the "
            "measurement rather than the transport."
        )
    probe.notes.append(
        "Only the medium regime of one collective was measured. No other verdict in "
        "the campaign is changed by this probe, and none was re-derived from it."
    )
    return probe
