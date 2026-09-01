"""Machine stability: is this environment steady enough to measure on?

Two measurement sessions hours apart differed by up to 2.7x on identical work.
That single fact invalidates every comparison this project makes across
sessions, and it was invisible until a number came out impossible -- adding
data made a fit worse, which cannot happen on one surface.

So stability stops being an assumption and becomes a measured quantity, with
three nested levels of repeatability that between them say *where* the
variation comes from:

``within_run``      repeats inside one process, one initialised transport.
                    This is the instrument's own noise floor.
``across_restart``  separate process launches, minutes apart. Adds process
                    placement, allocator state and transport setup.
``across_session``  launches separated in time. Adds whatever the machine does
                    while nobody is looking.

The three are nested, so their ratios identify the cause rather than merely
measuring it:

* restart ~ within        -> the extra machinery contributes nothing; variation
                             is the instrument's
* restart >> within       -> something about starting a process matters
* session >> restart      -> the machine's state drifts between sittings

and a drift that differs by collective or size is workload-specific, while one
that tracks a captured state variable is environment-state-dependent. Those
four have different remedies, which is why they are separated here rather than
summarised into one number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "DriftKind",
    "SessionState",
    "SentinelReading",
    "RepeatabilityLevel",
    "StabilityEnvelope",
    "ComparabilityVerdict",
    "SessionComparison",
    "compare_sessions",
]


class DriftKind(str, Enum):
    """What kind of thing the observed variation is."""

    RANDOM = "random"
    """Between-restart variation is no larger than within-run. Nothing to fix;
    average over more repeats."""

    PERSISTENT = "persistent"
    """A level shift that holds across restarts within a sitting and differs
    between sittings. Sessions cannot be pooled; each must be self-contained."""

    RESTART_LEVEL = "restart_level"
    """The variation lives between process launches, and time between sittings
    adds nothing on top of it. Two launches minutes apart are as different as two
    days apart, so 'measure it all in one session' is not the remedy people
    assume -- only 'measure it all in one process' would be, and that is usually
    impossible."""

    WORKLOAD_SPECIFIC = "workload_specific"
    """The drift depends on what is being measured -- one collective or one
    message size moves and the others do not. Points at an algorithm switch or
    a cache effect rather than a machine-wide state change."""

    ENVIRONMENT_DEPENDENT = "environment_dependent"
    """The drift tracks a captured state variable. Actionable: control that
    variable, or record it and condition on it."""

    UNDETERMINED = "undetermined"

    @property
    def remedy(self) -> str:
        return {
            DriftKind.RANDOM:
                "average over more repeats; no session rule is needed",
            DriftKind.PERSISTENT:
                "never pool across sessions. Every comparison must live inside "
                "one sitting, and a protocol that spans sittings is invalid",
            DriftKind.RESTART_LEVEL:
                "pooling across process launches is as unsafe as pooling across "
                "sittings. Either keep a comparison inside one process, or carry "
                "enough launches per condition that the launch-level variation is "
                "averaged rather than confounded with the effect",
            DriftKind.WORKLOAD_SPECIFIC:
                "the variation is a property of the work, not the machine; model "
                "it rather than trying to stabilise it",
            DriftKind.ENVIRONMENT_DEPENDENT:
                "control or record the state variable it tracks, and condition "
                "comparisons on it",
            DriftKind.UNDETERMINED:
                "not enough evidence to classify; do not pool until it is",
        }[self]


class ComparabilityVerdict(str, Enum):
    """Whether two sessions may be pooled, and why not when they may not.

    Three states, not two. A measurement can be invalid, or perfectly valid and
    still not poolable with another -- and collapsing those loses the
    distinction between "this run is broken" and "these two runs describe
    different machine states".
    """

    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"
    INVALID = "invalid"

    @property
    def poolable(self) -> bool:
        return self is ComparabilityVerdict.COMPARABLE


@dataclass
class SessionState:
    """Everything about the machine at the moment of one measurement.

    Captured for every reading, not once per session: a state that is only
    recorded at the start cannot explain a drift that happens during.
    """

    captured_at: str = ""
    gpu_name: str = ""
    gpu_clock_mhz: float = float("nan")
    gpu_mem_clock_mhz: float = float("nan")
    gpu_temperature_c: float = float("nan")
    gpu_utilisation_pct: float = float("nan")
    gpu_memory_used_mb: float = float("nan")
    gpu_power_w: float = float("nan")
    cpu_percent: float = float("nan")
    load_avg_1m: float = float("nan")
    n_cpu: int = 0
    free_ram_gb: float = float("nan")
    process_id: int = 0
    session_id: str = ""
    restart_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionState:
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def numeric(self) -> dict[str, float]:
        """The state variables a drift can be correlated against."""
        return {
            k: float(v) for k, v in self.__dict__.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in ("process_id", "restart_index", "n_cpu")
            and float(v) == float(v)
        }


@dataclass
class SentinelReading:
    """One timed sentinel probe, with the machine state that produced it."""

    session_id: str
    restart_index: int
    collective: str
    world: int
    nbytes: int
    size_class: str
    timings_s: list[float] = field(default_factory=list)
    state: SessionState = field(default_factory=SessionState)
    valid: bool = True
    invalid_reason: str = ""

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.collective, self.world, self.nbytes)

    @property
    def median_s(self) -> float:
        return float(np.median(self.timings_s)) if self.timings_s else float("nan")

    @property
    def within_cv(self) -> float:
        t = np.array(self.timings_s, dtype=float)
        return float(np.std(t) / np.mean(t)) if t.size > 1 and np.mean(t) > 0 else float("nan")

    @property
    def bandwidth_gbs(self) -> float:
        m = self.median_s
        if not math.isfinite(m) or m <= 0:
            return float("nan")
        factor = 2.0 if self.collective == "all_reduce" else 1.0
        steps = (self.world - 1) / self.world if self.world > 1 else 1.0
        return factor * steps * self.nbytes / m / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "restart_index": self.restart_index,
            "collective": self.collective, "world": self.world,
            "nbytes": self.nbytes, "size_class": self.size_class,
            "timings_s": self.timings_s, "median_s": self.median_s,
            "within_cv": self.within_cv, "bandwidth_gbs": self.bandwidth_gbs,
            "valid": self.valid, "invalid_reason": self.invalid_reason,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SentinelReading:
        return cls(
            session_id=d["session_id"], restart_index=d["restart_index"],
            collective=d["collective"], world=d["world"], nbytes=d["nbytes"],
            size_class=d.get("size_class", ""),
            timings_s=list(d.get("timings_s", [])),
            state=SessionState.from_dict(d.get("state", {}) or {}),
            valid=d.get("valid", True), invalid_reason=d.get("invalid_reason", ""),
        )


@dataclass
class RepeatabilityLevel:
    """Variation at one of the three nested levels."""

    name: str
    cv: float = float("nan")
    n_groups: int = 0
    per_probe: dict[str, float] = field(default_factory=dict)
    expected_cv: float = float("nan")
    """What this level's dispersion would be if the level below it were the only
    source. A level is only *adding* something when it exceeds this."""
    n_aggregated: int = 0

    @property
    def excess(self) -> float:
        """Observed dispersion over what the level below already explains."""
        if not (self.expected_cv > 0) or not math.isfinite(self.cv):
            return float("nan")
        return self.cv / self.expected_cv

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "cv": self.cv, "n_groups": self.n_groups,
                "expected_cv": self.expected_cv, "excess": self.excess,
                "n_aggregated": self.n_aggregated, "per_probe": self.per_probe}


@dataclass
class StabilityEnvelope:
    """What this machine's variation is, and what it therefore permits."""

    within_run: RepeatabilityLevel = field(
        default_factory=lambda: RepeatabilityLevel("within_run"))
    across_restart: RepeatabilityLevel = field(
        default_factory=lambda: RepeatabilityLevel("across_restart"))
    across_session: RepeatabilityLevel = field(
        default_factory=lambda: RepeatabilityLevel("across_session"))
    drift_kind: DriftKind = DriftKind.UNDETERMINED
    drift_evidence: list[str] = field(default_factory=list)
    workload_spread: float = float("nan")
    workload_spread_p: float = float("nan")
    """How often chance alone produces a spread this large.

    The spread between per-probe session CVs looks dramatic and mostly is not:
    each CV is estimated from a handful of session medians, so the ratio between
    the largest and smallest of six such estimates runs to six- or sevenfold
    under a null in which every probe behaves identically. A fixed threshold on
    this quantity reports structure that is not there."""
    state_correlations: dict[str, float] = field(default_factory=dict)
    max_session_drift_observed: float = float("nan")
    recommended_criterion: float = float("nan")
    n_sessions: int = 0
    n_restarts: int = 0
    n_readings: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def restart_over_within(self) -> float:
        """How much restarting adds *beyond* what averaging already predicts.

        Not the raw CV ratio. A restart-level reading is the median of several
        repeats, so its dispersion is smaller than the repeats' by construction;
        comparing the two raw numbers makes every level look like an improvement
        on the one below and would report a drifting machine as a stable one.
        """
        return self.across_restart.excess

    @property
    def session_over_restart(self) -> float:
        return self.across_session.excess

    @property
    def pooling_permitted(self) -> bool:
        """Whether cross-session pooling is defensible on this machine at all.

        Conjunctive on purpose. The drift classification is built from median
        behaviour across probes, while the comparability criterion is applied to
        the worst probe -- so a machine can be typically well behaved and still
        have moved more than the criterion allows on the one probe that matters.
        Permitting pooling on the median alone would let the typical case
        vouch for the exception.
        """
        if self.drift_kind is not DriftKind.RANDOM:
            return False
        obs, crit = self.max_session_drift_observed, self.recommended_criterion
        if math.isfinite(obs) and math.isfinite(crit) and obs > crit:
            return False
        return True

    @property
    def median_random_but_worst_drifts(self) -> bool:
        """The informative disagreement between the two statistics."""
        obs, crit = self.max_session_drift_observed, self.recommended_criterion
        return (self.drift_kind is DriftKind.RANDOM and math.isfinite(obs)
                and math.isfinite(crit) and obs > crit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "within_run": self.within_run.to_dict(),
            "across_restart": self.across_restart.to_dict(),
            "across_session": self.across_session.to_dict(),
            "restart_over_within": self.restart_over_within,
            "session_over_restart": self.session_over_restart,
            "drift_kind": self.drift_kind.value,
            "drift_remedy": self.drift_kind.remedy,
            "drift_evidence": self.drift_evidence,
            "workload_spread": self.workload_spread,
            "workload_spread_p": self.workload_spread_p,
            "state_correlations": self.state_correlations,
            "max_session_drift_observed": self.max_session_drift_observed,
            "recommended_criterion": self.recommended_criterion,
            "pooling_permitted": self.pooling_permitted,
            "median_random_but_worst_drifts": self.median_random_but_worst_drifts,
            "n_sessions": self.n_sessions, "n_restarts": self.n_restarts,
            "n_readings": self.n_readings,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "| Level | observed CV | expected from below | probes | what it adds |",
            "|---|---|---|---|---|",
            f"| within run | {self.within_run.cv:.1%} | n/a | {self.within_run.n_groups} "
            "| the instrument's own noise floor |",
            f"| across restart | {self.across_restart.cv:.1%} | "
            f"{self.across_restart.expected_cv:.1%} | {self.across_restart.n_groups} "
            "| process placement, allocator and transport setup |",
            f"| across session | {self.across_session.cv:.1%} | "
            f"{self.across_session.expected_cv:.1%} | {self.across_session.n_groups} "
            "| whatever the machine does between sittings |",
            "",
            f"Excess over the level below: restart {self.restart_over_within:.2f}x "
            f"&nbsp;&middot;&nbsp; session {self.session_over_restart:.2f}x. "
            "A level that adds nothing of its own sits at 1.0x, because a median "
            "of *n* is less dispersed than the *n* it averages.",
            "",
            f"**Drift is {self.drift_kind.value.upper()}.** {self.drift_kind.remedy}.",
            "",
        ]
        lines += [f"- {e}" for e in self.drift_evidence]
        if self.state_correlations:
            top = sorted(self.state_correlations.items(),
                         key=lambda kv: -abs(kv[1]))[:5]
            lines += ["", "Correlation of per-reading bandwidth with machine state:", ""]
            lines += [f"- `{k}`: r = {v:+.2f}" for k, v in top]
        lines += [
            "",
            f"Largest session-to-session drift observed: "
            f"**{self.max_session_drift_observed:.2f}x**. "
            f"Recommended comparability criterion: "
            f"**{self.recommended_criterion:.2f}x**.",
        ]
        if self.notes:
            lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


MEDIAN_SE_FACTOR = 1.2533
"""sqrt(pi/2): the standard error of a median relative to that of a mean."""

WORKLOAD_ALPHA = 0.05
"""Significance the permutation test must clear before workload-specificity is
claimed. Registered here rather than at the call site so it cannot differ
between two analyses of the same data."""

MIN_SESSIONS_FOR_WORKLOAD = 3
"""Below this the per-probe session CV has too few degrees of freedom to compare."""


def _cv(xs: Sequence[float]) -> float:
    a = np.array([x for x in xs if math.isfinite(x)], dtype=float)
    if a.size < 2 or a.mean() <= 0:
        return float("nan")
    return float(a.std() / a.mean())


def workload_specificity_p(readings: Sequence[SentinelReading], *,
                          n_perm: int = 4000, seed: int = 20260831) -> float:
    """How often chance produces a per-probe CV spread as large as observed.

    The null is that every probe shares one session-level variability and the
    session labels carry no probe-specific meaning, so shuffling which session
    label each launch-group wears should produce spreads like the real one.
    Permutation rather than a formula because each per-probe CV rests on a
    handful of session medians and no closed form for the ratio of the largest
    to the smallest of six such estimates is worth trusting at that sample size.
    """
    valid = [r for r in readings if r.valid and math.isfinite(r.median_s)]
    if not valid:
        return float("nan")

    def pk(r: SentinelReading) -> str:
        return f"{r.collective}/world{r.world}/{r.size_class}"

    probes = sorted({pk(r) for r in valid})
    sessions = sorted({r.session_id for r in valid})
    if len(probes) < 2 or len(sessions) < MIN_SESSIONS_FOR_WORKLOAD:
        return float("nan")

    # Per probe, the launch groups that can be relabelled.
    groups: dict[str, list[tuple[str, int, float]]] = {}
    for p_ in probes:
        seen: dict[tuple[str, int], list[float]] = {}
        for r in valid:
            if pk(r) == p_:
                seen.setdefault((r.session_id, r.restart_index), []).append(r.median_s)
        groups[p_] = [(sid, ri, float(np.median(v))) for (sid, ri), v in seen.items()]

    def spread(labelled: dict[str, list[tuple[str, float]]]) -> float:
        cvs = []
        for p_ in probes:
            by: dict[str, list[float]] = {}
            for sid, m in labelled[p_]:
                by.setdefault(sid, []).append(m)
            meds = [float(np.median(v)) for v in by.values()]
            c = _cv(meds)
            if math.isfinite(c) and c > 0:
                cvs.append(c)
        if len(cvs) < 2:
            return float("nan")
        return max(cvs) / min(cvs)

    observed = spread({p_: [(sid, m) for sid, _, m in groups[p_]] for p_ in probes})
    if not math.isfinite(observed):
        return float("nan")

    rng = np.random.default_rng(seed)
    hits = 0
    n_ok = 0
    for _ in range(n_perm):
        lab: dict[str, list[tuple[str, float]]] = {}
        for p_ in probes:
            sids = [g[0] for g in groups[p_]]
            rng.shuffle(sids)
            lab[p_] = [(sids[i], groups[p_][i][2]) for i in range(len(sids))]
        v = spread(lab)
        if math.isfinite(v):
            n_ok += 1
            if v >= observed:
                hits += 1
    if not n_ok:
        return float("nan")
    # Add-one correction: a permutation p of exactly zero overstates the evidence
    # available from a finite number of shuffles.
    return (hits + 1) / (n_ok + 1)


def build_envelope(readings: Sequence[SentinelReading], *,
                   random_tolerance: float = 1.5,
                   workload_tolerance: float = 2.0) -> StabilityEnvelope:
    """Decompose the variation into its three levels and name its cause."""
    valid = [r for r in readings if r.valid and math.isfinite(r.median_s)]
    env = StabilityEnvelope(n_readings=len(valid))
    if not valid:
        env.notes.append("no valid readings")
        return env

    env.n_sessions = len({r.session_id for r in valid})
    env.n_restarts = len({(r.session_id, r.restart_index) for r in valid})

    def probe_key(r: SentinelReading) -> str:
        return f"{r.collective}/world{r.world}/{r.size_class}"

    # within run: dispersion of the repeats inside each reading
    per_probe_within: dict[str, list[float]] = {}
    for r in valid:
        if math.isfinite(r.within_cv):
            per_probe_within.setdefault(probe_key(r), []).append(r.within_cv)
    env.within_run.per_probe = {k: float(np.median(v))
                                for k, v in per_probe_within.items()}
    env.within_run.cv = (float(np.median(list(env.within_run.per_probe.values())))
                         if env.within_run.per_probe else float("nan"))
    env.within_run.n_groups = len(env.within_run.per_probe)
    n_repeats = int(np.median([len(r.timings_s) for r in valid if r.timings_s]) or 1)
    env.within_run.n_aggregated = n_repeats

    # across restart: dispersion of medians between restarts, within a session
    per_probe_restart: dict[str, list[float]] = {}
    for sess in {r.session_id for r in valid}:
        for pk in {probe_key(r) for r in valid}:
            xs = [r.median_s for r in valid
                  if r.session_id == sess and probe_key(r) == pk]
            c = _cv(xs)
            if math.isfinite(c):
                per_probe_restart.setdefault(pk, []).append(c)
    env.across_restart.per_probe = {k: float(np.median(v))
                                    for k, v in per_probe_restart.items()}
    env.across_restart.cv = (float(np.median(list(env.across_restart.per_probe.values())))
                             if env.across_restart.per_probe else float("nan"))
    env.across_restart.n_groups = len(env.across_restart.per_probe)
    # A restart-level reading is a median of `n_repeats` timings. The standard
    # error of a median is about 1.253 sigma / sqrt(n), so this is the dispersion
    # the restart level would show if it added nothing of its own.
    n_rst = max(int(np.median([
        len({r.restart_index for r in valid if r.session_id == s_ and probe_key(r) == pk})
        for s_ in {r.session_id for r in valid}
        for pk in {probe_key(r) for r in valid}] or [1])), 1)
    env.across_restart.n_aggregated = n_rst
    if math.isfinite(env.within_run.cv) and n_repeats > 0:
        env.across_restart.expected_cv = MEDIAN_SE_FACTOR * env.within_run.cv / math.sqrt(n_repeats)

    # across session: dispersion of per-session medians
    per_probe_session: dict[str, list[float]] = {}
    session_medians: dict[str, dict[str, float]] = {}
    for pk in {probe_key(r) for r in valid}:
        by_sess = {}
        for sess in {r.session_id for r in valid}:
            xs = [r.median_s for r in valid
                  if r.session_id == sess and probe_key(r) == pk]
            if xs:
                by_sess[sess] = float(np.median(xs))
        session_medians[pk] = by_sess
        c = _cv(list(by_sess.values()))
        if math.isfinite(c):
            per_probe_session[pk] = c
    env.across_session.per_probe = per_probe_session
    env.across_session.cv = (float(np.median(list(per_probe_session.values())))
                             if per_probe_session else float("nan"))
    env.across_session.n_groups = len(per_probe_session)
    env.across_session.n_aggregated = env.n_sessions
    if math.isfinite(env.across_restart.cv) and n_rst > 0:
        env.across_session.expected_cv = MEDIAN_SE_FACTOR * env.across_restart.cv / math.sqrt(n_rst)

    # largest session-to-session ratio on any probe
    ratios = []
    for by_sess in session_medians.values():
        vals = [v for v in by_sess.values() if v > 0]
        if len(vals) >= 2:
            ratios.append(max(vals) / min(vals))
    env.max_session_drift_observed = float(max(ratios)) if ratios else float("nan")

    # Is the drift the same everywhere, or specific to some work? With only two
    # sessions each per-probe CV has one degree of freedom, so the ratio between
    # probes is dominated by chance and is not evidence of anything.
    if per_probe_session and env.n_sessions >= MIN_SESSIONS_FOR_WORKLOAD:
        vals = [v for v in per_probe_session.values() if math.isfinite(v) and v > 0]
        if len(vals) >= 2 and min(vals) > 0:
            env.workload_spread = float(max(vals) / min(vals))
            env.workload_spread_p = workload_specificity_p(valid)
    elif per_probe_session:
        env.notes.append(
            f"workload-specificity not tested: {env.n_sessions} session(s) give the "
            "per-probe session CV one degree of freedom, and the ratio between "
            "probes would be chance rather than structure"
        )

    # Does bandwidth track any captured state variable?
    keys: set[str] = set()
    for r in valid:
        keys |= set(r.state.numeric())
    for k in sorted(keys):
        xs, ys = [], []
        for r in valid:
            v = r.state.numeric().get(k)
            if v is not None and math.isfinite(r.bandwidth_gbs):
                xs.append(v)
                ys.append(r.bandwidth_gbs)
        if len(xs) >= 6 and np.std(xs) > 0 and np.std(ys) > 0:
            env.state_correlations[k] = float(np.corrcoef(xs, ys)[0, 1])

    env.drift_kind, env.drift_evidence = _classify(
        env, random_tolerance=random_tolerance, workload_tolerance=workload_tolerance
    )
    # The criterion is set from the observed session drift plus headroom, but
    # never tighter than the restart-level variation: a rule stricter than the
    # machine's own repeatability would reject every honest pair.
    floor = 1.0 + 3.0 * (env.across_restart.cv if math.isfinite(env.across_restart.cv) else 0.05)
    env.recommended_criterion = float(max(floor, 1.05))
    return env


def _classify(env: StabilityEnvelope, *, random_tolerance: float,
              workload_tolerance: float) -> tuple[DriftKind, list[str]]:
    ev: list[str] = []
    rw, sr = env.restart_over_within, env.session_over_restart

    strong_state = {k: v for k, v in env.state_correlations.items() if abs(v) >= 0.7}

    if math.isfinite(rw):
        ev.append(
            f"restarting the process leaves {rw:.2f}x the dispersion that averaging "
            f"alone predicts ({env.across_restart.cv:.1%} observed against "
            f"{env.across_restart.expected_cv:.1%} expected from the "
            f"{env.within_run.cv:.1%} within-run noise)"
        )
    if math.isfinite(sr):
        ev.append(
            f"a new session leaves {sr:.2f}x what restarts alone predict "
            f"({env.across_session.cv:.1%} observed against "
            f"{env.across_session.expected_cv:.1%} expected)"
        )
    if math.isfinite(env.workload_spread):
        pv = env.workload_spread_p
        if math.isfinite(pv):
            ev.append(
                f"the session-level variation differs by {env.workload_spread:.1f}x "
                f"between the probes, which chance alone reproduces {pv:.0%} of the "
                "time (" + ("so the difference is real"
                            if pv < WORKLOAD_ALPHA else
                            "so it is not evidence of anything: each per-probe CV "
                            "rests on a handful of session medians and spreads of "
                            "this size are ordinary under a null where every probe "
                            "behaves identically") + ")"
            )
        else:
            ev.append(
                f"the session-level variation differs by {env.workload_spread:.1f}x "
                "between the probes, untested for significance"
            )

    if strong_state:
        k, v = max(strong_state.items(), key=lambda kv: abs(kv[1]))
        ev.append(f"bandwidth correlates with `{k}` at r = {v:+.2f}")
        return DriftKind.ENVIRONMENT_DEPENDENT, ev

    # Workload-specificity now needs the permutation test to agree, not merely a
    # large ratio. The ratio is large under the null too.
    if (math.isfinite(env.workload_spread)
            and env.workload_spread >= workload_tolerance
            and math.isfinite(env.workload_spread_p)
            and env.workload_spread_p < WORKLOAD_ALPHA):
        return DriftKind.WORKLOAD_SPECIFIC, ev

    restart_adds = math.isfinite(rw) and rw >= random_tolerance
    session_adds = math.isfinite(sr) and sr >= random_tolerance

    if session_adds:
        return DriftKind.PERSISTENT, ev

    if restart_adds:
        ev.append(
            "the variation enters at the process launch and time between sittings "
            "adds nothing on top of it, so two launches minutes apart differ as "
            "much as two days apart"
        )
        return DriftKind.RESTART_LEVEL, ev

    if math.isfinite(rw) and not restart_adds and (
            not math.isfinite(sr) or not session_adds):
        ev.append(
            "no level adds materially to the one below it, so the variation is the "
            "instrument's own and averages down with repeats"
        )
        return DriftKind.RANDOM, ev

    return DriftKind.UNDETERMINED, ev


# --------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------


@dataclass
class SessionComparison:
    """Whether two sessions may be pooled, decided against a registered rule."""

    verdict: ComparabilityVerdict
    criterion: float
    observed_ratio: float = float("nan")
    per_probe_ratio: dict[str, float] = field(default_factory=dict)
    shape_divergence: float = float("nan")
    """How differently the probes moved *relative to each other*.

    1.0 means the whole surface scaled by one factor; large means it deformed.
    Separate from magnitude because the two fail independently: a session can
    match another's median exactly while being slower on small messages and
    faster on large, and pooling those two produces a shape neither measured.
    """
    failed_gate: str = ""
    level_shift: float = float("nan")
    """The median directed ratio: how far the whole surface moved.

    The high-power complement to the worst-probe gate. A maximum over hundreds
    of probes has to be judged against a wide noise floor and therefore sees
    only gross drift; a median over the same probes has a standard error that
    shrinks with their number and detects a systematic shift the maximum would
    miss. The two fail on different things, so both are gates.
    """
    effective_criterion: float = float("nan")
    """The registered criterion, or the noise floor where that is larger."""
    effective_shape_criterion: float = float("nan")
    n_shared_probes: int = 0
    reason: str = ""
    session_a: str = ""
    session_b: str = ""

    @property
    def poolable(self) -> bool:
        return self.verdict.poolable

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value, "poolable": self.poolable,
            "criterion": self.criterion, "observed_ratio": self.observed_ratio,
            "shape_divergence": self.shape_divergence,
            "level_shift": self.level_shift,
            "failed_gate": self.failed_gate,
            "effective_criterion": self.effective_criterion,
            "effective_shape_criterion": self.effective_shape_criterion,
            "per_probe_ratio": self.per_probe_ratio,
            "n_shared_probes": self.n_shared_probes,
            "session_a": self.session_a, "session_b": self.session_b,
            "reason": self.reason,
        }


def null_from_replicates(readings: Sequence[SentinelReading]
                         ) -> tuple[float, float, int]:
    """What the worst-probe and shape statistics reach on noise alone.

    Splits each reading's own repeats into two halves and compares those halves
    as if they were two sessions. They are the same launch by construction, so
    whatever spread appears is measurement noise -- measured at exactly this
    probe count and this replicate count.

    Both matter, and both were got wrong once. A criterion of the form
    ``1 + 3 sigma`` bounds *one* comparison; applied to the worst of 276 it is
    exceeded almost surely even by a perfectly stable machine. And a criterion
    derived from medians of seven repeats does not transfer to medians of two.
    Taking the null from the data removes both assumptions at once.

    Returns (worst_ratio, shape_divergence, n_probes), all NaN if the readings
    carry too few repeats to split.
    """
    # Split within each (session, probe) separately. Pooling readings across the
    # two sessions before splitting would average away the very noise the null is
    # meant to measure, and understates it by about half.
    per_session: dict[str, dict[tuple, list[tuple[float, float]]]] = {}
    for r in readings:
        if not r.valid or len(r.timings_s) < 2:
            continue
        t = list(r.timings_s)
        mid = max(len(t) // 2, 1)
        lo, hi = float(np.median(t[:mid])), float(np.median(t[mid:]))
        if lo > 0 and hi > 0:
            per_session.setdefault(r.session_id, {}).setdefault(r.key, []).append((lo, hi))

    worst_by_session, shape_by_session, counts = [], [], []
    for probes in per_session.values():
        ratios, directed = [], []
        for pairs in probes.values():
            lo = float(np.median([x for x, _ in pairs]))
            hi = float(np.median([y for _, y in pairs]))
            if lo > 0 and hi > 0:
                ratios.append(max(lo, hi) / min(lo, hi))
                directed.append(hi / lo)
        if len(ratios) >= 2:
            worst_by_session.append(max(ratios))
            shape_by_session.append(max(directed) / min(directed)
                                    if min(directed) > 0 else float("inf"))
            counts.append(len(ratios))

    if not worst_by_session:
        return float("nan"), float("nan"), 0
    # The largest of the per-session nulls: the threshold has to clear what the
    # noisier of the two sessions reaches on its own.
    return (float(max(worst_by_session)), float(max(shape_by_session)),
            int(max(counts)))


def compare_sessions(a: Sequence[SentinelReading], b: Sequence[SentinelReading], *,
                     criterion: float, min_shared_probes: int = 3,
                     use_internal_null: bool = False
                     ) -> SessionComparison:
    """Decide COMPARABLE / INCOMPARABLE / INVALID for two sessions.

    Compared probe by probe rather than on a pooled median. Two sessions can
    share a median and differ completely in shape -- one slower on small
    messages and faster on large -- and pooling those produces a surface
    neither session would recognise. The verdict turns on the worst probe.
    """
    va = [r for r in a if r.valid and math.isfinite(r.median_s) and r.median_s > 0]
    vb = [r for r in b if r.valid and math.isfinite(r.median_s) and r.median_s > 0]
    sid_a = va[0].session_id if va else ""
    sid_b = vb[0].session_id if vb else ""

    if not va or not vb:
        return SessionComparison(
            ComparabilityVerdict.INVALID, criterion, session_a=sid_a, session_b=sid_b,
            reason="one session has no valid readings, so nothing can be compared",
        )

    def medians(rs: Sequence[SentinelReading]) -> dict[tuple, float]:
        out: dict[tuple, list[float]] = {}
        for r in rs:
            out.setdefault(r.key, []).append(r.median_s)
        return {k: float(np.median(v)) for k, v in out.items()}

    ma, mb = medians(va), medians(vb)
    shared = sorted(set(ma) & set(mb))
    if len(shared) < min_shared_probes:
        return SessionComparison(
            ComparabilityVerdict.INVALID, criterion, n_shared_probes=len(shared),
            session_a=sid_a, session_b=sid_b,
            reason=(
                f"only {len(shared)} shared probe(s), below the {min_shared_probes} "
                "the protocol requires. Two sessions that measured different things "
                "cannot be shown comparable by measuring neither."
            ),
        )

    per: dict[str, float] = {}
    directed: dict[str, float] = {}
    for k in shared:
        x, y = ma[k], mb[k]
        label = f"{k[0]}/world{k[1]}/{k[2]}"
        per[label] = max(x, y) / min(x, y)
        directed[label] = y / x
    worst = max(per.values())
    dvals = list(directed.values())
    shape = max(dvals) / min(dvals) if min(dvals) > 0 else float("inf")

    # With many probes, or few repeats per probe, the registered criterion is the
    # wrong yardstick: it bounds one comparison at the sentinel's replicate
    # count. Where the data can supply its own null, a difference counts only if
    # it exceeds what noise alone reaches at this scale.
    eff_mag, eff_shape = criterion, criterion
    if use_internal_null:
        nw, ns, nk = null_from_replicates(list(a) + list(b))
        if math.isfinite(nw) and nk >= min_shared_probes:
            eff_mag = max(criterion, nw)
            eff_shape = max(criterion, ns) if math.isfinite(ns) else criterion

    # The level shift is a median, so multiplicity does not inflate it and the
    # registered criterion applies to it directly however many probes there are.
    level = float(np.median(dvals))
    level_ratio = max(level, 1.0 / level) if level > 0 else float("inf")

    magnitude_ok = worst <= eff_mag
    shape_ok = shape <= eff_shape
    level_ok = level_ratio <= criterion
    cmp = SessionComparison(
        ComparabilityVerdict.COMPARABLE if (magnitude_ok and shape_ok and level_ok)
        else ComparabilityVerdict.INCOMPARABLE,
        criterion, observed_ratio=worst, per_probe_ratio=per,
        shape_divergence=shape, level_shift=level, n_shared_probes=len(shared),
        session_a=sid_a, session_b=sid_b,
        effective_criterion=eff_mag, effective_shape_criterion=eff_shape,
    )
    if cmp.poolable:
        extra = ("" if eff_mag == criterion else
                 f", after widening to {eff_mag:.2f}x because measurement noise "
                 f"alone reaches that across {len(shared)} probes")
        cmp.reason = (
            f"the surface did not move ({level:.3f}x median shift), every shared "
            f"probe agrees within {worst:.2f}x, and the probes moved together "
            f"(shape divergence {shape:.2f}x){extra}"
        )
    elif not level_ok:
        cmp.failed_gate = "level"
        cmp.reason = (
            f"the whole surface moved by {level:.2f}x, past the {criterion:.2f}x the "
            "protocol registers. This is a median over "
            f"{len(shared)} probes rather than an extreme, so no widening for "
            "multiplicity applies to it: the shift is systematic, not one bad probe."
        )
    elif not magnitude_ok:
        cmp.failed_gate = "magnitude"
        offender = max(per, key=lambda k: per[k])
        cmp.reason = (
            f"the worst shared probe ({offender}) differs by {worst:.2f}x, beyond the "
            f"{eff_mag:.2f}x threshold. These sessions measured "
            "different machine states and pooling them would describe the difference "
            "between the sittings rather than the transport."
        )
    else:
        cmp.failed_gate = "shape"
        fast = min(directed, key=lambda k: directed[k])
        slow = max(directed, key=lambda k: directed[k])
        cmp.reason = (
            f"no single probe moved more than {worst:.2f}x, but they did not move "
            f"together: {slow} went one way and {fast} the other, a shape divergence "
            f"of {shape:.2f}x against a {eff_shape:.2f}x threshold. The surface "
            "deformed rather than shifted, and a fit spanning both sessions would "
            "describe neither."
        )
    return cmp
