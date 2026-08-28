"""One search procedure, applied identically to every engine.

Tuning-budget parity requires more than an equal trial count: it requires the
same *algorithm*, with the same exploration/exploitation split, the same
stopping behaviour and the same seed discipline. Two engines tuned by random
search and Bayesian optimisation respectively have not received equal budgets
in any meaningful sense, even at equal trial counts.

The procedure is deliberately simple and fully specified:

1. **Explore.** The first ``explore_frac`` of the budget is uniform random
   sampling over the engine's own space, without replacement.
2. **Refine.** The remainder is greedy coordinate descent from the best point
   found: cycle the knobs, try each untested value of the current knob, keep
   an improvement immediately.

Simple beats clever here. A stronger optimiser would find better configurations
for whichever engine happens to have the smoother response surface, and that
advantage would be indistinguishable in the results from a real difference
between the engines.

Every trial is recorded in the ledger, including failures, so a reader can
re-derive the search rather than trust its summary.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from losscolumn.core.parity import SearchSpace, Trial, TuningLedger
from losscolumn.core.provenance import content_hash, utcnow

Objective = Callable[[dict[str, Any]], tuple[float, str, dict[str, Any]]]


def engine_seed(master_seed: int, engine: str) -> int:
    """Per-engine seed derived deterministically from one master seed.

    Using the same integer for every engine would make them explore
    *correlated* positions in differently-shaped spaces, which is a subtle bias;
    deriving each from the master keeps them independent and still fully
    reproducible from the one number in the pre-registration.
    """
    return (master_seed + zlib.crc32(engine.encode())) % (2**31 - 1)


@dataclass
class SearchProcedure:
    """The shared tuning algorithm. Its name and version go in the ledger."""

    n_trials: int = 40
    explore_frac: float = 0.6
    name: str = "random-explore+coordinate-refine"
    version: str = "1.0"

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}(n={self.n_trials},explore={self.explore_frac})"

    def run(
        self,
        space: SearchSpace,
        objective: Objective,
        *,
        engine: str,
        seed: int,
        operator: str,
        hardware_fingerprint: str,
        workload_digest: str,
        objective_name: str,
        group: str = "",
    ) -> TuningLedger:
        rng = np.random.default_rng(seed)
        ledger = TuningLedger(
            system=engine,
            space=space,
            search_algorithm=self.identifier,
            search_seed=seed,
            operator=operator,
            hardware_fingerprint=hardware_fingerprint,
            workload_digest=workload_digest,
            objective_name=objective_name,
            group=group,
            budget_trials=self.n_trials,
        )
        keys = list(space.dims)
        tried: set[str] = set()
        best_cfg: dict[str, Any] | None = None
        best_val = -np.inf

        n_explore = max(int(round(self.n_trials * self.explore_frac)), 1)
        for _ in range(min(n_explore, self.n_trials)):
            cfg = {k: space.dims[k][int(rng.integers(len(space.dims[k])))] for k in keys}
            h = content_hash(cfg)
            guard = 0
            while h in tried and guard < 50:
                cfg = {k: space.dims[k][int(rng.integers(len(space.dims[k])))] for k in keys}
                h = content_hash(cfg)
                guard += 1
            tried.add(h)
            val = self._evaluate(ledger, cfg, objective)
            if val > best_val:
                best_val, best_cfg = val, dict(cfg)

        if best_cfg is None:
            best_cfg = {k: space.dims[k][0] for k in keys}

        ki = 0
        while ledger.n_trials < self.n_trials:
            key = keys[ki % len(keys)]
            ki += 1
            improved = False
            for value in space.dims[key]:
                if ledger.n_trials >= self.n_trials:
                    break
                cand = {**best_cfg, key: value}
                h = content_hash(cand)
                if h in tried:
                    continue
                tried.add(h)
                val = self._evaluate(ledger, cand, objective)
                if val > best_val:
                    best_val, best_cfg = val, cand
                    improved = True
            if not improved and ki > 4 * len(keys):
                # The neighbourhood is exhausted. Spend what is left on fresh
                # random points rather than stopping early: an unspent budget
                # would break parity with the engines that used theirs.
                while ledger.n_trials < self.n_trials:
                    cfg = {k: space.dims[k][int(rng.integers(len(space.dims[k])))] for k in keys}
                    h = content_hash(cfg)
                    if h in tried and len(tried) < space.cardinality:
                        continue
                    tried.add(h)
                    val = self._evaluate(ledger, cfg, objective)
                    if val > best_val:
                        best_val, best_cfg = val, dict(cfg)
                break

        ledger.seal()
        return ledger

    @staticmethod
    def _evaluate(ledger: TuningLedger, cfg: dict[str, Any], objective: Objective) -> float:
        started = utcnow()
        try:
            value, outcome, detail = objective(cfg)
        except Exception as e:  # a crash is a trial outcome, not an excuse to retry
            value, outcome, detail = -np.inf, "error", {"exception": f"{type(e).__name__}: {e}"}
        ledger.record(
            Trial(
                system=ledger.system,
                index=ledger.n_trials,
                config=cfg,
                objective=float(value) if np.isfinite(value) else float("-inf"),
                outcome=outcome,
                wall_time_s=float(detail.pop("wall_time_s", 0.0)),
                started_at=started,
                detail=detail,
            )
        )
        return value if np.isfinite(value) else -np.inf
