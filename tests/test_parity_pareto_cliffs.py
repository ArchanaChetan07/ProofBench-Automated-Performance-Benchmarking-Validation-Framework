"""Parity certification, Pareto frontiers and cliff detection."""

from __future__ import annotations

import numpy as np
import pytest

from losscolumn.core.cliffs import cliff_adjacency, detect_cliffs
from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.pareto import (
    OperatingPoint,
    ParetoFrontier,
    compare_frontiers,
)
from losscolumn.core.parity import (
    SearchSpace,
    Trial,
    TuningLedger,
    certify,
    certify_grouped,
)


def ledger(system: str, *, trials: int = 40, operator: str = "op", hw: str = "hw",
           wl: str = "wl", dims: int = 4, wall: float = 10.0, curve: str = "plateau",
           group: str = "") -> TuningLedger:
    space = SearchSpace(name=system, dims={f"k{i}": [1, 2, 3, 4] for i in range(dims)})
    led = TuningLedger(
        system=system, space=space, search_algorithm="rand@1", search_seed=0,
        operator=operator, hardware_fingerprint=hw, workload_digest=wl,
        objective_name="goodput", group=group, budget_trials=trials,
    )
    for i in range(trials):
        obj = (1.0 + i * 0.05) if curve == "rising" else (1.0 if i > 5 else 0.5 + i * 0.1)
        led.record(Trial(system=system, index=i, config={"k0": i % 4}, objective=obj,
                         wall_time_s=wall))
    led.seal()
    return led


class TestLedger:
    def test_is_append_only(self):
        led = ledger("a", trials=3)
        with pytest.raises(RuntimeError, match="sealed"):
            led.record(Trial(system="a", index=3, config={}, objective=1.0))

    def test_rejects_out_of_order_trials(self):
        space = SearchSpace(name="a", dims={"k": [1]})
        led = TuningLedger(system="a", space=space, search_algorithm="x", search_seed=0,
                           operator="o", hardware_fingerprint="h", workload_digest="w",
                           objective_name="g")
        with pytest.raises(ValueError, match="out-of-order"):
            led.record(Trial(system="a", index=7, config={}, objective=1.0))

    def test_still_improving_detects_a_binding_budget(self):
        assert ledger("a", curve="rising").still_improving()
        assert not ledger("a", curve="plateau").still_improving()

    def test_improvement_curve_is_monotone(self):
        curve = ledger("a").improvement_curve()
        assert all(b >= a for a, b in zip(curve, curve[1:], strict=False))


class TestCertification:
    def test_matched_ledgers_pass(self):
        cert = certify([ledger("a"), ledger("b")])
        assert cert.ok

    def test_unequal_trials_are_fatal(self):
        cert = certify([ledger("a", trials=40), ledger("b", trials=20)])
        assert not cert.ok
        assert any(v.code == "PAR-001" for v in cert.violations)

    def test_different_hardware_is_fatal(self):
        cert = certify([ledger("a", hw="h1"), ledger("b", hw="h2")])
        assert not cert.ok
        assert any(v.code == "PAR-004" for v in cert.violations)

    def test_different_workload_is_fatal(self):
        cert = certify([ledger("a", wl="w1"), ledger("b", wl="w2")])
        assert any(v.code == "PAR-005" for v in cert.violations)

    def test_different_operator_is_a_documented_warning(self):
        cert = certify([ledger("a", operator="x"), ledger("b", operator="y")])
        assert cert.ok, "operator skill is documented, not eliminated"
        assert any(v.code == "PAR-003" for v in cert.violations)

    def test_unequal_search_difficulty_warns(self):
        cert = certify([ledger("a", dims=2), ledger("b", dims=8)])
        assert cert.ok
        assert any(v.code == "PAR-010" for v in cert.violations)

    def test_unequal_space_coverage_warns(self):
        cert = certify([ledger("a", dims=2), ledger("b", dims=8)])
        assert any(v.code == "PAR-011" for v in cert.violations)

    def test_wall_clock_asymmetry_warns(self):
        """Equal trials can cost wildly unequal time -- a compiled engine pays
        a build per configuration. A reader choosing under a time budget rather
        than a trial budget would reach a different conclusion."""
        cert = certify([ledger("a", wall=10), ledger("b", wall=500)])
        assert any(v.code == "PAR-012" for v in cert.violations)

    def test_unsealed_ledger_is_fatal(self):
        led = ledger("a")
        led.finished_at = None
        cert = certify([led, ledger("b")])
        assert not cert.ok
        assert any(v.code == "PAR-007" for v in cert.violations)

    def test_single_system_cannot_have_parity(self):
        assert not certify([ledger("a")]).ok


class TestGroupedCertification:
    def test_per_group_workloads_do_not_trip_the_workload_rule(self):
        groups = {
            "chat": [ledger("a", wl="chat", group="chat"), ledger("b", wl="chat", group="chat")],
            "rag": [ledger("a", wl="rag", group="rag"), ledger("b", wl="rag", group="rag")],
        }
        cert = certify_grouped(groups)
        assert cert.ok
        assert not any(v.code == "PAR-005" for v in cert.violations)
        assert len(cert.ledgers) == 4

    def test_findings_repeated_in_every_group_are_stated_once(self):
        groups = {
            g: [ledger("a", wl=g, group=g, dims=2), ledger("b", wl=g, group=g, dims=8)]
            for g in ("w1", "w2", "w3")
        }
        cert = certify_grouped(groups)
        par10 = [v for v in cert.violations if v.code == "PAR-010"]
        assert len(par10) == 1
        assert "all 3 groups" in par10[0].message


class TestPareto:
    def _frontier(self) -> ParetoFrontier:
        pts = [
            OperatingPoint({}, latency_ms=10, throughput=100, label="a", tag="default"),
            OperatingPoint({}, latency_ms=20, throughput=180, label="b"),
            OperatingPoint({}, latency_ms=15, throughput=90, label="dominated"),
            OperatingPoint({}, latency_ms=40, throughput=200, label="c", tag="tuned"),
        ]
        return ParetoFrontier(system="e", points=pts)

    def test_dominated_points_are_excluded(self):
        labels = [p.label for p in self._frontier().frontier()]
        assert "dominated" not in labels
        assert labels == ["a", "b", "c"]

    def test_attainment_is_monotone(self):
        a = self._frontier().attainment(np.array([10.0, 20.0, 40.0, 80.0]))
        assert list(a) == [100.0, 180.0, 200.0, 200.0]

    def test_unmeetable_budget_is_nan_not_zero(self):
        """Zero would average into a favourable comparison; NaN cannot."""
        a = self._frontier().attainment(np.array([1.0]))
        assert np.isnan(a[0])

    def test_default_to_tuned_gap(self):
        gap = self._frontier().default_to_tuned_gap()
        assert gap is not None
        assert gap["throughput_gain_pct"] == pytest.approx(100.0)

    def test_robust_frontier_is_no_larger_than_the_possible_one(self):
        pts = [
            OperatingPoint({}, 10, 100, latency_ci=(9, 11), throughput_ci=(90, 110)),
            OperatingPoint({}, 12, 105, latency_ci=(11, 13), throughput_ci=(95, 115)),
        ]
        fr = ParetoFrontier(system="e", points=pts)
        assert len(fr.frontier("robust")) <= len(fr.frontier("possible"))

    def test_frontier_comparison_finds_the_crossover(self):
        fast = ParetoFrontier(system="fast", points=[
            OperatingPoint({}, 5, 50), OperatingPoint({}, 10, 80),
            OperatingPoint({}, 50, 100),
        ])
        big = ParetoFrontier(system="big", points=[
            OperatingPoint({}, 20, 40), OperatingPoint({}, 50, 300),
        ])
        cmp = compare_frontiers(fast, big, n_budgets=40)
        s = cmp.summary()
        assert s["frac_budgets_method_wins"] > 0
        assert s["frac_budgets_method_loses"] > 0
        assert cmp.crossover() is not None


class TestCliffs:
    def _env(self, values: dict, replicates: int = 9, noise: float = 0.01) -> Envelope:
        rng = np.random.default_rng(0)
        env = Envelope.allocate(
            (Factor("batch", (1, 2, 4, 8)), Factor("mode", ("a", "b"), ordered=False)),
            Metric("throughput", "tok/s", higher_is_better=True),
            ["sys"], replicates, interleaved=True,
        )
        for cell in env.cells():
            base = values[env.coords(cell)["batch"]]
            env.put("sys", cell, base * (1 + rng.normal(0, noise, replicates)))
        return env

    def test_finds_a_planted_cliff(self):
        env = self._env({1: 1000, 2: 1050, 4: 1100, 8: 500})
        rep = detect_cliffs(env, "sys", min_drop=0.15)
        assert rep.n_cliffs >= 2                     # one per level of the free factor
        worst = rep.worst(1)[0]
        assert worst.axis == "batch"
        assert (worst.from_level, worst.to_level) == (4, 8)
        assert worst.degradation_pct > 100

    def test_smooth_decline_is_not_a_cliff(self):
        """A cliff is a surprise, not a slope. A surface that degrades steadily
        everywhere has no discontinuity however steep it is."""
        env = self._env({1: 1000, 2: 700, 4: 490, 8: 343})
        assert detect_cliffs(env, "sys", min_drop=0.15).n_cliffs == 0

    def test_noise_alone_does_not_manufacture_a_cliff(self):
        env = self._env({1: 1000, 2: 1000, 4: 1000, 8: 1000}, noise=0.05)
        assert detect_cliffs(env, "sys", min_drop=0.15).n_cliffs == 0

    def test_an_unrunnable_level_is_the_steepest_cliff(self):
        env = self._env({1: 1000, 2: 1000, 4: 1000, 8: 1000})
        for cell in env.cells():
            if env.coords(cell)["batch"] == 8:
                env.mark_missing("sys", cell, "OOM")
        rep = detect_cliffs(env, "sys")
        assert rep.n_cliffs >= 1
        assert "stops running" in rep.worst(1)[0].describe()

    def test_adjacency_flags_a_config_next_to_a_cliff(self):
        env = self._env({1: 1000, 2: 1050, 4: 1100, 8: 500})
        rep = detect_cliffs(env, "sys", min_drop=0.15)
        rows = cliff_adjacency(env, rep, [{"name": "recommended", "batch": 4, "mode": "a"}])
        assert rows[0]["cliff_adjacent"]
        assert rows[0]["exposure_pct"] > 100

    def test_adjacency_reports_configs_outside_the_envelope(self):
        env = self._env({1: 1000, 2: 1000, 4: 1000, 8: 1000})
        rep = detect_cliffs(env, "sys")
        rows = cliff_adjacency(env, rep, [{"name": "off-grid", "batch": 999, "mode": "a"}])
        assert "error" in rows[0]
