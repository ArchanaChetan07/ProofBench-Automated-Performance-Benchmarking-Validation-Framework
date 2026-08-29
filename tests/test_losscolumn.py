"""Loss-column extraction: can it recover a loss region that was planted?

These are the load-bearing tests of the whole package. If the extractor cannot
recover a known region from synthetic data with known ground truth, nothing
downstream of it means anything.
"""

from __future__ import annotations

import numpy as np

from losscolumn.core.envelope import Envelope, Factor, Metric
from losscolumn.core.losscolumn import extract_loss_column
from losscolumn.core.stats import compare_cells


def build_envelope(
    *,
    planted: dict | None = None,
    regression: float = 0.30,
    replicates: int = 11,
    noise: float = 0.01,
    seed: int = 0,
) -> Envelope:
    """A grid where ``method`` matches ``baseline`` except inside ``planted``.

    ``planted`` maps factor name -> the levels that form the loss box.
    """
    rng = np.random.default_rng(seed)
    factors = (
        Factor("batch", (1, 2, 4, 8)),
        Factor("seq_len", (128, 512, 2048)),
        Factor("dtype", ("fp16", "bf16"), ordered=False),
    )
    env = Envelope.allocate(
        factors,
        Metric("throughput", "tok/s", higher_is_better=True),
        ["method", "baseline"],
        replicates,
        interleaved=True,
    )
    for cell in env.cells():
        coords = env.coords(cell)
        base = 1000.0 * (1 + 0.1 * cell[0])
        inside = planted is not None and all(
            coords[k] in v for k, v in planted.items()
        )
        factor = (1 - regression) if inside else 1.0
        # Paired replicates share a common shock, which is what interleaved
        # A/B measurement actually produces and what the paired test assumes.
        shock = rng.normal(0, noise, replicates)
        env.put("baseline", cell, base * (1 + shock + rng.normal(0, noise / 3, replicates)))
        env.put("method", cell,
                base * factor * (1 + shock + rng.normal(0, noise / 3, replicates)))
    return env


def extract(env, mde=0.05, **kw):
    cmps = compare_cells(env, "method", "baseline", mde=mde, q=0.05, seed=0)
    return cmps, extract_loss_column(env, cmps, method="method", baseline="baseline",
                                     mde=mde, q_level=0.05, **kw)


class TestRecovery:
    def test_recovers_a_planted_box(self):
        planted = {"seq_len": [2048], "dtype": ["fp16"]}
        env = build_envelope(planted=planted)
        cmps, lc = extract(env)

        assert not lc.is_empty, "a 30% regression must produce a loss region"
        assert lc.counts["loss"] == 4, "the planted box is 4 cells (all batches)"
        r = lc.regions[0]
        assert r.bounds["seq_len"] == [2048]
        assert r.bounds["dtype"] == ["fp16"]
        assert r.bounds["batch"] == [1, 2, 4, 8], "should grow across the free axis"
        assert r.purity == 1.0
        assert 40 < r.worst_regression_pct < 48   # 1/(1-0.30) - 1 = +42.9%

    def test_no_false_region_on_a_flat_surface(self):
        env = build_envelope(planted=None)
        _, lc = extract(env)
        assert lc.is_empty
        assert lc.counts["loss"] == 0
        assert lc.counts["tie"] > 0, "equivalence should be established, not just unrejected"

    def test_subthreshold_regression_is_a_tie_not_a_loss(self):
        """A 2% regression under a 5% MDE is a tie, however significant."""
        env = build_envelope(planted={"dtype": ["fp16"]}, regression=0.02, noise=0.002)
        _, lc = extract(env, mde=0.05)
        assert lc.counts["loss"] == 0

    def test_a_win_region_is_not_reported_as_a_loss(self):
        env = build_envelope(planted={"dtype": ["fp16"]}, regression=-0.30)
        _, lc = extract(env)
        assert lc.is_empty
        assert lc.counts["win"] == 12

    def test_regions_never_swallow_a_win(self):
        """Purity: a box may absorb inconclusive cells but never a win."""
        env = build_envelope(planted={"seq_len": [2048]}, regression=0.30)
        # Make one cell inside the planted region a large win instead.
        cell = env.cell_of(batch=1, seq_len=2048, dtype="fp16")
        env.put("method", cell, env.replicates_at("baseline", cell) * 2.0)
        cmps, lc = extract(env)
        by = {tuple(c.cell): c for c in cmps}
        for r in lc.regions:
            for c in env.cells():
                if all(env.coords(c)[k] in v for k, v in r.bounds.items()):
                    assert by[c].verdict != "win"

    def test_two_disjoint_regions(self):
        env = build_envelope(planted={"seq_len": [128]}, regression=0.30)
        # Plant a second, unrelated region.
        for b in (1, 2, 4, 8):
            cell = env.cell_of(batch=b, seq_len=2048, dtype="bf16")
            env.put("method", cell, env.replicates_at("baseline", cell) * 0.6)
        _, lc = extract(env)
        assert len(lc.regions) >= 2
        assert lc.counts["loss"] == 12  # 8 in the seq_len=128 slab + 4 planted


class TestUnrunnableCells:
    def test_oom_is_a_loss_not_a_gap(self):
        env = build_envelope(planted=None)
        cell = env.cell_of(batch=8, seq_len=2048, dtype="fp16")
        env.mark_missing("method", cell, "OOM at 81 GB")
        cmps, lc = extract(env)
        by = {tuple(c.cell): c for c in cmps}
        assert by[cell].verdict == "loss"
        assert lc.n_unrunnable == 1
        assert any("OOM" in r for reg in lc.regions for r in reg.missing_reasons)

    def test_both_missing_is_excluded_not_counted(self):
        env = build_envelope(planted=None)
        cell = env.cell_of(batch=8, seq_len=2048, dtype="fp16")
        env.mark_missing("method", cell, "OOM")
        env.mark_missing("baseline", cell, "OOM")
        cmps, lc = extract(env)
        by = {tuple(c.cell): c for c in cmps}
        assert by[cell].verdict == "missing"
        assert lc.counts["loss"] == 0


class TestUnderpoweredDesign:
    """The most dangerous failure mode: a clean-looking, undetectable sweep."""

    def test_low_replicates_flag_themselves(self):
        env = build_envelope(planted={"seq_len": [2048]}, regression=0.5, replicates=5)
        _, lc = extract(env)
        assert lc.underpowered
        assert lc.is_empty, "5 replicates cannot clear FDR over 24 cells"
        assert any("UNDERPOWERED" in n for n in lc.notes)
        assert "property of the design" in lc.summary_sentence()

    def test_it_says_what_it_missed(self):
        env = build_envelope(planted={"seq_len": [2048]}, regression=0.5, replicates=5)
        _, lc = extract(env)
        assert any("understates the losses" in n for n in lc.notes)

    def test_adequate_replicates_do_not_flag(self):
        env = build_envelope(planted={"seq_len": [2048]}, regression=0.5, replicates=11)
        _, lc = extract(env)
        assert not lc.underpowered
        assert not lc.is_empty


class TestRendering:
    def test_markdown_has_a_row_per_region(self):
        env = build_envelope(planted={"seq_len": [2048], "dtype": ["fp16"]})
        _, lc = extract(env)
        md = lc.to_markdown()
        assert "Loss column" in md
        assert "seq_len=2048" in md
        assert md.count("\n|") >= 3

    def test_unrunnable_renders_as_words_not_infinity(self):
        env = build_envelope(planted=None)
        env.mark_missing("method", env.cell_of(batch=1, seq_len=128, dtype="fp16"), "OOM")
        _, lc = extract(env)
        md = lc.to_markdown()
        assert "inf" not in md.lower()
        assert "cannot run" in md
