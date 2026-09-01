"""File the withdrawal of the first drift classification.

The readings were good and the protocol was registered before they were taken.
What failed was the statistic applied to them, twice, in two different ways --
which is why this needs a withdrawal reason the register did not previously
have.
"""
from pathlib import Path


def main() -> int:
    from losscolumn import history

    art = Path("artifacts")
    d = art / "history" / "2026-08-31-unsound-drift-classification"
    # Bytes, not text: the console log carries the terminal's own encoding and
    # re-encoding it would alter the artifact being preserved.
    files = {
        "original-run.log": (d / "original-run.log").read_bytes(),
        "stability-sentinel-v1.protocol.json":
            (d / "stability-sentinel-v1.protocol.json").read_bytes(),
    }

    w = history.Withdrawal(
        artifact_id="stability-machine@2026-08-31-first-analysis",
        title="Machine stability, first analysis: drift called WORKLOAD_SPECIFIC",
        reason="unsound-statistic",
        superseded_by="stability-machine@2026-08-31-corrected "
                      "(artifacts/stability-machine.json, drift RESTART_LEVEL)",
        what_was_wrong=(
            "The drift was classified WORKLOAD_SPECIFIC because the session-level "
            "variability differed 6.9x between the six probes, against a registered "
            "threshold of 2x. Each of those six figures is a coefficient of "
            "variation estimated from four session medians. A permutation test on "
            "the same readings -- shuffling which session label each launch wears, "
            "which is the null in which every probe behaves identically -- puts the "
            "median spread at 6.4x and reproduces the observed 6.9x forty percent "
            "of the time. The threshold was measuring the sample size, not the "
            "machine.\n\n"
            "A second, independent error sat in the comparability test the same "
            "analysis used. The criterion has the form 1 + 3 sigma, which bounds "
            "ONE comparison of medians-of-seven repeats. Applied to the campaign it "
            "was being asked to bound the worst of 276 comparisons of "
            "medians-of-two, where a perfectly stable machine breaches it almost "
            "surely. That error pointed the wrong way: it declared the campaign's "
            "two passes INCOMPARABLE at 2.90x when splitting the campaign's own "
            "repeats within a single pass -- the same machine state by construction "
            "-- reaches 4.91x. The passes are more alike than the instrument is "
            "with itself."
        ),
        how_it_was_found=(
            "The first error, by not believing a large ratio without testing it. "
            "6.9x is a striking number and the honest question was how often chance "
            "produces one; the permutation test cost no measurement time and the "
            "answer was 40%.\n\n"
            "The second, by checking the test before reporting its verdict. "
            "Declaring the campaign incomparable would have put every figure "
            "derived from it in question, so the statistic behind that verdict was "
            "worth one look -- and it was a worst-of-276 judged against a "
            "worst-of-6 threshold."
        ),
        what_changed=(
            "Workload-specificity now requires a permutation test at alpha = 0.05, "
            "not a ratio. The drift taxonomy gains RESTART_LEVEL, for variation "
            "that enters at the process launch while elapsed time between sittings "
            "adds nothing -- which is what these readings actually show, and which "
            "carries a different remedy from PERSISTENT: measuring everything in "
            "one sitting does not help. Comparability gained an internal null "
            "derived from the data's own replicate splits, at the same probe count "
            "and replicate count, and a third gate on the median directed ratio, "
            "which has no multiplicity problem and catches the systematic shift a "
            "maximum over hundreds of noisy probes cannot see."
        ),
        lesson=(
            "A threshold calibrated on one design does not transfer to another. "
            "Both errors here are the same error wearing different clothes: a "
            "number that means something at six probes and seven repeats was "
            "carried over to six probes and four sessions, and to 276 probes and "
            "two repeats, without asking what it meant there. Where the design "
            "changes, take the null from the data."
        ),
    )
    p = history.record(art, "2026-08-31-unsound-drift-classification", w, files)
    print(f"recorded {p}")
    problems = history.verify(art)
    print("verify:", problems or "all preserved digests match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
