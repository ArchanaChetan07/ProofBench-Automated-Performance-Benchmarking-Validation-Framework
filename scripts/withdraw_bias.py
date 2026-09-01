"""Withdraw the two-sample CV correction and everything derived from it.

The correction predicted, before the measurement ran, that measuring the same
grid with twenty-one repeats would push the recorded CV from 5.8% to about
10.7%. It went to 19.0%. The pre-registration said what a falsification means
and this is it being honoured.

The correction was not wrong in direction -- the CV did rise, substantially --
but a single multiplicative factor cannot describe what is happening, and the
reason is more interesting than the error.
"""
from pathlib import Path


def main() -> int:
    from losscolumn import history

    art = Path("artifacts")
    d = art / "history" / "2026-08-31-two-sample-cv-correction"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("coverage-debt.md", "coverage-debt.json",
                 "modellability.md", "modellability.json"):
        src = art / name
        if src.exists():
            (d / name).write_bytes(src.read_bytes())

    files = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}

    w = history.Withdrawal(
        artifact_id="coverage-debt+modellability@2026-08-31-bias-corrected",
        title="The coverage debt reclassified by a two-sample CV correction",
        reason="unsound-statistic",
        superseded_by="coverage-debt and modellability recomputed from the "
                      "21-repeat recampaign, which measures the dispersion "
                      "directly and needs no correction",
        what_was_wrong=(
            "Every CV in the original campaign was computed from two repeats. "
            "Measured against the stability sentinel's seven, a two-sample CV "
            "understated dispersion by 1.84x, and applying that factor moved six "
            "of sixteen uncovered cells from model-limited to noise-limited.\n\n"
            "The correction assumed the noise is stationary -- that a CV from two "
            "samples and a CV from seven estimate the same quantity, one just "
            "more precisely. It does not. Re-measuring the same grid at "
            "twenty-one repeats in one session puts the recorded CV at 19.0% "
            "against the 10.7% the correction predicted, and the excess "
            "decomposes cleanly within that single session: 2.25x is the "
            "estimator, and a further 1.35x is timescale, because twenty-one "
            "repeats span more wall-clock than two adjacent ones and see slower "
            "variation that adjacent repeats cannot.\n\n"
            "Which is why no single factor works. The 1.84x was measured against "
            "seven repeats and 2.25x against twenty-one; the 'true' CV a "
            "correction is supposed to recover keeps growing with the window it "
            "is measured over, so the correction factor depends on an arbitrary "
            "choice of reference."
        ),
        how_it_was_found=(
            "By registering a falsifiable prediction and honouring it. The "
            "prediction that the recorded CV would rise by the de-bias factor was "
            "sealed before the re-measurement, with an explicit statement that "
            "failure withdraws the correction and everything it reclassified.\n\n"
            "It also failed usefully rather than merely failing: the direction was "
            "right and the magnitude was not, which is what pointed at a second "
            "mechanism rather than at a mistake in arithmetic."
        ),
        what_changed=(
            "The debt and the modellability report are recomputed from the "
            "recampaign's own twenty-one-repeat CVs. A measured dispersion needs "
            "no correction, and it removes the arbitrary reference the factor "
            "depended on.\n\n"
            "The modellability model no longer assumes averaging reduces error as "
            "1/sqrt(n). Measured from 5520 disjoint block pairs inside one "
            "session, the standard error of a median of n falls as n^-0.31, so "
            "averaging is 62% as effective as it would be on white noise and "
            "halving an error costs nine times the repeats rather than four. That "
            "exponent is now measured and carried rather than assumed."
        ),
        lesson=(
            "A noise estimate is not a property of a machine; it is a property of "
            "a machine and an observation window. Two repeats taken back to back "
            "measure how much a call varies against its neighbour, twenty-one "
            "spread across a run measure how much it varies across the run, and "
            "those are different questions with different answers. Every coverage "
            "figure this project produced before today rested on the first while "
            "being read as the second."
        ),
    )
    p = history.record(art, "2026-08-31-two-sample-cv-correction", w, files)
    print(f"recorded {p}")
    print("verify:", history.verify(art) or "all preserved digests match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
