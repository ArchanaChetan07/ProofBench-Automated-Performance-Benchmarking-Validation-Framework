# Withdrawn: Machine stability, first analysis: drift called WORKLOAD_SPECIFIC

**Reason** &mdash; unsound-statistic (the measurement was sound and the statistic applied to it was not)  
**Withdrawn** 2026-09-01T01:25:07Z  
**Originally published at** `unrecorded`  
**Superseded by** `stability-machine@2026-08-31-corrected (artifacts/stability-machine.json, drift RESTART_LEVEL)`

> The files in this directory are preserved exactly as published, including the parts that were wrong. They are not corrected in place, because correcting them would destroy the evidence that the error happened and was caught. Nothing here is a live claim.

## What was wrong

The drift was classified WORKLOAD_SPECIFIC because the session-level variability differed 6.9x between the six probes, against a registered threshold of 2x. Each of those six figures is a coefficient of variation estimated from four session medians. A permutation test on the same readings -- shuffling which session label each launch wears, which is the null in which every probe behaves identically -- puts the median spread at 6.4x and reproduces the observed 6.9x forty percent of the time. The threshold was measuring the sample size, not the machine.

A second, independent error sat in the comparability test the same analysis used. The criterion has the form 1 + 3 sigma, which bounds ONE comparison of medians-of-seven repeats. Applied to the campaign it was being asked to bound the worst of 276 comparisons of medians-of-two, where a perfectly stable machine breaches it almost surely. That error pointed the wrong way: it declared the campaign's two passes INCOMPARABLE at 2.90x when splitting the campaign's own repeats within a single pass -- the same machine state by construction -- reaches 4.91x. The passes are more alike than the instrument is with itself.

## How it was found

The first error, by not believing a large ratio without testing it. 6.9x is a striking number and the honest question was how often chance produces one; the permutation test cost no measurement time and the answer was 40%.

The second, by checking the test before reporting its verdict. Declaring the campaign incomparable would have put every figure derived from it in question, so the statistic behind that verdict was worth one look -- and it was a worst-of-276 judged against a worst-of-6 threshold.

## What changed

Workload-specificity now requires a permutation test at alpha = 0.05, not a ratio. The drift taxonomy gains RESTART_LEVEL, for variation that enters at the process launch while elapsed time between sittings adds nothing -- which is what these readings actually show, and which carries a different remedy from PERSISTENT: measuring everything in one sitting does not help. Comparability gained an internal null derived from the data's own replicate splits, at the same probe count and replicate count, and a third gate on the median directed ratio, which has no multiplicity problem and catches the systematic shift a maximum over hundreds of noisy probes cannot see.

## What it taught

A threshold calibrated on one design does not transfer to another. Both errors here are the same error wearing different clothes: a number that means something at six probes and seven repeats was carried over to six probes and four sessions, and to 276 probes and two repeats, without asking what it meant there. Where the design changes, take the null from the data.

## Preserved files

| File | Digest at withdrawal |
|---|---|
| `original-run.log` | `sha256:a60be94f2069a32c...` |
| `stability-sentinel-v1.protocol.json` | `sha256:495d8bd8aa56ddc9...` |