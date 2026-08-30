# Withdrawn: Thrust I compute-model calibration (run of 2026-08-29 18:20)

**Reason** &mdash; vacuous-result (the result could not have come out any other way)  
**Withdrawn** 2026-08-30T02:51:08Z  
**Originally published at** `40d8a04`  
**Superseded by** `calibration-thrust1-compute @ 0d6d57b`

> The files in this directory are preserved exactly as published, including the parts that were wrong. They are not corrected in place, because correcting them would destroy the evidence that the error happened and was caught. Nothing here is a live claim.

## What was wrong

This report concludes **USABLE FOR RANKING: no false wins, and a typical error of 0.0 percentage points**, with a loss-region overlap of 1.00 and 100% verdict agreement. Every one of those numbers is vacuous.

Neither the predicted map nor the measured map contained a single loss. The intersection over union of two empty sets is 1.00; with no misclassified cells the error is necessarily 0.0; with no boundary in either map the boundary statistics are undefined. The predicted effects also had no spread, so no slope could be fitted and the model was in effect checked at a single operating point.

The study could not have detected a disagreement, and reported that as agreement.

## How it was found

From the shape of the numbers rather than their values. A calibration that reports *exactly* 1.00 overlap and *exactly* 0.0 percentage points of error has almost certainly divided by something empty, and `Cohen's kappa: nan` printed two lines above was the confirmation: kappa is undefined when one category never occurs.

## What changed

`CalibrationResult.degenerate` now detects both forms &mdash; no losses in either map, or no spread in the predicted effects &mdash; and the verdict leads with **ESTABLISHES NOTHING**, naming which condition applied. The statistics are still printed, under an explicit warning not to read them as agreement.

A stronger version of the same rule now gates the study before it runs: calibration data must contain at least one predicted or observed loss, or no overlap statistics are computed at all.

## What it taught

This is requirement LC-1.8 &mdash; *an empty result from a design that could not have produced a non-empty one is a property of the design, not evidence* &mdash; turned on the project's own calibration machinery. The standard already contained the rule that would have caught it; it had simply never been applied to anything except loss columns.

## Preserved files

| File | Digest at withdrawal |
|---|---|
| `calibration-thrust1-compute.json` | `sha256:eb10c35dd0fec310...` |
| `calibration-thrust1-compute.md` | `sha256:5de5f042c91eda09...` |