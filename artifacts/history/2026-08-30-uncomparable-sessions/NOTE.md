# Withdrawn: Medium-regime probe, first verdict (2026-08-30 22:33)

**Reason** &mdash; invalid-measurement (the measurement itself was not valid)  
**Withdrawn** 2026-08-31T05:35:12Z  
**Originally published at** `a1668eb`  
**Superseded by** `probe-thrust1-medium-regime, re-reported with a session check`

> The files in this directory are preserved exactly as published, including the parts that were wrong. They are not corrected in place, because correcting them would destroy the evidence that the error happened and was caught. Nothing here is a live claim.

## What was wrong

The probe reported 'PARTIAL: 1 of 3 groups clear the gate', concluding that thin evidence was part of the medium-regime problem. The conclusion rests on pooling the probe's records with the campaign's, and the two sessions are not comparable: over the same size range the probe measured 1.2x to 2.7x faster, a median drift of 1.42x.

Pooling them describes the drift rather than the transport. The signature was visible in the result and not recognised at first: for all_reduce at world 4 the best achievable medium-regime error ROSE from 13.5% to 32.5% when 32 points were added. Adding data cannot make the best achievable fit worse on a consistent surface; it can only do so when the added data describes a different machine state.

## How it was found

By disbelieving an impossible number. More points made the fit worse, which has no explanation on a single surface, so the next question was whether there was one surface. Comparing median bandwidth over the overlapping size range between the two sessions answered it immediately.

## What changed

session_comparability() now compares two sessions over their overlapping size range before any pooling, and the probe refuses to pool beyond a 1.15x drift. With the check active the probe reports INCONCLUSIVE rather than PARTIAL: the hypothesis is untested, not answered.

The remedy for the underlying question is to measure the campaign grid and the denser medium grid in ONE session, so the comparison happens within a single machine state.

## What it taught

The probe's own method was unsound in a way the campaign's was not: the campaign compared two passes taken back to back and used them only to test reproducibility, never pooling them into one fit. The probe pooled across sessions hours apart. A measurement protocol needs to state not only what it measures but which measurements may be combined, and that this was missing was invisible until a number came out impossible.

## Preserved files

| File | Digest at withdrawal |
|---|---|
| `json` | `sha256:a84ae4a9f3641a72...` |
| `md` | `sha256:95537d48ac88e949...` |