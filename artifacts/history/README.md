# Withdrawn artifacts and discarded runs

Nothing in this directory is a claim. Nothing here carries a conformance grade,
and nothing here should be cited as a result.

Three results produced during this project were wrong, and all three were wrong
in ways the project exists to talk about. Deleting them would remove the only
direct evidence that the discipline does what it claims, and would leave the
repository looking as though nothing had ever gone wrong — which is exactly the
impression a loss column is supposed to prevent.

## Two rules

**A withdrawn artifact is never edited.** Its content is what it was when it was
published, including the parts that were wrong. Correcting it in place would
destroy the thing that makes it evidence. Every preserved file is hashed at
withdrawal and re-checked by `losscolumn history verify`, so a silent edit is
detectable.

**A withdrawal is a separate document.** The reason, the superseding artifact,
and what the error taught live in `withdrawal.json` beside the artifact, never
inside it. That is how retraction works in publishing, and for the same reason:
the record of an error and the error itself are different documents, with
different authors and different dates.

## What is here

| Directory | What it holds |
|---|---|
| `2026-08-29-unfair-baseline/` | A Thrust III claim reporting **zero losses across 80 cells**. The sweep and the statistics were correct; the comparator was not. |
| `2026-08-29-vacuous-calibration/` | A calibration reporting **1.00 region overlap and 0.0pp error** — two empty sets agreeing. |
| `incidents/` | Runs discarded before publication. No artifact is preserved, because none should be citable. |

## Why these two, specifically

Both are the project's own rules turned on the project's own work.

The unfair baseline is the **untuned-baseline** failure that requirement LC-2
exists to foreclose, committed by the person writing the standard against it —
and caught because LC-1.8 makes an empty loss column suspicious rather than
satisfying.

The vacuous calibration is **LC-1.8 itself** — *an empty result from a design
that could not have produced a non-empty one is a property of the design, not
evidence* — applied to the calibration machinery instead of to a loss column.
The rule that would have caught it already existed. It had simply never been
pointed at anything except a loss column.
