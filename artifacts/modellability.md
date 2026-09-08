# NOISE_FLOOR_NOT_WELL_DEFINED

The launch-level noise floor computes to 0.0% from one observation window and 6.3% from another, on the same machine. That is not a discrepancy to be averaged: splitting variance into a part that averages down and a part that does not presumes the noise is stationary, and this machine's is not. A within-run estimate over seven repeats attributes to the launch what one over twenty-one absorbs into itself.

So no single floor can be quoted, and any statement of the form 'repeats converge on X%' is false precision. What can be said is measured directly and without a decomposition: the standard error of a median of n falls as n^-0.31, so more repeats do buy precision, and buy it more slowly than the usual assumption predicts.

## The noise budget

Measurement variance arrives at two levels and they behave differently under more work. One averages down; the other does not.

| Component | Value | Behaviour under more repeats |
|---|---|---|
| within a run | 15.7% | falls as n^-0.31, measured |
| across restarts, observed | 5.3% | n/a |
| across restarts, predicted from repeats alone | 7.7% | n/a |
| **launch-level component** | **0.0%** | **does not fall at all inside one process** |

Measured from 21 repeats per launch and 2 launches per probe. The launch component is what is left of the restart-level dispersion once averaging is accounted for; variances subtract, not the coefficients.

**There is no single floor.** This window puts it at 0.0%; the sentinel's seven-repeat window puts it at 6.3%. The decomposition is not stable, so neither number should be quoted as the machine's.

## What a target precision costs

| Target | Repeats | Launches | Reachable |
|---|---|---|---|
| 20% | 1 | 1 | yes |
| 15% | 3 | 1 | yes |
| 10% | 10 | 1 | yes |

Repeats first because they are much the cheaper of the two: a launch pays process startup and transport setup once for every probe it carries.

- **20%**: 1 repeats in one launch reaches 20%; the launch-level floor is 0.0%, below the target, so no extra launches are required
- **15%**: 3 repeats in one launch reaches 15%; the launch-level floor is 0.0%, below the target, so no extra launches are required
- **10%**: 10 repeats in one launch reaches 10%; the launch-level floor is 0.0%, below the target, so no extra launches are required

## Cell by cell

| Group | Regime | held-out | noise | Verdict | Why |
|---|---|---|---|---|---|
| all_reduce/world3 | medium | 19.3% | 20.4% | `MODEL_WORK` | the measurement is clean (20.4%) and the best family still misses by 19.3%: a real modelling gap |
| all_gather/world2 | small | 16.6% | 13.8% | `UNMODELLABLE` | an algorithm-selection threshold sits here, so the transport has two behaviours and no single-valued cost model applies. Neither more repeats nor a better family reaches this: an algorithm-selection threshold sits in this regime (band 185364-1641112 bytes, median step 4.2x at 276500B). The transport has two behaviours here and no single-valued cost model can be right about both |
| all_gather/world2 | medium | 19.6% | 21.1% | `UNMODELLABLE` | an algorithm-selection threshold sits here, so the transport has two behaviours and no single-valued cost model applies. Neither more repeats nor a better family reaches this: an algorithm-selection threshold sits in this regime (band 185364-1641112 bytes, median step 4.2x at 276500B). The transport has two behaviours here and no single-valued cost model can be right about both |
| all_reduce/world2 | tiny | 18.7% | 12.9% | `GROUP_BLOCKED` | this cell measures at 12.9% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 6.3% |
| all_reduce/world2 | small | 18.8% | 11.6% | `GROUP_BLOCKED` | this cell measures at 11.6% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 5.7% |
| all_reduce/world2 | medium | 34.2% | 26.3% | `GROUP_BLOCKED` | this cell measures at 26.3% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 12.9% |
| all_reduce/world2 | large | 4.3% | 19.1% | `GROUP_BLOCKED` | this cell measures at 19.1% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 9.3% |

## The determination

**4** group blocked, **1** model work, **2** unmodellable.

The question was never whether six of six groups could be made to pass. It is whether the operating surface this subsystem needs is one that can be modelled from this machine, and the honest answer separates three things that a single coverage percentage hides: cells where more measurement would help, cells where a better model would help, and cells where neither would because the instrument cannot resolve the error being chased.