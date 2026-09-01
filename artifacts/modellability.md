# NOISE_FLOOR_NOT_WELL_DEFINED

The launch-level noise floor computes to 0.0% from one observation window and 6.3% from another, on the same machine. That is not a discrepancy to be averaged: splitting variance into a part that averages down and a part that does not presumes the noise is stationary, and this machine's is not. A within-run estimate over seven repeats attributes to the launch what one over twenty-one absorbs into itself.

So no single floor can be quoted, and any statement of the form 'repeats converge on X%' is false precision. What can be said is measured directly and without a decomposition: the standard error of a median of n falls as n^-0.31, so more repeats do buy precision, and buy it more slowly than the usual assumption predicts.

## The noise budget

Measurement variance arrives at two levels and they behave differently under more work. One averages down; the other does not.

| Component | Value | Behaviour under more repeats |
|---|---|---|
| within a run | 19.0% | falls as n^-0.31, measured |
| across restarts, observed | 4.4% | n/a |
| across restarts, predicted from repeats alone | 9.3% | n/a |
| **launch-level component** | **0.0%** | **does not fall at all inside one process** |

Measured from 21 repeats per launch and 2 launches per probe. The launch component is what is left of the restart-level dispersion once averaging is accounted for; variances subtract, not the coefficients.

**There is no single floor.** This window puts it at 0.0%; the sentinel's seven-repeat window puts it at 6.3%. The decomposition is not stable, so neither number should be quoted as the machine's.

## What a target precision costs

| Target | Repeats | Launches | Reachable |
|---|---|---|---|
| 20% | 2 | 1 | yes |
| 15% | 5 | 1 | yes |
| 10% | 17 | 1 | yes |

Repeats first because they are much the cheaper of the two: a launch pays process startup and transport setup once for every probe it carries.

- **20%**: 2 repeats in one launch reaches 20%; the launch-level floor is 0.0%, below the target, so no extra launches are required
- **15%**: 5 repeats in one launch reaches 15%; the launch-level floor is 0.0%, below the target, so no extra launches are required
- **10%**: 17 repeats in one launch reaches 10%; the launch-level floor is 0.0%, below the target, so no extra launches are required

## Cell by cell

| Group | Regime | held-out | noise | Verdict | Why |
|---|---|---|---|---|---|
| all_gather/world3 | tiny | 17.5% | 17.7% | `BUYABLE` | the residual (17.5%) is the instrument's; 4 repeats per point reaches the gate |
| all_gather/world3 | small | 17.4% | 17.9% | `BUYABLE` | the residual (17.4%) is the instrument's; 4 repeats per point reaches the gate |
| all_gather/world3 | medium | 13.7% | 21.4% | `BUYABLE` | the residual (13.7%) is the instrument's; 7 repeats per point reaches the gate |
| all_reduce/world4 | tiny | 11.3% | 20.5% | `BUYABLE` | the residual (11.3%) is the instrument's; 6 repeats per point reaches the gate |
| all_reduce/world4 | medium | 9.8% | 22.6% | `BUYABLE` | the residual (9.8%) is the instrument's; 8 repeats per point reaches the gate |
| all_reduce/world4 | large | 5.0% | 23.6% | `BUYABLE` | the residual (5.0%) is the instrument's; 10 repeats per point reaches the gate |
| all_gather/world2 | tiny | 1.9% | 8.8% | `GROUP_BLOCKED` | this cell measures at 8.8% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 8.8% |
| all_gather/world2 | small | 2.7% | 4.9% | `GROUP_BLOCKED` | this cell measures at 4.9% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 4.9% |
| all_gather/world2 | medium | 30.6% | 30.6% | `GROUP_BLOCKED` | this cell measures at 30.6% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 30.6% |
| all_gather/world2 | large | 5.8% | 19.7% | `GROUP_BLOCKED` | this cell measures at 19.7% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 19.7% |
| all_reduce/world2 | tiny | 3.9% | 3.9% | `GROUP_BLOCKED` | this cell measures at 3.9% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 3.9% |
| all_reduce/world2 | small | 4.7% | 3.9% | `GROUP_BLOCKED` | this cell measures at 3.9% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 3.9% |
| all_reduce/world2 | medium | 37.4% | 30.3% | `GROUP_BLOCKED` | this cell measures at 30.3% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 30.3% |
| all_reduce/world2 | large | 5.6% | 18.7% | `GROUP_BLOCKED` | this cell measures at 18.7% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 18.7% |
| all_reduce/world3 | tiny | 23.7% | 42.3% | `EXPENSIVE` | reachable, at 60 repeats across 1 launch(es) per point, for a cell whose run-to-run variation is 42.3% |
| all_reduce/world3 | small | 8.8% | 32.3% | `GROUP_BLOCKED` | this cell measures at 32.3% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its tiny regime (42.3% run-to-run variation); this cell's own measurement is 32.3% |
| all_reduce/world3 | medium | 21.2% | 42.3% | `EXPENSIVE` | reachable, at 60 repeats across 1 launch(es) per point, for a cell whose run-to-run variation is 42.3% |
| all_reduce/world3 | large | 11.8% | 27.9% | `GROUP_BLOCKED` | this cell measures at 27.9% and could be graded; it is uncovered because no model was accepted for its group. the group has no accepted model, blocked by its tiny regime (42.3% run-to-run variation); this cell's own measurement is 27.9% |

## The determination

**6** buyable, **2** expensive, **10** group blocked.

The question was never whether six of six groups could be made to pass. It is whether the operating surface this subsystem needs is one that can be modelled from this machine, and the honest answer separates three things that a single coverage percentage hides: cells where more measurement would help, cells where a better model would help, and cells where neither would because the instrument cannot resolve the error being chased.