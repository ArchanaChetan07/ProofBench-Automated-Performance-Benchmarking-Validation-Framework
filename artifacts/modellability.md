# MODELLABLE_WITH_WORK

Every uncovered cell is measurable to the 15% gate. 3 of them need a model family that fits; the rest need replicates.

## The noise budget

Measurement variance arrives at two levels and they behave differently under more work. One averages down; the other does not.

| Component | Value | Behaviour under more repeats |
|---|---|---|
| within a run | 21.3% | falls as 1/sqrt(n) |
| across restarts, observed | 15.9% | n/a |
| across restarts, predicted from repeats alone | 10.1% | n/a |
| **launch-level component** | **12.3%** | **does not fall at all inside one process** |

Measured from 7 repeats per launch and 3 launches per probe. The launch component is what is left of the restart-level dispersion once averaging is accounted for; variances subtract, not the coefficients.

**The floor is 12.3%.** A campaign that adds repeats forever converges there and stops.

## What a target precision costs

| Target | Repeats | Launches | Reachable |
|---|---|---|---|
| 20% | 3 | 1 | yes |
| 15% | 10 | 1 | yes |
| 10% | 15 | 3 | yes |

Repeats first because they are much the cheaper of the two: a launch pays process startup and transport setup once for every probe it carries.

- **20%**: 3 repeats in one launch reaches 20%; the launch-level floor is 12.3%, below the target, so no extra launches are required
- **15%**: 10 repeats in one launch reaches 15%; the launch-level floor is 12.3%, below the target, so no extra launches are required
- **10%**: repeats alone converge on 12.3%, above the 10% target: the launch-level component does not average down inside one process, so this needs about 3 launches per point as well

## Cell by cell

| Group | Regime | held-out | noise | Verdict | Why |
|---|---|---|---|---|---|
| all_gather/world2 | small | 20.2% | 7.4% | `MODEL_WORK` | the measurement is clean (7.4%) and the best family still misses by 20.2%: a real modelling gap |
| all_reduce/world4 | tiny | 14.4% | 8.3% | `MODEL_WORK` | the measurement is clean (8.3%) and the best family still misses by 14.4%: a real modelling gap |
| all_reduce/world4 | medium | 22.3% | 11.2% | `MODEL_WORK` | the measurement is clean (11.2%) and the best family still misses by 22.3%: a real modelling gap |
| all_gather/world2 | tiny | 9.6% | 10.4% | `BUYABLE` | the residual (9.6%) is the instrument's; 3 repeats per point reaches the gate |
| all_gather/world2 | medium | 19.7% | 19.3% | `BUYABLE` | the residual (19.7%) is the instrument's; 9 repeats per point reaches the gate |
| all_gather/world2 | large | 5.3% | 10.7% | `BUYABLE` | the residual (5.3%) is the instrument's; 3 repeats per point reaches the gate |
| all_reduce/world2 | tiny | 7.1% | 11.2% | `BUYABLE` | the residual (7.1%) is the instrument's; 3 repeats per point reaches the gate |
| all_reduce/world2 | small | 8.6% | 8.1% | `BUYABLE` | the residual (8.6%) is the instrument's; 2 repeats per point reaches the gate |
| all_reduce/world2 | medium | 35.3% | 29.5% | `BUYABLE` | the residual (35.3%) is the instrument's; 19 repeats per point reaches the gate |
| all_reduce/world2 | large | 14.9% | 22.1% | `BUYABLE` | the residual (14.9%) is the instrument's; 11 repeats per point reaches the gate |
| all_reduce/world3 | tiny | 2.8% | 11.5% | `BUYABLE` | the residual (2.8%) is the instrument's; 3 repeats per point reaches the gate |
| all_reduce/world3 | small | 4.3% | 14.3% | `BUYABLE` | the residual (4.3%) is the instrument's; 5 repeats per point reaches the gate |
| all_reduce/world3 | medium | 35.2% | 26.0% | `BUYABLE` | the residual (35.2%) is the instrument's; 15 repeats per point reaches the gate |
| all_reduce/world3 | large | 5.7% | 16.4% | `BUYABLE` | the residual (5.7%) is the instrument's; 6 repeats per point reaches the gate |
| all_reduce/world4 | small | 7.6% | 7.4% | `BUYABLE` | the residual (7.6%) is the instrument's; 2 repeats per point reaches the gate |
| all_reduce/world4 | large | 5.2% | 13.3% | `BUYABLE` | the residual (5.2%) is the instrument's; 4 repeats per point reaches the gate |

## The determination

**13** buyable, **3** model work.

The question was never whether six of six groups could be made to pass. It is whether the operating surface this subsystem needs is one that can be modelled from this machine, and the honest answer separates three things that a single coverage percentage hides: cells where more measurement would help, cells where a better model would help, and cells where neither would because the instrument cannot resolve the error being chased.