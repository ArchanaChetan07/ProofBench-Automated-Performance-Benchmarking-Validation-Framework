### Thrust I memory model, version 2

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). Protocol seal `9e8525992c8f3477...`, grid split `e5623786935e6c21...`.

#### Calibration grid (the grid that diagnosed v1; not evidence about v2)

**EXACT: the model agrees with the hardware on all 8 cells.**

| | measured feasible | measured infeasible |
|---|---|---|
| **predicted feasible** | 2 correct | **0 FALSE WIN** |
| **predicted infeasible** | 0 false loss | 6 correct |

| Metric | Value |
|---|---|
| Accuracy | 100% |
| False-win count | **0** of 8 |
| False-win rate (of cleared) | 0% |
| False-win area (of grid) | 0% |
| False-loss count | 0 of 8 |
| False-loss rate (of rejected) | 0% |
| False-loss area (of grid) | 0% |
| Infeasible-region IoU | 1.00 |
| Dangerous-error score (w=10) | 0.00 |

**Boundary displacement** &mdash; positive means the model holds on too long, the direction that produces false wins.

| Axis | Mean signed shift | Max |shift| | Fibers |
|---|---|---|---|
| `micro_batch` | +0.00 level(s) | 0 | 4 |
| `seq_len` | +0.00 level(s) | 0 | 4 |
| `checkpointing` | +0.00 level(s) | 0 | 3 |


#### Validation grid (untouched until the model was frozen)

**EXACT: the model agrees with the hardware on all 10 cells.**

| | measured feasible | measured infeasible |
|---|---|---|
| **predicted feasible** | 4 correct | **0 FALSE WIN** |
| **predicted infeasible** | 0 false loss | 6 correct |

| Metric | Value |
|---|---|
| Accuracy | 100% |
| False-win count | **0** of 10 |
| False-win rate (of cleared) | 0% |
| False-win area (of grid) | 0% |
| False-loss count | 0 of 10 |
| False-loss rate (of rejected) | 0% |
| False-loss area (of grid) | 0% |
| Infeasible-region IoU | 1.00 |
| Dangerous-error score (w=10) | 0.00 |

**Boundary displacement** &mdash; positive means the model holds on too long, the direction that produces false wins.

| Axis | Mean signed shift | Max |shift| | Fibers |
|---|---|---|---|
| `micro_batch` | +0.00 level(s) | 0 | 6 |
| `seq_len` | +0.00 level(s) | 0 | 6 |
| `checkpointing` | +0.00 level(s) | 0 | 5 |


#### Old model against new

| Metric | v1 (diagnosis grid) | v2 (validation grid) |
|---|---|---|
| Infeasible-region IoU | 0.17 | 1.00 |
| Boundary error | 1 level late | **0 levels** on all 3 axes |
| False wins | 5/8 | **0/10** |
| False losses | 0/8 | 0/10 |
| Dangerous-error score (w=10) | 6.25 | 0.00 |
| Accuracy | 38% | 100% |

The two columns are **not** measured on the same cells and cannot be: the v1 numbers come from the grid that diagnosed it, which is this study's calibration grid and therefore inadmissible as evidence about v2. Each column is that model's result on the hardest grid available to it, not a paired contest.

#### Notes

- No parameter was fitted. Version 2's parameters are all REGISTERED, derived from what a block allocates rather than from these measurements, so the calibration grid confirms the replacement without conferring any fit on it. That makes the validation result a stronger statement, not a weaker one: the model had no opportunity to absorb the calibration data.