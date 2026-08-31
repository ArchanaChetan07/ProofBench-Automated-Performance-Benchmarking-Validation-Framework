### block-memory-v2, expanded validation

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). **The model is frozen**: nothing here is fitted, and a disagreement would be recorded rather than corrected.

Nine sequence lengths, 11 micro batches, two precisions and three effective capacities (35%, 60%, 92% of 8.6 GB).

**SAFE BUT CONSERVATIVE: no false wins; 10 configuration(s) the model rejected would in fact have run (36% of its rejections).**

| | measured feasible | measured infeasible |
|---|---|---|
| **predicted feasible** | 32 correct | **0 FALSE WIN** |
| **predicted infeasible** | 10 false loss | 18 correct |

| Metric | Value |
|---|---|
| Accuracy | 83% |
| False-win count | **0** of 60 |
| False-win rate (of cleared) | 0% |
| False-win area (of grid) | 0% |
| False-loss count | 10 of 60 |
| False-loss rate (of rejected) | 36% |
| False-loss area (of grid) | 17% |
| Infeasible-region IoU | 0.64 |
| Dangerous-error score (w=10) | 0.17 |

**Boundary displacement** &mdash; positive means the model holds on too long, the direction that produces false wins.

| Axis | Mean signed shift | Max |shift| | Fibers |
|---|---|---|---|
| `micro_batch` | +0.00 level(s) | 0 | 18 |
| `seq_len` | +0.00 level(s) | 0 | 13 |
| `checkpointing` | +0.00 level(s) | 0 | 18 |
| `dtype` | +0.00 level(s) | 0 | 13 |
| `budget_fraction` | +0.00 level(s) | 0 | 11 |


#### By effective capacity

| Budget | cells | false wins | false losses | accuracy |
|---|---|---|---|---|
| 35% (3.0 GB) | 20 | **0** | 1 | 95% |
| 60% (5.2 GB) | 20 | **0** | 8 | 60% |
| 92% (7.9 GB) | 20 | **0** | 1 | 95% |

#### By precision

| dtype | cells | false wins | false losses | accuracy |
|---|---|---|---|---|
| `float16` | 30 | **0** | 6 | 80% |
| `float32` | 30 | **0** | 4 | 87% |

#### Notes

- block-memory-v2 was not modified. Its parameters are the values validated at v0.3.0-memory-v2 and are asserted by test.