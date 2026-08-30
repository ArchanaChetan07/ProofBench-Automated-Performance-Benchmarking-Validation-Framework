### Calibration study -- Thrust I, compute half

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). The model was asked for its prediction before anything was measured, and was not adjusted afterwards.

| Model parameter | Assumed | Measured | Ratio |
|---|---|---|---|
| Achievable MFU | 0.480 | 1.406 | 2.93x |
| Checkpointing penalty | 1.33x | 1.22x | 0.92x |

### Calibration -- predicted against measured

**USABLE FOR RANKING: no false wins, and a typical error of 0.0 percentage points, inside the 5% MDE.**

- 4 of 4 cells measured and compared.
- Verdict agreement 100% (Cohen's kappa nan).
- Loss-region overlap (IoU) 1.00: 0 predicted, 0 measured.
- **0 false win(s)** -- predicted a win, measured a loss.
- 0 false loss(es) -- predicted a loss, measurement cleared it.

| Calibration statistic | Value | Reading |
|---|---|---|
| Bias | +0.00 pp | model is pessimistic |
| Mean absolute error | 0.00 pp | typical miss |
| RMSE | 0.00 pp | miss including the tails |
| Slope | nan | scales correctly |
| R-squared | nan | share of variation the model tracks |
| Within MDE | 100% | cells predicted to within 5% |

**Boundary displacement**

- seq_len: no fiber has a boundary in both maps
- checkpointing: no fiber has a boundary in both maps

- the predicted effects have no spread, so no slope can be fitted: the model is being checked at a single operating point

**Not calibrated by this study**

- Scale. Calibrated over micro batches [1, 2, 4] and sequence lengths [512, 1024] at hidden=1024, which is what a development card can measure in a usable time. The registered Thrust I grid is wider in every direction, and a slope fitted here is not evidence about the corners this grid does not reach.
- Collective time. Bus bandwidth, the per-collective latency floor and the bandwidth-against-message-size curve are the model's other half, and none of them is measurable on a single device. Thrust I's loss regions are driven by exposed communication, so the parameters that matter most to its published conclusion remain unchecked by this study.
- Whether a real runtime achieves the overlap schedule the model assumes. That is the question Thrust I exists to ask and it needs a cluster.
- Scaling across world size. The registered grid spans 8 and 32 GPUs; this study touches one.
- Model scale. Calibrated at hidden=1024, not the 7B-class hidden=4096 the protocol registers. What transfers is the residual of a model of this form, not the parameter value.