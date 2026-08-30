### Calibration study -- Thrust I, compute half

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). The model was asked for its prediction before anything was measured, and was not adjusted afterwards.

| Model parameter | Assumed | Measured | Ratio |
|---|---|---|---|
| Achievable MFU | 0.480 | 1.340 | 2.79x |
| Checkpointing penalty | 1.33x | 1.22x | 0.92x |

### Calibration -- predicted against measured

**ESTABLISHES NOTHING: this study could not have detected a disagreement. neither the predicted nor the measured map contains a loss, so the region overlap of 1.00 is two empty sets agreeing and the boundary statistics are undefined; the predicted effects have no spread, so no slope can be fitted and the model was checked at a single operating point. A wider grid, or one chosen so the model predicts a loss somewhere, is needed before any statement about the model's reliability is supported.**

> The statistics below are reported for completeness and should not be read as agreement. A study that could not have found a disagreement has not found their absence.

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

- The MFU reference is invalid on this device. A single large fp16 GEMM measured 0.34 TFLOP/s while the layer GEMMs it is supposed to bound reached more, giving a nominal utilisation of 1.34 -- above one, which is impossible. The cause is the absent tensor cores: the large fp16 GEMM falls on an emulated path that the smaller layer shapes avoid. No MFU calibration is available here, and the figure below is retained only to show the contradiction.
- Scale. Calibrated over micro batches [1, 2, 4] and sequence lengths [512, 1024] at hidden=1024, which is what a development card can measure in a usable time. The registered Thrust I grid is wider in every direction, and a slope fitted here is not evidence about the corners this grid does not reach.
- Collective time. Bus bandwidth, the per-collective latency floor and the bandwidth-against-message-size curve are the model's other half, and none of them is measurable on a single device. Thrust I's loss regions are driven by exposed communication, so the parameters that matter most to its published conclusion remain unchecked by this study.
- Whether a real runtime achieves the overlap schedule the model assumes. That is the question Thrust I exists to ask and it needs a cluster.
- Scaling across world size. The registered grid spans 8 and 32 GPUs; this study touches one.
- Model scale. Calibrated at hidden=1024, not the 7B-class hidden=4096 the protocol registers. What transfers is the residual of a model of this form, not the parameter value.