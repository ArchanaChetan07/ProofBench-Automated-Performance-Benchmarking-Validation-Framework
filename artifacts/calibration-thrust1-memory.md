### Calibration study -- Thrust I, memory feasibility

Measured on NVIDIA T1000 8GB (sm_75, 9 GB), budget 7.9 GB. The model was asked for its prediction before anything was measured, and was not adjusted afterwards.

The question is one the model can get wrong at a cost: **does the recommended micro batch fit?** A false win here is a run that dies.

### Calibration -- predicted against measured

**REGION STATISTICS ONLY: 5 false win(s) and 0 false loss(es) over 8 cells, overlap 0.17. The magnitude statistics are absent by construction: this is a feasibility study, so the prediction is a classification and there is no effect size to regress. EVERY false win is a configuration the model cleared and the hardware refused.**

- 8 of 8 cells measured and compared.
- Verdict agreement 38% (Cohen's kappa 0.09).
- Loss-region overlap (IoU) 0.17: 1 predicted, 6 measured.
- **5 false win(s)** -- predicted a win, measured a loss.
- 0 false loss(es) -- predicted a loss, measurement cleared it.

_No magnitude statistics: the prediction here is a classification, so there is no effect size to regress. The region statistics above are the whole of what this study measures._

**Boundary displacement**

- seq_len: the model puts the boundary 1.00 level(s) too late on average (|shift| 1.00 over 1 fiber(s))
- micro_batch: the model puts the boundary 1.00 level(s) too late on average (|shift| 1.00 over 1 fiber(s))
- checkpointing: the model puts the boundary 0.00 level(s) too late on average (|shift| 0.00 over 1 fiber(s))


**Feasibility agreement**: the model is right about 3 of 8 configurations.

| Sequence | Micro batch | Checkpointing | Model | Actual | Predicted GB |
|---|---|---|---|---|---|
| 2048 | 128 | False | fits | OOM | 6.44 |
| 2048 | 128 | True | fits | OOM | 3.63 |
| 4096 | 64 | False | fits | OOM | 6.44 |
| 4096 | 64 | True | fits | OOM | 3.63 |
| 4096 | 128 | True | fits | OOM | 6.18 |

**Not calibrated by this study**

- Communication. Bus bandwidth, the per-collective latency floor and the overlap schedule are the model's other half and need a cluster; Thrust I's loss regions are driven by exposed communication, so the parameters that matter most to its published conclusion are still unchecked.
- Model scale. Calibrated at hidden=1024 on one block, not a 7B-class model across 32 layers. What transfers is the residual of a memory model of this form, not the parameter values.
- Optimizer state. The measured runs hold gradients but no optimizer moments, while the model prices twelve bytes per parameter for them. That term is therefore predicted and not measured here.