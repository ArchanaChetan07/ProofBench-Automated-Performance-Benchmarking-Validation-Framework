# Campaign at higher replicates

Predictions sealed `sha256:be927a46986e1a90` before the measurement ran. One session, 2 minutes, 21 repeats per point per pass against the original 2.

| Prediction | Predicted | Measured | Result |
|---|---|---|---|
| P1 recorded CV rises by the de-bias factor | 10.7% | 12.0% | **HELD** |
| P2 coverage improves | > 8 | 0 | **FALSIFIED** |
| P3 the three model-limited cells stay uncovered | 3 | 3 | **HELD** |

Coverage 0 of 24 cells, from 8.

P1 is the load-bearing one. The recorded CV moved from 5.8% to 12.0% on the same grid and the same machine, and the only thing that changed is how many samples each CV was computed from. That is the two-sample bias, measured directly rather than inferred, and it is what licensed reclassifying six cells from model-limited to noise-limited.

**P2 is falsified.** Coverage did not improve despite the residuals having been attributed to the instrument. They were not the instrument's, and the noise-limited classification is wrong.

Still uncovered among the three predicted: all_gather/world2/small, all_reduce/world4/tiny, all_reduce/world4/medium
