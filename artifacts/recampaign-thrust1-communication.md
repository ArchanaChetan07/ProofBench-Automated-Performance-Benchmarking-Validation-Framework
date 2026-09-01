# Campaign at higher replicates

Predictions sealed `sha256:be927a46986e1a90` before the measurement ran. One session, 75 minutes, 21 repeats per point per pass against the original 2.

| Prediction | Predicted | Measured | Result |
|---|---|---|---|
| P1 recorded CV rises by the de-bias factor | 10.7% | 19.0% | **FALSIFIED** |
| P2 coverage improves | > 8 | 6 | **FALSIFIED** |
| P3 the three model-limited cells stay uncovered | 3 | 3 | **HELD** |

Coverage 6 of 24 cells, from 8.

**P1 is falsified.** The recorded CV went to 19.0% where 10.7% was predicted. The two-sample bias correction does not describe this data, and every cell reclassified under it has to be reclassified back. That correction is withdrawn.

**P2 is falsified.** Coverage did not improve despite the residuals having been attributed to the instrument. They were not the instrument's, and the noise-limited classification is wrong.

Still uncovered among the three predicted: all_gather/world2/small, all_reduce/world4/tiny, all_reduce/world4/medium
