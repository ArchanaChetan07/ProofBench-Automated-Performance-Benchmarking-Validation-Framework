### Coverage debt

**Coverage debt: 16 uncovered cell(s) of 24.**

| Kind | Cells | Remedy |
|---|---|---|
| `noise_limited` | 13 | more repeats, or a quieter machine. No modelling work will help: the residual is already below the measurement's own variation |
| `model_limited` | 3 | a model family that can express this shape, or an admission that the surface is not modellable at this granularity. More repeats will not help: the measurement is already clean |

| Group | Regime | Kind | held-out | noise | gap | why |
|---|---|---|---|---|---|---|
| all_gather/world2 | large | `noise_limited` | 5.3% | 10.7% | -9.7% | the best model misses by 5.3% against run-to-run variation of 10.7%; the residual is the instrument's, not the model's |
| all_gather/world2 | medium | `noise_limited` | 19.7% | 19.3% | +4.7% | the best model misses by 19.7% against run-to-run variation of 19.3%; the residual is the instrument's, not the model's |
| all_gather/world2 | small | `model_limited` | 20.2% | 7.4% | +5.2% | the measurement is clean (7.4%) and the best available family still misses by 20.2%: the gap is the model's |
| all_gather/world2 | tiny | `noise_limited` | 9.6% | 10.4% | -5.4% | the best model misses by 9.6% against run-to-run variation of 10.4%; the residual is the instrument's, not the model's |
| all_reduce/world2 | large | `noise_limited` | 14.9% | 22.1% | -0.1% | run-to-run variation 22.1% exceeds the 20% the protocol allows; nothing can be graded here |
| all_reduce/world2 | medium | `noise_limited` | 35.3% | 29.5% | +20.3% | run-to-run variation 29.5% exceeds the 20% the protocol allows; nothing can be graded here |
| all_reduce/world2 | small | `noise_limited` | 8.6% | 8.1% | -6.4% | the best model misses by 8.6% against run-to-run variation of 8.1%; the residual is the instrument's, not the model's |
| all_reduce/world2 | tiny | `noise_limited` | 7.1% | 11.2% | -7.9% | the best model misses by 7.1% against run-to-run variation of 11.2%; the residual is the instrument's, not the model's |
| all_reduce/world3 | large | `noise_limited` | 5.7% | 16.4% | -9.3% | the best model misses by 5.7% against run-to-run variation of 16.4%; the residual is the instrument's, not the model's |
| all_reduce/world3 | medium | `noise_limited` | 35.2% | 26.0% | +20.2% | run-to-run variation 26.0% exceeds the 20% the protocol allows; nothing can be graded here |
| all_reduce/world3 | small | `noise_limited` | 4.3% | 14.3% | -10.7% | the best model misses by 4.3% against run-to-run variation of 14.3%; the residual is the instrument's, not the model's |
| all_reduce/world3 | tiny | `noise_limited` | 2.8% | 11.5% | -12.2% | the best model misses by 2.8% against run-to-run variation of 11.5%; the residual is the instrument's, not the model's |
| all_reduce/world4 | large | `noise_limited` | 5.2% | 13.3% | -9.8% | the best model misses by 5.2% against run-to-run variation of 13.3%; the residual is the instrument's, not the model's |
| all_reduce/world4 | medium | `model_limited` | 22.3% | 11.2% | +7.3% | the measurement is clean (11.2%) and the best available family still misses by 22.3%: the gap is the model's |
| all_reduce/world4 | small | `noise_limited` | 7.6% | 7.4% | -7.4% | the best model misses by 7.6% against run-to-run variation of 7.4%; the residual is the instrument's, not the model's |
| all_reduce/world4 | tiny | `model_limited` | 14.4% | 8.3% | -0.6% | the measurement is clean (8.3%) and the best available family still misses by 14.4%: the gap is the model's |

**The single most discriminating next experiment**

all_gather/world2, small regime: measurement noise is 7.4% while the best model misses by 20.2%. The gap is the model's, not the instrument's, so the discriminating experiment is to add message sizes in that regime and re-fit: if a richer segmentation then clears the gate, the surface has more structure than the current families express; if it does not, the surface is not modellable at this granularity and that is the finding.

- Recorded CVs come from 2 repeats and are corrected by 1.83x before classification. Uncorrected they read as a clean measurement, which is the signature of a model-limited cell; several cells change kind under the correction.
- This supersedes an earlier ledger that read 9 model-limited and 7 noise-limited. Nothing was re-measured: the earlier one took each cell's recorded CV at face value, and every one of those was computed from two repeats. Correcting that estimator moved six cells, all of them from model-limited to noise-limited, and three of the four medium-regime cells among them.
- The correction inverts the plan the earlier ledger implied. Most of the supposed modelling work was never modelling work: the residuals it pointed at are smaller than the instrument's own variation, and no model family can beat the instrument. The remaining model-limited cells are worth attention precisely because there are only three of them.
- It also dissolves the medium-regime puzzle without appealing to session drift. all_reduce/world2 and world3 show 26% to 30% run-to-run variation there against a 20% ceiling, so nothing can be graded in those cells at all -- which is why a probe that added 32 points to that regime produced a worse fit rather than a better one.