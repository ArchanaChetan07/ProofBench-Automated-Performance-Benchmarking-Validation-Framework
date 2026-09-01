### Coverage debt

**Coverage debt: 16 uncovered cell(s) of 24.**

| Kind | Cells | Remedy |
|---|---|---|
| `model_limited` | 9 | a model family that can express this shape, or an admission that the surface is not modellable at this granularity. More repeats will not help: the measurement is already clean |
| `noise_limited` | 7 | more repeats, or a quieter machine. No modelling work will help: the residual is already below the measurement's own variation |

| Group | Regime | Kind | held-out | noise | gap | why |
|---|---|---|---|---|---|---|
| all_gather/world2 | large | `noise_limited` | 5.3% | 5.8% | -9.7% | the best model misses by 5.3% against run-to-run variation of 5.8%; the residual is the instrument's, not the model's |
| all_gather/world2 | medium | `model_limited` | 19.7% | 10.5% | +4.7% | the measurement is clean (10.5%) and the best available family still misses by 19.7%: the gap is the model's |
| all_gather/world2 | small | `model_limited` | 20.2% | 4.0% | +5.2% | the measurement is clean (4.0%) and the best available family still misses by 20.2%: the gap is the model's |
| all_gather/world2 | tiny | `model_limited` | 9.6% | 5.7% | -5.4% | the measurement is clean (5.7%) and the best available family still misses by 9.6%: the gap is the model's |
| all_reduce/world2 | large | `noise_limited` | 14.9% | 12.0% | -0.1% | the best model misses by 14.9% against run-to-run variation of 12.0%; the residual is the instrument's, not the model's |
| all_reduce/world2 | medium | `model_limited` | 35.3% | 16.1% | +20.3% | the measurement is clean (16.1%) and the best available family still misses by 35.3%: the gap is the model's |
| all_reduce/world2 | small | `model_limited` | 8.6% | 4.4% | -6.4% | the measurement is clean (4.4%) and the best available family still misses by 8.6%: the gap is the model's |
| all_reduce/world2 | tiny | `noise_limited` | 7.1% | 6.1% | -7.9% | the best model misses by 7.1% against run-to-run variation of 6.1%; the residual is the instrument's, not the model's |
| all_reduce/world3 | large | `noise_limited` | 5.7% | 9.0% | -9.3% | the best model misses by 5.7% against run-to-run variation of 9.0%; the residual is the instrument's, not the model's |
| all_reduce/world3 | medium | `model_limited` | 35.2% | 14.2% | +20.2% | the measurement is clean (14.2%) and the best available family still misses by 35.2%: the gap is the model's |
| all_reduce/world3 | small | `noise_limited` | 4.3% | 7.8% | -10.7% | the best model misses by 4.3% against run-to-run variation of 7.8%; the residual is the instrument's, not the model's |
| all_reduce/world3 | tiny | `noise_limited` | 2.8% | 6.3% | -12.2% | the best model misses by 2.8% against run-to-run variation of 6.3%; the residual is the instrument's, not the model's |
| all_reduce/world4 | large | `noise_limited` | 5.2% | 7.3% | -9.8% | the best model misses by 5.2% against run-to-run variation of 7.3%; the residual is the instrument's, not the model's |
| all_reduce/world4 | medium | `model_limited` | 22.3% | 6.1% | +7.3% | the measurement is clean (6.1%) and the best available family still misses by 22.3%: the gap is the model's |
| all_reduce/world4 | small | `model_limited` | 7.6% | 4.0% | -7.4% | the measurement is clean (4.0%) and the best available family still misses by 7.6%: the gap is the model's |
| all_reduce/world4 | tiny | `model_limited` | 14.4% | 4.5% | -0.6% | the measurement is clean (4.5%) and the best available family still misses by 14.4%: the gap is the model's |

**The single most discriminating next experiment**

all_gather/world2, small regime: measurement noise is 4.0% while the best model misses by 20.2%. The gap is the model's, not the instrument's, so the discriminating experiment is to add message sizes in that regime and re-fit: if a richer segmentation then clears the gate, the surface has more structure than the current families express; if it does not, the surface is not modellable at this granularity and that is the finding.

- The medium regime is model-limited in every all_reduce group. The targeted probe that would have settled whether that is evidence or structure came back INCONCLUSIVE: its two measurement sessions differed by 1.42x and could not be pooled. Re-measuring both grids in one session is the outstanding experiment.