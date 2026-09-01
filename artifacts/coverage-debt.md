### Coverage debt

**Coverage debt: 18 uncovered cell(s) of 24.**

| Kind | Cells | Remedy |
|---|---|---|
| `group_model_rejected` | 12 | nothing local to this cell. Either a family that fits the group's worst regime, or a decision about what the worst-regime rule should do when a regime is too noisy to grade at all -- which is a question about the gate, to be settled deliberately rather than by widening it |
| `noise_limited` | 6 | more repeats, or a quieter machine. No modelling work will help: the residual is already below the measurement's own variation |

| Group | Regime | Kind | held-out | noise | gap | why |
|---|---|---|---|---|---|---|
| all_gather/world2 | large | `group_model_rejected` | 5.8% | 19.7% | -9.2% | the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 19.7% |
| all_gather/world2 | medium | `group_model_rejected` | 30.6% | 30.6% | +15.6% | the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 30.6% |
| all_gather/world2 | small | `group_model_rejected` | 2.7% | 4.9% | -12.3% | the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 4.9% |
| all_gather/world2 | tiny | `group_model_rejected` | 1.9% | 8.8% | -13.1% | the group has no accepted model, blocked by its medium regime (30.6% run-to-run variation); this cell's own measurement is 8.8% |
| all_gather/world3 | medium | `noise_limited` | 13.7% | 21.4% | -1.3% | run-to-run variation 21.4% exceeds the 20% the protocol allows; nothing can be graded here |
| all_gather/world3 | small | `noise_limited` | 17.4% | 17.9% | +2.4% | the best model misses by 17.4% against run-to-run variation of 17.9%; the residual is the instrument's, not the model's |
| all_gather/world3 | tiny | `noise_limited` | 17.5% | 17.7% | +2.5% | the best model misses by 17.5% against run-to-run variation of 17.7%; the residual is the instrument's, not the model's |
| all_reduce/world2 | large | `group_model_rejected` | 5.6% | 18.7% | -9.4% | the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 18.7% |
| all_reduce/world2 | medium | `group_model_rejected` | 37.4% | 30.3% | +22.4% | the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 30.3% |
| all_reduce/world2 | small | `group_model_rejected` | 4.7% | 3.9% | -10.3% | the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 3.9% |
| all_reduce/world2 | tiny | `group_model_rejected` | 3.9% | 3.9% | -11.1% | the group has no accepted model, blocked by its medium regime (30.3% run-to-run variation); this cell's own measurement is 3.9% |
| all_reduce/world3 | large | `group_model_rejected` | 11.8% | 27.9% | -3.2% | the group has no accepted model, blocked by its tiny regime (42.3% run-to-run variation); this cell's own measurement is 27.9% |
| all_reduce/world3 | medium | `group_model_rejected` | 21.2% | 42.3% | +6.2% | the group has no accepted model, blocked by its tiny regime (42.3% run-to-run variation); this cell's own measurement is 42.3% |
| all_reduce/world3 | small | `group_model_rejected` | 8.8% | 32.3% | -6.2% | the group has no accepted model, blocked by its tiny regime (42.3% run-to-run variation); this cell's own measurement is 32.3% |
| all_reduce/world3 | tiny | `group_model_rejected` | 23.7% | 42.3% | +8.7% | the group has no accepted model, blocked by its tiny regime (42.3% run-to-run variation); this cell's own measurement is 42.3% |
| all_reduce/world4 | large | `noise_limited` | 5.0% | 23.6% | -10.0% | run-to-run variation 23.6% exceeds the 20% the protocol allows; nothing can be graded here |
| all_reduce/world4 | medium | `noise_limited` | 9.8% | 22.6% | -5.2% | run-to-run variation 22.6% exceeds the 20% the protocol allows; nothing can be graded here |
| all_reduce/world4 | tiny | `noise_limited` | 11.3% | 20.5% | -3.7% | run-to-run variation 20.5% exceeds the 20% the protocol allows; nothing can be graded here |

**The single most discriminating next experiment**

all_reduce/world4, large regime: the residual (5.0%) is at or below the measurement's own variation (23.6%). No model can do better than the instrument, so the discriminating experiment is more repeats there -- if the noise falls and the error follows, it was noise; if the error stays, it was never noise-limited.

- Recorded CVs come from 21 repeats per point, and are used as measured. No correction is applied: an earlier ledger corrected a two-repeat CV by a single factor and a pre-registered re-measurement falsified it.
- Computed from `recampaign-thrust1-communication.json`, whose CVs come from 21 repeats per point in a single session and are used exactly as measured.
- This supersedes a ledger that read 9 model-limited and 7 noise-limited. That one corrected the original campaign's two-repeat CVs by a single factor of 1.84x and moved six cells to noise-limited on the strength of it. A pre-registered re-measurement falsified the correction: it predicted the recorded CV would rise to 10.7% and it rose to 19.0%. The ledger and the correction are preserved at artifacts/history/2026-08-31-two-sample-cv-correction.
- No correction replaces it, because none can. Within the re-measurement's own session the excess splits into 2.25x from the estimator and a further 1.35x from timescale: twenty-one repeats span more wall-clock than two adjacent ones and see slower variation. The quantity a correction is meant to recover therefore grows with the window it is measured over, so the factor depends on an arbitrary reference -- 1.84x against seven repeats, 2.25x against twenty-one.
- The practical consequence is that every coverage figure this project produced before this measurement rested on a noise estimate roughly three times too small. Median cell CV moved from 5.7% to 18.4% and the number of cells above the 20% ceiling from 0 to 10. Coverage fell from 8 cells to 6, which is a correction rather than a regression: the earlier figure was inflated by an instrument that under-reported its own variation.