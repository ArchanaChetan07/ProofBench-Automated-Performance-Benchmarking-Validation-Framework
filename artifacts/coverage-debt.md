### Coverage debt

**Coverage debt: 7 uncovered cell(s) of 24.**

| Kind | Cells | Remedy |
|---|---|---|
| `group_model_rejected` | 4 | nothing local to this cell. Either a family that fits the group's worst regime, or a decision about what the worst-regime rule should do when a regime is too noisy to grade at all -- which is a question about the gate, to be settled deliberately rather than by widening it |
| `bimodal` | 2 | nothing, at this granularity. The remedy is a change to what is being claimed: model which branch the transport takes and fit each separately, or state the operating surface as excluding the unstable band. Both change the claim rather than improving the measurement, so both are decisions rather than work |
| `noise_limited` | 1 | more repeats, or a quieter machine. No modelling work will help: the residual is already below the measurement's own variation |

| Group | Regime | Kind | held-out | noise | gap | why |
|---|---|---|---|---|---|---|
| all_gather/world2 | medium | `bimodal` | 19.6% | 21.1% | +4.6% | an algorithm-selection threshold sits here |
| all_gather/world2 | small | `bimodal` | 16.6% | 13.8% | +1.6% | an algorithm-selection threshold sits here |
| all_reduce/world2 | large | `group_model_rejected` | 4.3% | 19.1% | -10.7% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 19.1% |
| all_reduce/world2 | medium | `group_model_rejected` | 34.2% | 26.3% | +19.2% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 26.3% |
| all_reduce/world2 | small | `group_model_rejected` | 18.8% | 11.6% | +3.8% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 11.6% |
| all_reduce/world2 | tiny | `group_model_rejected` | 18.7% | 12.9% | +3.7% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 12.9% |
| all_reduce/world3 | medium | `noise_limited` | 19.3% | 20.4% | +4.3% | run-to-run variation 20.4% exceeds the 20% the protocol allows; nothing can be graded here |

**The single most discriminating next experiment**

all_reduce/world3, medium regime: the residual (19.3%) is at or below the measurement's own variation (20.4%). No model can do better than the instrument, so the discriminating experiment is more repeats there -- if the noise falls and the error follows, it was noise; if the error stays, it was never noise-limited.

- Recorded CVs come from 21 repeats per point, and are used as measured. No correction is applied: an earlier ledger corrected a two-repeat CV by a single factor and a pre-registered re-measurement falsified it.
- Computed from `campaign-v2.json`, whose CVs come from 21 repeats per point in a single session and are used exactly as measured.
- This supersedes a ledger that read 9 model-limited and 7 noise-limited. That one corrected the original campaign's two-repeat CVs by a single factor of 1.84x and moved six cells to noise-limited on the strength of it. A pre-registered re-measurement falsified the correction: it predicted the recorded CV would rise to 10.7% and it rose to 19.0%. The ledger and the correction are preserved at artifacts/history/2026-08-31-two-sample-cv-correction.
- No correction replaces it, because none can. Within the re-measurement's own session the excess splits into 2.25x from the estimator and a further 1.35x from timescale: twenty-one repeats span more wall-clock than two adjacent ones and see slower variation. The quantity a correction is meant to recover therefore grows with the window it is measured over, so the factor depends on an arbitrary reference -- 1.84x against seven repeats, 2.25x against twenty-one.
- The practical consequence is that every coverage figure this project produced before this measurement rested on a noise estimate roughly three times too small. Median cell CV moved from 5.7% to 18.4% and the number of cells above the 20% ceiling from 0 to 10. Coverage fell from 8 cells to 6, which is a correction rather than a regression: the earlier figure was inflated by an instrument that under-reported its own variation.