### Coverage debt

**Coverage debt: 6 uncovered cell(s) of 24.**

| Kind | Cells | Remedy |
|---|---|---|
| `group_model_rejected` | 4 | nothing local to this cell. Either a family that fits the group's worst regime, or a decision about what the worst-regime rule should do when a regime is too noisy to grade at all -- which is a question about the gate, to be settled deliberately rather than by widening it |
| `model_limited` | 1 | a model family that can express this shape, or an admission that the surface is not modellable at this granularity. More repeats will not help: the measurement is already clean |
| `bimodal` | 1 | nothing, at this granularity. The remedy is a change to what is being claimed: model which branch the transport takes and fit each separately, or state the operating surface as excluding the unstable band. Both change the claim rather than improving the measurement, so both are decisions rather than work |

| Group | Regime | Kind | held-out | noise | gap | why |
|---|---|---|---|---|---|---|
| all_gather/world2 | medium | `bimodal` | 19.2% | 21.1% | +4.2% | an algorithm-selection threshold sits in this regime (band 124268-1641112 bytes, median step 4.2x at 276500B). The transport has two behaviours here and no single-valued cost model can be right about both |
| all_reduce/world2 | large | `group_model_rejected` | 5.1% | 19.1% | -9.9% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 9.3% |
| all_reduce/world2 | medium | `group_model_rejected` | 37.9% | 26.3% | +22.9% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 12.9% |
| all_reduce/world2 | small | `group_model_rejected` | 20.2% | 11.6% | +5.2% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 5.7% |
| all_reduce/world2 | tiny | `group_model_rejected` | 18.4% | 12.9% | +3.4% | the group has no accepted model, blocked by its medium regime (26.3% run-to-run variation); this cell's own measurement is 6.3% |
| all_reduce/world3 | medium | `model_limited` | 19.3% | 20.4% | +4.3% | the points are known to 10.0%, against which a perfect model would show about 6.7%, and the best available family misses by 19.3%: the gap is the model's |

**The single most discriminating next experiment**

all_reduce/world3, medium regime: measurement noise is 20.4% while the best model misses by 19.3%. The gap is the model's, not the instrument's, so the discriminating experiment is to add message sizes in that regime and re-fit: if a richer segmentation then clears the gate, the surface has more structure than the current families express; if it does not, the surface is not modellable at this granularity and that is the finding.

- Recorded CVs come from 21 repeats per point, and are used as measured. No correction is applied: an earlier ledger corrected a two-repeat CV by a single factor and a pre-registered re-measurement falsified it.
- Computed from `campaign-v2.json`, whose CVs come from 21 repeats per point in a single session and are used exactly as measured.
- This supersedes a ledger that read 9 model-limited and 7 noise-limited. That one corrected the original campaign's two-repeat CVs by a single factor of 1.84x and moved six cells to noise-limited on the strength of it. A pre-registered re-measurement falsified the correction: it predicted the recorded CV would rise to 10.7% and it rose to 19.0%. The ledger and the correction are preserved at artifacts/history/2026-08-31-two-sample-cv-correction.
- No correction replaces it, because none can. Within the re-measurement's own session the excess splits into 2.25x from the estimator and a further 1.35x from timescale: twenty-one repeats span more wall-clock than two adjacent ones and see slower variation. The quantity a correction is meant to recover therefore grows with the window it is measured over, so the factor depends on an arbitrary reference -- 1.84x against seven repeats, 2.25x against twenty-one.
- The practical consequence is that every coverage figure this project produced before this measurement rested on a noise estimate roughly three times too small. Median cell CV moved from 5.7% to 18.4% and the number of cells above the 20% ceiling from 0 to 10. Coverage fell from 8 cells to 6, which is a correction rather than a regression: the earlier figure was inflated by an instrument that under-reported its own variation.