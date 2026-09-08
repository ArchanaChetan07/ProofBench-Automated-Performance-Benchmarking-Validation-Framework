# Campaign v2

Predictions sealed `sha256:92163026f289523d` before the run. One session, 441 minutes, 21 repeats per point per pass -- the same replicate count as the run it is compared against, so any movement is the corrections and not the effort.

| Prediction | Baseline | Predicted | Measured | Result |
|---|---|---|---|---|
| P1 medium-regime CV falls a quarter | 27.4% | < 20.5% | 18.6% | **HELD** |
| P2 the 1 MiB noise cliff closes | 1.57x | < 1.25x | 1.33x | **FALSIFIED** |
| P3 coverage improves | 6 | > 6 | 17 | **HELD** |

Coverage **17 of 24** cells; **5 of 6** groups have an accepted model.

Timed blocks averaged 5-200 calls (median 45), chosen per point from its own measured variability rather than from its size.

**P2 is falsified.** The gap across 1 MiB survives a rule that does not know where 1 MiB is, so it is a property of the transport rather than of the measurement.
