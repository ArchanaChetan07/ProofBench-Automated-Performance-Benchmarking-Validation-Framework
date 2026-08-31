### Targeted probe: is the medium regime model-limited or evidence-thin?

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). 16 additional medium-regime sizes, all_reduce only, worlds [2, 3, 4]. Nothing else was measured.

**INCONCLUSIVE -- THE SESSIONS ARE NOT COMPARABLE. the two sessions differ by 1.42x in median bandwidth over the same size range (0.747 against 1.063 GB/s), beyond the 1.15x this protocol allows. Pooling them would describe the drift rather than the transport, and adding points would make the best achievable fit worse -- which is impossible for a consistent surface. The hypothesis is untested: this measurement cannot distinguish a model-limited surface from a drifting machine.**

| | prior session | this session | ratio | allowed |
|---|---|---|---|---|
| median bandwidth over the shared range | 0.747 | 1.063 | **1.42x** | 1.15x |

| Group | medium points before | after | error before | error after | segments after |
|---|---|---|---|---|---|
| all_reduce/world2 | 30 | 32 | 38.5% | 34.4% | 2 |
| all_reduce/world3 | 30 | 32 | 19.6% | 7.2% | 2 |
| all_reduce/world4 | 30 | 32 | 13.5% | 10.1% | 2 |

- The probe's own method was unsound: it pooled records across two measurement sessions without checking they were comparable. The check now exists, and it fails, which is how this was found.
- The signature was in the result and not recognised at first: adding 32 points made all_reduce/world4's best achievable medium error RISE from 13.5% to 32.5%. That is impossible on a consistent surface.
- The remedy is to measure the campaign grid and the denser medium grid in ONE session. Nothing here changes any campaign verdict.
- The first verdict is preserved unedited under artifacts/history.