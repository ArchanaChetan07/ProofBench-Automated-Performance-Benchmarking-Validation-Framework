### Communication coverage and readiness campaign

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). 46 message sizes, 2 independent passes, 2 repeats per point.

#### Peak reproducibility

| Group | pass 0 peak | pass 1 peak | magnitude ratio | ordering | reproducible |
|---|---|---|---|---|---|
| all_gather/world2 | 32768 KiB (large) | 121 KiB (small) | 1.37x | stable | **no** |
| all_gather/world3 | 512 KiB (medium) | 403 KiB (medium) | 1.03x | stable | yes |
| all_gather/world4 | 512 KiB (medium) | 512 KiB (medium) | 1.05x | stable | yes |
| all_reduce/world2 | 601 KiB (medium) | 403 KiB (medium) | 1.27x | CHANGED | **no** |
| all_reduce/world3 | 1337 KiB (medium) | 1337 KiB (medium) | 1.06x | stable | yes |
| all_reduce/world4 | 1603 KiB (medium) | 1994 KiB (medium) | 1.02x | stable | yes |

- **all_gather/world2**: NOT reproducible: the peak moved from large to small; its magnitude differs by 1.37x. The peak must not be reported as a physical property on this evidence.
- **all_gather/world3**: the peak lands in the medium regime in both passes, within 1.03x in magnitude and with the same regime ordering: a property of the transport, not of one run.
- **all_gather/world4**: the peak lands in the medium regime in both passes, within 1.05x in magnitude and with the same regime ordering: a property of the transport, not of one run.
- **all_reduce/world2**: NOT reproducible: its magnitude differs by 1.27x; the regime ordering changed between passes. The peak must not be reported as a physical property on this evidence.
- **all_reduce/world3**: the peak lands in the medium regime in both passes, within 1.06x in magnitude and with the same regime ordering: a property of the transport, not of one run.
- **all_reduce/world4**: the peak lands in the medium regime in both passes, within 1.02x in magnitude and with the same regime ordering: a property of the transport, not of one run.

#### Coverage matrix

**Subsystem readiness: NOT READY**

Parameter validity: 2 of 6 groups have an accepted model. Model coverage: 6 of 24 (group x regime) cells (25%). These are different numbers and neither implies readiness on its own.

| Collective | World | tiny | small | medium | large | Model | Verdict | Covered |
|---|---|---|---|---|---|---|---|---|
| all_reduce | 2 | _model_rejected_ | _model_rejected_ | _model_rejected_ | _model_rejected_ | — | diagnostic | **no** |
| all_reduce | 3 | _model_rejected_ | _model_rejected_ | _model_rejected_ | _model_rejected_ | — | diagnostic | **no** |
| all_reduce | 4 | _model_rejected_ | _model_rejected_ | _model_rejected_ | _model_rejected_ | — | diagnostic | **no** |
| all_gather | 2 | _model_rejected_ | _model_rejected_ | _model_rejected_ | _model_rejected_ | — | diagnostic | **no** |
| all_gather | 3 | 4% | 8% | 14% | 4% | `piecewise` | accepted | yes |
| all_gather | 4 | 6% | _error_too_high_ | _error_too_high_ | 12% | `linear` | accepted | **no** |

**Blocking gates**

- 5 of 6 required groups are not covered: all_reduce/world2 (insufficient_coverage), all_reduce/world3 (insufficient_coverage), all_reduce/world4 (insufficient_coverage), all_gather/world2 (insufficient_coverage), all_gather/world4 (insufficient_coverage)
- 4 group(s) have no accepted model: all_reduce/world2, all_reduce/world3, all_reduce/world4, all_gather/world2
- 18 of 24 (group x regime) cells are uncovered: error_too_high x2, model_rejected x16

- 24 calibration sizes and 22 validation sizes, alternating within each regime so neither half is missing a regime.