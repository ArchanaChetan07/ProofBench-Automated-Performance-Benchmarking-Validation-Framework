### 8x A100 decision gate

# NOT_READY_FOR_8X_A100

Every requirement is checked from an artifact on disk, so the gate cannot pass on an assumption.

| Gate | Result | Detail |
|---|---|---|
| communication quality gates pass | **FAIL** | 2 of 6 groups have an accepted model |
| required coverage achieved | **FAIL** | 8 of 24 (group x regime) cells covered (33%) |
| no rejected parameter is active | PASS | no group is counted as covered without an accepted model |
| validation is independent | PASS | 264 held-out validation points on sizes no fit saw; calibration and validation sizes alternate within each regime |
| fresh-clone reproducibility passes | PASS | fresh clone, isolated environment, full test suite and frozen standard verified; measured thrusts NOT re-run (no torch in the clean environment), so this establishes the standard and the analysis layer, not the measurements |
| 8-GPU protocol is fully prepared | PASS | 12 stages sealed (6.8 h), seal 16a8b46786fc9366 |

**Blocking: communication quality gates pass; required coverage achieved**

### 8x A100 campaign protocol

**PREPARED, NOT EXECUTED** &mdash; seal `16a8b46786fc9366...`, about 6.8 hours of allocation.

Local communication readiness at sealing: **not_ready**. That verdict gates whether this should be executed; it does not make the protocol wrong, it makes running it premature.

| # | Stage | Produces | Abort if | min |
|---|---|---|---|---|
| 1 | environment smoke test | a pass/fail record and the environment fingerprint | the suite does not pass on the target machine | 20 |
| 2 | topology verification | a topology map recorded in the artifact | the topology differs from the one the protocol assumes, in which case the grid of world sizes is re-derived and the change is recorded as a declared deviation before measurement | 15 |
| 3 | device and allocator verification | ProbeConfig with cap_installed true, per device | the cap cannot be installed on any device | 15 |
| 4 | fabric identification | a transport label per (world size, rank placement) | a group's transport cannot be identified | 30 |
| 5 | communication calibration | raw PointRecords, per collective per world per pass | peak reproducibility fails, which is reported rather than worked around | 90 |
| 6 | parameter quality gates | a ParameterVerdict per group and a coverage matrix | no group is accepted, which ends the campaign honestly rather than loosening the gate | 10 |
| 7 | model freeze | a sealed Fabric with parameter provenance | any rejected parameter is reachable as a model input | 5 |
| 8 | Thrust I prediction generation | a predicted loss map and its regions | the memory or communication model is not frozen | 15 |
| 9 | prediction lock | a sealed prediction hash | the prediction is not sealed before measurement begins | 5 |
| 10 | independent 8-GPU measurement | the measured envelope | the measurement shares hardware with another job | 180 |
| 11 | measured loss-region extraction | the measured loss map | the design check reports the sweep could not have declared a loss | 10 |
| 12 | prediction against measurement | the calibration report and the final Thrust I claim | the prediction hash does not match what was sealed at stage 9 | 10 |

**Ordering is binding.** Stages 8 and 9 run before stage 10: a prediction produced after seeing the measurement is not a prediction.

**No local parameter is used here.** The local campaign ran gloo over shared memory on one host. It cannot produce NVLink or InfiniBand numbers and none of its fits is copied into an A100 fabric. What transfers is the method.
