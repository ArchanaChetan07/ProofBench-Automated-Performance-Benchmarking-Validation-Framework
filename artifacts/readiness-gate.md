### 8x A100 decision gate

# NOT_READY_FOR_8X_A100

Every requirement is checked from an artifact on disk, so the gate cannot pass on an assumption.

| Gate | Result | Detail |
|---|---|---|
| evidence is not pooled across incomparable sessions | PASS | all evidence comes from a single session, so no cross-session pooling occurred and none had to be justified. For context, the machine's own drift is random (0 of 6 sentinel session pairs poolable at the registered 1.48x criterion; within-run 21.3%, across restart 15.9% against 14.6% expected from averaging) |
| the campaign's own evidence is internally poolable | PASS | its two passes agree to 2.90x against a 4.91x floor measured from the campaign's own repeats at 276 probes, and the surface did not move (0.998x median shift) |
| communication quality gates pass | **FAIL** | 5 of 6 groups have an accepted model |
| required coverage achieved | **FAIL** | 17 of 24 (group x regime) cells covered (71%) |
| no rejected parameter is active | PASS | no group is counted as covered without an accepted model |
| validation is independent | PASS | 264 held-out validation points on sizes no fit saw; calibration and validation sizes alternate within each regime |
| fresh-clone reproducibility passes | PASS | fresh clone, isolated environment, full test suite and frozen standard verified; measured thrusts NOT re-run (no torch in the clean environment), so this establishes the standard and the analysis layer, not the measurements |
| 8-GPU protocol is fully prepared | PASS | 13 stages sealed (7.2 h), seal c4f2344cf638807e |

**Blocking: communication quality gates pass; required coverage achieved**

### 8x A100 campaign protocol

**PREPARED, NOT EXECUTED** &mdash; seal `c4f2344cf638807e...`, about 7.2 hours of allocation.

Local communication readiness at sealing: **not_ready**. That verdict gates whether this should be executed; it does not make the protocol wrong, it makes running it premature.

| # | Stage | Produces | Abort if | min |
|---|---|---|---|---|
| 1 | environment smoke test | a pass/fail record and the environment fingerprint | the suite does not pass on the target machine | 20 |
| 2 | topology verification | a topology map recorded in the artifact | the topology differs from the one the protocol assumes, in which case the grid of world sizes is re-derived and the change is recorded as a declared deviation before measurement | 15 |
| 3 | machine stability sentinel | a stability envelope, a registered comparability criterion, and the replicate count the calibration stage needs | the launch-level noise floor exceeds the coverage gate, in which case the campaign cannot grade a model to the precision it requires and the allocation is released rather than spent proving it | 25 |
| 4 | device and allocator verification | ProbeConfig with cap_installed true, per device | the cap cannot be installed on any device | 15 |
| 5 | fabric identification | a transport label per (world size, rank placement) | a group's transport cannot be identified | 30 |
| 6 | communication calibration | raw PointRecords, per collective per world per pass | peak reproducibility fails, which is reported rather than worked around | 90 |
| 7 | parameter quality gates | a ParameterVerdict per group and a coverage matrix | no group is accepted, which ends the campaign honestly rather than loosening the gate | 10 |
| 8 | model freeze | a sealed Fabric with parameter provenance | any rejected parameter is reachable as a model input | 5 |
| 9 | Thrust I prediction generation | a predicted loss map and its regions | the memory or communication model is not frozen | 15 |
| 10 | prediction lock | a sealed prediction hash | the prediction is not sealed before measurement begins | 5 |
| 11 | independent 8-GPU measurement | the measured envelope | the measurement shares hardware with another job | 180 |
| 12 | measured loss-region extraction | the measured loss map | the design check reports the sweep could not have declared a loss | 10 |
| 13 | prediction against measurement | the calibration report and the final Thrust I claim | the prediction hash does not match what was sealed at stage 10 | 10 |

**Ordering is binding.** Stages 9 and 10 run before stage 11: a prediction produced after seeing the measurement is not a prediction.

**No local parameter is used here.** The local campaign ran gloo over shared memory on one host. It cannot produce NVLink or InfiniBand numbers and none of its fits is copied into an A100 fabric. What transfers is the method.
