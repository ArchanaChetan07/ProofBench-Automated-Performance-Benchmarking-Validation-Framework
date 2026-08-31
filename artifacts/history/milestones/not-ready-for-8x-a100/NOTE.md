# Milestone: Communication subsystem: NOT_READY_FOR_8X_A100

**Tag** `v0.7.0-not-ready` &middot; **commit** `a1668eb` &middot; recorded 2026-08-31T05:36:00Z

> Preserved unedited. The files here are hash-locked and re-checked by `losscolumn history verify`; a milestone nobody can quietly edit is the only kind that can still be cited later.

## Result

| Metric | Value |
|---|---|
| Verdict | NOT_READY_FOR_8X_A100 |
| Parameter validity | 2 of 6 groups accepted |
| Model coverage | 8 of 24 cells (33%) |
| Blocking gates | communication quality gates pass, required coverage achieved |
| Coverage debt | 16 cells: {'noise_limited': 7, 'model_limited': 9} |

## What this establishes

- The communication subsystem is not ready to be measured at scale, and the two blocking gates are named: model quality and operating-surface coverage.
- Acceptance and coverage are separate and the separation is load-bearing. all_gather/world4 has an accepted model and, before the parsimony fix, was not covered -- two of its four regimes exceeded the error threshold.
- The coverage debt is classified by what is actually blocking each cell: 9 model-limited, 7 noise-limited. Those have different remedies and the report says which.
- The bandwidth peak is a property of the transport in three groups -- medium regime in both passes, within 1.06x. In three others there is no well-defined peak at all.
- The 8x A100 protocol is sealed and executable: 12 stages, 6.8 hours, with prediction generation and lock before measurement.

## What it does NOT establish

Carried with the result on purpose: a narrow validation whose narrowness gets separated from it becomes a claim it never supported.

- Any NVLink or InfiniBand parameter. This is gloo over shared memory on one host, and nothing measured here is copied into an A100 fabric.
- Whether the medium regime is model-limited or evidence-thin. The probe that would have settled it came back INCONCLUSIVE: its two sessions differed by 1.42x and could not be pooled.
- That the local machine is stable enough for this work. Two sessions hours apart differed by up to 2.7x in throughput on the same sizes, which is the most important unresolved fact about this measurement environment.
- Anything about Thrust I's loss regions, which need the communication model this campaign declined to certify.

## Preserved files

| File | Digest |
|---|---|
| `a100-campaign-v1.protocol.json` | `sha256:ea50a4b7b66a09d9...` |
| `campaign-thrust1-communication.md` | `sha256:5a3aff60b93b4836...` |
| `coverage-debt.json` | `sha256:48086d14c90d1602...` |
| `coverage-debt.md` | `sha256:aae77d7f57b04970...` |
| `readiness-gate.json` | `sha256:aa17c39a85bc95f0...` |