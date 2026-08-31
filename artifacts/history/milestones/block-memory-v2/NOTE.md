# Milestone: Thrust I memory model v2, validated on an untouched 10-cell grid

**Tag** `v0.3.0-memory-v2` &middot; **commit** `f802694` &middot; recorded 2026-08-31T02:44:00Z

> Preserved unedited. The files here are hash-locked and re-checked by `losscolumn history verify`; a milestone nobody can quietly edit is the only kind that can still be cited later.

## Result

| Metric | Value |
|---|---|
| Infeasible-region IoU | 1.00 |
| Boundary error | 0 levels on all 3 axes |
| False wins | 0/10 |
| False losses | 0/10 |
| Dangerous-error score (w=10) | 0.00 |
| Accuracy | 100% |

## What this establishes

- On this device, at this block shape, block-memory-v2 made no dangerous error across ten configurations, six of which lay within 25% of its own predicted memory boundary.
- The component decomposition survives contact with hardware: checkpointing applied to the saved-activation term alone reproduces the measured feasibility boundary exactly, where a global multiplier did not.
- No parameter was fitted. Every value is registered from what a block allocates, so the validation had no fitted quantity to flatter.

## What it does NOT establish

Carried with the result on purpose: a narrow validation whose narrowness gets separated from it becomes a claim it never supported.

- Generalisation to another GPU. One card was available: an NVIDIA T1000, 8 GB, sm_75, with no tensor cores. Allocator behaviour, context cost and fragmentation all differ across devices and drivers.
- Generalisation to another memory capacity. Every cell was measured against one 7.9 GB budget, so the model's behaviour as the budget moves is untested.
- Generalisation to the simulated architecture. Calibrated on one block at hidden=1024; the target is 7B-class across 32 layers, and the ScalingMap records which parameters are claimed to transfer and which are not.
- Statistical strength. Ten cells is a small grid. Zero false wins in ten is consistent with a true false-win rate up to roughly 26% at 95% confidence; it is evidence of no gross error, not of a small one.
- Anything about communication. The Thrust I communication model is entirely uncalibrated, and Thrust I's published loss regions turn on it.

## Frozen parameters

Asserted by test. Any change is a new model version, not an edit to this one.

| Parameter | Value |
|---|---|
| `activation_tensors_per_block` | 4 |
| `allocator_reserve_fraction` | 0.12 |
| `attention_workspace_tensors` | 4 |
| `autograd_overhead_tensors` | 2 |
| `checkpoint_retained_fraction` | 0.25 |
| `context_bytes` | 7.5e+08 |
| `ffn_workspace_tensors` | 2 |
| `gradient_bytes_per_param` | 2 |
| `optimizer_bytes_per_param` | 0 |
| `safety_margin_fraction` | 0.1 |

## Preserved files

| File | Digest |
|---|---|
| `calibration-thrust1-memory-v2.json` | `sha256:22ffb1989d581408...` |
| `calibration-thrust1-memory-v2.md` | `sha256:e005313ee154e27d...` |
| `memory-model-v2.protocol.json` | `sha256:7bfca2b2b0442068...` |