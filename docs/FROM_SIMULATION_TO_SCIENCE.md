# From simulation to science

Two of this project's three thrusts are simulated, because the hardware they
need — an eight-GPU node for Thrust I, three inference engines under load for
Thrust II — is not available here. That is a real limitation, and the honest
response is neither to hide it nor to treat the simulated results as though
they were measurements.

This document states what a simulated thrust is claiming, and the path by which
it stops being a simulation.

---

## The claim a simulated thrust actually makes

A measured claim says:

> Across this envelope, the method loses here, by this much.

A simulated claim says something weaker and more conditional:

> **If** this model of the hardware is right, then across this envelope the
> method loses here, by this much.

Everything downstream inherits that conditional. The loss regions are real
consequences of the model; whether they are real consequences of the *hardware*
is exactly what has not been established. The standard makes this
non-negotiable rather than a matter of authorial care: `LC-0.2` requires every
claim to declare its evidence class, `LC-0.3` requires a simulated claim to say
what was modelled and what was measured, and the renderer puts a banner at the
top of the page that a reader cannot miss.

A simulated claim is a **prediction**. It is worth exactly as much as the
track record of the model that made it — which, before any calibration, is
nothing.

---

## The pipeline

```
   Simulated result
         │            a model, its parameters, and a loss map derived from them
         ▼
   Prediction
         │            the same loss map, stated as a falsifiable forecast about
         │            specific cells, registered before anyone measures them
         ▼
   Measured experiment
         │            a subset of those cells, measured under a sealed protocol
         ▼
   Calibration comparison
         │            predicted loss map scored against measured loss map:
         │            region overlap, boundary error, false wins, false losses,
         │            cliff agreement, calibration error
         ▼
   Model update
                      a correction, applied in a NEW sealed protocol and
                      validated on cells the calibration study did not touch
```

The last arrow is the one most easily got wrong. A model retuned against the
measurements used to score it has been **fitted**, not calibrated, and its
apparent accuracy afterwards is circular. `suggest_update()` therefore computes
the correction and deliberately refuses to apply it, recording why in the
artifact.

---

## What each simulated thrust assumes

### Thrust I — the overlap envelope

**Model.** A closed-form step-time model. Per layer:

- compute time = `FLOPs / (peak_tflops × achievable_mfu)`, plus a fixed
  per-kernel launch cost
- collective time from message size, bus bandwidth, and a per-collective
  latency floor
- a scheduling assumption about how much collective time hides behind compute

**Assumptions, in the order they are likely to be wrong:**

1. **The overlap schedule.** The model assumes a particular degree of
   compute/communication overlap. Whether a real FSDP implementation achieves
   it is precisely the question Thrust I exists to ask, so assuming it is close
   to assuming the conclusion. This is the model's weakest point.
2. **Bus bandwidth as a function of message size.** Real NCCL bandwidth is a
   curve with a latency floor, a ramp, and a plateau; the model uses a
   simplified form.
3. **`achievable_mfu` as a single constant.** Real MFU varies with shape.
4. **A flat `4/3` multiplier for activation checkpointing** — independent of
   sequence length or batch. Measurement (below) shows this is the wrong
   *shape*, not merely a mis-sized constant.

**Uncertainty.** The model is deterministic: it has no sampling noise, and
running paired statistics over its output would manufacture confidence
intervals from numbers that never varied. `deterministic_comparisons()`
therefore assigns verdicts by thresholding the predicted effect and sets the
interval to the point itself. The uncertainty in a simulated claim is
*parameter* uncertainty, not sampling uncertainty, and the two must not be
displayed in the same notation.

**Calibration procedure.** `losscolumn calibrate thrust1` restricts Thrust I's
own question — *is the recommended configuration beaten by another in the swept
set?* — to the axes a single GPU can vary: micro batch and activation
checkpointing. The model is asked for its prediction before anything is
measured. Then real transformer blocks are timed, forward and backward, and the
predicted loss map is scored against the measured one.

**What that calibrates, and what it cannot.** It calibrates the compute half
completely. It does not touch the communication half at all — and Thrust I's
loss regions are *driven* by exposed communication. So the calibration bounds
the model's error on the parameters that matter least to its published
conclusion. That is worth saying plainly rather than burying: this study makes
Thrust I better founded, not founded.

### Thrust II — the equal-tuning audit

**Model.** A serving simulator: a roofline on decode step time from weight
bytes over memory bandwidth, a queueing model for batching, and per-engine
overhead constants.

**Assumptions:**

1. Per-engine overhead constants are the whole of the difference between
   engines. Real engines differ in scheduling policy, kernel coverage and
   memory management in ways a constant cannot represent.
2. The batching model is an approximation of continuous batching.
3. Tuning response — how much each engine improves per trial — is modelled,
   and it is the quantity the thrust's *conclusion* is about.

The third is the serious one. Thrust II asks how much published engine
difference survives tuning-budget parity; simulating the tuning response means
the answer is partly assumed. The parity *machinery* — ledgers, certificates,
the `LC-2` rules — is real and mechanically checkable, and it is the part
intended to be reused. The numbers are illustrations of what that machinery
produces.

**Calibration path.** Not attempted here: it needs the engines installed and a
GPU each. The procedure would mirror Thrust I's — restrict to one engine and
one workload, measure the default-to-tuned gap over a small trial budget, and
score the predicted gap against it.

---

## Simulated provenance

A simulated result still has provenance, and it is recorded in the same shape
as a measurement's:

| Field | Meaning for a simulated run |
|---|---|
| `evidence_class` | `simulated`, and rendered as a banner |
| `evidence_note` | what was modelled, what was measured |
| `model_parameters` | every constant, with its source |
| `seed` | the draw, where the simulator is stochastic |
| `hardware` | the machine that *would* be needed, not the one that ran it |
| `commit`, `container_image` | as for a measurement |

The `hardware` row is the one that most invites carelessness. A simulated
Thrust I result must not record the laptop that produced it as though the
numbers came from there, and must not record an eight-GPU node as though one
had been used. It records the target configuration and the evidence class
together, so the pair is unambiguous.

---

## When a simulated thrust becomes a measured one

The transition is not a code change. It is:

1. **Seal a new protocol.** The measured version registers the same
   hypotheses against real hardware. The simulated protocol is archived and
   stays verifiable.
2. **Run the measurement**, unchanged in analysis code — the harness is
   backend-agnostic by construction, which is the point of the adapter layer.
3. **Score the old prediction against the new measurement.** The simulated
   loss map was published and hashed before the measurement existed, so this is
   a genuine out-of-sample test rather than a retrospective fit.
4. **Publish both**, with the calibration comparison between them.

Step 3 is what makes the earlier simulation worth having done. A prediction
that was recorded, sealed, and later checked is evidence about the model.
A prediction quietly revised after the fact is not evidence about anything.
