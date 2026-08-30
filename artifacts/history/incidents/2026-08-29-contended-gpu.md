# Incident: Two benchmark processes sharing one GPU

**Occurred** 2026-08-29T10:16:50-07:00 / 10:22:22-07:00  
**Artifacts published** none

> No artifact is preserved here, because none should be citable. What is recorded is that the run happened, why its numbers were discarded, and what changed so it cannot happen silently again.

## What happened

A Thrust III sweep was launched at 10:16. A Thrust I calibration study was launched at 10:22, while the sweep was still running. Both processes held CUDA contexts on the same device for roughly six minutes before the overlap was noticed.

## Why the data was unusable

Two benchmark processes on one device do not each receive half the machine in a way that averages out. They interleave at the scheduler, contend for L2 and memory bandwidth, and hold each other's clocks down. Every timing either process produced during the overlap is wrong by an amount that depends on what the other one happened to be doing at that instant.

The failure is invisible downstream. Replicates taken under contention are still self-consistent with one another, so the within-cell dispersion stays small, the bootstrap intervals stay tight, and the design check still reports adequate power. Nothing in the analysis can distinguish a contended measurement from a clean one -- which is precisely why it has to be prevented rather than detected after the fact.

## How it was detected

By checking process state rather than by anything in the data. `tasklist` showed two `losscolumn.cli.main` processes with creation timestamps six minutes apart, both holding roughly 900 MB.

## Disposition

Both processes were killed and both sets of numbers discarded. No artifact from either run was published, and the Thrust III sweep was re-run from the beginning on an idle device. Nothing is preserved here because nothing should be citable.

## Safeguard added

`losscolumn.thrusts.kernel.bench.MeasurementLock` -- an exclusive lock held for the duration of any timing run, checked by both the sweep and the calibration study. A second measurement now refuses to start and says which process holds the lock.

The first attempt at this guard queried `nvidia-smi --query-compute-apps` and was wrong: under Windows WDDM that call lists every graphics context on the machine, with `N/A` for memory, so it flagged 31 false positives including the desktop compositor. A guard that noisy gets switched off, which is worse than not having one. The lock targets the actual hazard -- another losscolumn measurement -- exactly and portably.

`scripts/measure_all.sh` is now the supported way to run every measurement, and runs them in series.
