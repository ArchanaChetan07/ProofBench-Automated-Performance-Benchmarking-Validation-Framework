# Machine stability

Protocol `machine-stability-sentinel-v1` sealed `sha256:acd3614e4e580a2a`, registered before measurement. 72 readings over 4 sessions and 12 process launches.

## Repeatability

| Level | observed CV | expected from below | probes | what it adds |
|---|---|---|---|---|
| within run | 21.3% | n/a | 6 | the instrument's own noise floor |
| across restart | 15.9% | 14.6% | 6 | process placement, allocator and transport setup |
| across session | 10.5% | 14.2% | 6 | whatever the machine does between sittings |

Excess over the level below: restart 1.09x &nbsp;&middot;&nbsp; session 0.74x. A level that adds nothing of its own sits at 1.0x, because a median of *n* is less dispersed than the *n* it averages.

**Drift is RANDOM.** average over more repeats; no session rule is needed.

- restarting the process leaves 1.09x the dispersion that averaging alone predicts (15.9% observed against 14.6% expected from the 21.3% within-run noise)
- a new session leaves 0.74x what restarts alone predict (10.5% observed against 14.2% expected)
- the session-level variation differs by 6.9x between the probes, which chance alone reproduces 40% of the time (so it is not evidence of anything: each per-probe CV rests on a handful of session medians and spreads of this size are ordinary under a null where every probe behaves identically)
- no level adds materially to the one below it, so the variation is the instrument's own and averages down with repeats

Correlation of per-reading bandwidth with machine state:

- `cpu_percent`: r = -0.07
- `gpu_mem_clock_mhz`: r = -0.03
- `free_ram_gb`: r = +0.03
- `load_avg_1m`: r = -0.02
- `gpu_clock_mhz`: r = -0.02

Largest session-to-session drift observed: **1.85x**. Recommended comparability criterion: **1.48x**.

## Session comparability

Three gates, failing on different things: the whole surface moving (`level`), one probe moving (`magnitude`), the probes moving against each other (`shape`).

| A | B | level | worst probe | shape | criterion | failed | verdict |
|---|---|---|---|---|---|---|---|
| S0 | S1 | 1.08x | 1.64x | 1.82x | 1.48x | magnitude | `INCOMPARABLE` |
| S0 | S2 | 1.05x | 1.50x | 1.55x | 1.48x | magnitude | `INCOMPARABLE` |
| S0 | S3 | 0.97x | 1.39x | 1.82x | 1.48x | shape | `INCOMPARABLE` |
| S1 | S2 | 1.02x | 1.29x | 1.65x | 1.48x | shape | `INCOMPARABLE` |
| S1 | S3 | 0.88x | 1.50x | 1.73x | 1.48x | magnitude | `INCOMPARABLE` |
| S2 | S3 | 0.92x | 1.85x | 1.98x | 1.48x | magnitude | `INCOMPARABLE` |

**0 of 6 session pairs are poolable.**

### A widening that was available and was not taken

The internal null used to correct the campaign's comparison is also computable here, from the sentinel's own repeats: it runs 1.17x to 1.43x on magnitude and up to 1.90x on shape. Applying it would move S1-S2 to COMPARABLE and make this 1 of 6 rather than 0 of 6.

It is not applied, and the reason is not a technical one. The sealed protocol registered a criterion of the form `max(1 + 3 * across_restart_CV, 1.05)` applied to the worst shared probe, and the sentinel is the design that criterion was registered for -- six probes, seven repeats, exactly as specified. Widening it now with a statistic the protocol does not name would be the post-hoc loosening LC-8.2 exists to forbid, and it would be done knowing which pair it admits.

The campaign is a different case and not a special one: 276 probes at two repeats is not the design the criterion was registered for, so carrying the number across was a category error rather than a loosening. Correcting a threshold that never applied is not the same act as relaxing one that does.

## What was corrected

The first analysis of these same readings called the drift `restart_level` on a 6.9x spread between the per-probe session variabilities. A permutation test on the readings shows chance alone reproduces a spread that large 40% of the time: each per-probe figure rests on four session medians, and the ratio between the largest and smallest of six such estimates is naturally enormous. The threshold was measuring the sample size, not the machine.
