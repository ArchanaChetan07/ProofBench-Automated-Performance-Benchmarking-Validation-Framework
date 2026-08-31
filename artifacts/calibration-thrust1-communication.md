### Thrust I communication model

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). Protocol seal `80cbfea2450c14a0...`.

#### Diagnosis: why one collective fits and the other does not

| Collective | world | linear R² | tiny GB/s | large GB/s | ratio | run-to-run |
|---|---|---|---|---|---|---|
| `all_gather` | 2 | 0.836 | 0.008 | 0.269 | 34.7x | 13.5% |
| `all_gather` | 3 | 0.806 | 0.009 | 0.238 | 27.5x | 5.8% |
| `all_gather` | 4 | 0.852 | 0.006 | 0.152 | 25.7x | 5.5% |
| `all_reduce` | 2 | 0.106 | 0.015 | 0.187 | 12.7x | 8.6% |
| `all_reduce` | 3 | 0.391 | 0.006 | 0.794 | 130.6x | 11.5% |
| `all_reduce` | 4 | 0.493 | 0.004 | 0.613 | 144.6x | 11.4% |

- **all_gather, world 2**: effective bandwidth rises 35x from the tiny tier to the large tier, so the transport is latency-dominated at the small end and bandwidth-dominated at the large end. One straight line has to pass through both, and cannot; run-to-run variation is 13.5%, small enough that the misfit is structural rather than noise.
- **all_gather, world 3**: effective bandwidth rises 28x from the tiny tier to the large tier, so the transport is latency-dominated at the small end and bandwidth-dominated at the large end. One straight line has to pass through both, and cannot; run-to-run variation is 5.8%, small enough that the misfit is structural rather than noise.
- **all_gather, world 4**: effective bandwidth rises 26x from the tiny tier to the large tier, so the transport is latency-dominated at the small end and bandwidth-dominated at the large end. One straight line has to pass through both, and cannot; run-to-run variation is 5.5%, small enough that the misfit is structural rather than noise.
- **all_reduce, world 2**: effective bandwidth rises 13x from the tiny tier to the large tier, so the transport is latency-dominated at the small end and bandwidth-dominated at the large end. One straight line has to pass through both, and cannot; run-to-run variation is 8.6%, small enough that the misfit is structural rather than noise.
- **all_reduce, world 3**: effective bandwidth rises 131x from the tiny tier to the large tier, so the transport is latency-dominated at the small end and bandwidth-dominated at the large end. One straight line has to pass through both, and cannot; run-to-run variation is 11.5%, small enough that the misfit is structural rather than noise.
- **all_reduce, world 4**: effective bandwidth rises 145x from the tiny tier to the large tier, so the transport is latency-dominated at the small end and bandwidth-dominated at the large end. One straight line has to pass through both, and cannot; run-to-run variation is 11.4%, small enough that the misfit is structural rather than noise.

#### Model family, chosen on held-out points

**gloo_shm / all_gather / world 2** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 13.5%

| Family | params | fit R² | held-out median | **worst tier** | AIC |
|---|---|---|---|---|---|
| `regime/weighted` | 5 | 0.854 | 12.9% | 69.9% | -258.3 |
| `piecewise/huber` | 5 | 0.853 | 13.2% | 70.0% | -258.2 |
| `regime/huber` | 5 | 0.853 | 13.2% | 70.0% | -258.2 |
| `piecewise/log` | 5 | 0.833 | 29.4% | 72.8% | -255.2 |
| `linear/huber` | 2 | 0.713 | 30.3% | 74.0% | -248.2 |
| `linear/weighted` | 2 | 0.724 | 29.1% | 75.2% | -249.1 |
| `regime/log` | 2 | 0.865 | 33.7% | 76.4% | -266.3 |
| `piecewise/weighted` | 5 | 0.860 | 14.4% | 77.7% | -259.3 |
| _1 further combinations_ | | | | | |

The best family, `regime/weighted`, still misses its worst-fitting regime by 70% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

**gloo_shm / all_gather / world 3** &mdash; chosen family: `piecewise/weighted` &nbsp;&middot;&nbsp; measurement noise floor 5.8%

| Family | params | fit R² | held-out median | **worst tier** | AIC |
|---|---|---|---|---|---|
| `piecewise/weighted` **<-** | 5 | 0.996 | 8.0% | 11.5% | -325.9 |
| `regime/weighted` | 5 | 0.996 | 9.3% | 16.2% | -323.6 |
| `piecewise/huber` | 5 | 0.992 | 12.1% | 16.2% | -306.6 |
| `regime/huber` | 5 | 0.992 | 12.1% | 16.2% | -306.6 |
| `piecewise/log` | 5 | 0.993 | 11.1% | 17.1% | -311.4 |
| `regime/log` | 5 | 0.993 | 11.1% | 17.1% | -311.4 |
| `linear/log` | 2 | 0.934 | 22.2% | 27.7% | -262.3 |
| `linear/weighted` | 2 | 0.880 | 24.1% | 31.0% | -247.9 |
| _1 further combinations_ | | | | | |

`piecewise/weighted` wins on worst-tier held-out error (11.5%) against `regime/weighted` 16.2%, `piecewise/huber` 16.2%. The decision is held-out error, not fit quality: a more flexible family always fits its own data better.

**gloo_shm / all_gather / world 4** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 5.5%

| Family | params | fit R² | held-out median | **worst tier** | AIC |
|---|---|---|---|---|---|
| `piecewise/huber` | 5 | 0.972 | 16.6% | 19.2% | -247.3 |
| `regime/huber` | 5 | 0.972 | 16.6% | 19.2% | -247.3 |
| `piecewise/log` | 5 | 0.986 | 17.5% | 22.9% | -264.4 |
| `regime/log` | 5 | 0.986 | 17.5% | 22.9% | -264.4 |
| `piecewise/weighted` | 5 | 0.993 | 17.9% | 24.2% | -279.4 |
| `regime/weighted` | 5 | 0.993 | 17.9% | 24.2% | -279.4 |
| `linear/log` | 2 | 0.898 | 22.3% | 28.6% | -222.1 |
| `linear/weighted` | 2 | 0.775 | 24.8% | 40.6% | -203.2 |
| _1 further combinations_ | | | | | |

The best family, `piecewise/huber`, still misses its worst-fitting regime by 19% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

**gloo_shm / all_reduce / world 2** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 8.6%

| Family | params | fit R² | held-out median | **worst tier** | AIC |
|---|---|---|---|---|---|
| `piecewise/weighted` | 5 | 0.865 | 21.8% | 43.0% | -207.5 |
| `regime/weighted` | 5 | 0.865 | 21.8% | 43.0% | -207.5 |
| `piecewise/log` | 5 | 0.670 | 22.9% | 52.8% | -186.1 |
| `regime/log` | 5 | 0.670 | 22.9% | 52.8% | -186.1 |
| `piecewise/huber` | 5 | 0.361 | 21.8% | 71.3% | -170.2 |
| `regime/huber` | 5 | 0.361 | 21.8% | 71.3% | -170.2 |
| `linear/huber` | 2 | -0.041 | 45.7% | 154.6% | -164.5 |
| `linear/weighted` | 2 | -0.009 | 46.3% | 155.6% | -165.2 |
| _1 further combinations_ | | | | | |

The best family, `piecewise/weighted`, still misses its worst-fitting regime by 43% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

**gloo_shm / all_reduce / world 3** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 11.5%

| Family | params | fit R² | held-out median | **worst tier** | AIC |
|---|---|---|---|---|---|
| `piecewise/huber` | 5 | 0.539 | 17.0% | 43.2% | -219.8 |
| `piecewise/weighted` | 5 | 0.546 | 22.7% | 45.4% | -220.1 |
| `regime/weighted` | 2 | 0.440 | 23.1% | 50.5% | -221.1 |
| `regime/huber` | 2 | 0.429 | 17.0% | 51.5% | -220.6 |
| `linear/weighted` | 2 | 0.440 | 22.9% | 59.4% | -221.1 |
| `linear/huber` | 2 | 0.429 | 25.0% | 61.0% | -220.6 |
| `regime/log` | 2 | 0.652 | 23.6% | 75.1% | -232.5 |
| `linear/log` | 2 | 0.652 | 36.2% | 91.3% | -232.5 |
| _1 further combinations_ | | | | | |

The best family, `piecewise/huber`, still misses its worst-fitting regime by 43% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

**gloo_shm / all_reduce / world 4** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 11.4%

| Family | params | fit R² | held-out median | **worst tier** | AIC |
|---|---|---|---|---|---|
| `piecewise/huber` | 5 | 0.764 | 27.8% | 53.3% | -233.0 |
| `piecewise/weighted` | 5 | 0.769 | 27.9% | 54.3% | -233.6 |
| `linear/huber` | 2 | 0.677 | 24.0% | 54.4% | -231.5 |
| `regime/huber` | 2 | 0.677 | 11.9% | 54.8% | -231.5 |
| `linear/weighted` | 2 | 0.683 | 22.7% | 54.8% | -232.0 |
| `regime/weighted` | 2 | 0.683 | 16.8% | 55.1% | -232.0 |
| `piecewise/log` | 5 | 0.884 | 30.4% | 67.4% | -250.2 |
| `regime/log` | 2 | 0.818 | 29.3% | 83.6% | -245.3 |
| _1 further combinations_ | | | | | |

The best family, `piecewise/huber`, still misses its worst-fitting regime by 53% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

#### Validation on untouched message sizes

| Collective | median err | max err | R² | latency err | bandwidth err | covered |
|---|---|---|---|---|---|---|
| `gloo_shm/all_gather/world3` | 12.1% | 52.5% | 0.991 | 9.6% | 15.8% | 75% |

Errors by regime:

- `gloo_shm/all_gather/world3`: tiny 6.1%, small 18.0%, medium 35.5%, large 8.3%

#### Parameters

| Parameter | Value | Provenance | Source |
|---|---|---|---|
| `gloo_shm/all_gather/world2/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |
| `gloo_shm/all_gather/world3/breakpoint_kib` | 2730.67 KiB | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world3/family` | 5 params | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world3/high_alpha_us` | 459.872 us | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world3/high_beta_gbs` | 0.242362 GB/s | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world3/low_alpha_us` | 208.791 us | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world3/low_beta_gbs` | 0.407424 GB/s | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world4/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |
| `gloo_shm/all_reduce/world2/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |
| `gloo_shm/all_reduce/world3/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |
| `gloo_shm/all_reduce/world4/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |

Parameters not used by the model, and why:

- `gloo_shm/all_gather/world2/alpha_us`: The best family, `regime/weighted`, still misses its worst-fitting regime by 70% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.
- `gloo_shm/all_gather/world4/alpha_us`: The best family, `piecewise/huber`, still misses its worst-fitting regime by 19% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.
- `gloo_shm/all_reduce/world2/alpha_us`: The best family, `piecewise/weighted`, still misses its worst-fitting regime by 43% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.
- `gloo_shm/all_reduce/world3/alpha_us`: The best family, `piecewise/huber`, still misses its worst-fitting regime by 43% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.
- `gloo_shm/all_reduce/world4/alpha_us`: The best family, `piecewise/huber`, still misses its worst-fitting regime by 53% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

#### Limitations

- gloo over shared memory between processes on one host. This is not NVLink and not InfiniBand, and no parameter fitted here is a value for the fabric the Thrust I protocol registers.
- One host, so no inter-node hop and no NIC is exercised.
- World sizes 2 to 4, against the 8 and 32 the protocol registers.