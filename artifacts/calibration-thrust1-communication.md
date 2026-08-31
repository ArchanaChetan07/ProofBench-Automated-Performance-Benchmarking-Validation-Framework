### Thrust I communication model

Measured on NVIDIA T1000 8GB (sm_75, 9 GB). Protocol seal `7863b8219ff9b483...`.

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

| Family | params | fit R² | held-out median err | held-out max err | AIC |
|---|---|---|---|---|---|
| `linear` | 2 | 0.602 | 25.2% | 87.3% | -138.2 |
| `piecewise` | 5 | 0.790 | 18.1% | 83.4% | -139.9 |
| `regime` | 5 | 0.790 | 18.1% | 83.4% | -139.9 |

The best family, `piecewise`, still misses held-out points by a median of 18% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer. Note that run-to-run variation on this group is 13.5%, so the residual is close to the measurement noise: a better model family would not fix it, and a cleaner measurement might. That is a statement about the instrument, not about the model.

**gloo_shm / all_gather / world 3** &mdash; chosen family: `piecewise` &nbsp;&middot;&nbsp; measurement noise floor 5.8%

| Family | params | fit R² | held-out median err | held-out max err | AIC |
|---|---|---|---|---|---|
| `linear` | 2 | 0.889 | 28.0% | 37.4% | -150.7 |
| `piecewise` **<-** | 5 | 0.992 | 5.7% | 74.2% | -176.0 |
| `regime` | 5 | 0.992 | 5.7% | 74.2% | -176.0 |

`piecewise` wins on held-out error (5.7% median) against `linear` 28.0%, `regime` 5.7%. The decision is held-out error, not fit quality: a more flexible family always fits its own data better.

**gloo_shm / all_gather / world 4** &mdash; chosen family: `piecewise` &nbsp;&middot;&nbsp; measurement noise floor 5.5%

| Family | params | fit R² | held-out median err | held-out max err | AIC |
|---|---|---|---|---|---|
| `linear` | 2 | 0.721 | 17.3% | 44.5% | -123.4 |
| `piecewise` **<-** | 5 | 0.944 | 10.6% | 45.8% | -136.6 |
| `regime` | 5 | 0.944 | 10.6% | 45.8% | -136.6 |

`piecewise` wins on held-out error (10.6% median) against `linear` 17.3%, `regime` 10.6%. The decision is held-out error, not fit quality: a more flexible family always fits its own data better.

**gloo_shm / all_reduce / world 2** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 8.6%

| Family | params | fit R² | held-out median err | held-out max err | AIC |
|---|---|---|---|---|---|
| `linear` | 2 | 0.030 | 53.9% | 93.4% | -117.2 |
| `piecewise` | 5 | 0.976 | 28.4% | 67.1% | -155.8 |
| `regime` | 5 | 0.976 | 28.4% | 67.1% | -155.8 |

The best family, `piecewise`, still misses held-out points by a median of 28% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

**gloo_shm / all_reduce / world 3** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 11.5%

| Family | params | fit R² | held-out median err | held-out max err | AIC |
|---|---|---|---|---|---|
| `linear` | 2 | 0.391 | 50.5% | 82.4% | -136.6 |
| `piecewise` | 5 | 0.474 | 39.1% | 89.6% | -132.4 |
| `regime` | 2 | 0.391 | 50.5% | 82.4% | -136.6 |

The best family, `piecewise`, still misses held-out points by a median of 39% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

**gloo_shm / all_reduce / world 4** &mdash; chosen family: `none` &nbsp;&middot;&nbsp; measurement noise floor 11.4%

| Family | params | fit R² | held-out median err | held-out max err | AIC |
|---|---|---|---|---|---|
| `linear` | 2 | 0.635 | 45.8% | 89.5% | -143.1 |
| `piecewise` | 5 | 0.686 | 39.3% | 86.7% | -138.9 |
| `regime` | 2 | 0.635 | 45.8% | 89.5% | -143.1 |

The best family, `piecewise`, still misses held-out points by a median of 39% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

#### Validation on untouched message sizes

| Collective | median err | max err | R² | latency err | bandwidth err | covered |
|---|---|---|---|---|---|---|
| `gloo_shm/all_gather/world3` | 12.1% | 52.5% | 0.991 | 9.6% | 15.8% | 75% |
| `gloo_shm/all_gather/world4` | 13.4% | 63.0% | 0.995 | 10.1% | 16.1% | 71% |

Errors by regime:

- `gloo_shm/all_gather/world3`: tiny 6.1%, small 18.0%, medium 35.5%, large 8.3%
- `gloo_shm/all_gather/world4`: tiny 4.6%, small 21.9%, medium 43.5%, large 4.9%

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
| `gloo_shm/all_gather/world4/breakpoint_kib` | 3072 KiB | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world4/family` | 5 params | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world4/high_alpha_us` | 772.034 us | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world4/high_beta_gbs` | 0.14744 GB/s | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world4/low_alpha_us` | 376.821 us | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_gather/world4/low_beta_gbs` | 0.281859 GB/s | accepted | thrust1-comm-v1 calibration |
| `gloo_shm/all_reduce/world2/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |
| `gloo_shm/all_reduce/world3/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |
| `gloo_shm/all_reduce/world4/alpha_us` | nan us | diagnostic **not used** | thrust1-comm-v1 family selection |

Parameters not used by the model, and why:

- `gloo_shm/all_gather/world2/alpha_us`: The best family, `piecewise`, still misses held-out points by a median of 18% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer. Note that run-to-run variation on this group is 13.5%, so the residual is close to the measurement noise: a better model family would not fix it, and a cleaner measurement might. That is a statement about the instrument, not about the model.
- `gloo_shm/all_reduce/world2/alpha_us`: The best family, `piecewise`, still misses held-out points by a median of 28% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.
- `gloo_shm/all_reduce/world3/alpha_us`: The best family, `piecewise`, still misses held-out points by a median of 39% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.
- `gloo_shm/all_reduce/world4/alpha_us`: The best family, `piecewise`, still misses held-out points by a median of 39% (gate: 15%). No family here describes this transport, so none is promoted; all remain diagnostic. Widening the gate to admit one would be choosing the answer.

#### Limitations

- gloo over shared memory between processes on one host. This is not NVLink and not InfiniBand, and no parameter fitted here is a value for the fabric the Thrust I protocol registers.
- One host, so no inter-node hop and no NIC is exercised.
- World sizes 2 to 4, against the 8 and 32 the protocol registers.