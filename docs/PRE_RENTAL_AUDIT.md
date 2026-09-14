# Pre-rental audit: what a GPU would find

Twenty-two findings from eight agents auditing the bug *classes* the first
rental exposed -- constants tuned on slow hardware, formulas never checked
against an external reference, code paths that only execute on GPU, and
anything that returns a plausible wrong number rather than failing.

Two are fixed and marked so. The rest are a work list, and they are
**unverified**: the audit had no adversarial stage, because an earlier
eight-finder run with three skeptics per finding exhausted the session limit
and returned nothing at all. Treat a finding here as a lead, not a
conclusion.

The two that were verified were both confirmed against the artifact the
rental produced. That is the only reason to expect the rest are worth
reading.

## blocks-measurement (4)

### `src/losscolumn/thrusts/kernel/reference.py:276`

**The sparse baseline rebuilds its attention mask with an O(nnz) Python loop inside the timed callable**

`sparse_reference` allocates a (seq_q, seq_k) bool tensor and fills it with one GPU slice-assignment per non-zero block, in a Python double loop, then does an arange/compare/in-place-`&=` for the causal term -- all before it calls SDPA. In `sparse_sweep.py` this whole function is what gets timed: line 338 binds it as `run_reference` and `paired_ab` (line 356) invokes it once per iteration, per replicate, per cell. So the baseline's recorded latency is `mask construction + SDPA`, while the method arm's is just the kernel. At seq_len=4096, density=0.25, causal, the layout has roughly 520 non-zero blocks, so every single baseline iteration issues ~520 Python-driven CUDA launches (~3 ms of pure host-side dispatch) on top of the attention it is supposed to be measuring. The same inflated callable is passed to `_peak` at line 368, so the memory envelope shows the same manufactured win: the baseline's peak includes a 16 MB bool mask plus its temporaries that the method never allocates.

*Why invisible locally:* On the T1000 the reference SDPA at these shapes takes milliseconds, so a few ms of mask-building was of the same order as the thing being measured and read as ordinary baseline slowness. Mask construction is fixed *host* cost -- it does not shrink when the GPU gets faster. On Blackwell the masked SDPA drops by an order of magnitude while the ~3 ms Python loop stays exactly where it is, so it goes from a fraction of the baseline to the dominant term in it. Nothing in the harness can detect this: replicates stay tight, the paired statistics stay valid, the loss map just fills with green. The 693-test suite never times anything, so it cannot see it either.

*Fix:* Build the mask once per cell, outside the timed region, and bind the prebuilt tensor into `run_reference` -- e.g. add a `mask=` parameter to `sparse_reference` and hoist `layout.dense_mask()` expansion to just after `build_layout` in `run_sparse_sweep`. Vectorise the expansion (np.repeat of the block grid) rather than looping. The same hoist fixes `_peak`. If the mask genuinely must be counted as part of the reference's cost, that is a deliberate protocol choice and belongs in `baseline_tuning_policy`, stated as such -- it currently claims the opposite, that the reference is given its best path.

*What a rental would see:* Every non-dense cell of both the portable and triton sweeps reports a large speedup over the reference, growing with seq_len and density, with tight confidence intervals. `_attribute` finds no loss regions to explain and the artifact publishes a sparse-attention win that is mostly the cost of a Python for-loop.

### [FIXED] `src/losscolumn/thrusts/overlap/campaign.py:767`

**analyse() silently discards every world-8 group; the headline 8-GPU numbers are measured, paid for, and never graded**

analyse() builds its coverage object from REQUIRED_WORLDS = (2, 3, 4) (line 58, via CommunicationCoverage.empty at line 754), then loops over the measured groups and does `g = cov.groups.get(key); if g is None: continue` (lines 766-768). scripts/gpu_campaign.py:246 sweeps `worlds = [w for w in (2, 3, 4, 8) if w <= len(ids)]`, so on the rented box every all_reduce/world8 and all_gather/world8 record hits `g is None` and is dropped before anything happens to it. Because the `continue` sits ABOVE `campaign.peaks[key] = analyse_peak(recs)` (line 770), world 8 gets no peak report, no threshold-band scan, no model selection, no held-out validation and no regime verdict. The records still land in campaign.records and are still serialised by to_dict(), so the raw world-8 bandwidths appear in the artifact with no grade attached.

*Why invisible locally:* The local campaign never runs world 8. REQUIRED_WORLDS and the local sweep list are the same tuple (2, 3, 4), so `cov.groups.get(key)` never misses and the `continue` is dead code on the development box. World 8 only becomes reachable when scripts/gpu_campaign.py derives worlds from a device set of eight, which requires eight GPUs.

*Fix:* Derive the coverage skeleton from the worlds actually swept rather than the module constant: pass the observed worlds into CommunicationCoverage.empty (e.g. `sorted({r.world for r in campaign.records}) or REQUIRED_WORLDS`), keeping REQUIRED_WORLDS as the set that must be present for readiness. At minimum, replace the bare `continue` with a recorded note (`campaign.notes.append(f"{key} measured but not in the required grid; not analysed")`) so a group that was measured and dropped is visible in the artifact, and make scripts/gpu_report.py print "not analysed" rather than an em-dash when `cov['groups']` has no entry for a group it is tabulating.

*What a rental would see:* artifacts/gpu-campaign.json contains thousands of world-8 PointRecords with bandwidths, but `peaks`, `model_selection`, `thresholds` and `coverage` contain nothing for `all_reduce/world8` or `all_gather/world8`. scripts/gpu_report.py:78-92 builds its group table from the valid RECORDS, so the world-8 rows DO appear with a peak GB/s and a median CV; its `uncovered` column comes from `(cov.get('groups') or {}).get(key, {})`, which returns `{}`, so `bad` is empty and the cell renders as an em-dash. The eight-GPU row is printed as having nothing uncovered precisely because it was never examined. The final print in gpu_campaign.py:300-304 also takes `max(bw)` over all records including world 8, so the quoted peak bandwidth for the fabric can come from the one world size that no gate ever looked at.

### `src/losscolumn/thrusts/overlap/campaign.py:378`

**A point where most repeats crashed is recorded as valid with cv = 0.0, and the crash reason is thrown away**

`_finalise` (lines 362-385) builds the PointRecord from whatever timings survived and never consults `n_failures` or `n_runs`. The inner `except Exception` in `_worker` (lines 313-315) catches a failure in the middle of a repeat, increments `failures`, and lets the repeat loop continue, so a point can come back with 1 of 3 repeats. `_finalise` then sets `rec.cv = np.std(t)/np.mean(t)` over that single survivor, which is exactly 0.0 (numpy's std of a one-element array), `rec.iqr_s = 0.0`, and leaves `rec.valid = True`. Worse, `raw["invalid_reason"]` -- the worker's exception text -- is only copied into the record inside the `if not rec.timings_s` branch (lines 371-374); when one timing survived it is silently dropped, so the artifact contains no trace that NCCL raised anything. cv 0.0 is not inert: `analyse` medians it into `noise_by_regime` (line 780), `_point_se(0.0, 3)` returns 0.0, the regime is therefore never added to `ungradable` (lines 802-808) and sails past the TOO_NOISY gate at line 923, so it can be marked COVERED. The regime with the most transport failures becomes the quietest regime in the campaign and the one most likely to be reported as covered.

*Why invisible locally:* On gloo over shared memory on one host a collective does not raise; every point gets all `repeats` timings, `n_failures` is always 0, and cv is always a real 3-sample dispersion. The partial-repeat branch is dead code locally. The 693-test suite exercises `_finalise` only through fully-populated rows (tests/test_collectives.py:104 passes `n_failures: 0`). Transient collective failures -- an aborted communicator, an ECC-retired page, a link flap -- are an NCCL-on-rented-hardware phenomenon.

*Fix:* Treat a point that did not complete its full repeat budget as not measured rather than as measured precisely: in `_finalise`, always carry `raw["invalid_reason"]` onto the record, and set `valid = False` (or a distinct status) when `n_runs < repeats` or `n_failures > 0`. At minimum make `cv` `float('nan')` when `len(timings_s) < 2` -- a dispersion estimated from one sample is not zero dispersion -- and make the `ungradable` test at campaign.py:802-808 and the gate at line 923 treat a non-finite `point_se` as ungradable instead of passing it.

*What a rental would see:* A coverage matrix in which the flakiest group/regime shows `noise_cv: 0.0`, `point_se: 0.0`, status `covered`, `detail: "held-out error x% over n point(s)"`, with `n_runs: 1, n_failures: 2` sitting unread in the same JSON record and `invalid_reason: ""`. Readiness can come back READY on the strength of cells whose measurements mostly crashed.

### `src/losscolumn/thrusts/overlap/torch_backend.py:210`

**build_fsdp_step selects the CUDA device inside an `if`, after process-group init; whole classes of launch put all eight ranks on cuda:0**

Lines 210-212 read `if cfg.world_size > 1 and not dist.is_initialized(): dist.init_process_group(backend="nccl"); torch.cuda.set_device(state["local_rank"])`. Three problems, all NCCL-only. (1) set_device runs AFTER init_process_group, the documented inversion for NCCL. (2) If the launcher or an enclosing harness already called init_process_group -- the normal torchrun pattern -- `not dist.is_initialized()` is False, the whole body is skipped, set_device is never called, and the `.cuda()` at line 233 and `device="cuda"` at line 242 resolve to device 0 on every rank. (3) If the first configuration in a sweep has cfg.world_size == 1, the body is skipped for the same reason; feasible() at lines 145-154 explicitly permits world_size=1 under a launch (it only rejects world_size>1 WITHOUT a launch), so eight ranks each build a full model on cuda:0.

*Why invisible locally:* build_fsdp_step has no caller anywhere in the repository (grep for it returns only its own definition), it is guarded by `if not HAS_TORCH: raise` at line 201, and every line from 203 onward imports torch.distributed.fsdp and calls .cuda(). None of the 693 tests can reach it on a single-GPU gloo box, and on that box local_rank is always 0 anyway, so the skipped set_device would be a no-op even if it did run.

*Fix:* Hoist device selection out of the conditional and above the process-group init: `torch.cuda.set_device(state["local_rank"]); dev = torch.device("cuda", state["local_rank"])` unconditionally, then `if cfg.world_size > 1 and not dist.is_initialized(): dist.init_process_group(backend="nccl", device_id=dev)`. Replace the bare `.cuda()` at line 233 and `device="cuda"` at line 242 with that explicit `dev`, so the tensor placement cannot silently depend on ambient process state.

*What a rental would see:* Under a torchrun launch that pre-initialises the group, eight ranks allocate an FSDP-wrapped model and an AdamW state on cuda:0 while GPUs 1-7 sit idle; NCCL then raises "Duplicate GPU detected" or hangs in the first all-gather, and the run dies with no traces written. Under a sweep whose first cell is world_size=1, the same eight models land on cuda:0 without any collective to expose it: the run completes and reports a single-GPU baseline throughput that is eight-way contended on one card. That baseline is the denominator for every scaling claim in Thrust I, so it does not fail -- it returns a number that is low by roughly the contention factor, in the direction that flatters every multi-GPU comparison made against it.

## wrong-number (9)

### `src/losscolumn/core/stability.py:418`

**Noise-level CVs use the population standard deviation (ddof=0), biasing the launch-level floor toward zero**

`_cv` returns `a.std() / a.mean()` with numpy's default ddof=0, so it estimates sigma low by sqrt((n-1)/n): 29% low at n=2, 18% at n=3, 11% at n=5. This matters because the two CVs it feeds are aggregated over very different group sizes and then subtracted in quadrature. `within_run.cv` (line 521) comes from `MeasurementRecord.within_cv` (line 196, also ddof=0) over ~21 repeats - essentially unbiased. `across_restart.cv` (line 531) comes from `_cv` over the handful of restart medians in one session, and `across_session` (line 559) over as few as 2-3 session medians - badly biased. `NoiseBudget.launch_cv` then computes `sqrt(max(across_restart_cv^2 - expected_restart_cv^2, 0))` (modellability.py:76-79), subtracting a nearly-unbiased quantity from a biased-low one. Worked through: with a true across-restart CV of 5.0% measured over n=3 restarts and an expected 4.0%, the measured 4.08% gives launch_cv = 0.8% against a true 3.0% - a 3.7x understatement of the floor, and the clamp to zero makes it disappear entirely whenever the bias pushes the measured value under the expected one.

*Why invisible locally:* This is a small-n bias, and it is hidden by the clamp: when it drives the launch component to zero the code reports "no launch component to speak of" (modellability.py:69-72), which reads as a clean result rather than a failed estimate. It gets worse exactly where it costs most - a rented box is run for few sessions and few restarts because sessions are expensive, so n drops and the bias grows, while the local desktop can afford many restarts. Nothing in the test suite can catch it because there is no external reference for what the launch-level dispersion should be.

*Fix:* Use `a.std(ddof=1)` in `_cv` (stability.py:418) and in `MeasurementRecord.within_cv` (stability.py:196) - both are sample estimates, not population parameters. Since the quadrature subtraction is a variance-components estimate, also record n per group and report a confidence interval (or at least the group size) alongside `launch_cv`, so a floor estimated from 2 restarts is not presented with the same authority as one from 20. Consumers affected: `NoiseBudget.floor` drives `plan_replicates`' `reachable_by_repeats_alone` and the launch count (modellability.py:178-196).

### `src/losscolumn/thrusts/kernel/sparse_sweep.py:362`

**The reference arm's TFLOP/s is computed from the sparse kernel's block count, not from the work the reference does**

Line 358 computes `flops = attention_flops_sparse(...)`, which by its own docstring is "FLOPs for the blocks actually computed" by the block-sparse kernel: `batch*heads*layout.nnz*2*2*block_q*block_k*head_dim` (sparsity.py:209-222). Line 361 divides it into the method's time - correct. Line 362 divides the same number into the BASELINE's time, but the baseline is not doing that work. For non-dense patterns `run_reference = sparse_reference(...)` computes the pattern densely under an explicit boolean mask (the comment at lines 330-334 says so outright), i.e. seq_q x seq_k, so its reported TFLOP/s is understated by roughly the realized density - a 4x understatement at density 0.25. At `phase == "decode"` it is worse and in the other direction: `seq_q = 1` (line 295), so nq=1 and the layout's block_q=64 counts 64 query rows, while the reference computes exactly one unpadded row - the baseline's TFLOP/s is overstated by up to 64x.

*Why invisible locally:* `compare_cells` and the loss column run on the `latency` envelope (lines 373-377), not on `throughput`, so the whole sweep's verdicts are correct regardless and nothing fails. The throughput envelope is still published verbatim into the claim artifact at runners.py:536 under `throughput_envelope`, with the METHOD and BASELINE arms side by side and the metric labelled "FLOPs over blocks actually computed, per second" (sparse_sweep.py:252-253). On the weak local card no one compares those absolute TFLOP/s against anything; on a real GPU a reader checking the reference against a known SDPA roofline sees a fused flash kernel apparently running at a quarter of its real rate, or at 64x it in the decode rows.

*Fix:* Compute the baseline's FLOPs from what the baseline actually executes and keep them separate: for the non-dense patterns that is `dense_equivalent_flops(bs, heads, head_dim, seq_q, seq, causal)` (already present at sparsity.py:224), and for decode it is the unpadded `seq_q=1` count rather than a padded 64-row block. Alternatively label the throughput envelope explicitly as "method-equivalent FLOPs per second" so the baseline column is not read as the reference's achieved throughput.

### `src/losscolumn/thrusts/kernel/triton_fa.py:282`

**Causal attention FLOPs counted as exactly half, which the function's own docstring says is the wrong number**

`attention_flops` returns `full * 0.5` for causal. Its docstring (lines 274-279) states the opposite: "The causal case does slightly more than half the work, not exactly half: the diagonal blocks are computed in full and then masked. Reporting the exact-half figure ... is the kind of small dishonesty that accumulates across a table." Both kernels this number is divided into do exactly what the docstring describes: the Triton kernel stops at `hi = tl.minimum(SEQ_K, (pid_m + 1) * BLOCK_M + offset)` (triton_fa.py:192) and the portable one at `k_end = min(q1 + offset, sk)` (reference.py:110), i.e. whole BLOCK_M x BLOCK_N tiles up to and including the diagonal, masked inside. Deriving the block-granular count independently: fraction of the s x s score matrix actually visited = sum_i ceil((i+1)*BM/BN)*BM*BN / s^2. With the autotuner's BM=128, BN=64 this is 0.625 at seq=512, 0.5625 at 1024, 0.5312 at 2048, 0.5156 at 4096 - never 0.5. Every causal TFLOP/s in the Thrust II envelope is therefore understated by 3%-25%, largest at the short sequences. The same literal appears at sparsity.py:227 in `dense_equivalent_flops`.

*Why invisible locally:* The number is only ever divided into the method's time and the baseline's time in the same cell, so the ratio cancels and every verdict, loss region and comparison is unaffected - exactly the "only ever compared against itself" pattern. The 693-test suite actively locks the wrong value in: tests/test_kernel.py:251-254 `test_causal_is_half` asserts `half == full / 2`, so the check that would have caught it certifies it instead. Locally the default is `causal=False` (sweep.py:109), so the causal path is rarely exercised; on the rented box a causal sweep publishes an absolute TFLOP/s table that a reader will compare against a datasheet roofline.

*Fix:* Count the blocks the kernel actually visits, the way `attention_flops_sparse` already does: take block_q/block_k (or the layout) as arguments and return `2*2*batch*heads*head_dim * sum_i(min(ceil(min((i+1)*block_q + offset, seq_k)/block_k), n_k_blocks) * block_q * block_k)`. Then fix tests/test_kernel.py:251 to assert the block-granular fraction (e.g. 0.5625 at seq=1024, BM=128/BN=64) rather than 0.5, and reconcile the docstring's "inflates" with the actual direction (it deflates).

### `src/losscolumn/thrusts/overlap/calibrate.py:452`

**Measured MFU counts a gated (3-matrix) FFN and non-causal attention against a block that has neither, inflating it ~37%**

`_measured_mfu` (line 452) and `_model_tokens_per_s` (line 408) both compute per-layer GEMM FLOPs as `2*mb*seq*(4*h*h + 3*h*hf)`. The `3*h*hf` term is the SwiGLU count (gate + up + down) inherited from `ModelSpec.params_per_layer` (simulate.py:72), which is correct for the llama-7b target. But the block these formulas are graded against is built by `_block()` at calibrate.py:225-231 and has `qkv`, `proj`, `up`, `down` only - no gate - i.e. `4*h*h + 2*h*hf`. At the calibration geometry (hidden=1024, ffn=4096, set at calibrate.py:252-254 and 300-301) that is 16,777,216 vs 12,582,912 per token-pair, a 33.3% overstatement. Second, independent error in the same expression: `attn_fwd = 4*mb*n_heads*seq*seq*head_dim` is the full non-causal count, but the block calls `scaled_dot_product_attention(..., is_causal=True)` (calibrate.py:229), which does roughly 0.53-0.56 of that. At mb=4, seq=1024 the two together make the FLOP numerator 1.37x the work the GPU actually did, so `measured_mfu` comes out ~37% too high.

*Why invisible locally:* On the T1000 the inflated MFU exceeds 1.0, and the code at lines 382-398 catches that and blames it entirely on the absent tensor cores making the large fp16 GEMM ceiling invalid - a plausible, well-written explanation that absorbs the arithmetic error and stops anyone looking further. On a rented box with real tensor cores the GEMM ceiling is valid and the MFU lands comfortably below 1.0, so nothing trips; the number just comes out 1.37x too high. It also cancels out of every comparison the study grades: `_model_tokens_per_s` uses the same inflated count for both RECOMMENDED and BEST, so the predicted-vs-measured verdicts are unaffected and the only casualty is the one absolute number.

*Fix:* Make the FFN and attention terms match what is actually benchmarked: use `4*h*h + 2*h*hf` (or give `_block` a gate projection so the 3x term is honest) and multiply the attention term by the causal fraction, since `_block` runs `is_causal=True`. Better, derive the count from `sum(p.numel() for p in block.parameters())` so the formula cannot drift from the module again. Note the table at calibrate.py:95-96 publishes `measured_mfu / assumed_mfu` as a parameter-correction ratio, so this value is acted on, not just displayed.

### `src/losscolumn/core/cliffs.py:166`

**Cliff exposure defaults to 0.0% for every cell, so a configuration that was never tested -- or never ran -- publishes as having no exposure**

`detect_cliffs` seeds `exposure = {env.label(c): 0.0 for c in env.cells()}` and thereafter only ever raises a cell's value (lines 230-233 and 255). Three distinct situations therefore share the value 0.0: (a) tested in every direction and genuinely flat; (b) never tested, because the step was skipped when either endpoint had no replicates (line 175, `if a.size == 0 or b.size == 0: continue` -- `Envelope.replicates_at` returns an empty array for an unmeasured or failed cell); (c) never tested, because the cell is the top level of an ordered axis and has no outgoing step, or because the axis is unordered and is excluded from `axes` altogether (line 163 filters to `f.ordered`). Note also that exposure is only ever written against `c0`, the from-cell of an increasing step, so a one-level move *down* an axis is never scored -- the `Cliff.direction` field at line 63 defaults to "increase" and is never assigned anything else. The 0.0 then propagates into the published artifact verbatim: `cliff_adjacency` reads it with `report.exposure.get(env.label(cell), 0.0)` (line 311) and `runners._adjacency_html` renders `r.get('exposure_pct', 0):.1f}%` (src/losscolumn/runners.py:217). That same renderer also drops the `error` key entirely for a configuration `cliff_adjacency` could not locate in the envelope (cliffs.py line 310), so an unlocatable config prints as "no / 0.0% / -": the most reassuring row the table can produce, from no data.

*Why invisible locally:* Thrust I runs on the calibratable model path locally (`evidence_class == "simulated"`), where every cell of the envelope has replicates and the surface is smooth, so every adjacent step is measurable and 0.0 genuinely means flat. There is no external reference the exposure number is checked against -- it is only ever read back from the artifact it was written to. On measured hardware, cells that OOM come back with all-NaN replicates and drop out of the step enumeration.

*Fix:* Make exposure tri-state rather than a float initialised to zero: leave a cell absent from the map (or store None) until a step out of it has actually been scored, and have `exposure_of`/`cliff_adjacency` report "not tested" rather than 0.0 for the absent case. Have `cliff_adjacency` refuse a cell whose `env.replicates_at` is empty, and have `_adjacency_html` render the `error` key when present instead of falling back to `.get(..., 0)`. Either score both directions of every adjacent pair or relabel the column to say it covers increasing steps on ordered axes only -- the current caption says "a single-level change in any direction".

*What a rental would see:* The RQ1 "cliff adjacency" table in thrust1-overlap-envelope.html reporting `Adjacent to a cliff: no`, `One-step exposure: 0.0%` for named recommendations such as "same recipe scaled to 4 nodes" (world_size=32, the top level of that axis) or "long-context default" (seq_len=8192) -- including for a configuration that failed to run at all on the rented node, since `cliff_adjacency` never checks that the cell has data before looking up its exposure.

### `src/losscolumn/core/modellability.py:95`

**An unmeasured launch-level noise floor is substituted with exactly zero, so "is this gate reachable?" is always answered yes**

`NoiseBudget.launch_cv` correctly returns NaN when `across_restart_cv` was never measured, and `floor` correctly propagates the NaN. But `point_noise` -- the function every reachability decision actually calls -- does `lv = 0.0 if not math.isfinite(lv) else ...` at line 95, converting "the launch component is unknown" into "the launch component is zero". `plan_replicates` then skips its floor branch (line 179 requires `math.isfinite(floor)`), runs the repeats-only loop, and returns `reachable_at_all=True` with a note that reads, literally, "the launch-level floor is nan%, below the target, so no extra launches are required". `_with_floor` compounds it: when `expected_restart_cv` is non-finite (which happens whenever `within_run.n_aggregated` is absent or 0) it returns the cell budget unchanged at lines 299-303, silently dropping the machine floor it exists to apply. And `assess_modellability` line 442 guards the pessimistic verdict on `math.isfinite(floor)`, so a NaN floor can only produce optimistic verdicts -- NOT_MODELLABLE_AT_THIS_GATE is unreachable, MODELLABLE is not.

*Why invisible locally:* The local envelope is built by scripts/modellability.py:`_envelope_from_recampaign`, which derives `across_restart.cv` from probes measured in two passes (`if len(v) >= 2`). Locally both passes always complete, so `cvs` is non-empty and the floor is finite -- the NaN branch never runs. It becomes live the moment a probe has fewer than two passes: a single-pass campaign (the obvious way to save money on a rented box), or a pass that errored out for one world size, leaves `cvs == []` and `across_restart_cv` NaN.

*Fix:* Make `point_noise` return NaN when `launch_cv` is NaN (unknown is not zero), and have `plan_replicates` return a plan with `reachable_at_all=False` and a note saying the floor was never measured, rather than a repeat count. `_with_floor` should signal that the floor could not be applied instead of returning the floor-free budget. Add a NOISE_FLOOR_NOT_MEASURED verdict to `assess_modellability` ahead of the `math.isfinite(floor)` checks.

*What a rental would see:* artifacts/modellability.json with `"launch_cv": NaN, "floor": NaN` and, in the same document, `"verdict": "MODELLABLE"`, `"reachable_at_all": true`, and per-target plans quoting a specific repeat count. The one question the module exists to answer -- whether more measurement can ever clear the gate -- is answered affirmatively from no restart data at all.

### `src/losscolumn/core/modellability.py:350`

**A cell with a missing or zero noise CV is costed at zero extra repeats and declared BUYABLE**

`assess_modellability` costs a cell only `if cv == cv and cv > 0` (line 350). When the debt item carries `noise_cv` NaN (never measured) or 0.0, no plan is built, `c.repeats_needed` keeps its dataclass default of 0 and `c.launches_needed` its default of 1, and the cell falls straight through to the `item["kind"]` branches. For `kind == "noise_limited"` it emerges as verdict BUYABLE with the reason "the residual (31.0%) is the instrument's; 0 repeats per point reaches the gate" -- a costing of zero work for a cell that missed the gate by a factor of two, and that nothing was ever costed for. This chains directly onto the first finding: a partial-repeat point produces `cv == 0.0`, which lands in exactly this branch.

*Why invisible locally:* Locally every debt item carries a real positive `noise_cv` (gloo never fails a repeat, so no cv collapses to 0.0, and the stability sweep populates every cell), so the guard is always true and a plan is always costed. The zero-work path only opens when upstream measurement degrades -- which is the rented-hardware case.

*Fix:* When no plan could be costed, set the verdict to UNDETERMINED (or a NOT_COSTED state) with the reason naming the missing CV, and leave `repeats_needed`/`launches_needed` as `None` rather than 0/1, so a consumer summing planned work cannot silently add zero. Do not let a cell reach BUYABLE without a plan object behind it.

*What a rental would see:* A cell-by-cell modellability table whose rows read BUYABLE / 0 repeats / 1 launch for the cells that measured worst, and an overall verdict of MODELLABLE ("Every uncovered cell is reachable with more replicates alone") derived from cells no plan was ever computed for.

### `src/losscolumn/thrusts/kernel/triton_fa.py:416`

**The sparse kernel wrapper re-uploads the layout from host memory on every timed call**

`flash_attention_triton_sparse` converts `layout.crow` and `layout.cols` -- numpy arrays living in pageable host memory -- to CUDA tensors on every invocation, plus three `.contiguous()` calls and a `torch.empty_like`. Because `run_method` (sparse_sweep.py:320) binds this wrapper, not the kernel, those two pageable host-to-device copies are inside the CUDA-event window for every one of the `iters` x `replicates` timed iterations. A pageable H2D copy is a blocking `cudaMemcpy` of tens of microseconds regardless of how small the array is; the layout itself is at most a few KB and is identical across every call in the cell.

*Why invisible locally:* On the T1000 a prefill kernel at seq 4096 runs for roughly a millisecond, so ~30 us of fixed per-call setup is a 3% tax that disappears into the replicate spread. On Blackwell with tensor cores the same kernel runs in tens of microseconds and the decode-phase kernel (seq_q=1, one query block, grid = (1, batch*heads)) runs in single-digit microseconds -- at which point the upload is comparable to or larger than the kernel, and the measured "kernel latency" is mostly memcpy. This is the identical shape to the already-fixed MAX_ITERS bug: a cost sized when a call took ~1 ms. No test covers it because the timed path only executes on a GPU.

*Fix:* Move the layout upload out of the per-call path: cache the device tensors on the `BlockLayout` (or keyed by `(id(layout), device)`), or accept already-device `crow`/`cols` and have `run_sparse_sweep` upload them once per cell, right after `build_layout`. Also hoist the output allocation if you want the timed callable to be the kernel and nothing else.

*What a rental would see:* The triton arm's decode-phase latency plateaus at a floor of a few tens of microseconds no matter the sequence length, density or pattern -- a suspiciously flat row across the decode facet of the loss map. `_attribute` (sparse_sweep.py:454-459) then confidently blames it on the 64-wide query block padding, which is the wrong mechanism.

### `src/losscolumn/thrusts/overlap/torch_backend.py:167`

**No memory is freed between sweep points, and feasible() is designed to read the resulting OOM as a property of the configuration**

TorchProfilerBackend.measure() calls `step_fn = self.build_step(cfg)` (line 167) once per configuration. build_fsdp_step allocates a fresh n_layers-deep model (line 233), an FSDP wrap (line 240), a fused AdamW with its exp_avg/exp_avg_sq state (line 241) and an input activation (line 242). Nothing in measure() or build_fsdp_step calls torch.cuda.empty_cache(), reset_peak_memory_stats(), or the free_memory() helper that already exists at thrusts/kernel/bench.py:155-159. Across a sweep the caching allocator keeps every previous configuration's blocks, and because successive configurations have different shapes (micro_batch, seq_len, strategy all vary) those blocks are the wrong sizes to be reused, so the arena fragments. The docstring of feasible() (lines 135-139) states the design intent explicitly: "an OOM is discovered by attempting the configuration". So an OOM caused by the previous point's residue is recorded as the current point being infeasible.

*Why invisible locally:* Neither measure() nor build_fsdp_step can execute without CUDA (line 171 `torch.cuda.synchronize()`, line 233 `.cuda()`), and build_fsdp_step has no caller in the repo, so no test allocates anything. On a single 8.6 GB card the sweep would OOM on its first real configuration anyway, so the accumulation across points has never had a chance to happen.

*Fix:* Bracket each configuration: at the end of measure(), `del step_fn` then call `losscolumn.thrusts.kernel.bench.free_memory()` (gc.collect + empty_cache + reset_peak_memory_stats), and call it again immediately before `self.build_step(cfg)` so each point starts from a clean arena. Record torch.cuda.memory_allocated() before and after into the trace config dict, so a point that starts with a non-empty arena is visible rather than inferred.

*What a rental would see:* The sweep runs cleanly through its first several sharding configurations, then starts reporting OOM on configurations that would fit in isolation -- typically the later, larger-micro-batch cells, which are exactly the cells Thrust I's H0 is about. Those cells enter the artifact as legitimately infeasible, producing a memory-feasibility frontier that is an artifact of sweep order rather than of the model. Re-running the failing configuration alone would succeed, but nothing in the harness prompts that check.

## wrong-label (5)

### `src/losscolumn/runners.py:543`

**Every published thrust3 claim asserts unconditionally that the device predates FlashAttention-3**

The limitations list interpolates the measured device name and capability into a sentence that then states, as fact, that this hardware "predates the architecture FlashAttention-3 targets" and that "FA-3's warp specialisation, TMA and ping-pong scheduling do not exist on this hardware for either arm to use". There is no condition on it -- unlike `_tensor_core_limitation` immediately above at line 542, which correctly suppresses itself when the probe finds tensor cores. On a Blackwell part the string renders as `Measured on NVIDIA RTX PRO 6000 Blackwell (sm_120), which predates the architecture FlashAttention-3 targets`, which is false on its face and is the one sentence in the artifact a reviewer will check.

*Why invisible locally:* On the T1000 (sm_75) the sentence is true, so it reads as a careful caveat rather than a hardcoded assumption. It only becomes a false claim when the artifact is produced on hardware newer than Hopper -- which is exactly the run it is being prepared for. No test asserts on limitation text.

*Fix:* Make it conditional on the measured capability, the way `_tensor_core_limitation` already is. Below sm_90, keep the current text. At sm_90, say FA-3's mechanisms exist and this kernel does not use them. Above sm_90 (sm_100/sm_120), say the device postdates Hopper, that FA-3's Hopper-specific asynchrony is not what torch dispatches here either, and name what the reference backend actually was -- rather than asserting an absence that is not true.

*What a rental would see:* A published HTML claim whose limitations section describes a 96 GB Blackwell as predating Hopper. The neighbouring bullets are correct, which makes the wrong one more damaging, not less.

### `src/losscolumn/thrusts/kernel/correctness.py:153`

**The `masked` check never constructs a fully-masked row, so the bug it exists to catch cannot fail it**

The comment at lines 150-152 states correctly that under a right-aligned causal mask with seq_k == seq_q every row sees at least one key, so "the degenerate case is constructed explicitly" -- and then line 153 constructs a spec with `seq_k` set to `spec.seq_len`, i.e. exactly the seq_k == seq_q case it just ruled out. `offset = sk - sq` is 0, row 0 attends to key 0, and no row is ever fully masked. The check re-runs the same non-degenerate shape with `seed + 1` and asserts finiteness, so it reports `passed: True` for a NaN class it never exercised. Reaching a fully-masked row requires `seq_k < seq_len` (e.g. seq_q=256, seq_k=64 masks rows 0..191 entirely). Note the sparse sweep -- the one that actually runs on the rental -- has no masked check at all: `_check_correctness` tests only output, determinism and dtype.

*Why invisible locally:* The check passes, which is what it is supposed to do, and it passes for the wrong reason. Both the Triton kernel (triton_fa.py:211-215) and the portable implementation do guard the -inf case, so there is no observable symptom on any hardware -- until an autotune config that the T1000 never selected changes the block geometry and the guard is exercised for the first time on Blackwell. That is precisely when the gate is needed and precisely when it is inert.

*Fix:* Build the degenerate case: `ShapeSpec(**{**spec.to_dict(), 'seq_len': spec.seq_len, 'seq_k': max(spec.seq_len // 4, 1)})` so `offset` is negative and the leading query blocks are fully masked, and assert the masked rows are exactly zero rather than merely finite. Add the same check to `sparse_sweep._check_correctness`, which currently has none.

*What a rental would see:* A correctness table reporting a passing `masked` check on every shape, with no indication that the fully-masked-row path was never entered.

### `src/losscolumn/thrusts/kernel/sparse_sweep.py:470`

**Loss attributions are keyed to absolute thresholds calibrated on the T1000, so they name the wrong mechanism on fast hardware**

`_attribute` decides *why* a region lost from hardcoded cut points rather than from any timing it just took. Line 470 says a region is "short sequences: the kernel runs for tens of microseconds and fixed per-launch cost is a large share of it" whenever `max(seqs) <= 1024` -- a cutoff that encodes the T1000's speed, where 1024 tokens took tens of microseconds and 4096 took milliseconds. On Blackwell with tensor cores every registered sequence length including 4096 runs in tens of microseconds, so the launch-bound explanation is true of cells the code refuses to apply it to and the >1024 cells get attributed elsewhere. The same pattern is in the dense sweep at sweep.py:268, where `< 32` FLOP/byte is the memory-bound test: arithmetic intensity here is seq/2, so with the registered seq_lens the branch can never fire at all -- while the actual ridge point of a tensor-core Blackwell (~250 TFLOP/s fp16 over ~1.8 TB/s, ~140 FLOP/byte) puts most of the grid on the memory-bound side of it. sweep.py:293-297 then asserts, for any region touching seq>=2048, that the reference "overlaps softmax with the next GEMM through warp specialisation and TMA" -- Hopper mechanisms that torch does not dispatch on sm_120.

*Why invisible locally:* These are prose thresholds, not numbers a test can check, and on the machine they were written for every one of them happened to be right. The attribution strings are what the artifact publishes as its explanatory contribution, so being wrong here is not cosmetic -- it is the finding being wrong while every measurement around it is fine.

*Fix:* Derive the cut points from the measurement instead of hardcoding them. For the launch-bound claim, compare the measured median latency against a per-run empty-kernel launch floor (time a trivial kernel once at startup) rather than against a sequence length. For the roofline, compute the ridge point from `tensor_core_probe()`'s measured fp16 TFLOP/s and a measured HBM bandwidth, instead of the literal 32. For the FA-3/TMA sentence, gate on capability the same way finding #3 needs to be gated.

*What a rental would see:* Loss regions carrying confident mechanistic explanations that contradict the roofline of the machine they were measured on -- cells labelled compute-bound that are memory-bound, and genuinely launch-bound long-sequence cells explained as code-generation quality.

### [FIXED] `src/losscolumn/thrusts/overlap/campaign.py:632`

**Campaign.protocol() hard-codes the transport as gloo shared memory and seals it into every NCCL artifact**

Campaign has no backend or fabric field. protocol() emits fixed strings describing the local machine: `"world_sizes": list(REQUIRED_WORLDS)` (line 631, always [2,3,4] even when 8 was swept), `"transport": "gloo over shared memory, one host"` (line 632), and a `transferability` paragraph beginning "NOTHING measured here is a value for NVLink or InfiniBand. This is gloo over shared memory on one host" (lines 688-694). Line 695 then computes `doc["seal_hash"] = content_hash(doc)` over that text, and to_dict() (line 703) embeds the whole block in the artifact. The same mislabelling repeats in analyse(), where the model selector is called with a literal `transport="gloo_shm"` (line 828), so every entry in campaign.selections is stamped gloo_shm regardless of backend. scripts/gpu_campaign.py:250 constructs Campaign() with no way to say otherwise and writes camp.to_dict() into artifacts/gpu-campaign.json.

*Why invisible locally:* On the development box the strings are true, so nothing reads wrong and no test compares the protocol text against the backend that was used. The backend parameter added to _worker and sweep_pass (campaign.py:224, 391) never reaches Campaign, and Campaign is the only object that writes a protocol.

*Fix:* Add `backend: str = "gloo"` and `fabric: str = ""` fields to Campaign, have scripts/gpu_campaign.py set them (`backend="nccl", fabric=fab["name"]`), and derive the protocol text from them: `"transport": f"{self.backend} over {self.fabric}"`, `"world_sizes": sorted({r.world for r in self.records}) or list(REQUIRED_WORLDS)`, and a transferability clause that is only emitted when self.backend == "gloo". Pass the same value into select_model at line 828 instead of the literal "gloo_shm".

*What a rental would see:* artifacts/gpu-campaign.json carries a `protocol` block, with a content hash over it, stating that the transport was gloo over shared memory on one host and that nothing in it is a value for NVLink or InfiniBand -- sitting directly beside `seal['backend'] = 'nccl'` and a topology block listing eight A100s. Every model_selection entry reports `transport: gloo_shm`. A reader who trusts the sealed protocol over the enclosing script concludes the eight-GPU run produced no fabric parameters; a reader who trusts the script concludes the seal is unreliable. Either way the artifact cannot be used to support a fabric claim, which was the purpose of the rental.

### `src/losscolumn/thrusts/overlap/microbench.py:467`

**run_microbenchmarks has no multi-GPU transport and asserts, unconditionally, that the machine has one GPU**

run_microbenchmarks (lines 436-472) is the only entry point in this module. Its collective measurements go exclusively through measure_gloo with `world=2` (lines 453-454), which spawns CPU-tensor gloo workers (line 270 `torch.ones(elems, dtype=torch.float32)` with no device). There is no NCCL path and no device-count check. After the measurement block it writes two fixed strings: `suite.unavailable["nvlink"] = "requires at least two GPUs on one node; this machine has one"` (lines 466-468). That claim is never derived from torch.cuda.device_count(). The same unconditional framing appears in MicrobenchSuite.to_dict, which always emits `"usable_for_registered_fabric": False` (line 384), in AlphaBetaFit.to_dict, which always emits `"usable_for_fabric": False` (line 133), and in to_markdown lines 423-428 ("**None of these is a value for the registered fabric.** ... neither exists on this machine"). measure_pcie also pins everything to `device="cuda"`, i.e. device 0 only (lines 223-224).

*Why invisible locally:* Every one of those statements is true on a one-GPU workstation, so the hard-coding is indistinguishable from a correct report. The tests exercise fit_alpha_beta and the Point/AlphaBetaFit arithmetic; nothing asserts that the unavailable dict matches the hardware, because on the development box there is only one hardware configuration to match.

*Fix:* Derive the claim: `n = torch.cuda.device_count() if torch.cuda.is_available() else 0`, and only write the nvlink entry when `n < 2`, with the real count in the message. Make `usable_for_fabric` / `usable_for_registered_fabric` a computed property of the transport name (True for an nccl transport on >= 2 devices, False otherwise) rather than a literal False, and gate the to_markdown disclaimer on the same value. If the module is meant to run on the rental at all, add an nccl branch alongside measure_gloo; if it is not, have run_microbenchmarks refuse on a multi-GPU host rather than emit a single-GPU report.

*What a rental would see:* Run on the eight-GPU box, the microbenchmark report states in prose that the machine has one GPU and that no NVLink measurement is possible, while its `device` field -- set from device_description() at line 441, which reads get_device_properties(0) -- names the A100. The two gloo fits it does report are CPU shared-memory numbers presented under an A100 heading, with `usable_for_fabric: False` on both, so the run consumes rented GPU time and returns nothing about the fabric while describing the fabric as absent.

## minor (4)

### `src/losscolumn/core/memory_model.py:228`

**BlockShape.params omits the gate projection, so the declared llama-7b target is 22% short on weights, gradients and optimizer state**

`BlockShape.params` returns `3*h*h + h*h + 2*h*ffn` - a GPT-2 style block with a two-matrix FFN, as its docstring says. `ScalingMap.to_target` (line 510-518) builds a `BlockShape` at `target_hidden=4096, target_ffn=11008, target_layers=32` (set in memcal2.py:316-319), which is the llama-7b geometry, and llama's FFN is gated: three matrices, `3*h*ffn`. The repo already knows this - `ModelSpec.params_per_layer` in simulate.py:70-73 uses `4*h*h + 3*h*hf` for the same named model. The external check settles it: with `4h^2 + 2*h*ffn` the 32-layer total plus embeddings is 5.30B parameters, not 7B; with `3*h*ffn` it is 6.74B, which is what "llama-7b-class" means. So `terms()` understates the target by 1.44B parameters, which at lines 265-267 propagates into `weights` (2.89 GB short at fp16), `gradients` (another 2.89 GB at the registered 2 bytes/param), and `optimizer_state` (17.3 GB short once `optimizer_bytes_per_param` is re-registered to 12 for Adam, which the parameter's own note at lines 96-101 says is the intended value for a real trainer).

*Why invisible locally:* The local calibration block really is two-matrix - `_shape` in memcal2.py:60-65 uses hidden=1024/ffn=4096 and the probe builds a matching block - so every cell in the calibration and validation grids agrees with the formula and the model validates clean. The mismatch only appears on the scaling step to the simulated architecture. It is currently latent rather than live: `to_target` is called only from tests/test_memory_model_v2.py:203, never from a production path. But `ScalingMap` publishes the local-to-7B correspondence and a "width ratio 4x" claim into the artifact (memcal2.py:124, 240), and the first caller that uses it to size a real run gets an understated requirement - the same direction as the false wins that killed version 1 (module docstring, lines 6-9).

*Fix:* Give `BlockShape` an explicit FFN-shape field (e.g. `ffn_matrices: int = 2`, or a `gated: bool`) and use it in `params`: `4*hidden**2 + ffn_matrices*hidden*ffn`. Set it to 3 in `ScalingMap.to_target` for the llama-7b target, and add an assertion that `BlockShape(target).params * n_layers + 2*vocab*hidden` lands within a few percent of the model's published parameter count - that is the external reference the term arithmetic has never been checked against. `ffn_workspace_tensors` (default 2.0, line 79-82) needs the same treatment for a gated FFN, where three tensors are live.

### `src/losscolumn/thrusts/kernel/sparse_sweep.py:149`

**The correctness gate is pinned to a reduced shape by an 8.6 GB constant, and a failure at the timed shape becomes an unexplained hole**

`CHECK_BATCH = 1` / `CHECK_HEADS = 2` are justified in the comment above them by the dev card: the fp64 ground truth at the largest registered cell (batch 8, 8 heads, 4096 tokens) needs 8.6 GB "on an 8.6 GB card". That arithmetic is right (8*8*4096*4096*8 = 8.59e9) and the constraint is real on a T1000 -- and gone on 96 GB, where the full timed shape is ~9% of memory. The constants stay, so the gate keeps validating a shape nobody times. That would be merely conservative except that the timing path swallows exceptions: `time_callable` catches everything (bench.py:92-93 and 115-116) and returns NaN, `paired_ab` appends the NaN, and `latency.put(METHOD, cell, a_ms)` writes it into the envelope with no `mark_missing` call. `is_measured` then returns False because `replicates_at` filters NaN, so the cell is a hole with no reason string -- and `_attribute`'s `if region.contains_missing and region.missing_reasons:` is False, so it falls through and invents a mechanism for a cell that crashed.

*Why invisible locally:* There was no way to observe the gap on the dev card because the full shape genuinely could not be ground-truthed there, and no failure mode that appears only at (batch 8, heads 8) was ever reachable. The published artifact will also keep asserting the 8.6 GB rationale in `CorrectnessSuite.note`, which is carried into `supporting.correctness` -- a limitation attributed to hardware the run is not using.

*Fix:* Size the check shape from available memory rather than from a literal: compute the fp64 score-matrix bytes for the cell and use the full (batch, heads) whenever it fits in, say, a quarter of `torch.cuda.get_device_properties(0).total_memory`, falling back to the reduced shape otherwise, and record which was used. Separately, make `paired_ab`/`run_sparse_sweep` propagate the `TimingResult.error` -- if every replicate is NaN, call `env.mark_missing(METHOD, cell, error)` so the hole carries its reason and `_attribute` stops fabricating one.

*What a rental would see:* Correctness reported as passing at batch=1/heads=2 while the timed batch=8/heads=8 shape is never checked; if it fails during timing, the cell appears in the loss map as a blank with a plausible but invented attribution rather than as a recorded failure.

### `src/losscolumn/thrusts/overlap/campaign.py:364`

**_finalise drops the backend and device_ids the worker deliberately recorded, so no measured point can say which fabric produced it**

_worker builds each row with `"backend": backend` and `"device_ids": list(device_ids[:world])` (campaign.py:323-325) -- added specifically so a number can name the devices it came from. _finalise (lines 362-369) constructs the PointRecord from seven named keys and neither of those is among them; PointRecord (lines 121-141) has no field to hold them, and PointRecord.from_dict (lines 154-157) filters to `cls.__dataclass_fields__`, so the fields could not survive a round-trip even if they were added to the dict. The information is discarded at the boundary between the worker and the record, one function after it was collected.

*Why invisible locally:* There is only one backend and no device set locally: `backend` is always "gloo" and `device_ids` is always empty, so the dropped fields carry no information and their absence changes nothing in any artifact or test. The fields exist only to be meaningful on a machine with more than one fabric.

*Fix:* Add `backend: str = ""` and `device_ids: list[int] = field(default_factory=list)` to PointRecord, populate them in _finalise from the raw row, and include them in to_dict() so they survive the round-trip through from_dict. Then have scripts/gpu_report.py print the device set on each group row rather than only in the per-fabric heading.

*What a rental would see:* scripts/gpu_campaign.py runs three overlapping fabrics -- `uniform` on a homogeneous quad, `one_domain` on the largest NUMA-clean set, `whole_node` across the SYS boundary -- and the module docstring states the reason: "A number that does not say which fabric produced it cannot be compared with one that does." In the artifact, the records themselves say nothing; the fabric is recoverable only from the enclosing `results[<name>]` key. Any downstream consumer that flattens or pools records across fabrics -- which is the natural thing to do with a list of PointRecords -- mixes 1.9 GB/s cross-NUMA points with 4.5 GB/s intra-quad points into one distribution, and the record-level data offers nothing that would reveal it.

### `src/losscolumn/thrusts/overlap/campaign.py:487`

**The peak's runner-up check defaults to 0.0 when there are no distant points, so the "well-defined peak" gate passes vacuously**

`_locate_peak` returns `runner_up = max(far)/bw[i] if far and bw[i] > 0 else 0.0`, where `far` holds only points more than `RUNNER_UP_DISTANCE` (3) grid steps from the argmax. When `far` is empty -- any pass with seven or fewer valid points, or a peak near the middle of a short grid -- the function returns 0.0, meaning "no distant rival", and `analyse_peak` line 544-548 then passes the `runner_up_ratio <= MAX_RUNNER_UP` half of its `well_defined` test with no evidence behind it. Note the asymmetry on the adjacent line: `support` falls back to 0.0 too (line 485), but 0.0 there fails the `>= MIN_NEIGHBOUR_SUPPORT` test, i.e. fail-closed; the identical fallback on line 487 is fail-open. The consequence is that `rep.reproducible` can be set, and the artifact's finding string asserts the peak is "a property of the transport, not of one run".

*Why invisible locally:* The default `build_grid` produces roughly 34 sizes, so `far` is always populated and the branch never runs. It opens on a reduced grid or on a pass where enough points came back invalid to leave seven or fewer -- both of which are how a first pass on rented hardware tends to look.

*Fix:* Return `float('nan')` when `far` is empty and treat a non-finite `runner_up_ratio` as failing `well_defined` in `analyse_peak` (line 544), with a finding that says the grid was too short to test for a distant rival -- matching the fail-closed behaviour the neighbour-support check already has.

*What a rental would see:* A peak report with `runner_up_ratio: 0.0`, `well_defined: true`, `reproducible: true`, and the finding text claiming the bandwidth peak is a property of the transport, computed from a pass too short to have tested the claim.
