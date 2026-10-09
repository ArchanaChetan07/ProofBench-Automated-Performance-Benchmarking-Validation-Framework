# Pre-rental audit, verified

Twenty-two findings from eight agents auditing the bug *classes* the first GPU
rental exposed. Two were fixed on sight. The other twenty were each put to an
independent skeptic instructed to refute by default and to prefer "I could not
confirm this" over a confident guess.

**11 real, 9 refuted** — a 45% false-positive rate, six of them
downgraded to `not-a-bug` outright. Acting on the unverified list would have
sent someone to fix nine pieces of working code, which is its own kind of
damage and the whole reason this pass exists.

## The one already fixed

`reference.py:276` built the sparse baseline's attention mask **inside the**
**timed region** — one CUDA launch per non-zero block, on the baseline arm
alone. Measured at 15–24% of baseline latency against a registered minimum
effect size of 10%, and the cost scales with the layout's occupancy, which is
the sweep's own independent variable, so it bent the speedup-against-density
curve rather than shifting it.

Fixed and pinned by `tests/test_baseline_fairness.py`. The Thrust III numbers
measured before the fix are marked provisional in
[`THRUST3_RESULTS.md`](THRUST3_RESULTS.md).

## Confirmed real, not yet fixed

| severity | location | finding |
|---|---|---|
| `wrong-number` | `campaign.py:378` | `_worker`'s inner `except` (campaign.py:313-315) catches a failure inside one repeat, increments `failures`, records a `reason`, and lets the repeat loop continue, so a row can reach `_final |
| `wrong-number` | `sparse_sweep.py:362` | sparse_sweep.py:358 computes one FLOP count from the block-sparse layout (nnz x block_q x block_k, the padded work the kernel does), then line 361 converts the method's per-replicate times t |
| `wrong-number` | `calibrate.py:452` | `_measured_mfu` (calibrate.py:439-455) computes achieved TFLOP/s by dividing a closed-form FLOP count into the measured wall time of `_block`, then divides by the measured peak GEMM. |
| `wrong-number` | `cliffs.py:166` | Line 166 initialises exposure to 0.0 for every cell in the grid, and thereafter only raises it: max(...) writes at lines 233 and 255 (gated on `d > 0`) and the float("inf") write at line 212 |
| `wrong-number` | `stability.py:418` | `_cv` at C:/Users/archa/OneDrive/Desktop/Sparse Attention/losscolumn/src/losscolumn/core/stability.py:414-418 returns `float(a.std() / a.mean())`. |
| `wrong-number` | `triton_fa.py:416` | flash_attention_triton_sparse (triton_fa.py:398-429) is the sweep's timed callable. |
| `wrong-label` | `runners.py:543` | src/losscolumn/runners.py:541-547 builds the `limitations` list for every thrust3 claim. |
| `wrong-label` | `correctness.py:153` | At correctness.py:152-160, when `spec.causal` is true, line 153 builds `qs = ShapeSpec(**{**spec.to_dict(), "seq_len": spec.seq_len, "seq_k": spec.seq_len})`. |
| `minor` | `microbench.py:467` | `run_microbenchmarks` (src/losscolumn/thrusts/overlap/microbench.py:436-472) is the module's only entry point. |
| `minor` | `campaign.py:487` | src/losscolumn/thrusts/overlap/campaign.py:486-487: far = [bw[j] for j in range(len(bw)) if abs(j - i) > RUNNER_UP_DISTANCE] then runner_up = float(max(far) / bw[i]) if far and bw[i] > 0 els |

### Fixes the verifiers proposed

**`campaign.py:378`** — In `_finalise` (src/losscolumn/thrusts/overlap/campaign.py:362-385), make two minimal changes that match conventions the project already relies on.

**`sparse_sweep.py:362`** — In run_sparse_sweep, compute a second FLOP count for the reference arm from what the reference actually executes, and use it at line 362: for the non-dense patterns the masked SDPA does the full dense score matrix, so `ref_flops = 2*2*bs*heads*seq_q*seq*head_d

**`calibrate.py:452`** — In `_measured_mfu` (calibrate.py:452-453) and `_model_tokens_per_s` (calibrate.py:408-409), make the count match what is actually benchmarked.

**`cliffs.py:166`** — Two independent fixes in src/losscolumn/core/cliffs.py, plus one in the renderer.

**`stability.py:418`** — Change stability.py:418 to `return float(a.std(ddof=1) / a.mean())` and stability.py:196 to `np.std(t, ddof=1)`.

**`triton_fa.py:416`** — Get the layout upload out of the timed callable.

**`runners.py:543`** — Replace the unconditional bullet at src/losscolumn/runners.py:543-547 with a helper that is gated on the measured capability, mirroring the existing `_tensor_core_limitation` pattern at lines 580-600.

**`correctness.py:153`** — In correctness.py, make line 153 build a genuinely degenerate shape and assert the stronger property:      n_pad = max(spec.seq_len // 4, 1)     qs = ShapeSpec(**{**spec.to_dict(), "seq_k": n_pad})     q2, k2, v2 = make_inputs(qs, device=device, seed=seed + 1)

**`microbench.py:467`** — In `run_microbenchmarks`, derive the claim instead of asserting it.

**`campaign.py:487`** — In _locate_peak (src/losscolumn/thrusts/overlap/campaign.py:487) return float("nan") when far is empty, so "not tested" is distinguishable from "no rival": runner_up = float(max(far) / bw[i]) if far and bw[i] > 0 else float("nan").

## Refuted

Kept rather than deleted. A refuted finding is evidence about the auditor, and
the next person to read the same code deserves to know it was already checked
and cleared.

| verdict | location | why not |
|---|---|---|
| `minor` | `torch_backend.py:210` | The audit's description of lines 210-212 is a fair reading of the source, so this is not a hallucinated snippet. |
| `minor` | `modellability.py:95` | Four independent reasons the finding does not survive. |
| `minor` | `sparse_sweep.py:470` | I could not confirm this finding at the severity or for the reason reported. |
| `not-a-bug` | `torch_backend.py:167` | I could not confirm this finding; its load-bearing claim is contradicted by the code, and the path is unreachable as the repo stands. |
| `not-a-bug` | `triton_fa.py:282` | The filed claim is "Every causal TFLOP/s in the Thrust II envelope is therefore understated by 3%-25%", severity wrong-number. |
| `not-a-bug` | `modellability.py:350` | The finding describes line 350's mechanics correctly in isolation -- with kind=="noise_limited" and noise_cv NaN or 0, no plan is built, repeats_needed/launches_needed keep their dataclass d |
| `not-a-bug` | `memory_model.py:228` | The finding's arithmetic is all correct — I checked it: 4h²+2·h·ffn at h=4096/ffn=11008 gives 157,286,400 vs 202,375,168 gated (22.3% short), 5.30B vs 6.74B over 32 layers plus embeddings, 2 |
| `not-a-bug` | `campaign.py:364` | The mechanical half of the finding is accurate: _finalise (campaign.py:364-369) builds PointRecord from seven named keys and ignores raw["backend"] and raw["device_ids"] that _worker put in |
| `not-a-bug` | `sparse_sweep.py:149` | The reduced check shape is a deliberate, documented choice (comment at lines 137-148, plus the invariance argument published in `CorrectnessSuite.note`), and every harmful consequence the fi |

## What it cost

Three runs. The first two died on session limits and returned nothing — the
first after 1.28M tokens and 328 tool calls across eight finders with three
skeptics each. The third, lean enough to survive, judged all twenty.

It found one defect that had already contaminated a published result, and it
stopped nine false leads from becoming nine commits.
