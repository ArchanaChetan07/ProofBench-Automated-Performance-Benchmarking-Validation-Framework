"""Validate the one memory-model term a single-layer grid cannot test.

block-memory-v2 scores 8 of 8 on its calibration grid, and that result says
nothing about checkpointing. The grid's block has one layer, and checkpointing
one layer saves nothing at the peak: torch.utils.checkpoint discards the
intermediates during forward and recomputes them during backward, reaching the
same high-water mark it would have reached by keeping them. The measured peak
for ckpt=True and ckpt=False is identical to the byte -- 5477763072 in both --
while the model predicts a 15% saving.

The model is not obviously wrong for it. Its saved-activation term scales with
layer count and checkpointing multiplies that term by a registered
`checkpoint_retained_fraction` of 0.25, which is a claim about what a stack of
blocks retains, not about what one block does. A stack of N blocks under
checkpointing stores N boundary tensors and recomputes one block's intermediates
at a time, so the saving is real and grows with N.

None of which has been measured. This measures it, across n_layers, with the
parameters frozen: the model gets no opportunity to absorb the result.
"""
import json
import sys
from pathlib import Path

LOCAL_HIDDEN = 1024
LOCAL_HEADS = 16
LOCAL_FFN = 4096

# Layer counts, chosen so the prediction separates. At one layer the model
# claims a saving that cannot be realised; if the term is right, agreement
# should improve as the stack deepens and the boundary tensors come to dominate.
LAYER_COUNTS = (1, 2, 4, 8)


def predictions(n_layers, mb, seq) -> dict:
    """Sealed before measurement, as the standard requires."""
    from losscolumn.core.memory_model import BlockMemoryModel, BlockShape

    model = BlockMemoryModel()
    from losscolumn.core.provenance import content_hash

    rows = []
    for n in n_layers:
        shape_off = BlockShape(hidden=LOCAL_HIDDEN, heads=LOCAL_HEADS, ffn=LOCAL_FFN,
                               micro_batch=mb, seq_len=seq, n_layers=n,
                               bytes_per_elem=2, checkpointing=False)
        shape_on = BlockShape(hidden=LOCAL_HIDDEN, heads=LOCAL_HEADS, ffn=LOCAL_FFN,
                              micro_batch=mb, seq_len=seq, n_layers=n,
                              bytes_per_elem=2, checkpointing=True)
        off = model.predict_bytes(shape_off)
        on = model.predict_bytes(shape_on)
        rows.append({
            "n_layers": n,
            "predicted_bytes_no_checkpoint": off,
            "predicted_bytes_checkpoint": on,
            "predicted_saving_fraction": (off - on) / off if off else 0.0,
        })

    body = {
        "name": "memory-v2-checkpointing-validation-v1",
        "question": (
            "The saved-activation term is multiplied by a registered "
            "checkpoint_retained_fraction of 0.25. Does the measured saving "
            "match, and does agreement improve with depth?"
        ),
        "why_this_grid": (
            "The existing calibration grid uses one layer, where checkpointing "
            "cannot save anything at the peak because backward recomputes to the "
            "same high-water mark. Its 8-of-8 result is therefore silent about "
            "this term. Depth is the axis that makes the term observable."
        ),
        "shape": {"hidden": LOCAL_HIDDEN, "heads": LOCAL_HEADS, "ffn": LOCAL_FFN,
                  "micro_batch": mb, "seq_len": seq, "dtype": "bfloat16"},
        "layer_counts": list(n_layers),
        "cells": rows,
        "parameters_are_frozen": True,
        "no_parameter_will_be_fitted": (
            "Every version 2 parameter is REGISTERED, derived from what a block "
            "allocates. This validates; it does not calibrate, and a result that "
            "disagrees is a finding rather than an input."
        ),
        "predictions": [
            {
                "id": "P1-one-layer-saves-nothing",
                "claim": (
                    "At n_layers=1 the measured peak is the same with and without "
                    "checkpointing, within 2%, because backward recomputes what "
                    "forward discarded."
                ),
                "falsified_if": "the measured peaks differ by more than 2%",
            },
            {
                "id": "P2-saving-grows-with-depth",
                "claim": (
                    "The measured saving fraction increases monotonically from "
                    "n_layers=1 to n_layers=8, because more of the stack's "
                    "activations are discarded and recomputed one block at a time."
                ),
                "falsified_if": (
                    "the measured saving does not increase with depth, which would "
                    "mean checkpointing is not doing what the term describes"
                ),
            },
            {
                "id": "P3-term-matches-at-depth",
                "claim": (
                    "At n_layers=8 the model's predicted saving fraction is within "
                    "10 percentage points of the measured one."
                ),
                "falsified_if": (
                    "it is not, which would put a number on how wrong "
                    "checkpoint_retained_fraction is and is the point of running this"
                ),
            },
        ],
        "committed_before_measurement": True,
    }
    body["seal_hash"] = content_hash(body)
    return body


def _measure(n_layers: int, mb: int, seq: int, ckpt: bool, dtype) -> dict:
    """Peak device bytes for a stack of blocks, forward and backward."""
    import torch
    from torch.utils.checkpoint import checkpoint

    from losscolumn.thrusts.overlap.calibrate import _block

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        blocks = [_block(LOCAL_HIDDEN, LOCAL_FFN, LOCAL_HEADS, dtype, "cuda")
                  for _ in range(n_layers)]
        x = torch.randn(mb, seq, LOCAL_HIDDEN, device="cuda", dtype=dtype,
                        requires_grad=True)
        h = x
        for blk in blocks:
            # Checkpointing is applied per block, which is what the term
            # describes: the stack keeps a boundary tensor per layer and
            # recomputes one layer's intermediates at a time.
            h = checkpoint(blk, h, use_reentrant=False) if ckpt else blk(h)
        h.sum().backward()
        peak = int(torch.cuda.max_memory_allocated(0))
        del blocks, x, h
        torch.cuda.empty_cache()
        return {"status": "ok", "peak_bytes": peak}
    except RuntimeError as e:
        torch.cuda.empty_cache()
        text = str(e).lower()
        if "out of memory" in text:
            return {"status": "oom", "peak_bytes": None, "detail": str(e)[:160]}
        return {"status": "invalid", "peak_bytes": None, "detail": str(e)[:160]}


def main() -> int:
    import torch

    from losscolumn.core.provenance import utcnow
    from losscolumn.thrusts.kernel.bench import MeasurementLock, device_description
    from losscolumn.thrusts.overlap.probe import install_budget
    from losscolumn.version import STANDARD_VERSION

    quick = "--quick" in sys.argv
    mb, seq = (8, 1024) if quick else (16, 2048)
    layers = (1, 2) if quick else LAYER_COUNTS

    art = Path("artifacts")
    (art / "prereg").mkdir(parents=True, exist_ok=True)
    pred = predictions(layers, mb, seq)
    (art / "prereg" / "memory-v2-checkpointing.protocol.json").write_text(
        json.dumps(pred, indent=2, default=str), encoding="utf-8")
    print(f"sealed {pred['seal_hash'][:23]} before measurement")
    print(f"shape hidden={LOCAL_HIDDEN} mb={mb} seq={seq}, layers {list(layers)}",
          flush=True)

    cfg = install_budget()
    if not cfg.cap_installed:
        print(f"REFUSED: {cfg.refusal or 'the memory cap could not be installed'}")
        return 1

    dtype = torch.bfloat16
    rows = []
    with MeasurementLock("memory checkpointing validation"):
        for n in layers:
            cell = {"n_layers": n}
            for ckpt in (False, True):
                r = _measure(n, mb, seq, ckpt, dtype)
                cell["checkpoint" if ckpt else "no_checkpoint"] = r
                print(f"  layers={n} ckpt={str(ckpt):5s} {r['status']:8s} "
                      f"{(r['peak_bytes'] or 0) / 1e9:6.2f} GB", flush=True)
            rows.append(cell)

    md = _report(rows, pred, device_description("cuda"))
    print()
    print(md)

    payload = {
        "kind": "memory-checkpointing-validation",
        "standard_version": STANDARD_VERSION, "generated_at": utcnow(),
        "protocol": pred, "device": device_description("cuda"),
        "runtime": cfg.to_dict(), "cells": rows,
    }
    (art / "validation-memory-v2-checkpointing.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (art / "validation-memory-v2-checkpointing.md").write_text(md, encoding="utf-8")
    print(f"\nwrote {art / 'validation-memory-v2-checkpointing.json'}")
    return 0


def _report(rows, pred, device) -> str:
    by_n = {c["n_layers"]: c for c in pred["cells"]}
    L = ["# Checkpointing, measured across depth", "",
         f"`{pred['name']}` sealed `{pred['seal_hash'][:23]}` before measurement, "
         f"on {device}.", "",
         "The existing calibration grid scores 8 of 8 and is silent about this "
         "term: its block has one layer, and checkpointing one layer saves "
         "nothing at the peak because backward recomputes what forward "
         "discarded. Depth is what makes the term observable.", "",
         "| layers | measured off | measured on | measured saving | predicted saving | gap |",
         "|---|---|---|---|---|---|"]
    measured = []
    for c in rows:
        off = (c.get("no_checkpoint") or {}).get("peak_bytes")
        on = (c.get("checkpoint") or {}).get("peak_bytes")
        p = by_n.get(c["n_layers"], {}).get("predicted_saving_fraction")
        if off and on:
            sav = (off - on) / off
            measured.append((c["n_layers"], sav, p))
            gap = f"{(sav - p) * 100:+.1f} pp" if p is not None else "—"
            L.append(f"| {c['n_layers']} | {off / 1e9:.2f} GB | {on / 1e9:.2f} GB | "
                     f"{sav:.1%} | {p:.1%} | {gap} |")
        else:
            L.append(f"| {c['n_layers']} | — | — | not measured | "
                     f"{p:.1%} | — |" if p is not None else
                     f"| {c['n_layers']} | — | — | not measured | — | — |")
    L.append("")

    if measured:
        one = [m for m in measured if m[0] == 1]
        if one:
            sav = one[0][1]
            held = abs(sav) <= 0.02
            L += [f"**P1 {'HELD' if held else 'FALSIFIED'}.** At one layer the "
                  f"measured saving is {sav:.1%}. " +
                  ("Checkpointing a single block cannot reduce the peak, and does "
                   "not." if held else
                   "That is outside the 2% the prediction allowed, so a single "
                   "block does change its peak under checkpointing."), ""]
        savs = [m[1] for m in measured]
        rising = all(b >= a - 0.01 for a, b in zip(savs, savs[1:], strict=False))
        L += [f"**P2 {'HELD' if rising else 'FALSIFIED'}.** The measured saving "
              f"runs {' -> '.join(f'{s:.1%}' for s in savs)} across "
              f"{len(savs)} depths. " +
              ("It grows with depth, which is what the term describes."
               if rising else
               "It does not grow with depth, so checkpointing is not doing what "
               "the term says it does."), ""]
        deep = max(measured, key=lambda m: m[0])
        if deep[2] is not None:
            gap = abs(deep[1] - deep[2])
            L += [f"**P3 {'HELD' if gap <= 0.10 else 'FALSIFIED'}.** At "
                  f"{deep[0]} layers the model predicts a {deep[2]:.1%} saving "
                  f"and {deep[1]:.1%} was measured, a gap of {gap * 100:.1f} "
                  "percentage points. " +
                  ("The registered fraction describes this machine."
                   if gap <= 0.10 else
                   "That gap is what the parameter is wrong by, and it is a "
                   "finding rather than an input: nothing here is refitted."), ""]
    L += ["No parameter was fitted. Every version 2 parameter is registered, so "
          "this validates the model without conferring anything on it.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
