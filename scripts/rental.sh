#!/usr/bin/env bash
# One rental, run end to end, checkpointed after every stage.
#
# The last two rentals were driven by hand. Both lost work: one to a queue
# assembled ad hoc, one to a campaign that wrote its artifact only at the end
# and was stopped two thirds through. Renting by the hour and improvising the
# order is how an expensive machine ends up debugging instead of measuring.
#
# Every stage writes its artifact before the next begins, and every stage is
# skipped if its artifact already exists, so an interrupted rental resumes by
# being run again rather than by being reasoned about.
#
# Order is deliberate. The cheap checks that can invalidate everything run
# first: if the suite fails or the machine is not steady, nothing measured
# afterwards would mean anything, and finding that out in minute three costs
# almost nothing.

set -uo pipefail

ART="${ART:-artifacts}"
LOG="${LOG:-$ART/rental.log}"
QUICK="${QUICK:-}"
mkdir -p "$ART"

say() { printf '\n=== %s ===\n' "$*" | tee -a "$LOG"; }
note() { printf '    %s\n' "$*" | tee -a "$LOG"; }

# Skip a stage whose artifact is already on disk. This is what makes the script
# resumable: a rental that dies at stage 4 is restarted by running it again.
done_already() {
  if [ -f "$1" ]; then note "already present, skipping: $1"; return 0; fi
  return 1
}

stage() {
  local name="$1" artifact="$2"; shift 2
  say "$name"
  if done_already "$artifact"; then return 0; fi
  local t0 rc
  t0=$(date +%s)
  if "$@" >>"$LOG" 2>&1; then rc=0; else rc=$?; fi
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then
    note "FAILED after ${dt}s (exit $rc) -- see $LOG"
    note "later stages continue: a failed stage is a missing result, not a"
    note "reason to waste the remaining rental hours"
    return $rc
  fi
  note "done in ${dt}s -> $artifact"
}

say "machine"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | tee -a "$LOG"
python3 -c "import torch;print(f'torch {torch.__version__} cuda {torch.version.cuda} nccl {torch.cuda.nccl.version()} devices {torch.cuda.device_count()}')" 2>&1 | tee -a "$LOG"
nvidia-smi topo -m 2>/dev/null | head -12 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Cheap and invalidating. If the suite does not pass on this machine, every
# number produced after it is suspect, and knowing that costs two minutes.
# ---------------------------------------------------------------------------
say "test suite"
if python3 -W ignore -m pytest tests/ -q -p no:faulthandler >>"$LOG" 2>&1; then
  note "suite passed"
else
  note "SUITE FAILED -- stopping. Measuring on a machine where the analysis"
  note "layer does not pass would produce artifacts nobody can trust."
  tail -30 "$LOG"
  exit 1
fi

# ---------------------------------------------------------------------------
# Is this machine steady enough to compare anything measured on it? Runs before
# the campaigns because it is the thing that decides whether their sessions may
# be pooled at all.
# ---------------------------------------------------------------------------
stage "stability sentinel" "$ART/stability-machine.json" \
  python3 -W ignore -u scripts/stability.py $QUICK

# ---------------------------------------------------------------------------
# The headline. The registered lattice is 144 cells at 11 replicates and has
# never been run in full: the development machine had 8.6 GB and no tensor
# cores, so it only ever ran in quick mode at two sequence lengths. This is the
# thrust the proposal is actually about.
# ---------------------------------------------------------------------------
stage "thrust III -- sparse attention, full registered lattice" \
  "$ART/thrust3a-portable-attention.claim.json" \
  python3 -W ignore -u -m losscolumn.cli.main run thrust3 --outdir "$ART" $QUICK

# ---------------------------------------------------------------------------
# The memory model is fitted at hidden 1024, sequence 2048-4096, on 8.6 GB. A
# 96 GB card is 11x outside that on the axis that matters most, so every
# prediction made here is an extrapolation until this runs.
# ---------------------------------------------------------------------------
stage "memory model -- recalibration in domain" \
  "$ART/calibration-thrust1-memory-v2.json" \
  python3 -W ignore -u scripts/memcal2.py $QUICK

# ---------------------------------------------------------------------------
# Communication last: it is the best understood of the three and the one whose
# local answer is already clear, so it gets whatever hours remain rather than
# the first ones.
# ---------------------------------------------------------------------------
stage "communication campaign" "$ART/gpu-campaign.json" \
  python3 -W ignore -u scripts/gpu_campaign.py $QUICK

say "summary"
for f in "$ART"/stability-machine.json \
         "$ART"/thrust3a-portable-attention.claim.json \
         "$ART"/calibration-thrust1-memory-v2.json \
         "$ART"/gpu-campaign.json; do
  if [ -f "$f" ]; then
    printf '    %-52s %8s KB\n' "$(basename "$f")" "$(( $(wc -c <"$f") / 1024 ))" | tee -a "$LOG"
  else
    printf '    %-52s %8s\n' "$(basename "$f")" "MISSING" | tee -a "$LOG"
  fi
done

say "collect"
note "tar the artifacts and pull them down before stopping the instance:"
note "  tar czf /workspace/rental-artifacts.tgz -C $(pwd) $ART"
note "  scp -P <port> root@<host>:/workspace/rental-artifacts.tgz ."
