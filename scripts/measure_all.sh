#!/usr/bin/env bash
# Every measurement this project publishes, in series.
#
# Series is not incidental. Two benchmark processes on one device contend at
# the scheduler and in cache, and produce timings that are wrong by an amount
# nothing downstream can detect. The measurement lock enforces this, and this
# script is the supported way to get it right.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Thrust III: the registered sparse-attention lattice ==="
python -W ignore -m losscolumn.cli.main run thrust3

echo
echo "=== Thrust I: compute-model calibration ==="
python -W ignore -m losscolumn.cli.main calibrate thrust1

echo
echo "=== validate every claim ==="
for f in artifacts/*.claim.json; do
  python -W ignore -m losscolumn.cli.main validate "$f" || true
done
