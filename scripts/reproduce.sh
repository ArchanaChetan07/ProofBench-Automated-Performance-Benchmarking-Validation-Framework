#!/usr/bin/env bash
#
# Clean-room reproduction.
#
# Requirement LC-5 says a conforming claim ships one command that regenerates
# it. A requirement nobody has executed is a wish, so this script executes it:
# it clones the repository into a fresh directory, builds a fresh virtual
# environment, installs from the pinned commit, runs the pipeline, and checks
# the artifacts it produces against the ones in the tree it cloned from.
#
# It runs from a CLONE, not from the working tree. That distinction is the
# whole point. Running the pipeline in place proves the pipeline works on a
# machine that already has everything it needs; the failures worth catching are
# uncommitted files, unpinned dependencies, and paths that only exist on the
# author's disk, and none of those are visible from inside the working tree.
#
#   scripts/reproduce.sh                  clone HEAD, quick run, compare
#   scripts/reproduce.sh --full           the full registered protocol
#   scripts/reproduce.sh --ref v1.0.0     reproduce a tagged release
#   scripts/reproduce.sh --keep           leave the clone in place for inspection
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="HEAD"
QUICK="--quick"
KEEP=0
WORK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)  QUICK=""; shift ;;
    --ref)   REF="$2"; shift 2 ;;
    --keep)  KEEP=1; shift ;;
    --work)  WORK="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/losscolumn-repro.XXXXXX")}"
CLONE="$WORK/losscolumn"

cleanup() {
  if [[ $KEEP -eq 0 && -d "$WORK" ]]; then rm -rf "$WORK"; fi
}
trap cleanup EXIT

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

say "1/6  Refusing to reproduce from a dirty tree"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "The working tree has uncommitted changes."
  echo "A reproduction from a dirty tree tests code that no commit identifies,"
  echo "which is the failure LC-5.4 exists to catch. Commit or stash first."
  git -C "$REPO_ROOT" status --short | head -20
  exit 1
fi
COMMIT="$(git -C "$REPO_ROOT" rev-parse "$REF")"
echo "reproducing $COMMIT"

say "2/6  Cloning into a fresh directory"
git clone --quiet --no-hardlinks "$REPO_ROOT" "$CLONE"
git -C "$CLONE" checkout --quiet "$COMMIT"
echo "$CLONE"

say "3/6  Building an isolated environment"
PY="${PYTHON:-python}"
"$PY" -m venv "$WORK/venv"
if [[ -x "$WORK/venv/bin/python" ]]; then VPY="$WORK/venv/bin/python"; else VPY="$WORK/venv/Scripts/python.exe"; fi
"$VPY" -m pip install --quiet --upgrade pip
# Install the package alone first: the analysis layer must be usable with numpy
# and nothing else, and this is where that claim gets tested rather than
# asserted.
"$VPY" -m pip install --quiet -e "$CLONE"
"$VPY" -c "import losscolumn, numpy; print('core imports with numpy alone:', losscolumn.__version__)"

say "4/6  Verifying the frozen standard from the clone"
( cd "$CLONE" && "$VPY" -m losscolumn.cli.main standard export --outdir "$WORK/standard" ) \
  || fail "the standard does not verify from a clean clone"

say "5/6  Running the pipeline"
# Install the test dependencies WITHOUT swallowing the error. This line used to
# end in `|| true`, so a failed install surfaced much later as "No module named
# pytest" -- a silent failure inside the script whose whole job is to catch
# silent failures. Found by running it.
#
# Extras are installed by name rather than through the `[dev]` extra: the
# bracket form is unreliable through a POSIX shell on Windows, where the clone
# path is translated on its way to pip.
# SciPy is optional for the package but required to exercise the log-relative
# and robust estimator families, so the clean-clone run installs it: a
# reproduction that silently tests fewer model families than the author did is
# not a reproduction of the author's result.
"$VPY" -m pip install --quiet pytest scipy \
  || fail "could not install the test dependencies into the clean environment"
"$VPY" -c "import pytest" \
  || fail "pytest did not import after installation"
( cd "$CLONE" && "$VPY" -W ignore -m pytest tests/ -q -p no:faulthandler ) \
  || fail "the test suite does not pass from a clean clone"
( cd "$CLONE" && "$VPY" -W ignore -m losscolumn.cli.main run all $QUICK ) \
  || fail "the pipeline does not run from a clean clone"

say "6/6  Comparing against the tracked artifacts"
"$VPY" - "$REPO_ROOT" "$CLONE" <<'PYEOF'
import json
import pathlib
import sys

tracked, fresh = (pathlib.Path(p) / "artifacts" for p in sys.argv[1:3])
problems, checked = [], 0

for a in sorted(tracked.glob("*.claim.json")):
    b = fresh / a.name
    if not b.exists():
        problems.append(f"{a.name}: not produced by the clean-room run")
        continue
    checked += 1
    ja, jb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))

    # Numbers are not compared. Two runs of a benchmark on the same machine do
    # not produce identical timings, and a reproduction check that demanded
    # they did would fail every time and be switched off. What must reproduce
    # is the STRUCTURE: the same standard, the same factor lattice, the same
    # protocol seal, and the same conformance verdict.
    for field in ("standard_version", "method", "baseline", "evidence_class"):
        if ja.get(field) != jb.get(field):
            problems.append(f"{a.name}: {field} differs ({ja.get(field)} != {jb.get(field)})")

    fa = [f["name"] for f in (ja.get("envelope") or {}).get("factors", [])]
    fb = [f["name"] for f in (jb.get("envelope") or {}).get("factors", [])]
    if fa != fb:
        problems.append(f"{a.name}: factor lattice differs ({fa} != {fb})")

    sa = (ja.get("prereg") or {}).get("normative_hash")
    sb = (jb.get("prereg") or {}).get("normative_hash")
    if sa != sb:
        problems.append(f"{a.name}: protocol seal differs ({sa} != {sb})")

print(f"compared {checked} claim(s)")
if problems:
    print("\nSTRUCTURAL DIFFERENCES:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("every tracked claim reproduces structurally from a clean clone")
PYEOF

say "Clean-room reproduction succeeded"
echo "commit    $COMMIT"
echo "clone     $CLONE"
[[ $KEEP -eq 1 ]] && echo "(kept for inspection; delete $WORK when done)"
exit 0
