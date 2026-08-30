"""Entry point for the communication microbenchmarks.

A real file with a __main__ guard, because the gloo collectives are measured in
spawned worker processes and the spawn start method re-imports the parent's
__main__ in each child. Run from `python -c`, there is nothing to re-import and
the children never start.
"""
import json
import sys
from pathlib import Path


def main() -> int:
    from losscolumn.thrusts.overlap.microbench import run_microbenchmarks

    quick = "--quick" in sys.argv
    suite = run_microbenchmarks(quick=quick)
    print(suite.to_markdown())
    out = Path("artifacts") / "microbench-communication.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite.to_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
