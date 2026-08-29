"""``losscolumn`` -- the one command a reproducer runs.

Requirement LC-5 says a conforming claim ships a single command that
regenerates it. That requirement is only credible if this package's own
artifacts are produced that way, so every artifact in ``artifacts/`` comes from
a subcommand here and each claim records the exact invocation that made it.

    losscolumn doctor                 what can this machine actually measure?
    losscolumn prereg seal I          seal a protocol before collecting data
    losscolumn prereg show II         print a sealed protocol
    losscolumn run thrust1|2|3        run a thrust and publish its artifact
    losscolumn run all                run everything and build the synthesis
    losscolumn validate claim.json    grade any claim against the standard
    losscolumn report claim.json      re-render an artifact from its claim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from losscolumn.version import STANDARD_VERSION, __version__

DEFAULT_ARTIFACTS = Path("artifacts")


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what this machine can measure, and what it will have to model.

    Printed before any run because the difference between a measurement and a
    simulation is the difference between two kinds of claim, and the operator
    should see which one they are about to produce.
    """
    from losscolumn.core.provenance import Provenance
    from losscolumn.thrusts.engines.adapters.external import availability
    from losscolumn.thrusts.kernel.triton_fa import triton_available

    prov = Provenance.capture()
    print(f"losscolumn {__version__}  (standard {STANDARD_VERSION})")
    print(f"  platform            {prov.platform}")
    print(f"  python              {prov.python}")
    print(f"  hardware            {prov.describe_hardware()}")
    print(f"  fingerprint         {prov.hardware_fingerprint()}")
    git = prov.git
    print(f"  commit              {(git.get('commit') or 'n/a')[:12]}"
          f"{'  (DIRTY TREE)' if git.get('dirty') else ''}")
    print()
    print("  Thrust I  -- overlap envelope")
    ws = int(prov.gpu.get("devices") and len(prov.gpu["devices"]) or 0)
    if ws >= 2:
        print(f"    measured path available ({ws} GPUs visible); "
              "launch under torchrun for collectives")
    else:
        print(f"    only {ws} GPU visible: the measured path needs a multi-GPU launch.")
        print("    the simulated path is available and is labelled SIMULATED in artifacts.")
    print()
    print("  Thrust II -- engine audit")
    for eng, state in availability().items():
        print(f"    {eng:8s} {state}")
    print("    the simulated path is available and is labelled SIMULATED in artifacts.")
    print()
    print("  Thrust III -- attention kernel")
    ok, why = triton_available()
    print(f"    triton kernel       {'available' if ok else 'unavailable: ' + why}")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"    portable kernel     available (measured on {torch.cuda.get_device_name(0)})")
        else:
            print("    portable kernel     CPU only; timings will not be meaningful")
    except Exception:
        print("    portable kernel     torch not installed")
    return 0


# --------------------------------------------------------------------------
# prereg
# --------------------------------------------------------------------------


def cmd_prereg(args: argparse.Namespace) -> int:
    from losscolumn.core.prereg import PreRegistration, seal
    from losscolumn.protocols import build

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.action == "seal":
        pre = build(args.thrust)
        path = outdir / f"prereg-thrust-{args.thrust.upper()}.json"
        if path.exists() and not args.force:
            print(f"{path} already exists. A sealed protocol is not re-sealed; publish a new "
                  "version with a declared deviation instead. Use --force only to supersede "
                  "it with a genuinely new revision.", file=sys.stderr)
            return 2
        if path.exists():
            # A superseded seal is archived, never deleted: it governed results
            # that are already published, and it has to stay verifiable.
            import json as _json

            old = _json.loads(path.read_text(encoding="utf-8"))
            arch = outdir / "archive"
            arch.mkdir(parents=True, exist_ok=True)
            stamp = str(old.get("sealed_at", "unknown")).replace(":", "").replace("-", "")
            dest = arch / f"prereg-thrust-{args.thrust.upper()}-v{old.get('version', '?')}-{stamp}.json"
            dest.write_text(_json.dumps(old, indent=2), encoding="utf-8")
            print(f"archived the superseded seal to {dest}")
        seal(pre, anchor=args.anchor)
        path.write_text(json.dumps(pre.to_dict(), indent=2), encoding="utf-8")
        (outdir / f"prereg-thrust-{args.thrust.upper()}.md").write_text(
            pre.to_markdown(), encoding="utf-8"
        )
        print(f"sealed  {pre.title}")
        print(f"  hash    {pre.seal_hash}")
        print(f"  at      {pre.sealed_at}")
        print(f"  anchor  {pre.anchor or 'none -- ordering rests on this timestamp alone'}")
        print(f"  written {path}")
        return 0

    if args.action == "show":
        path = outdir / f"prereg-thrust-{args.thrust.upper()}.json"
        if not path.exists():
            print(f"no sealed protocol at {path}; run `losscolumn prereg seal "
                  f"{args.thrust}` first", file=sys.stderr)
            return 2
        pre = PreRegistration.from_dict(json.loads(path.read_text(encoding="utf-8")))
        print(pre.to_markdown())
        ok = pre.compute_hash() == pre.seal_hash
        print(f"\nseal check: {'INTACT' if ok else 'BROKEN -- the protocol was edited'}")
        return 0 if ok else 1

    return 2


def load_prereg(thrust: str, outdir: Path):
    from losscolumn.core.prereg import PreRegistration

    path = Path(outdir) / f"prereg-thrust-{thrust.upper()}.json"
    if not path.exists():
        return None
    return PreRegistration.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from losscolumn.runners import run_all, run_thrust_one, run_thrust_three, run_thrust_two

    outdir = Path(args.outdir)
    prereg_dir = Path(args.prereg_dir)
    fns = {
        "thrust1": run_thrust_one,
        "thrust2": run_thrust_two,
        "thrust3": run_thrust_three,
    }
    if args.what == "all":
        paths = run_all(outdir=outdir, prereg_dir=prereg_dir, quick=args.quick)
    else:
        paths = fns[args.what](outdir=outdir, prereg_dir=prereg_dir, quick=args.quick)
    for k, v in paths.items():
        print(f"  {k:14s} {v}")
    return 0


# --------------------------------------------------------------------------
# validate / report
# --------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    from losscolumn.validate import validate_claim

    worst = 0
    for path in args.claims:
        claim: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        rep = validate_claim(claim)
        print(rep.to_markdown() if args.markdown else _terse(path, rep))
        if args.json:
            Path(args.json).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        worst = max(worst, 0 if rep.ok else 1)
    return worst


def _terse(path: str, rep: Any) -> str:
    lines = [f"{path}: {rep.grade.upper()}  "
             f"({len(rep.failures)} fatal, {len(rep.warnings)} warning)"]
    for f in rep.failures:
        lines.append(f"  FAIL {f.rule:8s} {f.message}")
    for f in rep.warnings:
        lines.append(f"  warn {f.rule:8s} {f.message}")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    """Re-render an artifact from a claim document, including third-party ones."""
    from losscolumn.pipeline import md_to_html
    from losscolumn.validate import validate_claim

    claim = json.loads(Path(args.claim).read_text(encoding="utf-8"))
    rep = validate_claim(claim)
    out = Path(args.out or Path(args.claim).with_suffix(".html"))
    body = [
        f"<title>{claim.get('title', 'claim')}</title>",
        f"<h1>{claim.get('title', 'claim')}</h1>",
        f"<p>{claim.get('headline_rendered', '')}</p>",
        md_to_html(rep.to_markdown()),
    ]
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {out}  ({rep.grade})")
    return 0 if rep.ok else 1


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="losscolumn",
        description="Falsifiable performance claims for LLM training and inference systems.",
    )
    p.add_argument("--version", action="version", version=f"losscolumn {__version__} "
                                                          f"(standard {STANDARD_VERSION})")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="report what this machine can measure")
    d.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("prereg", help="seal or inspect a pre-registered protocol")
    pr.add_argument("action", choices=["seal", "show"])
    pr.add_argument("thrust", help="I, II or III")
    pr.add_argument("--outdir", default=str(DEFAULT_ARTIFACTS / "prereg"))
    pr.add_argument("--anchor", default=None,
                    help="external timestamp anchor (git tag, OSF id, receipt)")
    pr.add_argument("--force", action="store_true",
                    help="replace an unpublished draft; never use on a published seal")
    pr.set_defaults(func=cmd_prereg)

    r = sub.add_parser("run", help="run a thrust and publish its artifact")
    r.add_argument("what", choices=["thrust1", "thrust2", "thrust3", "all"])
    r.add_argument("--outdir", default=str(DEFAULT_ARTIFACTS))
    r.add_argument("--prereg-dir", default=str(DEFAULT_ARTIFACTS / "prereg"))
    r.add_argument("--quick", action="store_true",
                   help="reduced grid for smoke testing; artifacts are labelled as such")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="grade a claim against the standard")
    v.add_argument("claims", nargs="+")
    v.add_argument("--markdown", action="store_true")
    v.add_argument("--json", default=None, help="write the conformance report here")
    v.set_defaults(func=cmd_validate)

    rp = sub.add_parser("report", help="render a claim document to HTML")
    rp.add_argument("claim")
    rp.add_argument("--out", default=None)
    rp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
