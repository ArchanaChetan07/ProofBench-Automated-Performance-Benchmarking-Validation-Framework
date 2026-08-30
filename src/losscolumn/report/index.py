"""The artifact tree, and the index that makes it navigable.

The repository holds three tiers of artifact, and they are different kinds of
thing:

``standard/``       The frozen standard: schemas, the rule registry, the
                    coverage matrix, the adversarial corpus. Generated from the
                    registry, not maintained beside it.
``protocols/``      One directory per thrust: the sealed protocol, the claims
                    it governs, their conformance reports and their rendered
                    artifacts.
``reproduction/``   The record of a clean-room run: what a third party gets
                    when they clone and execute the one command.

Keeping them apart matters because they answer different questions. The
standard is what you must agree to before a claim means anything; a protocol is
what was committed to before data existed; a reproduction is whether any of it
survives contact with a machine that is not the author's.

The index is generated rather than written, so it cannot drift from the files
it describes -- an index that lists an artifact nobody produced is exactly the
kind of unfalsifiable furniture this project exists to argue against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from losscolumn.core.provenance import utcnow
from losscolumn.history import load, load_incidents, verify
from losscolumn.spec import registry, schema

THRUST_TITLES = {
    "I": "Thrust I -- the overlap envelope",
    "II": "Thrust II -- the equal-tuning audit",
    "III": "Thrust III -- sparse attention, prefill and decode",
}


@dataclass
class ClaimEntry:
    path: Path
    id: str = ""
    title: str = ""
    thrust: str = ""
    evidence: str = ""
    grade: str = ""
    n_fatal: int = 0
    n_warn: int = 0
    method: str = ""
    baseline: str = ""
    loss_cells: int = 0
    n_cells: int = 0
    headline: str = ""
    siblings: dict[str, Path] = field(default_factory=dict)


def _load_claims(root: Path) -> list[ClaimEntry]:
    """Live claims only.

    Withdrawn artifacts live under ``history/`` and are deliberately excluded.
    They are preserved so the record of an error survives, not so they can be
    counted, graded or cited -- and a recursive glob that swept them back into
    the claim list would undo the whole point of withdrawing them.
    """
    out: list[ClaimEntry] = []
    history = history_dir_of(root)
    for p in sorted(root.rglob("*.claim.json")):
        if history in p.parents:
            continue
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        stem = p.name[: -len(".claim.json")]
        e = ClaimEntry(
            path=p,
            id=c.get("id", stem),
            title=c.get("title", stem),
            thrust=str(c.get("thrust", "")),
            evidence=c.get("evidence_class", "?"),
            method=c.get("method", ""),
            baseline=c.get("baseline", ""),
            headline=(c.get("headline") or {}).get("statement", ""),
        )
        lc = c.get("loss_column") or {}
        e.loss_cells = int((lc.get("counts") or {}).get("loss", 0))
        e.n_cells = int(lc.get("n_cells", 0))
        conf = p.with_name(f"{stem}.conformance.json")
        if conf.exists():
            try:
                r = json.loads(conf.read_text(encoding="utf-8"))
                e.grade = r.get("grade", "")
                e.n_fatal = int(r.get("n_fatal", 0))
                e.n_warn = int(r.get("n_warnings", 0))
            except Exception:
                pass
        for ext in (".md", ".html", ".conformance.json"):
            q = p.with_name(f"{stem}{ext}")
            if q.exists():
                e.siblings[ext.lstrip(".")] = q
        out.append(e)
    return out


def history_dir_of(root: Path) -> Path:
    return Path(root) / "history"


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.name


def build_index(root: Path) -> str:
    """Render INDEX.md for an artifact tree."""
    root = Path(root)
    claims = _load_claims(root)
    by_thrust: dict[str, list[ClaimEntry]] = {}
    for c in claims:
        by_thrust.setdefault(c.thrust or "?", []).append(c)

    conforming = sum(1 for c in claims if c.grade == "conforming")
    measured = sum(1 for c in claims if c.evidence == "measured")

    L: list[str] = [
        "# Artifact index",
        "",
        f"Generated {utcnow()} by `losscolumn index`. Do not edit: an index "
        "maintained by hand drifts from the files it describes, and an index that "
        "lists an artifact nobody produced is the kind of unfalsifiable furniture "
        "this project exists to argue against.",
        "",
        f"**{len(claims)} claims** &mdash; {conforming} conforming, "
        f"{measured} measured, {len(claims) - measured} simulated.",
        "",
        "---",
        "",
        "## STANDARD",
        "",
        f"LC-{registry.compute_digest()[7:19]}... &mdash; **{registry.N_RULES} rules**, "
        f"{registry.N_FATAL} fatal.",
        "",
    ]

    std = root / "standard"
    if std.exists():
        L += [
            "| Document | What it is |",
            "|---|---|",
            f"| [`rules.md`]({_rel(std / 'rules.md', root)}) | Every rule, its severity, "
            "and the failure mode it forecloses |",
            f"| [`validator-coverage.md`]({_rel(std / 'validator-coverage.md', root)}) | "
            "Which rules have a passing case and a failing case |",
            f"| [`adversarial-corpus.json`]({_rel(std / 'adversarial-corpus.json', root)}) | "
            "Deliberately defective artifacts, for checking another implementation |",
            f"| [`schema/`]({_rel(std / 'schema', root)}) | The four frozen document "
            "shapes as JSON Schema |",
            "",
            f"Registry digest `{registry.compute_digest()}`  ",
            f"Schema digest `{schema.schema_digest()}`",
            "",
            "Both are asserted by `tests/test_standard_frozen.py`. Changing a rule id "
            "or a severity fails that test, which makes a version bump a deliberate "
            "act rather than an edit that slips through.",
            "",
        ]
    else:
        L += ["_Not exported. Run `losscolumn standard export`._", ""]

    L += ["---", "", "## PROTOCOLS", ""]
    prereg_dir = root / "prereg"
    for t in ("I", "II", "III"):
        entries = by_thrust.get(t, [])
        L += [f"### {THRUST_TITLES[t]}", ""]

        seal = prereg_dir / f"prereg-thrust-{t}.json"
        if seal.exists():
            try:
                s = json.loads(seal.read_text(encoding="utf-8"))
                L += [
                    f"**Registered protocol** &mdash; [`{seal.name}`]({_rel(seal, root)}) "
                    f"(revision {s.get('version', '1')}), sealed {s.get('sealed_at')}  ",
                    f"MDE {float(s.get('mde', 0)):.0%} &middot; q {s.get('q_level')} "
                    f"&middot; {s.get('replicates')} replicates &middot; "
                    f"seal `{str(s.get('seal_hash', ''))[7:23]}...`",
                    "",
                ]
                arch = sorted((prereg_dir / "archive").glob(f"prereg-thrust-{t}-*.json"))
                if arch:
                    names = ", ".join(f"[`{a.name}`]({_rel(a, root)})" for a in arch)
                    L += [f"Superseded and still verifiable: {names}", ""]
            except Exception:
                pass
        else:
            L += ["_No sealed protocol._", ""]

        if entries:
            L += ["| Claim | Evidence | Grade | Loss cells | Artifacts |",
                  "|---|---|---|---|---|"]
            for c in entries:
                links = " ".join(
                    f"[{k}]({_rel(v, root)})" for k, v in sorted(c.siblings.items())
                )
                grade = c.grade or "?"
                if c.n_warn:
                    grade += f" ({c.n_warn}w)"
                L.append(
                    f"| **{c.title}**<br><code>{c.method}</code> vs "
                    f"<code>{c.baseline}</code> | {c.evidence} | {grade} | "
                    f"{c.loss_cells}/{c.n_cells} | {links} |"
                )
            L.append("")
        else:
            L += ["_No claim published._", ""]

    cal = sorted(root.glob("calibration-*.json"))
    if cal:
        L += ["---", "", "## CALIBRATION", "",
              "Evidence *about* a claim rather than a claim. A simulated thrust "
              "predicts a loss map; a calibration study measures a subset of it and "
              "reports how far the prediction can be trusted.", ""]
        for p in cal:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                v = (d.get("calibration") or {}).get("verdict", "")
                L.append(f"- [`{p.name}`]({_rel(p, root)}) &mdash; {v}")
            except Exception:
                L.append(f"- [`{p.name}`]({_rel(p, root)})")
        L.append("")

    withdrawn = load(root)
    incidents = load_incidents(root)
    if withdrawn or incidents:
        L += ["---", "", "## WITHDRAWN AND DISCARDED", "",
              "Preserved rather than deleted. Nothing in this section is a claim, and "
              "nothing here carries a conformance grade: grading a withdrawn artifact "
              "would invite it to be cited.", ""]
        problems = verify(root)
        L += [
            f"Integrity: **{'all preserved files match their withdrawal digests' if not problems else str(len(problems)) + ' PRESERVED FILE(S) HAVE CHANGED'}**.",
            "",
        ]
        if problems:
            L += [f"- {p}" for p in problems] + [""]
        if withdrawn:
            L += ["| Withdrawn | Reason | Superseded by | Record |", "|---|---|---|---|"]
            for d, w in withdrawn:
                L.append(
                    f"| {w.title} | `{w.reason}` | `{w.superseded_by or '-'}` | "
                    f"[NOTE.md]({_rel(d / 'NOTE.md', root)}) |"
                )
            L.append("")
        if incidents:
            L += ["**Discarded runs** &mdash; no artifact was published, so there is "
                  "nothing to withdraw; what is recorded is that the run happened and "
                  "what changed as a result.", ""]
            for i in incidents:
                L.append(
                    f"- [{i.title}]({_rel(history_dir_of(root) / 'incidents' / (i.incident_id + '.md'), root)})"
                    f" &mdash; {i.occurred_at}"
                )
            L.append("")

    L += [
        "---",
        "",
        "## REPRODUCTION",
        "",
        "```bash",
        "scripts/reproduce.sh --full",
        "```",
        "",
        "Clones the repository into a fresh directory, builds an isolated "
        "environment, installs from the pinned commit, runs the pipeline, and "
        "compares what comes out against what is tracked here.",
        "",
        "It runs from a clone rather than the working tree on purpose. Running in "
        "place proves the pipeline works on a machine that already has everything "
        "it needs; the failures worth catching -- uncommitted files, unpinned "
        "dependencies, paths that exist only on one disk -- are invisible from "
        "inside the tree.",
        "",
        "Structure is compared, not numbers. Two benchmark runs on the same machine "
        "never produce identical timings, and a check that demanded they did would "
        "fail every time and be switched off. What must reproduce is the standard "
        "version, the factor lattice, the protocol seal and the conformance verdict.",
        "",
    ]
    return "\n".join(L)


def write_index(root: Path) -> Path:
    root = Path(root)
    p = root / "INDEX.md"
    p.write_text(build_index(root), encoding="utf-8")
    return p


def summarise(root: Path) -> dict[str, Any]:
    claims = _load_claims(Path(root))
    return {
        "n_withdrawn": len(load(Path(root))),
        "n_incidents": len(load_incidents(Path(root))),
        "history_intact": not verify(Path(root)),
        "n_claims": len(claims),
        "n_conforming": sum(1 for c in claims if c.grade == "conforming"),
        "n_measured": sum(1 for c in claims if c.evidence == "measured"),
        "claims": [
            {"id": c.id, "thrust": c.thrust, "grade": c.grade,
             "evidence": c.evidence, "loss_cells": c.loss_cells, "n_cells": c.n_cells}
            for c in claims
        ],
    }
