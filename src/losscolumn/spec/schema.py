"""Frozen document schemas for LC-1.0.

Four document shapes are frozen at v1.0: the measurement envelope, the loss
column, the claim, and the pre-registration seal. Freezing them is what lets
one person's tooling grade another person's artifact -- the validator reads
documents, not objects, and a document shape that drifts silently breaks every
consumer downstream.

The schemas are expressed as JSON Schema (draft 2020-12) and shipped as data,
so a consumer in another language can use them. Validation here is a small
self-contained checker rather than a dependency: ``losscolumn.core`` and the
validator must stay installable with numpy alone, and the subset of JSON Schema
these documents use is narrow enough that implementing it is cheaper than
taking a dependency that would have to be present wherever anyone wants to
check an artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from losscolumn.core.provenance import content_hash

SCHEMA_VERSION = "LC-1.1"

# Every revision whose claims this implementation can read. A 1.0 claim must go
# on validating: the revision that superseded it did not change what it said.
KNOWN_STANDARD_VERSIONS = ("LC-1.0", "LC-1.1")

# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------

FACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "levels"],
    "properties": {
        "name": {"type": "string"},
        "levels": {"type": "array", "minItems": 1},
        "ordered": {"type": "boolean"},
        "unit": {"type": ["string", "null"]},
        "log_scale": {"type": "boolean"},
    },
}

METRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "unit", "higher_is_better"],
    "properties": {
        "name": {"type": "string"},
        "unit": {"type": "string"},
        "higher_is_better": {"type": "boolean"},
        "description": {"type": ["string", "null"]},
    },
}

ENVELOPE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://losscolumn.org/schema/LC-1.0/envelope.json",
    "title": "Measurement envelope",
    "type": "object",
    "required": ["kind", "factors", "metric", "systems", "replicates"],
    "properties": {
        "kind": {"const": "envelope"},
        "factors": {"type": "array", "minItems": 1, "items": FACTOR_SCHEMA},
        "metric": METRIC_SCHEMA,
        "systems": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "replicates": {"type": "integer", "minimum": 1},
        # Interleaving is required information, not an optional nicety: paired
        # statistics are invalid without it and the reader cannot tell from the
        # numbers alone.
        "interleaved": {"type": "boolean"},
        "workload": {"type": ["string", "null"]},
        "data": {"type": "object"},
        "missing": {"type": "object"},
        "coverage": {"type": "object"},
        "meta": {"type": "object"},
        "provenance": {"type": ["object", "null"]},
    },
}

LOSS_REGION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["bounds", "n_cells", "n_loss", "worst_regression_pct"],
    "properties": {
        "bounds": {"type": "object"},
        "n_cells": {"type": "integer", "minimum": 1},
        "n_loss": {"type": "integer", "minimum": 0},
        "frac_of_envelope": {"type": "number"},
        "median_regression_pct": {"type": "number"},
        "worst_regression_pct": {"type": "number"},
        "max_q": {"type": ["number", "null"]},
        "attribution": {"type": ["string", "null"]},
        "missing_reasons": {"type": "array", "items": {"type": "string"}},
        "label": {"type": "string"},
    },
}

LOSS_COLUMN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://losscolumn.org/schema/LC-1.0/loss-column.json",
    "title": "Loss column",
    "type": "object",
    "required": ["metric", "method", "baseline", "mde_pct", "q_level", "regions",
                 "counts", "n_cells", "design_check"],
    "properties": {
        "metric": {"type": "string"},
        "unit": {"type": "string"},
        "method": {"type": "string"},
        "baseline": {"type": "string"},
        "mde_pct": {"type": "number", "exclusiveMinimum": 0},
        "q_level": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "regions": {"type": "array", "items": LOSS_REGION_SCHEMA},
        "counts": {
            "type": "object",
            "required": ["loss", "win", "tie", "inconclusive", "missing"],
            "properties": {k: {"type": "integer", "minimum": 0}
                           for k in ("loss", "win", "tie", "inconclusive", "missing")},
        },
        "n_cells": {"type": "integer", "minimum": 1},
        # The power check is required, not optional. An empty loss column from
        # a design that could not have found one is the failure mode LC-1.8
        # exists to catch, and it is uncatchable if this may be absent.
        "design_check": {
            "type": "object",
            "required": ["can_reject", "replicates", "n_tests"],
            "properties": {
                "can_reject": {"type": "boolean"},
                "replicates": {"type": "integer", "minimum": 1},
                "n_tests": {"type": "integer", "minimum": 0},
                "min_replicates": {"type": "integer"},
                "p_floor": {"type": "number"},
                "message": {"type": "string"},
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
        "n_unrunnable": {"type": "integer", "minimum": 0},
    },
}

SEAL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://losscolumn.org/schema/LC-1.0/preregistration.json",
    "title": "Pre-registration seal",
    "type": "object",
    "required": ["title", "hypotheses", "metrics", "factors", "systems", "mde",
                 "q_level", "replicates", "sealed_at", "seal_hash", "normative_hash"],
    "properties": {
        "title": {"type": "string"},
        "hypotheses": {"type": "object"},
        "metrics": {"type": "array", "minItems": 1, "items": METRIC_SCHEMA},
        "factors": {"type": "array", "minItems": 1, "items": FACTOR_SCHEMA},
        "systems": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "mde": {"type": "number", "exclusiveMinimum": 0},
        "q_level": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "replicates": {"type": "integer", "minimum": 1},
        "interleaved": {"type": "boolean"},
        "decision_rules": {"type": "array", "items": {"type": "string"}},
        "exclusion_rules": {"type": "array", "items": {"type": "string"}},
        "stopping_rule": {"type": "string"},
        "sealed_at": {"type": ["string", "null"]},
        "seal_hash": {"type": ["string", "null"]},
        "normative_hash": {"type": "string"},
        "anchor": {"type": ["string", "null"]},
        "deviations": {"type": "array"},
    },
}

CLAIM_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://losscolumn.org/schema/LC-1.0/claim.json",
    "title": "Conforming performance claim",
    "type": "object",
    "required": ["kind", "standard_version", "id", "title", "method", "baseline",
                 "evidence_class", "headline", "loss_column", "reproduction"],
    "properties": {
        "kind": {"const": "claim"},
        "standard_version": {"enum": list(KNOWN_STANDARD_VERSIONS)},
        "id": {"type": "string", "minLength": 1},
        "title": {"type": "string", "minLength": 1},
        "thrust": {"type": "string"},
        "method": {"type": "string"},
        "baseline": {"type": "string"},
        "evidence_class": {"enum": ["measured", "simulated", "mixed"]},
        "evidence_note": {"type": "string"},
        "loss_column_placement": {"enum": ["main", "appendix"]},
        "headline": {
            "type": "object",
            "required": ["statement", "best_pct", "worst_pct", "median_pct"],
            "properties": {
                "statement": {"type": "string", "minLength": 1},
                "best_pct": {"type": "number"},
                "worst_pct": {"type": "number"},
                "median_pct": {"type": "number"},
                "conditions_best": {"type": "object"},
                "conditions_worst": {"type": "object"},
            },
        },
        "loss_column": LOSS_COLUMN_SCHEMA,
        "reproduction": {
            "type": "object",
            "required": ["command", "hardware"],
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "repo": {"type": ["string", "null"]},
                "commit": {"type": ["string", "null"]},
                "dirty": {"type": ["boolean", "null"]},
                "container_image": {"type": ["string", "null"]},
                "hardware": {"type": "string"},
                "estimated_runtime_min": {"type": ["number", "null"]},
            },
        },
        "envelope": {"type": ["object", "null"]},
        "prereg": {"type": ["object", "null"]},
        "parity": {"type": ["object", "null"]},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "supporting": {"type": "object"},
        "digest": {"type": "string"},
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "envelope": ENVELOPE_SCHEMA,
    "loss_column": LOSS_COLUMN_SCHEMA,
    "claim": CLAIM_SCHEMA,
    "preregistration": SEAL_SCHEMA,
}


def schema_digest() -> str:
    """Content hash over all four schemas. Frozen at v1.0."""
    return content_hash(SCHEMAS)


# LC-1.1: standard_version widened from a const to an enum so LC-1.0
# claims keep validating. No other shape changed.
FROZEN_SCHEMA_DIGEST = "sha256:49729bd2610b893b183ba5aa27a7df0f7a6319ea496c9265ae172b43a595f529"


# --------------------------------------------------------------------------
# a small validator for the subset of JSON Schema these documents use
# --------------------------------------------------------------------------

_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, spec: Any) -> bool:
    names = spec if isinstance(spec, list) else [spec]
    for n in names:
        t = _TYPES.get(n)
        if t is None:
            continue
        # bool is a subclass of int in Python; a boolean is not an integer here.
        if n in ("number", "integer") and isinstance(value, bool):
            continue
        if isinstance(value, t):
            return True
    return False


def validate_document(doc: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable schema violations. Empty means valid."""
    errs: list[str] = []

    if "const" in schema and doc != schema["const"]:
        errs.append(f"{path}: expected the constant {schema['const']!r}, got {doc!r}")
        return errs
    if "enum" in schema and doc not in schema["enum"]:
        errs.append(f"{path}: {doc!r} is not one of {schema['enum']}")
        return errs
    if "type" in schema and not _type_ok(doc, schema["type"]):
        errs.append(f"{path}: expected type {schema['type']}, got {type(doc).__name__}")
        return errs

    if isinstance(doc, dict):
        for req in schema.get("required", []):
            if req not in doc:
                errs.append(f"{path}.{req}: required field is missing")
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in doc and doc[k] is not None or (k in doc and sub.get("type") == "null"):
                errs += validate_document(doc[k], sub, f"{path}.{k}")
            elif k in doc and doc[k] is None:
                if not _type_ok(None, sub.get("type", [])):
                    errs.append(f"{path}.{k}: null is not permitted here")
    elif isinstance(doc, list):
        if "minItems" in schema and len(doc) < schema["minItems"]:
            errs.append(f"{path}: needs at least {schema['minItems']} item(s), has {len(doc)}")
        item = schema.get("items")
        if item:
            for i, v in enumerate(doc):
                errs += validate_document(v, item, f"{path}[{i}]")
    elif isinstance(doc, str):
        if "minLength" in schema and len(doc) < schema["minLength"]:
            errs.append(f"{path}: string is shorter than {schema['minLength']}")
    elif isinstance(doc, (int, float)) and not isinstance(doc, bool):
        if "minimum" in schema and doc < schema["minimum"]:
            errs.append(f"{path}: {doc} is below the minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and doc <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {doc} must exceed {schema['exclusiveMinimum']}")
        if "maximum" in schema and doc > schema["maximum"]:
            errs.append(f"{path}: {doc} is above the maximum {schema['maximum']}")
    return errs


def validate_claim_document(claim: dict[str, Any]) -> list[str]:
    return validate_document(claim, CLAIM_SCHEMA)


def export(outdir: str | Path) -> list[Path]:
    """Write the schemas as .json so other tooling can consume them."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, sch in SCHEMAS.items():
        p = out / f"{name}.schema.json"
        p.write_text(json.dumps(sch, indent=2), encoding="utf-8")
        written.append(p)
    (out / "schemas.digest.txt").write_text(schema_digest() + "\n", encoding="utf-8")
    return written
