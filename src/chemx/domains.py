from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chemx.models import DomainSpec

DOMAIN_SLUGS = (
    "eyedrops",
    "benzimidazoles",
    "oxazolidinones",
    "co-crystals",
    "complexes",
    "nanozymes",
    "synergy",
    "nanomag",
    "cytotox",
    "seltox",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def skills_root() -> Path:
    return project_root() / ".agents" / "skills"


def load_domain(slug: str, root: Path | None = None) -> DomainSpec:
    normalized = slug.strip().lower().replace("_", "-")
    aliases = {"co-crystal": "co-crystals", "eye-drops": "eyedrops"}
    normalized = aliases.get(normalized, normalized)
    path = (root or skills_root()) / normalized / "domain.json"
    if not path.exists():
        raise ValueError(f"unknown ChemX domain: {slug}")
    return DomainSpec.model_validate_json(path.read_text(encoding="utf-8"))


def list_domains(root: Path | None = None) -> list[DomainSpec]:
    return [load_domain(slug, root) for slug in DOMAIN_SLUGS]


def detect_domain(pdf: Path, text: str = "", root: Path | None = None) -> DomainSpec:
    haystack = f"{pdf.as_posix()} {text[:20_000]}".lower()
    scores: list[tuple[int, DomainSpec]] = []
    for spec in list_domains(root):
        terms = {spec.slug, spec.name.lower(), *[alias.lower() for alias in spec.aliases]}
        score = sum(
            len(re.findall(re.escape(term), haystack, flags=re.IGNORECASE)) for term in terms
        )
        scores.append((score, spec))
    score, best = max(scores, key=lambda item: item[0])
    if score == 0:
        raise ValueError(f"cannot detect domain from path or text: {pdf}")
    return best


def output_schema(spec: DomainSpec) -> dict[str, Any]:
    value_properties: dict[str, Any] = {}
    required: list[str] = []
    evidence_properties: dict[str, Any] = {}
    source_ref = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page": {"type": "integer", "minimum": 1},
            "kind": {
                "type": "string",
                "enum": [
                    "text",
                    "table",
                    "figure",
                    "caption",
                    "metadata",
                    "layout",
                    "marker",
                    "ocr",
                    "ocsr",
                ],
            },
            "bbox": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {key: {"type": "number"} for key in ("x0", "y0", "x1", "y1")},
                        "required": ["x0", "y0", "x1", "y1"],
                    },
                    {"type": "null"},
                ]
            },
            "text": {"type": ["string", "null"]},
            "asset_path": {"type": ["string", "null"]},
        },
        "required": ["page", "kind", "bbox", "text", "asset_path"],
    }
    for field in spec.fields:
        base: dict[str, Any] = {"type": field.type}
        if field.enum:
            base["enum"] = field.enum
        value_properties[field.name] = {"anyOf": [base, {"type": "null"}]}
        evidence_properties[field.name] = {"type": "array", "items": source_ref}
        if field.required:
            required.append(field.name)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string", "const": "1.0"},
            "domain": {"type": "string", "const": spec.slug},
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": value_properties,
                            "required": required,
                        },
                        "evidence": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": evidence_properties,
                            "required": list(evidence_properties),
                        },
                    },
                    "required": ["values", "evidence"],
                },
            },
        },
        "required": ["schema_version", "domain", "records"],
    }


def write_output_schema(spec: DomainSpec, path: Path) -> Path:
    path.write_text(json.dumps(output_schema(spec), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
