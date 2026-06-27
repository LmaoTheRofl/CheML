from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chemx.chemistry import canonicalize_smiles_required
from chemx.models import DomainSpec, Prediction, PredictionRecord

_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
}


def validate_prediction(prediction: Prediction, spec: DomainSpec) -> Prediction:
    if prediction.domain != spec.slug:
        raise ValueError(f"prediction domain {prediction.domain!r} != {spec.slug!r}")
    expected = {field.name for field in spec.fields}
    for index, record in enumerate(prediction.records):
        actual = set(record.values)
        missing = expected - actual
        unknown = actual - expected
        if missing or unknown:
            raise ValueError(
                f"record {index} fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        unknown_evidence = set(record.evidence) - expected
        if unknown_evidence:
            raise ValueError(
                f"record {index} has unknown evidence fields: {sorted(unknown_evidence)}"
            )
        for field in spec.fields:
            value = record.values[field.name]
            if value is None:
                continue
            allowed = _PYTHON_TYPES[field.type]
            if isinstance(value, bool) and field.type in {"number", "integer"}:
                raise ValueError(f"record {index}.{field.name} must be {field.type}")
            if not isinstance(value, allowed):
                raise ValueError(f"record {index}.{field.name} must be {field.type}")
            if field.enum and value not in field.enum:
                raise ValueError(f"record {index}.{field.name} is outside enum: {value!r}")
    return prediction


def _coerce_value(value: Any, field_type: str) -> tuple[Any, bool]:
    if value is None:
        return None, False
    if field_type == "string":
        return str(value), not isinstance(value, str)
    if field_type == "number":
        if isinstance(value, bool):
            return None, True
        if isinstance(value, int | float):
            return value, False
        if isinstance(value, str):
            raw = value.strip().replace(",", ".")
            if not raw:
                return None, True
            try:
                return float(raw), True
            except ValueError:
                return None, True
    if field_type == "integer":
        if isinstance(value, bool):
            return None, True
        if isinstance(value, int):
            return value, False
        if isinstance(value, float) and value.is_integer():
            return int(value), True
        if isinstance(value, str):
            raw = value.strip()
            if re.fullmatch(r"[+-]?\d+", raw):
                return int(raw), True
            return None, True
    if field_type == "boolean":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true", True
    return value, False


def repair_and_canonicalize_prediction(
    prediction: Prediction,
    spec: DomainSpec,
    workspace: Path,
    *,
    require_rdkit: bool = True,
) -> Prediction:
    """Force contract shape and RDKit canonical SMILES before strict validation."""
    schema_repairs: list[dict[str, Any]] = []
    chemistry_repairs: list[dict[str, Any]] = []
    expected = [field.name for field in spec.fields]
    field_by_name = {field.name: field for field in spec.fields}
    repaired_records: list[PredictionRecord] = []
    for record_index, record in enumerate(prediction.records):
        values: dict[str, Any] = {}
        unknown = sorted(set(record.values) - set(expected))
        if unknown:
            schema_repairs.append(
                {"record": record_index, "kind": "drop_unknown_fields", "fields": unknown}
            )
        for field_name in expected:
            field = field_by_name[field_name]
            if field_name not in record.values:
                values[field_name] = None
                schema_repairs.append(
                    {"record": record_index, "kind": "add_missing_field", "field": field_name}
                )
                continue
            coerced, changed = _coerce_value(record.values[field_name], field.type)
            if changed:
                schema_repairs.append(
                    {
                        "record": record_index,
                        "kind": "coerce_type",
                        "field": field_name,
                        "from": record.values[field_name],
                        "to": coerced,
                    }
                )
            if require_rdkit and field.smiles and coerced not in {None, "", "NOT_DETECTED"}:
                canonical, valid = canonicalize_smiles_required(coerced)
                chemistry_repairs.append(
                    {
                        "record": record_index,
                        "field": field_name,
                        "raw": coerced,
                        "canonical": canonical,
                        "valid": valid,
                    }
                )
                coerced = canonical
            values[field_name] = coerced
        evidence = {field: record.evidence.get(field, []) for field in expected}
        evidence_unknown = sorted(set(record.evidence) - set(expected))
        if evidence_unknown:
            schema_repairs.append(
                {
                    "record": record_index,
                    "kind": "drop_unknown_evidence",
                    "fields": evidence_unknown,
                }
            )
        repaired_records.append(PredictionRecord(values=values, evidence=evidence))
    (workspace / "schema_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repair_count": len(schema_repairs),
                "repairs": schema_repairs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (workspace / "chemistry_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "smiles_count": len(chemistry_repairs),
                "invalid_smiles_count": sum(
                    1 for repair in chemistry_repairs if not repair["valid"]
                ),
                "smiles": chemistry_repairs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return Prediction(domain=spec.slug, records=repaired_records)


def deduplicate_prediction(prediction: Prediction) -> Prediction:
    """Remove only byte-equivalent duplicate records, including identical evidence."""
    seen: set[str] = set()
    records: list[PredictionRecord] = []
    for record in prediction.records:
        key = json.dumps(record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            records.append(record)
    return prediction.model_copy(update={"records": records})
