from __future__ import annotations

import json

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
