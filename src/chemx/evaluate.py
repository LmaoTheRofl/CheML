from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

from chemx.domains import detect_domain, load_domain, project_root
from chemx.models import Prediction, RunManifest
from chemx.normalize import is_missing, multiset_metric, normalize_number, normalize_value

DEFAULT_DATASETS_DIR = Path("datasets")
GOLD_MARKERS = ("gold", "answer", "ground_truth", "exp_final")
QA_COLUMNS = {"Question ID", "Domain", "Question Final", "Answer"}
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s<>\"'\\\]\)]+", re.IGNORECASE)


@dataclass(frozen=True)
class GoldTable:
    columns: dict[str, list[Any]]
    path: Path
    rows: int
    doi_matches: list[str]
    pdf_matches: list[str]
    title_matches: list[str]
    match_mode: str
    schema: dict[str, str] | None = None


def assert_gold_isolated(workspace: Path) -> None:
    violations = [
        path
        for path in workspace.rglob("*")
        if path.is_file() and any(marker in path.name.lower() for marker in GOLD_MARKERS)
    ]
    if violations:
        names = ", ".join(path.relative_to(workspace).as_posix() for path in violations)
        raise RuntimeError(f"gold data leaked into inference workspace: {names}")


def _prediction_columns(prediction: Prediction) -> dict[str, list[Any]]:
    fields = {key for record in prediction.records for key in record.values}
    return {field: [record.values.get(field) for record in prediction.records] for field in fields}


def write_prediction_csv(
    prediction: Prediction,
    path: Path,
    fields: list[str] | None = None,
) -> Path:
    import csv

    ordered = fields or [field.name for field in load_domain(prediction.domain).fields]
    expected = set(ordered)
    for index, record in enumerate(prediction.records):
        actual = set(record.values)
        missing = expected - actual
        unknown = actual - expected
        if missing or unknown:
            raise ValueError(
                f"record {index} fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for record in prediction.records:
            writer.writerow({field: record.values.get(field) for field in ordered})
    return path


def _normalize_doi(value: Any) -> str:
    if is_missing(value):
        return ""
    raw = str(value).strip().lower()
    raw = re.sub(r"^(doi:\s*|https?://(dx\.)?doi\.org/)", "", raw)
    return raw.rstrip(".,;:)]}")


def _doi_candidates(run_dir: Path, manifest: RunManifest) -> list[str]:
    candidates: list[str] = []
    texts: list[str] = [manifest.source_pdf]
    bundle_path = run_dir / "bundle.json"
    if bundle_path.exists():
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
        texts.extend(page.get("text", "") for page in raw.get("pages", []))
    source_pdf = Path(manifest.source_pdf)
    if source_pdf.exists():
        try:
            import fitz

            document = fitz.open(source_pdf)
            texts.extend(page.get_text("text") for page in document)
            document.close()
        except Exception:
            pass
    seen: set[str] = set()
    for text in texts:
        for match in DOI_PATTERN.findall(text or ""):
            doi = _normalize_doi(match)
            if doi and doi not in seen:
                seen.add(doi)
                candidates.append(doi)
    return candidates


def _normalize_pdf_key(value: Any) -> str:
    if is_missing(value):
        return ""
    raw = str(value).strip().lower().replace("\\", "/")
    name = raw.rsplit("/", 1)[-1]
    if name.endswith(".pdf"):
        name = name[:-4]
    return re.sub(r"\s+", "", name)


def _pdf_candidates(manifest: RunManifest) -> list[str]:
    source = Path(manifest.source_pdf)
    candidates = [
        _normalize_pdf_key(manifest.source_pdf),
        _normalize_pdf_key(source.name),
        _normalize_pdf_key(source.stem),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _normalize_title_key(value: Any) -> str:
    if is_missing(value):
        return ""
    raw = unescape(str(value)).strip().lower()
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = raw.replace("&", " and ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _title_candidates(run_dir: Path, manifest: RunManifest) -> list[str]:
    source = Path(manifest.source_pdf)
    candidates = [_normalize_title_key(source.stem.replace("_", " "))]
    prediction_path = run_dir / "prediction.json"
    if prediction_path.is_file():
        try:
            prediction = Prediction.model_validate_json(
                prediction_path.read_text(encoding="utf-8")
            )
        except Exception:
            prediction = None
        if prediction is not None:
            for record in prediction.records:
                candidates.append(_normalize_title_key(record.values.get("title")))
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _filter_frame_to_run(
    frame: Any,
    run_dir: Path,
    manifest: RunManifest,
    *,
    strict: bool,
) -> tuple[Any, list[str], list[str], list[str], str]:
    if QA_COLUMNS.issubset(set(frame.columns)):
        raise ValueError(
            "incompatible Q&A gold: ChemX table evaluation requires local domain-column parquet"
        )
    doi_candidates = set(_doi_candidates(run_dir, manifest))
    if "doi" in frame.columns and doi_candidates:
        normalized = frame["doi"].map(_normalize_doi)
        matches = sorted(doi_candidates & set(normalized.dropna()))
        if matches:
            return frame[normalized.isin(matches)].reset_index(drop=True), matches, [], [], "doi"
    if "pdf" in frame.columns:
        pdf_candidates = set(_pdf_candidates(manifest))
        normalized_pdf = frame["pdf"].map(_normalize_pdf_key)
        pdf_matches = sorted(pdf_candidates & set(normalized_pdf.dropna()))
        if pdf_matches:
            return (
                frame[normalized_pdf.isin(pdf_matches)].reset_index(drop=True),
                [],
                pdf_matches,
                [],
                "pdf",
            )
    if "title" in frame.columns:
        title_candidates = set(_title_candidates(run_dir, manifest))
        normalized_title = frame["title"].map(_normalize_title_key)
        title_matches = sorted(title_candidates & set(normalized_title.dropna()))
        if title_matches:
            return (
                frame[normalized_title.isin(title_matches)].reset_index(drop=True),
                [],
                [],
                title_matches,
                "title",
            )
    if strict:
        if not doi_candidates and "pdf" not in frame.columns and "title" not in frame.columns:
            raise ValueError(
                "could not detect DOI and parquet has no pdf/title columns for current PDF"
            )
        preview = ", ".join(sorted(doi_candidates)[:5])
        raise ValueError(
            "current PDF is absent from parquet gold by DOI, pdf column, and title"
            + (f": {preview}" if preview else "")
        )
    return frame, [], [], [], "unfiltered"


def _frame_to_gold(
    path: Path,
    frame: Any,
    run_dir: Path,
    manifest: RunManifest,
    *,
    strict: bool,
) -> GoldTable:
    frame, doi_matches, pdf_matches, title_matches, match_mode = _filter_frame_to_run(
        frame, run_dir, manifest, strict=strict
    )
    return GoldTable(
        columns={str(column): frame[column].tolist() for column in frame.columns},
        path=path,
        rows=len(frame),
        doi_matches=doi_matches,
        pdf_matches=pdf_matches,
        title_matches=title_matches,
        match_mode=match_mode,
    )


def _write_reference_csv(gold_table: GoldTable, path: Path, fields: list[str]) -> Path:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index in range(gold_table.rows):
            writer.writerow(
                {
                    field: gold_table.columns.get(field, [None] * gold_table.rows)[index]
                    for field in fields
                }
            )
    return path


def _read_gold(
    path: Path,
    domain: str,
    run_dir: Path,
    manifest: RunManifest,
    *,
    strict_doi: bool = True,
) -> GoldTable:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping) and domain in raw:
            raw = raw[domain]
        if isinstance(raw, Mapping):
            columns = {str(key): list(value) for key, value in raw.items()}
            rows = max((len(values) for values in columns.values()), default=0)
            return GoldTable(
                columns=columns,
                path=path,
                rows=rows,
                doi_matches=[],
                pdf_matches=[],
                title_matches=[],
                match_mode="unfiltered",
            )
        if isinstance(raw, list):
            keys = {key for row in raw for key in row}
            return GoldTable(
                columns={key: [row.get(key) for row in raw] for key in keys},
                path=path,
                rows=len(raw),
                doi_matches=[],
                pdf_matches=[],
                title_matches=[],
                match_mode="unfiltered",
            )
        raise ValueError("unsupported JSON gold format")
    if path.suffix.lower() == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        return _frame_to_gold(path, frame, run_dir, manifest, strict=False)
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        schema = validate_parquet_contract(path, domain)
        frame = pd.read_parquet(path)
        table = _frame_to_gold(path, frame, run_dir, manifest, strict=strict_doi)
        return GoldTable(
            columns=table.columns,
            path=table.path,
            rows=table.rows,
            doi_matches=table.doi_matches,
            pdf_matches=table.pdf_matches,
            title_matches=table.title_matches,
            match_mode=table.match_mode,
            schema=schema,
        )
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None)
        if any(QA_COLUMNS.issubset(set(frame.columns)) for frame in sheets.values()):
            raise ValueError(
                "incompatible Q&A gold workbook: ChemX table evaluation requires "
                "domain-column JSON/CSV/XLSX/parquet gold"
            )
        candidates = {name.lower().replace("_", "-"): frame for name, frame in sheets.items()}
        key = min(candidates, key=lambda name: (domain not in name, len(name)))
        frame = candidates[key]
        return _frame_to_gold(path, frame, run_dir, manifest, strict=False)
    else:
        raise ValueError(f"unsupported gold file: {path}")


def _default_gold_path(manifest: RunManifest, domain: str, datasets_dir: Path | None) -> Path:
    source_pdf = Path(manifest.source_pdf)
    local = sorted(source_pdf.parent.glob("*.parquet")) if source_pdf.parent.exists() else []
    if len(local) == 1:
        return local[0]
    root = datasets_dir or DEFAULT_DATASETS_DIR
    if not root.is_absolute():
        root = project_root() / root
    normalized_domain = domain.replace("-", "").lower()
    candidates = [
        path
        for path in root.rglob("*.parquet")
        if normalized_domain in path.parent.name.replace("-", "").lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no local parquet gold found for domain {domain!r} below {root}")
    raise ValueError(f"ambiguous local parquet gold for domain {domain!r}: {candidates}")


def _numeric_like(values: list[Any]) -> bool:
    checked = 0
    numeric = 0
    for value in values:
        if is_missing(value):
            continue
        checked += 1
        if normalize_number(value) != str(value).strip():
            numeric += 1
            continue
        try:
            float(str(value).strip().replace(",", "."))
        except ValueError:
            continue
        numeric += 1
    return checked > 0 and numeric / checked >= 0.8


def _arrow_scalar_type(data_type: Any) -> str:
    import pyarrow.types as arrow_types

    if arrow_types.is_string(data_type) or arrow_types.is_large_string(data_type):
        return "string"
    if arrow_types.is_integer(data_type):
        return "integer"
    if arrow_types.is_floating(data_type) or arrow_types.is_decimal(data_type):
        return "number"
    if arrow_types.is_boolean(data_type):
        return "boolean"
    raise ValueError(f"unsupported parquet scalar type: {data_type}")


def validate_parquet_contract(path: Path, domain: str) -> dict[str, str]:
    """Require the domain field names, order, and scalar types to match parquet."""
    import pyarrow.parquet as parquet

    schema = parquet.read_schema(path)
    actual = {field.name: _arrow_scalar_type(field.type) for field in schema}
    spec = load_domain(domain)
    expected = {field.name: field.type for field in spec.fields}
    if list(actual) != list(expected):
        missing = [name for name in expected if name not in actual]
        unknown = [name for name in actual if name not in expected]
        raise ValueError(
            f"parquet fields do not match {domain!r} contract: "
            f"missing={missing}, unknown={unknown}, order_matches={set(actual) == set(expected)}"
        )
    mismatches = {
        name: {"parquet": actual[name], "contract": expected[name]}
        for name in actual
        if actual[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"parquet scalar types do not match {domain!r} contract: {mismatches}")
    return actual


def audit_parquet_contracts(datasets_dir: Path, output_path: Path) -> tuple[int, Path]:
    """Write an auditable field/type comparison for every local ChemX parquet."""
    import csv

    parquet_paths = sorted(datasets_dir.rglob("*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no parquet files below {datasets_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parquet_path",
                "domain",
                "field",
                "parquet_type",
                "contract_type",
                "matches",
            ],
        )
        writer.writeheader()
        for parquet_path in parquet_paths:
            spec = detect_domain(parquet_path)
            actual = validate_parquet_contract(parquet_path, spec.slug)
            expected = {field.name: field.type for field in spec.fields}
            for field, parquet_type in actual.items():
                writer.writerow(
                    {
                        "parquet_path": str(parquet_path.resolve()),
                        "domain": spec.slug,
                        "field": field,
                        "parquet_type": parquet_type,
                        "contract_type": expected[field],
                        "matches": parquet_type == expected[field],
                    }
                )
    return len(parquet_paths), output_path


def _write_metrics_csv(result: dict[str, Any], path: Path) -> Path:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "field",
                "precision",
                "recall",
                "f1",
                "true_positive",
                "predicted",
                "expected",
                "gold_rows",
                "gold_doi",
                "gold_pdf",
                "gold_title",
                "gold_match_mode",
                "gold_path",
                "reference_path",
                "macro_f1",
            ],
        )
        writer.writeheader()
        for field, metric in result["fields"].items():
            writer.writerow(
                {
                    "domain": result["domain"],
                    "field": field,
                    "gold_rows": result["gold_rows"],
                    "gold_doi": ";".join(result["gold_doi_matches"]),
                    "gold_pdf": ";".join(result["gold_pdf_matches"]),
                    "gold_title": ";".join(result["gold_title_matches"]),
                    "gold_match_mode": result["gold_match_mode"],
                    "gold_path": result["gold_path"],
                    "reference_path": result["reference_path"],
                    "macro_f1": result["macro_f1"],
                    **metric,
                }
            )
    return path


def _write_article_summary_csv(results: list[dict[str, Any]], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "source_pdf",
        "domain",
        "gold_doi",
        "gold_pdf",
        "gold_title",
        "gold_match_mode",
        "predicted_records",
        "gold_rows",
        "field_count",
        "macro_f1",
        "all_fields_f1_1",
        "gold_schema_matches_contract",
        "gold_path",
        "reference_path",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **{field: result.get(field) for field in fields},
                    "gold_doi": ";".join(result["gold_doi_matches"]),
                    "gold_pdf": ";".join(result["gold_pdf_matches"]),
                    "gold_title": ";".join(result["gold_title_matches"]),
                }
            )
    return path


def evaluate_run(
    run_dir: Path,
    *,
    gold_path: Path | None = None,
    datasets_dir: Path | None = None,
    strict_doi: bool = True,
) -> dict[str, Any]:
    manifest = RunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.state != "inference_complete":
        raise RuntimeError("gold can only be loaded after inference_complete")
    assert_gold_isolated(run_dir)
    prediction = Prediction.model_validate_json(
        (run_dir / "prediction.json").read_text(encoding="utf-8")
    )
    selected_gold = gold_path.resolve() if gold_path else _default_gold_path(
        manifest, prediction.domain, datasets_dir
    )
    gold_table = _read_gold(
        selected_gold,
        prediction.domain,
        run_dir,
        manifest,
        strict_doi=strict_doi,
    )
    gold = gold_table.columns
    predicted = _prediction_columns(prediction)
    spec = load_domain(prediction.domain)
    reference_path = _write_reference_csv(
        gold_table,
        run_dir / "reference.csv",
        [field.name for field in spec.fields],
    )
    field_specs = {field.name: field for field in spec.fields}
    metrics: dict[str, Any] = {}
    for column in sorted(set(gold) | set(predicted)):
        field = field_specs.get(column)
        values = predicted.get(column, []) + gold.get(column, [])
        numeric = (
            field is not None and field.type in {"number", "integer"}
        ) or _numeric_like(values)
        smiles = field is not None and field.smiles
        pred_values = [
            normalize_value(value, numeric=numeric, smiles=smiles)
            for value in predicted.get(column, [])
        ]
        gold_values = [
            normalize_value(value, numeric=numeric, smiles=smiles)
            for value in gold.get(column, [])
        ]
        metric = multiset_metric(pred_values, gold_values)
        metrics[column] = {
            "precision": metric.precision,
            "recall": metric.recall,
            "f1": metric.f1,
            "true_positive": metric.true_positive,
            "predicted": metric.predicted,
            "expected": metric.expected,
        }
    result = {
        "run_id": manifest.run_id,
        "run_dir": str(run_dir.resolve()),
        "source_pdf": manifest.source_pdf,
        "domain": prediction.domain,
        "gold_path": str(gold_table.path),
        "gold_rows": gold_table.rows,
        "gold_doi_matches": gold_table.doi_matches,
        "gold_pdf_matches": gold_table.pdf_matches,
        "gold_title_matches": gold_table.title_matches,
        "gold_match_mode": gold_table.match_mode,
        "reference_path": str(reference_path.resolve()),
        "predicted_records": len(prediction.records),
        "field_count": len(metrics),
        "macro_f1": sum(value["f1"] for value in metrics.values()) / len(metrics)
        if metrics
        else 0.0,
        "gold_schema_matches_contract": gold_table.schema is not None,
        "gold_schema": gold_table.schema,
        "all_fields_f1_1": bool(metrics) and all(value["f1"] == 1.0 for value in metrics.values()),
        "fields": metrics,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_metrics_csv(result, run_dir / "evaluation_metrics.csv")
    return result


def evaluate_runs(
    runs_dir: Path,
    *,
    datasets_dir: Path | None = None,
    output_path: Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Evaluate the newest completed run for every distinct source article."""
    completed: dict[str, tuple[RunManifest, Path]] = {}
    for manifest_path in sorted(runs_dir.rglob("manifest.json")):
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.state != "inference_complete":
            continue
        current = completed.get(manifest.source_pdf)
        if current is None or manifest.created_at > current[0].created_at:
            completed[manifest.source_pdf] = (manifest, manifest_path.parent)
    if not completed:
        raise ValueError(f"no completed inference runs below {runs_dir}")
    selected = sorted(completed.values(), key=lambda item: item[0].source_pdf)
    results = [
        evaluate_run(run_dir, datasets_dir=datasets_dir)
        for _, run_dir in selected
    ]
    summary_path = output_path or runs_dir / "article_macro_f1.csv"
    return results, _write_article_summary_csv(results, summary_path)
