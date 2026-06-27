import importlib.util
from pathlib import Path

import pandas as pd

from chemx.models import RunManifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_prediction_chemical_metrics.py"
SPEC = importlib.util.spec_from_file_location(
    "write_prediction_chemical_metrics",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
write_prediction_chemical_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(write_prediction_chemical_metrics)

aggregate_chemical_metric_rows = (
    write_prediction_chemical_metrics.aggregate_chemical_metric_rows
)
chemical_fields = write_prediction_chemical_metrics.chemical_fields
chemical_metric_rows = write_prediction_chemical_metrics.chemical_metric_rows
latest_completed_article_runs = (
    write_prediction_chemical_metrics.latest_completed_article_runs
)
write_metric_csv = write_prediction_chemical_metrics.write_metric_csv


def test_chemical_fields_reads_first_metrics_column(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": ["compound_id", "smiles"],
            "tp": [1, 0],
            "fp": [0, 1],
            "fn": [0, 1],
            "precision": [1.0, 0.0],
            "recall": [1.0, 0.0],
            "f1": [1.0, 0.0],
        }
    ).to_csv(path, index=False)

    assert chemical_fields(path) == ["compound_id", "smiles"]


def test_chemical_metric_rows_use_prediction_counts() -> None:
    result = {
        "fields": {
            "compound_id": {
                "true_positive": 2,
                "predicted": 5,
                "expected": 4,
                "precision": 0.4,
                "recall": 0.5,
                "f1": 4 / 9,
            }
        }
    }

    assert chemical_metric_rows(result, ["compound_id"]) == [
        {
            "field": "compound_id",
            "tp": 2,
            "fp": 3,
            "fn": 2,
            "precision": 0.4,
            "recall": 0.5,
            "f1": 4 / 9,
        }
    ]


def test_aggregate_chemical_metric_rows_sums_article_counts() -> None:
    results = [
        {
            "fields": {
                "compound_id": {
                    "true_positive": 2,
                    "predicted": 4,
                    "expected": 5,
                }
            }
        },
        {
            "fields": {
                "compound_id": {
                    "true_positive": 1,
                    "predicted": 2,
                    "expected": 1,
                }
            }
        },
    ]

    assert aggregate_chemical_metric_rows(results, ["compound_id"]) == [
        {
            "field": "compound_id",
            "tp": 3,
            "fp": 3,
            "fn": 3,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }
    ]


def test_latest_completed_article_runs_searches_nested_run_directories(tmp_path: Path) -> None:
    source_pdf = tmp_path / "article.pdf"
    old_run = tmp_path / "complexes" / "article" / "20240101T000000Z-article"
    new_run = tmp_path / "complexes" / "article" / "20240102T000000Z-article"
    for run_dir in [old_run, new_run]:
        run_dir.mkdir(parents=True)
        manifest = RunManifest(
            run_id=run_dir.name,
            source_pdf=str(source_pdf),
            domain="complexes",
            backend="fake",
            state="inference_complete",
        )
        (run_dir / "manifest.json").write_text(
            manifest.model_dump_json(),
            encoding="utf-8",
        )
        (run_dir / "prediction.json").write_text("{}", encoding="utf-8")

    assert latest_completed_article_runs(tmp_path) == {"complexes": [new_run]}


def test_write_metric_csv_matches_baseline_header_shape(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"

    write_metric_csv(
        [
            {
                "field": "compound_id",
                "tp": 1,
                "fp": 2,
                "fn": 3,
                "precision": 0.25,
                "recall": 0.5,
                "f1": 1 / 3,
            }
        ],
        path,
    )

    assert list(pd.read_csv(path, nrows=0).columns) == [
        "Unnamed: 0",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
