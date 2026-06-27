from pathlib import Path

import pandas as pd

from scripts.write_prediction_chemical_metrics import (
    aggregate_chemical_metric_rows,
    chemical_fields,
    chemical_metric_rows,
    write_metric_csv,
)


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
