from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from chemx.evaluate import evaluate_run
from chemx.models import RunManifest

BASELINE_DOMAIN_BY_FILE = {
    "metrics_benzimidazole_from_single_agent.csv": "benzimidazoles",
    "metrics_cocrystals_from_single_agent.csv": "co-crystals",
    "metrics_complexes_from_single_agent.csv": "complexes",
    "metrics_cytotoxicity_from_single_agent.csv": "cytotox",
    "metrics_magnetic_from_single_agent.csv": "nanomag",
    "metrics_nanozymes_from_single_agent.csv": "nanozymes",
    "metrics_oxazolidinone_from_single_agent.csv": "oxazolidinones",
    "metrics_seltox_from_single_agent.csv": "seltox",
    "metrics_synergy_from_single_agent.csv": "synergy",
}


def chemical_fields(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return [str(value) for value in frame.iloc[:, 0].tolist()]


def latest_completed_runs(runs_dir: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for manifest_path in sorted(runs_dir.rglob("manifest.json")):
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        if manifest.state != "inference_complete":
            continue
        if not (run_dir / "prediction.json").is_file():
            continue
        previous = runs.get(manifest.domain)
        if previous is None or run_dir.name > previous.name:
            runs[manifest.domain] = run_dir
    return runs


def latest_completed_article_runs(runs_dir: Path) -> dict[str, list[Path]]:
    latest_by_article: dict[tuple[str, str], Path] = {}
    for manifest_path in sorted(runs_dir.rglob("manifest.json")):
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        if manifest.state != "inference_complete":
            continue
        if not (run_dir / "prediction.json").is_file():
            continue
        source_pdf = str(Path(manifest.source_pdf).resolve())
        key = (manifest.domain, source_pdf)
        previous = latest_by_article.get(key)
        if previous is None or run_dir.name > previous.name:
            latest_by_article[key] = run_dir

    runs: dict[str, list[Path]] = {}
    for (domain, _), run_dir in latest_by_article.items():
        runs.setdefault(domain, []).append(run_dir)
    for domain_runs in runs.values():
        domain_runs.sort(key=lambda path: path.name)
    return runs


def chemical_metric_rows(result: dict[str, Any], fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        metric = result["fields"].get(field)
        if metric is None:
            raise ValueError(f"field {field!r} is absent from evaluation result")
        true_positive = int(metric["true_positive"])
        predicted = int(metric["predicted"])
        expected = int(metric["expected"])
        rows.append(
            {
                "field": field,
                "tp": true_positive,
                "fp": predicted - true_positive,
                "fn": expected - true_positive,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
            }
        )
    return rows


def aggregate_chemical_metric_rows(
    results: list[dict[str, Any]],
    fields: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        true_positive = 0
        predicted = 0
        expected = 0
        for result in results:
            metric = result["fields"].get(field)
            if metric is None:
                raise ValueError(f"field {field!r} is absent from evaluation result")
            true_positive += int(metric["true_positive"])
            predicted += int(metric["predicted"])
            expected += int(metric["expected"])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / expected if expected else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "field": field,
                "tp": true_positive,
                "fp": predicted - true_positive,
                "fn": expected - true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def write_metric_csv(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows).set_index("field")
    frame.index.name = None
    frame.to_csv(path, encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "domain",
        "baseline_file",
        "status",
        "gold_match_mode",
        "gold_doi",
        "gold_pdf",
        "gold_title",
        "runs_evaluated",
        "runs_failed",
        "run_dir",
        "source_pdf",
        "output_csv",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_name(baseline_path: Path) -> str:
    return baseline_path.name.replace("_from_single_agent.csv", "_from_predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    runs = latest_completed_article_runs(args.runs_dir)
    summary: list[dict[str, Any]] = []
    for baseline_path in sorted(args.metrics_dir.glob("metrics_*_from_single_agent.csv")):
        domain = BASELINE_DOMAIN_BY_FILE.get(baseline_path.name)
        if domain is None:
            summary.append(
                {
                    "domain": "",
                    "baseline_file": str(baseline_path),
                    "status": "skipped",
                    "gold_match_mode": "",
                    "gold_doi": "",
                    "gold_pdf": "",
                    "gold_title": "",
                    "runs_evaluated": 0,
                    "runs_failed": 0,
                    "run_dir": "",
                    "source_pdf": "",
                    "output_csv": "",
                    "error": "unknown baseline filename",
                }
            )
            continue
        run_dirs = runs.get(domain, [])
        if not run_dirs:
            summary.append(
                {
                    "domain": domain,
                    "baseline_file": str(baseline_path),
                    "status": "skipped",
                    "gold_match_mode": "",
                    "gold_doi": "",
                    "gold_pdf": "",
                    "gold_title": "",
                    "runs_evaluated": 0,
                    "runs_failed": 0,
                    "run_dir": "",
                    "source_pdf": "",
                    "output_csv": "",
                    "error": "no completed prediction run",
                }
            )
            continue
        try:
            fields = chemical_fields(baseline_path)
            results: list[dict[str, Any]] = []
            errors: list[str] = []
            for run_dir in run_dirs:
                try:
                    results.append(evaluate_run(run_dir, datasets_dir=args.datasets_dir))
                except Exception as exc:
                    errors.append(f"{run_dir.name}: {exc}")
            if not results:
                raise ValueError("; ".join(errors) or "no evaluable prediction run")
            output_path = args.metrics_dir / output_name(baseline_path)
            write_metric_csv(aggregate_chemical_metric_rows(results, fields), output_path)
            summary.append(
                {
                    "domain": domain,
                    "baseline_file": str(baseline_path),
                    "status": "partial" if errors else "written",
                    "gold_match_mode": ";".join(result["gold_match_mode"] for result in results),
                    "gold_doi": ";".join(
                        ";".join(result["gold_doi_matches"]) for result in results
                    ),
                    "gold_pdf": ";".join(
                        ";".join(result["gold_pdf_matches"]) for result in results
                    ),
                    "gold_title": ";".join(
                        ";".join(result["gold_title_matches"]) for result in results
                    ),
                    "runs_evaluated": len(results),
                    "runs_failed": len(errors),
                    "run_dir": ";".join(result["run_dir"] for result in results),
                    "source_pdf": ";".join(result["source_pdf"] for result in results),
                    "output_csv": str(output_path),
                    "error": "; ".join(errors),
                }
            )
        except Exception as exc:
            summary.append(
                {
                    "domain": domain,
                    "baseline_file": str(baseline_path),
                    "status": "failed",
                    "gold_match_mode": "",
                    "gold_doi": "",
                    "gold_pdf": "",
                    "gold_title": "",
                    "runs_evaluated": 0,
                    "runs_failed": len(run_dirs),
                    "run_dir": ";".join(str(path) for path in run_dirs),
                    "source_pdf": "",
                    "output_csv": "",
                    "error": str(exc),
                }
            )
    summary_path = args.metrics_dir / "prediction_metrics_summary.csv"
    write_summary(summary, summary_path)
    for row in summary:
        print(f"{row['domain']}: {row['status']} {row['output_csv'] or row['error']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
