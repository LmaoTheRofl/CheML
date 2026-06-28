import csv
import json
from pathlib import Path

import pytest

from chemx.domains import load_domain
from chemx.evaluate import (
    evaluate_run,
    evaluate_runs,
    validate_parquet_contract,
    write_prediction_csv,
)
from chemx.models import Prediction, RunManifest


def write_run(
    root: Path,
    state: str = "inference_complete",
    source_pdf: Path | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    spec = load_domain("eyedrops")
    values = {
        "smiles": "CCO",
        "name": "A",
        "perm (cm/s)": "0.1",
        "logP": "1.25",
        "doi": "10.1234/test.doi",
        "PMID": 1.0,
        "title": "T",
        "publisher": "P",
        "year": 2024,
        "access": 1,
        "page": 2,
        "origin": "table 1",
    }
    evidence = {field.name: [] for field in spec.fields}
    prediction = Prediction(domain=spec.slug, records=[{"values": values, "evidence": evidence}])
    manifest = RunManifest(
        run_id="test",
        source_pdf=str(source_pdf or "article.pdf"),
        domain=spec.slug,
        backend="fake",
        state=state,
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (root / "prediction.json").write_text(prediction.model_dump_json(), encoding="utf-8")
    return root


def test_local_column_gold_can_reach_f1_one(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    gold = tmp_path.parent / "reference.json"
    gold.write_text(
        json.dumps(
            {
                "smiles": ["CCO"],
                "name": ["A"],
                "perm (cm/s)": ["0.1"],
                "logP": ["1,25"],
                "doi": ["10.1234/test.doi"],
                "PMID": [1.0],
                "title": ["T"],
                "publisher": ["P"],
                "year": [2024],
                "access": [1],
                "page": [2],
                "origin": ["table 1"],
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_run(run, gold_path=gold)
    assert result["all_fields_f1_1"] is True
    assert result["macro_f1"] == 1.0
    assert all(metric["f1"] == 1.0 for metric in result["fields"].values())
    assert (run / "evaluation_metrics.csv").is_file()


def test_evaluation_is_blocked_before_inference(tmp_path: Path) -> None:
    run = write_run(tmp_path, state="bundled")
    gold = tmp_path.parent / "reference.json"
    gold.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="after inference_complete"):
        evaluate_run(run, gold_path=gold)


def test_parquet_gold_is_filtered_by_current_pdf_doi(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import fitz

    pdf = tmp_path / "article.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "DOI: 10.1234/test.doi")
    document.save(pdf)
    document.close()
    run = write_run(tmp_path / "run", source_pdf=pdf)
    gold = tmp_path / "train-00000-of-00001.parquet"
    prediction = Prediction.model_validate_json((run / "prediction.json").read_text())
    matching = dict(prediction.records[0].values)
    matching["logP"] = "1,25"
    other = {**matching, "smiles": "CCC", "name": "B", "logP": "3.0"}
    other["doi"] = "10.9999/other"
    pd.DataFrame([matching, other]).to_parquet(gold)

    result = evaluate_run(run, gold_path=gold)

    assert result["gold_rows"] == 1
    assert result["gold_doi_matches"] == ["10.1234/test.doi"]
    assert result["fields"]["logP"]["f1"] == 1.0
    assert result["gold_schema_matches_contract"] is True
    assert result["reference_path"].endswith("reference.csv")
    assert (run / "evaluation_metrics.csv").read_text(encoding="utf-8").startswith("domain,field")
    with (run / "reference.csv").open(encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    assert list(reference_rows[0]) == [field.name for field in load_domain("eyedrops").fields]
    assert reference_rows[0]["logP"] == "1,25"


def test_parquet_gold_falls_back_to_pdf_column_when_doi_is_absent(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import fitz

    spec = load_domain("oxazolidinones")
    pdf = tmp_path / "article-key.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "DOI: 10.1234/current")
    document.save(pdf)
    document.close()
    values = {
        field.name: (1 if field.type == "integer" else 1.0 if field.type == "number" else "x")
        for field in spec.fields
    }
    values["doi"] = "10.1234/current"
    values["pdf"] = "article-key.pdf"
    prediction = Prediction(domain=spec.slug, records=[{"values": values, "evidence": {}}])
    run = tmp_path / "run"
    run.mkdir()
    manifest = RunManifest(
        run_id="test",
        source_pdf=str(pdf),
        domain=spec.slug,
        backend="fake",
        state="inference_complete",
    )
    (run / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (run / "prediction.json").write_text(prediction.model_dump_json(), encoding="utf-8")
    gold_values = {**values, "doi": "10.9999/other", "pdf": "article-key"}
    other_values = {**values, "doi": "10.9999/other", "pdf": "other-article"}
    pd.DataFrame([gold_values, other_values]).to_parquet(tmp_path / "gold.parquet")

    result = evaluate_run(run, gold_path=tmp_path / "gold.parquet")

    assert result["gold_rows"] == 1
    assert result["gold_doi_matches"] == []
    assert result["gold_pdf_matches"] == ["article-key"]
    assert result["gold_match_mode"] == "pdf"


def test_parquet_gold_falls_back_to_title_when_doi_and_pdf_do_not_match(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import fitz

    spec = load_domain("complexes")
    pdf = tmp_path / "Technetium_and_rhenium_coordination_chemistry_and_.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "No DOI on this synthetic page")
    document.save(pdf)
    document.close()
    values = {
        field.name: (1 if field.type == "integer" else 1.0 if field.type == "number" else "x")
        for field in spec.fields
    }
    values["doi"] = ""
    values["pdf"] = "Technetium_and_rhenium_coordination_chemistry_and_.pdf"
    values["title"] = (
        "Technetium and Rhenium - Coordination Chemistry and Nuclear Medical Applications"
    )
    prediction = Prediction(domain=spec.slug, records=[{"values": values, "evidence": {}}])
    run = tmp_path / "run"
    run.mkdir()
    manifest = RunManifest(
        run_id="test",
        source_pdf=str(pdf),
        domain=spec.slug,
        backend="fake",
        state="inference_complete",
    )
    (run / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (run / "prediction.json").write_text(prediction.model_dump_json(), encoding="utf-8")
    gold_values = {
        **values,
        "doi": "10.1590/S0103-50532006000800004",
        "pdf": "abram2006",
        "title": (
            "Technetium and rhenium: coordination chemistry and nuclear medical applications"
        ),
    }
    other_values = {**gold_values, "pdf": "other-key", "title": "Different title"}
    pd.DataFrame([gold_values, other_values]).to_parquet(tmp_path / "gold.parquet")

    result = evaluate_run(run, gold_path=tmp_path / "gold.parquet")

    assert result["gold_rows"] == 1
    assert result["gold_doi_matches"] == []
    assert result["gold_pdf_matches"] == []
    assert result["gold_title_matches"] == [
        "technetium and rhenium coordination chemistry and nuclear medical applications"
    ]
    assert result["gold_match_mode"] == "title"


def test_prediction_csv_defaults_to_domain_field_order(tmp_path: Path) -> None:
    spec = load_domain("eyedrops")
    values = {field.name: None for field in reversed(spec.fields)}
    prediction = Prediction(
        domain=spec.slug,
        records=[{"values": values, "evidence": {field.name: [] for field in spec.fields}}],
    )

    path = write_prediction_csv(prediction, tmp_path / "prediction.csv")

    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == [field.name for field in spec.fields]


def test_prediction_csv_rejects_field_name_mismatch(tmp_path: Path) -> None:
    spec = load_domain("eyedrops")
    values = {field.name: None for field in spec.fields}
    values.pop(spec.fields[0].name)
    values["unexpected"] = "value"
    prediction = Prediction(
        domain=spec.slug,
        records=[{"values": values, "evidence": {field.name: [] for field in spec.fields}}],
    )

    with pytest.raises(ValueError, match="fields mismatch"):
        write_prediction_csv(prediction, tmp_path / "prediction.csv")


def test_evaluate_runs_writes_one_article_macro_f1_row(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import fitz

    pdf = tmp_path / "article.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "DOI: 10.1234/test.doi")
    document.save(pdf)
    document.close()
    run = write_run(tmp_path / "runs" / "one", source_pdf=pdf)
    prediction = Prediction.model_validate_json((run / "prediction.json").read_text())
    pd.DataFrame([prediction.records[0].values]).to_parquet(tmp_path / "gold.parquet")

    results, summary = evaluate_runs(tmp_path / "runs", datasets_dir=tmp_path)

    assert len(results) == 1
    assert results[0]["macro_f1"] == 1.0
    assert results[0]["gold_schema_matches_contract"] is True
    lines = summary.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "macro_f1" in lines[0]


def test_downloaded_parquet_types_match_all_domain_contracts() -> None:
    pytest.importorskip("pyarrow")
    datasets = Path(__file__).resolve().parents[1] / "datasets"
    parquet_paths = sorted(datasets.rglob("*.parquet"))
    pdf_dirs = {path.parent for path in datasets.rglob("*.pdf")}
    parquet_dirs = {path.parent for path in parquet_paths}
    assert parquet_dirs == pdf_dirs
    for parquet_path in parquet_paths:
        domain = parquet_path.parent.name.strip().lower()
        assert validate_parquet_contract(parquet_path, domain)
