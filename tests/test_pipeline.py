from __future__ import annotations

import shutil
from pathlib import Path

import fitz
import pytest

from chemx.evaluate import assert_gold_isolated
from chemx.models import Prediction, RunManifest
from chemx.pipeline import parse_article, resume_article


class FakeBackend:
    name = "fake"

    def run(self, workspace: Path, spec) -> Prediction:
        values = {field.name: None for field in spec.fields}
        evidence = {field.name: [] for field in spec.fields}
        prediction = Prediction(
            domain=spec.slug,
            records=[{"values": values, "evidence": evidence}],
        )
        (workspace / "prediction.json").write_text(
            prediction.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return prediction


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Selective toxicity result")
    document.save(path)
    document.close()


def test_full_pipeline_with_fake_backend(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "SelTox"
    pdf_dir.mkdir()
    pdf = pdf_dir / "article.pdf"
    make_pdf(pdf)
    run = parse_article(pdf, backend=FakeBackend())
    try:
        assert (run / "bundle.json").is_file()
        assert (run / "prediction.json").is_file()
        assert (run / "prediction.csv").is_file()
        assert (run / "review.json").is_file()
        assert (run / "review_report.md").is_file()
        manifest = (run / "manifest.json").read_text().replace(" ", "")
        assert '"state":"inference_complete"' in manifest
        assert_gold_isolated(run)
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_pipeline_can_skip_reviewer(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "SelTox"
    pdf_dir.mkdir()
    pdf = pdf_dir / "article.pdf"
    make_pdf(pdf)
    run = parse_article(pdf, backend=FakeBackend(), skip_reviewer=True)
    try:
        assert (run / "prediction.json").is_file()
        assert (run / "prediction.csv").is_file()
        assert not (run / "review.json").exists()
        assert not (run / "review_report.md").exists()
        completed = RunManifest.model_validate_json(
            (run / "manifest.json").read_text(encoding="utf-8")
        )
        assert completed.state == "inference_complete"
        assert completed.error is None
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_resume_reuses_existing_bundle_without_reparsing(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "SelTox"
    pdf_dir.mkdir()
    pdf = pdf_dir / "article.pdf"
    make_pdf(pdf)
    run = parse_article(pdf, backend=FakeBackend())
    try:
        bundle = run / "bundle.json"
        bundle_mtime = bundle.stat().st_mtime_ns
        for name in ("prediction.json", "prediction.csv", "review.json", "review_report.md"):
            (run / name).unlink(missing_ok=True)
        manifest_path = run / "manifest.json"
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        manifest.state = "failed"
        manifest.error = "external backend limit"
        manifest.prediction_path = None
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        resumed = resume_article(run, backend=FakeBackend())

        assert resumed == run
        assert bundle.stat().st_mtime_ns == bundle_mtime
        assert (run / "prediction.csv").is_file()
        completed = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        assert completed.state == "inference_complete"
        assert completed.error is None
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_tmp_run_output_directory_is_rejected(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "SelTox"
    pdf_dir.mkdir()
    pdf = pdf_dir / "article.pdf"
    make_pdf(pdf)
    with pytest.raises(ValueError, match="cannot be inside"):
        parse_article(pdf, backend=FakeBackend(), runs_dir=tmp_path / "runs")


def test_gold_leak_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "exp_final.xlsx").write_bytes(b"gold")
    with pytest.raises(RuntimeError, match="gold data leaked"):
        assert_gold_isolated(tmp_path)
