from __future__ import annotations

import shutil
from pathlib import Path

import fitz
import pytest

from chemx.evaluate import assert_gold_isolated
from chemx.models import Prediction
from chemx.pipeline import parse_article


class FakeBackend:
    name = "fake"

    def run(self, workspace: Path, spec) -> Prediction:
        values = {field.name: None for field in spec.fields}
        evidence = {field.name: [] for field in spec.fields}
        prediction = Prediction(
            domain=spec.slug,
            records=[{"values": values, "evidence": evidence}],
        )
        (workspace / "prediction.json").write_text(prediction.model_dump_json(indent=2))
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
        manifest = (run / "manifest.json").read_text().replace(" ", "")
        assert '"state":"inference_complete"' in manifest
        assert_gold_isolated(run)
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
