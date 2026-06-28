import json
import subprocess
import sys
from pathlib import Path

import pytest

from chemx.ui import (
    available_metric_domains,
    discover_evaluable_runs,
    discover_runs,
    filter_evaluable_columns,
    metric_fields_for_domain,
    read_job_state,
    run_stage,
    safe_upload_name,
    save_uploaded_pdf,
    stop_extraction_job,
    tail_text,
)


def test_ui_discovers_newest_runs_first(tmp_path: Path) -> None:
    for name in ("001-old", "002-new"):
        run = tmp_path / name
        run.mkdir()
        (run / "manifest.json").write_text("{}")
    assert [path.name for path in discover_runs(tmp_path)] == ["002-new", "001-old"]


def test_ui_discovers_only_evaluable_runs(tmp_path: Path) -> None:
    for name, domain in (("001-old", "eyedrops"), ("002-new", "benzimidazoles")):
        run = tmp_path / name
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"domain": domain}))

    runs = discover_evaluable_runs(tmp_path, {"benzimidazoles"})

    assert [path.name for path in runs] == ["002-new"]


def test_ui_excludes_domains_without_metrics_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "metrics_benzimidazole_from_single_agent.csv"
    baseline.write_text("field,macro_f1\nsmiles,1\n")

    domains = available_metric_domains(tmp_path)

    assert domains == {"benzimidazoles": baseline}
    assert "eyedrops" not in domains


def test_metric_fields_for_domain_reads_first_column(tmp_path: Path) -> None:
    (tmp_path / "metrics_seltox_from_single_agent.csv").write_text(
        "field,macro_f1\nnp_synthesis,0.5\nzoi_np_mm,0.7\n"
    )

    assert metric_fields_for_domain("seltox", tmp_path) == ["np_synthesis", "zoi_np_mm"]


def test_filter_evaluable_columns_keeps_metric_order() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "extra": ["ignored"],
            "smiles": ["C"],
            "target_value": ["1"],
            "compound_id": ["1a"],
        }
    )

    filtered = filter_evaluable_columns(frame, ["compound_id", "smiles", "target_value"])

    assert list(filtered.columns) == ["compound_id", "smiles", "target_value"]
    assert filtered.to_dict(orient="records") == [
        {"compound_id": "1a", "smiles": "C", "target_value": "1"}
    ]


def test_safe_upload_name_strips_paths_and_nonportable_characters() -> None:
    assert safe_upload_name("../bad file.pdf") == "bad_file.pdf"
    assert safe_upload_name("../../article") == "article.pdf"


def test_save_uploaded_pdf_writes_sanitized_upload(tmp_path: Path) -> None:
    class Uploaded:
        name = "../bad file.pdf"

        def getvalue(self) -> bytes:
            return b"%PDF-1.7"

    path = save_uploaded_pdf(Uploaded(), tmp_path)

    assert path.parent == tmp_path / "_uploads"
    assert path.name.endswith("-bad_file.pdf")
    assert path.read_bytes() == b"%PDF-1.7"


def test_tail_text_returns_last_non_empty_lines(tmp_path: Path) -> None:
    log = tmp_path / "stderr.log"
    log.write_text("one\n\n two \nthree\nfour\n", encoding="utf-8")

    assert tail_text(log, max_lines=2) == ["three", "four"]


def test_run_stage_reports_marker_preprocessing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"state": "prepared"}), encoding="utf-8")

    assert run_stage(run) == "Preparing bundle: Marker layout/OCR/text recognition"


def test_stop_extraction_job_terminates_process_group(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    job_path = tmp_path / "job.json"
    state = {
        "job_path": str(job_path),
        "runs_dir": str(tmp_path),
        "pdf_path": str(tmp_path / "article.pdf"),
        "pid": process.pid,
    }
    job_path.write_text(json.dumps(state), encoding="utf-8")

    try:
        stopped = stop_extraction_job(state, timeout_seconds=2)
        process.wait(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()

    assert stopped
    assert read_job_state(job_path)["status"] == "stopped"
