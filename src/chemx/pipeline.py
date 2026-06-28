from __future__ import annotations

import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from chemx.bundle import BundleBuilder
from chemx.domains import detect_domain, load_domain, project_root
from chemx.evaluate import assert_gold_isolated, write_prediction_csv
from chemx.models import Prediction, ReviewResult, RunManifest
from chemx.runner import (
    Backend,
    CodexBackend,
    CodexReviewer,
    DeterministicReviewer,
    OllamaBackend,
    Reviewer,
    backend_runtime,
    install_run_skills,
)
from chemx.validation import (
    deduplicate_prediction,
    repair_and_canonicalize_prediction,
    validate_prediction,
)


def _run_id(pdf: Path) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", pdf.stem).strip("-")[:48].lower()
    return f"{timestamp}-{stem}"


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def backend_from_name(name: str) -> Backend:
    if name == "codex":
        return CodexBackend()
    if name == "ollama":
        return OllamaBackend()
    raise ValueError(f"unsupported backend: {name}")


def _is_production_backend(backend: Backend) -> bool:
    return backend.name in {"codex", "ollama"}


def _default_reviewer(backend: Backend) -> Reviewer:
    if backend.name == "codex":
        return CodexReviewer()
    return DeterministicReviewer()


def _candidate_count(workspace: Path) -> int:
    count = 0
    tables_path = workspace / "tables.json"
    if tables_path.is_file():
        tables = json.loads(tables_path.read_text(encoding="utf-8"))
        for table in tables.get("tables", []):
            rows = table.get("rows") or []
            count += max(0, len(rows) - 1)
    chemistry_path = workspace / "chemistry_candidates.json"
    if chemistry_path.is_file():
        chemistry = json.loads(chemistry_path.read_text(encoding="utf-8"))
        count += len(chemistry.get("smiles", []))
    ocr_path = workspace / "ocr.json"
    if ocr_path.is_file():
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        count += sum(1 for page in ocr.get("pages", []) if str(page.get("text", "")).strip())
    return count


def _write_quality_flags(workspace: Path, flags: list[dict[str, object]]) -> None:
    (workspace / "quality_flags.json").write_text(
        json.dumps({"schema_version": "1.0", "flags": flags}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _prepare_prediction(
    prediction: Prediction,
    selected,
    target: Path,
    *,
    require_rdkit: bool,
) -> Prediction:
    prediction = repair_and_canonicalize_prediction(
        prediction,
        selected,
        target,
        require_rdkit=require_rdkit,
    )
    return deduplicate_prediction(validate_prediction(prediction, selected))


def _write_prediction_artifacts(prediction: Prediction, selected, target: Path) -> None:
    (target / "prediction.json").write_text(
        prediction.model_dump_json(indent=2), encoding="utf-8"
    )
    write_prediction_csv(
        prediction,
        target / "prediction.csv",
        fields=[field.name for field in selected.fields],
    )


def _run_review(
    reviewer: Reviewer,
    target: Path,
    selected,
) -> ReviewResult:
    return reviewer.review(target, selected)


def resolve_runs_dir(runs_dir: Path | None = None) -> Path:
    root = project_root()
    resolved = runs_dir or root / "runs"
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.expanduser().resolve()
    tmp_roots = {
        Path("/tmp").resolve(),
        Path("/var/tmp").resolve(),
        Path("/dev/shm").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    for tmp_root in tmp_roots:
        if resolved == tmp_root or tmp_root in resolved.parents:
            raise ValueError(
                f"run output directory must be persistent and cannot be inside {tmp_root}"
            )
    return resolved


def _complete_inference(
    target: Path,
    selected,
    active_backend: Backend,
    active_reviewer: Reviewer | None,
    manifest: RunManifest,
    *,
    production: bool,
) -> None:
    prediction = _prepare_prediction(
        Prediction.model_validate(active_backend.run(target, selected)),
        selected,
        target,
        require_rdkit=production,
    )
    candidate_count = _candidate_count(target)
    if production and not prediction.records and candidate_count:
        feedback_path = target / "reviewer_feedback.md"
        empty_feedback = (
            "The previous extraction returned records=[] despite non-empty tables/OCR/"
            "chemistry candidates. Re-extract all domain rows from the artifacts and do "
            "not return an empty records array."
        )
        if feedback_path.is_file():
            existing_feedback = feedback_path.read_text(encoding="utf-8").strip()
            if existing_feedback and empty_feedback not in existing_feedback:
                empty_feedback = existing_feedback + "\n\n" + empty_feedback
        feedback_path.write_text(empty_feedback, encoding="utf-8")
        prediction = _prepare_prediction(
            Prediction.model_validate(active_backend.run(target, selected)),
            selected,
            target,
            require_rdkit=production,
        )
    _write_prediction_artifacts(prediction, selected, target)
    if production and not prediction.records and candidate_count:
        _write_quality_flags(
            target,
            [
                {
                    "flag": "empty_prediction_with_candidates",
                    "candidate_count": candidate_count,
                }
            ],
        )
        manifest.state = "failed_quality_review"
        manifest.error = "empty prediction with non-empty extraction candidates"
        _write_manifest(target / "manifest.json", manifest)
        raise RuntimeError(manifest.error)
    if active_reviewer is None:
        manifest.state = "inference_complete"
        manifest.prediction_path = str(target / "prediction.json")
        manifest.error = None
        _write_manifest(target / "manifest.json", manifest)
        return
    review = _run_review(active_reviewer, target, selected)
    if review.status == "fail":
        _write_quality_flags(
            target,
            [{"flag": "reviewer_failed", "summary": review.summary}],
        )
        manifest.state = "failed_quality_review"
        manifest.error = f"reviewer failed extraction: {review.summary}"
        _write_manifest(target / "manifest.json", manifest)
        raise RuntimeError(manifest.error)
    if production and review.status == "needs_retry":
        (target / "reviewer_feedback.md").write_text(
            review.summary
            + "\n\n"
            + "\n".join(f"- {finding.message}" for finding in review.findings),
            encoding="utf-8",
        )
        prediction = _prepare_prediction(
            Prediction.model_validate(active_backend.run(target, selected)),
            selected,
            target,
            require_rdkit=production,
        )
        _write_prediction_artifacts(prediction, selected, target)
        review = _run_review(active_reviewer, target, selected)
        if review.status != "pass":
            _write_quality_flags(
                target,
                [
                    {
                        "flag": "reviewer_not_passed_after_retry",
                        "status": review.status,
                        "summary": review.summary,
                    }
                ],
            )
            manifest.state = "failed_quality_review"
            manifest.error = f"reviewer did not pass after retry: {review.summary}"
            _write_manifest(target / "manifest.json", manifest)
            raise RuntimeError(manifest.error)
    manifest.state = "inference_complete"
    manifest.prediction_path = str(target / "prediction.json")
    manifest.error = None
    _write_manifest(target / "manifest.json", manifest)


def resume_article(
    run: Path,
    *,
    backend: Backend | None = None,
    reviewer: Reviewer | None = None,
    skip_reviewer: bool = False,
) -> Path:
    target = run.expanduser().resolve()
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest.json: {target}")
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    selected = load_domain(manifest.domain)
    active_backend = backend or backend_from_name(manifest.backend)
    production = _is_production_backend(active_backend)
    active_reviewer: Reviewer | None = (
        None if skip_reviewer else reviewer or _default_reviewer(active_backend)
    )
    required = ["bundle.json"]
    if production:
        required.extend(
            [
                "layout.json",
                "marker.md",
                "marker.json",
                "tables.json",
                "ocr.json",
                "ocsr.json",
                "chemistry_candidates.json",
                "tool_manifest.json",
            ]
        )
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise ValueError(f"run is not ready for inference resume; missing: {', '.join(missing)}")
    manifest.backend = active_backend.name
    manifest.state = "bundled"
    manifest.bundle_path = str(target / "bundle.json")
    manifest.error = None
    _write_manifest(manifest_path, manifest)
    install_run_skills(project_root(), target, selected)
    assert_gold_isolated(target)
    try:
        with backend_runtime(active_backend):
            _complete_inference(
                target,
                selected,
                active_backend,
                active_reviewer,
                manifest,
                production=production,
            )
    except Exception as exc:
        if manifest.state != "failed_quality_review":
            manifest.state = "failed"
        manifest.error = str(exc)
        _write_manifest(manifest_path, manifest)
        raise
    return target


def parse_article(
    pdf: Path,
    *,
    domain: str = "auto",
    backend: Backend | None = None,
    runs_dir: Path | None = None,
    builder: BundleBuilder | None = None,
    reviewer: Reviewer | None = None,
    skip_reviewer: bool = False,
) -> Path:
    pdf = pdf.resolve()
    root = project_root()
    target = resolve_runs_dir(runs_dir) / _run_id(pdf)
    target.mkdir(parents=True, exist_ok=False)
    selected = detect_domain(pdf) if domain == "auto" else load_domain(domain)
    active_backend = backend or CodexBackend()
    production = _is_production_backend(active_backend)
    active_reviewer: Reviewer | None = (
        None if skip_reviewer else reviewer or _default_reviewer(active_backend)
    )
    manifest = RunManifest(
        run_id=target.name,
        source_pdf=str(pdf),
        domain=selected.slug,
        backend=active_backend.name,
        state="prepared",
    )
    _write_manifest(target / "manifest.json", manifest)
    try:
        (builder or BundleBuilder(require_full_stack=production)).build(pdf, target)
        manifest.state = "bundled"
        manifest.bundle_path = str(target / "bundle.json")
        _write_manifest(target / "manifest.json", manifest)
        install_run_skills(root, target, selected)
        assert_gold_isolated(target)
        with backend_runtime(active_backend):
            _complete_inference(
                target,
                selected,
                active_backend,
                active_reviewer,
                manifest,
                production=production,
            )
    except Exception as exc:
        if manifest.state != "failed_quality_review":
            manifest.state = "failed"
        manifest.error = str(exc)
        _write_manifest(target / "manifest.json", manifest)
        raise
    return target


def batch_articles(
    dataset_dir: Path,
    *,
    backend_name: str = "codex",
    runs_dir: Path | None = None,
    skip_reviewer: bool = False,
) -> list[Path]:
    pdfs = sorted(dataset_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"no PDF files below {dataset_dir}")
    return [
        parse_article(
            pdf,
            backend=backend_from_name(backend_name),
            runs_dir=runs_dir,
            skip_reviewer=skip_reviewer,
        )
        for pdf in pdfs
    ]
