from __future__ import annotations

import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from chemx.bundle import BundleBuilder
from chemx.domains import detect_domain, load_domain, project_root
from chemx.evaluate import assert_gold_isolated, write_prediction_csv
from chemx.models import Prediction, RunManifest
from chemx.runner import Backend, CodexBackend, OllamaBackend, install_run_skills
from chemx.validation import deduplicate_prediction, validate_prediction


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


def parse_article(
    pdf: Path,
    *,
    domain: str = "auto",
    backend: Backend | None = None,
    runs_dir: Path | None = None,
    builder: BundleBuilder | None = None,
) -> Path:
    pdf = pdf.resolve()
    root = project_root()
    target = resolve_runs_dir(runs_dir) / _run_id(pdf)
    target.mkdir(parents=True, exist_ok=False)
    selected = detect_domain(pdf) if domain == "auto" else load_domain(domain)
    active_backend = backend or CodexBackend()
    manifest = RunManifest(
        run_id=target.name,
        source_pdf=str(pdf),
        domain=selected.slug,
        backend=active_backend.name,
        state="prepared",
    )
    _write_manifest(target / "manifest.json", manifest)
    try:
        (builder or BundleBuilder()).build(pdf, target)
        manifest.state = "bundled"
        manifest.bundle_path = str(target / "bundle.json")
        _write_manifest(target / "manifest.json", manifest)
        install_run_skills(root, target, selected)
        assert_gold_isolated(target)
        prediction = Prediction.model_validate(active_backend.run(target, selected))
        prediction = deduplicate_prediction(validate_prediction(prediction, selected))
        (target / "prediction.json").write_text(
            prediction.model_dump_json(indent=2), encoding="utf-8"
        )
        write_prediction_csv(
            prediction,
            target / "prediction.csv",
            fields=[field.name for field in selected.fields],
        )
        manifest.state = "inference_complete"
        manifest.prediction_path = str(target / "prediction.json")
        _write_manifest(target / "manifest.json", manifest)
    except Exception as exc:
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
) -> list[Path]:
    pdfs = sorted(dataset_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"no PDF files below {dataset_dir}")
    return [
        parse_article(pdf, backend=backend_from_name(backend_name), runs_dir=runs_dir)
        for pdf in pdfs
    ]
