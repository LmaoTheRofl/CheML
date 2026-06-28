from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chemx.domains import list_domains, project_root
from chemx.pipeline import resolve_runs_dir

METRIC_BASELINES_BY_DOMAIN = {
    "benzimidazoles": "metrics_benzimidazole_from_single_agent.csv",
    "co-crystals": "metrics_cocrystals_from_single_agent.csv",
    "complexes": "metrics_complexes_from_single_agent.csv",
    "cytotox": "metrics_cytotoxicity_from_single_agent.csv",
    "nanomag": "metrics_magnetic_from_single_agent.csv",
    "nanozymes": "metrics_nanozymes_from_single_agent.csv",
    "oxazolidinones": "metrics_oxazolidinone_from_single_agent.csv",
    "seltox": "metrics_seltox_from_single_agent.csv",
    "synergy": "metrics_synergy_from_single_agent.csv",
}
UPLOADS_DIRNAME = "_uploads"
JOBS_DIRNAME = "_ui_jobs"
JOB_POLL_SECONDS = 2.0


def discover_runs(root: Path) -> list[Path]:
    return sorted(
        (path.parent for path in root.glob("*/manifest.json")),
        key=lambda path: path.name,
        reverse=True,
    )


def _manifest_for_run(path: Path) -> dict:
    try:
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def discover_evaluable_runs(root: Path, domains: set[str]) -> list[Path]:
    return [
        run
        for run in discover_runs(root)
        if str(_manifest_for_run(run).get("domain", "")) in domains
    ]


def available_metric_domains(metrics_dir: Path | None = None) -> dict[str, Path]:
    root = metrics_dir or project_root() / "metrics"
    return {
        domain: path
        for domain, filename in METRIC_BASELINES_BY_DOMAIN.items()
        if (path := root / filename).is_file()
    }


def metric_fields_for_domain(domain: str, metrics_dir: Path | None = None) -> list[str]:
    baseline = available_metric_domains(metrics_dir).get(domain)
    if baseline is None:
        raise ValueError(f"no metrics baseline for domain: {domain}")
    with baseline.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[0] for row in reader if row and row[0]]


def filter_evaluable_columns(frame, fields: list[str]):
    selected = [field for field in fields if field in frame.columns]
    return frame.loc[:, selected]


def missing_evaluable_columns(frame, fields: list[str]) -> list[str]:
    return [field for field in fields if field not in frame.columns]


def safe_upload_name(filename: str) -> str:
    name = Path(filename or "article.pdf").name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        safe = "article.pdf"
    if Path(safe).suffix.lower() != ".pdf":
        safe = f"{safe}.pdf"
    return safe


def save_uploaded_pdf(uploaded_file, runs_dir: Path) -> Path:
    uploads_dir = runs_dir / UPLOADS_DIRNAME
    uploads_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = uploads_dir / f"{timestamp}-{safe_upload_name(uploaded_file.name)}"
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    target.write_bytes(data)
    return target


def _job_path(runs_dir: Path, job_id: str) -> Path:
    return runs_dir / JOBS_DIRNAME / f"{job_id}.json"


def read_job_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def update_job_state(path: Path, **updates: Any) -> dict[str, Any]:
    state = read_job_state(path) or {}
    state.update(updates)
    state["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(path, state)
    return state


def start_extraction_job(
    pdf_path: Path,
    *,
    domain: str,
    backend: str,
    runs_dir: Path,
    reviewer: bool,
) -> dict[str, Any]:
    job_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    job_path = _job_path(runs_dir, job_id)
    log_path = job_path.with_suffix(".log")
    state: dict[str, Any] = {
        "job_id": job_id,
        "status": "starting",
        "pdf_path": str(pdf_path),
        "domain": domain,
        "backend": backend,
        "runs_dir": str(runs_dir),
        "reviewer": reviewer,
        "job_path": str(job_path),
        "log_path": str(log_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(job_path, state)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "chemx.ui_job", str(job_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return update_job_state(job_path, pid=process.pid, status="running")


def _pid_status(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        data = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    parts = data.split()
    return parts[2] if len(parts) >= 3 else None


def process_is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    status = _pid_status(pid)
    if status is not None:
        return status != "Z"
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_process_exit(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_is_running(pid):
            return True
        time.sleep(0.1)
    return not process_is_running(pid)


def find_latest_run_for_pdf(runs_dir: Path, pdf_path: Path) -> Path | None:
    source = str(pdf_path.resolve())
    candidates: list[Path] = []
    for manifest_path in runs_dir.glob("*/manifest.json"):
        manifest = _manifest_for_run(manifest_path.parent)
        if manifest.get("source_pdf") == source:
            candidates.append(manifest_path.parent)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def mark_run_stopped(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = _manifest_for_run(run_dir)
    if not manifest or manifest.get("state") == "inference_complete":
        return
    manifest["state"] = "failed"
    manifest["error"] = "stopped by user from Streamlit UI"
    _write_json_atomic(manifest_path, manifest)


def stop_extraction_job(job_state: dict[str, Any], timeout_seconds: float = 5.0) -> bool:
    pid = job_state.get("pid")
    if not isinstance(pid, int):
        return False
    stopped = False
    if process_is_running(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            stopped = True
        except OSError:
            os.kill(pid, signal.SIGTERM)
        stopped = stopped or wait_for_process_exit(pid, timeout_seconds)
        if not stopped and process_is_running(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                stopped = True
            except OSError:
                os.kill(pid, signal.SIGKILL)
            stopped = stopped or wait_for_process_exit(pid, timeout_seconds)
    else:
        stopped = True

    job_path = Path(str(job_state["job_path"]))
    run_dir = find_latest_run_for_pdf(
        Path(str(job_state["runs_dir"])),
        Path(str(job_state["pdf_path"])),
    )
    if run_dir is not None:
        mark_run_stopped(run_dir)
    update_job_state(
        job_path,
        status="stopped",
        run_dir=str(run_dir) if run_dir is not None else job_state.get("run_dir"),
        error="stopped by user",
        stopped_at=datetime.now(UTC).isoformat(),
    )
    return stopped


def refresh_job_state(job_state: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job_state["job_path"]))
    state = read_job_state(path) or job_state
    pid = state.get("pid")
    if state.get("status") == "running" and isinstance(pid, int) and not process_is_running(pid):
        state = update_job_state(
            path,
            status="failed",
            error="extraction process exited before reporting completion",
        )
    if state.get("status") == "running" and not state.get("run_dir"):
        run_dir = find_latest_run_for_pdf(
            Path(str(state["runs_dir"])),
            Path(str(state["pdf_path"])),
        )
        if run_dir is not None:
            state = update_job_state(path, run_dir=str(run_dir))
    return state


def job_is_active(job_state: dict[str, Any] | None) -> bool:
    if not job_state or job_state.get("status") != "running":
        return False
    pid = job_state.get("pid")
    return isinstance(pid, int) and process_is_running(pid)


def tail_text(path: Path, *, max_lines: int = 12) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-max_lines:]


def run_stage(run_dir: Path | None, job_state: dict[str, Any] | None = None) -> str:
    if job_state and job_state.get("status") == "stopped":
        return "Stopped by user"
    if run_dir is None:
        return "Waiting for run workspace"
    manifest = _manifest_for_run(run_dir)
    state = manifest.get("state")
    if state == "prepared":
        return "Preparing bundle: Marker layout/OCR/text recognition"
    if state == "bundled":
        return "Bundle complete: running backend extraction/review"
    if state == "inference_complete":
        return "Inference complete"
    if state in {"failed", "failed_quality_review"}:
        return f"Failed: {manifest.get('error') or 'unknown error'}"
    return f"State: {state or 'unknown'}"


def run_log_lines(run_dir: Path | None, job_state: dict[str, Any] | None = None) -> list[str]:
    lines: list[str] = []
    if run_dir is not None:
        lines.extend(tail_text(run_dir / "marker" / "stderr.log", max_lines=10))
    if job_state and job_state.get("log_path"):
        lines.extend(tail_text(Path(str(job_state["log_path"])), max_lines=8))
    return lines[-12:]


def prediction_frame(run_dir: Path, metrics_dir: Path | None = None):
    import pandas as pd

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    fields = metric_fields_for_domain(str(manifest["domain"]), metrics_dir)
    frame = pd.read_csv(run_dir / "prediction.csv")
    return filter_evaluable_columns(frame, fields), missing_evaluable_columns(frame, fields)


def _run_label(path: Path) -> str:
    manifest = _manifest_for_run(path)
    if not manifest:
        return path.name
    return f"{path.name} · {manifest.get('domain', 'unknown')} · {manifest.get('state', 'unknown')}"


def main(runs_dir: Path | None = None) -> None:
    import streamlit as st

    requested_root = runs_dir or Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    st.set_page_config(page_title="ChemX Parser", layout="wide")
    requested_root = runs_dir or Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    st.set_page_config(page_title="ChemX Parser", layout="wide")
    st.title("ChemX Article Parser")

    try:
        root = resolve_runs_dir(requested_root)
        root.mkdir(parents=True, exist_ok=True)
    except ValueError as exc:
        st.error(str(exc))
        return

    metric_domains = available_metric_domains()
    specs = [spec for spec in list_domains() if spec.slug in metric_domains]
    if not specs:
        st.error("No evaluable ChemX domains found in metrics/")
        return

    active_job = None
    active_job_path = st.session_state.get("active_job_path")
    if active_job_path:
        active_job = read_job_state(Path(str(active_job_path)))
        if active_job:
            active_job = refresh_job_state(active_job)
            if active_job.get("run_dir"):
                st.session_state["selected_run"] = str(active_job["run_dir"])
            if active_job.get("status") in {"completed", "failed", "stopped"}:
                st.session_state.pop("active_job_path", None)

    active = job_is_active(active_job)
    if active_job and active:
        run_dir = Path(str(active_job["run_dir"])) if active_job.get("run_dir") else None
        st.subheader("Active extraction")
        st.caption(run_stage(run_dir, active_job))
        log_lines = run_log_lines(run_dir, active_job)
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        if st.button("Stop extraction", type="primary"):
            stopped = stop_extraction_job(active_job)
            st.session_state.pop("active_job_path", None)
            if stopped:
                st.warning("Extraction stopped")
            else:
                st.error("Stop signal was sent, but the process is still running")
            st.rerun()

    with st.form("chemx_parse_form"):
        uploaded = st.file_uploader("PDF article", type="pdf")
        domain = st.selectbox(
            "Domain",
            [spec.slug for spec in specs],
            format_func=lambda slug: next(
                f"{spec.name} ({spec.slug})" for spec in specs if spec.slug == slug
            ),
        )
        backend = st.radio("Backend", ["codex", "ollama"], horizontal=True)
        reviewer = st.checkbox("Reviewer", value=True)
        submitted = st.form_submit_button("Run extraction", disabled=active)

    if submitted:
        if uploaded is None:
            st.error("Upload a PDF article first")
        else:
            try:
                pdf_path = save_uploaded_pdf(uploaded, root)
                job = start_extraction_job(
                    pdf_path,
                    domain=domain,
                    backend=backend,
                    runs_dir=root,
                    reviewer=reviewer,
                )
                st.session_state["active_job_path"] = str(job["job_path"])
                st.success("Extraction started")
                st.rerun()
            except Exception as exc:  # pragma: no cover - surfaced in Streamlit
                st.exception(exc)

    runs = discover_evaluable_runs(root, set(metric_domains))

    try:
        root = resolve_runs_dir(requested_root)
        root.mkdir(parents=True, exist_ok=True)
    except ValueError as exc:
        st.error(str(exc))
        return

    metric_domains = available_metric_domains()
    specs = [spec for spec in list_domains() if spec.slug in metric_domains]
    if not specs:
        st.error("No evaluable ChemX domains found in metrics/")
        return

    active_job = None
    active_job_path = st.session_state.get("active_job_path")
    if active_job_path:
        active_job = read_job_state(Path(str(active_job_path)))
        if active_job:
            active_job = refresh_job_state(active_job)
            if active_job.get("run_dir"):
                st.session_state["selected_run"] = str(active_job["run_dir"])
            if active_job.get("status") in {"completed", "failed", "stopped"}:
                st.session_state.pop("active_job_path", None)

    active = job_is_active(active_job)
    if active_job and active:
        run_dir = Path(str(active_job["run_dir"])) if active_job.get("run_dir") else None
        st.subheader("Active extraction")
        st.caption(run_stage(run_dir, active_job))
        log_lines = run_log_lines(run_dir, active_job)
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        if st.button("Stop extraction", type="primary"):
            stopped = stop_extraction_job(active_job)
            st.session_state.pop("active_job_path", None)
            if stopped:
                st.warning("Extraction stopped")
            else:
                st.error("Stop signal was sent, but the process is still running")
            st.rerun()

    with st.form("chemx_parse_form"):
        uploaded = st.file_uploader("PDF article", type="pdf")
        domain = st.selectbox(
            "Domain",
            [spec.slug for spec in specs],
            format_func=lambda slug: next(
                f"{spec.name} ({spec.slug})" for spec in specs if spec.slug == slug
            ),
        )
        backend = st.radio("Backend", ["codex", "ollama"], horizontal=True)
        reviewer = st.checkbox("Reviewer", value=True)
        submitted = st.form_submit_button("Run extraction", disabled=active)

    if submitted:
        if uploaded is None:
            st.error("Upload a PDF article first")
        else:
            try:
                pdf_path = save_uploaded_pdf(uploaded, root)
                job = start_extraction_job(
                    pdf_path,
                    domain=domain,
                    backend=backend,
                    runs_dir=root,
                    reviewer=reviewer,
                )
                st.session_state["active_job_path"] = str(job["job_path"])
                st.success("Extraction started")
                st.rerun()
            except Exception as exc:  # pragma: no cover - surfaced in Streamlit
                st.exception(exc)

    runs = discover_evaluable_runs(root, set(metric_domains))
    if not runs:
        st.info(f"No runs found in {root}")
        if active:
            time.sleep(JOB_POLL_SECONDS)
            st.rerun()
        if active:
            time.sleep(JOB_POLL_SECONDS)
            st.rerun()
        return

    selected_default = Path(st.session_state.get("selected_run", runs[0]))
    if selected_default not in runs:
        selected_default = runs[0]
    selected = Path(
        st.selectbox(
            "Run",
            runs,
            index=runs.index(selected_default),
            format_func=_run_label,
        )
    )
    st.session_state["selected_run"] = str(selected)


    selected_default = Path(st.session_state.get("selected_run", runs[0]))
    if selected_default not in runs:
        selected_default = runs[0]
    selected = Path(
        st.selectbox(
            "Run",
            runs,
            index=runs.index(selected_default),
            format_func=_run_label,
        )
    )
    st.session_state["selected_run"] = str(selected)

    manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    st.json(manifest)
    st.caption(run_stage(selected))
    log_lines = run_log_lines(selected)
    if log_lines:
        st.code("\n".join(log_lines), language="text")
    prediction_json_path = selected / "prediction.json"
    prediction_csv_path = selected / "prediction.csv"
    if not prediction_json_path.exists() or not prediction_csv_path.exists():
        st.warning("Inference is not complete")
        if active:
            time.sleep(JOB_POLL_SECONDS)
            st.rerun()
        return
    prediction = json.loads(prediction_json_path.read_text(encoding="utf-8"))
    frame, missing = prediction_frame(selected)
    if missing:
        st.warning(f"Missing evaluable columns: {', '.join(missing)}")
    st.dataframe(frame, use_container_width=True)
    st.download_button(
        "Export CSV",
        frame.to_csv(index=False),
        "prediction_evaluable.csv",
        "text/csv",
    )
    st.download_button(
        "Export JSON",
        prediction_json_path.read_bytes(),
        "prediction.json",
        "application/json",
    )
    rows = [record["values"] for record in prediction["records"]]
    index = st.number_input("Record", min_value=0, max_value=max(0, len(rows) - 1), step=1)
    if rows:
        st.subheader("Evidence")
        st.json(prediction["records"][int(index)].get("evidence", {}))

    if active:
        time.sleep(JOB_POLL_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
