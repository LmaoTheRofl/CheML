from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chemx.pipeline import backend_from_name, parse_article


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_state(path: Path, **updates: Any) -> dict[str, Any]:
    state = _read_state(path)
    state.update(updates)
    state["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(path, state)
    return state


def run_job(job_path: Path) -> Path:
    state = _update_state(job_path, status="running", worker_pid=os.getpid())
    print(
        "Starting ChemX extraction "
        f"domain={state['domain']} backend={state['backend']} pdf={state['pdf_path']}",
        flush=True,
    )
    run_dir = parse_article(
        Path(str(state["pdf_path"])),
        domain=str(state["domain"]),
        backend=backend_from_name(str(state["backend"])),
        runs_dir=Path(str(state["runs_dir"])),
        skip_reviewer=not bool(state["reviewer"]),
    )
    _update_state(
        job_path,
        status="completed",
        run_dir=str(run_dir),
        completed_at=datetime.now(UTC).isoformat(),
    )
    print(f"Completed ChemX extraction run_dir={run_dir}", flush=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m chemx.ui_job <job.json>", file=sys.stderr)
        return 2
    job_path = Path(args[0])
    try:
        run_job(job_path)
    except BaseException as exc:
        _update_state(
            job_path,
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
            failed_at=datetime.now(UTC).isoformat(),
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
