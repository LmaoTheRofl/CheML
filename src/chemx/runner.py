from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from chemx.domains import output_schema
from chemx.evaluate import assert_gold_isolated
from chemx.models import DomainSpec, Prediction, ReviewResult

GEMMA_CODEX_INSTRUCTIONS = """You are a tool-using ChemX extraction agent.
You are already inside a prepared ChemX run directory. The required artifacts are
local files in the current directory. Do not install packages, do not use pip, do
not use the network, and do not claim that the chemx package is unavailable.
Use the supplied tools whenever the task depends on workspace files. Never invent
file contents or command results. Use local shell/Python stdlib commands to read
JSON, Markdown, and CSV files when needed. Never print a complete artifact: filter
with rg/sed or parse JSON in Python and print only concise, relevant summaries.
Keep every tool result below 200 lines and 20 KB. For chemistry candidates, print
only counts and short molecules relevant to the extracted drug/coformer names.
Emit shell calls exactly as:
<tool_call>
{"name":"exec_command","parameters":{"cmd":"COMMAND"}}
</tool_call>
Never wrap tool calls in Markdown. Wait for each tool result before continuing.
Return only the final JSON requested by the user when extraction is complete:
no prose, no Markdown fences, no explanation, and never `{}`.
The final object must have exactly the top-level keys `schema_version`, `domain`,
and `records`. Each record must have exactly `values` and `evidence`; `values`
must contain every field name from domain.json and `evidence` must be an object.
Never return bundle metadata keys or an ad-hoc simplified record schema.
"""


class Backend(Protocol):
    name: str

    def run(self, workspace: Path, spec: DomainSpec) -> Prediction: ...


class Reviewer(Protocol):
    name: str

    def review(self, workspace: Path, spec: DomainSpec) -> ReviewResult: ...


@dataclass
class CodexBackend:
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    executable: str = "codex"
    timeout_seconds: float = 3600
    name: str = "codex"

    def command(self, workspace: Path) -> list[str]:
        prompt = self._prompt()
        feedback = workspace / "reviewer_feedback.md"
        if feedback.is_file():
            prompt = (
                prompt
                + "\n\nMandatory reviewer_feedback.md contents:\n"
                + feedback.read_text(encoding="utf-8")
            )
        return [
            self.executable,
            "exec",
            "--model",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--output-schema",
            str(workspace / "output-schema.json"),
            "--output-last-message",
            str(workspace / "prediction.json"),
            "-C",
            str(workspace),
            prompt,
        ]

    @staticmethod
    def _prompt() -> str:
        return (
            "Extract every ChemX record from bundle.json, layout.json, marker.md, marker.json, "
            "tables.json, ocr.json, ocsr.json, and chemistry_candidates.json. Follow the installed "
            "chemx-parser and selected domain skill exactly. Use the exact domain schema and never "
            "rename columns. Do not return an empty records array when tables, OCR text, OCSR "
            "structures, chemistry candidates, compounds, targets, bacteria, metals, coformers, or "
            "photostability candidates are present. If reviewer_feedback.md exists, address it. "
            "Return only JSON matching output-schema.json. Never access gold, answers, reference "
            "outputs, HuggingFace, or the network."
        )

    def environment(self, workspace: Path) -> dict[str, str]:
        env = os.environ.copy()
        source_codex_home = Path(env.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
        local_codex_home = workspace.parent / ".codex-home"
        if source_codex_home.is_dir():
            local_codex_home.mkdir(parents=True, exist_ok=True)
            local_codex_home.chmod(0o700)
            for name in ("auth.json", "installation_id"):
                source = source_codex_home / name
                if source.is_file():
                    target = local_codex_home / name
                    shutil.copy2(source, target)
                    target.chmod(0o600)
            env["CODEX_HOME"] = str(local_codex_home.resolve())
        if "XDG_CACHE_HOME" not in env:
            cache_home = workspace.parent / ".codex-cache"
            cache_home.mkdir(parents=True, exist_ok=True)
            env["XDG_CACHE_HOME"] = str(cache_home.resolve())
        return env

    def run(self, workspace: Path, spec: DomainSpec) -> Prediction:
        assert_gold_isolated(workspace)
        env = self.environment(workspace)
        command = self.command(workspace)
        command[0] = shutil.which(command[0], path=env.get("PATH")) or command[0]
        local_codex_home = (workspace.parent / ".codex-home").resolve()
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        finally:
            if env.get("CODEX_HOME") == str(local_codex_home):
                shutil.rmtree(local_codex_home, ignore_errors=True)
        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.splitlines()[-20:])
            raise RuntimeError(f"codex exec failed ({completed.returncode}):\n{tail}")
        assert_gold_isolated(workspace)
        return self._validate_prediction(
            (workspace / "prediction.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def _validate_prediction(text: str) -> Prediction:
        return Prediction.model_validate_json(text)


@dataclass
class OllamaBackend(CodexBackend):
    model: str = "lukaspetrik/gemma3-tools:27b"
    startup_timeout_seconds: float = 30
    name: str = "ollama"

    def _adapter_address(self) -> tuple[str, int]:
        value = os.environ.get("CHEMX_OLLAMA_ADAPTER_URL", "http://127.0.0.1:11434")
        parsed = urlsplit(value if "://" in value else f"http://{value}")
        return parsed.hostname or "127.0.0.1", parsed.port or 11434

    def _adapter_ready(self) -> bool:
        try:
            with socket.create_connection(self._adapter_address(), timeout=0.25):
                return True
        except OSError:
            return False

    @staticmethod
    def _stop_adapter(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _start_adapter(self) -> subprocess.Popen[str] | None:
        if self._adapter_ready():
            return None
        ollama_bin = os.environ.get("OLLAMA_BIN") or shutil.which("ollama")
        if not ollama_bin:
            fallback = Path.home() / ".local" / "bin" / "ollama"
            ollama_bin = str(fallback) if fallback.is_file() else None
        if not ollama_bin:
            raise RuntimeError("Ollama executable not found; set OLLAMA_BIN")
        host, port = self._adapter_address()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "chemx.ollama_adapter",
                "--listen",
                f"{host}:{port}",
                "--upstream",
                os.environ.get("OLLAMA_UPSTREAM_URL", "127.0.0.1:11435"),
                "--ollama-bin",
                ollama_bin,
            ],
            env=os.environ.copy(),
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"ChemX Ollama adapter exited with code {process.returncode}"
                )
            if self._adapter_ready():
                return process
            time.sleep(0.2)
        self._stop_adapter(process)
        raise TimeoutError("ChemX Ollama adapter did not become ready")

    @contextmanager
    def runtime(self) -> Iterator[None]:
        process = self._start_adapter()
        try:
            yield
        finally:
            if process is not None:
                self._stop_adapter(process)

    def command(self, workspace: Path) -> list[str]:
        instructions = workspace / "gemma-codex-instructions.txt"
        instructions.write_text(GEMMA_CODEX_INSTRUCTIONS, encoding="utf-8")
        command = super().command(workspace)
        model_index = command.index("--model")
        command[model_index:model_index + 2] = []
        command[2:2] = ["--oss", "--local-provider", "ollama", "--model", self.model]
        schema_index = command.index("--output-schema")
        command[schema_index:schema_index + 2] = []
        command[2:2] = [
            "-c",
            f"model_instructions_file={json.dumps(str(instructions.resolve()))}",
        ]
        return command

    def environment(self, workspace: Path) -> dict[str, str]:
        env = super().environment(workspace)
        env["OLLAMA_HOST"] = env.get("CHEMX_OLLAMA_ADAPTER_URL", "http://127.0.0.1:11434")
        return env

    @staticmethod
    def _validate_prediction(text: str) -> Prediction:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1]
            candidate = candidate.rsplit("```", 1)[0].strip()
        object_start = candidate.find("{")
        if object_start < 0:
            return Prediction.model_validate_json(candidate)
        payload, _ = json.JSONDecoder().raw_decode(candidate[object_start:])
        for record in payload.get("records", []):
            if record.get("evidence") == []:
                record["evidence"] = {}
            elif isinstance(record.get("evidence"), dict):
                record["evidence"] = {
                    field: (
                        []
                        if isinstance(refs, str)
                        or (
                            isinstance(refs, list)
                            and all(isinstance(ref, str) for ref in refs)
                        )
                        else refs
                    )
                    for field, refs in record["evidence"].items()
                }
        return Prediction.model_validate(payload)


@contextmanager
def backend_runtime(backend: Backend) -> Iterator[None]:
    runtime = getattr(backend, "runtime", None)
    if runtime is None:
        yield
        return
    with runtime():
        yield


REVIEW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "const": "1.0"},
        "status": {"type": "string", "enum": ["pass", "needs_retry", "fail"]},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                    "field": {"type": ["string", "null"]},
                    "message": {"type": "string"},
                },
                "required": ["severity", "field", "message"],
            },
        },
    },
    "required": ["schema_version", "status", "summary", "findings"],
}


@dataclass
class CodexReviewer:
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    executable: str = "codex"
    timeout_seconds: float = 1800
    name: str = "codex-reviewer"

    def command(self, workspace: Path) -> list[str]:
        return [
            self.executable,
            "exec",
            "--model",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--output-schema",
            str(workspace / "review-schema.json"),
            "--output-last-message",
            str(workspace / "review.json"),
            "-C",
            str(workspace),
            self._prompt(),
        ]

    @staticmethod
    def _prompt() -> str:
        return (
            "Review the ChemX extraction result. Read prediction.json, prediction.csv, "
            "domain.json, output-schema.json, bundle.json, layout.json, marker.md/json, "
            "tables.json, ocr.json, ocsr.json, chemistry_candidates.json, "
            "schema_diagnostics.json, and "
            "chemistry_diagnostics.json. Never access gold, reference.csv, parquet files, "
            "HuggingFace, or the network. Check schema compliance, non-empty extraction when "
            "candidates exist, RDKit canonical SMILES, evidence quality, missed candidate rows, "
            "numeric precision, and hallucinations against artifacts. Return only review-schema "
            "JSON."
        )

    def review(self, workspace: Path, spec: DomainSpec) -> ReviewResult:
        del spec
        assert_gold_isolated(workspace)
        (workspace / "review-schema.json").write_text(
            json.dumps(REVIEW_SCHEMA, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        backend = CodexBackend(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            executable=self.executable,
            timeout_seconds=self.timeout_seconds,
        )
        env = backend.environment(workspace)
        command = self.command(workspace)
        command[0] = shutil.which(command[0], path=env.get("PATH")) or command[0]
        local_codex_home = (workspace.parent / ".codex-home").resolve()
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        finally:
            if env.get("CODEX_HOME") == str(local_codex_home):
                shutil.rmtree(local_codex_home, ignore_errors=True)
        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.splitlines()[-20:])
            raise RuntimeError(f"codex reviewer failed ({completed.returncode}):\n{tail}")
        result = ReviewResult.model_validate_json(
            (workspace / "review.json").read_text(encoding="utf-8")
        )
        (workspace / "review_report.md").write_text(
            "# ChemX reviewer report\n\n"
            f"Status: {result.status}\n\n"
            f"{result.summary}\n\n"
            + "\n".join(
                f"- {finding.severity}: "
                f"{finding.field + ': ' if finding.field else ''}{finding.message}"
                for finding in result.findings
            ),
            encoding="utf-8",
        )
        return result


@dataclass
class DeterministicReviewer:
    name: str = "deterministic-reviewer"

    def review(self, workspace: Path, spec: DomainSpec) -> ReviewResult:
        prediction = Prediction.model_validate_json(
            (workspace / "prediction.json").read_text(encoding="utf-8")
        )
        findings = []
        if prediction.domain != spec.slug:
            findings.append(
                {"severity": "error", "field": None, "message": "prediction domain mismatch"}
            )
        expected = {field.name for field in spec.fields}
        for index, record in enumerate(prediction.records):
            missing = expected - set(record.values)
            unknown = set(record.values) - expected
            if missing or unknown:
                findings.append(
                    {
                        "severity": "error",
                        "field": None,
                        "message": f"record {index} schema mismatch",
                    }
                )
        result = ReviewResult(
            status="fail" if findings else "pass",
            summary="deterministic schema review",
            findings=findings,
        )
        (workspace / "review.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (workspace / "review_report.md").write_text(
            f"# ChemX reviewer report\n\nStatus: {result.status}\n",
            encoding="utf-8",
        )
        return result


def install_run_skills(project: Path, workspace: Path, spec: DomainSpec) -> None:
    destination = workspace / ".agents" / "skills"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("chemx-parser", spec.slug):
        source = project / ".agents" / "skills" / name
        if not source.is_dir():
            raise FileNotFoundError(f"missing project skill: {source}")
        shutil.copytree(source, destination / name, dirs_exist_ok=True)
    shutil.copy2(
        project / ".agents" / "skills" / spec.slug / "domain.json",
        workspace / "domain.json",
    )
    schema = output_schema(spec)
    (workspace / "output-schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
