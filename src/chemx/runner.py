from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chemx.domains import output_schema
from chemx.evaluate import assert_gold_isolated
from chemx.models import DomainSpec, Prediction


class Backend(Protocol):
    name: str

    def run(self, workspace: Path, spec: DomainSpec) -> Prediction: ...


@dataclass
class CodexBackend:
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    executable: str = "codex"
    timeout_seconds: float = 3600
    name: str = "codex"

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
            str(workspace / "output-schema.json"),
            "--output-last-message",
            str(workspace / "prediction.json"),
            "-C",
            str(workspace),
            self._prompt(),
        ]

    @staticmethod
    def _prompt() -> str:
        return (
            "Extract every ChemX record from bundle.json. Follow the installed chemx-parser and "
            "selected domain skill exactly. Inspect assets when tables or chemical structures are "
            "not recoverable from text. Return only JSON matching output-schema.json. Never access "
            "gold, answers, reference outputs, HuggingFace, or the network."
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
        return Prediction.model_validate_json(
            (workspace / "prediction.json").read_text(encoding="utf-8")
        )


@dataclass
class OllamaBackend(CodexBackend):
    model: str = "gpt-oss:20b"
    name: str = "ollama"

    def command(self, workspace: Path) -> list[str]:
        command = super().command(workspace)
        model_index = command.index("--model")
        command[model_index:model_index + 2] = []
        command[2:2] = ["--oss", "--local-provider", "ollama", "--model", self.model]
        return command


def install_run_skills(project: Path, workspace: Path, spec: DomainSpec) -> None:
    destination = workspace / ".agents" / "skills"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("chemx-parser", spec.slug):
        source = project / ".agents" / "skills" / name
        if not source.is_dir():
            raise FileNotFoundError(f"missing project skill: {source}")
        shutil.copytree(source, destination / name, dirs_exist_ok=True)
    schema = output_schema(spec)
    (workspace / "output-schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
