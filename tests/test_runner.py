from pathlib import Path

import pytest

from chemx.domains import load_domain
from chemx.runner import CodexBackend, OllamaBackend, install_run_skills


class FakeAdapterProcess:
    def __init__(self, returncode=None) -> None:
        self.returncode = returncode
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, value) -> None:
        self.signals.append(value)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_codex_command_has_isolation_and_structured_output(tmp_path: Path) -> None:
    command = CodexBackend().command(tmp_path)
    assert command[:2] == ["codex", "exec"]
    model_index = command.index("--model")
    assert ["--model", "gpt-5.5"] == command[model_index : model_index + 2]
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert "--output-schema" in command
    assert 'model_reasoning_effort="xhigh"' in command


def test_ollama_uses_same_output_contract(tmp_path: Path) -> None:
    command = OllamaBackend().command(tmp_path)
    assert "--oss" in command
    assert command[command.index("--local-provider") + 1] == "ollama"
    assert command[command.index("--model") + 1] == "lukaspetrik/gemma3-tools:27b"
    assert "--output-schema" not in command
    assert command[command.index("--model") + 1] == "lukaspetrik/gemma3-tools:27b"
    assert "--output-schema" not in command
    assert "--output-last-message" in command
    assert "model_instructions_file=" in " ".join(command)
    assert (tmp_path / "gemma-codex-instructions.txt").is_file()


def test_ollama_backend_uses_local_adapter(tmp_path: Path) -> None:
    env = OllamaBackend().environment(tmp_path / "runs" / "one")
    assert env["OLLAMA_HOST"] == "http://127.0.0.1:11434"


def test_ollama_accepts_markdown_fenced_prediction() -> None:
    prediction = OllamaBackend._validate_prediction(
        '```json\n{"schema_version":"1.0","domain":"eyedrops","records":[]}\n```'
    )
    assert prediction.domain == "eyedrops"
    assert prediction.records == []


def test_codex_backend_sets_writable_codex_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "auth.json").write_text("{}")
    (codex_home / "installation_id").write_text("test")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    env = CodexBackend().environment(tmp_path / "runs" / "one")

    assert env["CODEX_HOME"] == str((tmp_path / "runs" / ".codex-home").resolve())
    assert env["XDG_CACHE_HOME"] == str((tmp_path / "runs" / ".codex-cache").resolve())
    assert (tmp_path / "runs" / ".codex-home" / "auth.json").is_file()
    assert (tmp_path / "runs" / ".codex-home" / "installation_id").is_file()


def test_codex_backend_cleans_local_codex_home_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    workspace = tmp_path / "runs" / "one"
    workspace.mkdir(parents=True)
    (workspace / "output-schema.json").write_text("{}")

    def fake_run(command, **kwargs):
        payload = '{"schema_version":"1.0","domain":"eyedrops","records":[]}'
        (workspace / "prediction.json").write_text(payload)
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("chemx.runner.subprocess.run", fake_run)

    CodexBackend().run(workspace, load_domain("eyedrops"))

    assert not (tmp_path / "runs" / ".codex-home").exists()


def test_codex_backend_resolves_windows_command_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "runs" / "one"
    workspace.mkdir(parents=True)
    (workspace / "output-schema.json").write_text("{}")
    resolved = str(tmp_path / "codex.CMD")

    def fake_run(command, **kwargs):
        assert command[0] == resolved
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        payload = '{"schema_version":"1.0","domain":"eyedrops","records":[]}'
        (workspace / "prediction.json").write_text(payload)
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("chemx.runner.shutil.which", lambda *args, **kwargs: resolved)
    monkeypatch.setattr("chemx.runner.subprocess.run", fake_run)

    prediction = CodexBackend().run(workspace, load_domain("eyedrops"))

    assert prediction.domain == "eyedrops"


def test_run_workspace_gets_only_router_and_selected_skill(tmp_path: Path) -> None:
    install_run_skills(Path.cwd(), tmp_path, load_domain("seltox"))
    installed = {path.name for path in (tmp_path / ".agents" / "skills").iterdir()}
    assert installed == {"chemx-parser", "seltox"}
    assert (tmp_path / "domain.json").is_file()
    assert (tmp_path / "domain.json").is_file()
    assert (tmp_path / "output-schema.json").is_file()
