from pathlib import Path
from types import SimpleNamespace

import pytest

import chemx.toolchain as toolchain_module
from chemx.bundle import BundleBuilder
from chemx.toolchain import FullStackToolchain, ToolchainError


def test_full_stack_toolchain_reports_missing_configured_commands() -> None:
    toolchain = FullStackToolchain(
        marker_command="definitely-missing-marker",
        ocr_command="definitely-missing-ocr",
        molscribe_command="definitely-missing-molscribe",
        codex_command="definitely-missing-codex",
    )

    statuses = {status.name: status for status in toolchain.check()}

    assert statuses["marker"].available is False
    assert statuses["ocr"].available is False
    assert statuses["molscribe"].available is False
    assert statuses["codex"].available is False
    with pytest.raises(ToolchainError, match="mandatory ChemX parser toolchain"):
        toolchain.require()


def test_toolchain_handles_unquoted_executable_paths_with_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "Program Files" / "Tesseract-OCR" / "tesseract.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        toolchain_module,
        "os",
        SimpleNamespace(name="nt", environ=toolchain_module.os.environ),
    )
    toolchain = FullStackToolchain(
        ocr_command=f"{executable.as_posix()} {{image}} stdout -l eng"
    )

    command = toolchain.ocr_command(Path("page.png"))

    assert command == [executable.as_posix(), "page.png", "stdout", "-l", "eng"]


def test_toolchain_discovers_project_local_ocr_and_molscribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "runs" / "tools" / "molscribe-py39" / "bin" / "python"
    adapter = tmp_path / "scripts" / "molscribe_predict.py"
    weights = tmp_path / "swin_base_char_aux_1m680k.pth"
    traineddata = tmp_path / "runs" / "tools" / "tesseract" / "tessdata" / "eng.traineddata"
    for path in (python, adapter, weights, traineddata):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"installed")
    monkeypatch.delenv("CHEMX_OCR_COMMAND", raising=False)
    monkeypatch.delenv("CHEMX_MOLSCRIBE_COMMAND", raising=False)
    monkeypatch.setattr(toolchain_module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(toolchain_module, "_which", lambda command: f"/bin/{command}")

    toolchain = FullStackToolchain()

    assert toolchain.ocr_command(Path("page.png")) == [
        toolchain_module.sys.executable,
        "-m",
        "chemx.tesseract_ocr",
        "--tessdata-prefix",
        str(tmp_path / "runs" / "tools" / "tesseract"),
        "page.png",
    ]
    assert toolchain.molscribe_command(Path("figure.png")) == [
        str(python),
        str(adapter),
        "--model",
        str(weights),
        "figure.png",
    ]


def test_toolchain_reports_missing_molscribe_model(tmp_path: Path) -> None:
    python = tmp_path / "molscribe-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"installed")
    missing = tmp_path / "missing.pth"
    toolchain = FullStackToolchain(
        molscribe_command=f"{python} scripts/molscribe_predict.py --model {missing} {{image}}"
    )

    statuses = {status.name: status for status in toolchain.check()}

    assert statuses["molscribe"].available is False
    assert "missing model file" in statuses["molscribe"].detail


def test_strict_bundle_requires_full_stack_before_fallback(tmp_path: Path) -> None:
    pdf = tmp_path / "article.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%invalid-for-toolchain-check\n")
    builder = BundleBuilder(
        require_full_stack=True,
        toolchain=FullStackToolchain(
            marker_command="definitely-missing-marker",
            ocr_command="definitely-missing-ocr",
            molscribe_command="definitely-missing-molscribe",
            codex_command="definitely-missing-codex",
        ),
    )

    with pytest.raises(ToolchainError):
        builder.build(pdf, tmp_path / "run")
