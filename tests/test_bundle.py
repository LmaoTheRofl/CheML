import subprocess
from pathlib import Path

import fitz
import pytest

import chemx.bundle as bundle_module
from chemx.bundle import BundleBuilder
from chemx.models import ArticleBundle


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((30, 40), "Compound A permeability 1.25 cm/s")
    document.set_metadata({"title": "Synthetic ChemX article", "author": "Tests"})
    document.save(path)
    document.close()


def test_bundle_contains_text_layout_render_and_metadata(tmp_path: Path) -> None:
    pdf = tmp_path / "article.pdf"
    make_pdf(pdf)
    output = tmp_path / "bundle"
    bundle = BundleBuilder(use_marker=False, render_scale=1.0).build(pdf, output)
    loaded = ArticleBundle.model_validate_json((output / "bundle.json").read_text())
    assert bundle.parser == "pymupdf-fallback"
    assert loaded.metadata.title == "Synthetic ChemX article"
    assert loaded.metadata.page_count == 1
    assert "permeability" in loaded.pages[0].text
    assert loaded.pages[0].blocks
    assert (output / loaded.pages[0].render_path).is_file()


def test_marker_uses_writable_project_cache_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "article.pdf"
    make_pdf(pdf)
    output = tmp_path / "bundle"
    output.mkdir()
    font = tmp_path / "DejaVuSans.ttf"
    font.write_bytes(b"font")
    captured: dict[str, object] = {}

    class FakeToolchain:
        marker_command = "marker_single"

        def marker_executable(self) -> str:
            return "marker_single"

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["env"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        captured["stdout_name"] = Path(kwargs["stdout"].name).name  # type: ignore[union-attr]
        captured["stderr_name"] = Path(kwargs["stderr"].name).name  # type: ignore[union-attr]
        marker_dir = Path(command[command.index("--output_dir") + 1])
        (marker_dir / "article.md").write_text("# ok", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("FONT_PATH", raising=False)
    monkeypatch.delenv("PARALLEL_DOWNLOAD_WORKERS", raising=False)
    monkeypatch.setattr(bundle_module, "MARKER_FONT_CANDIDATES", (font,))
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        BundleBuilder,
        "_require_marker_model_cache",
        lambda self, env: None,
    )

    result = BundleBuilder(toolchain=FakeToolchain())._run_marker(
        pdf,
        output,
        strict=True,
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert Path(env["XDG_CACHE_HOME"]).parts[-3:] == ("runs", "tools", "cache")
    assert Path(env["XDG_CACHE_HOME"]).is_dir()
    assert env["FONT_PATH"] == str(font.resolve())
    assert env["PARALLEL_DOWNLOAD_WORKERS"] == "1"
    assert captured["timeout"] == 10800
    assert captured["stdout_name"] == "pages-0001-0001.stdout.log"
    assert captured["stderr_name"] == "pages-0001-0001.stderr.log"
    assert result == output / "marker" / "marker.md"


def test_marker_timeout_can_be_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHEMX_MARKER_TIMEOUT_SECONDS", "7200")
    builder = BundleBuilder()
    assert builder.marker_timeout_seconds == 7200


def test_marker_refuses_incomplete_model_cache_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "article.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    output = tmp_path / "bundle"
    output.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("marker subprocess should not run with incomplete cache")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="Marker model cache is incomplete"):
        BundleBuilder()._run_marker(pdf, output, strict=True)
