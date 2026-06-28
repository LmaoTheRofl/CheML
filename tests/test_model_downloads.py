import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


datalab = load_script("download_datalab_cache", ROOT / "scripts/download_datalab_cache.py")
molscribe = load_script("download_molscribe_model", ROOT / "scripts/download_molscribe_model.py")


def test_datalab_cache_downloads_manifests_and_weights_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datalab, "CHECKPOINTS", ("layout/test",))
    monkeypatch.setattr(datalab, "model_root", lambda: tmp_path)
    monkeypatch.setattr(datalab, "content_length", lambda url: 12)

    def fake_download(url: str, destination: Path, expected_size: int | None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "manifest.json":
            destination.write_text(
                json.dumps({"files": ["config.json", "model.safetensors"]}),
                encoding="utf-8",
            )
        else:
            destination.write_bytes(f"{url}:{expected_size}".encode())

    monkeypatch.setattr(datalab, "download", fake_download)

    datalab.sync_cache(include_weights=True)

    assert (tmp_path / "layout/test/manifest.json").is_file()
    assert (tmp_path / "layout/test/config.json").is_file()
    assert (tmp_path / "layout/test/model.safetensors").is_file()


def test_datalab_cache_requires_explicit_weight_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datalab, "CHECKPOINTS", ("layout/test",))
    monkeypatch.setattr(datalab, "model_root", lambda: tmp_path)
    monkeypatch.setattr(datalab, "content_length", lambda url: 12)

    def fake_download(url: str, destination: Path, expected_size: int | None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "manifest.json":
            destination.write_text(json.dumps({"files": ["model.safetensors"]}), encoding="utf-8")

    monkeypatch.setattr(datalab, "download", fake_download)

    with pytest.raises(FileNotFoundError, match="--include-weights"):
        datalab.sync_cache(include_weights=False)


def test_molscribe_downloader_uses_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path, int | None]] = []
    output = tmp_path / "swin_base_char_aux_1m680k.pth"
    monkeypatch.setattr(molscribe, "content_length", lambda url: 123)
    monkeypatch.setattr(
        molscribe,
        "download",
        lambda url, destination, expected_size: calls.append((url, destination, expected_size)),
    )

    assert molscribe.main(["--url", "https://example.test/model.pth", "--output", str(output)]) == 0

    assert calls == [("https://example.test/model.pth", output, 123)]
