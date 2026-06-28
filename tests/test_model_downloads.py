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


class FakeStreamResponse:
    def __init__(self, body: bytes, status_code: int = 206) -> None:
        self.body = body
        self.status_code = status_code

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int):
        yield self.body


class FakeRangeClient:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def __enter__(self) -> "FakeRangeClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, method: str, url: str, headers: dict[str, str]):
        assert method == "GET"
        assert url.startswith("https://example.test/model")
        range_header = headers["Range"]
        self.calls.append(range_header)
        start_text, end_text = range_header.removeprefix("bytes=").split("-")
        start = int(start_text)
        end = int(end_text)
        return FakeStreamResponse(self.payload[start : end + 1])


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


def test_datalab_downloader_fetches_known_size_files_by_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefghi"
    client = FakeRangeClient(payload)
    destination = tmp_path / "model.bin"

    monkeypatch.setattr(datalab, "CHUNK_SIZE", 3)
    monkeypatch.setattr(datalab, "model_root", lambda: tmp_path)
    monkeypatch.setattr(datalab.httpx, "Client", lambda **kwargs: client)

    datalab.download("https://example.test/model.bin", destination, len(payload))

    assert destination.read_bytes() == payload
    assert client.calls == ["bytes=0-2", "bytes=3-5", "bytes=6-8"]


def test_molscribe_downloader_fetches_known_size_files_by_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefghi"
    client = FakeRangeClient(payload)
    destination = tmp_path / "model.pth"

    monkeypatch.setattr(molscribe, "CHUNK_SIZE", 3)
    monkeypatch.setattr(molscribe.httpx, "Client", lambda **kwargs: client)

    molscribe.download("https://example.test/model.pth", destination, len(payload))

    assert destination.read_bytes() == payload
    assert client.calls == ["bytes=0-2", "bytes=3-5", "bytes=6-8"]
