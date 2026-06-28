from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

import chemx.tesseract_ocr as ocr


def _tessdata(tmp_path: Path) -> Path:
    prefix = tmp_path / "tesseract"
    data = prefix / "tessdata" / "eng.traineddata"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"model")
    return prefix


def _fake_success(
    captured: dict[str, object],
):
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "recognized\n", "")

    return run


def test_main_converts_png_and_runs_tesseract(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    image = tmp_path / "page.png"
    Image.new("RGBA", (20, 10), (255, 255, 255, 128)).save(image)
    captured: dict[str, object] = {}
    monkeypatch.setattr(ocr.subprocess, "run", _fake_success(captured))

    assert ocr.main(["--tessdata-prefix", str(_tessdata(tmp_path)), str(image)]) == 0
    converted = tmp_path / "page.ocr.jpg"
    assert converted.is_file()
    with Image.open(converted) as loaded:
        assert loaded.mode == "RGB"
        assert loaded.size == (20, 10)
    assert capsys.readouterr().out == "recognized\n"
    assert captured["command"][1] == str(converted)  # type: ignore[index]


def test_main_accepts_one_pixel_grayscale_image(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "tiny.png"
    Image.new("L", (1, 1), 255).save(image)
    monkeypatch.setattr(ocr.subprocess, "run", _fake_success({}))
    assert ocr.main(["--tessdata-prefix", str(_tessdata(tmp_path)), str(image)]) == 0


def test_main_rejects_missing_image(tmp_path: Path, capsys) -> None:
    result = ocr.main(
        ["--tessdata-prefix", str(_tessdata(tmp_path)), str(tmp_path / "missing.png")]
    )
    assert result == 2
    assert "image not found" in capsys.readouterr().err


def test_main_rejects_missing_traineddata(tmp_path: Path, capsys) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)
    result = ocr.main(["--tessdata-prefix", str(tmp_path / "absent"), str(image)])
    assert result == 2
    assert "traineddata not found" in capsys.readouterr().err


def test_main_propagates_tesseract_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image)

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7, "", "ocr failed\n")

    monkeypatch.setattr(ocr.subprocess, "run", fail)
    result = ocr.main(["--tessdata-prefix", str(_tessdata(tmp_path)), str(image)])
    assert result == 7
    assert "ocr failed" in capsys.readouterr().err
