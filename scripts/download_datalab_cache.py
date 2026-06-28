from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx


BASE_URL = "https://models.datalab.to"
CHECKPOINTS = (
    "layout/2025_09_23",
    "text_detection/2025_05_07",
    "text_recognition/2025_09_23",
    "table_recognition/2025_02_18",
    "ocr_error_detection/2025_02_18",
)
CHUNK_SIZE = 4 * 1024 * 1024


def model_root() -> Path:
    cache_home = Path(
        os.environ.get(
            "XDG_CACHE_HOME",
            Path(__file__).resolve().parents[1] / "runs" / "tools" / "cache",
        )
    )
    return cache_home / "datalab" / "models"


def content_length(url: str) -> int | None:
    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=httpx.Timeout(60, connect=20, read=60),
    ) as client:
        response = client.head(url)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
    return int(value) if value else None


def download(url: str, destination: Path, expected_size: int | None) -> None:
    if destination.is_file() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        print(f"skip {destination.relative_to(model_root())}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 11):
        downloaded = partial.stat().st_size if partial.exists() else 0
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            timeout = httpx.Timeout(60, connect=20, read=30)
            separator = "&" if "?" in url else "?"
            request_url = f"{url}{separator}cachebust={time.time_ns()}-{attempt}"
            with httpx.stream(
                "GET",
                request_url,
                headers=headers,
                follow_redirects=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                if downloaded and response.status_code != 206:
                    partial.unlink(missing_ok=True)
                    downloaded = 0
                with partial.open("ab" if downloaded else "wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)
            break
        except httpx.HTTPError as exc:
            if attempt == 10:
                raise
            size = partial.stat().st_size if partial.exists() else 0
            print(
                f"retry {attempt}/10 {destination.relative_to(model_root())} "
                f"after {type(exc).__name__}, partial={size}",
                flush=True,
            )
            time.sleep(2 * attempt)

    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"bad size for {destination}: {size} != {expected_size}")
    partial.replace(destination)
    print(f"downloaded {destination.relative_to(model_root())} {size}")


def checkpoint_files(root: Path, checkpoint: str) -> list[str]:
    manifest_path = root / checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError(f"invalid manifest files list: {manifest_path}")
    return [str(file) for file in files]


def main() -> int:
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    for checkpoint in CHECKPOINTS:
        files = checkpoint_files(root, checkpoint)
        for file_name in files:
            destination = root / checkpoint / file_name
            if destination.is_file():
                print(f"skip {destination.relative_to(root)}", flush=True)
                continue
            if file_name == "model.safetensors":
                raise FileNotFoundError(
                    f"missing local model weight; automatic weight download is disabled: {destination}"
                )
            url = f"{BASE_URL}/{checkpoint}/{file_name}"
            download(url, destination, expected_size=None)
            time.sleep(0.1)
    print("datalab model cache complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
