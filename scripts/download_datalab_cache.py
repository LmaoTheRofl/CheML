from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
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
DOWNLOAD_ATTEMPTS = int(os.environ.get("CHEMX_MODEL_DOWNLOAD_ATTEMPTS", "20"))
READ_TIMEOUT_SECONDS = float(os.environ.get("CHEMX_MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS", "180"))
CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("CHEMX_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECONDS", "20")
)


def model_root() -> Path:
    cache_home = Path(
        os.environ.get(
            "XDG_CACHE_HOME",
            Path(__file__).resolve().parents[1] / "runs" / "tools" / "cache",
        )
    )
    return cache_home / "datalab" / "models"


def content_length(url: str) -> int | None:
    try:
        with httpx.Client(
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=httpx.Timeout(60, connect=20, read=60),
        ) as client:
            response = client.head(url)
            response.raise_for_status()
            value = response.headers.get("Content-Length")
    except httpx.HTTPError:
        return None
    return int(value) if value else None


def cachebusted_url(url: str, attempt: int) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}cachebust={time.time_ns()}-{attempt}"


def retry_delay(attempt: int) -> int:
    return min(30, 2 * attempt)


def request_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        READ_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
        read=READ_TIMEOUT_SECONDS,
    )


def retry_message(destination: Path, attempt: int, exc: Exception, partial: Path) -> None:
    size = partial.stat().st_size if partial.exists() else 0
    print(
        f"retry {attempt}/{DOWNLOAD_ATTEMPTS} {destination.relative_to(model_root())} "
        f"after {type(exc).__name__}, partial={size}",
        flush=True,
    )


def download_known_size(url: str, destination: Path, partial: Path, expected_size: int) -> None:
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    with httpx.Client(
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        timeout=request_timeout(),
    ) as client:
        while offset < expected_size:
            range_end = min(offset + CHUNK_SIZE - 1, expected_size - 1)
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                try:
                    with client.stream(
                        "GET",
                        cachebusted_url(url, attempt),
                        headers={"Range": f"bytes={offset}-{range_end}"},
                    ) as response:
                        if response.status_code == 200 and offset == 0:
                            mode = "wb"
                        elif response.status_code == 206:
                            mode = "ab"
                        else:
                            response.raise_for_status()
                            raise RuntimeError(
                                f"range request failed for {destination}: "
                                f"HTTP {response.status_code}"
                            )

                        with partial.open(mode) as handle:
                            if mode == "wb":
                                offset = 0
                            for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    handle.write(chunk)
                                    offset += len(chunk)

                    if response.status_code == 206 and offset != range_end + 1:
                        raise RuntimeError(
                            f"short read for {destination}: {offset} < {range_end + 1}"
                        )
                    break
                except (httpx.HTTPError, RuntimeError) as exc:
                    offset = partial.stat().st_size if partial.exists() else 0
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise
                    retry_message(destination, attempt, exc, partial)
                    time.sleep(retry_delay(attempt))


def download_unknown_size(url: str, destination: Path, partial: Path) -> None:
    with httpx.Client(
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        timeout=request_timeout(),
    ) as client:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            downloaded = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            try:
                with client.stream(
                    "GET",
                    cachebusted_url(url, attempt),
                    headers=headers,
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
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                retry_message(destination, attempt, exc, partial)
                time.sleep(retry_delay(attempt))


def download(url: str, destination: Path, expected_size: int | None) -> None:
    if destination.is_file() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        print(f"skip {destination.relative_to(model_root())}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if expected_size is None:
        download_unknown_size(url, destination, partial)
    else:
        download_known_size(url, destination, partial, expected_size)

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


def sync_cache(*, include_weights: bool) -> None:
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    for checkpoint in CHECKPOINTS:
        manifest_url = f"{BASE_URL}/{checkpoint}/manifest.json"
        manifest_path = root / checkpoint / "manifest.json"
        download(manifest_url, manifest_path, expected_size=content_length(manifest_url))
        files = checkpoint_files(root, checkpoint)
        for file_name in files:
            destination = root / checkpoint / file_name
            if destination.is_file():
                print(f"skip {destination.relative_to(root)}", flush=True)
                continue
            if file_name == "model.safetensors" and not include_weights:
                raise FileNotFoundError(
                    "missing Marker model weight; rerun with --include-weights: "
                    f"{destination}"
                )
            url = f"{BASE_URL}/{checkpoint}/{file_name}"
            download(url, destination, expected_size=content_length(url))
            time.sleep(0.1)
    print("datalab model cache complete")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Download large Marker/Surya model.safetensors files.",
    )
    args = parser.parse_args(argv)
    sync_cache(include_weights=args.include_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
