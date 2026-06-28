from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

DEFAULT_URL = (
    "https://huggingface.co/yujieq/MolScribe/resolve/main/"
    "swin_base_char_aux_1m680k.pth"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("CHEMX_MOLSCRIBE_MODEL_PATH", "swin_base_char_aux_1m680k.pth")
)
CHUNK_SIZE = 4 * 1024 * 1024
DOWNLOAD_ATTEMPTS = int(os.environ.get("CHEMX_MODEL_DOWNLOAD_ATTEMPTS", "20"))
READ_TIMEOUT_SECONDS = float(os.environ.get("CHEMX_MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS", "180"))
CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("CHEMX_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECONDS", "20")
)


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
        f"retry {attempt}/{DOWNLOAD_ATTEMPTS} {destination} "
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
        print(f"skip {destination}")
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
    print(f"downloaded {destination} {size}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("CHEMX_MOLSCRIBE_MODEL_URL", DEFAULT_URL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    download(args.url, args.output, expected_size=content_length(args.url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
