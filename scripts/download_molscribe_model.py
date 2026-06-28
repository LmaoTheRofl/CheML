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


def download(url: str, destination: Path, expected_size: int | None) -> None:
    if destination.is_file() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        print(f"skip {destination}")
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
            separator = "&" if "?" in url else "?"
            request_url = f"{url}{separator}cachebust={time.time_ns()}-{attempt}"
            with httpx.stream(
                "GET",
                request_url,
                headers=headers,
                follow_redirects=True,
                timeout=httpx.Timeout(60, connect=20, read=30),
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
                f"retry {attempt}/10 {destination} after {type(exc).__name__}, "
                f"partial={size}",
                flush=True,
            )
            time.sleep(2 * attempt)

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
