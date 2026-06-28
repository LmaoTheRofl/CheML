from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError


def jpeg_path(image: Path) -> Path:
    return image.with_name(f"{image.stem}.ocr.jpg")


def convert_to_jpeg(image: Path) -> Path:
    destination = jpeg_path(image)
    with Image.open(image) as source:
        source.convert("RGB").save(destination, format="JPEG", quality=95)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--tessdata-prefix", required=True, type=Path)
    parser.add_argument("--language", default="eng")
    parser.add_argument("image", type=Path)
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2
    traineddata = args.tessdata_prefix / "tessdata" / f"{args.language}.traineddata"
    if not traineddata.is_file():
        print(f"traineddata not found: {traineddata}", file=sys.stderr)
        return 2
    try:
        compatible_image = convert_to_jpeg(args.image)
    except (OSError, UnidentifiedImageError) as exc:
        print(f"cannot convert image {args.image}: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(args.tessdata_prefix.resolve())
    completed = subprocess.run(
        [
            args.tesseract,
            str(compatible_image),
            "stdout",
            "-l",
            args.language,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
