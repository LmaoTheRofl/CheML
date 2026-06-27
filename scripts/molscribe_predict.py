from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    from molscribe import MolScribe

    model = MolScribe(str(args.model))
    result = model.predict_image_file(str(args.image), return_confidence=True)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    smiles = str(result.get("smiles") or "").strip()
    if not smiles:
        return 1
    print(smiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
