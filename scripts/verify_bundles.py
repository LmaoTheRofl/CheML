from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from chemx.bundle import BundleBuilder
from chemx.domains import detect_domain


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate every local ChemX PDF bundle")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--render-scale", type=float, default=0.35)
    args = parser.parse_args()
    pdfs = sorted(args.dataset.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit("no PDFs found")
    summaries = []
    with tempfile.TemporaryDirectory(prefix="chemx-bundle-check-", dir="/tmp") as temporary:
        root = Path(temporary)
        builder = BundleBuilder(use_marker=False, render_scale=args.render_scale)
        for index, pdf in enumerate(pdfs):
            bundle = builder.build(pdf, root / f"bundle-{index:02d}")
            summaries.append(
                {
                    "pdf": pdf.as_posix(),
                    "domain": detect_domain(pdf).slug,
                    "pages": len(bundle.pages),
                    "blocks": sum(len(page.blocks) for page in bundle.pages),
                    "tables": len(bundle.tables),
                    "figures": len(bundle.figures),
                }
            )
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
