from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import fitz

from chemx.models import (
    ArticleBundle,
    ArticleMetadata,
    BoundingBox,
    FigureBundle,
    LayoutBlock,
    PageBundle,
    TableBundle,
)


def _bbox(rect: Any) -> BoundingBox:
    return BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])


class BundleBuilder:
    def __init__(self, *, render_scale: float = 1.5, use_marker: bool = True) -> None:
        self.render_scale = render_scale
        self.use_marker = use_marker

    def build(self, pdf: Path, output_dir: Path) -> ArticleBundle:
        pdf = pdf.resolve()
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError(f"PDF does not exist: {pdf}")
        output_dir.mkdir(parents=True, exist_ok=True)
        assets = output_dir / "assets"
        assets.mkdir(exist_ok=True)
        marker_path = self._run_marker(pdf, output_dir) if self.use_marker else None

        pages: list[PageBundle] = []
        tables: list[TableBundle] = []
        figures: list[FigureBundle] = []
        document = fitz.open(pdf)
        try:
            for index, page in enumerate(document):
                number = index + 1
                blocks = self._blocks(page, number)
                render = assets / f"page-{number:04d}.png"
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(self.render_scale, self.render_scale), alpha=False
                )
                pixmap.save(render)
                pages.append(
                    PageBundle(
                        number=number,
                        width=page.rect.width,
                        height=page.rect.height,
                        text=page.get_text("text"),
                        blocks=blocks,
                        render_path=render.relative_to(output_dir).as_posix(),
                    )
                )
                page_tables = self._tables(page, number)
                for table in page_tables:
                    table.caption = self._nearest_caption(blocks, table.bbox, "table")
                tables.extend(page_tables)
                page_figures = self._figures(document, page, number, assets, output_dir)
                for figure in page_figures:
                    if figure.bbox is not None:
                        figure.caption = self._nearest_caption(blocks, figure.bbox, "figure")
                figures.extend(page_figures)

            metadata = document.metadata or {}
            bundle = ArticleBundle(
                parser="marker+pymupdf" if marker_path else "pymupdf-fallback",
                metadata=ArticleMetadata(
                    title=metadata.get("title") or None,
                    author=metadata.get("author") or None,
                    subject=metadata.get("subject") or None,
                    keywords=metadata.get("keywords") or None,
                    creator=metadata.get("creator") or None,
                    producer=metadata.get("producer") or None,
                    page_count=document.page_count,
                    sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    source_name=pdf.name,
                ),
                pages=pages,
                tables=tables,
                figures=figures,
                marker_markdown_path=(
                    marker_path.relative_to(output_dir).as_posix() if marker_path else None
                ),
            )
        finally:
            document.close()
        (output_dir / "bundle.json").write_text(
            bundle.model_dump_json(indent=2), encoding="utf-8"
        )
        return bundle

    def _run_marker(self, pdf: Path, output_dir: Path) -> Path | None:
        executable = shutil.which("marker_single")
        if executable is None:
            return None
        marker_dir = output_dir / "marker"
        marker_dir.mkdir(exist_ok=True)
        try:
            subprocess.run(
                [
                    executable,
                    str(pdf),
                    "--output_dir",
                    str(marker_dir),
                    "--output_format",
                    "markdown",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        candidates = sorted(marker_dir.rglob("*.md"))
        return candidates[0] if candidates else None

    @staticmethod
    def _blocks(page: fitz.Page, number: int) -> list[LayoutBlock]:
        result: list[LayoutBlock] = []
        for raw in page.get_text("blocks"):
            if len(raw) < 7 or not str(raw[4]).strip():
                continue
            kind = "text" if raw[6] == 0 else "image"
            result.append(
                LayoutBlock(page=number, kind=kind, text=str(raw[4]).strip(), bbox=_bbox(raw[:4]))
            )
        return result

    @staticmethod
    def _tables(page: fitz.Page, number: int) -> list[TableBundle]:
        if not hasattr(page, "find_tables"):
            return []
        try:
            found = page.find_tables()
            return [
                TableBundle(
                    page=number,
                    bbox=_bbox(table.bbox),
                    rows=[
                        ["" if cell is None else str(cell) for cell in row]
                        for row in table.extract()
                    ],
                )
                for table in found.tables
            ]
        except (AttributeError, TypeError, ValueError):
            return []

    @staticmethod
    def _figures(
        document: fitz.Document,
        page: fitz.Page,
        number: int,
        assets: Path,
        output_dir: Path,
    ) -> list[FigureBundle]:
        result: list[FigureBundle] = []
        seen: set[int] = set()
        for ordinal, image in enumerate(page.get_images(full=True), start=1):
            xref = int(image[0])
            if xref in seen:
                continue
            seen.add(xref)
            try:
                raw = document.extract_image(xref)
                extension = raw.get("ext", "bin")
                path = assets / f"page-{number:04d}-image-{ordinal:03d}.{extension}"
                path.write_bytes(raw["image"])
                rects = page.get_image_rects(xref)
                result.append(
                    FigureBundle(
                        page=number,
                        bbox=_bbox(rects[0]) if rects else None,
                        asset_path=path.relative_to(output_dir).as_posix(),
                    )
                )
            except (KeyError, RuntimeError, ValueError):
                continue
        return result

    @staticmethod
    def _nearest_caption(
        blocks: list[LayoutBlock], bbox: BoundingBox, kind: str
    ) -> str | None:
        pattern = r"^\s*table\s+\w+" if kind == "table" else r"^\s*(fig(?:ure)?\.?\s+\w+)"
        candidates = [
            block
            for block in blocks
            if re.search(pattern, block.text, flags=re.IGNORECASE)
            and block.bbox.y0 <= bbox.y1 + 80
            and block.bbox.y1 >= bbox.y0 - 120
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda block: abs(block.bbox.y0 - bbox.y0))
        return nearest.text
