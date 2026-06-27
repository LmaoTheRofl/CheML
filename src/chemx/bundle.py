from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import fitz

from chemx.chemistry import canonicalize_smiles_required
from chemx.models import (
    ArticleBundle,
    ArticleMetadata,
    BoundingBox,
    FigureBundle,
    LayoutBlock,
    PageBundle,
    TableBundle,
)
from chemx.toolchain import (
    FullStackToolchain,
    ToolchainError,
    ToolStatus,
    run_command,
    write_tool_manifest,
)


def _bbox(rect: Any) -> BoundingBox:
    return BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])


class BundleBuilder:
    def __init__(
        self,
        *,
        render_scale: float = 1.5,
        use_marker: bool = False,
        require_full_stack: bool = False,
        toolchain: FullStackToolchain | None = None,
    ) -> None:
        self.render_scale = render_scale
        self.use_marker = use_marker
        self.require_full_stack = require_full_stack
        self.toolchain = toolchain or FullStackToolchain()

    def build(self, pdf: Path, output_dir: Path) -> ArticleBundle:
        pdf = pdf.resolve()
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError(f"PDF does not exist: {pdf}")
        output_dir.mkdir(parents=True, exist_ok=True)
        assets = output_dir / "assets"
        assets.mkdir(exist_ok=True)
        tool_statuses: list[ToolStatus] = (
            self.toolchain.require() if self.require_full_stack else self.toolchain.check()
        )
        marker_path = (
            self._run_marker(pdf, output_dir, strict=self.require_full_stack)
            if self.use_marker or self.require_full_stack
            else None
        )

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
        if self.require_full_stack:
            self._write_enriched_artifacts(output_dir, bundle, marker_path, tool_statuses)
        return bundle

    def _run_marker(self, pdf: Path, output_dir: Path, *, strict: bool) -> Path | None:
        marker_parts = shlex.split(self.toolchain.marker_command)
        try:
            executable = self.toolchain.marker_executable()
        except ToolchainError as err:
            if strict:
                raise RuntimeError("Marker is required but marker_single is not available") from err
            return None
        marker_dir = output_dir / "marker"
        marker_dir.mkdir(exist_ok=True)
        try:
            subprocess.run(
                [
                    executable,
                    *marker_parts[1:],
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
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if strict:
                raise RuntimeError(f"Marker failed for {pdf}") from exc
            return None
        candidates = sorted(marker_dir.rglob("*.md"))
        if not candidates and strict:
            raise RuntimeError(f"Marker produced no markdown for {pdf}")
        return candidates[0] if candidates else None

    def _write_enriched_artifacts(
        self,
        output_dir: Path,
        bundle: ArticleBundle,
        marker_path: Path | None,
        tool_statuses: list[ToolStatus],
    ) -> None:
        artifacts: dict[str, str | None] = {}
        artifacts["layout"] = self._write_layout_artifact(output_dir, bundle)
        artifacts["marker_markdown"] = self._write_marker_artifacts(output_dir, marker_path)
        artifacts["tables"] = self._write_tables_artifact(output_dir, bundle)
        artifacts["ocr"] = self._write_ocr_artifact(output_dir, bundle)
        artifacts["ocsr"] = self._write_ocsr_artifact(output_dir, bundle)
        artifacts["chemistry_candidates"] = self._write_chemistry_candidates(output_dir, bundle)
        write_tool_manifest(output_dir / "tool_manifest.json", tool_statuses, artifacts=artifacts)

    @staticmethod
    def _write_layout_artifact(output_dir: Path, bundle: ArticleBundle) -> str:
        path = output_dir / "layout.json"
        payload = {
            "schema_version": "1.0",
            "source": "pymupdf+pymupdf_layout",
            "pages": [
                {
                    "page": page.number,
                    "width": page.width,
                    "height": page.height,
                    "reading_order": [
                        {
                            "kind": block.kind,
                            "text": block.text,
                            "bbox": block.bbox.model_dump(),
                        }
                        for block in sorted(
                            page.blocks,
                            key=lambda block: (block.bbox.y0, block.bbox.x0),
                        )
                    ],
                }
                for page in bundle.pages
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path.name

    @staticmethod
    def _write_marker_artifacts(output_dir: Path, marker_path: Path | None) -> str | None:
        if marker_path is None:
            return None
        markdown = output_dir / "marker.md"
        markdown.write_text(marker_path.read_text(encoding="utf-8"), encoding="utf-8")
        metadata = {
            "schema_version": "1.0",
            "source_path": marker_path.relative_to(output_dir).as_posix(),
            "markdown_path": markdown.name,
        }
        (output_dir / "marker.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return markdown.name

    @staticmethod
    def _write_tables_artifact(output_dir: Path, bundle: ArticleBundle) -> str:
        path = output_dir / "tables.json"
        payload = {
            "schema_version": "1.0",
            "tables": [
                {
                    "source": "pymupdf",
                    "page": table.page,
                    "bbox": table.bbox.model_dump(),
                    "caption": table.caption,
                    "rows": table.rows,
                }
                for table in bundle.tables
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path.name

    def _write_ocr_artifact(self, output_dir: Path, bundle: ArticleBundle) -> str:
        path = output_dir / "ocr.json"
        pages: list[dict[str, object]] = []
        for page in bundle.pages:
            if not page.render_path:
                continue
            image = output_dir / page.render_path
            try:
                completed = run_command(self.toolchain.ocr_command(image), timeout=180)
            except Exception as exc:
                raise RuntimeError(f"OCR failed for page {page.number}") from exc
            pages.append(
                {
                    "page": page.number,
                    "asset_path": page.render_path,
                    "text": completed.stdout,
                }
            )
        path.write_text(
            json.dumps({"schema_version": "1.0", "pages": pages}, indent=2),
            encoding="utf-8",
        )
        return path.name

    def _write_ocsr_artifact(self, output_dir: Path, bundle: ArticleBundle) -> str:
        path = output_dir / "ocsr.json"
        entries: list[dict[str, object]] = []
        for figure in bundle.figures:
            image = output_dir / figure.asset_path
            try:
                completed = run_command(self.toolchain.molscribe_command(image), timeout=180)
            except Exception as exc:
                raise RuntimeError(f"MolScribe failed for {figure.asset_path}") from exc
            raw = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
            canonical, valid = canonicalize_smiles_required(raw)
            entries.append(
                {
                    "page": figure.page,
                    "asset_path": figure.asset_path,
                    "bbox": figure.bbox.model_dump() if figure.bbox else None,
                    "raw_smiles": raw,
                    "canonical_smiles": canonical,
                    "valid": valid,
                }
            )
        path.write_text(
            json.dumps({"schema_version": "1.0", "structures": entries}, indent=2),
            encoding="utf-8",
        )
        return path.name

    @staticmethod
    def _write_chemistry_candidates(output_dir: Path, bundle: ArticleBundle) -> str:
        path = output_dir / "chemistry_candidates.json"
        seen: set[str] = set()
        candidates: list[dict[str, object]] = []
        pattern = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{4,}")
        for page in bundle.pages:
            for match in pattern.findall(page.text):
                if match in seen:
                    continue
                seen.add(match)
                try:
                    canonical, valid = canonicalize_smiles_required(match)
                except RuntimeError:
                    raise
                if not valid or canonical is None:
                    continue
                candidates.append(
                    {
                        "source": "text",
                        "page": page.number,
                        "raw": match,
                        "canonical_smiles": canonical,
                    }
                )
                if len(candidates) >= 500:
                    break
        ocsr_path = output_dir / "ocsr.json"
        if ocsr_path.is_file():
            ocsr = json.loads(ocsr_path.read_text(encoding="utf-8"))
            for structure in ocsr.get("structures", []):
                if structure.get("canonical_smiles"):
                    candidates.append(
                        {
                            "source": "ocsr",
                            "page": structure.get("page"),
                            "asset_path": structure.get("asset_path"),
                            "raw": structure.get("raw_smiles"),
                            "canonical_smiles": structure.get("canonical_smiles"),
                        }
                    )
        path.write_text(
            json.dumps({"schema_version": "1.0", "smiles": candidates}, indent=2),
            encoding="utf-8",
        )
        return path.name

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
