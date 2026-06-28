from __future__ import annotations

import hashlib
import json
import os
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

SURYA_MARKER_CHECKPOINTS = (
    "layout/2025_09_23",
    "text_detection/2025_05_07",
    "text_recognition/2025_09_23",
    "table_recognition/2025_02_18",
    "ocr_error_detection/2025_02_18",
)
MARKER_FONT_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "runs" / "tools" / "fonts" / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _bbox(rect: Any) -> BoundingBox:
    return BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


class BundleBuilder:
    def __init__(
        self,
        *,
        render_scale: float = 1.5,
        use_marker: bool = False,
        require_full_stack: bool = False,
        toolchain: FullStackToolchain | None = None,
        marker_timeout_seconds: float | None = None,
    ) -> None:
        self.render_scale = render_scale
        self.use_marker = use_marker
        self.require_full_stack = require_full_stack
        self.toolchain = toolchain or FullStackToolchain()
        self.marker_timeout_seconds = (
            marker_timeout_seconds
            if marker_timeout_seconds is not None
            else float(os.environ.get("CHEMX_MARKER_TIMEOUT_SECONDS", "10800"))
        )

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
        env = os.environ.copy()
        if not env.get("XDG_CACHE_HOME"):
            cache_home = Path(__file__).resolve().parents[2] / "runs" / "tools" / "cache"
            cache_home.mkdir(parents=True, exist_ok=True)
            env["XDG_CACHE_HOME"] = str(cache_home)
        if not env.get("FONT_PATH"):
            font = next((path for path in MARKER_FONT_CANDIDATES if path.is_file()), None)
            if font is not None:
                env["FONT_PATH"] = str(font.resolve())
        env.setdefault("PARALLEL_DOWNLOAD_WORKERS", "1")
        try:
            self._require_marker_model_cache(env)
        except RuntimeError:
            if strict:
                raise
            return None
        if strict:
            return self._run_marker_chunked(pdf, marker_dir, executable, marker_parts, env)
        command = self._marker_command(executable, marker_parts, pdf, marker_dir)
        ok, _detail = self._run_marker_process(command, marker_dir, "full", env)
        if not ok:
            return None
        candidates = sorted(marker_dir.rglob("*.md"))
        return candidates[0] if candidates else None

    def _run_marker_chunked(
        self,
        pdf: Path,
        marker_dir: Path,
        executable: str,
        marker_parts: list[str],
        env: dict[str, str],
    ) -> Path:
        document = fitz.open(pdf)
        try:
            page_count = document.page_count
        finally:
            document.close()
        if page_count < 1:
            raise RuntimeError(f"Marker produced no markdown for {pdf}")
        chunk_size = self._marker_chunk_size()
        parts: list[str] = []
        for start in range(0, page_count, chunk_size):
            end = min(start + chunk_size - 1, page_count - 1)
            parts.extend(
                self._run_marker_page_range(
                    pdf,
                    marker_dir,
                    executable,
                    marker_parts,
                    env,
                    start,
                    end,
                    allow_page_retry=chunk_size > 1,
                )
            )
        combined = marker_dir / "marker.md"
        combined.write_text("\n\n".join(parts), encoding="utf-8")
        return combined

    def _run_marker_page_range(
        self,
        pdf: Path,
        marker_dir: Path,
        executable: str,
        marker_parts: list[str],
        env: dict[str, str],
        start: int,
        end: int,
        *,
        allow_page_retry: bool,
    ) -> list[str]:
        label = f"pages-{start + 1:04d}-{end + 1:04d}"
        page_range = str(start) if start == end else f"{start}-{end}"
        run_dir = marker_dir / label
        run_dir.mkdir(exist_ok=True)
        command = [
            *self._marker_command(executable, marker_parts, pdf, run_dir),
            "--page_range",
            page_range,
        ]
        ok, detail = self._run_marker_process(command, marker_dir, label, env)
        if not ok and allow_page_retry:
            texts: list[str] = []
            for page_index in range(start, end + 1):
                texts.extend(
                    self._run_marker_page_range(
                        pdf,
                        marker_dir,
                        executable,
                        marker_parts,
                        env,
                        page_index,
                        page_index,
                        allow_page_retry=False,
                    )
                )
            return texts
        if not ok:
            raise RuntimeError(f"Marker failed for {pdf} page range {page_range}: {detail}")
        candidates = sorted(run_dir.rglob("*.md"))
        if not candidates:
            raise RuntimeError(f"Marker produced no markdown for {pdf} page range {page_range}")
        markdown = candidates[0].read_text(encoding="utf-8")
        return [f"<!-- Marker pages {start + 1}-{end + 1} -->\n\n{markdown}"]

    @staticmethod
    def _marker_command(
        executable: str,
        marker_parts: list[str],
        pdf: Path,
        output_dir: Path,
    ) -> list[str]:
        return [
            executable,
            *marker_parts[1:],
            str(pdf),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "markdown",
        ]

    @staticmethod
    def _marker_chunk_size() -> int:
        raw = os.environ.get("CHEMX_MARKER_PAGE_CHUNK_SIZE", "1")
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    def _run_marker_process(
        self,
        command: list[str],
        log_dir: Path,
        label: str,
        env: dict[str, str],
    ) -> tuple[bool, str]:
        stdout_path = log_dir / f"{label}.stdout.log"
        stderr_path = log_dir / f"{label}.stderr.log"
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_log:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout_log,
                    stderr=stderr_log,
                    text=True,
                    timeout=self.marker_timeout_seconds,
                    env=env,
                )
        except subprocess.TimeoutExpired as exc:
            detail = f"timed out after {exc.timeout}s; see {stdout_path} and {stderr_path}"
            return False, detail
        if completed.returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            return (
                False,
                f"exit {completed.returncode}; see {stdout_path} and {stderr_path}; "
                f"stderr={_tail(stderr)}",
            )
        return True, ""

    @staticmethod
    def _require_marker_model_cache(env: dict[str, str]) -> None:
        cache_home = Path(env["XDG_CACHE_HOME"])
        model_root = cache_home / "datalab" / "models"
        missing: list[str] = []
        for checkpoint in SURYA_MARKER_CHECKPOINTS:
            checkpoint_dir = model_root / checkpoint
            manifest_path = checkpoint_dir / "manifest.json"
            if not manifest_path.is_file():
                missing.append(f"{checkpoint}/manifest.json")
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid Marker model manifest: {manifest_path}") from exc
            for file_name in manifest.get("files", []):
                if not (checkpoint_dir / file_name).is_file():
                    missing.append(f"{checkpoint}/{file_name}")
        if missing:
            preview = "\n".join(f"- {item}" for item in missing[:40])
            extra = "" if len(missing) <= 40 else f"\n... and {len(missing) - 40} more"
            raise RuntimeError(
                "Marker model cache is incomplete; refusing to let Marker auto-download "
                f"large model files. Missing files:\n{preview}{extra}"
            )

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
