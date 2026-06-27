from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class SourceRef(BaseModel):
    page: int = Field(ge=1)
    kind: Literal[
        "text",
        "table",
        "figure",
        "caption",
        "metadata",
        "layout",
        "marker",
        "ocr",
        "ocsr",
    ]
    bbox: BoundingBox | None = None
    text: str | None = None
    asset_path: str | None = None


class LayoutBlock(BaseModel):
    page: int = Field(ge=1)
    kind: str
    text: str = ""
    bbox: BoundingBox


class TableBundle(BaseModel):
    page: int = Field(ge=1)
    bbox: BoundingBox
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None


class FigureBundle(BaseModel):
    page: int = Field(ge=1)
    bbox: BoundingBox | None = None
    asset_path: str
    caption: str | None = None


class PageBundle(BaseModel):
    number: int = Field(ge=1)
    width: float
    height: float
    text: str
    blocks: list[LayoutBlock] = Field(default_factory=list)
    render_path: str | None = None


class ArticleMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    page_count: int
    sha256: str
    source_name: str


class ArticleBundle(BaseModel):
    schema_version: str = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parser: str
    metadata: ArticleMetadata
    pages: list[PageBundle]
    tables: list[TableBundle] = Field(default_factory=list)
    figures: list[FigureBundle] = Field(default_factory=list)
    marker_markdown_path: str | None = None


Scalar = str | int | float | bool | None


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Scalar]
    evidence: dict[str, list[SourceRef]] = Field(default_factory=dict)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    domain: str
    records: list[PredictionRecord]


class FieldSpec(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "integer", "boolean"] = "string"
    required: bool = True
    enum: list[str] | None = None
    unit: str | None = None
    smiles: bool = False


class DomainSpec(BaseModel):
    slug: str
    name: str
    family: Literal["small_molecule", "nanomaterials"]
    description: str
    aliases: list[str] = Field(default_factory=list)
    fields: list[FieldSpec]

    @field_validator("fields")
    @classmethod
    def unique_fields(cls, value: list[FieldSpec]) -> list[FieldSpec]:
        names = [field.name for field in value]
        if len(names) != len(set(names)):
            raise ValueError("domain field names must be unique")
        return value


class RunManifest(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    source_pdf: str
    domain: str
    backend: str
    state: Literal[
        "prepared",
        "bundled",
        "inference_complete",
        "failed",
        "failed_quality_review",
    ]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    bundle_path: str | None = None
    prediction_path: str | None = None
    error: str | None = None

    @property
    def workspace(self) -> Path:
        return Path(self.bundle_path or self.prediction_path or ".").parent


class ReviewFinding(BaseModel):
    severity: Literal["info", "warning", "error"]
    field: str | None = None
    message: str


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    status: Literal["pass", "needs_retry", "fail"]
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
