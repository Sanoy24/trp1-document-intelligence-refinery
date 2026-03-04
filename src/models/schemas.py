"""
Pydantic schemas for the Document Intelligence Refinery.

All core data models for the 5-stage pipeline:
  - DocumentProfile (Stage 1: Triage)
  - ExtractedDocument / ExtractedPage (Stage 2: Extraction)
  - LDU (Stage 3: Semantic Chunking)
  - PageIndexNode (Stage 4: PageIndex Builder)
  - ProvenanceChain (Stage 5: Query Agent)
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    OTHER = "other"


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    OTHER = "other"


# ─────────────────────────────────────────────────────────────
# Spatial primitives
# ─────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Bounding box in PDF points (72 dpi). Origin = bottom-left of page."""
    x0: float = Field(ge=0.0)
    y0: float = Field(ge=0.0)
    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    page_number: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_box_coords(self) -> "BoundingBox":
        """Ensure x0 <= x1 and y0 <= y1."""
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0
        return self

    @computed_field
    @property
    def area(self) -> float:
        return abs(self.x1 - self.x0) * abs(self.y1 - self.y0)


# ─────────────────────────────────────────────────────────────
# Stage 1: Triage — Document Profiling
# ─────────────────────────────────────────────────────────────

class PageSignal(BaseModel):
    """Per-page signals collected during triage."""
    page_number: int = Field(ge=1)
    char_count: int = Field(default=0, ge=0)
    char_density: float = Field(default=0.0, ge=0.0)
    word_count: int = Field(default=0, ge=0)
    image_area_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    font_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    has_text_layer: bool = True
    whitespace_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_hint: str = "fast_text" # fast_text | layout_model | vision_model


class DocumentProfile(BaseModel):
    """
    Output of the Triage Agent (Stage 1).
    Governs which extraction strategy downstream stages will use.
    Stored at .refinery/profiles/{doc_id}.json
    """
    doc_id: str
    filename: str
    file_size_bytes: int = Field(default=0, ge=0)
    page_count: int = Field(default=0, ge=0)

    # Classification dimensions
    origin_type: OriginType = OriginType.NATIVE_DIGITAL
    layout_complexity: LayoutComplexity = LayoutComplexity.SINGLE_COLUMN
    language: str = "en"
    language_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_hint: DomainHint = DomainHint.GENERAL
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    # Aggregate signals
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    pages_below_threshold: int = Field(default=0, ge=0)
    per_page_signals: list[PageSignal] = Field(default_factory=list)

    # Metadata
    profiled_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Stage 2: Structure Extraction
# ─────────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    """A contiguous block of text with spatial coordinates."""
    content: str
    bbox: Optional[BoundingBox] = None
    block_type: BlockType = BlockType.PARAGRAPH
    reading_order: int = 0


class TableCell(BaseModel):
    """A single cell in a table."""
    text: str = ""
    row: int = 0
    col: int = 0
    row_span: int = 1
    col_span: int = 1


class ExtractedTable(BaseModel):
    """A table extracted as structured data."""
    table_id: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[TableCell]] = Field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    caption: Optional[str] = None
    page_number: int = 0


class FigureBlock(BaseModel):
    """A figure/image with optional caption."""
    figure_id: str = ""
    caption: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    image_ref: Optional[str] = None    # path to extracted image file
    page_number: int = 0


class ExtractedPage(BaseModel):
    """All extracted content from a single page."""
    page_number: int = Field(ge=1)
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    figures: list[FigureBlock] = Field(default_factory=list)
    raw_text: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy_used: str = "fast_text"


class ExtractedDocument(BaseModel):
    """
    Normalized output of any extraction strategy (A, B, or C).
    This is the canonical intermediate representation all strategies
    must produce before passing to the Chunking Engine.
    """
    doc_id: str
    filename: str
    pages: list[ExtractedPage] = Field(default_factory=list)
    strategy_used: str = "fast_text"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_s: float = Field(default=0.0, ge=0.0)
    needs_review: bool = False          # True when final strategy still low confidence
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def total_tables(self) -> int:
        return sum(len(p.tables) for p in self.pages)

    @computed_field
    @property
    def total_figures(self) -> int:
        return sum(len(p.figures) for p in self.pages)

    @computed_field
    @property
    def total_text_blocks(self) -> int:
        return sum(len(p.text_blocks) for p in self.pages)


# ─────────────────────────────────────────────────────────────
# Stage 3: Semantic Chunking — Logical Document Units
# ─────────────────────────────────────────────────────────────

class LDU(BaseModel):
    """
    Logical Document Unit — a semantically coherent, self-contained
    chunk that preserves structural context. RAG-ready.
    """
    ldu_id: str = ""
    content: str
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    page_refs: list[int] = Field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    parent_section: Optional[str] = None
    token_count: int = Field(default=0, ge=0)
    content_hash: str = ""
    related_chunks: list[str] = Field(
        default_factory=list,
        description="IDs of related LDUs (cross-references, e.g. 'see Table 3')",
    )

    @field_validator("page_refs")
    @classmethod
    def validate_page_refs(cls, v: list[int]) -> list[int]:
        """Ensure all page references are positive."""
        for page in v:
            if page < 1:
                raise ValueError(f"page_ref must be >= 1, got {page}")
        return v

    def compute_hash(self) -> str:
        """SHA-256 hash of content for provenance verification."""
        self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return self.content_hash


# ─────────────────────────────────────────────────────────────
# Stage 4: PageIndex
# ─────────────────────────────────────────────────────────────

class PageIndexNode(BaseModel):
    """A node in the hierarchical document navigation tree."""
    title: str
    page_start: int
    page_end: int
    child_sections: list[PageIndexNode] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    summary: str = ""                  # LLM-generated 2-3 sentence summary
    data_types_present: list[str] = Field(default_factory=list)  # tables, figures, equations


# ─────────────────────────────────────────────────────────────
# Stage 5: Provenance
# ─────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single source citation with spatial provenance."""
    doc_name: str
    page_number: int
    bbox: Optional[BoundingBox] = None
    content_hash: str = ""
    excerpt: str = ""                  # short text snippet from source


class ProvenanceChain(BaseModel):
    """Full provenance trail for a query answer."""
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    verified: bool = False
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Extraction Ledger Entry
# ─────────────────────────────────────────────────────────────

class LedgerEntry(BaseModel):
    """A single log entry in the extraction ledger."""
    doc_id: str
    filename: str
    strategy_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    cost_estimate_usd: float = Field(default=0.0, ge=0.0)
    processing_time_s: float = Field(default=0.0, ge=0.0)
    pages_processed: int = Field(default=0, ge=0)
    escalated_from: Optional[str] = None  # which strategy it escalated from
    needs_review: bool = False             # flagged for manual review
    timestamp: datetime = Field(default_factory=datetime.utcnow)
