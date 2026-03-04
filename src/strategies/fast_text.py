"""
Strategy A — Fast Text Extraction (Cost: Low).

Uses pdfplumber for fast, direct text extraction with per-page
confidence scoring. Suitable for native digital PDFs with
single-column or simple layouts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pdfplumber

from src.config import get_confidence_thresholds, get_page_signal_thresholds
from src.models.schemas import (
    BoundingBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
    TableCell,
    TextBlock,
    BlockType,
)
from src.strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)


class FastTextExtractor(ExtractionStrategy):
    """
    Strategy A: Direct text extraction via pdfplumber.

    Triggers when: origin_type=native_digital AND layout_complexity=single_column.
    Confidence gate: pages must have char_count > threshold and
    image_area < 50% of page area.
    """

    @property
    def name(self) -> str:
        return "fast_text"

    @property
    def cost_tier(self) -> str:
        return "low"

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """Extract text, tables, and basic structure using pdfplumber."""
        start_time = time.time()
        pages: list[ExtractedPage] = []

        logger.info(f"[Strategy A] Fast text extraction: {pdf_path.name}")

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_page = self._extract_page(page)
                pages.append(extracted_page)

        processing_time = time.time() - start_time

        # Aggregate confidence
        confidences = [p.confidence_score for p in pages]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            pages=pages,
            strategy_used=self.name,
            confidence_score=round(avg_confidence, 4),
            processing_time_s=round(processing_time, 2),
        )

        logger.info(
            f"[Strategy A] Done: {len(pages)} pages, "
            f"confidence={avg_confidence:.3f}, "
            f"tables={doc.total_tables}, "
            f"time={processing_time:.1f}s"
        )

        return doc

    def _extract_page(self, page) -> ExtractedPage:
        """Extract content from a single pdfplumber page."""
        page_number = page.page_number
        page_area = float(page.width) * float(page.height)

        # ── Raw text ──
        raw_text = page.extract_text() or ""

        # ── Text blocks ──
        text_blocks = self._extract_text_blocks(page)

        # ── Tables ──
        tables = self._extract_tables(page)

        # ── Confidence score ──
        confidence = self._compute_page_confidence(page, page_area)

        return ExtractedPage(
            page_number=page_number,
            text_blocks=text_blocks,
            tables=tables,
            figures=[],  # pdfplumber doesn't extract figure captions
            raw_text=raw_text,
            confidence_score=confidence,
            strategy_used=self.name,
        )

    def _extract_text_blocks(self, page) -> list[TextBlock]:
        """Extract text as blocks using pdfplumber words, grouped by proximity."""
        blocks: list[TextBlock] = []
        words = page.extract_words(keep_blank_chars=True, extra_attrs=["fontname", "size"]) or []

        if not words:
            return blocks

        # Group words into lines by y-coordinate proximity
        lines: list[list[dict]] = []
        current_line: list[dict] = [words[0]]

        for word in words[1:]:
            # Same line if y-coordinates are close
            prev_top = current_line[-1].get("top", 0)
            curr_top = word.get("top", 0)
            if abs(curr_top - prev_top) < 5:  # within 5 points
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        # Convert lines to text blocks
        for idx, line_words in enumerate(lines):
            text = " ".join(w.get("text", "") for w in line_words)
            if not text.strip():
                continue

            # Bounding box from first and last word
            x0 = min(float(w.get("x0", 0)) for w in line_words)
            y0 = min(float(w.get("top", 0)) for w in line_words)
            x1 = max(float(w.get("x1", 0)) for w in line_words)
            y1 = max(float(w.get("bottom", 0)) for w in line_words)

            bbox = BoundingBox(
                x0=x0, y0=y0, x1=x1, y1=y1,
                page_number=page.page_number,
            )

            blocks.append(TextBlock(
                content=text.strip(),
                bbox=bbox,
                block_type=BlockType.PARAGRAPH,
                reading_order=idx,
            ))

        return blocks

    def _extract_tables(self, page) -> list[ExtractedTable]:
        """Extract tables as structured data."""
        extracted_tables: list[ExtractedTable] = []
        tables = page.find_tables() or []

        for t_idx, table in enumerate(tables):
            raw_data = table.extract()
            if not raw_data or len(raw_data) == 0:
                continue

            # First row as headers
            headers = [str(cell) if cell else "" for cell in raw_data[0]]

            # Remaining rows as TableCell objects
            rows: list[list[TableCell]] = []
            for r_idx, row in enumerate(raw_data[1:], start=1):
                cells = []
                for c_idx, cell_val in enumerate(row):
                    cells.append(TableCell(
                        text=str(cell_val) if cell_val else "",
                        row=r_idx,
                        col=c_idx,
                    ))
                rows.append(cells)

            # Table bounding box
            bbox = None
            if table.bbox:
                bbox = BoundingBox(
                    x0=float(table.bbox[0]),
                    y0=float(table.bbox[1]),
                    x1=float(table.bbox[2]),
                    y1=float(table.bbox[3]),
                    page_number=page.page_number,
                )

            extracted_tables.append(ExtractedTable(
                table_id=f"p{page.page_number}_t{t_idx}",
                headers=headers,
                rows=rows,
                bbox=bbox,
                page_number=page.page_number,
            ))

        return extracted_tables

    def _compute_page_confidence(self, page, page_area: float) -> float:
        """Compute Strategy A confidence for a single page."""
        thresholds = get_page_signal_thresholds()
        min_chars = thresholds.get("min_char_count", 100)
        max_image_ratio = thresholds.get("max_image_area_ratio", 0.50)
        min_density = thresholds.get("min_char_density", 0.001)
        min_fonts = thresholds.get("min_font_count", 1)

        chars = page.chars or []
        char_count = len(chars)
        char_density = char_count / page_area if page_area > 0 else 0.0

        # Image area
        images = page.images or []
        image_area = sum(
            abs(float(img.get("x1", 0)) - float(img.get("x0", 0)))
            * abs(float(img.get("y1", 0)) - float(img.get("y0", 0)))
            for img in images
        )
        image_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0.0

        fonts = set(ch.get("fontname", "") for ch in chars if ch.get("fontname"))
        font_count = len(fonts)

        # Component scores
        density_score = min(char_density / 0.01, 1.0) if char_density > 0 else 0.0
        char_score = min(char_count / min_chars, 1.0)
        image_score = max(0.0, 1.0 - max(0, image_ratio - max_image_ratio) / max(0.01, 1.0 - max_image_ratio))
        font_score = 1.0 if font_count >= min_fonts else 0.2

        confidence = (
            0.30 * density_score
            + 0.25 * char_score
            + 0.30 * image_score
            + 0.15 * font_score
        )

        return round(min(max(confidence, 0.0), 1.0), 4)
