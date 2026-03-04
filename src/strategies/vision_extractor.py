"""
Strategy C — Vision-Augmented Extraction (Cost: High).

Uses a Vision Language Model (VLM) via OpenRouter or Google Gemini API
to extract structure from scanned or complex documents.
Includes a budget guard to prevent cost overruns.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from src.config import get_budget_config
from src.models.schemas import (
    BoundingBox,
    BlockType,
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
    TableCell,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)


# Cost estimates per 1K tokens (approximate)
_COST_PER_1K_INPUT = {
    "google/gemini-flash-1.5": 0.000075,
    "google/gemini-2.0-flash-001": 0.0001,
    "openai/gpt-4o-mini": 0.00015,
    "default": 0.0001,
}

_COST_PER_1K_OUTPUT = {
    "google/gemini-flash-1.5": 0.0003,
    "google/gemini-2.0-flash-001": 0.0004,
    "openai/gpt-4o-mini": 0.0006,
    "default": 0.0004,
}


class VisionExtractor(ExtractionStrategy):
    """
    Strategy C: Vision-augmented extraction via VLM.

    Triggers when: scanned_image OR Strategy A/B confidence < threshold
    OR handwriting detected.
    Converts PDF pages to images and sends to a VLM with structured
    extraction prompts.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        budget_cfg = get_budget_config()
        self.max_cost_per_doc = budget_cfg.get("max_usd_per_document", 0.50)
        self.max_cost_per_page = budget_cfg.get("max_usd_per_page", 0.05)
        self.model = model or budget_cfg.get("default_model", "google/gemini-flash-1.5")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        self._total_cost = 0.0

    @property
    def name(self) -> str:
        return "vision"

    @property
    def cost_tier(self) -> str:
        return "high"

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """Extract content by sending page images to a VLM."""
        start_time = time.time()
        pages: list[ExtractedPage] = []
        self._total_cost = 0.0

        logger.info(f"[Strategy C] Vision extraction ({self.model}): {pdf_path.name}")

        if not self.api_key:
            logger.error(
                "No API key found. Set OPENROUTER_API_KEY or GEMINI_API_KEY env var."
            )
            raise ValueError("VLM API key not configured")

        # Convert PDF pages to images
        page_images, page_dims = self._pdf_to_images(pdf_path)

        for page_num, image_bytes in enumerate(page_images, start=1):
            # Budget guard
            if self._total_cost >= self.max_cost_per_doc:
                logger.warning(
                    f"[Strategy C] Budget cap reached (${self._total_cost:.4f}). "
                    f"Stopping at page {page_num}/{len(page_images)}"
                )
                break

            # Get page dimensions (width, height in PDF points)
            width, height = page_dims[page_num - 1] if page_num - 1 < len(page_dims) else (612.0, 792.0)

            try:
                extracted_page = self._extract_page_with_vlm(
                    image_bytes, page_num, profile, page_width=width, page_height=height
                )
                pages.append(extracted_page)
            except Exception as e:
                logger.error(f"[Strategy C] Page {page_num} failed: {e}")
                pages.append(ExtractedPage(
                    page_number=page_num,
                    confidence_score=0.0,
                    strategy_used=self.name,
                ))

        processing_time = time.time() - start_time

        confidences = [p.confidence_score for p in pages if p.confidence_score > 0]
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
            f"[Strategy C] Done: {len(pages)} pages, "
            f"cost=${self._total_cost:.4f}, "
            f"time={processing_time:.1f}s"
        )

        return doc

    @property
    def total_cost(self) -> float:
        """Total cost spent so far in current extraction."""
        return self._total_cost

    def _pdf_to_images(self, pdf_path: Path) -> tuple[list[bytes], list[tuple[float, float]]]:
        """Convert PDF pages to PNG images. Returns (images, page_dimensions)."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            images = []
            dims = []
            for page in doc:
                # Store page dimensions in PDF points for bbox
                dims.append((float(page.rect.width), float(page.rect.height)))
                # Render at 150 DPI for good quality without excessive size
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
            doc.close()
            return images, dims
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            raise

    def _extract_page_with_vlm(
        self, image_bytes: bytes, page_num: int, profile: DocumentProfile,
        page_width: float = 612.0, page_height: float = 792.0,
    ) -> ExtractedPage:
        """Send a page image to the VLM and parse the response."""
        import httpx

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = self._build_extraction_prompt(profile)

        # Call OpenRouter API
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
            timeout=60.0,
        )

        response.raise_for_status()
        result = response.json()

        # Track cost
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost_key = self.model if self.model in _COST_PER_1K_INPUT else "default"
        page_cost = (
            input_tokens / 1000 * _COST_PER_1K_INPUT[cost_key]
            + output_tokens / 1000 * _COST_PER_1K_OUTPUT[cost_key]
        )
        self._total_cost += page_cost

        # Parse VLM response
        vlm_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_vlm_response(vlm_content, page_num, page_width, page_height)

    def _build_extraction_prompt(self, profile: DocumentProfile) -> str:
        """Build a structured extraction prompt for the VLM."""
        domain = profile.domain_hint.value

        return f"""You are a document extraction system. Analyze this document page and extract all content in structured JSON format.

Domain context: This is a {domain} document.

Return a JSON object with exactly this structure:
{{
  "text_blocks": [
    {{"content": "...", "type": "paragraph|heading|list_item|caption", "reading_order": 0}}
  ],
  "tables": [
    {{
      "headers": ["col1", "col2"],
      "rows": [["val1", "val2"]],
      "caption": "Table caption if visible"
    }}
  ],
  "raw_text": "Full page text in reading order"
}}

Rules:
- Preserve exact numbers, currencies, and dates
- Maintain table structure precisely (do not merge cells)
- Identify headings vs body text
- Extract ALL text, do not summarize
- Return ONLY valid JSON, no markdown fencing"""

    def _parse_vlm_response(
        self, content: str, page_num: int,
        page_width: float = 612.0, page_height: float = 792.0,
    ) -> ExtractedPage:
        """Parse the VLM's JSON response into an ExtractedPage."""
        text_blocks: list[TextBlock] = []
        tables: list[ExtractedTable] = []
        raw_text = ""

        # Full-page bounding box fallback — VLMs cannot provide precise bboxes
        full_page_bbox = BoundingBox(
            x0=0.0, y0=0.0, x1=page_width, y1=page_height, page_number=page_num
        )

        try:
            # Try to extract JSON from the response
            json_str = content.strip()
            if json_str.startswith("```"):
                # Strip markdown code fences
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])

            data = json.loads(json_str)

            # Text blocks
            for idx, block in enumerate(data.get("text_blocks", [])):
                block_type_str = block.get("type", "paragraph")
                try:
                    block_type = BlockType(block_type_str)
                except ValueError:
                    block_type = BlockType.PARAGRAPH

                text_blocks.append(TextBlock(
                    content=block.get("content", ""),
                    block_type=block_type,
                    reading_order=block.get("reading_order", idx),
                    bounding_box=full_page_bbox,  # full-page fallback
                ))

            # Tables
            for t_idx, table_data in enumerate(data.get("tables", [])):
                headers = table_data.get("headers", [])
                rows: list[list[TableCell]] = []
                for r_idx, row in enumerate(table_data.get("rows", [])):
                    cells = [
                        TableCell(text=str(cell), row=r_idx + 1, col=c_idx)
                        for c_idx, cell in enumerate(row)
                    ]
                    rows.append(cells)

                tables.append(ExtractedTable(
                    table_id=f"p{page_num}_t{t_idx}",
                    headers=headers,
                    rows=rows,
                    caption=table_data.get("caption"),
                    page_number=page_num,
                    bounding_box=full_page_bbox,  # full-page fallback
                ))

            raw_text = data.get("raw_text", "")

        except json.JSONDecodeError:
            logger.warning(f"[Strategy C] Failed to parse VLM JSON for page {page_num}")
            # Fallback: treat entire response as raw text
            raw_text = content
            text_blocks.append(TextBlock(
                content=content,
                block_type=BlockType.PARAGRAPH,
                reading_order=0,
                bounding_box=full_page_bbox,  # full-page fallback
            ))

        return ExtractedPage(
            page_number=page_num,
            text_blocks=text_blocks,
            tables=tables,
            raw_text=raw_text,
            confidence_score=0.70,  # VLM output assumed decent
            strategy_used=self.name,
        )
