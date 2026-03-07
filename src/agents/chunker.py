"""
Semantic Chunking Engine (Stage 3).

Converts the normalized ExtractedDocument (from Stage 2) into
Logical Document Units (LDUs) suitable for Vector Store ingestion.

Enforces the 5 Core Chunking Rules via ChunkValidator.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.models.schemas import (
    BlockType,
    ChunkType,
    ExtractedDocument,
    ExtractedPage,
    LDU,
)

logger = logging.getLogger(__name__)


class ChunkValidator:
    """Enforces the 5 Core Chunking Rules."""

    @staticmethod
    def validate_rules(ldus: list[LDU]) -> None:
        """Runs basic validation checks on a batch of LDUs."""
        for ldu in ldus:
            # Rule 1 & 2 implicitly handled during assembly, but we check constraints here.
            
            # Basic validation
            if not ldu.content.strip():
                logger.warning(f"LDU {ldu.ldu_id} is empty.")
            
            if not ldu.page_refs:
                logger.warning(f"LDU {ldu.ldu_id} has no page references.")
            
            if not ldu.content_hash:
                logger.warning(f"LDU {ldu.ldu_id} missing content_hash.")


class ChunkingEngine:
    """
    Stateful chunking engine that sweeps through an ExtractedDocument
    and emits LDUs, tracking section headers and lists.
    """

    def __init__(self, max_tokens_per_chunk: int = 500):
        self.max_tokens = max_tokens_per_chunk
        self.validator = ChunkValidator()

    def process_document(self, document: ExtractedDocument) -> list[LDU]:
        """Convert a full document into semantic LDUs."""
        logger.info(f"Chunking document: {document.filename}")
        ldus: list[LDU] = []
        
        current_section = "Root"
        current_list_items = []
        
        # Helper to flush pending list items into a single LDU (Rule 3)
        def flush_list():
            if not current_list_items:
                return
            
            # Combine all list items
            combined_content = "\n".join(item[0].content for item in current_list_items)
            pages = sorted(list(set(p for _, p in current_list_items)))
            
            # Simple bbox combining (just take the first one for now as approximation)
            bbox = current_list_items[0][0].bbox if current_list_items else None
            
            ldu = LDU(
                ldu_id=f"{document.doc_id}_list_{len(ldus)}",
                content=combined_content,
                chunk_type=ChunkType.LIST,
                page_refs=pages,
                bounding_box=bbox,
                parent_section=current_section,
                token_count=self._estimate_tokens(combined_content)
            )
            ldu.compute_hash()
            ldus.append(ldu)
            current_list_items.clear()

        # Iterate through pages sequentially
        for page in sorted(document.pages, key=lambda p: p.page_number):
            
            # Strategy A (Fast Text) currently returns blocks out of order if we aren't careful,
            # but Docling (Strategy B) usually orders them. We'll sort text blocks by reading_order if present,
            # otherwise just take them as they come.
            sorted_blocks = sorted(page.text_blocks, key=lambda b: (b.reading_order, b.bbox.y0 if b.bbox else 0))

            for block in sorted_blocks:

                # Heuristic heading detection for sources that don't label headings explicitly.
                # If a paragraph "looks like" a section title, treat it as a heading so that
                # PageIndex can build a useful navigation tree.
                if block.block_type == BlockType.PARAGRAPH and self._looks_like_heading(
                    block.content
                ):
                    block.block_type = BlockType.HEADING

                # Rule 4: Tracking Section Headers
                if block.block_type == BlockType.HEADING:
                    flush_list()
                    current_section = block.content.strip()
                    
                    # We also emit the heading itself as an LDU (optional, but good for search)
                    ldu = LDU(
                        ldu_id=f"{document.doc_id}_heading_{len(ldus)}",
                        content=block.content,
                        chunk_type=ChunkType.HEADING,
                        page_refs=[page.page_number],
                        bounding_box=block.bbox,
                        parent_section=current_section,
                        token_count=self._estimate_tokens(block.content)
                    )
                    ldu.compute_hash()
                    ldus.append(ldu)
                    continue

                # Rule 3: Collect list items
                if block.block_type == BlockType.LIST_ITEM:
                    current_list_items.append((block, page.page_number))
                    # Check token limit safeguard
                    if self._estimate_tokens("\n".join(i.content for i, _ in current_list_items)) > self.max_tokens:
                        flush_list()
                    continue
                else:
                    # If we hit a normal paragraph, flush active list
                    flush_list()

                # Normal Paragraphs
                if block.block_type == BlockType.PARAGRAPH:
                    # Detect cross references (Rule 5 simple implementation)
                    related = self._detect_cross_references(block.content)
                    
                    ldu = LDU(
                        ldu_id=f"{document.doc_id}_text_{len(ldus)}",
                        content=block.content,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_refs=[page.page_number],
                        bounding_box=block.bbox,
                        parent_section=current_section,
                        token_count=self._estimate_tokens(block.content),
                        related_chunks=related
                    )
                    ldu.compute_hash()
                    ldus.append(ldu)

            # Process Tables for this page
            # Rule 1: Table cell never split from header
            for table in page.tables:
                table_content = self._serialize_table(table)
                related = self._detect_cross_references(table_content)
                
                ldu = LDU(
                    ldu_id=f"{document.doc_id}_table_{len(ldus)}",
                    content=table_content,
                    chunk_type=ChunkType.TABLE,
                    page_refs=[page.page_number],
                    bounding_box=table.bbox,
                    parent_section=current_section,
                    token_count=self._estimate_tokens(table_content),
                    related_chunks=related
                )
                ldu.compute_hash()
                ldus.append(ldu)

            # Process Figures for this page
            # Rule 2: Figure caption is metadata of parent figure chunk
            for figure in page.figures:
                caption_text = figure.caption or "Figure (No Caption)"
                
                ldu = LDU(
                    ldu_id=f"{document.doc_id}_figure_{len(ldus)}",
                    content=caption_text,
                    chunk_type=ChunkType.FIGURE,
                    page_refs=[page.page_number],
                    bounding_box=figure.bbox,
                    parent_section=current_section,
                    token_count=self._estimate_tokens(caption_text)
                )
                ldu.compute_hash()
                ldus.append(ldu)

        # Flush any remaining list items at end of document
        flush_list()

        # Validate before returning
        self.validator.validate_rules(ldus)
        
        logger.info(f"Generated {len(ldus)} LDUs for {document.filename}")
        return ldus

    def _looks_like_heading(self, text: str) -> bool:
        """
        Lightweight heuristic to decide if a line of text is likely a section heading.
        This improves PageIndex quality for extractors that do not label headings.
        Rejects numeric table rows and single-letter fragments to avoid spurious sections.

        Heuristics:
          - Short lines (<= 80 chars)
          - Must contain at least some letters (reject pure numeric rows)
          - Not a single character (reject column headers like "T", "E")
          - Not a row of numeric tokens (e.g. "68.7  76.4  120.7")
          - Either:
              * Start with a numbering pattern (e.g., "1.", "2.1", "I.", "A.")
              * OR are mostly uppercase words (titles)
        """
        stripped = text.strip()
        if not stripped:
            return False

        # Ignore very long lines — unlikely to be headings
        if len(stripped) > 80:
            return False

        # Reject lines with no letters (numeric table rows like "0.09  0.01  0.00  0.11")
        letters = [ch for ch in stripped if ch.isalpha()]
        if not letters:
            return False

        # Reject single-character "headings" (often table column headers like "T", "E")
        if len(stripped) <= 1:
            return False

        # Reject lines that look like table data rows: multiple tokens that are all numeric
        tokens = stripped.split()
        if len(tokens) >= 2:
            numeric_tokens = sum(1 for t in tokens if re.match(r"^[\d.,\-]+$", t))
            if numeric_tokens == len(tokens):
                return False

        # Numbered heading patterns (e.g. "3.1  Defining...", "A.  Negative...")
        if re.match(r"^(\d+(\.\d+)*|[IVXLCM]+\.|[A-Z]\.)\s", stripped):
            return True

        # All-caps / title-like headings (allow digits and punctuation)
        if letters:
            upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
            if upper_ratio > 0.8:
                return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (fast)."""
        return int(len(text.split()) * 1.3)  # Rough word-to-token ratio

    def _serialize_table(self, table: ExtractedTable) -> str:
        """
        Serialize a table into Markdown or structural text.
        Enforces Rule 1 (Headers kept with cells).
        """
        lines = []
        if table.caption:
            lines.append(f"Caption: {table.caption}")
            
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("|".join(["---"] * len(table.headers)))
            
        for row in table.rows:
            # Sort cells by column index
            sorted_cells = sorted(row, key=lambda c: c.col)
            lines.append(" | ".join(c.text.replace("\n", " ") for c in sorted_cells))
            
        return "\n".join(lines)

    def _detect_cross_references(self, text: str) -> list[str]:
        """
        Rule 5: Detect explicit cross-references.
        E.g., "see Table 3", "refer to Figure 2".
        Returns generic string labels that can be matched later.
        """
        related = []
        # Simple regex for Table/Figure X
        matches = re.findall(r"(?:see|refer to)\s+(Table|Figure)\s+(\d+)", text, re.IGNORECASE)
        for entity_type, num in matches:
            related.append(f"{entity_type.lower()}_{num}")
        return related
