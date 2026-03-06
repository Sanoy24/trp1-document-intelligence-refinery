"""
Strategy B — Layout-Aware Extraction (Cost: Medium).

Uses Docling for layout-aware extraction with DoclingDocument adapter
to normalize output into the internal ExtractedDocument schema.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.models.schemas import (
    BoundingBox,
    BlockType,
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
    FigureBlock,
    TableCell,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy

logger = logging.getLogger(__name__)


class LayoutExtractor(ExtractionStrategy):
    """
    Strategy B: Layout-aware extraction via Docling.

    Triggers when: multi_column OR table_heavy OR mixed origin.
    Extracts text blocks with bounding boxes, tables as structured JSON,
    figures with captions, and reading order reconstruction.
    """

    @property
    def name(self) -> str:
        return "layout"

    @property
    def cost_tier(self) -> str:
        return "medium"

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """Run Docling on the PDF and normalize output."""
        start_time = time.time()

        logger.info(f"[Strategy B] Layout extraction (Docling): {pdf_path.name}")

        try:
            from docling.document_converter import DocumentConverter
            import os

            # Determine a safe number of threads to use (leave 1 core free)
            num_threads = max(1, (os.cpu_count() or 4) - 1)

            # Define pipeline options — Optimized for speed and low memory
            pipeline_options = ThreadedPdfPipelineOptions(
                do_ocr=False,
                generate_page_images=False,
                do_table_structure=True,  # Always enable for Strategy B
                ocr_batch_size=1,
                layout_batch_size=4,
                table_batch_size=4,
                queue_max_size=10,
                accelerator_options=AcceleratorOptions(
                    num_threads=num_threads, device="cpu"
                ),
            )

            # Initialize the converter
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            result = converter.convert(str(pdf_path))
            docling_doc = result.document

            pages = self._adapt_docling_document(docling_doc, profile)

        except ImportError:
            logger.error("Docling not installed. Run: pip install docling")
            raise
        except Exception as e:
            logger.error(f"[Strategy B] Docling extraction failed: {e}")
            raise

        processing_time = time.time() - start_time

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
            f"[Strategy B] Done: {len(pages)} pages, "
            f"tables={doc.total_tables}, "
            f"figures={doc.total_figures}, "
            f"time={processing_time:.1f}s"
        )

        return doc

    def _adapt_docling_document(
        self, docling_doc, profile: DocumentProfile
    ) -> list[ExtractedPage]:
        """
        DoclingDocumentAdapter: normalize Docling's output into
        our internal ExtractedPage schema.
        """
        from docling.datamodel.document import DoclingDocument

        # Build page-level containers
        page_map: dict[int, ExtractedPage] = {}

        for page_num in range(1, profile.page_count + 1):
            page_map[page_num] = ExtractedPage(
                page_number=page_num,
                confidence_score=0.75,  # Docling produces decent output
                strategy_used=self.name,
            )

        # Process document items
        reading_order = 0

        # Iterate through the document's main body
        if hasattr(docling_doc, "texts") and docling_doc.texts:
            for text_item in docling_doc.texts:
                page_num = self._get_page_number(text_item)
                if page_num not in page_map:
                    page_map[page_num] = ExtractedPage(
                        page_number=page_num,
                        confidence_score=0.75,
                        strategy_used=self.name,
                    )

                bbox = self._extract_bbox(text_item, page_num)
                block_type = self._map_block_type(text_item)

                text_content = ""
                if hasattr(text_item, "text"):
                    text_content = text_item.text
                elif hasattr(text_item, "export_to_text"):
                    text_content = text_item.export_to_text()

                if text_content.strip():
                    page_map[page_num].text_blocks.append(TextBlock(
                        content=text_content.strip(),
                        bbox=bbox,
                        block_type=block_type,
                        reading_order=reading_order,
                    ))
                    reading_order += 1

        # Process tables
        if hasattr(docling_doc, "tables") and docling_doc.tables:
            for t_idx, table_item in enumerate(docling_doc.tables):
                page_num = self._get_page_number(table_item)
                if page_num not in page_map:
                    continue

                extracted_table = self._extract_table(table_item, page_num, t_idx)
                if extracted_table:
                    page_map[page_num].tables.append(extracted_table)

        # Process figures/pictures
        if hasattr(docling_doc, "pictures") and docling_doc.pictures:
            for f_idx, fig_item in enumerate(docling_doc.pictures):
                page_num = self._get_page_number(fig_item)
                if page_num not in page_map:
                    continue

                figure = self._extract_figure(fig_item, page_num, f_idx)
                if figure:
                    page_map[page_num].figures.append(figure)

        # Build raw text for each page
        for page_num, page in page_map.items():
            page.raw_text = "\n".join(
                block.content for block in page.text_blocks
            )

        return sorted(page_map.values(), key=lambda p: p.page_number)

    def _get_page_number(self, item) -> int:
        """Extract page number from a Docling document item."""
        # Docling items may have prov (provenance) with page info
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "page_no"):
                    return prov.page_no
                if hasattr(prov, "page"):
                    return prov.page
        return 1

    def _extract_bbox(self, item, page_num: int) -> BoundingBox | None:
        """Extract bounding box from a Docling item's provenance."""
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "bbox") and prov.bbox is not None:
                    bbox = prov.bbox
                    if hasattr(bbox, "l"):
                        return BoundingBox(
                            x0=float(bbox.l),
                            y0=float(bbox.t),
                            x1=float(bbox.r),
                            y1=float(bbox.b),
                            page_number=page_num,
                        )
                    elif hasattr(bbox, "x0"):
                        return BoundingBox(
                            x0=float(bbox.x0),
                            y0=float(bbox.y0),
                            x1=float(bbox.x1),
                            y1=float(bbox.y1),
                            page_number=page_num,
                        )
        return None

    def _map_block_type(self, item) -> BlockType:
        """Map Docling item type to our BlockType enum."""
        if hasattr(item, "label"):
            label = str(item.label).lower()
            if "heading" in label or "title" in label:
                return BlockType.HEADING
            elif "list" in label:
                return BlockType.LIST_ITEM
            elif "caption" in label:
                return BlockType.CAPTION
            elif "footer" in label:
                return BlockType.FOOTER
            elif "header" in label:
                return BlockType.HEADER
        return BlockType.PARAGRAPH

    def _extract_table(self, table_item, page_num: int, idx: int) -> ExtractedTable | None:
        """Extract a Docling table into our ExtractedTable schema."""
        try:
            if hasattr(table_item, "data") and hasattr(table_item.data, "grid"):
                grid = table_item.data.grid
                if not grid or len(grid) == 0:
                    return None

                # First row as headers
                headers = [str(cell.text) if hasattr(cell, "text") else "" for cell in grid[0]]

                rows: list[list[TableCell]] = []
                for r_idx, row in enumerate(grid[1:], start=1):
                    cells = []
                    for c_idx, cell in enumerate(row):
                        cells.append(TableCell(
                            text=str(cell.text) if hasattr(cell, "text") else "",
                            row=r_idx,
                            col=c_idx,
                            row_span=getattr(cell, "row_span", 1) or 1,
                            col_span=getattr(cell, "col_span", 1) or 1,
                        ))
                    rows.append(cells)

                bbox = self._extract_bbox(table_item, page_num)

                caption_text = None
                if hasattr(table_item, "caption_text"):
                    try:
                        caption_text = table_item.caption_text()
                    except TypeError:
                        caption_text = getattr(table_item, "text", None)

                return ExtractedTable(
                    table_id=f"p{page_num}_t{idx}",
                    headers=headers,
                    rows=rows,
                    bbox=bbox,
                    caption=caption_text,
                    page_number=page_num,
                )

            # Fallback: try export_to_dataframe or export_to_dict
            if hasattr(table_item, "export_to_dict"):
                table_data = table_item.export_to_dict()
                # Simplified extraction from dict
                return ExtractedTable(
                    table_id=f"p{page_num}_t{idx}",
                    headers=list(table_data.keys()) if isinstance(table_data, dict) else [],
                    rows=[],
                    bbox=self._extract_bbox(table_item, page_num),
                    page_number=page_num,
                )

        except Exception as e:
            logger.warning(f"Failed to extract table on page {page_num}: {e}")

        return None

    def _extract_figure(self, fig_item, page_num: int, idx: int) -> FigureBlock | None:
        """Extract a Docling figure/picture into our FigureBlock schema."""
        try:
            caption = None
            if hasattr(fig_item, "caption_text"):
                try:
                    caption = fig_item.caption_text()
                except TypeError:
                    caption = getattr(fig_item, "text", None)
            elif hasattr(fig_item, "text"):
                caption = fig_item.text

            bbox = self._extract_bbox(fig_item, page_num)

            return FigureBlock(
                figure_id=f"p{page_num}_f{idx}",
                caption=caption,
                bbox=bbox,
                page_number=page_num,
            )
        except Exception as e:
            logger.warning(f"Failed to extract figure on page {page_num}: {e}")
            return None
