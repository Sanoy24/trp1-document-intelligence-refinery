"""
PageIndex Builder (Stage 4).

Generates a hierarchical navigation tree (PageIndexNode) from LDUs.
Uses local LLM (Ollama) to generate section summaries.
Routes table-heavy document text through the FactTable Extractor.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from src.data.fact_store import FactStore
from src.models.schemas import (
    ChunkType,
    DocumentProfile,
    DomainHint,
    LDU,
    OriginType,
    PageIndexNode,
)
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class PageIndexBuilder:
    """Builds the PageIndex tree and populates the FactTable."""

    def __init__(self, llm_client: LLMClient | None = None, fact_store: FactStore | None = None):
        self.llm = llm_client or LLMClient()
        self.fact_store = fact_store or FactStore()
        self.out_dir = Path(".refinery/pageindex")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, doc_profile: DocumentProfile, ldus: list[LDU]) -> PageIndexNode:
        """
        Group LDUs by section, summarize them, extract facts (if financial).
        Returns the root PageIndexNode.
        """
        logger.info(f"Building PageIndex for {doc_profile.doc_id}")
        
        # Group LDUs by section
        section_groups: dict[str, list[LDU]] = defaultdict(list)
        for ldu in ldus:
            section_name = ldu.parent_section or "Root"
            section_groups[section_name].append(ldu)

        # Build root node
        pages = [p for ldu in ldus for p in ldu.page_refs]
        min_page = min(pages) if pages else 1
        max_page = max(pages) if pages else 1
        
        root_node = PageIndexNode(
            title=doc_profile.filename,
            page_start=min_page,
            page_end=max_page,
            summary=f"Automated PageIndex for {doc_profile.filename}",
            data_types_present=["mixed"] if doc_profile.origin_type == OriginType.MIXED else ["text"]
        )

        # Identify if we should run heavy Fact extraction (financial/table-heavy)
        run_fact_extraction = doc_profile.domain_hint == DomainHint.FINANCIAL or doc_profile.layout_complexity.value == "table_heavy"

        # Process each section
        # For simplicity in this demo, it's a flat hierarchy under Root.
        for section_name, section_ldus in section_groups.items():
            if section_name == "Root" and len(section_groups) > 1:
                # Skip naming a generic 'Root' section if we have specific sections
                continue

            sec_pages = [p for l in section_ldus for p in l.page_refs]
            sec_min_p = min(sec_pages) if sec_pages else min_page
            sec_max_p = max(sec_pages) if sec_pages else max_page
            
            # Gather data types (tables, figures, equations)
            data_types = set()
            section_text_parts = []
            
            for ldu in section_ldus:
                if ldu.chunk_type == ChunkType.TABLE:
                    data_types.add("tables")
                elif ldu.chunk_type == ChunkType.FIGURE:
                    data_types.add("figures")
                elif ldu.chunk_type == ChunkType.HEADING:
                    continue # exclude heading from summary payload
                    
                section_text_parts.append(ldu.content)

            section_text = "\n\n".join(section_text_parts)
            
            # 1. Summarize section via LLM
            # Cap text at 3000 chars to avoid overloading the local context window for a simple summary
            summary = self.llm.generate_summary(section_text[:3000])
            
            # 2. Extract Key-Value Facts (if rules match)
            if run_fact_extraction and "tables" in data_types:
                logger.info(f"Extracting structured facts for section: {section_name}")
                facts = self.llm.extract_facts(section_text[:4000])
                if facts:
                    # Map back to LDU source hash for strict provenance
                    # We just use the first table's hash as a proxy for the section fact source
                    table_ldus = [l for l in section_ldus if l.chunk_type == ChunkType.TABLE]
                    src_hash = table_ldus[0].content_hash if table_ldus else section_ldus[0].content_hash
                    
                    inserted = self.fact_store.insert_facts(
                        doc_id=doc_profile.doc_id,
                        page_number=sec_min_p,
                        source_hash=src_hash,
                        facts=facts
                    )
                    logger.debug(f"Inserted {inserted} facts into FactTable.")

            # Append as child node
            child_node = PageIndexNode(
                title=section_name,
                page_start=sec_min_p,
                page_end=sec_max_p,
                summary=summary,
                data_types_present=list(data_types),
                key_entities=[]  # Could be added with an NER step
            )
            root_node.child_sections.append(child_node)

        # Sort children by page appearance
        root_node.child_sections.sort(key=lambda n: n.page_start)

        self._save_index(doc_profile.doc_id, root_node)
        return root_node

    def _save_index(self, doc_id: str, node: PageIndexNode) -> None:
        """Persist the PageIndex tree to JSON."""
        out_path = self.out_dir / f"{doc_id}_index.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(node.model_dump(mode="json"), f, indent=2)
            
        logger.info(f"Saved PageIndex to {out_path}")
