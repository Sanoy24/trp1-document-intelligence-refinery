"""
Vector Store Interface for the Document Intelligence Refinery.

Wraps a local ChromaDB instance to ingest Logical Document Units (LDUs)
and perform semantic search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.models.schemas import LDU

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages the local ChromaDB vector database."""

    def __init__(self, db_dir: str | Path = ".refinery/chroma_db", collection_name: str = "refinery_ldus"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB in persistent mode
        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Refinery Semantic Chunks (LDUs)"}
        )
        
        logger.info(f"Initialized ChromaDB VectorStore at {self.db_dir}")

    def ingest_ldus(self, ldus: list[LDU]) -> None:
        """
        Embed and store a batch of LDUs in ChromaDB.
        Uses ChromaDB's default SentenceTransformers embedding function
        if no custom function is provided.
        """
        if not ldus:
            return

        ids = []
        documents = []
        metadatas = []

        for ldu in ldus:
            ids.append(ldu.ldu_id)
            documents.append(ldu.content)
            
            # Pack complex types (lists, dicts) into JSON strings for Chroma metadata
            bbox_str = json.dumps(ldu.bounding_box.model_dump()) if ldu.bounding_box else ""
            
            metadata = {
                "chunk_type": ldu.chunk_type.value,
                "parent_section": ldu.parent_section or "Root",
                "token_count": ldu.token_count,
                "content_hash": ldu.content_hash,
                "page_refs": json.dumps(ldu.page_refs),
                "bounding_box": bbox_str,
                "related_chunks": json.dumps(ldu.related_chunks),
            }
            metadatas.append(metadata)

        # Upsert the batch
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Ingested {len(ldus)} chunks into ChromaDB.")

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Perform a semantic search against the LDU collection.
        Returns a formatted list of results with metadata.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        
        if not results["ids"] or not results["ids"][0]:
            return formatted_results

        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
            
        return formatted_results

