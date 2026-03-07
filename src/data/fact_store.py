"""
FactTable Extractor Database Layer.

Handles secure, structured extraction of key-value metrics
into a SQLite database for precise querying (Rubric Stage 4).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FactStore:
    """Manages the SQLite FactTable for financial/numerical documents."""

    def __init__(self, db_path: str | Path = ".refinery/facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # Return dictionaries from rows
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"Initialized SQLite FactStore at {self.db_path}")

    def _init_schema(self) -> None:
        """Create the table schema if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ExtractedFacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_number INTEGER,
                metric_name TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                date_context TEXT,
                source_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast querying by the LangGraph agent
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON ExtractedFacts(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric ON ExtractedFacts(metric_name)")
        
        self.conn.commit()

    def insert_facts(self, doc_id: str, page_number: int, source_hash: str, facts: list[dict[str, str]]) -> int:
        """Insert a batch of LLM-extracted facts into the table."""
        if not facts:
            return 0
            
        cursor = self.conn.cursor()
        count = 0
        
        for fact in facts:
            name = fact.get("metric_name", "").strip()
            val = fact.get("metric_value", "").strip()
            date = fact.get("date_context", "").strip()
            
            # Skip invalid extractions
            if not name or not val:
                continue
                
            cursor.execute(
                """
                INSERT INTO ExtractedFacts 
                (doc_id, page_number, metric_name, metric_value, date_context, source_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, page_number, name, val, date, source_hash)
            )
            count += 1
            
        self.conn.commit()
        return count

    def query_facts(self, sql_query: str) -> list[dict[str, Any]]:
        """
        Execute a safe SELECT query on the FactTable.
        Used by the standard structured_query tool in the LangGraph Agent.
        """
        if not sql_query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
            
        cursor = self.conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
