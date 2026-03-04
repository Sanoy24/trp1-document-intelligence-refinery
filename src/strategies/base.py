"""
Base class for extraction strategies.

All strategies must implement the ExtractionStrategy interface
to produce a normalized ExtractedDocument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.schemas import DocumentProfile, ExtractedDocument


class ExtractionStrategy(ABC):
    """Abstract base class for document extraction strategies."""

    @abstractmethod
    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Extract structured content from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            profile: DocumentProfile from the Triage Agent.

        Returns:
            ExtractedDocument with normalized page-level content.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier: 'fast_text', 'layout', or 'vision'."""
        ...

    @property
    @abstractmethod
    def cost_tier(self) -> str:
        """Cost tier: 'low', 'medium', or 'high'."""
        ...
