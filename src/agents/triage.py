"""
Triage Agent — Stage 1 of the Document Intelligence Refinery.

Classifies incoming documents to determine extraction strategy:
  - Origin type: native_digital | scanned_image | mixed | form_fillable
  - Layout complexity: single_column | multi_column | table_heavy | figure_heavy | mixed
  - Domain hint: financial | legal | technical | medical | general
  - Language detection
  - Estimated extraction cost
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pdfplumber

from src.config import (
    get_config,
    get_document_classification_thresholds,
    get_domain_hints_config,
    get_logging_config,
    get_page_signal_thresholds,
)
from src.models.schemas import (
    DocumentProfile,
    DomainHint,
    LayoutComplexity,
    OriginType,
    PageSignal,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Swappable Domain Classification Strategy
# ─────────────────────────────────────────────────────────

class DomainClassifier(ABC):
    """Abstract base class for domain hint classification.

    Implement this interface to swap in a VLM-based or
    model-based domain classifier without modifying TriageAgent.
    """

    @abstractmethod
    def classify(self, text: str, filename: str = "") -> DomainHint:
        """Classify the domain of a document from sample text."""
        ...


class KeywordDomainClassifier(DomainClassifier):
    """Keyword-based domain classifier using lists from config."""

    def __init__(self, keywords_config: dict[str, Any] | None = None):
        self.domain_keywords = keywords_config or get_domain_hints_config()

    def classify(self, text: str, filename: str = "") -> DomainHint:
        """Score each domain by keyword matches and return the best."""
        if not text.strip():
            return DomainHint.GENERAL

        text_lower = text.lower()
        best_domain = DomainHint.GENERAL
        best_score = 0

        for domain_name, domain_cfg in self.domain_keywords.items():
            keywords = domain_cfg.get("keywords", [])
            if not keywords:
                continue

            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                try:
                    best_domain = DomainHint(domain_name)
                except ValueError:
                    pass

        return best_domain


class TriageAgent:
    """
    Analyses a PDF and produces a DocumentProfile that governs
    which extraction strategy downstream stages will use.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        domain_classifier: DomainClassifier | None = None,
    ):
        self.config = config or get_config()
        self.page_thresholds = get_page_signal_thresholds()
        self.doc_thresholds = get_document_classification_thresholds()
        self.domain_classifier = domain_classifier or KeywordDomainClassifier()
        self.confidence_gates = self.config.get("confidence_gates", {})

    # ─────────────────────────────────────────────────────────
    # Main Pipeline
    # ─────────────────────────────────────────────────────────

    def profile(self, pdf_path: Path) -> DocumentProfile:
        """Main entry point: analyzes a PDF and returns a DocumentProfile."""
        doc_id = f"{pdf_path.stem.replace(' ', '_')}_{hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]}"
        
        logger.info(f"Triage: {pdf_path.name}")
        
        page_signals: list[PageSignal] = []
        sample_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            # Collect signals for all pages
            for page in pdf.pages:
                signal = self._analyze_page(page)
                page_signals.append(signal)
            
            # Collect sample text from first 5 pages for domain/language
            for page in pdf.pages[:5]:
                text = page.extract_text() or ""
                sample_text += text + " "

        profile = self._build_profile(doc_id, pdf_path, page_signals, sample_text)
        self._save_profile(profile)
        return profile

    def _analyze_page(self, page: pdfplumber.page.Page) -> PageSignal:
        """Extract textual and structural signals from a single page."""
        width = float(page.width)
        height = float(page.height)
        page_area = width * height

        # Text signals
        text = page.extract_text() or ""
        char_count = len(text)
        word_count = len(text.split())
        char_density = char_count / page_area if page_area > 0 else 0

        # Structural signals
        images = page.images
        image_area = sum(float(img["width"]) * float(img["height"]) for img in images)
        image_area_ratio = image_area / page_area if page_area > 0 else 0
        image_area_ratio = max(0.0, min(1.0, image_area_ratio))

        tables = page.find_tables()
        table_count = len(tables)

        # Font metadata (multi-signal)
        fonts = set()
        has_text_layer = False
        try:
            # pdfplumber .chars contains font info if digital
            chars = page.chars
            if chars:
                has_text_layer = True
                fonts = {c.get("fontname") for c in chars if c.get("fontname")}
        except Exception:
            pass

        # Whitespace ratio (heuristic)
        # Using a simple trick: char_count vs expected capacity
        # Or better: bbox coverage
        whitespace_ratio = 1.0 - (char_count * 50 / page_area) if page_area > 0 else 1.0
        whitespace_ratio = max(0.0, min(1.0, whitespace_ratio))

        # Initial confidence score for Strategy A (pdfplumber)
        confidence = self._compute_confidence(
            char_density, image_area_ratio, len(fonts), char_count
        )

        return PageSignal(
            page_number=page.page_number,
            char_count=char_count,
            char_density=char_density,
            word_count=word_count,
            image_area_ratio=image_area_ratio,
            font_count=len(fonts),
            table_count=table_count,
            has_text_layer=has_text_layer,
            whitespace_ratio=whitespace_ratio,
            confidence_score=confidence,
            extraction_hint=self._classify_extraction_hint(
                confidence, image_area_ratio, char_count
            ),
        )

    def _compute_confidence(
        self,
        char_density: float,
        image_area_ratio: float,
        font_count: int,
        char_count: int,
    ) -> float:
        """
        Compute a multi-signal confidence score (0.0–1.0)
        for Strategy A (pypdf/pdfplumber).
        
        Formula weights:
        - 30% Char Density (relative to standard 3000 chars/page)
        - 25% Image Area Ratio (inversely proportional)
        - 30% Char Count 
        - 15% Font Metadata
        """
        if char_count == 0:
            return 0.0

        # 1. Density Score (Target: ~0.005 chars/pt²)
        density_score = min(1.0, char_density / 0.005)

        # 2. Image Ratio Score (Target: < 20%)
        image_score = 1.0 - image_area_ratio

        # 3. Char Count Score (Target: > 500)
        count_score = min(1.0, char_count / 500)

        # 4. Font metadata (Target: > 2 fonts)
        font_score = min(1.0, font_count / 2)

        raw_score = (
            (density_score * 0.30) +
            (image_score * 0.25) +
            (count_score * 0.30) +
            (font_score * 0.15)
        )
        return round(float(raw_score), 4)

    def _classify_extraction_hint(
        self, confidence: float, image_area_ratio: float, char_count: int
    ) -> str:
        """Map confidence + signals to an extraction strategy hint."""
        strategy_a_min = self.confidence_gates.get("strategy_a_min", 0.70)
        strategy_b_min = self.confidence_gates.get("strategy_b_min", 0.45)

        if confidence >= strategy_a_min:
            return "fast_text"
        elif confidence >= strategy_b_min:
            return "layout_model"
        else:
            return "vision_model"

    # ─────────────────────────────────────────────────────────
    # Document-level classification
    # ─────────────────────────────────────────────────────────

    def _build_profile(
        self,
        doc_id: str,
        pdf_path: Path,
        page_signals: list[PageSignal],
        sample_text: str = "",
    ) -> DocumentProfile:
        """Aggregate page signals into a DocumentProfile."""
        page_count = len(page_signals)
        file_size = pdf_path.stat().st_size

        # Aggregate confidence
        confidences = [s.confidence_score for s in page_signals]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        strategy_a_min = self.confidence_gates.get("strategy_a_min", 0.70)
        pages_below = sum(1 for c in confidences if c < strategy_a_min)

        origin_type = self._detect_origin_type(pdf_path, page_signals)
        layout_complexity = self._detect_layout_complexity(page_signals)
        domain_hint = self.domain_classifier.classify(sample_text, pdf_path.name)
        language, lang_confidence = self._detect_language(sample_text)
        
        # Logic for strategy selection based on characteristics
        if origin_type == OriginType.SCANNED_IMAGE:
            cost_per_page = 0.01  # Vision tier
        elif (origin_type == OriginType.MIXED or 
              layout_complexity in (LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN, LayoutComplexity.MIXED) or
              avg_confidence < self.confidence_gates.get("strategy_a_min", 0.70)):
            cost_per_page = 0.001 # Layout tier
        else:
            cost_per_page = 0.0001 # Fast Text tier
            
        estimated_cost_usd = round(page_count * cost_per_page, 4)

        return DocumentProfile(
            doc_id=doc_id,
            filename=pdf_path.name,
            file_size_bytes=file_size,
            page_count=page_count,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language=language,
            language_confidence=lang_confidence,
            domain_hint=domain_hint,
            estimated_cost_usd=estimated_cost_usd,
            avg_confidence=round(avg_confidence, 4),
            pages_below_threshold=pages_below,
            per_page_signals=page_signals,
        )

    def _detect_origin_type(self, pdf_path: Path, signals: list[PageSignal]) -> OriginType:
        """
        Classify document origin based on page-level text/image analysis.
        Uses thresholds from extraction_rules.yaml.
        Also detects form-fillable PDFs via /AcroForm dictionary.
        """
        total = len(signals)
        if total == 0:
            return OriginType.NATIVE_DIGITAL

        # Check for form-fillable PDFs (AcroForm)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if hasattr(pdf, 'doc') and pdf.doc.catalog.get('AcroForm'):
                    return OriginType.FORM_FILLABLE
        except Exception:
            pass

        no_text_pct_thresh = self.doc_thresholds.get("scanned_threshold_no_text_pct", 0.80)
        mixed_no_text_thresh = self.doc_thresholds.get("mixed_threshold_no_text_pct", 0.10)

        pages_no_text = sum(1 for s in signals if not s.has_text_layer)
        no_text_pct = pages_no_text / total

        if no_text_pct >= no_text_pct_thresh:
            return OriginType.SCANNED_IMAGE
        elif no_text_pct >= mixed_no_text_thresh:
            return OriginType.MIXED
        else:
            return OriginType.NATIVE_DIGITAL

    def _detect_layout_complexity(self, signals: list[PageSignal]) -> LayoutComplexity:
        """
        Classify layout complexity based on table counts, image ratios,
        and character density patterns.
        """
        total = len(signals)
        if total == 0:
            return LayoutComplexity.SINGLE_COLUMN

        table_heavy_thresh = self.doc_thresholds.get("table_heavy_tables_per_page", 0.5)
        figure_heavy_thresh = self.doc_thresholds.get("figure_heavy_image_ratio", 0.3)

        # Tables per page
        total_tables = sum(s.table_count for s in signals)
        tables_per_page = total_tables / total

        # Image-dominated pages
        img_dominated = sum(1 for s in signals if s.image_area_ratio > figure_heavy_thresh)
        img_dominated_pct = img_dominated / total

        # Density variance → multi-column if high variance
        densities = [s.char_density for s in signals if s.char_density > 0]
        density_std = statistics.stdev(densities) if len(densities) > 1 else 0.0

        if tables_per_page >= table_heavy_thresh:
            return LayoutComplexity.TABLE_HEAVY
        elif img_dominated_pct > 0.30:
            return LayoutComplexity.FIGURE_HEAVY
        elif density_std > 0.003:
            return LayoutComplexity.MULTI_COLUMN
        elif tables_per_page > 0 or density_std > 0.001:
            return LayoutComplexity.MIXED
        else:
            return LayoutComplexity.SINGLE_COLUMN

    def _detect_language(self, sample_text: str) -> tuple[str, float]:
        """
        Detect document language using langdetect on sample text.
        Returns (language_code, confidence).
        """
        if len(sample_text.strip()) < 50:
            return ("en", 0.0)  # Not enough text to detect

        try:
            from langdetect import detect_langs
            results = detect_langs(sample_text)
            if results:
                best = results[0]
                return (str(best.lang), round(float(best.prob), 4))
        except ImportError:
            logger.warning("langdetect not installed, defaulting to English")
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

        return ("en", 0.0)

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def _save_profile(self, profile: DocumentProfile) -> None:
        """Persist DocumentProfile to .refinery/profiles/."""
        profile_dir = Path(".refinery/profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        profile_path = profile_dir / f"{profile.doc_id}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile.model_dump(mode="json"), f, indent=2, sort_keys=True)
        # logger.info(f"Profile saved: {profile_path}")
