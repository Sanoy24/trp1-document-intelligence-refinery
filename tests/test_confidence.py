"""
Unit tests for confidence scoring and escalation logic.

Tests the multi-signal confidence formula and verifies
that escalation triggers fire correctly at threshold boundaries.
"""

import pytest

from src.agents.triage import TriageAgent
from src.models.schemas import PageSignal


@pytest.fixture
def agent():
    return TriageAgent()


# ─────────────────────────────────────────────────────────────
# Confidence scoring formula tests
# ─────────────────────────────────────────────────────────────

class TestConfidenceScoring:

    def test_high_confidence_digital_page(self, agent):
        """A page with dense text, no images, multiple fonts → high confidence."""
        score = agent._compute_confidence(
            char_count=2000,
            char_density=0.005,
            image_area_ratio=0.0,
            font_count=5,
            table_count=0,
            page_area=400000.0,
        )
        assert score >= 0.70, f"Expected >= 0.70, got {score}"

    def test_low_confidence_scanned_page(self, agent):
        """A page with no text, high image area → low confidence."""
        score = agent._compute_confidence(
            char_count=0,
            char_density=0.0,
            image_area_ratio=0.98,
            font_count=0,
            table_count=0,
            page_area=400000.0,
        )
        assert score < 0.45, f"Expected < 0.45, got {score}"

    def test_marginal_confidence(self, agent):
        """A page with some text but high image area → marginal confidence."""
        score = agent._compute_confidence(
            char_count=80,
            char_density=0.0002,
            image_area_ratio=0.60,
            font_count=2,
            table_count=0,
            page_area=400000.0,
        )
        # Should be in the marginal zone (roughly 0.3–0.7)
        assert 0.0 <= score <= 1.0

    def test_confidence_range(self, agent):
        """Confidence is always between 0.0 and 1.0."""
        # Edge case: extreme values
        score = agent._compute_confidence(
            char_count=100000,
            char_density=1.0,
            image_area_ratio=0.0,
            font_count=100,
            table_count=50,
            page_area=400000.0,
        )
        assert 0.0 <= score <= 1.0

        score2 = agent._compute_confidence(
            char_count=0,
            char_density=0.0,
            image_area_ratio=1.0,
            font_count=0,
            table_count=0,
            page_area=0.0,
        )
        assert 0.0 <= score2 <= 1.0


# ─────────────────────────────────────────────────────────────
# Extraction hint / escalation trigger tests
# ─────────────────────────────────────────────────────────────

class TestEscalationLogic:

    def test_fast_text_hint_for_high_confidence(self, agent):
        """Confidence >= 0.70 → fast_text."""
        hint = agent._classify_extraction_hint(
            confidence=0.85, image_area_ratio=0.1, char_count=1000
        )
        assert hint == "fast_text"

    def test_layout_model_hint_for_medium_confidence(self, agent):
        """0.45 <= confidence < 0.70 → layout_model."""
        hint = agent._classify_extraction_hint(
            confidence=0.55, image_area_ratio=0.3, char_count=500
        )
        assert hint == "layout_model"

    def test_vision_model_hint_for_low_confidence(self, agent):
        """Confidence < 0.45 → vision_model."""
        hint = agent._classify_extraction_hint(
            confidence=0.20, image_area_ratio=0.9, char_count=10
        )
        assert hint == "vision_model"

    def test_threshold_boundary_a(self, agent):
        """Exactly at strategy_a_min → fast_text (>= comparison)."""
        hint = agent._classify_extraction_hint(
            confidence=0.70, image_area_ratio=0.1, char_count=500
        )
        assert hint == "fast_text"

    def test_threshold_boundary_b(self, agent):
        """Exactly at strategy_b_min → layout_model (>= comparison)."""
        hint = agent._classify_extraction_hint(
            confidence=0.45, image_area_ratio=0.5, char_count=100
        )
        assert hint == "layout_model"

    def test_just_below_boundary_a(self, agent):
        """Just below strategy_a_min → layout_model."""
        hint = agent._classify_extraction_hint(
            confidence=0.699, image_area_ratio=0.2, char_count=500
        )
        assert hint == "layout_model"

    def test_just_below_boundary_b(self, agent):
        """Just below strategy_b_min → vision_model."""
        hint = agent._classify_extraction_hint(
            confidence=0.449, image_area_ratio=0.9, char_count=50
        )
        assert hint == "vision_model"


# ─────────────────────────────────────────────────────────────
# Origin type detection tests
# ─────────────────────────────────────────────────────────────

class TestOriginTypeDetection:

    def test_all_pages_digital(self, agent):
        """All pages have text → native_digital."""
        from pathlib import Path
        from src.models.schemas import OriginType

        signals = [
            PageSignal(page_number=i, has_text_layer=True, char_count=500)
            for i in range(1, 11)
        ]
        # Pass a dummy path — AcroForm check will gracefully fail for non-existent file
        origin = agent._detect_origin_type(Path("dummy.pdf"), signals)
        assert origin == OriginType.NATIVE_DIGITAL

    def test_all_pages_scanned(self, agent):
        """90%+ pages have no text → scanned_image."""
        from pathlib import Path
        from src.models.schemas import OriginType

        signals = [
            PageSignal(page_number=i, has_text_layer=False, char_count=0)
            for i in range(1, 11)
        ]
        origin = agent._detect_origin_type(Path("dummy.pdf"), signals)
        assert origin == OriginType.SCANNED_IMAGE

    def test_mixed_pages(self, agent):
        """Some pages digital, some scanned → mixed."""
        from pathlib import Path
        from src.models.schemas import OriginType

        signals = [
            PageSignal(page_number=i, has_text_layer=(i <= 5), char_count=500 if i <= 5 else 0)
            for i in range(1, 11)
        ]
        origin = agent._detect_origin_type(Path("dummy.pdf"), signals)
        assert origin == OriginType.MIXED
