"""
Unit tests for the Triage Agent — classification of corpus documents.

Tests that each of the 4 document classes receives the correct
origin_type, layout_complexity, and domain_hint.
"""

import json
from pathlib import Path

import pytest

from src.agents.triage import TriageAgent
from src.models.schemas import (
    DomainHint,
    LayoutComplexity,
    OriginType,
)

# ─────────────────────────────────────────────────────────────
# Corpus paths — skip tests if documents not present
# ─────────────────────────────────────────────────────────────

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

CLASS_A = CORPUS_DIR / "CBE ANNUAL REPORT 2023-24.pdf"
CLASS_B = CORPUS_DIR / "Audit Report - 2023.pdf"
CLASS_C = CORPUS_DIR / "fta_performance_survey_final_report_2022.pdf"
CLASS_D = CORPUS_DIR / "tax_expenditure_ethiopia_2021_22.pdf"

requires_corpus = pytest.mark.skipif(
    not CORPUS_DIR.exists(), reason="Corpus directory not found"
)


@pytest.fixture(scope="module")
def triage_agent():
    """Create a single TriageAgent for all tests."""
    return TriageAgent()


# ─────────────────────────────────────────────────────────────
# Class A: CBE Annual Report (native digital, table-heavy)
# ─────────────────────────────────────────────────────────────

@requires_corpus
@pytest.mark.skipif(not CLASS_A.exists(), reason="Class A document not found")
class TestClassA:

    def test_origin_type(self, triage_agent):
        profile = triage_agent.profile(CLASS_A)
        assert profile.origin_type in (
            OriginType.NATIVE_DIGITAL,
            OriginType.FORM_FILLABLE,
        )

    def test_layout_complexity(self, triage_agent):
        profile = triage_agent.profile(CLASS_A)
        assert profile.layout_complexity in (
            LayoutComplexity.TABLE_HEAVY,
            LayoutComplexity.MULTI_COLUMN,
            LayoutComplexity.MIXED,
        )

    def test_domain_hint(self, triage_agent):
        profile = triage_agent.profile(CLASS_A)
        assert profile.domain_hint == DomainHint.FINANCIAL

    def test_extraction_cost_reasonable(self, triage_agent):
        """Estimated cost should be non-negative and reasonable."""
        profile = triage_agent.profile(CLASS_A)
        assert profile.estimated_cost_usd >= 0.0
        assert profile.estimated_cost_usd < 10.0  # Sanity cap

    def test_profile_saved(self, triage_agent):
        profile = triage_agent.profile(CLASS_A)
        profile_path = Path(".refinery/profiles") / f"{profile.doc_id}.json"
        assert profile_path.exists()


# ─────────────────────────────────────────────────────────────
# Class B: Audit Report (scanned image)
# ─────────────────────────────────────────────────────────────

@requires_corpus
@pytest.mark.skipif(not CLASS_B.exists(), reason="Class B document not found")
class TestClassB:

    def test_origin_type(self, triage_agent):
        profile = triage_agent.profile(CLASS_B)
        assert profile.origin_type == OriginType.SCANNED_IMAGE

    def test_extraction_cost_vision_tier(self, triage_agent):
        """Scanned documents should have higher cost (vision tier)."""
        profile = triage_agent.profile(CLASS_B)
        assert profile.estimated_cost_usd > 0.0

    def test_low_confidence(self, triage_agent):
        profile = triage_agent.profile(CLASS_B)
        assert profile.avg_confidence < 0.50


# ─────────────────────────────────────────────────────────────
# Class C: FTA Report (mixed layout, digital text)
# ─────────────────────────────────────────────────────────────

@requires_corpus
@pytest.mark.skipif(not CLASS_C.exists(), reason="Class C document not found")
class TestClassC:

    def test_origin_type(self, triage_agent):
        profile = triage_agent.profile(CLASS_C)
        assert profile.origin_type == OriginType.NATIVE_DIGITAL

    def test_has_text_on_all_pages(self, triage_agent):
        profile = triage_agent.profile(CLASS_C)
        pages_no_text = sum(
            1 for s in profile.per_page_signals if not s.has_text_layer
        )
        assert pages_no_text == 0


# ─────────────────────────────────────────────────────────────
# Class D: Tax Expenditure (table-heavy, cleanest)
# ─────────────────────────────────────────────────────────────

@requires_corpus
@pytest.mark.skipif(not CLASS_D.exists(), reason="Class D document not found")
class TestClassD:

    def test_origin_type(self, triage_agent):
        profile = triage_agent.profile(CLASS_D)
        assert profile.origin_type == OriginType.NATIVE_DIGITAL

    def test_layout_complexity(self, triage_agent):
        profile = triage_agent.profile(CLASS_D)
        # Tax report is table-heavy or mixed
        assert profile.layout_complexity in (
            LayoutComplexity.TABLE_HEAVY,
            LayoutComplexity.MIXED,
            LayoutComplexity.SINGLE_COLUMN,
        )
