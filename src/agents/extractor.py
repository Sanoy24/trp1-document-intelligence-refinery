"""
Extraction Router — Confidence-gated strategy selection with escalation guard.

Reads the DocumentProfile and delegates to the correct extraction strategy.
If Strategy A confidence is low, automatically escalates to Strategy B,
then to Strategy C if B is also low.

Logs every extraction to .refinery/extraction_ledger.jsonl.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.config import get_confidence_thresholds, get_logging_config
from src.models.schemas import (
    DocumentProfile,
    OriginType,
    LayoutComplexity,
    ExtractedDocument,
    LedgerEntry,
)
from src.strategies.base import ExtractionStrategy
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor

logger = logging.getLogger(__name__)


class ExtractionRouter:
    """
    Strategy-pattern router that selects and executes the
    appropriate extraction strategy based on DocumentProfile.

    Implements the Escalation Guard:
        Strategy A (confidence < 0.70) → retry with Strategy B
        Strategy B (confidence < 0.45) → escalate to Strategy C
    """

    def __init__(self):
        self.strategies: dict[str, ExtractionStrategy] = {
            "fast_text": FastTextExtractor(),
            "layout": LayoutExtractor(),
            "vision": VisionExtractor(),
        }
        self.confidence_gates = get_confidence_thresholds()

    def extract(
        self, pdf_path: Path, profile: DocumentProfile
    ) -> ExtractedDocument:
        """
        Run extraction with automatic escalation on low confidence.

        Returns:
            ExtractedDocument from the best-performing strategy.
        """
        pdf_path = Path(pdf_path)

        # Pick initial strategy from profile
        initial_strategy = self._select_initial_strategy(profile)

        logger.info(
            f"Router: {pdf_path.name} → initial strategy={initial_strategy} "
            f"(cost=${profile.estimated_cost_usd:.4f})"
        )

        strategy_a_min = self.confidence_gates.get("strategy_a_min", 0.70)
        strategy_b_min = self.confidence_gates.get("strategy_b_min", 0.45)
        escalated_from: str | None = None

        # ── Attempt extraction with escalation guard ──

        # If starting at Strategy C, just run it
        if initial_strategy == "vision":
            result = self._run_strategy("vision", pdf_path, profile)
            self._check_graceful_degradation(result, strategy_b_min)
            self._log_to_ledger(result, profile, escalated_from=None)
            return result

        # If starting at Strategy B, run it, escalate to C if needed
        if initial_strategy == "layout":
            result = self._run_strategy("layout", pdf_path, profile)

            if result.confidence_score < strategy_b_min:
                logger.warning(
                    f"Strategy B confidence ({result.confidence_score:.3f}) "
                    f"< threshold ({strategy_b_min}). Escalating to Strategy C."
                )
                escalated_from = "layout"
                result = self._run_strategy(
                    "vision", pdf_path, profile, escalated_from=escalated_from
                )
                self._check_graceful_degradation(result, strategy_b_min)

            self._log_to_ledger(result, profile, escalated_from=escalated_from)
            return result

        # Start with Strategy A (fast_text)
        result = self._run_strategy("fast_text", pdf_path, profile)

        if result.confidence_score < strategy_a_min:
            logger.warning(
                f"Strategy A confidence ({result.confidence_score:.3f}) "
                f"< threshold ({strategy_a_min}). Escalating to Strategy B."
            )
            escalated_from = "fast_text"
            result = self._run_strategy(
                "layout", pdf_path, profile, escalated_from=escalated_from
            )

            if result.confidence_score < strategy_b_min:
                logger.warning(
                    f"Strategy B confidence ({result.confidence_score:.3f}) "
                    f"< threshold ({strategy_b_min}). Escalating to Strategy C."
                )
                escalated_from = "layout"
                result = self._run_strategy(
                    "vision", pdf_path, profile, escalated_from=escalated_from
                )
                self._check_graceful_degradation(result, strategy_b_min)

        self._log_to_ledger(result, profile, escalated_from=escalated_from)
        return result

    def _select_initial_strategy(self, profile: DocumentProfile) -> str:
        """
        Pick the initial strategy based on the DocumentProfile decision tree:
        1. Scanned Image -> Strategy C (Vision)
        2. Mixed Origin OR Table Heavy OR Multi-Column -> Strategy B (Layout)
        3. Low Confidence (< 0.70) -> Strategy B (Layout)
        4. Single Column Native Digital -> Strategy A (Fast Text)
        """
        # 1. Scanned Image escalation
        if profile.origin_type == OriginType.SCANNED_IMAGE:
            return "vision"

        # 2. Layout Complexity escalation (Table Heavy / Multi Column / Mixed)
        if (profile.origin_type == OriginType.MIXED or 
            profile.layout_complexity in (LayoutComplexity.TABLE_HEAVY, 
                                        LayoutComplexity.MULTI_COLUMN, 
                                        LayoutComplexity.MIXED)):
            return "layout"

        # 3. Confidence-based triage
        if profile.avg_confidence < 0.70:
            return "layout"

        # 4. Default to Fast Text
        return "fast_text"

    def _run_strategy(
        self,
        strategy_name: str,
        pdf_path: Path,
        profile: DocumentProfile,
        escalated_from: str | None = None,
    ) -> ExtractedDocument:
        """Execute a named strategy and return the result."""
        strategy = self.strategies[strategy_name]

        logger.info(
            f"Running {strategy.name} (cost_tier={strategy.cost_tier})"
            + (f" [escalated from {escalated_from}]" if escalated_from else "")
        )

        start = time.time()
        result = strategy.extract(pdf_path, profile)
        elapsed = time.time() - start

        result.processing_time_s = round(elapsed, 2)

        logger.info(
            f"{strategy.name} complete: confidence={result.confidence_score:.3f}, "
            f"pages={len(result.pages)}, time={elapsed:.1f}s"
        )

        return result

    def _check_graceful_degradation(
        self,
        result: ExtractedDocument,
        min_threshold: float,
    ) -> None:
        """
        When the final strategy still returns low confidence,
        flag the result for manual review rather than silently passing bad data.
        """
        if result.confidence_score < min_threshold:
            result.needs_review = True
            logger.warning(
                f"⚠️ Final strategy '{result.strategy_used}' confidence "
                f"({result.confidence_score:.3f}) is below threshold "
                f"({min_threshold}). Document flagged for review."
            )

    def _log_to_ledger(
        self,
        result: ExtractedDocument,
        profile: DocumentProfile,
        escalated_from: str | None = None,
    ) -> None:
        """Append an entry to .refinery/extraction_ledger.jsonl."""
        log_cfg = get_logging_config()
        ledger_path = Path(log_cfg.get(
            "extraction_ledger_path", ".refinery/extraction_ledger.jsonl"
        ))
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate actual cost based on strategy used (matching Triage tier pricing)
        page_count = len(result.pages)
        if result.strategy_used == "vision":
            cost_estimate = page_count * 0.01
        elif result.strategy_used == "layout":
            cost_estimate = page_count * 0.001
        else:
            cost_estimate = page_count * 0.0001

        entry = LedgerEntry(
            doc_id=result.doc_id,
            filename=result.filename,
            strategy_used=result.strategy_used,
            confidence_score=result.confidence_score,
            cost_estimate_usd=cost_estimate,
            processing_time_s=result.processing_time_s,
            pages_processed=page_count,
            escalated_from=escalated_from,
            needs_review=result.needs_review,
        )

        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
            f.flush()

        logger.info(f"Ledger entry written: {ledger_path}")
