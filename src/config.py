"""
Configuration loader for the Document Intelligence Refinery.

Loads extraction_rules.yaml and provides a singleton config accessor.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "rubric" / "extraction_rules.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """
    Load and cache the extraction rules configuration.

    Resolution order:
      1. REFINERY_CONFIG_PATH environment variable
      2. rubric/extraction_rules.yaml relative to project root
    """
    config_path_str = os.environ.get("REFINERY_CONFIG_PATH")
    if config_path_str:
        config_path = Path(config_path_str)
    else:
        config_path = _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Extraction rules config not found at {config_path}. "
            "Set REFINERY_CONFIG_PATH or ensure rubric/extraction_rules.yaml exists."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_confidence_thresholds() -> dict[str, float]:
    """Return the confidence gate thresholds."""
    cfg = get_config()
    return cfg.get("confidence_gates", {})


def get_page_signal_thresholds() -> dict[str, Any]:
    """Return the page-level signal thresholds."""
    cfg = get_config()
    return cfg.get("page_signals", {})


def get_document_classification_thresholds() -> dict[str, Any]:
    """Return document-level classification thresholds."""
    cfg = get_config()
    return cfg.get("document_classification", {})


def get_chunking_config() -> dict[str, Any]:
    """Return chunking constitution parameters."""
    cfg = get_config()
    return cfg.get("chunking", {})


def get_budget_config() -> dict[str, Any]:
    """Return budget guard configuration."""
    cfg = get_config()
    return cfg.get("budget", {})


def get_domain_hints_config() -> dict[str, Any]:
    """Return domain hint keyword lists."""
    cfg = get_config()
    return cfg.get("domain_hints", {})


def get_logging_config() -> dict[str, Any]:
    """Return logging and ledger path configuration."""
    cfg = get_config()
    return cfg.get("logging", {})
