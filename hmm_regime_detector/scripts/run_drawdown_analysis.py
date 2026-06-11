#!/usr/bin/env python3
"""
Generate drawdown comparison plots for all validation frameworks.

Visualization only — uses saved models and existing validation logic.

Usage
-----
    python scripts/run_drawdown_analysis.py
"""

from __future__ import annotations

import json
import logging

from _bootstrap import PROJECT_ROOT  # noqa: F401

from src.data_loader import load_market_data
from src.drawdown_analysis import run_drawdown_analysis
from src.features import build_features, load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== HMM Regime Detector — Drawdown Analysis ===")

    from src.config import FEATURES_PATH, RAW_DATA_PATH

    if FEATURES_PATH.exists() and RAW_DATA_PATH.exists():
        features = load_features()
        logger.info("Loaded cached features from %s", FEATURES_PATH)
    else:
        features = build_features(load_market_data())

    summary = run_drawdown_analysis(features)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
