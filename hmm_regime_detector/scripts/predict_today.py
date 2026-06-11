#!/usr/bin/env python3
"""
Generate today's regime report using the trained HMM model.

Downloads the latest data, runs inference, and writes a JSON report
to reports/latest_report.json.

Usage
-----
    python scripts/predict_today.py
"""

from __future__ import annotations

import logging
import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401 — sets sys.path

from src.config import MODEL_PATH
from src.data_loader import download_market_data
from src.features import build_features
from src.model import HMMRegimeDetector
from src.report import generate_daily_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== HMM Regime Detector — Daily Prediction ===")

    if not MODEL_PATH.exists():
        logger.error("Model not found. Run src/train.py first.")
        sys.exit(1)

    market_data = download_market_data()
    features = build_features(market_data)

    detector = HMMRegimeDetector.load()
    regime_result = detector.predict_proba(features)

    generate_daily_report(regime_result)


if __name__ == "__main__":
    main()
