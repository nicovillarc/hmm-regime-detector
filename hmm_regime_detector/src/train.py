#!/usr/bin/env python3
"""
Train the HMM regime detector on historical SPY / VIX data.

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_market_data
from src.features import build_features
from src.model import HMMRegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== HMM Regime Detector — Training ===")

    market_data = download_market_data()
    features = build_features(market_data)

    detector = HMMRegimeDetector()
    detector.fit(features)
    detector.save()

    logger.info("Training complete. Model saved to data/hmm_model.pkl")


if __name__ == "__main__":
    main()
