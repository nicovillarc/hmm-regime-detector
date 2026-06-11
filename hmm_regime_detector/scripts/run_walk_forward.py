#!/usr/bin/env python3
"""
Run walk-forward validation (v1.2).

Expanding-window retraining with one-year test folds from 2015 onward.
Strategy returns use a one-day signal shift.

Usage
-----
    python scripts/run_walk_forward.py
"""

from __future__ import annotations

import logging

from _bootstrap import PROJECT_ROOT  # noqa: F401 — sets sys.path

from src.data_loader import download_market_data
from src.features import build_features
from src.walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== HMM Regime Detector — Walk-Forward Validation (v1.2) ===")

    market_data = download_market_data()
    features = build_features(market_data)
    run_walk_forward(features)


if __name__ == "__main__":
    main()
