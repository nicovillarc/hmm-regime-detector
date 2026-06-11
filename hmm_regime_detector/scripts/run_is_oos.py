#!/usr/bin/env python3
"""
Run strict in-sample / out-of-sample validation (v1.1).

The HMM is trained exclusively on 2005-02-01 → 2018-12-31 and evaluated
on 2019-01-01 → present without retraining.

Usage
-----
    python scripts/run_is_oos.py
"""

from __future__ import annotations

import logging

from _bootstrap import PROJECT_ROOT  # noqa: F401 — sets sys.path

from src.covid_oos_forensics import run_covid_oos_forensics
from src.data_loader import download_market_data
from src.features import build_features
from src.is_oos_validation import run_is_oos_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== HMM Regime Detector — IS/OOS Validation (v1.1) ===")

    market_data = download_market_data()
    features = build_features(market_data)
    run_is_oos_validation(features)
    run_covid_oos_forensics(features)


if __name__ == "__main__":
    main()
