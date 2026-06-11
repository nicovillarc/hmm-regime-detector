#!/usr/bin/env python3
"""
Run a full backtest comparing buy-and-hold SPY vs. the HMM risk overlay.

Trains (or loads) the model, computes regime probabilities, applies
hysteresis, and produces plots + summary statistics.

Usage
-----
    python scripts/run_backtest.py
"""

from __future__ import annotations

import logging

import pandas as pd

from _bootstrap import PROJECT_ROOT

from src.backtest import run_backtest
from src.data_loader import download_market_data
from src.features import build_features
from src.model import HMMRegimeDetector
from src.covid_forensics import run_covid_forensics
from src.covid_oos_forensics import run_covid_oos_forensics
from src.is_oos_validation import run_is_oos_validation
from src.model_audit import run_model_audit
from src.regime_diagnostics import run_regime_diagnostics
from src.report import plot_regime_analysis, save_backtest_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== HMM Regime Detector — Backtest ===")

    market_data = download_market_data()
    features = build_features(market_data)

    model_path = PROJECT_ROOT / "data" / "hmm_model.pkl"
    if model_path.exists():
        detector = HMMRegimeDetector.load()
        logger.info("Loaded existing model.")
    else:
        detector = HMMRegimeDetector()
        detector.fit(features)
        detector.save()
        logger.info("No saved model found — trained a new one.")

    regime_result = detector.predict_proba(features)

    backtest_result = run_backtest(
        spy_prices=features["spy_close"],
        p_high_vol=pd.Series(
            regime_result.p_high_vol,
            index=regime_result.dates,
            name="p_high_vol",
        ),
        regimes=pd.Series(
            regime_result.most_likely_regime,
            index=regime_result.dates,
            name="regime",
        ),
    )

    plot_regime_analysis(
        spy_prices=features["spy_close"],
        regime_result=regime_result,
        backtest_result=backtest_result,
    )
    save_backtest_summary(backtest_result)
    run_model_audit(detector, features)
    run_regime_diagnostics(detector, features, regime_result)
    run_covid_forensics(detector, features, regime_result)
    run_is_oos_validation(features)
    run_covid_oos_forensics(features)

    stats = backtest_result.stats
    print("\n--- Backtest Summary ---")
    print(f"  Buy & Hold CAGR     : {stats['bh_cagr']:.2%}")
    print(f"  HMM Strategy CAGR   : {stats['hmm_cagr']:.2%}")
    print(f"  Buy & Hold Max DD   : {stats['bh_max_drawdown']:.2%}")
    print(f"  HMM Strategy Max DD : {stats['hmm_max_drawdown']:.2%}")
    print(f"  Days Invested       : {stats['pct_days_invested']:.1%}")
    print("  Plots saved to reports/regime_analysis.png\n")


if __name__ == "__main__":
    main()
