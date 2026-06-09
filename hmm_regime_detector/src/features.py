"""Feature engineering for the HMM regime detector."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    REALIZED_VOL_WINDOW,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


def build_features(
    market_data: pd.DataFrame,
    save_path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    """
    Construct model features from raw SPY / VIX prices.

    Features
    --------
    spy_log_return      : daily log return of SPY
    spy_realized_vol    : 20-day rolling std of log returns × sqrt(252)
    log_vix             : natural log of VIX level

    Rows with insufficient history (first 20 trading days) are dropped.

    Parameters
    ----------
    market_data : pd.DataFrame
        Must contain columns spy_close and vix_close.
    save_path : Path
        Optional CSV persistence path.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with spy_close for backtesting / plotting.
    """
    df = market_data.copy()

    df["spy_log_return"] = np.log(df["spy_close"] / df["spy_close"].shift(1))
    df["spy_realized_vol"] = (
        df["spy_log_return"]
        .rolling(window=REALIZED_VOL_WINDOW)
        .std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    df["log_vix"] = np.log(df["vix_close"])

    features = df[FEATURE_COLUMNS + ["spy_close"]].dropna()
    features = features.sort_index()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(save_path)
    logger.info("Built %d feature rows → %s", len(features), save_path)

    return features


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load a previously saved feature matrix."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
