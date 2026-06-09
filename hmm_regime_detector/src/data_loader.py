"""Download and persist SPY / VIX daily market data via yfinance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import DATA_START, RAW_DATA_PATH, SPY_TICKER, VIX_TICKER

logger = logging.getLogger(__name__)


def download_market_data(
    start: str = DATA_START,
    end: str | None = None,
    save_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for SPY and VIX.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD).
    end : str | None
        End date; defaults to today.
    save_path : Path
        CSV path where the merged dataset is persisted.

    Returns
    -------
    pd.DataFrame
        Columns: spy_close, vix_close. Index: DatetimeIndex (UTC-normalised).
    """
    logger.info("Downloading %s and %s from %s …", SPY_TICKER, VIX_TICKER, start)

    spy = yf.download(SPY_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    vix = yf.download(VIX_TICKER, start=start, end=end, progress=False, auto_adjust=True)

    if spy.empty or vix.empty:
        raise ValueError("Downloaded data is empty. Check tickers and date range.")

    # yfinance may return MultiIndex columns when a single ticker is requested.
    spy_close = _extract_close(spy, SPY_TICKER)
    vix_close = _extract_close(vix, VIX_TICKER)

    df = pd.DataFrame({"spy_close": spy_close, "vix_close": vix_close})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    logger.info("Saved %d rows to %s", len(df), save_path)

    return df


def load_market_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load previously saved market data from CSV."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _extract_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Normalise yfinance output to a single Close price series."""
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Adjusted Close when available.
        for col_name in ("Close", "Adj Close"):
            if (col_name, ticker) in df.columns:
                return df[(col_name, ticker)].squeeze()
        raise KeyError(f"Close column not found for {ticker}")
    if "Close" in df.columns:
        return df["Close"].squeeze()
    raise KeyError(f"Close column not found in downloaded data for {ticker}")
