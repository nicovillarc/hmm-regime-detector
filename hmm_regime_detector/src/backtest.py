"""Backtest engine: buy-and-hold SPY vs. HMM risk-overlay strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    HIGH_VOL_ENTER_THRESHOLD,
    HIGH_VOL_EXIT_THRESHOLD,
    HYSTERESIS_CONSECUTIVE_DAYS,
    POSITION_CASH,
    POSITION_INVESTED,
    RISK_FREE_RATE,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary statistics and time-series for a backtest run."""

    equity_buy_hold: pd.Series
    equity_hmm: pd.Series
    positions: pd.Series
    p_high_vol: pd.Series
    regimes: pd.Series
    stats: dict[str, float]


def apply_hysteresis(p_high_vol: pd.Series) -> pd.Series:
    """
    Convert raw posterior probabilities into discrete positions with hysteresis.

    Rules
    -----
    * Start in INVESTED.
    * Switch to CASH  when P(high_vol) >= 0.60 for 3 consecutive days.
    * Switch to INVESTED when P(high_vol) <= 0.40 for 3 consecutive days.
    * Otherwise keep the current position.
    """
    positions: list[str] = []
    current = POSITION_INVESTED
    consecutive_high = 0
    consecutive_low = 0

    for prob in p_high_vol:
        if prob >= HIGH_VOL_ENTER_THRESHOLD:
            consecutive_high += 1
            consecutive_low = 0
        elif prob <= HIGH_VOL_EXIT_THRESHOLD:
            consecutive_low += 1
            consecutive_high = 0
        else:
            consecutive_high = 0
            consecutive_low = 0

        if current == POSITION_INVESTED and consecutive_high >= HYSTERESIS_CONSECUTIVE_DAYS:
            current = POSITION_CASH
        elif current == POSITION_CASH and consecutive_low >= HYSTERESIS_CONSECUTIVE_DAYS:
            current = POSITION_INVESTED

        positions.append(current)

    return pd.Series(positions, index=p_high_vol.index, name="position")


def run_backtest(
    spy_prices: pd.Series,
    p_high_vol: pd.Series,
    regimes: pd.Series | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> BacktestResult:
    """
    Compare buy-and-hold SPY against the HMM risk-overlay strategy.

    The HMM strategy is fully invested in SPY when the hysteresis position
    is INVESTED and earns the risk-free rate (default 0 %) when in CASH.

    Parameters
    ----------
    spy_prices : pd.Series
        Daily SPY close prices, aligned with p_high_vol.
    p_high_vol : pd.Series
        Posterior probability of the high-volatility regime.
    regimes : pd.Series | None
        Optional most-likely regime labels (for plotting).
    risk_free_rate : float
        Annualised risk-free rate for cash periods.

    Returns
    -------
    BacktestResult
    """
    aligned = pd.DataFrame(
        {"spy_close": spy_prices, "p_high_vol": p_high_vol}
    ).dropna()

    positions = apply_hysteresis(aligned["p_high_vol"])
    daily_returns = aligned["spy_close"].pct_change().fillna(0.0)
    daily_rf = risk_free_rate / 252.0

    invested_mask = positions == POSITION_INVESTED
    strategy_returns = np.where(invested_mask, daily_returns, daily_rf)

    equity_bh = (1.0 + daily_returns).cumprod()
    equity_hmm = pd.Series(
        (1.0 + strategy_returns).cumprod(),
        index=aligned.index,
        name="equity_hmm",
    )

    stats = _compute_stats(
        daily_returns, strategy_returns, equity_bh, equity_hmm, invested_mask
    )

    regime_series = regimes.reindex(aligned.index) if regimes is not None else None

    logger.info(
        "Backtest complete — BH CAGR: %.2f%%, HMM CAGR: %.2f%%",
        stats["bh_cagr"] * 100,
        stats["hmm_cagr"] * 100,
    )

    return BacktestResult(
        equity_buy_hold=equity_bh,
        equity_hmm=equity_hmm,
        positions=positions,
        p_high_vol=aligned["p_high_vol"],
        regimes=regime_series,
        stats=stats,
    )


def _compute_stats(
    bh_returns: pd.Series,
    hmm_returns: np.ndarray,
    equity_bh: pd.Series,
    equity_hmm: pd.Series,
    invested_mask: pd.Series,
) -> dict[str, float]:
    """Compute CAGR, volatility, Sharpe, and max drawdown for both strategies."""
    n_years = len(bh_returns) / 252.0

    bh_cagr = (equity_bh.iloc[-1]) ** (1.0 / n_years) - 1.0
    hmm_cagr = (equity_hmm.iloc[-1]) ** (1.0 / n_years) - 1.0

    bh_vol = bh_returns.std() * np.sqrt(252)
    hmm_vol = pd.Series(hmm_returns).std() * np.sqrt(252)

    bh_sharpe = bh_cagr / bh_vol if bh_vol > 0 else np.nan
    hmm_sharpe = hmm_cagr / hmm_vol if hmm_vol > 0 else np.nan

    return {
        "bh_cagr": bh_cagr,
        "hmm_cagr": hmm_cagr,
        "bh_volatility": bh_vol,
        "hmm_volatility": hmm_vol,
        "bh_sharpe": bh_sharpe,
        "hmm_sharpe": hmm_sharpe,
        "bh_max_drawdown": _max_drawdown(equity_bh),
        "hmm_max_drawdown": _max_drawdown(equity_hmm),
        "pct_days_invested": float(invested_mask.mean()),
    }


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a positive fraction."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min())
