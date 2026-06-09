"""Report generation and visualisation for regime detection."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.backtest import BacktestResult, apply_hysteresis
from src.config import (
    FIGURE_DPI,
    PLOT_STYLE,
    REGIME_HIGH_VOL,
    REPORTS_DIR,
)
from src.model import RegimeResult

logger = logging.getLogger(__name__)


def generate_daily_report(
    regime_result: RegimeResult,
    output_dir: Path = REPORTS_DIR,
) -> dict:
    """
    Build a JSON report for the most recent trading day.

    Returns
    -------
    dict
        Keys: date, p_normal, p_high_vol, current_regime, recommendation.
    """
    latest_idx = -1
    p_high = float(regime_result.p_high_vol[latest_idx])
    p_normal = float(regime_result.p_normal[latest_idx])
    current_regime = str(regime_result.most_likely_regime[latest_idx])
    latest_date = regime_result.dates[latest_idx]

    # Hysteresis requires the last N days of probabilities.
    p_series = pd.Series(
        regime_result.p_high_vol, index=regime_result.dates, name="p_high_vol"
    )
    position = apply_hysteresis(p_series).iloc[-1]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_date": str(latest_date.date()),
        "p_normal": round(p_normal, 4),
        "p_high_volatility": round(p_high, 4),
        "current_regime": current_regime,
        "recommendation": position,
        "hysteresis_rules": {
            "enter_cash": f"P(high_vol) >= 0.60 for 3 consecutive days",
            "enter_invested": f"P(high_vol) <= 0.40 for 3 consecutive days",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latest_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    _print_report(report)
    logger.info("Report saved → %s", report_path)
    return report


def _print_report(report: dict) -> None:
    """Pretty-print the daily report to stdout."""
    sep = "=" * 55
    print(f"\n{sep}")
    print("  HMM REGIME DETECTOR — DAILY REPORT")
    print(sep)
    print(f"  Latest date          : {report['latest_date']}")
    print(f"  P(normal regime)     : {report['p_normal']:.2%}")
    print(f"  P(high vol regime)   : {report['p_high_volatility']:.2%}")
    print(f"  Current regime       : {report['current_regime']}")
    print(f"  Recommendation       : {report['recommendation']}")
    print(f"{sep}\n")


def plot_regime_analysis(
    spy_prices: pd.Series,
    regime_result: RegimeResult,
    backtest_result: BacktestResult,
    output_dir: Path = REPORTS_DIR,
) -> Path:
    """
    Create a three-panel figure:
      1. SPY price with regime shading
      2. P(high volatility) through time
      3. Equity curves (buy-and-hold vs. HMM)
    """
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    dates = regime_result.dates

    # --- Panel 1: SPY price with regime highlights ---
    ax1 = axes[0]
    ax1.plot(dates, spy_prices.reindex(dates).values, color="black", linewidth=0.8, label="SPY")

    high_vol_mask = regime_result.most_likely_regime == REGIME_HIGH_VOL
    _shade_regimes(ax1, dates, high_vol_mask, color="#ffcccc", alpha=0.5)

    ax1.set_title("SPY Price with Detected Regimes", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price ($)")
    normal_patch = mpatches.Patch(color="#ccffcc", alpha=0.5, label="Normal")
    crisis_patch = mpatches.Patch(color="#ffcccc", alpha=0.5, label="High Volatility")
    ax1.legend(handles=[normal_patch, crisis_patch], loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: P(high volatility) ---
    ax2 = axes[1]
    ax2.plot(dates, regime_result.p_high_vol, color="#c0392b", linewidth=0.8, label="P(high vol)")
    ax2.axhline(0.60, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="Enter CASH (0.60)")
    ax2.axhline(0.40, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="Enter INVESTED (0.40)")
    ax2.fill_between(
        dates,
        regime_result.p_high_vol,
        alpha=0.15,
        color="#c0392b",
    )
    ax2.set_title("Posterior P(High Volatility Regime)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Equity curves ---
    ax3 = axes[2]
    bt_dates = backtest_result.equity_buy_hold.index
    ax3.plot(
        bt_dates,
        backtest_result.equity_buy_hold.values,
        color="#2c3e50",
        linewidth=1.2,
        label="Buy & Hold SPY",
    )
    ax3.plot(
        bt_dates,
        backtest_result.equity_hmm.values,
        color="#27ae60",
        linewidth=1.2,
        label="HMM Risk Overlay",
    )
    ax3.set_title("Equity Curves", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Growth of $1")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Annotate backtest stats
    stats = backtest_result.stats
    stats_text = (
        f"BH  CAGR: {stats['bh_cagr']:.1%}  |  MaxDD: {stats['bh_max_drawdown']:.1%}\n"
        f"HMM CAGR: {stats['hmm_cagr']:.1%}  |  MaxDD: {stats['hmm_max_drawdown']:.1%}"
    )
    ax3.text(
        0.02, 0.05, stats_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "regime_analysis.png"
    fig.savefig(plot_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Plot saved → %s", plot_path)
    return plot_path


def _shade_regimes(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    mask: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    """Shade contiguous high-volatility periods on a price chart."""
    in_region = False
    start = None

    for i, (date, is_high) in enumerate(zip(dates, mask)):
        if is_high and not in_region:
            start = date
            in_region = True
        elif not is_high and in_region:
            ax.axvspan(start, date, color=color, alpha=alpha)
            in_region = False

    if in_region:
        ax.axvspan(start, dates[-1], color=color, alpha=alpha)


def save_backtest_summary(
    backtest_result: BacktestResult,
    output_dir: Path = REPORTS_DIR,
) -> Path:
    """Persist backtest statistics as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "backtest_summary.json"

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {k: round(v, 6) if isinstance(v, float) else v
                  for k, v in backtest_result.stats.items()},
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Backtest summary saved → %s", summary_path)
    return summary_path
