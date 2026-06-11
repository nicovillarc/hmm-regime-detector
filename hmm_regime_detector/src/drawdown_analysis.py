"""Drawdown analysis plots for all validation frameworks (visualization only)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import run_backtest
from src.config import (
    DRAWDOWN_COMPARISON_PNG,
    DRAWDOWN_SUMMARY_JSON,
    FIGURE_DPI,
    IS_END,
    IS_MODEL_PATH,
    IS_OOS_DRAWDOWNS_PNG,
    IS_START,
    MODEL_PATH,
    OOS_START,
    PLOT_STYLE,
    REPORTS_DIR,
    WF_FIRST_TEST_YEAR,
    WALK_FORWARD_DRAWDOWNS_PNG,
)
from src.is_oos_validation import _run_period_backtest, _split_features
from src.model import HMMRegimeDetector
from src.walk_forward import _get_test_years, _run_fold

logger = logging.getLogger(__name__)

FULL_SAMPLE_START = "2005-02-01"


def run_drawdown_analysis(
    features: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """
    Generate drawdown comparison plots and summary JSON.

    Uses saved models where available (full-sample, IS/OOS).
    Walk-forward drawdowns reuse the v1.2 fold execution (1-day signal shift).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    full = _full_sample_drawdowns(features)
    is_oos = _is_oos_drawdowns(features)
    wf = _walk_forward_drawdowns(features)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "full_sample": {
            "buy_hold_max_dd": round(full["bh_max_dd"], 6),
            "hmm_max_dd": round(full["hmm_max_dd"], 6),
        },
        "is_oos": {
            "is_buy_hold_max_dd": round(is_oos["is_bh_max_dd"], 6),
            "is_hmm_max_dd": round(is_oos["is_hmm_max_dd"], 6),
            "oos_buy_hold_max_dd": round(is_oos["oos_bh_max_dd"], 6),
            "oos_hmm_max_dd": round(is_oos["oos_hmm_max_dd"], 6),
        },
        "walk_forward": {
            "buy_hold_max_dd": round(wf["bh_max_dd"], 6),
            "hmm_max_dd": round(wf["hmm_max_dd"], 6),
        },
    }

    plot_full_sample_drawdowns(
        full["bh_dd"], full["hmm_dd"],
        full["bh_max_dd"], full["hmm_max_dd"],
        output_dir / DRAWDOWN_COMPARISON_PNG.name,
    )
    plot_is_oos_drawdowns(
        is_oos["is_bh_dd"], is_oos["is_hmm_dd"],
        is_oos["is_bh_max_dd"], is_oos["is_hmm_max_dd"],
        is_oos["oos_bh_dd"], is_oos["oos_hmm_dd"],
        is_oos["oos_bh_max_dd"], is_oos["oos_hmm_max_dd"],
        output_dir / IS_OOS_DRAWDOWNS_PNG.name,
    )
    plot_walk_forward_drawdowns(
        wf["bh_dd"], wf["hmm_dd"],
        wf["bh_max_dd"], wf["hmm_max_dd"],
        output_dir / WALK_FORWARD_DRAWDOWNS_PNG.name,
    )

    summary_path = output_dir / DRAWDOWN_SUMMARY_JSON.name
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Drawdown analysis saved → %s", output_dir)
    return summary


# ---------------------------------------------------------------------------
# Drawdown helpers
# ---------------------------------------------------------------------------

def equity_from_returns(returns: pd.Series) -> pd.Series:
    """Cumulative equity curve from daily returns."""
    return (1.0 + returns).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown as a negative percentage series."""
    running_max = equity.cummax()
    return (equity / running_max - 1.0) * 100.0


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a positive fraction."""
    dd = drawdown_series(equity)
    return float(-dd.min() / 100.0)


# ---------------------------------------------------------------------------
# Data paths per validation framework
# ---------------------------------------------------------------------------

def _full_sample_drawdowns(features: pd.DataFrame) -> dict[str, Any]:
    """v1.0 full-sample buy-and-hold vs HMM overlay."""
    sample = features[features.index >= pd.Timestamp(FULL_SAMPLE_START)].copy()
    detector = _load_model(MODEL_PATH)
    result = detector.predict_proba(sample)

    bt = run_backtest(
        spy_prices=sample["spy_close"],
        p_high_vol=pd.Series(result.p_high_vol, index=result.dates),
    )

    bh_dd = drawdown_series(bt.equity_buy_hold)
    hmm_dd = drawdown_series(bt.equity_hmm)
    return {
        "bh_dd": bh_dd,
        "hmm_dd": hmm_dd,
        "bh_max_dd": bt.stats["bh_max_drawdown"],
        "hmm_max_dd": bt.stats["hmm_max_drawdown"],
    }


def _is_oos_drawdowns(features: pd.DataFrame) -> dict[str, Any]:
    """v1.1 IS/OOS drawdowns using the IS-trained model."""
    is_features, oos_features = _split_features(features)
    detector = _load_model(IS_MODEL_PATH)

    is_result = detector.predict_proba(is_features)
    oos_result = detector.predict_proba(oos_features)

    is_bt = _run_period_backtest(is_features, is_result)
    oos_bt = _run_period_backtest(oos_features, oos_result)

    return {
        "is_bh_dd": drawdown_series(is_bt.equity_buy_hold),
        "is_hmm_dd": drawdown_series(is_bt.equity_hmm),
        "is_bh_max_dd": is_bt.stats["bh_max_drawdown"],
        "is_hmm_max_dd": is_bt.stats["hmm_max_drawdown"],
        "oos_bh_dd": drawdown_series(oos_bt.equity_buy_hold),
        "oos_hmm_dd": drawdown_series(oos_bt.equity_hmm),
        "oos_bh_max_dd": oos_bt.stats["bh_max_drawdown"],
        "oos_hmm_max_dd": oos_bt.stats["hmm_max_drawdown"],
    }


def _walk_forward_drawdowns(features: pd.DataFrame) -> dict[str, Any]:
    """v1.2 walk-forward stitched drawdowns (1-day signal shift)."""
    bh_parts: list[pd.Series] = []
    hmm_parts: list[pd.Series] = []

    for test_year in _get_test_years(features):
        fold = _run_fold(features, test_year)
        bh_parts.append(fold["bh_returns"])
        hmm_parts.append(fold["hmm_returns"])

    bh_returns = pd.concat(bh_parts).sort_index()
    hmm_returns = pd.concat(hmm_parts).sort_index()

    equity_bh = equity_from_returns(bh_returns)
    equity_hmm = equity_from_returns(hmm_returns)

    return {
        "bh_dd": drawdown_series(equity_bh),
        "hmm_dd": drawdown_series(equity_hmm),
        "bh_max_dd": max_drawdown(equity_bh),
        "hmm_max_dd": max_drawdown(equity_hmm),
    }


def _load_model(path: Path) -> HMMRegimeDetector:
    """Load a saved model; raises if not found (no retraining)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train the model before running drawdown analysis."
        )
    return HMMRegimeDetector.load(path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_full_sample_drawdowns(
    bh_dd: pd.Series,
    hmm_dd: pd.Series,
    bh_max: float,
    hmm_max: float,
    path: Path,
) -> None:
    """Single-panel full-sample drawdown comparison."""
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        bh_dd.index, bh_dd.values,
        color="#2c3e50", linewidth=0.9,
        label=f"Buy & Hold (Max DD: {bh_max:.1%})",
    )
    ax.plot(
        hmm_dd.index, hmm_dd.values,
        color="#27ae60", linewidth=0.9,
        label=f"HMM Overlay (Max DD: {hmm_max:.1%})",
    )

    ax.set_title(
        "Full-Sample Drawdown Comparison (v1.0) — 2005-02-01 → present",
        fontweight="bold",
    )
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_is_oos_drawdowns(
    is_bh_dd: pd.Series,
    is_hmm_dd: pd.Series,
    is_bh_max: float,
    is_hmm_max: float,
    oos_bh_dd: pd.Series,
    oos_hmm_dd: pd.Series,
    oos_bh_max: float,
    oos_hmm_max: float,
    path: Path,
) -> None:
    """Two-panel IS/OOS drawdown comparison."""
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    _drawdown_panel(
        axes[0], is_bh_dd, is_hmm_dd, is_bh_max, is_hmm_max,
        f"In-Sample Drawdowns (v1.1) — {IS_START} → {IS_END}",
    )
    _drawdown_panel(
        axes[1], oos_bh_dd, oos_hmm_dd, oos_bh_max, oos_hmm_max,
        f"Out-of-Sample Drawdowns (v1.1) — {OOS_START} → present",
    )
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_walk_forward_drawdowns(
    bh_dd: pd.Series,
    hmm_dd: pd.Series,
    bh_max: float,
    hmm_max: float,
    path: Path,
) -> None:
    """Walk-forward stitched drawdown comparison."""
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        bh_dd.index, bh_dd.values,
        color="#2c3e50", linewidth=0.9,
        label=f"Buy & Hold (Max DD: {bh_max:.1%})",
    )
    ax.plot(
        hmm_dd.index, hmm_dd.values,
        color="#27ae60", linewidth=0.9,
        label=f"HMM Walk-Forward (Max DD: {hmm_max:.1%})",
    )

    ax.set_title(
        f"Walk-Forward Drawdown Comparison (v1.2) — "
        f"{WF_FIRST_TEST_YEAR} → present (1-day signal shift)",
        fontweight="bold",
    )
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _drawdown_panel(
    ax: plt.Axes,
    bh_dd: pd.Series,
    hmm_dd: pd.Series,
    bh_max: float,
    hmm_max: float,
    title: str,
) -> None:
    ax.plot(
        bh_dd.index, bh_dd.values,
        color="#2c3e50", linewidth=0.9,
        label=f"Buy & Hold (Max DD: {bh_max:.1%})",
    )
    ax.plot(
        hmm_dd.index, hmm_dd.values,
        color="#27ae60", linewidth=0.9,
        label=f"HMM Overlay (Max DD: {hmm_max:.1%})",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
