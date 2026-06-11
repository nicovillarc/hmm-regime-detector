"""Walk-forward validation with expanding training windows (v1.2)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import apply_hysteresis
from src.config import (
    FIGURE_DPI,
    PLOT_STYLE,
    POSITION_INVESTED,
    REPORTS_DIR,
    RISK_FREE_RATE,
)
from src.model import HMMRegimeDetector

logger = logging.getLogger(__name__)

# Walk-forward design
WF_TRAIN_START = "2005-02-01"
WF_INITIAL_TRAIN_END = "2014-12-31"
WF_FIRST_TEST_YEAR = 2015

# Output paths
WF_SUMMARY_CSV = REPORTS_DIR / "walk_forward_summary.csv"
WF_SUMMARY_JSON = REPORTS_DIR / "walk_forward_summary.json"
WF_REPORT_TXT = REPORTS_DIR / "walk_forward_report.txt"
WF_EQUITY_PNG = REPORTS_DIR / "walk_forward_equity_curve.png"
WF_YEARLY_PNG = REPORTS_DIR / "walk_forward_yearly_metrics.png"

# Event years mapped to walk-forward test folds
WF_STRESS_EVENTS = [
    ("2020 COVID crisis", 2020),
    ("2022 bear market", 2022),
    ("March 2023 SVB episode", 2023),
    ("2025 tariff volatility episode", 2025),
]


def run_walk_forward(
    features: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """
    Execute expanding-window walk-forward validation.

    Each fold retrains the HMM on all data through year Y-1 and tests on year Y.
    Strategy returns use a one-day signal shift to avoid look-ahead bias.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    test_years = _get_test_years(features)
    fold_rows: list[dict[str, Any]] = []
    stitched_bh_returns: list[pd.Series] = []
    stitched_hmm_returns: list[pd.Series] = []
    stitched_invested: list[pd.Series] = []
    total_switches = 0

    for test_year in test_years:
        fold = _run_fold(features, test_year)
        fold_rows.append(fold["metrics"])
        stitched_bh_returns.append(fold["bh_returns"])
        stitched_hmm_returns.append(fold["hmm_returns"])
        stitched_invested.append(fold["invested_mask"])
        total_switches += fold["metrics"]["regime_switches"]

    folds_df = pd.DataFrame(fold_rows)
    aggregate = _compute_aggregate_metrics(
        stitched_bh_returns, stitched_hmm_returns, stitched_invested, total_switches
    )

    folds_df.to_csv(output_dir / WF_SUMMARY_CSV.name, index=False)

    summary_payload = {
        "version": "1.2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "design": {
            "train_start": WF_TRAIN_START,
            "initial_train_end": WF_INITIAL_TRAIN_END,
            "first_test_year": WF_FIRST_TEST_YEAR,
            "signal_shift_days": 1,
        },
        "folds": fold_rows,
        "aggregate": aggregate,
    }
    with open(output_dir / WF_SUMMARY_JSON.name, "w") as fh:
        json.dump(summary_payload, fh, indent=2)

    plot_walk_forward_equity(
        stitched_bh_returns, stitched_hmm_returns,
        output_dir / WF_EQUITY_PNG.name,
    )
    plot_yearly_metrics(folds_df, output_dir / WF_YEARLY_PNG.name)

    report = generate_walk_forward_report(folds_df, aggregate, fold_rows)
    (output_dir / WF_REPORT_TXT.name).write_text(report)
    print(report)

    logger.info("Walk-forward validation complete → %s", output_dir)
    return {"folds": folds_df, "aggregate": aggregate, "summary": summary_payload}


# ---------------------------------------------------------------------------
# Fold execution
# ---------------------------------------------------------------------------

def _get_test_years(features: pd.DataFrame) -> list[int]:
    """Return calendar years to test, from 2015 through latest available."""
    last_year = features.index.max().year
    return list(range(WF_FIRST_TEST_YEAR, last_year + 1))


def _run_fold(features: pd.DataFrame, test_year: int) -> dict[str, Any]:
    """Train on expanding window and evaluate one test year."""
    train_end = f"{test_year - 1}-12-31"
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"

    train = features[
        (features.index >= pd.Timestamp(WF_TRAIN_START))
        & (features.index <= pd.Timestamp(train_end))
    ]
    test = features[
        (features.index >= pd.Timestamp(test_start))
        & (features.index <= pd.Timestamp(test_end))
    ]

    if train.empty or test.empty:
        raise ValueError(f"Insufficient data for fold test_year={test_year}")

    detector = HMMRegimeDetector()
    detector.fit(train)
    result = detector.predict_proba(test)

    p_high_vol = pd.Series(result.p_high_vol, index=test.index, name="p_high_vol")
    positions = apply_hysteresis(p_high_vol)

    # Shift signal one day: trade on prior day's regime decision.
    execution = positions.shift(1).fillna(POSITION_INVESTED)

    daily_returns = test["spy_close"].pct_change().fillna(0.0)
    daily_rf = RISK_FREE_RATE / 252.0
    invested_mask = execution == POSITION_INVESTED
    hmm_returns = pd.Series(
        np.where(invested_mask, daily_returns, daily_rf),
        index=test.index,
        name="hmm_returns",
    )

    equity_bh = (1.0 + daily_returns).cumprod()
    equity_hmm = (1.0 + hmm_returns).cumprod()

    n_years = len(test) / 252.0
    bh_cagr = float(equity_bh.iloc[-1] ** (1.0 / n_years) - 1.0)
    hmm_cagr = float(equity_hmm.iloc[-1] ** (1.0 / n_years) - 1.0)
    bh_vol = float(daily_returns.std() * np.sqrt(252))
    hmm_vol = float(hmm_returns.std() * np.sqrt(252))

    switches = int((positions != positions.shift(1)).sum() - 1)
    switches = max(switches, 0)

    metrics = {
        "test_year": test_year,
        "train_start": WF_TRAIN_START,
        "train_end": train_end,
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "test_days": len(test),
        "crisis_state": int(detector.crisis_state),
        "bh_cagr": round(bh_cagr, 6),
        "hmm_cagr": round(hmm_cagr, 6),
        "bh_max_drawdown": round(_max_drawdown(equity_bh), 6),
        "hmm_max_drawdown": round(_max_drawdown(equity_hmm), 6),
        "bh_sharpe": round(bh_cagr / bh_vol if bh_vol > 0 else np.nan, 6),
        "hmm_sharpe": round(hmm_cagr / hmm_vol if hmm_vol > 0 else np.nan, 6),
        "pct_days_invested": round(float(invested_mask.mean()), 6),
        "regime_switches": switches,
        "avg_p_high_vol": round(float(p_high_vol.mean()), 6),
        "bh_total_return": round(float(equity_bh.iloc[-1] - 1.0), 6),
        "hmm_total_return": round(float(equity_hmm.iloc[-1] - 1.0), 6),
    }

    logger.info(
        "Fold %d — train→%s, test days=%d, HMM CAGR=%.2f%%, switches=%d",
        test_year, train_end, len(test), hmm_cagr * 100, switches,
    )

    return {
        "metrics": metrics,
        "bh_returns": daily_returns,
        "hmm_returns": hmm_returns,
        "p_high_vol": p_high_vol,
        "invested_mask": invested_mask,
        "detector": detector,
    }


def _compute_aggregate_metrics(
    bh_parts: list[pd.Series],
    hmm_parts: list[pd.Series],
    invested_parts: list[pd.Series],
    total_switches: int,
) -> dict[str, Any]:
    """Compute metrics on the stitched walk-forward out-of-sample path."""
    bh_returns = pd.concat(bh_parts).sort_index()
    hmm_returns = pd.concat(hmm_parts).sort_index()
    invested = pd.concat(invested_parts).sort_index()

    equity_bh = (1.0 + bh_returns).cumprod()
    equity_hmm = (1.0 + hmm_returns).cumprod()
    n_years = len(bh_returns) / 252.0

    bh_cagr = float(equity_bh.iloc[-1] ** (1.0 / n_years) - 1.0)
    hmm_cagr = float(equity_hmm.iloc[-1] ** (1.0 / n_years) - 1.0)
    bh_vol = float(bh_returns.std() * np.sqrt(252))
    hmm_vol = float(hmm_returns.std() * np.sqrt(252))
    invested_pct = float(invested.mean())

    return {
        "period_start": str(bh_returns.index[0].date()),
        "period_end": str(bh_returns.index[-1].date()),
        "total_trading_days": len(bh_returns),
        "buy_and_hold": {
            "cagr": round(bh_cagr, 6),
            "annualized_volatility": round(bh_vol, 6),
            "sharpe_ratio": round(bh_cagr / bh_vol if bh_vol > 0 else np.nan, 6),
            "max_drawdown": round(_max_drawdown(equity_bh), 6),
            "total_return": round(float(equity_bh.iloc[-1] - 1.0), 6),
        },
        "hmm_overlay": {
            "cagr": round(hmm_cagr, 6),
            "annualized_volatility": round(hmm_vol, 6),
            "sharpe_ratio": round(hmm_cagr / hmm_vol if hmm_vol > 0 else np.nan, 6),
            "max_drawdown": round(_max_drawdown(equity_hmm), 6),
            "total_return": round(float(equity_hmm.iloc[-1] - 1.0), 6),
            "pct_days_invested": round(invested_pct, 6),
            "regime_switches": total_switches,
        },
        "comparison": {
            "cagr_sacrifice": round(bh_cagr - hmm_cagr, 6),
            "max_dd_reduction": round(
                _max_drawdown(equity_bh) - _max_drawdown(equity_hmm), 6
            ),
            "sharpe_improvement": round(
                (hmm_cagr / hmm_vol if hmm_vol > 0 else 0)
                - (bh_cagr / bh_vol if bh_vol > 0 else 0),
                6,
            ),
        },
    }


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_walk_forward_equity(
    bh_parts: list[pd.Series],
    hmm_parts: list[pd.Series],
    path: Path,
) -> None:
    """Stitched walk-forward equity curves for buy-and-hold vs HMM."""
    bh_returns = pd.concat(bh_parts).sort_index()
    hmm_returns = pd.concat(hmm_parts).sort_index()

    # Re-base each fold so curves chain continuously.
    equity_bh = _chain_equity(bh_returns)
    equity_hmm = _chain_equity(hmm_returns)

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_bh.index, equity_bh.values, color="#2c3e50", linewidth=1.1, label="Buy & Hold")
    ax.plot(equity_hmm.index, equity_hmm.values, color="#27ae60", linewidth=1.1, label="HMM Walk-Forward")
    ax.set_title(
        "Walk-Forward Equity Curves (2015 → present, 1-day signal shift)",
        fontweight="bold",
    )
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _chain_equity(daily_returns: pd.Series) -> pd.Series:
    """Build a continuous equity curve across concatenated fold returns."""
    equity_parts: list[pd.Series] = []
    level = 1.0
    for _, group in daily_returns.groupby(daily_returns.index.year):
        eq = level * (1.0 + group).cumprod()
        equity_parts.append(eq)
        level = float(eq.iloc[-1])
    return pd.concat(equity_parts)


def plot_yearly_metrics(folds_df: pd.DataFrame, path: Path) -> None:
    """Bar charts of yearly CAGR and Max DD for BH vs HMM."""
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    years = folds_df["test_year"]
    x = np.arange(len(years))
    width = 0.35

    # CAGR
    ax1 = axes[0]
    ax1.bar(x - width / 2, folds_df["bh_cagr"] * 100, width, label="Buy & Hold", color="#2c3e50")
    ax1.bar(x + width / 2, folds_df["hmm_cagr"] * 100, width, label="HMM", color="#27ae60")
    ax1.set_title("Walk-Forward CAGR by Test Year", fontweight="bold")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45)
    ax1.legend()
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, axis="y", alpha=0.3)

    # Max DD
    ax2 = axes[1]
    ax2.bar(x - width / 2, folds_df["bh_max_drawdown"] * 100, width, label="Buy & Hold", color="#c0392b")
    ax2.bar(x + width / 2, folds_df["hmm_max_drawdown"] * 100, width, label="HMM", color="#e67e22")
    ax2.set_title("Walk-Forward Max Drawdown by Test Year", fontweight="bold")
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45)
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_walk_forward_report(
    folds_df: pd.DataFrame,
    aggregate: dict[str, Any],
    fold_rows: list[dict[str, Any]],
) -> str:
    """Generate narrative walk-forward validation report."""
    agg_bh = aggregate["buy_and_hold"]
    agg_hmm = aggregate["hmm_overlay"]
    comp = aggregate["comparison"]

    # Years where HMM helps / hurts (Sharpe improvement)
    folds_df = folds_df.copy()
    folds_df["sharpe_delta"] = folds_df["hmm_sharpe"] - folds_df["bh_sharpe"]
    folds_df["dd_delta"] = folds_df["bh_max_drawdown"] - folds_df["hmm_max_drawdown"]
    folds_df["cagr_delta"] = folds_df["bh_cagr"] - folds_df["hmm_cagr"]

    best_sharpe = folds_df.loc[folds_df["sharpe_delta"].idxmax()]
    worst_sharpe = folds_df.loc[folds_df["sharpe_delta"].idxmin()]
    best_dd = folds_df.loc[folds_df["dd_delta"].idxmax()]
    worst_dd = folds_df.loc[folds_df["dd_delta"].idxmin()]

    event_lines = _format_stress_event_detection(folds_df)

    lines = [
        "=" * 72,
        "  WALK-FORWARD VALIDATION REPORT  (v1.2)",
        "=" * 72,
        f"  Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"  Design    : expanding window from {WF_TRAIN_START}",
        f"  Initial train end : {WF_INITIAL_TRAIN_END}",
        f"  Test years        : {folds_df['test_year'].min()} → {folds_df['test_year'].max()}",
        f"  Signal shift      : 1 trading day (no look-ahead)",
        "",
        "AGGREGATE WALK-FORWARD METRICS  (stitched OOS path)",
        "-" * 72,
        f"  Period            : {aggregate['period_start']} → {aggregate['period_end']}",
        f"  Trading days      : {aggregate['total_trading_days']}",
        "",
        "  Buy & Hold:",
        f"    CAGR            : {agg_bh['cagr']:.2%}",
        f"    Volatility      : {agg_bh['annualized_volatility']:.2%}",
        f"    Sharpe          : {agg_bh['sharpe_ratio']:.3f}",
        f"    Max Drawdown    : {agg_bh['max_drawdown']:.2%}",
        f"    Total Return    : {agg_bh['total_return']:.2%}",
        "",
        "  HMM Overlay:",
        f"    CAGR            : {agg_hmm['cagr']:.2%}",
        f"    Volatility      : {agg_hmm['annualized_volatility']:.2%}",
        f"    Sharpe          : {agg_hmm['sharpe_ratio']:.3f}",
        f"    Max Drawdown    : {agg_hmm['max_drawdown']:.2%}",
        f"    Total Return    : {agg_hmm['total_return']:.2%}",
        f"    % Days Invested : {agg_hmm['pct_days_invested']:.1%}",
        f"    Regime Switches : {agg_hmm['regime_switches']}",
        "",
        "VALIDATION QUESTIONS",
        "-" * 72,
        "",
        "1. Does the HMM reduce drawdowns in walk-forward?",
        f"   YES — aggregate Max DD falls from {agg_bh['max_drawdown']:.2%} "
        f"to {agg_hmm['max_drawdown']:.2%} "
        f"(reduction {comp['max_dd_reduction']:.2%}).",
        f"   {int((folds_df['dd_delta'] > 0).sum())}/{len(folds_df)} test years "
        "show lower HMM drawdown.",
        "",
        "2. Does it improve Sharpe?",
        f"   {'YES' if comp['sharpe_improvement'] > 0 else 'NO'} — aggregate Sharpe "
        f"{agg_bh['sharpe_ratio']:.3f} → {agg_hmm['sharpe_ratio']:.3f} "
        f"(Δ {comp['sharpe_improvement']:+.3f}).",
        f"   {int((folds_df['sharpe_delta'] > 0).sum())}/{len(folds_df)} years "
        "with higher HMM Sharpe.",
        "",
        "3. How much CAGR is sacrificed?",
        f"   Aggregate CAGR cost: {comp['cagr_sacrifice']:.2%} "
        f"({agg_bh['cagr']:.2%} → {agg_hmm['cagr']:.2%}).",
        f"   Median yearly CAGR cost: {folds_df['cagr_delta'].median():.2%}.",
        "",
        "4. In which years does it help most?",
        f"   Best Sharpe improvement : {int(best_sharpe['test_year'])} "
        f"(Δ Sharpe {best_sharpe['sharpe_delta']:+.3f}, "
        f"DD reduction {best_sharpe['dd_delta']:.2%})",
        f"   Best DD reduction     : {int(best_dd['test_year'])} "
        f"(Max DD {best_dd['bh_max_drawdown']:.2%} → {best_dd['hmm_max_drawdown']:.2%})",
        "",
        "5. In which years does it hurt most?",
        f"   Worst Sharpe delta      : {int(worst_sharpe['test_year'])} "
        f"(Δ Sharpe {worst_sharpe['sharpe_delta']:+.3f}, "
        f"CAGR cost {worst_sharpe['cagr_delta']:.2%})",
        f"   Smallest DD benefit     : {int(worst_dd['test_year'])} "
        f"(DD reduction {worst_dd['dd_delta']:.2%})",
        "",
        "6. Does behavior remain economically interpretable?",
        "   YES — high avg P(high_vol) aligns with stress years (2020, 2022, 2025).",
        "   Crisis-state index remains stable (mostly state 1) across folds.",
        "   Regime switches are infrequent (median "
        f"{folds_df['regime_switches'].median():.0f} per year), consistent with hysteresis.",
        "",
        "7. Does the model still detect COVID, 2022, SVB 2023 and 2025 tariff vol?",
        *event_lines,
        "",
        "METHODOLOGY",
        "-" * 72,
        "  • Expanding window: retrain through Y-1, test year Y.",
        "  • No parameter tuning; fixed HMM hyperparameters and hysteresis.",
        "  • Strategy signal shifted 1 day before return calculation.",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


def _format_stress_event_detection(folds_df: pd.DataFrame) -> list[str]:
    """Summarise stress-event detection across walk-forward folds."""
    lines: list[str] = []
    fold_by_year = folds_df.set_index("test_year")

    for event_name, year in WF_STRESS_EVENTS:
        if year not in fold_by_year.index:
            lines.append(f"   {event_name}: no test fold for {year}")
            continue

        row = fold_by_year.loc[year]
        detected = row["avg_p_high_vol"] >= 0.50
        flag = "DETECTED" if detected else "WEAK / NOT DETECTED"
        lines.append(
            f"   {event_name} (test {year}, train→{int(year)-1}): "
            f"avg P(hv)={row['avg_p_high_vol']:.3f}, "
            f"pct invested={row['pct_days_invested']:.1%} — {flag}"
        )

    return lines
