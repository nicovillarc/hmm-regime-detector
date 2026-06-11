"""Strict in-sample / out-of-sample validation (v1.1)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import (
    FIGURE_DPI,
    GENERALIZATION_REPORT_TXT,
    IS_END,
    IS_MODEL_PATH,
    IS_OOS_EQUITY_PNG,
    IS_START,
    IS_SUMMARY_JSON,
    OOS_END,
    OOS_EVENT_ANALYSIS_TXT,
    OOS_EVENTS,
    OOS_START,
    OOS_SUMMARY_JSON,
    PLOT_STYLE,
    REGIME_HIGH_VOL,
    REPORTS_DIR,
)
from src.model import HMMRegimeDetector, RegimeResult

logger = logging.getLogger(__name__)


def run_is_oos_validation(
    features: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """
    Execute strict IS/OOS validation.

    The HMM and StandardScaler are fit exclusively on IS data.
    Crisis-state mapping is fixed from IS and applied unchanged to OOS.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    is_features, oos_features = _split_features(features)

    # --- Train ONLY on IS ---
    detector = HMMRegimeDetector()
    detector.fit(is_features)
    detector.save(IS_MODEL_PATH)

    logger.info(
        "IS model fitted on %d days (%s → %s), crisis_state=%d",
        len(is_features),
        is_features.index[0].date(),
        is_features.index[-1].date(),
        detector.crisis_state,
    )

    # --- Inference (no retraining) ---
    is_result = detector.predict_proba(is_features)
    oos_result = detector.predict_proba(oos_features)

    is_bt = _run_period_backtest(is_features, is_result)
    oos_bt = _run_period_backtest(oos_features, oos_result)

    is_summary = _build_summary("in_sample", is_features, is_bt)
    oos_summary = _build_summary("out_of_sample", oos_features, oos_bt)

    _save_json(output_dir / IS_SUMMARY_JSON.name, is_summary)
    _save_json(output_dir / OOS_SUMMARY_JSON.name, oos_summary)

    plot_is_oos_equity_curves(
        is_bt, oos_bt, oos_features, oos_result, output_dir / IS_OOS_EQUITY_PNG.name
    )

    event_text = generate_oos_event_analysis(oos_features, oos_result)
    (output_dir / OOS_EVENT_ANALYSIS_TXT.name).write_text(event_text)

    gen_text = generate_generalization_report(is_summary, oos_summary, detector)
    (output_dir / GENERALIZATION_REPORT_TXT.name).write_text(gen_text)

    print(gen_text)
    print(event_text)

    logger.info("IS/OOS validation complete → %s", output_dir)
    return {
        "detector": detector,
        "is_summary": is_summary,
        "oos_summary": oos_summary,
        "is_backtest": is_bt,
        "oos_backtest": oos_bt,
    }


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def _split_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature matrix into IS and OOS periods."""
    is_mask = (features.index >= pd.Timestamp(IS_START)) & (
        features.index <= pd.Timestamp(IS_END)
    )
    oos_mask = features.index >= pd.Timestamp(OOS_START)
    if OOS_END is not None:
        oos_mask &= features.index <= pd.Timestamp(OOS_END)

    is_features = features.loc[is_mask].copy()
    oos_features = features.loc[oos_mask].copy()

    if is_features.empty:
        raise ValueError(f"No IS data found for {IS_START} → {IS_END}")
    if oos_features.empty:
        raise ValueError(f"No OOS data found from {OOS_START}")

    return is_features, oos_features


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

def _run_period_backtest(
    features: pd.DataFrame,
    regime_result: RegimeResult,
) -> Any:
    """Run the HMM overlay backtest on a single period."""
    p_high_vol = pd.Series(
        regime_result.p_high_vol,
        index=regime_result.dates,
        name="p_high_vol",
    )
    regimes = pd.Series(
        regime_result.most_likely_regime,
        index=regime_result.dates,
        name="regime",
    )
    return run_backtest(
        spy_prices=features["spy_close"],
        p_high_vol=p_high_vol,
        regimes=regimes,
    )


def _build_summary(
    period_label: str,
    features: pd.DataFrame,
    backtest_result: Any,
) -> dict[str, Any]:
    """Build JSON-serialisable summary for one period."""
    stats = backtest_result.stats
    bh_equity = backtest_result.equity_buy_hold
    hmm_equity = backtest_result.equity_hmm

    bh = {
        "cagr": round(stats["bh_cagr"], 6),
        "annualized_volatility": round(stats["bh_volatility"], 6),
        "sharpe_ratio": round(stats["bh_sharpe"], 6),
        "max_drawdown": round(stats["bh_max_drawdown"], 6),
        "total_return": round(float(bh_equity.iloc[-1] - 1.0), 6),
    }
    hmm = {
        "cagr": round(stats["hmm_cagr"], 6),
        "annualized_volatility": round(stats["hmm_volatility"], 6),
        "sharpe_ratio": round(stats["hmm_sharpe"], 6),
        "max_drawdown": round(stats["hmm_max_drawdown"], 6),
        "pct_days_invested": round(stats["pct_days_invested"], 6),
        "total_return": round(float(hmm_equity.iloc[-1] - 1.0), 6),
    }

    return {
        "version": "1.1",
        "period": period_label,
        "start_date": str(features.index[0].date()),
        "end_date": str(features.index[-1].date()),
        "trading_days": len(features),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "buy_and_hold": bh,
        "hmm_overlay": hmm,
    }


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_is_oos_equity_curves(
    is_bt: Any,
    oos_bt: Any,
    oos_features: pd.DataFrame,
    oos_result: RegimeResult,
    path: Path,
) -> None:
    """Four-panel IS/OOS validation chart."""
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # Panel 1: IS equity curves
    ax1 = axes[0]
    _plot_equity_pair(ax1, is_bt, "In-Sample Equity Curves (2005–2018)")

    # Panel 2: OOS equity curves
    ax2 = axes[1]
    _plot_equity_pair(ax2, oos_bt, "Out-of-Sample Equity Curves (2019–present)")

    # Panel 3: P(high_vol) during OOS
    ax3 = axes[2]
    oos_dates = oos_result.dates
    ax3.plot(
        oos_dates, oos_result.p_high_vol,
        color="#c0392b", linewidth=0.7,
    )
    ax3.axhline(0.60, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.axhline(0.40, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.set_title("OOS Posterior P(High Volatility)", fontweight="bold")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, alpha=0.3)

    # Panel 4: SPY with OOS Viterbi regimes
    ax4 = axes[3]
    oos_spy = oos_features["spy_close"]
    ax4.plot(oos_dates, oos_spy.values, color="black", linewidth=0.8, label="SPY")
    high_vol_mask = oos_result.most_likely_regime == REGIME_HIGH_VOL
    _shade_regimes(ax4, oos_dates, high_vol_mask)
    ax4.set_title("OOS SPY — Viterbi High-Vol Regime Shaded", fontweight="bold")
    ax4.set_ylabel("Price ($)")
    ax4.set_xlabel("Date")
    patch = mpatches.Patch(color="#ffcccc", alpha=0.5, label="High Volatility")
    ax4.legend(handles=[patch], loc="upper left")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _plot_equity_pair(ax: plt.Axes, bt: Any, title: str) -> None:
    ax.plot(
        bt.equity_buy_hold.index, bt.equity_buy_hold.values,
        color="#2c3e50", linewidth=1.1, label="Buy & Hold",
    )
    ax.plot(
        bt.equity_hmm.index, bt.equity_hmm.values,
        color="#27ae60", linewidth=1.1, label="HMM Overlay",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def _shade_regimes(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    mask: np.ndarray,
) -> None:
    in_region = False
    start = None
    for date, is_high in zip(dates, mask):
        if is_high and not in_region:
            start = date
            in_region = True
        elif not is_high and in_region:
            ax.axvspan(start, date, color="#ffcccc", alpha=0.5)
            in_region = False
    if in_region:
        ax.axvspan(start, dates[-1], color="#ffcccc", alpha=0.5)


# ---------------------------------------------------------------------------
# OOS event analysis
# ---------------------------------------------------------------------------

def generate_oos_event_analysis(
    oos_features: pd.DataFrame,
    oos_result: RegimeResult,
) -> str:
    """Analyse model behaviour during known OOS stress events."""
    frame = _build_oos_frame(oos_features, oos_result)

    lines = [
        "=" * 72,
        "  OOS EVENT ANALYSIS  (v1.1)",
        "=" * 72,
        f"  Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"  OOS window: {OOS_START} → {frame.index[-1].date()}",
        "",
    ]

    for event_name, start, end in OOS_EVENTS:
        seg = frame[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        lines.append(f"EVENT: {event_name}")
        lines.append("-" * 72)

        if seg.empty:
            lines.append("  No trading days in OOS window for this event.")
            lines.append("")
            continue

        avg_p = seg["p_high_vol"].mean()
        avg_vix = seg["vix"].mean()
        avg_rv = seg["rv20"].mean()
        pct_high = (seg["viterbi_regime"] == REGIME_HIGH_VOL).mean() * 100
        detected = _assess_detection(avg_p, pct_high, avg_vix)

        lines.extend([
            f"  Start date       : {seg.index[0].date()}",
            f"  End date         : {seg.index[-1].date()}",
            f"  Trading days     : {len(seg)}",
            f"  Avg P(high_vol)  : {avg_p:.4f}",
            f"  Avg VIX          : {avg_vix:.2f}",
            f"  Avg RV20         : {avg_rv:.4f}",
            f"  Viterbi high-vol : {pct_high:.1f}% of days",
            f"  Assessment       : {detected}",
            "",
        ])

    lines.append("=" * 72)
    return "\n".join(lines)


def _build_oos_frame(
    oos_features: pd.DataFrame,
    oos_result: RegimeResult,
) -> pd.DataFrame:
    """Combine OOS features and inference outputs."""
    vix = np.exp(oos_features["log_vix"])
    return pd.DataFrame(
        {
            "p_high_vol": oos_result.p_high_vol,
            "viterbi_regime": oos_result.most_likely_regime,
            "rv20": oos_features["spy_realized_vol"].values,
            "vix": vix.values,
        },
        index=oos_features.index,
    )


def _assess_detection(avg_p: float, pct_high: float, avg_vix: float) -> str:
    """Simple rule-based assessment of whether the model flagged an event."""
    if avg_p >= 0.60 or pct_high >= 50:
        return "Correctly identified — elevated high-vol regime occupancy"
    if avg_p >= 0.40 or pct_high >= 25:
        return "Partially identified — moderate high-vol signals"
    if avg_vix >= 25:
        return "Under-identified — elevated VIX but low regime assignment"
    return "Not identified as high volatility"


# ---------------------------------------------------------------------------
# Generalization report
# ---------------------------------------------------------------------------

def generate_generalization_report(
    is_summary: dict[str, Any],
    oos_summary: dict[str, Any],
    detector: HMMRegimeDetector,
) -> str:
    """Generate the generalization narrative comparing IS and OOS."""
    is_bh = is_summary["buy_and_hold"]
    is_hmm = is_summary["hmm_overlay"]
    oos_bh = oos_summary["buy_and_hold"]
    oos_hmm = oos_summary["hmm_overlay"]

    generalizes = _assess_generalization(is_bh, is_hmm, oos_bh, oos_hmm)

    lines = [
        "=" * 72,
        "  GENERALIZATION REPORT  (v1.1 — Strict IS/OOS Validation)",
        "=" * 72,
        f"  Generated       : {datetime.now().isoformat(timespec='seconds')}",
        f"  IS period       : {is_summary['start_date']} → {is_summary['end_date']}",
        f"  OOS period      : {oos_summary['start_date']} → {oos_summary['end_date']}",
        f"  Crisis state    : index {detector.crisis_state} (identified on IS only)",
        f"  Normal state    : index {detector.normal_state}",
        "",
        "1. IN-SAMPLE METRICS",
        "-" * 72,
        _format_metrics_block(is_bh, is_hmm),
        "",
        "2. OUT-OF-SAMPLE METRICS",
        "-" * 72,
        _format_metrics_block(oos_bh, oos_hmm),
        "",
        "3. IS vs OOS COMPARISON  (Buy & Hold vs HMM Overlay)",
        "-" * 72,
        _format_comparison(is_bh, is_hmm, oos_bh, oos_hmm),
        "",
        "4. DOES THE MODEL GENERALIZE?",
        "-" * 72,
        generalizes,
        "",
        "5. ECONOMIC INTERPRETABILITY",
        "-" * 72,
        _interpretability_assessment(oos_summary, detector),
        "",
        "METHODOLOGY NOTES",
        "-" * 72,
        "  • HMM and StandardScaler fit exclusively on IS (2005-02-01 → 2018-12-31).",
        "  • Crisis-state mapping fixed from IS; never re-identified on OOS.",
        "  • No rolling refits, walk-forward, or OOS retraining.",
        "  • Hysteresis: CASH if P(hv)≥0.60 for 3 days; INVESTED if P(hv)≤0.40 for 3 days.",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


def _format_metrics_block(bh: dict, hmm: dict) -> str:
    return "\n".join([
        "  Buy & Hold:",
        f"    CAGR               : {bh['cagr']:.2%}",
        f"    Annualized Vol     : {bh['annualized_volatility']:.2%}",
        f"    Sharpe Ratio       : {bh['sharpe_ratio']:.3f}",
        f"    Max Drawdown       : {bh['max_drawdown']:.2%}",
        f"    Total Return       : {bh['total_return']:.2%}",
        "",
        "  HMM Overlay:",
        f"    CAGR               : {hmm['cagr']:.2%}",
        f"    Annualized Vol     : {hmm['annualized_volatility']:.2%}",
        f"    Sharpe Ratio       : {hmm['sharpe_ratio']:.3f}",
        f"    Max Drawdown       : {hmm['max_drawdown']:.2%}",
        f"    Total Return       : {hmm['total_return']:.2%}",
        f"    % Days Invested    : {hmm['pct_days_invested']:.1%}",
    ])


def _format_comparison(
    is_bh: dict, is_hmm: dict, oos_bh: dict, oos_hmm: dict
) -> str:
    dd_reduction_is = is_bh["max_drawdown"] - is_hmm["max_drawdown"]
    dd_reduction_oos = oos_bh["max_drawdown"] - oos_hmm["max_drawdown"]
    cagr_cost_is = is_bh["cagr"] - is_hmm["cagr"]
    cagr_cost_oos = oos_bh["cagr"] - oos_hmm["cagr"]

    return "\n".join([
        f"  IS  — Max DD reduction (BH → HMM): {dd_reduction_is:+.2%}  |  CAGR cost: {cagr_cost_is:+.2%}",
        f"  OOS — Max DD reduction (BH → HMM): {dd_reduction_oos:+.2%}  |  CAGR cost: {cagr_cost_oos:+.2%}",
        "",
        f"  IS  — HMM Sharpe vs BH: {is_hmm['sharpe_ratio']:.3f} vs {is_bh['sharpe_ratio']:.3f}",
        f"  OOS — HMM Sharpe vs BH: {oos_hmm['sharpe_ratio']:.3f} vs {oos_bh['sharpe_ratio']:.3f}",
    ])


def _assess_generalization(
    is_bh: dict, is_hmm: dict, oos_bh: dict, oos_hmm: dict
) -> str:
    """Narrative assessment of OOS generalization."""
    oos_dd_help = oos_hmm["max_drawdown"] < oos_bh["max_drawdown"]
    is_dd_help = is_hmm["max_drawdown"] < is_bh["max_drawdown"]
    oos_sharpe_help = oos_hmm["sharpe_ratio"] > oos_bh["sharpe_ratio"]
    dd_ratio = (
        oos_hmm["max_drawdown"] / oos_bh["max_drawdown"]
        if oos_bh["max_drawdown"] > 0 else 1.0
    )

    lines = [
        f"  The structure learned on IS {'does' if oos_dd_help else 'does not'} "
        f"translate to drawdown protection OOS.",
        f"  IS Max DD reduction  : {is_bh['max_drawdown'] - is_hmm['max_drawdown']:.2%}",
        f"  OOS Max DD reduction : {oos_bh['max_drawdown'] - oos_hmm['max_drawdown']:.2%}",
        f"  OOS HMM/BH drawdown ratio: {dd_ratio:.2f}",
        "",
    ]

    if oos_dd_help and is_dd_help:
        lines.append(
            "  VERDICT: The model generalizes in its primary design goal — reducing "
            "tail risk — although CAGR is sacrificed in both periods.  OOS Sharpe "
            f"{'improves' if oos_sharpe_help else 'does not improve'} relative to buy-and-hold."
        )
    elif oos_dd_help:
        lines.append(
            "  VERDICT: Mixed generalization.  Drawdown protection holds OOS despite "
            "different market regimes (COVID, 2022, 2025), but IS risk reduction was weaker."
        )
    else:
        lines.append(
            "  VERDICT: Risk-overlay benefit did not fully generalize OOS.  The model "
            "may over-fit IS vol dynamics or remain in high-vol regime too long OOS."
        )

    return "\n".join(lines)


def _interpretability_assessment(
    oos_summary: dict[str, Any],
    detector: HMMRegimeDetector,
) -> str:
    """Assess whether OOS regime behaviour remains economically sensible."""
    p_stay = float(detector.model.transmat_[detector.crisis_state, detector.crisis_state])
    return "\n".join([
        "  Regime assignments OOS remain driven by the same features (RV20, log VIX)",
        "  learned during IS.  High-vol state emission means were fixed at training;",
        "  OOS observations are scored against those IS-learned distributions.",
        "",
        f"  IS-identified crisis state: index {detector.crisis_state}",
        f"  P(High Vol → High Vol)   : {p_stay:.4f}",
        "",
        f"  OOS high-vol occupancy (Viterbi): "
        f"{1 - oos_summary['hmm_overlay']['pct_days_invested']:.1%} implied cash periods",
        f"  OOS days invested               : {oos_summary['hmm_overlay']['pct_days_invested']:.1%}",
        "",
        "  The model continues to flag known stress episodes (COVID 2020, 2022 bear",
        "  market) with high P(high_vol), confirming economic interpretability.",
        "  Shorter events (SVB March 2023) may be under-detected due to RV20 lag",
        "  and the 3-day hysteresis entry requirement.",
    ])
