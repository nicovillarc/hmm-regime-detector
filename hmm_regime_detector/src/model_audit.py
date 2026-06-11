"""Internal model audit — interpretability report for the fitted Gaussian HMM."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MODEL_AUDIT_JSON,
    MODEL_AUDIT_TXT,
    REGIME_HIGH_VOL,
    REGIME_NORMAL,
    REPORTS_DIR,
)
from src.model import HMMRegimeDetector

logger = logging.getLogger(__name__)

# Human-readable feature labels for reports.
FEATURE_LABELS = {
    "spy_log_return": "SPY log return",
    "spy_realized_vol": "RV20",
    "log_vix": "log(VIX)",
}

REGIME_LABELS = {
    REGIME_NORMAL: "Normal",
    REGIME_HIGH_VOL: "High Volatility",
}


def build_model_audit(
    detector: HMMRegimeDetector,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compile a full interpretability audit for a fitted HMM.

    Crisis / high-volatility state identification follows the same rule as
    training: highest combined average of RV20 + log(VIX) on original features
    under Viterbi-decoded state assignments.
    """
    if detector.model is None or detector.crisis_state is None:
        raise RuntimeError("Detector must be fitted before running audit.")

    X_scaled = detector.scaler.transform(features[detector.feature_columns].values)
    decoded_states = detector.model.predict(X_scaled)

    normal_idx = detector.normal_state
    crisis_idx = detector.crisis_state

    state_means_original = _inverse_transform_means(detector)
    state_covariances_scaled = _extract_scaled_covariances(detector)
    state_covariances_original = _transform_covariances_to_original(
        detector, state_covariances_scaled
    )

    transmat = detector.model.transmat_
    labeled_transitions = _label_transition_matrix(transmat, normal_idx, crisis_idx)
    expected_durations = _expected_durations(labeled_transitions)
    assignment_stats = _regime_assignment_stats(decoded_states, normal_idx, crisis_idx)
    regime_stats = _regime_feature_statistics(features, decoded_states, normal_idx, crisis_idx)

    audit: dict[str, Any] = {
        "version": "1.0.1",
        "report_type": "model_interpretability_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_period": {
            "start": str(features.index[0].date()),
            "end": str(features.index[-1].date()),
            "n_observations": int(len(features)),
        },
        "state_mapping": {
            "normal_state_index": int(normal_idx),
            "high_volatility_state_index": int(crisis_idx),
            "crisis_identification_rule": (
                "State with highest combined average of RV20 + log(VIX) "
                "on original (non-scaled) features under Viterbi decoding."
            ),
        },
        "state_means_original": {
            REGIME_NORMAL: _means_dict(state_means_original[normal_idx]),
            REGIME_HIGH_VOL: _means_dict(state_means_original[crisis_idx]),
        },
        "state_covariances_scaled": {
            REGIME_NORMAL: state_covariances_scaled[normal_idx].tolist(),
            REGIME_HIGH_VOL: state_covariances_scaled[crisis_idx].tolist(),
        },
        "state_covariances_original": {
            REGIME_NORMAL: state_covariances_original[normal_idx].tolist(),
            REGIME_HIGH_VOL: state_covariances_original[crisis_idx].tolist(),
        },
        "transition_matrix": labeled_transitions,
        "transition_matrix_raw": {
            "states_order": [int(normal_idx), int(crisis_idx)],
            "matrix": transmat.tolist(),
        },
        "expected_duration_days": expected_durations,
        "regime_assignments": assignment_stats,
        "regime_statistics": regime_stats,
    }
    return audit


def save_model_audit(
    audit: dict[str, Any],
    json_path: Path = MODEL_AUDIT_JSON,
    txt_path: Path = MODEL_AUDIT_TXT,
) -> tuple[Path, Path]:
    """Persist audit results as JSON and human-readable text."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as fh:
        json.dump(audit, fh, indent=2)

    text = format_model_audit_text(audit)
    with open(txt_path, "w") as fh:
        fh.write(text)

    logger.info("Model audit saved → %s, %s", json_path, txt_path)
    return json_path, txt_path


def print_model_audit(audit: dict[str, Any]) -> None:
    """Print the audit report to stdout."""
    print(format_model_audit_text(audit))


def format_model_audit_text(audit: dict[str, Any]) -> str:
    """Render the audit as a formatted plain-text report."""
    lines: list[str] = []
    sep = "=" * 62
    subsep = "-" * 62

    lines.append(sep)
    lines.append("  HMM MODEL AUDIT — INTERPRETABILITY REPORT  (v1.0.1)")
    lines.append(sep)
    lines.append(f"  Generated : {audit['generated_at']}")
    period = audit["sample_period"]
    lines.append(
        f"  Sample    : {period['start']} → {period['end']}  "
        f"({period['n_observations']:,} days)"
    )
    mapping = audit["state_mapping"]
    lines.append(
        f"  States    : normal={mapping['normal_state_index']}, "
        f"high_vol={mapping['high_volatility_state_index']}"
    )
    lines.append("")

    # --- 1. State means (original scale) ---
    lines.append("1. STATE MEANS  (original, non-scaled features)")
    lines.append(subsep)
    for regime in (REGIME_NORMAL, REGIME_HIGH_VOL):
        lines.append(f"  [{REGIME_LABELS[regime]}]")
        means = audit["state_means_original"][regime]
        for col in FEATURE_COLUMNS:
            lines.append(f"    {FEATURE_LABELS[col]:18s}: {means[col]:+.6f}")
        lines.append("")

    # --- 2. State covariances ---
    lines.append("2. STATE COVARIANCES  (from HMM, scaled feature space)")
    lines.append(subsep)
    for regime in (REGIME_NORMAL, REGIME_HIGH_VOL):
        lines.append(f"  [{REGIME_LABELS[regime]}]")
        cov = np.array(audit["state_covariances_scaled"][regime])
        header = "    " + "".join(f"{FEATURE_LABELS[c]:>14s}" for c in FEATURE_COLUMNS)
        lines.append(header)
        for i, row_label in enumerate(FEATURE_COLUMNS):
            row = "    " + f"{FEATURE_LABELS[row_label]:>14s}"
            row += "".join(f"{cov[i, j]:14.6f}" for j in range(len(FEATURE_COLUMNS)))
            lines.append(row)
        lines.append("")

    lines.append("   (original-scale covariances saved in model_audit.json)")
    lines.append("")

    # --- 3. Transition matrix ---
    lines.append("3. TRANSITION MATRIX  (regime labels)")
    lines.append(subsep)
    tm = audit["transition_matrix"]
    lines.append(f"    Normal       → Normal       : {tm['normal_to_normal']:.4f}")
    lines.append(f"    Normal       → High Vol     : {tm['normal_to_high_vol']:.4f}")
    lines.append(f"    High Vol     → Normal       : {tm['high_vol_to_normal']:.4f}")
    lines.append(f"    High Vol     → High Vol     : {tm['high_vol_to_high_vol']:.4f}")
    lines.append("")

    # --- 4. Expected durations ---
    lines.append("4. AVERAGE EXPECTED REGIME DURATION  (days)")
    lines.append(subsep)
    dur = audit["expected_duration_days"]
    lines.append(f"    Normal         : {dur[REGIME_NORMAL]:.2f} days")
    lines.append(f"    High Volatility: {dur[REGIME_HIGH_VOL]:.2f} days")
    lines.append("    Formula: 1 / (1 − P(state → same state))")
    lines.append("")

    # --- 5–6. Regime assignments ---
    lines.append("5–6. REGIME ASSIGNMENTS  (Viterbi-decoded)")
    lines.append(subsep)
    assign = audit["regime_assignments"]
    for regime in (REGIME_NORMAL, REGIME_HIGH_VOL):
        stats = assign[regime]
        lines.append(
            f"    {REGIME_LABELS[regime]:16s}: "
            f"{stats['count']:5,d} days  ({stats['pct']:.2%})"
        )
    lines.append("")

    # --- 7–9. Regime statistics ---
    lines.append("7–9. AVERAGE FEATURE VALUES BY INFERRED REGIME")
    lines.append(subsep)
    rs = audit["regime_statistics"]
    for regime in (REGIME_NORMAL, REGIME_HIGH_VOL):
        s = rs[regime]
        lines.append(f"  [{REGIME_LABELS[regime]}]")
        lines.append(f"    Avg SPY log return : {s['avg_spy_log_return']:+.6f}")
        lines.append(f"    Avg RV20           : {s['avg_rv20']:.6f}")
        lines.append(f"    Avg log(VIX)       : {s['avg_log_vix']:.6f}")
        lines.append(f"    Avg VIX            : {s['avg_vix']:.2f}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inverse_transform_means(detector: HMMRegimeDetector) -> np.ndarray:
    """Map HMM emission means from scaled space back to original features."""
    scale = detector.scaler.scale_
    mean = detector.scaler.mean_
    return detector.model.means_ * scale + mean


def _extract_scaled_covariances(detector: HMMRegimeDetector) -> np.ndarray:
    """Return per-state covariance matrices as learned by the HMM."""
    covars = detector.model.covars_
    if detector.model.covariance_type == "full":
        return covars.copy()
    raise ValueError(
        f"Unsupported covariance_type: {detector.model.covariance_type}"
    )


def _transform_covariances_to_original(
    detector: HMMRegimeDetector,
    covars_scaled: np.ndarray,
) -> np.ndarray:
    """Transform covariance matrices from scaled to original feature space."""
    scale_matrix = np.diag(detector.scaler.scale_)
    return scale_matrix @ covars_scaled @ scale_matrix


def _means_dict(mean_vector: np.ndarray) -> dict[str, float]:
    """Zip feature columns with their mean values."""
    return {
        col: round(float(val), 8)
        for col, val in zip(FEATURE_COLUMNS, mean_vector)
    }


def _label_transition_matrix(
    transmat: np.ndarray,
    normal_idx: int,
    crisis_idx: int,
) -> dict[str, float]:
    """Extract the four labelled transition probabilities."""
    return {
        "normal_to_normal": round(float(transmat[normal_idx, normal_idx]), 6),
        "normal_to_high_vol": round(float(transmat[normal_idx, crisis_idx]), 6),
        "high_vol_to_normal": round(float(transmat[crisis_idx, normal_idx]), 6),
        "high_vol_to_high_vol": round(float(transmat[crisis_idx, crisis_idx]), 6),
    }


def _expected_durations(labeled_transitions: dict[str, float]) -> dict[str, float]:
    """Compute average regime duration from self-transition probabilities."""
    p_nn = labeled_transitions["normal_to_normal"]
    p_hh = labeled_transitions["high_vol_to_high_vol"]
    return {
        REGIME_NORMAL: round(_duration_from_self_prob(p_nn), 2),
        REGIME_HIGH_VOL: round(_duration_from_self_prob(p_hh), 2),
    }


def _duration_from_self_prob(p_stay: float) -> float:
    """Expected duration = 1 / (1 − P(stay))."""
    if p_stay >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - p_stay)


def _regime_assignment_stats(
    decoded_states: np.ndarray,
    normal_idx: int,
    crisis_idx: int,
) -> dict[str, dict[str, float | int]]:
    """Count and percentage of Viterbi-assigned days per regime."""
    n = len(decoded_states)
    counts = {
        REGIME_NORMAL: int((decoded_states == normal_idx).sum()),
        REGIME_HIGH_VOL: int((decoded_states == crisis_idx).sum()),
    }
    return {
        regime: {
            "count": count,
            "pct": round(count / n, 6) if n > 0 else 0.0,
        }
        for regime, count in counts.items()
    }


def _regime_feature_statistics(
    features: pd.DataFrame,
    decoded_states: np.ndarray,
    normal_idx: int,
    crisis_idx: int,
) -> dict[str, dict[str, float]]:
    """Empirical averages of key market variables by Viterbi-inferred regime."""
    regime_index = pd.Series(decoded_states, index=features.index)
    vix = np.exp(features["log_vix"])

    stats: dict[str, dict[str, float]] = {}
    for regime, state_idx in (
        (REGIME_NORMAL, normal_idx),
        (REGIME_HIGH_VOL, crisis_idx),
    ):
        mask = regime_index == state_idx
        stats[regime] = {
            "avg_spy_log_return": round(float(features.loc[mask, "spy_log_return"].mean()), 8),
            "avg_rv20": round(float(features.loc[mask, "spy_realized_vol"].mean()), 8),
            "avg_log_vix": round(float(features.loc[mask, "log_vix"].mean()), 8),
            "avg_vix": round(float(vix.loc[mask].mean()), 4),
        }
    return stats


def run_model_audit(
    detector: HMMRegimeDetector,
    features: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """
    Build, save, and print the model audit report.

    Convenience wrapper used by run_backtest.py.
    """
    audit = build_model_audit(detector, features)
    json_path = output_dir / MODEL_AUDIT_JSON.name
    txt_path = output_dir / MODEL_AUDIT_TXT.name
    save_model_audit(audit, json_path=json_path, txt_path=txt_path)
    print_model_audit(audit)
    return audit
