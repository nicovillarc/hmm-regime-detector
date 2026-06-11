"""COVID episode forensics using the IS-trained model (v1.1 extension)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    IS_END,
    IS_MODEL_PATH,
    IS_START,
    MODEL_PATH,
    OOS_START,
    REGIME_HIGH_VOL,
    REPORTS_DIR,
)
from src.model import HMMRegimeDetector
from src.regime_diagnostics import detect_high_vol_episodes

logger = logging.getLogger(__name__)

COVID_OOS_FORENSICS_TXT = REPORTS_DIR / "covid_oos_forensics.txt"

# Reference episode from full-sample v1.0 diagnostics.
FULL_SAMPLE_COVID = {
    "model": "full-sample v1.0 (trained 2005 → present)",
    "start_date": "2020-02-24",
    "end_date": "2021-03-24",
    "trading_days": 274,
    "avg_vix": 29.97,
    "avg_rv20": 0.2594,
    "avg_p_high_vol": 0.958,
}

FEB_2020_START = "2020-02-01"
FEB_2020_END = "2020-02-29"


def run_covid_oos_forensics(
    features: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """
    Identify the IS-model high-vol episode containing February 2020.

    Uses the IS-trained HMM (2005-2018) with Viterbi decoding on OOS data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    is_detector = _load_is_detector(features)
    oos_features = features[features.index >= pd.Timestamp(OOS_START)].copy()

    oos_result = is_detector.predict_proba(oos_features)
    viterbi = _build_oos_frame(oos_features, oos_result, is_detector)

    episodes = detect_high_vol_episodes(viterbi)
    covid_episode = find_episode_containing_feb_2020(episodes, viterbi, oos_result)

    full_sample_episode = _compute_full_sample_episode(features)

    report = generate_covid_oos_report(
        covid_episode=covid_episode,
        full_sample_episode=full_sample_episode,
        is_detector=is_detector,
    )

    report_path = output_dir / COVID_OOS_FORENSICS_TXT.name
    report_path.write_text(report)
    print(report)

    logger.info("COVID OOS forensics saved → %s", report_path)
    return {
        "is_episode": covid_episode,
        "full_sample_episode": full_sample_episode,
        "report_path": report_path,
    }


def _load_is_detector(features: pd.DataFrame) -> HMMRegimeDetector:
    """Load or fit the IS-trained model."""
    if IS_MODEL_PATH.exists():
        return HMMRegimeDetector.load(IS_MODEL_PATH)

    is_features = features[
        (features.index >= pd.Timestamp(IS_START))
        & (features.index <= pd.Timestamp(IS_END))
    ]
    detector = HMMRegimeDetector()
    detector.fit(is_features)
    detector.save(IS_MODEL_PATH)
    return detector


def _build_oos_frame(
    oos_features: pd.DataFrame,
    oos_result: Any,
    detector: HMMRegimeDetector,
) -> pd.DataFrame:
    """Build Viterbi frame with P(high_vol) for OOS episode detection."""
    X_scaled = detector.scaler.transform(oos_features[detector.feature_columns].values)
    decoded = detector.model.predict(X_scaled)
    regime = np.where(
        decoded == detector.crisis_state, REGIME_HIGH_VOL, "normal"
    )
    vix = np.exp(oos_features["log_vix"])

    return pd.DataFrame(
        {
            "regime": regime,
            "spy_log_return": oos_features["spy_log_return"],
            "rv20": oos_features["spy_realized_vol"],
            "log_vix": oos_features["log_vix"],
            "vix": vix,
            "p_high_vol": oos_result.p_high_vol,
            "is_high_vol": regime == REGIME_HIGH_VOL,
        },
        index=oos_features.index,
    )


def find_episode_containing_feb_2020(
    episodes: pd.DataFrame,
    viterbi: pd.DataFrame,
    oos_result: Any,
) -> dict[str, Any]:
    """Return the contiguous high-vol episode overlapping February 2020."""
    if episodes.empty:
        raise ValueError("No high-volatility episodes found in OOS Viterbi path.")

    target_start = pd.Timestamp(FEB_2020_START)
    target_end = pd.Timestamp(FEB_2020_END)

    match = episodes[
        (pd.to_datetime(episodes["start_date"]) <= target_end)
        & (pd.to_datetime(episodes["end_date"]) >= target_start)
    ]

    if match.empty:
        raise ValueError("No high-vol episode overlaps February 2020.")

    row = match.iloc[0]
    seg = viterbi[
        (viterbi.index >= pd.Timestamp(row["start_date"]))
        & (viterbi.index <= pd.Timestamp(row["end_date"]))
    ]

    return {
        "model": f"IS-trained (fit {IS_START} → {IS_END})",
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "trading_days": int(row["trading_days"]),
        "avg_vix": round(float(seg["vix"].mean()), 2),
        "avg_rv20": round(float(seg["rv20"].mean()), 4),
        "avg_p_high_vol": round(float(seg["p_high_vol"].mean()), 4),
    }


def _compute_full_sample_episode(features: pd.DataFrame) -> dict[str, Any]:
    """Recompute full-sample v1.0 COVID episode stats for comparison."""
    if MODEL_PATH.exists():
        detector = HMMRegimeDetector.load(MODEL_PATH)
    else:
        detector = HMMRegimeDetector()
        detector.fit(features)

    result = detector.predict_proba(features)
    X_scaled = detector.scaler.transform(features[detector.feature_columns].values)
    decoded = detector.model.predict(X_scaled)
    regime = np.where(decoded == detector.crisis_state, REGIME_HIGH_VOL, "normal")
    vix = np.exp(features["log_vix"])

    viterbi = pd.DataFrame(
        {
            "vix": vix,
            "rv20": features["spy_realized_vol"],
            "p_high_vol": result.p_high_vol,
            "is_high_vol": regime == REGIME_HIGH_VOL,
            "spy_log_return": features["spy_log_return"],
        },
        index=features.index,
    )

    episodes = detect_high_vol_episodes(viterbi)
    ep = find_episode_containing_feb_2020(
        episodes, viterbi, result
    )
    ep["model"] = FULL_SAMPLE_COVID["model"]
    return ep


def generate_covid_oos_report(
    covid_episode: dict[str, Any],
    full_sample_episode: dict[str, Any],
    is_detector: HMMRegimeDetector,
) -> str:
    """Format the comparison report."""
    is_ep = covid_episode
    fs_ep = full_sample_episode

    delta_days = is_ep["trading_days"] - fs_ep["trading_days"]
    delta_vix = is_ep["avg_vix"] - fs_ep["avg_vix"]
    delta_rv = is_ep["avg_rv20"] - fs_ep["avg_rv20"]
    delta_p = is_ep["avg_p_high_vol"] - fs_ep["avg_p_high_vol"]

    lines = [
        "=" * 72,
        "  COVID OOS FORENSICS — IS-TRAINED MODEL vs FULL-SAMPLE v1.0",
        "=" * 72,
        f"  Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"  Target    : contiguous high-vol Viterbi episode containing Feb 2020",
        f"  IS model  : fit {IS_START} → {IS_END}, crisis_state={is_detector.crisis_state}",
        "",
        "IS-TRAINED MODEL EPISODE  (OOS Viterbi, 2019 → present)",
        "-" * 72,
        f"  Start date       : {is_ep['start_date']}",
        f"  End date         : {is_ep['end_date']}",
        f"  Trading days     : {is_ep['trading_days']}",
        f"  Average VIX      : {is_ep['avg_vix']:.2f}",
        f"  Average RV20     : {is_ep['avg_rv20']:.4f}",
        f"  Average P(hv)    : {is_ep['avg_p_high_vol']:.4f}",
        "",
        "FULL-SAMPLE v1.0 COVID EPISODE  (reference)",
        "-" * 72,
        f"  Start date       : {fs_ep['start_date']}",
        f"  End date         : {fs_ep['end_date']}",
        f"  Trading days     : {fs_ep['trading_days']}",
        f"  Average VIX      : {fs_ep['avg_vix']:.2f}",
        f"  Average RV20     : {fs_ep['avg_rv20']:.4f}",
        f"  Average P(hv)    : {fs_ep['avg_p_high_vol']:.4f}",
        "",
        "COMPARISON  (IS-trained minus full-sample)",
        "-" * 72,
        f"  Δ Trading days   : {delta_days:+d}",
        f"  Δ Average VIX    : {delta_vix:+.2f}",
        f"  Δ Average RV20   : {delta_rv:+.4f}",
        f"  Δ Average P(hv)  : {delta_p:+.4f}",
        "",
        "INTERPRETATION",
        "-" * 72,
        *_interpretation(is_ep, fs_ep),
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


def _interpretation(is_ep: dict, fs_ep: dict) -> list[str]:
    """Narrative comparison of the two COVID episodes."""
    same_start = is_ep["start_date"] == fs_ep["start_date"]
    same_end = is_ep["end_date"] == fs_ep["end_date"]
    duration_diff = is_ep["trading_days"] - fs_ep["trading_days"]

    lines: list[str] = []

    if same_start and same_end:
        lines.append(
            "  Both models identify the identical Viterbi episode boundaries."
        )
    else:
        if not same_start:
            lines.append(
                f"  Entry differs: IS model enters high vol on {is_ep['start_date']} "
                f"vs full-sample {fs_ep['start_date']}."
            )
        if not same_end:
            lines.append(
                f"  Exit differs: IS model exits on {is_ep['end_date']} "
                f"vs full-sample {fs_ep['end_date']} "
                f"({duration_diff:+d} trading days)."
            )

    if abs(duration_diff) <= 5:
        lines.append(
            "  Episode duration is essentially unchanged — IS-learned emission "
            "distributions produce the same persistence dynamics OOS."
        )
    elif duration_diff > 0:
        lines.append(
            "  The IS-trained model holds the high-vol state longer OOS, "
            "likely because IS emission means are calibrated to pre-2019 data "
            "and COVID observations appear even more extreme relative to them."
        )
    else:
        lines.append(
            "  The IS-trained model exits high vol sooner than the full-sample "
            "model, suggesting the full-sample fit partially absorbed post-2019 "
            "vol levels into its emission parameters."
        )

    if abs(is_ep["avg_p_high_vol"] - fs_ep["avg_p_high_vol"]) < 0.05:
        lines.append(
            f"  Posterior confidence is similar (IS P(hv)={is_ep['avg_p_high_vol']:.3f} "
            f"vs full-sample {fs_ep['avg_p_high_vol']:.3f})."
        )
    elif is_ep["avg_p_high_vol"] > fs_ep["avg_p_high_vol"]:
        lines.append(
            "  IS model assigns higher average P(high_vol) — COVID features are "
            "more extreme relative to IS-learned state distributions."
        )
    else:
        lines.append(
            "  Full-sample model assigns higher P(high_vol) on average."
        )

    lines.append(
        "  Market inputs (VIX, RV20) are identical between models; any boundary "
        "or probability difference arises purely from IS vs full-sample emission "
        "and transition parameters."
    )

    return lines
