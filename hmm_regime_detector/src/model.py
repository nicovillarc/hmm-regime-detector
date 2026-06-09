"""Gaussian Hidden Markov Model for volatility regime detection."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.config import (
    COVARIANCE_TYPE,
    CRISIS_IDENTIFICATION_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_PATH,
    N_ITER,
    N_STATES,
    RANDOM_STATE,
    REGIME_HIGH_VOL,
    REGIME_NORMAL,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """Container for regime inference output on a single date or series."""

    dates: pd.DatetimeIndex
    p_normal: np.ndarray
    p_high_vol: np.ndarray
    most_likely_regime: np.ndarray
    crisis_state: int
    normal_state: int


class HMMRegimeDetector:
    """
    Two-state Gaussian HMM that infers normal vs. high-volatility regimes.

    The model is fit on standardised features.  Crisis-state identification
    uses the original (non-scaled) realised-volatility and log(VIX) averages
    conditional on the Viterbi-decoded state assignments.
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        covariance_type: str = COVARIANCE_TYPE,
        n_iter: int = N_ITER,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: GaussianHMM | None = None
        self.scaler = StandardScaler()
        self.crisis_state: int | None = None
        self.normal_state: int | None = None
        self.feature_columns: list[str] = FEATURE_COLUMNS.copy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, features: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the HMM on standardised features.

        Parameters
        ----------
        features : pd.DataFrame
            Must contain all columns listed in FEATURE_COLUMNS.

        Returns
        -------
        self
        """
        X_raw = features[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X_raw)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        # Identify crisis state using original-scale volatility features.
        decoded_states = self.model.predict(X_scaled)
        self.crisis_state, self.normal_state = self._identify_crisis_state(
            features, decoded_states
        )

        logger.info(
            "Fitted HMM — crisis_state=%d, normal_state=%d",
            self.crisis_state,
            self.normal_state,
        )
        return self

    def _identify_crisis_state(
        self,
        features: pd.DataFrame,
        decoded_states: np.ndarray,
    ) -> tuple[int, int]:
        """
        Label the crisis state as the one with the highest combined average
        of realised volatility and log(VIX) on original (non-scaled) data.
        """
        combined_scores: dict[int, float] = {}

        for state in range(self.n_states):
            mask = decoded_states == state
            if mask.sum() == 0:
                combined_scores[state] = -np.inf
                continue

            vol_mean = features.loc[mask, CRISIS_IDENTIFICATION_COLUMNS[0]].mean()
            vix_mean = features.loc[mask, CRISIS_IDENTIFICATION_COLUMNS[1]].mean()
            combined_scores[state] = vol_mean + vix_mean

        crisis = max(combined_scores, key=combined_scores.get)
        normal = 1 - crisis if self.n_states == 2 else min(
            combined_scores, key=combined_scores.get
        )
        return crisis, normal

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, features: pd.DataFrame) -> RegimeResult:
        """
        Compute posterior state probabilities for each observation.

        Returns
        -------
        RegimeResult
            Posterior probabilities mapped to normal / high-volatility labels.
        """
        if self.model is None or self.crisis_state is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_scaled = self.scaler.transform(features[self.feature_columns].values)
        state_proba = self.model.predict_proba(X_scaled)

        p_high_vol = state_proba[:, self.crisis_state]
        p_normal = state_proba[:, self.normal_state]

        most_likely = np.where(
            p_high_vol >= p_normal, REGIME_HIGH_VOL, REGIME_NORMAL
        )

        return RegimeResult(
            dates=features.index,
            p_normal=p_normal,
            p_high_vol=p_high_vol,
            most_likely_regime=most_likely,
            crisis_state=self.crisis_state,
            normal_state=self.normal_state,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        """Serialise the fitted model, scaler, and state labels."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model is not fitted.")

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "crisis_state": self.crisis_state,
            "normal_state": self.normal_state,
            "feature_columns": self.feature_columns,
            "hyperparameters": {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "random_state": self.random_state,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "HMMRegimeDetector":
        """Load a previously serialised model."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        detector = cls(**payload["hyperparameters"])
        detector.model = payload["model"]
        detector.scaler = payload["scaler"]
        detector.crisis_state = payload["crisis_state"]
        detector.normal_state = payload["normal_state"]
        detector.feature_columns = payload["feature_columns"]

        logger.info("Model loaded ← %s", path)
        return detector
