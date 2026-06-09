"""Central configuration for the HMM regime detector."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = DATA_DIR / "hmm_model.pkl"
RAW_DATA_PATH = DATA_DIR / "market_data.csv"
FEATURES_PATH = DATA_DIR / "features.csv"

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"
DATA_START = "2005-01-01"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
REALIZED_VOL_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252

FEATURE_COLUMNS = [
    "spy_log_return",
    "spy_realized_vol",
    "log_vix",
]

# Columns used to identify the crisis state (original, non-scaled scale).
CRISIS_IDENTIFICATION_COLUMNS = ["spy_realized_vol", "log_vix"]

# ---------------------------------------------------------------------------
# HMM hyper-parameters
# ---------------------------------------------------------------------------
N_STATES = 2
COVARIANCE_TYPE = "full"
N_ITER = 1000
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------
REGIME_NORMAL = "normal"
REGIME_HIGH_VOL = "high_volatility"

# ---------------------------------------------------------------------------
# Hysteresis thresholds (position switching)
# ---------------------------------------------------------------------------
HIGH_VOL_ENTER_THRESHOLD = 0.60  # switch to CASH when P(high_vol) >= this
HIGH_VOL_EXIT_THRESHOLD = 0.40   # switch to INVESTED when P(high_vol) <= this
HYSTERESIS_CONSECUTIVE_DAYS = 3

POSITION_INVESTED = "INVESTED"
POSITION_CASH = "CASH"

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.0  # cash earns 0 % (simplification)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"
