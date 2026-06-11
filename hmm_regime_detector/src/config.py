"""Central configuration for the HMM regime detector."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_AUDIT_JSON = REPORTS_DIR / "model_audit.json"
MODEL_AUDIT_TXT = REPORTS_DIR / "model_audit.txt"
REGIME_DIAGNOSTICS_TXT = REPORTS_DIR / "regime_diagnostics.txt"
COVID_FORENSICS_TXT = REPORTS_DIR / "covid_forensics.txt"
COVID_OOS_FORENSICS_TXT = REPORTS_DIR / "covid_oos_forensics.txt"
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
# In-sample / out-of-sample split  (v1.1)
# ---------------------------------------------------------------------------
IS_START = "2005-02-01"
IS_END = "2018-12-31"
OOS_START = "2019-01-01"
OOS_END = None  # present (latest available data)

IS_MODEL_PATH = DATA_DIR / "hmm_model_is.pkl"
IS_SUMMARY_JSON = REPORTS_DIR / "is_summary.json"
OOS_SUMMARY_JSON = REPORTS_DIR / "oos_summary.json"
IS_OOS_EQUITY_PNG = REPORTS_DIR / "is_oos_equity_curves.png"
OOS_EVENT_ANALYSIS_TXT = REPORTS_DIR / "oos_event_analysis.txt"
GENERALIZATION_REPORT_TXT = REPORTS_DIR / "generalization_report.txt"

# OOS stress events for event analysis.
OOS_EVENTS = [
    ("2020 COVID crisis", "2020-02-20", "2020-06-30"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("March 2023 SVB episode", "2023-03-01", "2023-03-31"),
    ("2025 tariff volatility episode", "2025-02-01", "2025-05-31"),
]

# ---------------------------------------------------------------------------
# Walk-forward validation  (v1.2)
# ---------------------------------------------------------------------------
WF_SUMMARY_CSV = REPORTS_DIR / "walk_forward_summary.csv"
WF_SUMMARY_JSON = REPORTS_DIR / "walk_forward_summary.json"
WF_REPORT_TXT = REPORTS_DIR / "walk_forward_report.txt"
WF_EQUITY_PNG = REPORTS_DIR / "walk_forward_equity_curve.png"
WF_YEARLY_PNG = REPORTS_DIR / "walk_forward_yearly_metrics.png"
WF_FIRST_TEST_YEAR = 2015

# ---------------------------------------------------------------------------
# Drawdown analysis (visualization only)
# ---------------------------------------------------------------------------
DRAWDOWN_COMPARISON_PNG = REPORTS_DIR / "drawdown_comparison.png"
IS_OOS_DRAWDOWNS_PNG = REPORTS_DIR / "is_oos_drawdowns.png"
WALK_FORWARD_DRAWDOWNS_PNG = REPORTS_DIR / "walk_forward_drawdowns.png"
DRAWDOWN_SUMMARY_JSON = REPORTS_DIR / "drawdown_summary.json"

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"
