# HMM Regime Detector

A production-style **Hidden Markov Model** regime detector for the US equity market. The model infers whether the market is in a **normal volatility** or **high volatility / crisis** regime using daily SPY and VIX data — it does **not** predict returns.

## Features

| Feature | Description |
|---|---|
| `spy_log_return` | Daily log return of SPY |
| `spy_realized_vol` | 20-day rolling std of log returns × √252 (annualised) |
| `log_vix` | Natural log of the VIX index level |

## Model

- **Gaussian HMM** with 2 hidden states
- `covariance_type="full"`, `n_iter=1000`, `random_state=42`
- Features standardised with `StandardScaler` before fitting
- Crisis state identified as the state with the highest combined average of realised volatility and log(VIX) on original (non-scaled) data

## Regime Decision Logic

Posterior probabilities are computed via `model.predict_proba()`. Position recommendations use **hysteresis** to avoid whipsawing:

| Action | Condition |
|---|---|
| Switch to **CASH** | P(high vol) ≥ 0.60 for **3 consecutive days** |
| Switch to **INVESTED** | P(high vol) ≤ 0.40 for **3 consecutive days** |

## Project Structure

```
hmm_regime_detector/
├── data/                  # Downloaded data & saved model
├── reports/               # JSON reports & plots
├── src/
│   ├── config.py          # Central configuration
│   ├── data_loader.py     # yfinance data download
│   ├── features.py        # Feature engineering
│   ├── model.py           # HMM training & inference
│   ├── backtest.py        # Strategy backtest engine
│   └── report.py          # Report generation & plots
├── train.py               # Train and save the model
├── predict_today.py       # Daily regime report
├── run_backtest.py        # Full backtest + plots
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (downloads data, fits HMM, saves to data/)
python train.py

# 3. Get today's regime recommendation
python predict_today.py

# 4. Run backtest and generate plots
python run_backtest.py
```

## Output

### Daily Report (`reports/latest_report.json`)

```json
{
  "latest_date": "2026-06-06",
  "p_normal": 0.8234,
  "p_high_volatility": 0.1766,
  "current_regime": "normal",
  "recommendation": "INVESTED"
}
```

### Backtest Plots (`reports/regime_analysis.png`)

Three-panel chart:
1. **SPY price** with normal / crisis regime shading
2. **P(high volatility)** through time with hysteresis thresholds
3. **Equity curves** — buy-and-hold SPY vs. HMM risk overlay

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past performance of any strategy does not guarantee future results.
