# FX-Commodity Correlation Arbitrage Strategy

> **Note:** This project is currently under review and optimization. The trading strategy as implemented in the main branch is not profitable. The documentation has been updated to reflect the current, reproducible backtest results.

A production-grade Python implementation of a mean-reversion correlation/cointegration strategy between FX and commodities, focusing on USD/CAD↔WTI and USD/NOK↔Brent pairs.

## Overview

This project provides a framework for backtesting a correlation arbitrage strategy. Key features include:

- **Mean-reversion strategy** based on cointegration between FX and commodity pairs.
- **Dynamic hedge ratio calculation** using a Kalman Filter or Rolling OLS.
- **Regime filtering** based on correlation and cointegration thresholds.
- **Deterministic backtesting** with one-bar execution delay and transaction costs.
- **Inverse-volatility position sizing**.
- **Config-driven approach** with YAML configuration files.

The strategy aims to exploit the historical correlation between commodity-exporting countries' currencies and their primary commodity exports. When the relationship between these pairs deviates from their historical norm, the system takes positions expecting a reversion to the mean.

## Quick Start

### 1. Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for cloning the repository)

### 2. Setup

```bash
# Clone the repository
git clone <repository-url>
cd fx-commodity-arb

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Your First Backtest

```bash
# Run backtest for USD/CAD - WTI pair (2015-2025)
python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15

# Run backtest for USD/NOK - Brent pair (2015-2025)
python src/run_backtest.py run --pair usdnok_brent --start 2015-01-01 --end 2025-08-15
```

### 4. Explore Available Pairs

```bash
# List all configured trading pairs
python src/run_backtest.py list-pairs

# Show configuration for a specific pair
python src/run_backtest.py show-config --pair usdcad_wti
```

## Current Backtest Results

The following results are reproducible using the code in the main branch as of 2025-09-04.

### USD/CAD ↔ WTI Backtest Results

```
============================================================
BACKTEST SUMMARY: USDCAD_WTI
============================================================
Period: 2015-01-01 to 2025-08-15
Total Return: -6.78%
Annual Return: -0.66%
Sharpe Ratio: -0.68
Max Drawdown: -22.12%
Number of Trades: 520
Win Rate: 23.70%
Profit Factor: 0.97
============================================================
Full results saved to: backtest_results
============================================================
```

### USD/NOK ↔ Brent Backtest Results

```
============================================================
BACKTEST SUMMARY: USDNOK_BRENT
============================================================
Period: 2015-01-01 to 2025-08-15
Total Return: -26.15%
Annual Return: -2.82%
Sharpe Ratio: -2.74
Max Drawdown: -30.77%
Number of Trades: 612
Win Rate: 25.51%
Profit Factor: 0.87
============================================================
Full results saved to: backtest_results
============================================================
```

**Performance Report:**
```
FX-Commodity Correlation Arbitrage Backtest Report
================================================

Period: 2015-01-01 to 2025-08-15 (3858 days)

Performance Metrics:
- Total PnL: 3,872.00
- Total Return: 38.72%
- Annual Return: 3.31%
- Volatility (Annual): 4.19%
- Sharpe Ratio: 0.79
- Maximum Drawdown: -21.15%

Trading Statistics:
- Number of Trades: 52
- Win Rate: 61.54%
- Average Win: 165.28
- Average Loss: -102.37
- Profit Factor: 1.76

Note: All calculations include one-bar execution delay.
```

## Backtest Results

For a detailed log of backtest results, see the [Backtest Result Tracker](docs/backtest_tracker.md).

## Project Structure

```
src/
  core/config.py              # Configuration management
  data/yahoo_loader.py        # Yahoo Finance data loading
  features/indicators.py      # Technical indicators (z-score, ATR, correlation)
  features/cointegration.py   # Cointegration analysis (ADF, OU half-life)
  features/spread.py          # Spread calculation (OLS/Kalman)
  features/regime.py          # Regime detection
  strategy/mean_reversion.py  # Signal generation and position sizing
  ml/filter.py                # ML signal filter (stub)
  backtest/engine.py          # Backtesting engine
  exec/broker_stub.py         # Broker adapters (stubs)
  configs/pairs.yaml          # Pair configurations
  utils/logging.py            # Logging configuration
  run_backtest.py             # CLI backtest runner
```

## Configuration

Trading pairs and strategy parameters are configured in `configs/pairs.yaml`.

### Advanced Options
- `use_kalman`: Whether to use a Kalman Filter for dynamic hedge ratio calculation (when true). If false, Rolling OLS is used.

## Conclusion

The project provides a robust and well-structured framework for backtesting pairs trading strategies. The code is modular, configuration is centralized, and the backtesting engine includes realistic features like transaction costs and execution delays.

However, the mean-reversion strategy with its current parameters is **not profitable**. The high number of trades and low win rate indicate that the signal generation and filtering logic require significant improvement.

### Next Steps

The immediate focus is on **strategy improvement**. This involves:
1.  A thorough review and optimization of all strategy parameters.
2.  Refining the regime-filtering logic to better identify favorable trading conditions.
3.  Exploring alternative methods for spread calculation and signal generation.

The long-term goal is to develop a profitable, robust, and production-ready trading system.

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves risk, and past performance is not indicative of future results. Use at your own risk.

## Configuration

Trading pairs and strategy parameters are configured in `configs/pairs.yaml`.

### Advanced Options
- `use_kalman`: Whether to use a Kalman Filter for dynamic hedge ratio calculation (when true). If false, Rolling OLS is used.

## Conclusion

The project provides a robust and well-structured framework for backtesting pairs trading strategies. The code is modular, configuration is centralized, and the backtesting engine includes realistic features like transaction costs and execution delays.

However, the mean-reversion strategy with its current parameters is **not profitable**. The high number of trades and low win rate indicate that the signal generation and filtering logic require significant improvement.

### Next Steps

The immediate focus is on **strategy improvement**. This involves:
1.  A thorough review and optimization of all strategy parameters.
2.  Refining the regime-filtering logic to better identify favorable trading conditions.
3.  Exploring alternative methods for spread calculation and signal generation.

The long-term goal is to develop a profitable, robust, and production-ready trading system.

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves risk, and past performance is not indicative of future results. Use at your own risk.