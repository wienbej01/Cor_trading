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
## System Description and Trading Logic

This section provides a precise, testable rulebook for the strategy, links to the exact code, and current backtest performance across pairs.

Architecture overview
- Data: Yahoo Finance daily closes for FX and commodities.
- Signal engine: Mean-reversion on a beta-hedged spread with robust z-scores and regime gates.
- Execution: One-bar delay, parameterized costs, inverse-volatility sizing.
- Risk: Time-stop, z-stop, size and loss caps.

Key modules (click to open code)
- Signal generation: [mean_reversion.generate_signals()](src/strategy/mean_reversion.py:48)
- Regime gating: [regime.combined_regime_filter()](src/features/regime.py:355)
- Corr gate: [regime.correlation_gate()](src/features/regime.py:17)
- Robust z-score: [indicators.zscore_robust()](src/features/indicators.py:233)
- OU half-life: [cointegration.ou_half_life()](src/features/cointegration.py:52)
- Hurst exponent: [cointegration.hurst_exponent()](src/features/cointegration.py:124)
- Backtest engine: [backtest.engine.backtest_pair()](src/backtest/engine.py:82), [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:333)
- CLI runner: [run_backtest.py](src/run_backtest.py:1)

Rulebook (entries, exits, filters, sizing)
- Session/timeframe:
  - Daily bars. Signals evaluated and applied at bar_close + 1 bar (one-bar execution delay).
- Spread and z-score:
  - Compute beta-hedged spread S_t = FX_t − β_t × COMD_t using rolling OLS or Kalman (config: lookbacks.beta_window; use_kalman).
    - Implementation: [features.spread.compute_spread()](src/features/spread.py:1)
  - Robust z-score: z_t = (S_t − rolling_median(S, z_window)) / (1.4826 × MAD + 1e−12).
    - Implementation: [indicators.zscore_robust()](src/features/indicators.py:233)
- Regime gating (no leakage; uses only past info):
  1) Correlation gate: |ρ(FX, COMD)| over corr_window must exceed min_abs_corr.
     - [regime.correlation_gate()](src/features/regime.py:17)
  2) External regime filter (configurable): combined of volatility, trend, HMM, VIX overlay.
     - [regime.combined_regime_filter()](src/features/regime.py:355)
  3) Structural diagnostics (soft, not hard gate): ADF p-value, OU half-life, and Hurst on spread. If all pass (adf_p ≤ thresholds.adf_p, 2 ≤ OU HL ≤ 252, Hurst &lt; 0.6), z-thresholds are slightly relaxed; otherwise tightened.
     - Applied in: [mean_reversion.generate_signals()](src/strategy/mean_reversion.py:138)
- Dynamic thresholds:
  - Vol-scaling: vol_scale = stdev(S, 20) / rolling_median(stdev(S,20), 252), clipped [0.7, 1.5].
  - Structural scaling: structural_scale = 0.9 if diagnostics OK else 1.1.
  - Entry threshold: entry_z_dyn = clip(entry_z × structural_scale × vol_scale, lower=0.8).
  - Exit threshold: exit_z_dyn = clip(exit_z × (1/structural_scale) × vol_scale, upper=1.25).
    - Implementation: [mean_reversion.generate_signals()](src/strategy/mean_reversion.py:176)
- Entries/exits and stops:
  - Entry long: z ≤ −entry_z_dyn and good_regime True.
  - Entry short: z ≥ entry_z_dyn and good_regime True.
  - Exit flat: |z| ≤ exit_z_dyn (state-based exit logic).
  - Z-stop: long_stop if z ≤ −stop_z; short_stop if z ≥ stop_z.
  - Time-stop: close position after N days where N = min(max_days, 3 × OU half-life), bounded to [2, max_days].
    - Implementation: [mean_reversion.apply_time_stop()](src/strategy/mean_reversion.py:333)
- Position sizing and execution:
  - Inverse-volatility per leg: size = target_vol_per_leg / ATR_proxy(close, atr_window); FX leg is further divided by FX price if inverse_fx_for_quote_ccy_strength is true.
    - [mean_reversion.calculate_position_sizes()](src/strategy/mean_reversion.py:287)
  - Opposite legs: FX leg aligned with signal; commodity leg opposed (pair trade).
  - Execution: all trades applied at next bar close (one-bar delay).
- Costs model:
  - Parameterized in per-pair config under exec.costs; applied inside backtest engine.
- Risk guardrails (enforced in config):
  - Max position, daily/weekly loss caps, circuit breaker toggles.

Economic rationale
- Exporters’ currencies co-move with their key commodities over medium horizons. Large, regime-consistent spread dislocations exhibit mean-reversion, captured with robust z and structural diagnostics. Correlation and regime filters reduce false positives in trending/high-vol states.

Exact labels (for supervised research, optional)
- Enter at bar t+1 if z_t crosses entry_z_dyn and gate_t True.
- Exit at bar t+1 if |z_t| ≤ exit_z_dyn or time-stop/z-stop triggers at t.
- Forward return labels can be defined leakage-free as r_{t+1→t+k} using only future prices post-entry; ensure purged/embargoed splits for ML.

Reproducible backtests (examples)
- Single pair:
  - python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15 --save-data
- Multi-pair loop (bash):
  - for p in usdcad_wti usdnok_brent audusd_gold usdzar_platinum usdclp_copper usdmxn_wti usdbrl_soybeans; do python src/run_backtest.py run --pair $p --start 2015-01-01 --end 2025-08-15 --save-data; done

Latest backtest performance (as of 2025-09-17; see backtest_results/)
- Notes:
  - Performance includes one-bar delay and configured costs.
  - Some newly added pairs were previously over-filtered; regime softening now permits signals. A known issue is being investigated where reported PnL aggregates to 0.00 for some runs despite non-zero trade counts; see [backtest.engine.run_backtest()](src/backtest/engine.py:393).
- Summary:
  - USDCAD–WTI (2015–2025): Total Return −6.78%, Sharpe −0.68, MaxDD −22.12%, Trades 520. Source: Backtest report in backtest_results/ (earlier run; see section “Current Backtest Results” above).
  - USDNOK–Brent (2015–2025): Total Return −26.15%, Sharpe −2.74, MaxDD −30.77%, Trades 612. Source: Backtest report in backtest_results/.
  - AUDUSD–Gold (2015–2025): Previously 0 trades under strict gating; after regime softening, re-run required to refresh metrics (see backtest_results/).
  - USDZAR–Platinum (2015–2025): Previously 0 trades; re-run required after softening (see backtest_results/).
  - USDCLP–Copper (2015–2025): Previously 0 trades; re-run required after softening (see backtest_results/).
  - USDMXN–WTI (2015–2025): Signals observed (e.g., 60 active signals; ~35–36 trades in latest run). Current report shows 0.00 PnL due to aggregation issue under investigation (see [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:333)).
  - USDBRL–Soybeans (2015–2025): Re-run in progress; see backtest_results/ for final report.

How to fetch latest pair reports
- Each run saves:
  - Signals CSV: backtest_results/&lt;pair&gt;_signals_YYYYMMDD_HHMMSS.csv
  - Backtest CSV: backtest_results/&lt;pair&gt;_backtest_YYYYMMDD_HHMMSS.csv
  - Text Report: backtest_results/&lt;pair&gt;_report_YYYYMMDD_HHMMSS.txt
- Parse reports programmatically to build a performance table:
  - See [docs/backtest_tracker.md](docs/backtest_tracker.md)

Known issues and next steps
- Backtest PnL zeroing bug: Investigate position propagation after time-stop and ensure costs/positions are aggregated correctly across trades. Areas to inspect:
  - [mean_reversion.apply_time_stop()](src/strategy/mean_reversion.py:333)
  - [backtest.engine._calculate_trade_stats()](src/backtest/engine.py:251)
  - [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:333)
- Regime thresholds: Continue calibrated softening and ablations (corr_window, min_abs_corr, HMM window/df/tol) to balance trade frequency and quality.
- Documentation: This section supersedes older summaries; all new reports are under backtest_results/.
