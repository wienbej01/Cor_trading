# Backtest Result Tracker

This document tracks the results of the backtests run on the FX-Commodity correlation arbitrage strategy.

| Date       | Pair         | HMM Filter | HMM Parameters                               | Total Return | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate | Profit Factor |
|------------|--------------|------------|----------------------------------------------|--------------|---------------|--------------|--------------|----------|---------------|
| 2025-09-17 | usdcad_wti   | Disabled   | N/A                                          | -1.72%       | -0.16%        | -0.02        | -22.21%      | 23.70%   | 0.97          |
| 2025-09-17 | usdcad_wti   | Enabled    | `n_states: 3`, `window: 126`, `df: 3.0`, `tol: 1e-4` | -15.80%      | -1.61%        | -0.21        | -30.23%      | 15.19%   | 0.81          |
| 2025-09-17 | usdcad_wti   | Enabled    | `n_states: 3`, `window: 126`, `df: 5.0`, `tol: 1e-3`, `n_iter: 100` | -15.80%      | -1.61%        | -0.21        | -30.23%      | 15.19%   | 0.81          |
