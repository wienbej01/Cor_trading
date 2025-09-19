# Metrics Catalog

This document catalogs the metrics used in the backtesting engine.

## Equity Statistics

- **total_return**: The total return of the strategy over the entire backtest period.
- **annual_return**: The annualized return of the strategy.
- **volatility**: The annualized volatility of the strategy's returns.
- **sharpe_ratio**: The Sharpe ratio of the strategy.
- **max_drawdown**: The maximum drawdown of the strategy.
- **calmar_ratio**: The Calmar ratio of the strategy.
- **ulcer_index**: The Ulcer index of the strategy.

## Trade Statistics

- **total_trades**: The total number of trades executed.
- **winning_trades**: The number of winning trades.
- **losing_trades**: The number of losing trades.
- **win_rate**: The percentage of winning trades.
- **avg_win**: The average profit of a winning trade.
- **avg_loss**: The average loss of a losing trade.
- **profit_factor**: The ratio of the sum of profits from winning trades to the sum of losses from losing trades.
- **max_win**: The maximum profit of a single trade.
- **max_loss**: The maximum loss of a single trade.
- **avg_duration**: The average duration of a trade in bars.

## Per-Day Statistics

- **best_day**: The best daily return.
- **worst_day**: The worst daily return.
- **avg_daily_return**: The average daily return.
- **daily_volatility**: The standard deviation of daily returns.
- **positive_days**: The number of days with positive returns.
- **negative_days**: The number of days with negative returns.
- **daily_win_rate**: The percentage of days with positive returns.

## Cost and Slippage Statistics

- **total_costs**: The total costs incurred from slippage and commissions.
- **costs_per_trade**: The average cost per trade.
- **costs_pct_of_pnl**: The total costs as a percentage of the total profit and loss.
