# Metrics Catalog

This document describes all metrics calculated and reported by the FX-Commodity correlation arbitrage system.

## Equity Metrics

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| `total_return` | Total return over the backtest period | (Final Equity / Initial Equity) - 1 | Percentage |
| `annual_return` | Annualized return | (1 + Total Return)^(1/Years) - 1 | Percentage |
| `volatility` | Annualized volatility of returns | std(daily_returns) * sqrt(252) | Percentage |
| `sharpe_ratio` | Risk-adjusted return | Annual Return / Volatility | Ratio |
| `max_drawdown` | Maximum peak-to-trough decline | min((equity - running_max) / running_max) | Percentage |
| `calmar_ratio` | Return-to-max-drawdown ratio | abs(Annual Return / Max Drawdown) | Ratio |
| `ulcer_index` | Measure of downside risk | sqrt(mean(drawdown^2)) | Index |

## Trade Metrics

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| `total_trades` | Total number of trades executed | count | Number |
| `winning_trades` | Number of profitable trades | count(pnl > 0) | Number |
| `losing_trades` | Number of unprofitable trades | count(pnl < 0) | Number |
| `win_rate` | Percentage of winning trades | Winning Trades / Total Trades | Percentage |
| `avg_win` | Average profit per winning trade | mean(pnl where pnl > 0) | Currency |
| `avg_loss` | Average loss per losing trade | mean(pnl where pnl < 0) | Currency |
| `profit_factor` | Gross wins to gross losses ratio | Gross Wins / abs(Gross Losses) | Ratio |
| `max_win` | Largest winning trade | max(pnl) | Currency |
| `max_loss` | Largest losing trade | min(pnl) | Currency |
| `avg_duration` | Average trade duration | mean(duration) | Days |

## Daily Metrics

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| `best_day` | Best daily return | max(daily_pnl) | Currency |
| `worst_day` | Worst daily return | min(daily_pnl) | Currency |
| `avg_daily_return` | Average daily return | mean(daily_pnl) | Currency |
| `daily_volatility` | Volatility of daily returns | std(daily_pnl) | Currency |
| `positive_days` | Number of positive return days | count(daily_pnl > 0) | Number |
| `negative_days` | Number of negative return days | count(daily_pnl < 0) | Number |
| `daily_win_rate` | Percentage of positive return days | Positive Days / Total Days | Percentage |

## Cost Metrics

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| `total_costs` | Total transaction costs | sum(costs) | Currency |
| `costs_per_trade` | Average cost per trade | Total Costs / Total Trades | Currency |
| `costs_pct_of_pnl` | Costs as percentage of gross PnL | abs(Total Costs / Gross PnL) * 100 | Percentage |

## Configuration Metadata

| Field | Description | Example |
|-------|-------------|---------|
| `pair` | Trading pair identifier | "usdcad_wti" |
| `start_date` | Backtest start date | "2015-01-01" |
| `end_date` | Backtest end date | "2025-08-15" |

## Run Artifacts

Each backtest run produces the following artifacts in `reports/<pair>/<run_id>/`:

1. `summary.json` - Comprehensive performance metrics in JSON format
2. `trades.csv` - Detailed trade-level data in CSV format
3. `config.json` - Configuration used for the backtest in JSON format

### summary.json Structure

```json
{
  "timestamp": "2025-09-18T02:23:45.123456",
  "equity": {
    "total_return": 0.15,
    "annual_return": 0.08,
    "volatility": 0.12,
    "sharpe_ratio": 0.67,
    "max_drawdown": -0.05,
    "calmar_ratio": 1.6,
    "ulcer_index": 0.02
  },
  "trades": {
    "total_trades": 120,
    "winning_trades": 70,
    "losing_trades": 50,
    "win_rate": 0.58,
    "avg_win": 1500.0,
    "avg_loss": -800.0,
    "profit_factor": 1.8,
    "max_win": 5000.0,
    "max_loss": -3000.0,
    "avg_duration": 15.2
  },
  "daily": {
    "best_day": 2500.0,
    "worst_day": -4000.0,
    "avg_daily_return": 120.0,
    "daily_volatility": 800.0,
    "positive_days": 180,
    "negative_days": 120,
    "daily_win_rate": 0.6
  },
  "costs": {
    "total_costs": -12000.0,
    "costs_per_trade": -100.0,
    "costs_pct_of_pnl": 8.5
  },
  "config": {
    "pair": "usdcad_wti",
    "start_date": "2015-01-01",
    "end_date": "2025-08-15"
  }
}
```

### trades.csv Schema

| Column | Description | Type |
|--------|-------------|------|
| `trade_id` | Unique trade identifier | int64 |
| `entry_date` | Trade entry date | datetime |
| `exit_date` | Trade exit date | datetime |
| `direction` | Trade direction (1=long, -1=short) | int64 |
| `duration` | Trade duration in days | int64 |
| `pnl` | Trade profit/loss | float64 |
| `fx_entry` | FX entry price | float64 |
| `fx_exit` | FX exit price | float64 |
| `comd_entry` | Commodity entry price | float64 |
| `comd_exit` | Commodity exit price | float64 |
| `entry_z` | Z-score at entry (if available) | float64 |
| `exit_z` | Z-score at exit (if available) | float64 |

### config.json Structure

Contains the complete configuration used for the backtest, including:
- Strategy parameters
- Risk management settings
- Data configuration
- Pair-specific settings