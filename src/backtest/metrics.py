"""
This module contains functions for calculating various performance metrics for backtests.
"""

import pandas as pd

import numpy as np

import json
import os
from datetime import datetime

def calculate_equity_stats(equity_curve: pd.Series) -> dict:
    """Calculates equity statistics."""
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "ulcer_index": 0.0,
        }

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0.0

    returns = equity_curve.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)

    sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    ulcer_index = np.sqrt(np.mean(drawdown**2))

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "ulcer_index": ulcer_index,
    }

def calculate_trade_stats(trades: pd.DataFrame) -> dict:
    """Calculates trade statistics."""
    if trades.empty:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "avg_duration": 0.0,
        }

    total_trades = len(trades)
    winning_trades = trades[trades["pnl"] > 0]
    losing_trades = trades[trades["pnl"] < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

    avg_win = winning_trades["pnl"].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades["pnl"].mean() if not losing_trades.empty else 0.0

    profit_factor = abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum()) if losing_trades["pnl"].sum() != 0 else 0.0

    max_win = trades["pnl"].max()
    max_loss = trades["pnl"].min()

    avg_duration = trades["duration"].mean()

    return {
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_win": max_win,
        "max_loss": max_loss,
        "avg_duration": avg_duration,
    }

def calculate_per_day_stats(daily_returns: pd.Series) -> dict:
    """Calculates per-day statistics."""
    if daily_returns.empty:
        return {
            "best_day": 0.0,
            "worst_day": 0.0,
            "avg_daily_return": 0.0,
            "daily_volatility": 0.0,
            "positive_days": 0,
            "negative_days": 0,
            "daily_win_rate": 0.0,
        }

    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    avg_daily_return = daily_returns.mean()
    daily_volatility = daily_returns.std()
    positive_days = (daily_returns > 0).sum()
    negative_days = (daily_returns < 0).sum()
    daily_win_rate = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0.0

    return {
        "best_day": best_day,
        "worst_day": worst_day,
        "avg_daily_return": avg_daily_return,
        "daily_volatility": daily_volatility,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "daily_win_rate": daily_win_rate,
    }

def calculate_cost_slippage_stats(trades: pd.DataFrame) -> dict:
    """Calculates cost and slippage statistics."""
    if trades.empty or "pnl" not in trades.columns or "cost" not in trades.columns:
        return {
            "total_costs": 0.0,
            "costs_per_trade": 0.0,
            "costs_pct_of_pnl": 0.0,
        }

    total_costs = trades["cost"].sum()
    total_pnl = trades["pnl"].sum()
    costs_per_trade = total_costs / len(trades) if len(trades) > 0 else 0.0
    costs_pct_of_pnl = total_costs / total_pnl if total_pnl != 0 else 0.0

    return {
        "total_costs": total_costs,
        "costs_per_trade": costs_per_trade,
        "costs_pct_of_pnl": costs_pct_of_pnl,
    }

def generate_run_id() -> str:
    """Generates a unique run ID based on the current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_run_artifacts(
    pair: str,
    backtest_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: dict,
    metrics: dict,
    run_id: str,
) -> str:
    """Saves backtest artifacts to the reports directory."""
    reports_path = f"reports/{pair}/{run_id}"
    os.makedirs(reports_path, exist_ok=True)

    # Save summary
    summary_path = os.path.join(reports_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save trades
    trades_path = os.path.join(reports_path, "trades.parquet")
    trades_df.to_parquet(trades_path)

    # Save config
    config_path = os.path.join(reports_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return reports_path

def calculate_comprehensive_metrics(backtest_df: pd.DataFrame, trades_df: pd.DataFrame, config: dict) -> dict:
    """Calculates a comprehensive set of metrics for a backtest run."""
    equity_curve = backtest_df["equity"]
    daily_returns = backtest_df["pnl"]

    equity_stats = calculate_equity_stats(equity_curve)
    trade_stats = calculate_trade_stats(trades_df)
    per_day_stats = calculate_per_day_stats(daily_returns)
    cost_slippage_stats = calculate_cost_slippage_stats(trades_df)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "equity": equity_stats,
        "trades": trade_stats,
        "daily": per_day_stats,
        "costs": cost_slippage_stats,
        "config": config,
    }
