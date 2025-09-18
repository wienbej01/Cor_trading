"""
This module contains functions for calculating various performance metrics for backtests.
"""

import pandas as pd

import numpy as np

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
    pass

def calculate_cost_slippage_stats(trades: pd.DataFrame) -> dict:
    """Calculates cost and slippage statistics."""
    pass
