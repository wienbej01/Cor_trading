"""
Metrics module for FX-Commodity correlation arbitrage strategy.
Implements comprehensive performance metrics calculation and reporting.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from loguru import logger


def calculate_equity_stats(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate equity curve statistics.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Dictionary with equity statistics
    """
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            " ulcer_index": 0.0
        }
    
    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Annualized return
    years = len(equity_curve) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    # Volatility (annualized)
    returns = equity_curve.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0.0
    
    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0.0
    
    # Ulcer index
    ulcer_index = np.sqrt(np.mean(drawdown**2))
    
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        "ulcer_index": float(ulcer_index)
    }


def calculate_trade_stats(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate trade-level statistics.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        Dictionary with trade statistics
    """
    if trades_df.empty:
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
            "avg_duration": 0.0
        }
    
    # Basic trade counts
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["pnl"] > 0])
    losing_trades = len(trades_df[trades_df["pnl"] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Profit/loss statistics
    avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0.0
    avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0.0
    
    # Profit factor
    gross_wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # Max win/loss
    max_win = trades_df["pnl"].max()
    max_loss = trades_df["pnl"].min()
    
    # Average duration
    avg_duration = trades_df["duration"].mean() if "duration" in trades_df.columns else 0.0
    
    return {
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "max_win": float(max_win),
        "max_loss": float(max_loss),
        "avg_duration": float(avg_duration)
    }


def calculate_per_day_stats(backtest_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate per-day statistics.
    
    Args:
        backtest_df: DataFrame with backtest results
        
    Returns:
        Dictionary with per-day statistics
    """
    if backtest_df.empty:
        return {
            "best_day": 0.0,
            "worst_day": 0.0,
            "avg_daily_return": 0.0,
            "daily_volatility": 0.0,
            "positive_days": 0,
            "negative_days": 0,
            "daily_win_rate": 0.0
        }
    
    # Daily returns
    daily_returns = backtest_df["pnl"].groupby(backtest_df.index.date).sum()
    
    # Best/worst day
    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    
    # Average daily return
    avg_daily_return = daily_returns.mean()
    
    # Daily volatility
    daily_volatility = daily_returns.std()
    
    # Positive/negative days
    positive_days = len(daily_returns[daily_returns > 0])
    negative_days = len(daily_returns[daily_returns < 0])
    daily_win_rate = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0.0
    
    return {
        "best_day": float(best_day),
        "worst_day": float(worst_day),
        "avg_daily_return": float(avg_daily_return),
        "daily_volatility": float(daily_volatility),
        "positive_days": int(positive_days),
        "negative_days": int(negative_days),
        "daily_win_rate": float(daily_win_rate)
    }


def calculate_cost_slippage_stats(backtest_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate cost and slippage statistics.
    
    Args:
        backtest_df: DataFrame with backtest results
        
    Returns:
        Dictionary with cost/slippage statistics
    """
    if backtest_df.empty or "total_pnl" not in backtest_df.columns:
        return {
            "total_costs": 0.0,
            "costs_per_trade": 0.0,
            "costs_pct_of_pnl": 0.0
        }
    
    # Calculate total costs (negative values in total_pnl due to costs)
    total_pnl = backtest_df["total_pnl"].sum()
    if "pnl" in backtest_df.columns:
        gross_pnl = backtest_df["pnl"].sum()
        total_costs = gross_pnl - total_pnl
    else:
        total_costs = 0.0
    
    # Costs per trade
    trade_count = int((backtest_df["delayed_signal"].diff().abs() == 1).sum() / 2) if "delayed_signal" in backtest_df.columns else 0
    costs_per_trade = total_costs / trade_count if trade_count > 0 else 0.0
    
    # Costs as percentage of PnL
    costs_pct_of_pnl = abs(total_costs / gross_pnl * 100) if gross_pnl != 0 else 0.0
    
    return {
        "total_costs": float(total_costs),
        "costs_per_trade": float(costs_per_trade),
        "costs_pct_of_pnl": float(costs_pct_of_pnl)
    }


def generate_run_id() -> str:
    """
    Generate a unique run ID based on timestamp.
    
    Returns:
        String with run ID
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run_artifacts(
    pair: str,
    backtest_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    run_id: str = None
) -> str:
    """
    Save run artifacts to reports directory.
    
    Args:
        pair: Trading pair name
        backtest_df: DataFrame with backtest results
        trades_df: DataFrame with trade data
        config: Configuration dictionary
        metrics: Performance metrics dictionary
        run_id: Optional run ID (generated if not provided)
        
    Returns:
        Path to reports directory
    """
    if run_id is None:
        run_id = generate_run_id()
    
    # Create reports directory structure
    reports_dir = f"reports/{pair}/{run_id}"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save summary metrics
    summary_path = f"{reports_dir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save trades data
    trades_path = f"{reports_dir}/trades.csv"
    trades_df.to_csv(trades_path, index=False)
    
    # Save configuration
    config_path = f"{reports_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved run artifacts to {reports_dir}")
    return reports_dir


def calculate_comprehensive_metrics(
    backtest_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        backtest_df: DataFrame with backtest results
        trades_df: DataFrame with trade data
        config: Configuration dictionary
        
    Returns:
        Dictionary with all performance metrics
    """
    # Calculate equity curve if not present
    if "equity" not in backtest_df.columns:
        backtest_df["equity"] = (1.0 + backtest_df["pnl"]).cumprod()
    
    # Calculate all metrics
    equity_stats = calculate_equity_stats(backtest_df["equity"])
    trade_stats = calculate_trade_stats(trades_df)
    daily_stats = calculate_per_day_stats(backtest_df)
    cost_stats = calculate_cost_slippage_stats(backtest_df)
    
    # Combine all metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "equity": equity_stats,
        "trades": trade_stats,
        "daily": daily_stats,
        "costs": cost_stats,
        "config": {
            "pair": config.get("pair", "unknown"),
            "start_date": config.get("start_date", "unknown"),
            "end_date": config.get("end_date", "unknown")
        }
    }
    
    return metrics