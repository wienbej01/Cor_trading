"""
Rolling window performance metrics module for FX-Commodity correlation arbitrage strategy.
Implements rolling calculations for Sharpe ratio, Sortino ratio, and maximum drawdown.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger


def _safe_sharpe(returns: pd.Series, ann_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio with safe handling of edge cases.

    Args:
        returns: Series of returns.
        ann_factor: Annualization factor (default 252 for daily data).

    Returns:
        Sharpe ratio as a float, or 0.0 if calculation is not possible.
    """
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=0)
    if sd <= 1e-12:
        return 0.0
    return float((mu * ann_factor) / (sd * np.sqrt(ann_factor)))


def _safe_sortino(returns: pd.Series, ann_factor: int = 252) -> float:
    """
    Calculate Sortino ratio with safe handling of edge cases.

    Args:
        returns: Series of returns.
        ann_factor: Annualization factor (default 252 for daily data).

    Returns:
        Sortino ratio as a float, or 0.0 if calculation is not possible.
    """
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    # Calculate downside deviation (standard deviation of negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        downside_dev = 1e-12  # Avoid division by zero
    else:
        downside_dev = negative_returns.std(ddof=0)
    if downside_dev <= 1e-12:
        return 0.0
    return float((mu * ann_factor) / (downside_dev * np.sqrt(ann_factor)))


def _safe_max_drawdown(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown with safe handling of edge cases.

    Args:
        equity: Series of equity values.

    Returns:
        Maximum drawdown as a float, or 0.0 if calculation is not possible.
    """
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _safe_sharpe_from_array(returns_array: np.ndarray, ann_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio from numpy array.

    Args:
        returns_array: Numpy array of returns.
        ann_factor: Annualization factor (default 252 for daily data).

    Returns:
        Sharpe ratio as a float, or 0.0 if calculation is not possible.
    """
    if len(returns_array) < 2:
        return 0.0
    mu = np.mean(returns_array)
    sd = np.std(returns_array, ddof=0)
    if sd <= 1e-12:
        return 0.0
    return float((mu * ann_factor) / (sd * np.sqrt(ann_factor)))


def _safe_sortino_from_array(returns_array: np.ndarray, ann_factor: int = 252) -> float:
    """
    Calculate Sortino ratio from numpy array.

    Args:
        returns_array: Numpy array of returns.
        ann_factor: Annualization factor (default 252 for daily data).

    Returns:
        Sortino ratio as a float, or 0.0 if calculation is not possible.
    """
    if len(returns_array) < 2:
        return 0.0
    mu = np.mean(returns_array)
    # Calculate downside deviation (standard deviation of negative returns)
    negative_returns = returns_array[returns_array < 0]
    if len(negative_returns) == 0:
        downside_dev = 1e-12  # Avoid division by zero
    else:
        downside_dev = np.std(negative_returns, ddof=0)
    if downside_dev <= 1e-12:
        return 0.0
    return float((mu * ann_factor) / (downside_dev * np.sqrt(ann_factor)))


def _safe_max_drawdown_from_array(equity_array: np.ndarray) -> float:
    """
    Calculate maximum drawdown from numpy array.

    Args:
        equity_array: Numpy array of equity values.

    Returns:
        Maximum drawdown as a float, or 0.0 if calculation is not possible.
    """
    if len(equity_array) < 2:
        return 0.0
    # Convert to pandas Series to use cummax
    equity_series = pd.Series(equity_array)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return float(drawdown.min())


def calculate_rolling_sharpe(
    equity: pd.Series, window: int, ann_factor: int = 252
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        equity: Series of equity values.
        window: Rolling window size in periods.
        ann_factor: Annualization factor (default 252 for daily data).

    Returns:
        Series of rolling Sharpe ratios.
    """
    returns = equity.pct_change().fillna(0)
    rolling_sharpe = returns.rolling(window=window).apply(
        lambda x: _safe_sharpe_from_array(x, ann_factor), raw=True
    )
    return rolling_sharpe


def calculate_rolling_sortino(
    equity: pd.Series, window: int, ann_factor: int = 252
) -> pd.Series:
    """
    Calculate rolling Sortino ratio.

    Args:
        equity: Series of equity values.
        window: Rolling window size in periods.
        ann_factor: Annualization factor (default 252 for daily data).

    Returns:
        Series of rolling Sortino ratios.
    """
    returns = equity.pct_change().fillna(0)
    rolling_sortino = returns.rolling(window=window).apply(
        lambda x: _safe_sortino_from_array(x, ann_factor), raw=True
    )
    return rolling_sortino


def calculate_rolling_max_drawdown(equity: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling maximum drawdown.

    Args:
        equity: Series of equity values.
        window: Rolling window size in periods.

    Returns:
        Series of rolling maximum drawdowns.
    """
    rolling_drawdown = equity.rolling(window=window).apply(
        lambda x: _safe_max_drawdown_from_array(x), raw=True
    )
    return rolling_drawdown


def calculate_rolling_metrics(
    equity: pd.Series, windows: Dict[str, int] = None
) -> Dict[str, pd.Series]:
    """
    Calculate all rolling performance metrics.

    Args:
        equity: Series of equity values.
        windows: Dictionary of window names and sizes (default: 30D, 60D, 90D).

    Returns:
        Dictionary with rolling metrics series.
    """
    if windows is None:
        windows = {"30D": 30, "60D": 60, "90D": 90}

    metrics = {}

    for window_name, window_size in windows.items():
        if len(equity) >= window_size:
            metrics[f"rolling_sharpe_{window_name}"] = calculate_rolling_sharpe(
                equity, window_size
            )
            metrics[f"rolling_sortino_{window_name}"] = calculate_rolling_sortino(
                equity, window_size
            )
            metrics[f"rolling_max_drawdown_{window_name}"] = (
                calculate_rolling_max_drawdown(equity, window_size)
            )
        else:
            # Not enough data for this window size, fill with zeros
            metrics[f"rolling_sharpe_{window_name}"] = pd.Series(
                0.0, index=equity.index
            )
            metrics[f"rolling_sortino_{window_name}"] = pd.Series(
                0.0, index=equity.index
            )
            metrics[f"rolling_max_drawdown_{window_name}"] = pd.Series(
                0.0, index=equity.index
            )

    return metrics


def add_rolling_metrics_to_config(config: Dict) -> Dict:
    """
    Add rolling metrics configuration to the pair configuration.

    Args:
        config: Configuration dictionary for a pair.

    Returns:
        Configuration dictionary with rolling metrics parameters added.
    """
    # Add rolling metrics configuration if not present
    if "rolling_metrics" not in config:
        config["rolling_metrics"] = {"windows": {"30D": 30, "60D": 60, "90D": 90}}

    return config
