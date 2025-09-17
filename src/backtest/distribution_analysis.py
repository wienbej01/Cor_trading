"""
Trade return distribution analysis for FX-Commodity correlation strategy.

Implements skewness, kurtosis, VaR, and CVaR calculations for trade returns.
"""

from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


def calculate_skewness(returns: pd.Series) -> float:
    """
    Calculate skewness of returns.

    Args:
        returns: Series of returns.

    Returns:
        Skewness value.
    """
    if len(returns) < 3:
        return 0.0
    return float(stats.skew(returns, nan_policy="omit"))


def calculate_kurtosis(returns: pd.Series) -> float:
    """
    Calculate kurtosis of returns.

    Args:
        returns: Series of returns.

    Returns:
        Kurtosis value.
    """
    if len(returns) < 4:
        return 0.0
    return float(stats.kurtosis(returns, nan_policy="omit"))


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) at specified confidence level.

    Args:
        returns: Series of returns.
        confidence_level: Confidence level (default: 0.05 for 95%).

    Returns:
        VaR value.
    """
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, confidence_level * 100))


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) at specified confidence level.

    Args:
        returns: Series of returns.
        confidence_level: Confidence level (default: 0.05 for 95%).

    Returns:
        CVaR value.
    """
    if len(returns) == 0:
        return 0.0
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return float(cvar) if not np.isnan(cvar) else 0.0


def calculate_distribution_metrics(returns: pd.Series) -> Dict:
    """
    Calculate comprehensive distribution metrics for trade returns.

    Args:
        returns: Series of trade returns.

    Returns:
        Dictionary with distribution metrics.
    """
    if len(returns) == 0:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
        }

    metrics = {
        "skewness": calculate_skewness(returns),
        "kurtosis": calculate_kurtosis(returns),
        "var_95": calculate_var(returns, 0.05),
        "var_99": calculate_var(returns, 0.01),
        "cvar_95": calculate_cvar(returns, 0.05),
        "cvar_99": calculate_cvar(returns, 0.01),
    }

    return metrics


def plot_return_distribution(
    returns: pd.Series, title: str = "Trade Return Distribution"
) -> None:
    """
    Plot histogram of trade returns with normal distribution overlay.

    Args:
        returns: Series of trade returns.
        title: Title for the plot.
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=30, alpha=0.7, density=True, label="Actual Returns")

        # Add normal distribution overlay
        mean_return = returns.mean()
        std_return = returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x, mean_return, std_return)
        plt.plot(x, normal_dist, "r-", label="Normal Distribution")

        plt.title(title)
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def analyze_return_distribution(returns: pd.Series) -> Dict:
    """
    Perform comprehensive analysis of return distribution.

    Args:
        returns: Series of trade returns.

    Returns:
        Dictionary with detailed distribution analysis.
    """
    if len(returns) == 0:
        return {
            "distribution_metrics": {
                "skewness": 0.0,
                "kurtosis": 0.0,
                "var_95": 0.0,
                "var_99": 0.0,
                "cvar_95": 0.0,
                "cvar_99": 0.0,
            },
            "distribution_interpretation": "No trades to analyze",
        }

    # Calculate distribution metrics
    metrics = calculate_distribution_metrics(returns)

    # Interpretation
    interpretation = []
    if metrics["skewness"] > 0.5:
        interpretation.append("Positive skewness: More large positive returns")
    elif metrics["skewness"] < -0.5:
        interpretation.append("Negative skewness: More large negative returns")
    else:
        interpretation.append("Approximately symmetric distribution")

    if metrics["kurtosis"] > 3:
        interpretation.append("Leptokurtic: Fat tails, more extreme values")
    elif metrics["kurtosis"] < 3:
        interpretation.append("Platykurtic: Thin tails, fewer extreme values")
    else:
        interpretation.append("Mesokurtic: Normal-like tail behavior")

    # Risk assessment
    if metrics["var_95"] < -0.02:  # 2% threshold
        interpretation.append("High VaR at 95% confidence: Significant tail risk")
    if metrics["var_99"] < -0.05:  # 5% threshold
        interpretation.append("High VaR at 99% confidence: Extreme tail risk")

    return {
        "distribution_metrics": metrics,
        "distribution_interpretation": "; ".join(interpretation),
    }
