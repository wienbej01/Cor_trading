"""
Technical indicators module for FX-Commodity correlation arbitrage strategy.
Provides common financial indicators like z-score, ATR, and rolling correlation.
"""

from typing import Union

import numpy as np
import pandas as pd
from loguru import logger


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the z-score of a series over a rolling window.
    
    Args:
        series: Input time series.
        window: Rolling window size for mean and std calculation.
        
    Returns:
        Series with z-scores.
        
    Raises:
        ValueError: If window is less than 2 or greater than series length.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    if window > len(series):
        raise ValueError("Window cannot be greater than series length")
    
    logger.debug(f"Calculating z-score with window {window}")
    
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Handle division by zero
    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    
    return z_scores


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) as a volatility proxy.
    
    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        window: Rolling window size for ATR calculation.
        
    Returns:
        Series with ATR values.
        
    Raises:
        ValueError: If window is less than 2 or series lengths don't match.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close series must have the same length")
    
    logger.debug(f"Calculating ATR with window {window}")
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using simple moving average
    atr_values = true_range.rolling(window=window).mean()
    
    return atr_values


def atr_proxy(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate a simplified ATR proxy using only close prices.
    This is useful when high/low data is not available.
    
    Args:
        close: Series of close prices.
        window: Rolling window size for calculation.
        
    Returns:
        Series with ATR proxy values.
        
    Raises:
        ValueError: If window is less than 2.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    
    logger.debug(f"Calculating ATR proxy with window {window}")
    
    # Use daily price changes as a proxy for true range
    price_changes = abs(close.diff())
    
    # Simple moving average of price changes
    atr_proxy_values = price_changes.rolling(window=window).mean()
    
    return atr_proxy_values


def rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series_a: First time series.
        series_b: Second time series.
        window: Rolling window size for correlation calculation.
        
    Returns:
        Series with rolling correlation values.
        
    Raises:
        ValueError: If window is less than 2 or series lengths don't match.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    if len(series_a) != len(series_b):
        raise ValueError("Series must have the same length")
    
    logger.debug(f"Calculating rolling correlation with window {window}")
    
    # Align series and drop NA values
    df = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    
    if len(df) < window:
        logger.warning(f"Not enough data points ({len(df)}) for window size {window}")
        return pd.Series(index=series_a.index, dtype=float)
    
    # Calculate rolling correlation
    correlation = df["a"].rolling(window=window).corr(df["b"])
    
    return correlation


def rolling_beta(series_y: pd.Series, series_x: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling beta (slope coefficient) between two series using OLS.
    
    Args:
        series_y: Dependent variable series.
        series_x: Independent variable series.
        window: Rolling window size for beta calculation.
        
    Returns:
        Series with rolling beta values.
        
    Raises:
        ValueError: If window is less than 2 or series lengths don't match.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    if len(series_y) != len(series_x):
        raise ValueError("Series must have the same length")
    
    logger.debug(f"Calculating rolling beta with window {window}")
    
    # Align series and drop NA values
    df = pd.DataFrame({"y": series_y, "x": series_x}).dropna()
    
    if len(df) < window:
        logger.warning(f"Not enough data points ({len(df)}) for window size {window}")
        return pd.Series(index=series_y.index, dtype=float)
    
    # Calculate rolling covariance and variance
    cov = df["y"].rolling(window=window).cov(df["x"])
    var = df["x"].rolling(window=window).var()
    
    # Calculate beta
    beta = cov / var
    
    return beta


def rolling_alpha(series_y: pd.Series, series_x: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling alpha (intercept) between two series using OLS.
    
    Args:
        series_y: Dependent variable series.
        series_x: Independent variable series.
        window: Rolling window size for alpha calculation.
        
    Returns:
        Series with rolling alpha values.
        
    Raises:
        ValueError: If window is less than 2 or series lengths don't match.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    if len(series_y) != len(series_x):
        raise ValueError("Series must have the same length")
    
    logger.debug(f"Calculating rolling alpha with window {window}")
    
    # Align series and drop NA values
    df = pd.DataFrame({"y": series_y, "x": series_x}).dropna()
    
    if len(df) < window:
        logger.warning(f"Not enough data points ({len(df)}) for window size {window}")
        return pd.Series(index=series_y.index, dtype=float)
    
    # Calculate rolling means
    mean_y = df["y"].rolling(window=window).mean()
    mean_x = df["x"].rolling(window=window).mean()
    
    # Calculate rolling beta
    beta = rolling_beta(series_y, series_x, window)
    
    # Calculate alpha
    alpha = mean_y - beta * mean_x
    
    return alpha


def zscore_robust(s: pd.Series, window: int) -> pd.Series:
    """
    Calculate robust z-score using median and median absolute deviation (MAD).
    
    Args:
        s: Input time series.
        window: Rolling window size for median and MAD calculation.
        
    Returns:
        Series with robust z-scores.
        
    Raises:
        ValueError: If window is less than 2 or greater than series length.
    """
    if window < 2:
        raise ValueError("Window must be at least 2")
    if window > len(s):
        raise ValueError("Window cannot be greater than series length")
    
    logger.debug(f"Calculating robust z-score with window {window}")
    
    roll = s.rolling(window)
    med = roll.median()
    mad = roll.apply(lambda v: np.median(np.abs(v - np.median(v))) if len(v.dropna()) else np.nan)
    return (s - med) / (1.4826 * (mad.replace(0, np.nan)) + 1e-12)