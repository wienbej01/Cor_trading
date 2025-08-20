"""
Market regime detection module for FX-Commodity correlation arbitrage strategy.
Provides functions to detect market regimes and filter signals based on correlation.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .indicators import rolling_corr


def correlation_gate(
    series_a: pd.Series, 
    series_b: pd.Series, 
    corr_window: int = 60,
    min_abs_corr: float = 0.3
) -> pd.Series:
    """
    Apply correlation gate to filter signals based on rolling correlation.
    
    Args:
        series_a: First time series (e.g., FX rate).
        series_b: Second time series (e.g., commodity price).
        corr_window: Window size for rolling correlation calculation.
        min_abs_corr: Minimum absolute correlation threshold for signal validity.
        
    Returns:
        Boolean series indicating valid signals (True when correlation is strong enough).
        
    Raises:
        ValueError: If series have mismatched lengths or parameters are invalid.
    """
    if len(series_a) != len(series_b):
        raise ValueError("Series must have the same length")
    
    if corr_window < 5:
        raise ValueError("Correlation window must be at least 5")
    
    if not 0 <= min_abs_corr <= 1:
        raise ValueError("Minimum absolute correlation must be between 0 and 1")
    
    logger.debug(f"Applying correlation gate with window={corr_window}, min_abs_corr={min_abs_corr}")
    
    # Calculate rolling correlation
    correlation = rolling_corr(series_a, series_b, corr_window)
    
    # Check if absolute correlation exceeds threshold
    valid_signals = correlation.abs() >= min_abs_corr
    
    logger.info(f"Correlation gate: {valid_signals.sum()}/{len(valid_signals)} signals pass threshold")
    
    return valid_signals


def dcc_garch_filter(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 60,
    corr_threshold: float = 0.3
) -> pd.Series:
    """
    Apply DCC-GARCH filter for dynamic correlation estimation (placeholder implementation).
    
    Args:
        returns_a: Returns of first time series.
        returns_b: Returns of second time series.
        window: Window size for estimation.
        corr_threshold: Correlation threshold for signal validity.
        
    Returns:
        Boolean series indicating valid signals based on DCC-GARCH correlation.
        
    Raises:
        ValueError: If series have mismatched lengths or parameters are invalid.
    """
    if len(returns_a) != len(returns_b):
        raise ValueError("Returns series must have the same length")
    
    if window < 20:
        raise ValueError("DCC-GARCH window must be at least 20")
    
    logger.warning("DCC-GARCH filter called but not fully implemented (using rolling correlation as placeholder)")
    
    # For now, use rolling correlation as a placeholder
    # In a full implementation, this would use DCC-GARCH for dynamic correlation estimation
    correlation = rolling_corr(returns_a, returns_b, window)
    
    # Check if correlation exceeds threshold
    valid_signals = correlation.abs() >= corr_threshold
    
    logger.info(f"DCC-GARCH filter (placeholder): {valid_signals.sum()}/{len(valid_signals)} signals pass threshold")
    
    return valid_signals


def volatility_regime(
    series: pd.Series,
    window: int = 20,
    high_vol_threshold: float = 0.02,
    low_vol_threshold: float = 0.005
) -> pd.Series:
    """
    Detect volatility regime of a series.
    
    Args:
        series: Time series to analyze.
        window: Window size for volatility calculation.
        high_vol_threshold: Threshold for high volatility regime (as decimal).
        low_vol_threshold: Threshold for low volatility regime (as decimal).
        
    Returns:
        Series with regime labels: 0=low vol, 1=normal vol, 2=high vol.
        
    Raises:
        ValueError: If parameters are invalid.
    """
    if window < 5:
        raise ValueError("Volatility window must be at least 5")
    
    if high_vol_threshold <= low_vol_threshold:
        raise ValueError("High volatility threshold must be greater than low volatility threshold")
    
    logger.debug(f"Detecting volatility regime with window={window}")
    
    # Calculate returns
    returns = series.pct_change().dropna()
    
    # Calculate rolling volatility
    volatility = returns.rolling(window=window).std()
    
    # Classify regime
    regime = pd.Series(1, index=volatility.index)  # Default to normal
    
    # Low volatility regime
    regime[volatility <= low_vol_threshold] = 0
    
    # High volatility regime
    regime[volatility >= high_vol_threshold] = 2
    
    return regime


def trend_regime(
    series: pd.Series,
    window: int = 20,
    trend_threshold: float = 0.01
) -> pd.Series:
    """
    Detect trend regime of a series.
    
    Args:
        series: Time series to analyze.
        window: Window size for trend calculation.
        trend_threshold: Threshold for significant trend (as decimal).
        
    Returns:
        Series with regime labels: -1=downtrend, 0=range-bound, 1=uptrend.
        
    Raises:
        ValueError: If parameters are invalid.
    """
    if window < 5:
        raise ValueError("Trend window must be at least 5")
    
    logger.debug(f"Detecting trend regime with window={window}")
    
    # Calculate returns
    returns = series.pct_change().dropna()
    
    # Calculate cumulative returns over window
    cumulative_returns = returns.rolling(window=window).sum()
    
    # Classify regime
    regime = pd.Series(0, index=cumulative_returns.index)  # Default to range-bound
    
    # Uptrend
    regime[cumulative_returns > trend_threshold] = 1
    
    # Downtrend
    regime[cumulative_returns < -trend_threshold] = -1
    
    return regime


def combined_regime_filter(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict
) -> pd.Series:
    """
    Apply combined regime filter based on correlation, volatility, and trend.
    
    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.
        config: Configuration dictionary with regime parameters.
        
    Returns:
        Boolean series indicating valid signals based on all regime filters.
        
    Raises:
        ValueError: If required config parameters are missing.
    """
    logger.debug("Applying combined regime filter")
    
    # Extract config parameters
    try:
        corr_window = config["lookbacks"]["corr_window"]
        min_abs_corr = config["regime"]["min_abs_corr"]
    except KeyError as e:
        raise ValueError(f"Missing required config parameter: {e}")
    
    # Initialize with all signals valid
    valid_signals = pd.Series(True, index=fx_series.index)
    
    # Apply correlation gate
    corr_gate = correlation_gate(fx_series, comd_series, corr_window, min_abs_corr)
    valid_signals = valid_signals & corr_gate
    
    # Apply volatility regime filter (optional)
    if "volatility_window" in config.get("regime", {}):
        vol_window = config["regime"]["volatility_window"]
        high_vol_threshold = config["regime"].get("high_vol_threshold", 0.02)
        low_vol_threshold = config["regime"].get("low_vol_threshold", 0.005)
        
        fx_vol_regime = volatility_regime(fx_series, vol_window, high_vol_threshold, low_vol_threshold)
        comd_vol_regime = volatility_regime(comd_series, vol_window, high_vol_threshold, low_vol_threshold)
        
        # Filter out extreme volatility regimes (optional)
        if config["regime"].get("filter_extreme_vol", False):
            vol_filter = (fx_vol_regime != 2) & (comd_vol_regime != 2)
            valid_signals = valid_signals & vol_filter
    
    # Apply trend regime filter (optional)
    if "trend_window" in config.get("regime", {}):
        trend_window = config["regime"]["trend_window"]
        trend_threshold = config["regime"].get("trend_threshold", 0.01)
        
        fx_trend_regime = trend_regime(fx_series, trend_window, trend_threshold)
        comd_trend_regime = trend_regime(comd_series, trend_window, trend_threshold)
        
        # Filter out strong trending regimes (optional)
        if config["regime"].get("filter_strong_trend", False):
            trend_filter = (fx_trend_regime.abs() != 1) & (comd_trend_regime.abs() != 1)
            valid_signals = valid_signals & trend_filter
    
    logger.info(f"Combined regime filter: {valid_signals.sum()}/{len(valid_signals)} signals pass all filters")
    
    return valid_signals