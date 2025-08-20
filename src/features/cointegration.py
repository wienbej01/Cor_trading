"""
Cointegration analysis module for FX-Commodity correlation arbitrage strategy.
Provides functions to test for cointegration and calculate mean reversion parameters.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller


def adf_pvalue(series: pd.Series, max_lag: int = 1) -> float:
    """
    Calculate the Augmented Dickey-Fuller test p-value for stationarity.
    
    Args:
        series: Time series to test for stationarity.
        max_lag: Maximum lag for the ADF test.
        
    Returns:
        p-value from the ADF test.
        
    Raises:
        ValueError: If series has insufficient data points.
    """
    if len(series) < 10:
        raise ValueError("Series must have at least 10 data points for ADF test")
    
    logger.debug(f"Calculating ADF p-value with max_lag={max_lag}")
    
    # Drop NA values
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        raise ValueError("Series must have at least 10 non-NA data points for ADF test")
    
    try:
        # Perform ADF test
        result = adfuller(clean_series, maxlag=max_lag)
        p_value = result[1]
        
        logger.debug(f"ADF p-value: {p_value:.6f}")
        
        return p_value
    except Exception as e:
        logger.error(f"Error calculating ADF p-value: {e}")
        raise ValueError(f"Failed to calculate ADF p-value: {e}")


def ou_half_life(spread: pd.Series, cap: float = 100.0) -> float:
    """
    Calculate the Ornstein-Uhlenbeck half-life of mean reversion.
    
    Args:
        spread: Time series of spread values.
        cap: Upper bound on half-life in days to prevent extreme values.
        
    Returns:
        Half-life of mean reversion in days.
        
    Raises:
        ValueError: If series has insufficient data points.
    """
    if len(spread) < 10:
        raise ValueError("Spread series must have at least 10 data points")
    
    logger.debug(f"Calculating OU half-life with cap={cap}")
    
    # Drop NA values
    clean_spread = spread.dropna()
    
    if len(clean_spread) < 10:
        raise ValueError("Spread series must have at least 10 non-NA data points")
    
    try:
        # Calculate lagged spread and spread changes
        lagged_spread = clean_spread.shift(1).dropna()
        spread_changes = clean_spread.diff().dropna()
        
        # Align series
        df = pd.DataFrame({
            "spread": clean_spread,
            "lagged_spread": lagged_spread,
            "spread_changes": spread_changes
        }).dropna()
        
        if len(df) < 10:
            raise ValueError("Insufficient data points after alignment")
        
        # Calculate regression coefficient
        cov = df["spread_changes"].cov(df["lagged_spread"])
        var = df["lagged_spread"].var()
        
        if var == 0:
            logger.warning("Zero variance in lagged spread, returning large half-life")
            return cap
        
        # OU process coefficient (negative mean reversion speed)
        lambda_ = cov / var
        
        # Calculate half-life
        if lambda_ >= 0:
            # No mean reversion (positive lambda)
            logger.warning("Positive lambda detected, no mean reversion")
            return cap
        
        half_life = -np.log(2) / lambda_
        
        # Cap the half-life to prevent extreme values
        half_life = min(half_life, cap)
        
        logger.debug(f"OU half-life: {half_life:.2f} days")
        
        return half_life
    except Exception as e:
        logger.error(f"Error calculating OU half-life: {e}")
        raise ValueError(f"Failed to calculate OU half-life: {e}")


def hurst_exponent(series: pd.Series, max_lags: int = 20) -> float:
    """
    Calculate the Hurst exponent to determine if a series is mean-reverting, random, or trending.
    
    Args:
        series: Time series to analyze.
        max_lags: Maximum number of lags to use in calculation.
        
    Returns:
        Hurst exponent value.
        
    Raises:
        ValueError: If series has insufficient data points.
    """
    if len(series) < 100:
        raise ValueError("Series must have at least 100 data points for Hurst exponent")
    
    logger.debug(f"Calculating Hurst exponent with max_lags={max_lags}")
    
    # Drop NA values
    clean_series = series.dropna()
    
    if len(clean_series) < 100:
        raise ValueError("Series must have at least 100 non-NA data points for Hurst exponent")
    
    try:
        # Calculate range of lags
        lags = range(2, min(max_lags, len(clean_series) // 4))
        
        # Calculate variance of differences for each lag
        tau = [np.var(np.subtract(clean_series[lag:], clean_series[:-lag])) for lag in lags]
        
        # Calculate log-log slope
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        # Linear regression to find slope
        slope = np.polyfit(log_lags, log_tau, 1)[0]
        
        # Hurst exponent is half the slope
        hurst = slope / 2
        
        logger.debug(f"Hurst exponent: {hurst:.4f}")
        
        return hurst
    except Exception as e:
        logger.error(f"Error calculating Hurst exponent: {e}")
        raise ValueError(f"Failed to calculate Hurst exponent: {e}")


def is_cointegrated(
    series_y: pd.Series, 
    series_x: pd.Series, 
    adf_threshold: float = 0.05,
    max_lag: int = 1
) -> Tuple[bool, float, pd.Series]:
    """
    Test if two series are cointegrated.
    
    Args:
        series_y: First time series.
        series_x: Second time series.
        adf_threshold: Threshold p-value for cointegration test.
        max_lag: Maximum lag for the ADF test.
        
    Returns:
        Tuple of (is_cointegrated, p_value, spread).
        
    Raises:
        ValueError: If series have insufficient data points or mismatched lengths.
    """
    if len(series_y) != len(series_x):
        raise ValueError("Series must have the same length")
    
    logger.debug(f"Testing cointegration with ADF threshold={adf_threshold}")
    
    # Align series and drop NA values
    df = pd.DataFrame({"y": series_y, "x": series_x}).dropna()
    
    if len(df) < 20:
        raise ValueError("Need at least 20 aligned data points for cointegration test")
    
    try:
        # Calculate beta using OLS
        cov = df["y"].cov(df["x"])
        var = df["x"].var()
        
        if var == 0:
            logger.warning("Zero variance in independent variable")
            return False, 1.0, pd.Series()
        
        beta = cov / var
        
        # Calculate spread
        spread = df["y"] - beta * df["x"]
        
        # Test spread for stationarity
        p_value = adf_pvalue(spread, max_lag)
        
        # Determine if cointegrated
        is_coint = p_value < adf_threshold
        
        logger.debug(f"Cointegration test: p-value={p_value:.6f}, is_cointegrated={is_coint}")
        
        return is_coint, p_value, spread
    except Exception as e:
        logger.error(f"Error testing cointegration: {e}")
        raise ValueError(f"Failed to test cointegration: {e}")