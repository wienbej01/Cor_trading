"""
Yahoo Finance data loader module for FX-Commodity correlation arbitrage strategy.
Handles downloading and aligning financial time series data.
"""

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
from loguru import logger


def download_daily(symbol: str, start: str, end: str) -> pd.Series:
    """
    Download daily price data for a symbol from Yahoo Finance.
    
    Args:
        symbol: Financial instrument symbol (e.g., "USDCAD=X", "CL=F").
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        
    Returns:
        pandas Series with daily close prices, indexed by date.
        
    Raises:
        ValueError: If symbol is invalid or date range is invalid.
    """
    logger.info(f"Downloading daily data for {symbol} from {start} to {end}")
    
    try:
        # Validate date format
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol} in the specified date range")
            
        # Extract close prices and ensure proper index
        close_prices = data["Close"]
        
        # Remove timezone information if present to avoid alignment issues
        if close_prices.index.tz is not None:
            close_prices.index = close_prices.index.tz_localize(None)
            
        logger.info(f"Downloaded {len(close_prices)} data points for {symbol}")
        
        return close_prices
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        raise ValueError(f"Failed to download data for {symbol}: {e}")


def align_series(series_a: pd.Series, series_b: pd.Series, method: str = "inner") -> pd.DataFrame:
    """
    Align two time series based on their dates.
    
    Args:
        series_a: First time series.
        series_b: Second time series.
        method: Alignment method ("inner", "outer", "left", "right").
        
    Returns:
        DataFrame with aligned series, column names match input series names.
        
    Raises:
        ValueError: If series have overlapping dates but no common dates after alignment.
    """
    logger.debug(f"Aligning series with {len(series_a)} and {len(series_b)} points using {method} join")
    
    # Create DataFrame from series
    df = pd.DataFrame({
        series_a.name if series_a.name else "series_a": series_a,
        series_b.name if series_b.name else "series_b": series_b
    })
    
    # Align based on index (dates) - drop rows with NaN values in either column
    aligned_df = df.dropna(subset=[df.columns[0], df.columns[1]])
    
    if aligned_df.empty:
        raise ValueError("No overlapping dates found between the two series")
        
    logger.debug(f"Aligned series have {len(aligned_df)} common data points")
    
    return aligned_df


def download_and_align_pair(
    fx_symbol: str, 
    comd_symbol: str, 
    start: str, 
    end: str,
    fx_name: Optional[str] = None,
    comd_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Download and align FX and commodity data for a pair.
    
    Args:
        fx_symbol: FX symbol (e.g., "USDCAD=X").
        comd_symbol: Commodity symbol (e.g., "CL=F").
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        fx_name: Optional name for FX series (defaults to fx_symbol).
        comd_name: Optional name for commodity series (defaults to comd_symbol).
        
    Returns:
        DataFrame with aligned FX and commodity data.
    """
    logger.info(f"Downloading and aligning pair: {fx_symbol} and {comd_symbol}")
    
    # Download individual series
    fx_series = download_daily(fx_symbol, start, end)
    comd_series = download_daily(comd_symbol, start, end)
    
    # Set names if provided
    if fx_name:
        fx_series.name = fx_name
    if comd_name:
        comd_series.name = comd_name
        
    # Align series
    aligned_data = align_series(fx_series, comd_series)
    
    logger.info(f"Successfully aligned pair with {len(aligned_data)} data points")
    
    return aligned_data