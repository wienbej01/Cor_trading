"""
Yahoo Finance data loader module for FX-Commodity correlation arbitrage strategy.
Handles downloading and aligning financial time series data with comprehensive logging and error handling.
"""

from datetime import datetime
from typing import Optional
import time
import os

import pandas as pd
import yfinance as yf
from loguru import logger


def validate_date_format(date_str: str) -> None:
    """Validate date string format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD: {e}")


def validate_symbol(symbol: str) -> None:
    """Validate symbol format."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid symbol: {symbol}")
    if len(symbol.strip()) == 0:
        raise ValueError("Symbol cannot be empty")


def load_local_series(path: str) -> pd.Series:
    """
    Load a local CSV or Parquet file as a price Series.
    - Detects date index: uses 'date' column if present, else first column when it looks like dates, else tries index.
    - Detects price column: one of ['close','adj_close','adj close','Close','Adj Close','price','Price'].
    Returns a Series indexed by datetime.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq", ".parq"]:
        df = pd.read_parquet(path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type for local series: {ext}")

    # Normalize columns
    cols_lower = {c: str(c).lower() for c in df.columns}
    # Determine datetime index
    if "date" in cols_lower.values():
        date_col = [c for c, lc in cols_lower.items() if lc == "date"][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif df.shape[1] >= 1:
        # Try first column as date if convertible
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col])
            df = df.set_index(first_col)
        except Exception:
            # Assume index is already datetime-like
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # Find price/close column
    price_candidates = [
        "close",
        "adj_close",
        "adj close",
        "price",
        "last",
        "settle",
    ]
    chosen = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in price_candidates or "close" in lc or "price" in lc:
            chosen = c
            break
    if chosen is None:
        # If a single column, take it
        if df.shape[1] == 1:
            chosen = df.columns[0]
        else:
            raise ValueError(f"Could not detect price/close column in {path}. Columns: {list(df.columns)}")

    s = df[chosen].astype(float).copy()
    s.name = chosen
    # Dropna and forward-fill small gaps
    if s.isna().any():
        s = s.ffill()
    # De-tz if any
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def download_daily(
    symbol: str,
    start: str,
    end: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    local_path: Optional[str] = None,
) -> pd.Series:
    """
    Download daily price data for a symbol.

    Priority:
      1) If local_path is provided (or available via environment variables), try loading CSV/Parquet locally.
      2) Fallback to Yahoo Finance download with retries.

    Env support:
      - If local_path is None, will check environment variables:
          FX_LOCAL_PATH / COMD_LOCAL_PATH when called for FX/Commodity upstream
          YF_LOCAL_PATH as a generic fallback (single path)
    Args:
        symbol: Financial instrument symbol (e.g., "USDCAD=X", "CL=F").
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        max_retries: Maximum number of retry attempts for failed downloads.
        retry_delay: Delay between retry attempts in seconds.
        local_path: Optional path to CSV/Parquet file to load locally.

    Returns:
        pandas Series with daily close prices, indexed by date.
    """
    logger.info(f"Starting load for symbol: {symbol}, date range: {start} to {end}")

    # Validate inputs
    validate_symbol(symbol)
    validate_date_format(start)
    validate_date_format(end)

    if start >= end:
        raise ValueError(f"Invalid date range: start ({start}) must be before end ({end})")

    # Clean symbol
    symbol = symbol.strip().upper()

    # Local path resolution via environment, only if not explicitly provided
    if not local_path:
        # Generic single local file path fallback
        local_env = os.getenv("YF_LOCAL_PATH")
        if local_env and local_env.strip():
            local_path = local_env.strip()

    # Try local first if provided
    if local_path:
        try:
            logger.info(f"Attempting local load for {symbol} from: {local_path}")
            local_series = load_local_series(local_path)
            # Slice by date
            local_series = local_series.loc[
                (local_series.index >= pd.to_datetime(start)) & (local_series.index <= pd.to_datetime(end))
            ]
            if not local_series.empty:
                local_series.name = local_series.name or symbol
                logger.success(f"Loaded {len(local_series)} rows locally for {symbol}")
                return local_series
            else:
                logger.warning(f"Local file provided for {symbol} but yielded 0 rows in range; falling back to Yahoo")
        except Exception as e:
            logger.warning(f"Local load failed for {symbol} from {local_path}: {e}. Falling back to Yahoo")

    # Fallback to Yahoo
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Download attempt {attempt + 1}/{max_retries + 1} for {symbol}")

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end)

            if data.empty:
                logger.warning(f"No data found for symbol {symbol} in range {start} to {end}")
                raise ValueError(f"No data found for symbol {symbol} in the specified date range")

            if "Close" not in data.columns:
                logger.error(f"Invalid data structure for {symbol}: missing 'Close' column")
                raise ValueError(f"Invalid data structure for {symbol}: missing required columns")

            close_prices = data["Close"].copy()

            logger.info(f"Downloaded {len(close_prices)} data points for {symbol}")
            logger.debug(f"Date range: {close_prices.index.min()} to {close_prices.index.max()}")
            logger.debug(f"Missing values: {close_prices.isna().sum()}")

            if close_prices.isna().any():
                logger.warning(f"Found {close_prices.isna().sum()} missing values in {symbol}, filling forward")
                close_prices = close_prices.fillna(method="ffill")

            if getattr(close_prices.index, "tz", None) is not None:
                close_prices.index = close_prices.index.tz_localize(None)
                logger.debug("Removed timezone information from index")

            if close_prices.empty:
                raise ValueError("No valid data points after processing")

            logger.success(f"Successfully downloaded data for {symbol}")
            return close_prices

        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Download failed for {symbol} (attempt {attempt + 1}): {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"Failed to download data for {symbol} after {max_retries + 1} attempts: {e}")
                if "Connection" in str(e) or "Timeout" in str(e):
                    raise ConnectionError(f"Unable to connect to Yahoo Finance: {e}")
                else:
                    raise ValueError(f"Failed to download data for {symbol}: {e}")


def align_series(
    series_a: pd.Series, series_b: pd.Series, method: str = "inner"
) -> pd.DataFrame:
    """
    Align two time series based on their dates with comprehensive logging and validation.

    Args:
        series_a: First time series.
        series_b: Second time series.
        method: Alignment method ("inner", "outer", "left", "right").

    Returns:
        DataFrame with aligned series, column names match input series names.

    Raises:
        ValueError: If series have no overlapping dates or are invalid.
    """
    logger.info(
        f"Aligning series: {series_a.name} ({len(series_a)} points) and {series_b.name} ({len(series_b)} points)"
    )

    # Validate inputs
    if series_a.empty or series_b.empty:
        raise ValueError("Cannot align empty series")

    if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
        raise ValueError("Inputs must be pandas Series")

    # Ensure series have names
    name_a = series_a.name or "series_a"
    name_b = series_b.name or "series_b"

    try:
        # Create DataFrame from series
        df = pd.DataFrame({name_a: series_a, name_b: series_b})

        # Log alignment details
        logger.debug(
            f"Original date ranges: {name_a}: {series_a.index.min()} to {series_a.index.max()}"
        )
        logger.debug(
            f"Original date ranges: {name_b}: {series_b.index.min()} to {series_b.index.max()}"
        )

        # Align based on index (dates) - drop rows with NaN values in either column
        aligned_df = df.dropna(subset=[name_a, name_b])

        if aligned_df.empty:
            logger.error(f"No overlapping dates found between {name_a} and {name_b}")
            raise ValueError("No overlapping dates found between the two series")

        # Log alignment results
        logger.info(
            f"Successfully aligned series with {len(aligned_df)} common data points"
        )
        logger.debug(
            f"Aligned date range: {aligned_df.index.min()} to {aligned_df.index.max()}"
        )
        logger.debug(
            f"Aligned index has duplicates: {aligned_df.index.duplicated().any()}"
        )
        if aligned_df.index.duplicated().any():
            logger.warning(
                f"Duplicate dates in aligned data: {aligned_df.index[aligned_df.index.duplicated()].unique()}"
            )

        return aligned_df

    except Exception as e:
        logger.error(f"Error aligning series: {e}")
        raise


def download_and_align_pair(
    fx_symbol: str,
    comd_symbol: str,
    start: str,
    end: str,
    fx_name: Optional[str] = None,
    comd_name: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fx_local_path: Optional[str] = None,
    comd_local_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download and align FX and commodity data for a pair with comprehensive logging and validation.

    Args:
        fx_symbol: FX symbol (e.g., "USDCAD=X").
        comd_symbol: Commodity symbol (e.g., "CL=F").
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        fx_name: Optional name for FX series (defaults to fx_symbol).
        comd_name: Optional name for commodity series (defaults to comd_symbol).
        max_retries: Maximum retry attempts for failed downloads.
        retry_delay: Delay between retry attempts in seconds.
        fx_local_path: Optional local CSV/Parquet path for FX time series. If provided, used before Yahoo.
        comd_local_path: Optional local CSV/Parquet path for commodity. If provided, used before Yahoo.

    Returns:
        DataFrame with aligned FX and commodity data.
    """
    logger.info(f"Starting pair download: {fx_symbol} vs {comd_symbol}")
    logger.info(f"Date range: {start} to {end}")

    # Validate inputs
    validate_symbol(fx_symbol)
    validate_symbol(comd_symbol)
    validate_date_format(start)
    validate_date_format(end)

    if start >= end:
        raise ValueError(f"Invalid date range: start ({start}) must be before end ({end})")

    # Clean symbols
    fx_symbol = fx_symbol.strip().upper()
    comd_symbol = comd_symbol.strip().upper()

    # Set names
    fx_name = fx_name or fx_symbol
    comd_name = comd_name or comd_symbol

    # Environment fallback for local paths if not explicitly provided
    if not fx_local_path:
        fx_local_path = os.getenv("FX_LOCAL_PATH") or fx_local_path
    if not comd_local_path:
        comd_local_path = os.getenv("COMD_LOCAL_PATH") or comd_local_path

    try:
        logger.info(f"Loading FX data: {fx_symbol}")
        fx_series = download_daily(
            fx_symbol, start, end, max_retries, retry_delay, local_path=fx_local_path
        )
        fx_series.name = fx_name

        logger.info(f"Loading commodity data: {comd_symbol}")
        comd_series = download_daily(
            comd_symbol, start, end, max_retries, retry_delay, local_path=comd_local_path
        )
        comd_series.name = comd_name

        logger.info(
            f"FX data: {len(fx_series)} points from {fx_series.index.min()} to {fx_series.index.max()}"
        )
        logger.info(
            f"Commodity data: {len(comd_series)} points from {comd_series.index.min()} to {comd_series.index.max()}"
        )

        aligned_data = align_series(fx_series, comd_series)

        if aligned_data.empty:
            raise ValueError("No valid data points after alignment")

        logger.info("Successfully downloaded and aligned pair")
        logger.info(f"Final dataset: {len(aligned_data)} data points")
        logger.info(f"Date range: {aligned_data.index.min()} to {aligned_data.index.max()}")
        logger.debug(f"Columns: {list(aligned_data.columns)}")

        return aligned_data

    except Exception as e:
        logger.error(f"Error downloading and aligning pair {fx_symbol} vs {comd_symbol}: {e}")
        raise


# Convenience function for testing
def test_data_loading():
    """Test function to verify data loading functionality."""
    try:
        logger.info("Testing data loading functionality...")

        # Test with a small date range
        test_data = download_and_align_pair(
            fx_symbol="USDCAD=X",
            comd_symbol="CL=F",
            start="2024-01-01",
            end="2024-01-31",
        )

        logger.success("Data loading test successful!")
        logger.info(f"Test data shape: {test_data.shape}")
        logger.info(f"Test data columns: {list(test_data.columns)}")

        return test_data

    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test when module is executed directly
    test_data_loading()
