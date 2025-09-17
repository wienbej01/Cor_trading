"""
Broker API integration module for multi-timeframe data fetching.
Provides unified interface for H1 data retrieval with rate limiting and fallbacks.
"""

from typing import Dict, Optional, List
import asyncio
import time
import pandas as pd
from loguru import logger
import os

# Rate limiting: 5 calls per minute for free tier
RATE_LIMIT_CALLS = 5
RATE_LIMIT_WINDOW = 60  # seconds
_call_times: List[float] = []


def _check_rate_limit() -> bool:
    """Check if we're within rate limits."""

    # Clean old calls
    current_time = time.time()
    _call_times = [t for t in _call_times if current_time - t < RATE_LIMIT_WINDOW]

    # Check if we can make another call
    if len(_call_times) >= RATE_LIMIT_CALLS:
        return False

    return True


def _record_call():
    """Record a successful API call."""
    _call_times.append(time.time())


def _wait_for_rate_limit():
    """Wait until we can make another API call."""

    if not _check_rate_limit():
        # Calculate wait time
        oldest_call = min(_call_times)
        wait_time = RATE_LIMIT_WINDOW - (time.time() - oldest_call)

        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)


class BrokerAPI:
    """
    Unified broker API for multi-timeframe data fetching.
    Supports Polygon.io and Interactive Brokers with automatic fallbacks.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize broker API client.

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.polygon_available = self._check_polygon_availability()
        self.ibkr_available = self._check_ibkr_availability()

        logger.info(
            f"BrokerAPI initialized - Polygon: {self.polygon_available}, IBKR: {self.ibkr_available}"
        )

    def _check_polygon_availability(self) -> bool:
        """Check if Polygon.io is available and configured."""
        try:
            # Check for API key in environment or config
            api_key = self.config.get("polygon_api_key") or os.getenv("POLYGON_API_KEY")
            return api_key is not None
        except Exception:
            return False

    def _check_ibkr_availability(self) -> bool:
        """Check if Interactive Brokers is available."""
        try:
            # This would check IBKR connection in a real implementation
            return False  # Placeholder - assume not available for now
        except Exception:
            return False

    async def get_h1_data_async(
        self, symbol: str, start_date: str, end_date: str, max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch H1 (hourly) data asynchronously with rate limiting.

        Args:
            symbol: Trading symbol (e.g., "AAPL", "CL=F")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            max_retries: Maximum retry attempts

        Returns:
            DataFrame with H1 OHLCV data
        """
        logger.info(f"Fetching H1 data for {symbol} from {start_date} to {end_date}")

        # Try Polygon.io first (if available)
        if self.polygon_available:
            try:
                return await self._get_polygon_h1_data(
                    symbol, start_date, end_date, max_retries
                )
            except Exception as e:
                logger.warning(f"Polygon.io H1 data failed for {symbol}: {e}")

        # Fallback to Interactive Brokers (if available)
        if self.ibkr_available:
            try:
                return await self._get_ibkr_h1_data(symbol, start_date, end_date)
            except Exception as e:
                logger.warning(f"IBKR H1 data failed for {symbol}: {e}")

        # Final fallback: resample from daily data
        logger.info(f"Using daily data resampling for {symbol} H1 data")
        return await self._resample_daily_to_h1(symbol, start_date, end_date)

    async def _get_polygon_h1_data(
        self, symbol: str, start_date: str, end_date: str, max_retries: int
    ) -> pd.DataFrame:
        """
        Get H1 data from Polygon.io with rate limiting.
        Note: Polygon has limited commodity data, so this may not work for all symbols.
        """
        try:
            # Import here to avoid circular imports
            from trade_system_modules.data.polygon_adapter import get_agg_minute

            # Wait for rate limit if needed
            _wait_for_rate_limit()

            # Get minute data and resample to hourly
            minute_data = await get_agg_minute(
                symbol, start_date, end_date, concurrency=5
            )

            # Record the API call
            _record_call()

            # Resample minute data to hourly OHLCV
            h1_data = self._resample_minute_to_hourly(minute_data)

            logger.info(
                f"Retrieved {len(h1_data)} H1 bars from Polygon.io for {symbol}"
            )
            return h1_data

        except Exception as e:
            logger.error(f"Polygon.io H1 data error for {symbol}: {e}")
            raise

    async def _get_ibkr_h1_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get H1 data from Interactive Brokers.
        Placeholder implementation - would need IBKR integration.
        """
        # This would implement IBKR data fetching
        # For now, raise NotImplementedError
        raise NotImplementedError("IBKR H1 data fetching not yet implemented")

    async def _resample_daily_to_h1(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fallback: Get daily data and resample to H1.
        This provides basic H1 data when minute data is unavailable.
        """
        try:
            # Import existing yahoo loader for daily data
            from .yahoo_loader import download_daily

            # Get daily data
            daily_data = download_daily(symbol, start_date, end_date)

            # Resample daily to hourly (interpolate)
            h1_data = self._resample_daily_to_hourly(daily_data)

            logger.info(
                f"Resampled {len(daily_data)} daily bars to {len(h1_data)} H1 bars for {symbol}"
            )
            return h1_data

        except Exception as e:
            logger.error(f"Daily data fallback failed for {symbol}: {e}")
            raise

    def _resample_minute_to_hourly(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample minute OHLCV data to hourly.
        """
        # Ensure we have the right columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in minute_data.columns for col in required_cols):
            raise ValueError(f"Minute data missing required columns: {required_cols}")

        # Set timestamp as index if not already
        if "ts" in minute_data.columns:
            minute_data = minute_data.set_index("ts")

        # Resample to hourly
        hourly = (
            minute_data.resample("H")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        return hourly.reset_index()

    def _resample_daily_to_hourly(self, daily_data: pd.Series) -> pd.DataFrame:
        """
        Resample daily price data to hourly (interpolated).
        This is a simplified approach for when minute data is unavailable.
        """
        # Create hourly index
        start = pd.Timestamp(daily_data.index[0])
        end = pd.Timestamp(daily_data.index[-1]) + pd.Timedelta(days=1)
        hourly_index = pd.date_range(start=start, end=end, freq="H")

        # Interpolate daily data to hourly
        # This is a basic linear interpolation - not ideal but provides a fallback
        hourly_prices = daily_data.reindex(
            hourly_index.union(daily_data.index)
        ).interpolate(method="linear")

        # Create OHLCV structure (simplified - all OHLC same for interpolated data)
        h1_data = pd.DataFrame(
            {
                "ts": hourly_index,
                "open": hourly_prices.loc[hourly_index],
                "high": hourly_prices.loc[hourly_index],
                "low": hourly_prices.loc[hourly_index],
                "close": hourly_prices.loc[hourly_index],
                "volume": 0,  # No volume data available
            }
        )

        return h1_data.dropna()

    async def get_multi_symbol_h1_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch H1 data for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format

        Returns:
            Dictionary mapping symbols to their H1 DataFrames
        """
        logger.info(f"Fetching H1 data for {len(symbols)} symbols concurrently")

        # Create tasks for concurrent fetching
        tasks = []
        for symbol in symbols:
            task = self.get_h1_data_async(symbol, start_date, end_date)
            tasks.append((symbol, task))

        # Execute concurrently with rate limiting
        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                results[symbol] = data
                logger.info(
                    f"Successfully fetched H1 data for {symbol}: {len(data)} bars"
                )
            except Exception as e:
                logger.error(f"Failed to fetch H1 data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols

        return results


# Convenience functions
async def get_h1_data(
    symbol: str, start_date: str, end_date: str, config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Convenience function to get H1 data for a single symbol.

    Args:
        symbol: Trading symbol
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        config: Optional configuration dictionary

    Returns:
        DataFrame with H1 OHLCV data
    """
    api = BrokerAPI(config)
    return await api.get_h1_data_async(symbol, start_date, end_date)


async def get_multi_symbol_h1_data(
    symbols: List[str], start_date: str, end_date: str, config: Optional[Dict] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to get H1 data for multiple symbols.

    Args:
        symbols: List of trading symbols
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping symbols to their H1 DataFrames
    """
    api = BrokerAPI(config)
    return await api.get_multi_symbol_h1_data(symbols, start_date, end_date)


# Synchronous wrapper for backward compatibility
def get_h1_data_sync(
    symbol: str, start_date: str, end_date: str, config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Synchronous wrapper for get_h1_data.

    Args:
        symbol: Trading symbol
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        config: Optional configuration dictionary

    Returns:
        DataFrame with H1 OHLCV data
    """
    return asyncio.run(get_h1_data(symbol, start_date, end_date, config))
