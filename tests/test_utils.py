"""
Common test utilities for FX-Commodity correlation arbitrage tests.
Provides data generation functions, custom assertions, and mock objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings


def generate_synthetic_market_data(
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
    freq: str = "D",
    base_fx_price: float = 1.30,
    base_commodity_price: float = 60.0,
    fx_volatility: float = 0.007,
    commodity_volatility: float = 0.02,
    correlation: float = 0.3,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate synthetic market data for FX and commodity prices.

    Args:
        start_date: Start date for the data.
        end_date: End date for the data.
        freq: Frequency of the data.
        base_fx_price: Starting price for FX.
        base_commodity_price: Starting price for commodity.
        fx_volatility: Daily volatility for FX.
        commodity_volatility: Daily volatility for commodity.
        correlation: Correlation between FX and commodity returns.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (fx_series, commodity_series).
    """
    np.random.seed(seed)

    # Generate date range
    dates = pd.date_range(start_date, end_date, freq=freq)
    n_periods = len(dates)

    # Generate correlated returns using Cholesky decomposition
    mean_returns = [0.0005, 0.0003]  # Slight upward bias
    cov_matrix = np.array(
        [
            [fx_volatility**2, correlation * fx_volatility * commodity_volatility],
            [
                correlation * fx_volatility * commodity_volatility,
                commodity_volatility**2,
            ],
        ]
    )

    # Generate random returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)

    # Create price series
    fx_prices = [base_fx_price]
    commodity_prices = [base_commodity_price]

    for fx_ret, commodity_ret in returns[1:]:
        fx_prices.append(fx_prices[-1] * (1 + fx_ret))
        commodity_prices.append(commodity_prices[-1] * (1 + commodity_ret))

    fx_series = pd.Series(fx_prices, index=dates, name="FX")
    commodity_series = pd.Series(commodity_prices, index=dates, name="Commodity")

    return fx_series, commodity_series


def generate_synthetic_volume_data(
    dates: pd.DatetimeIndex,
    base_volume: float = 100000,
    volume_volatility: float = 0.2,
    seed: int = 456,
) -> pd.Series:
    """
    Generate synthetic volume data.

    Args:
        dates: Date index for the volume data.
        base_volume: Base volume level.
        volume_volatility: Volatility of volume changes.
        seed: Random seed for reproducibility.

    Returns:
        Series with synthetic volume data.
    """
    np.random.seed(seed)

    # Generate volume with random noise
    volume_noise = np.random.normal(0, volume_volatility, len(dates))
    volumes = [base_volume * (1 + noise) for noise in volume_noise]

    # Ensure positive volumes
    volumes = [max(vol, base_volume * 0.1) for vol in volumes]

    return pd.Series(volumes, index=dates, name="Volume")


def generate_synthetic_high_low_data(
    close_prices: pd.Series, avg_spread_pct: float = 0.005, seed: int = 789
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate synthetic high and low prices from close prices.

    Args:
        close_prices: Series of close prices.
        avg_spread_pct: Average spread as percentage of close price.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (high_prices, low_prices).
    """
    np.random.seed(seed)

    high_prices = []
    low_prices = []

    for close_price in close_prices:
        # Generate random spreads
        high_spread = np.random.uniform(0.001, avg_spread_pct * 2)
        low_spread = np.random.uniform(0.001, avg_spread_pct * 2)

        high_price = close_price * (1 + high_spread)
        low_price = close_price * (1 - low_spread)

        high_prices.append(high_price)
        low_prices.append(low_price)

    high_series = pd.Series(high_prices, index=close_prices.index, name="High")
    low_series = pd.Series(low_prices, index=close_prices.index, name="Low")

    return high_series, low_series


def generate_synthetic_signals(
    dates: pd.DatetimeIndex,
    entry_probability: float = 0.1,
    exit_probability: float = 0.8,
    persistence: float = 0.8,
    seed: int = 654,
) -> pd.Series:
    """
    Generate synthetic trading signals.

    Args:
        dates: Date index for the signals.
        entry_probability: Probability of entering a position.
        exit_probability: Probability of exiting when in position.
        persistence: Probability of maintaining current position.
        seed: Random seed for reproducibility.

    Returns:
        Series with synthetic trading signals (-1, 0, 1).
    """
    np.random.seed(seed)

    signals = [0]  # Start with no position
    current_position = 0

    for i in range(1, len(dates)):
        if current_position != 0:
            # Higher probability of staying in position
            if np.random.random() < persistence:
                signals.append(current_position)
            else:
                signals.append(0)
                current_position = 0
        else:
            # Random entry signals
            rand_val = np.random.random()
            if rand_val < entry_probability / 2:
                signals.append(1)
                current_position = 1
            elif rand_val < entry_probability:
                signals.append(-1)
                current_position = -1
            else:
                signals.append(0)

    return pd.Series(signals, index=dates, name="Signals")


def generate_synthetic_equity_curve(
    dates: pd.DatetimeIndex,
    base_equity: float = 100000,
    annual_return: float = 0.10,
    annual_volatility: float = 0.15,
    max_drawdown: float = 0.20,
    seed: int = 321,
) -> pd.Series:
    """
    Generate synthetic equity curve with realistic characteristics.

    Args:
        dates: Date index for the equity curve.
        base_equity: Starting equity value.
        annual_return: Expected annual return.
        annual_volatility: Annual volatility.
        max_drawdown: Maximum drawdown to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Series with synthetic equity curve.
    """
    np.random.seed(seed)

    # Convert annual parameters to daily
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # Generate random returns
    returns = np.random.normal(daily_return, daily_volatility, len(dates))

    # Create equity series
    equity = [base_equity]
    for ret in returns:
        equity.append(equity[-1] * (1 + ret))

    # Simulate a drawdown period
    drawdown_start = len(dates) // 3
    drawdown_length = len(dates) // 6
    drawdown_intensity = max_drawdown / drawdown_length

    for i in range(drawdown_start, min(drawdown_start + drawdown_length, len(equity))):
        equity[i] *= 1 - drawdown_intensity

    return pd.Series(equity[: len(dates)], index=dates, name="Equity")


def generate_synthetic_feature_data(
    dates: pd.DatetimeIndex,
    n_features: int = 5,
    feature_types: List[str] = None,
    seed: int = 987,
) -> pd.DataFrame:
    """
    Generate synthetic feature data for ML models.

    Args:
        dates: Date index for the features.
        n_features: Number of features to generate.
        feature_types: List of feature types ('normal', 'uniform', 'exponential', 'beta').
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic features.
    """
    np.random.seed(seed)

    if feature_types is None:
        feature_types = ["normal", "uniform", "exponential", "beta", "gamma"]

    n_samples = len(dates)
    data = {}

    for i in range(n_features):
        feature_type = feature_types[i % len(feature_types)]

        if feature_type == "normal":
            data[f"feature_{i+1}"] = np.random.normal(0, 1, n_samples)
        elif feature_type == "uniform":
            data[f"feature_{i+1}"] = np.random.uniform(-1, 1, n_samples)
        elif feature_type == "exponential":
            data[f"feature_{i+1}"] = np.random.exponential(1, n_samples)
        elif feature_type == "beta":
            data[f"feature_{i+1}"] = np.random.beta(2, 5, n_samples)
        elif feature_type == "gamma":
            data[f"feature_{i+1}"] = np.random.gamma(2, 2, n_samples)

    # Add time-based features
    data["day_of_week"] = [d.dayofweek for d in dates]
    data["month"] = [d.month for d in dates]
    data["quarter"] = [d.quarter for d in dates]

    return pd.DataFrame(data, index=dates)


def generate_synthetic_target_data(
    dates: pd.DatetimeIndex,
    feature_data: pd.DataFrame = None,
    noise_level: float = 0.1,
    seed: int = 246,
) -> pd.Series:
    """
    Generate synthetic target data with optional relationship to features.

    Args:
        dates: Date index for the target.
        feature_data: Feature data to create relationship with.
        noise_level: Level of noise to add to the target.
        seed: Random seed for reproducibility.

    Returns:
        Series with synthetic target values.
    """
    np.random.seed(seed)

    n_samples = len(dates)

    if feature_data is not None and len(feature_data) > 0:
        # Create target with relationship to first few features
        feature_cols = [
            col for col in feature_data.columns if col.startswith("feature_")
        ][:3]
        target = np.zeros(n_samples)

        for i, col in enumerate(feature_cols):
            target += (i + 1) * feature_data[col].values * 0.1
    else:
        # Generate random target
        target = np.random.normal(0, 0.1, n_samples)

    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    target += noise

    return pd.Series(target, index=dates, name="Target")


class CustomAssertions:
    """Custom assertion methods for financial calculations."""

    @staticmethod
    def assert_series_close(
        actual: pd.Series,
        expected: pd.Series,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: str = None,
    ) -> None:
        """
        Assert that two series are close within tolerance.

        Args:
            actual: Actual series.
            expected: Expected series.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            msg: Custom error message.
        """
        if actual.shape != expected.shape:
            raise AssertionError(
                f"Series shapes differ: {actual.shape} vs {expected.shape}"
            )

        # Align series
        actual_aligned, expected_aligned = actual.align(expected, join="inner")

        if len(actual_aligned) != len(actual) or len(expected_aligned) != len(expected):
            warnings.warn("Series indices don't match perfectly, using aligned subset")

        # Check values
        close = np.isclose(
            actual_aligned.values, expected_aligned.values, rtol=rtol, atol=atol
        )
        if not close.all():
            diff_mask = ~close
            diff_indices = actual_aligned.index[diff_mask]
            diff_values = list(
                zip(
                    actual_aligned[diff_mask].values, expected_aligned[diff_mask].values
                )
            )

            error_msg = msg or f"Series values differ at {len(diff_indices)} positions"
            error_msg += (
                f"\nFirst few differences at indices: {diff_indices[:5].tolist()}"
            )
            error_msg += f"\nFirst few different values: {diff_values[:5]}"

            raise AssertionError(error_msg)

    @staticmethod
    def assert_financial_quantity(
        value: float, expected_range: Tuple[float, float], msg: str = None
    ) -> None:
        """
        Assert that a financial quantity is within expected range.

        Args:
            value: Actual value.
            expected_range: Tuple of (min, max) expected values.
            msg: Custom error message.
        """
        min_val, max_val = expected_range

        if not (min_val <= value <= max_val):
            error_msg = (
                msg
                or f"Financial quantity {value} not in expected range [{min_val}, {max_val}]"
            )
            raise AssertionError(error_msg)

    @staticmethod
    def assert_no_extreme_values(
        series: pd.Series, threshold: float = 10.0, msg: str = None
    ) -> None:
        """
        Assert that a series contains no extreme values.

        Args:
            series: Series to check.
            threshold: Threshold for extreme values (in standard deviations).
            msg: Custom error message.
        """
        if len(series) == 0:
            return

        mean = series.mean()
        std = series.std()

        if std == 0:
            return

        z_scores = np.abs((series - mean) / std)
        extreme_mask = z_scores > threshold

        if extreme_mask.any():
            extreme_count = extreme_mask.sum()
            extreme_indices = series.index[extreme_mask][:5].tolist()
            extreme_values = series[extreme_mask][:5].tolist()

            error_msg = (
                msg or f"Found {extreme_count} extreme values (z-score > {threshold})"
            )
            error_msg += f"\nFirst few extreme values at indices: {extreme_indices}"
            error_msg += f"\nFirst few extreme values: {extreme_values}"

            raise AssertionError(error_msg)

    @staticmethod
    def assert_monotonic(
        series: pd.Series, increasing: bool = True, msg: str = None
    ) -> None:
        """
        Assert that a series is monotonic.

        Args:
            series: Series to check.
            increasing: Whether series should be increasing (True) or decreasing (False).
            msg: Custom error message.
        """
        if len(series) <= 1:
            return

        diff = series.diff().dropna()

        if increasing:
            violations = (diff < 0).sum()
        else:
            violations = (diff > 0).sum()

        if violations > 0:
            error_msg = (
                msg
                or f"Series is not monotonic {'increasing' if increasing else 'decreasing'}"
            )
            error_msg += (
                f"\nFound {violations} violations out of {len(diff)} differences"
            )

            raise AssertionError(error_msg)


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, data: Dict[str, pd.DataFrame] = None):
        """
        Initialize mock data loader.

        Args:
            data: Dictionary mapping symbols to dataframes.
        """
        self.data = data or {}

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Mock method to load data.

        Args:
            symbol: Symbol to load.
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with mock data.
        """
        if symbol in self.data:
            return self.data[symbol]
        else:
            # Generate synthetic data if not available
            dates = pd.date_range(start_date, end_date)
            return pd.DataFrame(
                {
                    "Open": np.random.uniform(90, 110, len(dates)),
                    "High": np.random.uniform(95, 115, len(dates)),
                    "Low": np.random.uniform(85, 105, len(dates)),
                    "Close": np.random.uniform(90, 110, len(dates)),
                    "Volume": np.random.uniform(1000000, 5000000, len(dates)),
                },
                index=dates,
            )


class MockMarketSimulator:
    """Mock market simulator for testing."""

    def __init__(self, price_data: Dict[str, pd.Series] = None):
        """
        Initialize mock market simulator.

        Args:
            price_data: Dictionary mapping symbols to price series.
        """
        self.price_data = price_data or {}
        self.current_time = None

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Symbol to get price for.

        Returns:
            Current price.
        """
        if symbol not in self.price_data:
            raise ValueError(f"Symbol {symbol} not found in price data")

        if self.current_time is None:
            return self.price_data[symbol].iloc[-1]

        try:
            return self.price_data[symbol].loc[self.current_time]
        except KeyError:
            # Use last available price if current time not found
            return self.price_data[symbol].iloc[-1]

    def set_current_time(self, time: datetime) -> None:
        """
        Set current simulation time.

        Args:
            time: Current time.
        """
        self.current_time = time
