"""
Pytest configuration and common fixtures for FX-Commodity correlation arbitrage tests.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def sample_config() -> Dict[str, Any]:
    """
    Sample configuration dictionary for testing.

    Returns:
        Dictionary with sample configuration parameters.
    """
    return {
        "lookbacks": {"beta_window": 60, "z_window": 20, "corr_window": 30},
        "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        "sizing": {"target_vol_per_leg": 0.02, "atr_window": 14},
        "regime": {"min_abs_corr": 0.3},
        "use_kalman": True,
        "volatility_scaling": True,
    }


@pytest.fixture(scope="session")
def sample_fx_data() -> pd.Series:
    """
    Sample FX price data for testing.

    Returns:
        Series with sample FX price data.
    """
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

    # Generate realistic FX price movements
    base_price = 1.30  # Starting price for USD/CAD
    returns = np.random.normal(
        0.0005, 0.007, len(dates)
    )  # Daily returns with slight upward bias

    # Create price series
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    return pd.Series(prices, index=dates, name="USDCAD")


@pytest.fixture(scope="session")
def sample_commodity_data() -> pd.Series:
    """
    Sample commodity price data for testing.

    Returns:
        Series with sample commodity price data.
    """
    np.random.seed(123)  # Different seed for commodity data

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

    # Generate realistic commodity price movements (WTI)
    base_price = 60.0  # Starting price for WTI
    returns = np.random.normal(
        0.0003, 0.02, len(dates)
    )  # Higher volatility for commodities

    # Create price series
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    return pd.Series(prices, index=dates, name="WTI")


@pytest.fixture(scope="session")
def sample_volume_data() -> pd.Series:
    """
    Sample volume data for testing.

    Returns:
        Series with sample volume data.
    """
    np.random.seed(456)

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

    # Generate realistic volume patterns
    base_volume = 100000
    volume_noise = np.random.normal(0, 0.2, len(dates))
    volumes = [base_volume * (1 + noise) for noise in volume_noise]

    return pd.Series(volumes, index=dates, name="Volume")


@pytest.fixture(scope="session")
def sample_high_low_data() -> Tuple[pd.Series, pd.Series]:
    """
    Sample high and low price data for testing.

    Returns:
        Tuple of (high_prices, low_prices) series.
    """
    np.random.seed(789)

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    base_price = 1.30

    # Generate high and low prices based on close prices
    close_prices = [base_price]
    high_prices = []
    low_prices = []

    for i in range(1, len(dates)):
        # Generate close price
        daily_return = np.random.normal(0.0005, 0.007)
        close_price = close_prices[-1] * (1 + daily_return)
        close_prices.append(close_price)

        # Generate high and low with realistic spreads
        high_range = np.random.uniform(0.002, 0.008)
        low_range = np.random.uniform(0.002, 0.008)

        high_price = close_price * (1 + high_range)
        low_price = close_price * (1 - low_range)

        high_prices.append(high_price)
        low_prices.append(low_price)

    # Add first day values
    high_prices.insert(0, base_price * 1.005)
    low_prices.insert(0, base_price * 0.995)

    return (
        pd.Series(high_prices, index=dates, name="High"),
        pd.Series(low_prices, index=dates, name="Low"),
    )


@pytest.fixture(scope="session")
def sample_equity_curve() -> pd.Series:
    """
    Sample equity curve data for testing.

    Returns:
        Series with sample equity curve data.
    """
    np.random.seed(321)

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

    # Generate realistic equity curve with some drawdowns
    base_equity = 100000
    returns = np.random.normal(0.001, 0.015, len(dates))  # Positive expected return

    # Create equity series
    equity = [base_equity]
    for ret in returns[1:]:
        equity.append(equity[-1] * (1 + ret))

    return pd.Series(equity, index=dates, name="Equity")


@pytest.fixture(scope="session")
def sample_signals() -> pd.Series:
    """
    Sample trading signals for testing.

    Returns:
        Series with sample trading signals (-1, 0, 1).
    """
    np.random.seed(654)

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

    # Generate signals with some persistence
    signals = [0]  # Start with no position
    current_position = 0

    for i in range(1, len(dates)):
        # Add some persistence to signals
        if current_position != 0:
            # Higher probability of staying in position
            if np.random.random() < 0.8:
                signals.append(current_position)
            else:
                signals.append(0)
                current_position = 0
        else:
            # Random entry signals
            rand_val = np.random.random()
            if rand_val < 0.1:
                signals.append(1)
                current_position = 1
            elif rand_val < 0.2:
                signals.append(-1)
                current_position = -1
            else:
                signals.append(0)

    return pd.Series(signals, index=dates, name="Signals")


@pytest.fixture(scope="function")
def clean_logger():
    """
    Fixture to clean up logger handlers between tests.
    """
    import loguru

    # Remove all handlers
    loguru.logger.remove()

    # Add a null handler for tests
    loguru.logger.add(lambda _: None, level="DEBUG")

    yield

    # Clean up again after test
    loguru.logger.remove()


@pytest.fixture(scope="session")
def feature_data() -> pd.DataFrame:
    """
    Sample feature data for ML model testing.

    Returns:
        DataFrame with sample features.
    """
    np.random.seed(987)

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    n_samples = len(dates)

    # Generate various features
    data = {
        "feature_1": np.random.normal(0, 1, n_samples),
        "feature_2": np.random.normal(5, 2, n_samples),
        "feature_3": np.random.exponential(1, n_samples),
        "feature_4": np.random.uniform(-1, 1, n_samples),
        "feature_5": np.random.beta(2, 5, n_samples),
    }

    # Add some time-based features
    data["day_of_week"] = [d.dayofweek for d in dates]
    data["month"] = [d.month for d in dates]

    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def target_data() -> pd.Series:
    """
    Sample target data for ML model testing.

    Returns:
        Series with sample target values.
    """
    np.random.seed(246)

    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    n_samples = len(dates)

    # Generate target with some relationship to features
    target = np.random.normal(0, 0.1, n_samples)

    return pd.Series(target, index=dates, name="Target")


@pytest.fixture(scope="function")
def temp_directory(tmp_path_factory):
    """
    Fixture to create a temporary directory for file-based tests.

    Returns:
        Path to temporary directory.
    """
    return tmp_path_factory.mktemp("test_data")


# Custom markers for test categorization
def pytest_configure(config):
    """
    Configure custom markers for pytest.
    """
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line(
        "markers", "signal_optimization: marks tests for signal optimization module"
    )
    config.addinivalue_line(
        "markers", "regime_expansion: marks tests for regime expansion module"
    )
    config.addinivalue_line(
        "markers", "risk_execution: marks tests for risk and execution module"
    )
    config.addinivalue_line(
        "markers", "performance_metrics: marks tests for performance metrics module"
    )
    config.addinivalue_line(
        "markers", "model_diversification: marks tests for model diversification module"
    )
    config.addinivalue_line(
        "markers", "architecture: marks tests for architecture module"
    )
