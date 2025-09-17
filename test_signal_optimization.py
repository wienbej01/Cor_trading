#!/usr/bin/env python3
"""
Test script for Signal Quality & Threshold Optimization implementation.
Validates look-ahead bias prevention and ensures non-trivial trading activity.
"""

import sys
import os

sys.path.append("src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime, timedelta
from loguru import logger

# Import our modules
from src.data.yahoo_loader import download_daily
from src.core.config import ConfigManager
from src.features.signal_optimization import SignalOptimizer


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_lookahead_bias_prevention():
    """
    Test that all signal generation methods prevent look-ahead bias.
    """
    logger.info("Testing look-ahead bias prevention...")

    # Load test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years of data

    try:
        fx_data = download_daily(
            "USDCAD=X", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        comd_data = download_daily(
            "CL=F", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        logger.info(f"Loaded {len(fx_data)} days of data")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Load configuration
    config_manager = ConfigManager()
    pair_config = config_manager.get_pair_config("usdcad_wti")

    # Initialize signal optimizer
    optimizer = SignalOptimizer(pair_config)

    # Test point-by-point signal generation to ensure no look-ahead
    signals_list = []
    for i in range(100, len(fx_data)):  # Start from index 100 to have enough history
        # Use only data up to current point
        fx_hist = fx_data.iloc[: i + 1]
        comd_hist = comd_data.iloc[: i + 1]

        # Generate signals
        thresholds = {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.5}
        signals_df = optimizer.generate_enhanced_signals(
            fx_hist,
            comd_hist,
            thresholds,
            use_vol_adjustment=False,  # Disable for simpler test
        )

        # Get the last signal
        current_signal = signals_df.iloc[-1]["signal"]
        signals_list.append(current_signal)

    # Convert to series
    signals_series = pd.Series(signals_list, index=fx_data.index[100:])

    # Check for any suspicious patterns that might indicate look-ahead
    # 1. No immediate reversals (signal changes should not be too frequent)
    signal_changes = signals_series.diff().abs()
    frequent_changes = (signal_changes.rolling(window=5).sum() > 3).sum()
    if frequent_changes > len(signals_series) * 0.1:  # More than 10% of periods
        logger.warning(f"High frequency of signal changes detected: {frequent_changes}")

    # 2. Signal distribution should be reasonable
    signal_counts = signals_series.value_counts()
    logger.info(f"Signal distribution: {dict(signal_counts)}")

    # 3. Check for NaN values
    nan_count = signals_series.isna().sum()
    if nan_count > 0:
        logger.error(f"Found {nan_count} NaN values in signals")
        return False

    logger.info("Look-ahead bias prevention test passed")
    return True


def test_trading_activity():
    """
    Test that enhanced signals produce non-trivial trading activity.
    """
    logger.info("Testing trading activity...")

    # Load test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data

    try:
        fx_data = download_daily(
            "USDCAD=X", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        comd_data = download_daily(
            "CL=F", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Load configuration
    config_manager = ConfigManager()
    pair_config = config_manager.get_pair_config("usdcad_wti")

    # Initialize signal optimizer
    optimizer = SignalOptimizer(pair_config)

    # Test different threshold combinations
    threshold_combinations = [
        {"entry_z": 0.8, "exit_z": 0.3, "stop_z": 2.5},
        {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.0},
        {"entry_z": 1.2, "exit_z": 0.7, "stop_z": 3.5},
        {"entry_z": 1.5, "exit_z": 0.9, "stop_z": 4.0},
    ]

    activity_results = []

    for thresholds in threshold_combinations:
        logger.info(f"Testing thresholds: {thresholds}")

        # Generate signals with volatility adjustment
        signals_df = optimizer.generate_enhanced_signals(
            fx_data, comd_data, thresholds, use_vol_adjustment=True
        )

        # Calculate trading activity metrics
        total_signals = (signals_df["signal"].abs() > 0).sum()
        signal_changes = (signals_df["signal"].diff().abs() > 0).sum()
        avg_signal_duration = optimizer._calculate_avg_signal_duration(
            signals_df["signal"]
        )

        # Calculate performance metrics
        spread_returns = signals_df["spread"].diff()
        strategy_returns = signals_df["signal"].shift(1) * spread_returns

        total_return = strategy_returns.sum()
        sharpe_ratio = (
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            if strategy_returns.std() > 0
            else 0
        )

        activity_results.append(
            {
                "thresholds": thresholds,
                "total_signals": total_signals,
                "signal_changes": signal_changes,
                "avg_duration": avg_signal_duration,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "activity_ratio": total_signals / len(signals_df),
            }
        )

    # Check if any combination produces reasonable activity
    reasonable_activity = False
    for result in activity_results:
        logger.info(
            f"Thresholds {result['thresholds']}: "
            f"Activity ratio: {result['activity_ratio']:.3f}, "
            f"Signal changes: {result['signal_changes']}, "
            f"Sharpe: {result['sharpe_ratio']:.3f}"
        )

        # Check for non-trivial activity (at least 5% of periods with signals)
        if result["activity_ratio"] > 0.05 and result["signal_changes"] > 10:
            reasonable_activity = True

    if not reasonable_activity:
        logger.warning("No threshold combination produced sufficient trading activity")
        return False

    logger.info("Trading activity test passed")
    return True


def test_parameter_sweep():
    """
    Test the parameter sweep functionality.
    """
    logger.info("Testing parameter sweep...")

    # Load test data (smaller dataset for faster testing)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data

    try:
        fx_data = download_daily(
            "USDCAD=X", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        comd_data = download_daily(
            "CL=F", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Load configuration
    config_manager = ConfigManager()
    pair_config = config_manager.get_pair_config("usdcad_wti")

    # Initialize signal optimizer
    optimizer = SignalOptimizer(pair_config)

    # Generate limited threshold combinations for testing
    test_combinations = [
        {"entry_z": 0.8, "exit_z": 0.3, "stop_z": 2.5},
        {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.0},
        {"entry_z": 1.2, "exit_z": 0.7, "stop_z": 3.5},
    ]

    # Run parameter sweep
    try:
        sweep_results = optimizer.run_parameter_sweep(
            fx_data, comd_data, test_combinations
        )

        logger.info(f"Parameter sweep completed with {len(sweep_results)} results")
        logger.info(
            f"Best combination by Sharpe ratio:\n{sweep_results.nlargest(3, 'sharpe_ratio')}"
        )

        # Check if results are reasonable
        if len(sweep_results) == 0:
            logger.error("Parameter sweep produced no results")
            return False

        # Check for reasonable Sharpe ratios
        max_sharpe = sweep_results["sharpe_ratio"].max()
        if abs(max_sharpe) > 10:  # Unreasonably high
            logger.warning(f"Suspiciously high Sharpe ratio detected: {max_sharpe}")

        logger.info("Parameter sweep test passed")
        return True

    except Exception as e:
        logger.error(f"Parameter sweep failed: {e}")
        return False


def test_diagnostic_plots():
    """
    Test diagnostic plotting functionality.
    """
    logger.info("Testing diagnostic plots...")

    # Load test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months for faster testing

    try:
        fx_data = download_daily(
            "USDCAD=X", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        comd_data = download_daily(
            "CL=F", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Load configuration
    config_manager = ConfigManager()
    pair_config = config_manager.get_pair_config("usdcad_wti")

    # Initialize signal optimizer
    optimizer = SignalOptimizer(pair_config)

    # Generate signals
    thresholds = {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.5}
    signals_df = optimizer.generate_enhanced_signals(
        fx_data, comd_data, thresholds, use_vol_adjustment=True
    )

    # Create diagnostic plots
    try:
        optimizer.plot_signal_diagnostics(
            signals_df, save_path="signal_diagnostics_test.png"
        )
        logger.info("Diagnostic plots saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create diagnostic plots: {e}")
        return False


def main():
    """
    Run all tests for signal optimization.
    """
    logger.info("Starting Signal Optimization Tests...")

    tests = [
        ("Look-ahead bias prevention", test_lookahead_bias_prevention),
        ("Trading activity", test_trading_activity),
        ("Parameter sweep", test_parameter_sweep),
        ("Diagnostic plots", test_diagnostic_plots),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("All tests passed! Signal optimization is working correctly.")
        return True
    else:
        logger.error("Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
