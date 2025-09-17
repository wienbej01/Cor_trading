#!/usr/bin/env python3
"""
Unit tests for Performance Metrics & Short-Window Validation implementation.
Tests NaN handling, rolling window metrics, distribution analysis, and integration.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, "src")

from src.backtest.engine import (
    _safe_cagr,
    _safe_sharpe,
    _safe_sortino,
    _safe_max_drawdown,
    calculate_performance_metrics,
)
from src.backtest.rolling_metrics import (
    calculate_rolling_sharpe,
    calculate_rolling_sortino,
    calculate_rolling_max_drawdown,
    calculate_rolling_metrics,
)
from src.backtest.distribution_analysis import (
    calculate_skewness,
    calculate_kurtosis,
    calculate_var,
    calculate_cvar,
    calculate_distribution_metrics,
    analyze_return_distribution,
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test suite for performance metrics functions."""

    def setUp(self):
        """Set up test data."""
        # Create sample equity curve
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        self.equity = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)), index=dates
        )

        # Create sample trade returns
        self.trade_returns = pd.Series(np.random.normal(0.01, 0.05, 50))

    def test_safe_cagr(self):
        """Test safe CAGR calculation."""
        # Normal case
        cagr = _safe_cagr(self.equity)
        self.assertIsInstance(cagr, float)
        self.assertGreaterEqual(cagr, -1.0)  # CAGR should be >= -100%

        # Edge case: insufficient data
        short_series = self.equity[:1]
        cagr_short = _safe_cagr(short_series, min_periods=10)
        self.assertEqual(cagr_short, 0.0)

    def test_safe_sharpe(self):
        """Test safe Sharpe ratio calculation."""
        returns = self.equity.pct_change().fillna(0)

        # Normal case
        sharpe = _safe_sharpe(returns)
        self.assertIsInstance(sharpe, float)

        # Edge case: insufficient data
        short_returns = returns[:1]
        sharpe_short = _safe_sharpe(short_returns, min_periods=10)
        self.assertEqual(sharpe_short, 0.0)

    def test_safe_sortino(self):
        """Test safe Sortino ratio calculation."""
        returns = self.equity.pct_change().fillna(0)

        # Normal case
        sortino = _safe_sortino(returns)
        self.assertIsInstance(sortino, float)

        # Edge case: insufficient data
        short_returns = returns[:1]
        sortino_short = _safe_sortino(short_returns, min_periods=10)
        self.assertEqual(sortino_short, 0.0)

    def test_safe_max_drawdown(self):
        """Test safe maximum drawdown calculation."""
        # Normal case
        max_dd = _safe_max_drawdown(self.equity)
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0.0)  # Drawdown should be <= 0

        # Edge case: insufficient data
        short_equity = self.equity[:1]
        max_dd_short = _safe_max_drawdown(short_equity)
        self.assertEqual(max_dd_short, 0.0)

    def test_rolling_metrics(self):
        """Test rolling metrics calculations."""
        # Test rolling Sharpe
        rolling_sharpe = calculate_rolling_sharpe(self.equity, 30)
        self.assertIsInstance(rolling_sharpe, pd.Series)
        self.assertEqual(len(rolling_sharpe), len(self.equity))

        # Test rolling Sortino
        rolling_sortino = calculate_rolling_sortino(self.equity, 30)
        self.assertIsInstance(rolling_sortino, pd.Series)
        self.assertEqual(len(rolling_sortino), len(self.equity))

        # Test rolling max drawdown
        rolling_dd = calculate_rolling_max_drawdown(self.equity, 30)
        self.assertIsInstance(rolling_dd, pd.Series)
        self.assertEqual(len(rolling_dd), len(self.equity))

        # Test all rolling metrics
        rolling_metrics = calculate_rolling_metrics(self.equity)
        self.assertIsInstance(rolling_metrics, dict)
        self.assertIn("rolling_sharpe_30D", rolling_metrics)

    def test_distribution_analysis(self):
        """Test distribution analysis functions."""
        # Test skewness
        skewness = calculate_skewness(self.trade_returns)
        self.assertIsInstance(skewness, float)

        # Test kurtosis
        kurtosis = calculate_kurtosis(self.trade_returns)
        self.assertIsInstance(kurtosis, float)

        # Test VaR
        var_95 = calculate_var(self.trade_returns, 0.05)
        self.assertIsInstance(var_95, float)

        # Test CVaR
        cvar_95 = calculate_cvar(self.trade_returns, 0.05)
        self.assertIsInstance(cvar_95, float)

        # Test distribution metrics
        dist_metrics = calculate_distribution_metrics(self.trade_returns)
        self.assertIsInstance(dist_metrics, dict)
        self.assertIn("skewness", dist_metrics)
        self.assertIn("kurtosis", dist_metrics)
        self.assertIn("var_95", dist_metrics)
        self.assertIn("var_99", dist_metrics)
        self.assertIn("cvar_95", dist_metrics)
        self.assertIn("cvar_99", dist_metrics)

        # Test distribution analysis
        dist_analysis = analyze_return_distribution(self.trade_returns)
        self.assertIsInstance(dist_analysis, dict)
        self.assertIn("distribution_metrics", dist_analysis)
        self.assertIn("distribution_interpretation", dist_analysis)

    def test_performance_metrics_with_insufficient_trades(self):
        """Test performance metrics with insufficient trades."""
        # Create DataFrame with minimal data
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "equity": [1.0, 1.01, 1.02, 1.01, 1.03],
                "total_pnl": [0.0, 0.01, 0.01, -0.01, 0.02],
                "delayed_signal": [0, 0, 0, 0, 0],
            },
            index=dates,
        )

        # Add required columns
        df["trade_id"] = 0

        # Calculate metrics with high minimum trade count
        metrics = calculate_performance_metrics(df, min_trade_count=10)

        # Check that metrics are properly defaulted
        self.assertEqual(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["sortino_ratio"], 0.0)
        self.assertEqual(metrics["annual_return"], 0.0)
        self.assertEqual(metrics["win_rate"], 0)
        self.assertEqual(metrics["profit_factor"], 0)

    def test_performance_metrics_with_sufficient_trades(self):
        """Test performance metrics with sufficient trades."""
        # Create DataFrame with sufficient data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        equity_values = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))

        df = pd.DataFrame(
            {
                "equity": equity_values,
                "total_pnl": np.random.normal(0.01, 0.05, 100),
                "delayed_signal": np.concatenate(
                    [np.ones(50), np.zeros(50)]
                ),  # 50 trades
            },
            index=dates,
        )

        # Add trade IDs
        df["trade_id"] = np.concatenate(
            [np.repeat(i, 2) for i in range(25)] + [np.repeat(25, 50)]
        )

        # Calculate metrics with low minimum trade count
        metrics = calculate_performance_metrics(df, min_trade_count=10)

        # Check that metrics are calculated
        self.assertIsInstance(metrics["sharpe_ratio"], float)
        self.assertIsInstance(metrics["sortino_ratio"], float)
        self.assertIsInstance(metrics["annual_return"], float)


if __name__ == "__main__":
    unittest.main()
