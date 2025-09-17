"""
Unit tests for the parallel backtesting framework.
"""

import unittest
import pandas as pd
import numpy as np
from src.backtest.parallel import (
    BacktestConfig,
    BacktestResult,
    ParallelBacktester,
    create_default_backtest_config,
    _run_single_backtest,
)


class TestBacktestConfig(unittest.TestCase):
    """Test cases for the backtest configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = create_default_backtest_config()
        self.assertIsInstance(config, BacktestConfig)

        # Check default values
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.timeout, 3600)
        self.assertIn("ols", config.models_to_test)
        self.assertIn("kalman", config.models_to_test)
        self.assertIn("corr", config.models_to_test)

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            max_workers=2,
            timeout=1800,
            models_to_test=["ols", "gb"],
            entry_z_range=[1.0, 1.5],
            exit_z_range=[0.5, 0.7],
            stop_z_range=[3.0, 4.0],
        )

        self.assertEqual(config.max_workers, 2)
        self.assertEqual(config.timeout, 1800)
        self.assertEqual(config.models_to_test, ["ols", "gb"])
        self.assertEqual(config.entry_z_range, [1.0, 1.5])


class TestBacktestResult(unittest.TestCase):
    """Test cases for the backtest result."""

    def test_backtest_result_creation(self):
        """Test backtest result creation."""
        # Create sample data
        equity_curve = pd.Series([1.0, 1.1, 1.2, 1.15, 1.3])
        signals = pd.DataFrame({"signal": [0, 1, 1, 0, -1]})
        metrics = {"sharpe_ratio": 1.5, "max_drawdown": -0.1}

        result = BacktestResult(
            model_name="test_model",
            params={"entry_z": 1.0, "exit_z": 0.5},
            metrics=metrics,
            equity_curve=equity_curve,
            signals=signals,
            execution_time=1.23,
        )

        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.params["entry_z"], 1.0)
        self.assertEqual(result.metrics["sharpe_ratio"], 1.5)
        self.assertEqual(len(result.equity_curve), 5)
        self.assertEqual(len(result.signals), 5)
        self.assertEqual(result.execution_time, 1.23)


class TestParallelBacktester(unittest.TestCase):
    """Test cases for the parallel backtester."""

    def setUp(self):
        """Set up test data."""
        # Create simple test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "fx_price": 1.3 + np.cumsum(np.random.randn(100) * 0.01),
                "comd_price": 50 + np.cumsum(np.random.randn(100) * 0.5),
                "spread": np.random.randn(100),
                "spread_z": np.random.randn(100),
                "signal": np.random.choice([-1, 0, 1], 100),
                "fx_position": np.random.randn(100),
                "comd_position": np.random.randn(100),
            },
            index=dates,
        )

        # Add required columns for backtesting
        self.data["delayed_signal"] = self.data["signal"].shift(1)
        self.data["delayed_fx_position"] = self.data["fx_position"].shift(1)
        self.data["delayed_comd_position"] = self.data["comd_position"].shift(1)
        self.data["fx_return"] = self.data["fx_price"].pct_change()
        self.data["comd_return"] = self.data["comd_price"].pct_change()
        self.data["fx_pnl"] = (
            self.data["delayed_fx_position"] * self.data["fx_price"].diff()
        )
        self.data["comd_pnl"] = (
            self.data["delayed_comd_position"] * self.data["comd_price"].diff()
        )
        self.data["total_pnl"] = self.data["fx_pnl"] + self.data["comd_pnl"]
        self.data["cumulative_pnl"] = self.data["total_pnl"].cumsum()
        self.data["equity"] = 1.0 + self.data["cumulative_pnl"]
        self.data["running_max"] = self.data["equity"].cummax()
        self.data["drawdown"] = (
            self.data["equity"] - self.data["running_max"]
        ) / self.data["running_max"]

        # Base config
        self.base_config = {
            "thresholds": {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.5},
            "time_stop": {"max_days": 10},
            "inverse_fx_for_quote_ccy_strength": True,
            "min_trade_count": 5,
        }

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        config = BacktestConfig(max_workers=2)
        backtester = ParallelBacktester(config)

        self.assertEqual(backtester.config.max_workers, 2)
        self.assertEqual(len(backtester.results), 0)

    def test_generate_param_combinations(self):
        """Test parameter combination generation."""
        config = BacktestConfig(
            entry_z_range=[1.0, 1.5], exit_z_range=[0.5], stop_z_range=[3.0, 4.0]
        )
        backtester = ParallelBacktester(config)

        combinations = backtester._generate_param_combinations()

        # Should generate 2 * 1 * 2 = 4 combinations with valid constraints
        # entry_z > exit_z and stop_z > entry_z
        valid_combinations = [
            combo
            for combo in combinations
            if combo["entry_z"] > combo["exit_z"] and combo["stop_z"] > combo["entry_z"]
        ]

        self.assertGreater(len(valid_combinations), 0)

    def test_get_best_results(self):
        """Test getting best results."""
        backtester = ParallelBacktester()

        # Add some mock results
        result1 = BacktestResult(
            model_name="model1",
            params={"entry_z": 1.0},
            metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.1},
            equity_curve=pd.Series(),
            signals=pd.DataFrame(),
            execution_time=1.0,
        )

        result2 = BacktestResult(
            model_name="model2",
            params={"entry_z": 1.2},
            metrics={"sharpe_ratio": 2.0, "max_drawdown": -0.05},
            equity_curve=pd.Series(),
            signals=pd.DataFrame(),
            execution_time=1.5,
        )

        result3 = BacktestResult(
            model_name="model3",
            params={"entry_z": 0.8},
            metrics={"sharpe_ratio": 1.8, "max_drawdown": -0.15},
            equity_curve=pd.Series(),
            signals=pd.DataFrame(),
            execution_time=0.8,
        )

        backtester.results = [result1, result2, result3]

        # Get top 2 results by Sharpe ratio
        best_results = backtester.get_best_results(metric="sharpe_ratio", top_n=2)

        self.assertEqual(len(best_results), 2)
        # Should be sorted by Sharpe ratio (descending)
        self.assertEqual(best_results[0].model_name, "model2")  # Highest Sharpe
        self.assertEqual(best_results[1].model_name, "model3")  # Second highest Sharpe

    def test_aggregate_results(self):
        """Test aggregating results."""
        backtester = ParallelBacktester()

        # Add some mock results
        result1 = BacktestResult(
            model_name="model1",
            params={"entry_z": 1.0, "exit_z": 0.5},
            metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.1},
            equity_curve=pd.Series(),
            signals=pd.DataFrame(),
            execution_time=1.0,
        )

        result2 = BacktestResult(
            model_name="model2",
            params={"entry_z": 1.2, "exit_z": 0.5},
            metrics={"sharpe_ratio": 2.0, "max_drawdown": -0.05},
            equity_curve=pd.Series(),
            signals=pd.DataFrame(),
            execution_time=1.5,
        )

        backtester.results = [result1, result2]

        # Aggregate results
        df = backtester.aggregate_results()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("model_name", df.columns)
        self.assertIn("sharpe_ratio", df.columns)
        self.assertIn("max_drawdown", df.columns)
        self.assertIn("entry_z", df.columns)
        self.assertIn("exit_z", df.columns)

        # Should be sorted by Sharpe ratio (descending)
        self.assertEqual(df.iloc[0]["model_name"], "model2")
        self.assertEqual(df.iloc[1]["model_name"], "model1")


class TestRunSingleBacktest(unittest.TestCase):
    """Test cases for the single backtest function."""

    def setUp(self):
        """Set up test data."""
        # Create simple test data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "fx_price": 1.3 + np.cumsum(np.random.randn(50) * 0.01),
                "comd_price": 50 + np.cumsum(np.random.randn(50) * 0.5),
                "spread": np.random.randn(50),
                "spread_z": np.random.randn(50),
                "signal": np.random.choice([-1, 0, 1], 50),
                "fx_position": np.random.randn(50),
                "comd_position": np.random.randn(50),
            },
            index=dates,
        )

        # Add required columns for backtesting
        self.data["delayed_signal"] = self.data["signal"].shift(1)
        self.data["delayed_fx_position"] = self.data["fx_position"].shift(1)
        self.data["delayed_comd_position"] = self.data["comd_position"].shift(1)
        self.data["fx_return"] = self.data["fx_price"].pct_change()
        self.data["comd_return"] = self.data["comd_price"].pct_change()
        self.data["fx_pnl"] = (
            self.data["delayed_fx_position"] * self.data["fx_price"].diff()
        )
        self.data["comd_pnl"] = (
            self.data["delayed_comd_position"] * self.data["comd_price"].diff()
        )
        self.data["total_pnl"] = self.data["fx_pnl"] + self.data["comd_pnl"]
        self.data["cumulative_pnl"] = self.data["total_pnl"].cumsum()
        self.data["equity"] = 1.0 + self.data["cumulative_pnl"]
        self.data["running_max"] = self.data["equity"].cummax()
        self.data["drawdown"] = (
            self.data["equity"] - self.data["running_max"]
        ) / self.data["running_max"]

        self.config = {
            "thresholds": {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.5},
            "time_stop": {"max_days": 10},
            "inverse_fx_for_quote_ccy_strength": True,
            "min_trade_count": 5,
        }

    def test_run_single_backtest_success(self):
        """Test successful single backtest execution."""
        args = (
            "test_model",
            {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.5},
            self.data,
            self.config,
        )

        result = _run_single_backtest(args)

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.model_name, "test_model")
        self.assertIn("entry_z", result.params)
        self.assertGreaterEqual(result.execution_time, 0)

        # Check that metrics were calculated
        self.assertIn("total_pnl", result.metrics)
        self.assertIn("sharpe_ratio", result.metrics)

    def test_run_single_backtest_with_invalid_params(self):
        """Test single backtest with invalid parameters."""
        # Invalid config with stop_z <= entry_z
        invalid_config = self.config.copy()
        invalid_config["thresholds"] = {
            "entry_z": 1.0,
            "exit_z": 0.5,
            "stop_z": 0.8,  # Invalid: stop_z should be > entry_z
        }

        args = (
            "test_model",
            {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 0.8},
            self.data,
            invalid_config,
        )

        result = _run_single_backtest(args)

        self.assertIsInstance(result, BacktestResult)
        # Should still return a result, possibly with error metrics


if __name__ == "__main__":
    unittest.main()
