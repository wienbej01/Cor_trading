"""
Unit tests for the metrics module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import tempfile

from src.backtest.metrics import (
    calculate_equity_stats,
    calculate_trade_stats,
    calculate_per_day_stats,
    calculate_cost_slippage_stats,
    generate_run_id,
    save_run_artifacts,
    calculate_comprehensive_metrics
)


@pytest.fixture
def sample_equity_curve():
    """Create a sample equity curve for testing."""
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
    # Create a realistic equity curve with some volatility
    returns = np.random.normal(0.0005, 0.02, 252)  # 0.05% daily return, 2% daily volatility
    equity = 100000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_trades_df():
    """Create a sample trades DataFrame for testing."""
    trades_data = {
        "trade_id": [1, 2, 3, 4, 5],
        "entry_date": pd.date_range(start="2020-01-01", periods=5, freq="D"),
        "exit_date": pd.date_range(start="2020-01-02", periods=5, freq="D"),
        "direction": [1, -1, 1, -1, 1],
        "duration": [1, 1, 1, 1, 1],
        "pnl": [1000, -500, 1500, -300, 800],
        "fx_entry": [1.2, 1.3, 1.4, 1.5, 1.6],
        "fx_exit": [1.25, 1.25, 1.45, 1.48, 1.65],
        "comd_entry": [50, 55, 60, 65, 70],
        "comd_exit": [52, 54, 62, 64, 72]
    }
    return pd.DataFrame(trades_data)


@pytest.fixture
def sample_backtest_df():
    """Create a sample backtest DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
    backtest_data = {
        "pnl": np.random.normal(100, 50, 252),
        "total_pnl": np.random.normal(90, 50, 252),
        "delayed_signal": np.random.choice([0, 1, -1], 252),
        "equity": 100000 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
    }
    return pd.DataFrame(backtest_data, index=dates)


def test_calculate_equity_stats(sample_equity_curve):
    """Test equity statistics calculation."""
    stats = calculate_equity_stats(sample_equity_curve)
    
    # Check that all expected keys are present
    expected_keys = ["total_return", "annual_return", "volatility", "sharpe_ratio", 
                     "max_drawdown", "calmar_ratio", "ulcer_index"]
    for key in expected_keys:
        assert key in stats
        assert isinstance(stats[key], float)
    
    # Check that total return is reasonable
    assert stats["total_return"] > -1  # Can't lose more than 100%
    
    # Check that volatility is positive
    assert stats["volatility"] >= 0
    
    # Check that max drawdown is negative or zero
    assert stats["max_drawdown"] <= 0


def test_calculate_trade_stats(sample_trades_df):
    """Test trade statistics calculation."""
    stats = calculate_trade_stats(sample_trades_df)
    
    # Check that all expected keys are present
    expected_keys = ["total_trades", "winning_trades", "losing_trades", "win_rate",
                     "avg_win", "avg_loss", "profit_factor", "max_win", "max_loss", "avg_duration"]
    for key in expected_keys:
        assert key in stats
    
    # Check specific values
    assert stats["total_trades"] == 5
    assert stats["winning_trades"] == 3
    assert stats["losing_trades"] == 2
    assert stats["win_rate"] == 0.6
    assert stats["max_win"] == 1500
    assert stats["max_loss"] == -500


def test_calculate_per_day_stats(sample_backtest_df):
    """Test per-day statistics calculation."""
    stats = calculate_per_day_stats(sample_backtest_df)
    
    # Check that all expected keys are present
    expected_keys = ["best_day", "worst_day", "avg_daily_return", "daily_volatility",
                     "positive_days", "negative_days", "daily_win_rate"]
    for key in expected_keys:
        assert key in stats
        assert isinstance(stats[key], (int, float))
    
    # Check that best day is greater than or equal to worst day
    assert stats["best_day"] >= stats["worst_day"]


def test_calculate_cost_slippage_stats(sample_backtest_df):
    """Test cost and slippage statistics calculation."""
    stats = calculate_cost_slippage_stats(sample_backtest_df)
    
    # Check that all expected keys are present
    expected_keys = ["total_costs", "costs_per_trade", "costs_pct_of_pnl"]
    for key in expected_keys:
        assert key in stats
        assert isinstance(stats[key], float)


def test_generate_run_id():
    """Test run ID generation."""
    run_id = generate_run_id()
    
    # Check that run_id is a string
    assert isinstance(run_id, str)
    
    # Check that run_id has the expected format (YYYYMMDD_HHMMSS)
    assert len(run_id) == 15
    assert run_id[8] == "_"


def test_save_run_artifacts(sample_backtest_df, sample_trades_df):
    """Test saving run artifacts."""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create sample config
            config = {
                "pair": "usdcad_wti",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31"
            }
            
            # Create sample metrics
            metrics = {
                "equity": {"total_return": 0.15},
                "trades": {"total_trades": 5},
                "daily": {"best_day": 1000},
                "costs": {"total_costs": -500}
            }
            
            # Test saving artifacts
            run_id = "20200101_120000"
            reports_path = save_run_artifacts(
                "usdcad_wti", sample_backtest_df, sample_trades_df, config, metrics, run_id
            )
            
            # Check that reports directory was created
            assert os.path.exists(reports_path)
            assert "reports/usdcad_wti/20200101_120000" in reports_path
            
            # Check that files were created
            assert os.path.exists(f"{reports_path}/summary.json")
            assert os.path.exists(f"{reports_path}/trades.csv")
            assert os.path.exists(f"{reports_path}/config.json")
            
            # Check that JSON files contain valid data
            with open(f"{reports_path}/summary.json", "r") as f:
                summary_data = json.load(f)
                assert "equity" in summary_data
            
            with open(f"{reports_path}/config.json", "r") as f:
                config_data = json.load(f)
                assert config_data["pair"] == "usdcad_wti"
        finally:
            # Change back to original directory
            os.chdir(original_cwd)


def test_calculate_comprehensive_metrics(sample_backtest_df, sample_trades_df):
    """Test comprehensive metrics calculation."""
    # Create sample config
    config = {
        "pair": "usdcad_wti",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31"
    }
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(sample_backtest_df, sample_trades_df, config)
    
    # Check that all expected sections are present
    expected_sections = ["timestamp", "equity", "trades", "daily", "costs", "config"]
    for section in expected_sections:
        assert section in metrics
    
    # Check that timestamp is present
    assert "timestamp" in metrics
    assert isinstance(metrics["timestamp"], str)
    
    # Check that config section contains expected data
    assert metrics["config"]["pair"] == "usdcad_wti"