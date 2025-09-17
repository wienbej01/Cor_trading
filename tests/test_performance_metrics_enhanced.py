"""
Unit tests for Performance Metrics Enhanced Module.

Tests for:
- NaN handling with insufficient trade counts
- Rolling window metric calculations (Sharpe, Sortino, drawdown)
- Trade return distribution analysis (skewness, kurtosis, VaR, CVaR)
- Integration with existing performance framework
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings
from scipy import stats

from src.backtest.rolling_metrics import RollingMetrics
from src.backtest.distribution_analysis import DistributionAnalysis
from tests.test_utils import (
    generate_synthetic_market_data,
    generate_synthetic_equity_curve,
    CustomAssertions,
)


class TestRollingMetrics:
    """Test RollingMetrics class."""

    @pytest.fixture
    def rolling_metrics(self):
        """Create a RollingMetrics instance for testing."""
        return RollingMetrics()

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        # Generate returns with some characteristics
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)

        # Add some drawdown periods
        returns.iloc[200:250] = np.random.normal(-0.005, 0.01, 50)
        returns.iloc[400:450] = np.random.normal(-0.008, 0.015, 50)

        return returns

    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve for testing."""
        return generate_synthetic_equity_curve(
            pd.date_range("2020-01-01", "2022-12-31", freq="D"),
            base_equity=100000,
            annual_return=0.10,
            annual_volatility=0.15,
            max_drawdown=0.20,
            seed=42,
        )

    def test_calculate_rolling_sharpe(self, rolling_metrics, sample_returns):
        """Test rolling Sharpe ratio calculation."""
        window = 63  # ~3 months of daily data

        rolling_sharpe = rolling_metrics.calculate_rolling_sharpe(
            sample_returns, window=window
        )

        # Check that result is a Series
        assert isinstance(rolling_sharpe, pd.Series)

        # Check that index matches input
        assert rolling_sharpe.index.equals(sample_returns.index)

        # Check that values are reasonable (NaN for early periods)
        assert rolling_sharpe.iloc[: window - 1].isna().all()
        assert not rolling_sharpe.iloc[window:].isna().all()

        # Check that Sharpe ratios are within reasonable bounds
        valid_sharpe = rolling_sharpe.dropna()
        assert not valid_sharpe.empty
        assert (valid_sharpe.abs() < 10).all()  # Reasonable bounds

    def test_calculate_rolling_sortino(self, rolling_metrics, sample_returns):
        """Test rolling Sortino ratio calculation."""
        window = 63

        rolling_sortino = rolling_metrics.calculate_rolling_sortino(
            sample_returns, window=window
        )

        # Check that result is a Series
        assert isinstance(rolling_sortino, pd.Series)

        # Check that index matches input
        assert rolling_sortino.index.equals(sample_returns.index)

        # Check that values are reasonable
        assert rolling_sortino.iloc[: window - 1].isna().all()
        assert not rolling_sortino.iloc[window:].isna().all()

        # Sortino should generally be higher than Sharpe for same data
        rolling_sharpe = rolling_metrics.calculate_rolling_sharpe(
            sample_returns, window=window
        )
        valid_sortino = rolling_sortino.dropna()
        valid_sharpe = rolling_sharpe.dropna()

        # For most periods, Sortino should be >= Sharpe
        assert (valid_sortino >= valid_sharpe).mean() > 0.5

    def test_calculate_rolling_drawdown(self, rolling_metrics, sample_returns):
        """Test rolling drawdown calculation."""
        window = 63

        rolling_dd = rolling_metrics.calculate_rolling_drawdown(
            sample_returns, window=window
        )

        # Check that result is a Series
        assert isinstance(rolling_dd, pd.Series)

        # Check that index matches input
        assert rolling_dd.index.equals(sample_returns.index)

        # Check that values are reasonable
        assert rolling_dd.iloc[: window - 1].isna().all()
        assert not rolling_dd.iloc[window:].isna().all()

        # Drawdown should be <= 0
        valid_dd = rolling_dd.dropna()
        assert (valid_dd <= 0).all()

        # Check that we have some significant drawdowns
        assert valid_dd.min() < -0.05  # At least 5% drawdown

    def test_calculate_rolling_calmar(self, rolling_metrics, sample_returns):
        """Test rolling Calmar ratio calculation."""
        window = 126  # ~6 months

        rolling_calmar = rolling_metrics.calculate_rolling_calmar(
            sample_returns, window=window
        )

        # Check that result is a Series
        assert isinstance(rolling_calmar, pd.Series)

        # Check that index matches input
        assert rolling_calmar.index.equals(sample_returns.index)

        # Check that values are reasonable
        assert rolling_calmar.iloc[: window - 1].isna().all()
        assert not rolling_calmar.iloc[window:].isna().all()

        # Calmar ratio can be positive or negative
        valid_calmar = rolling_calmar.dropna()
        assert not valid_calmar.empty

    def test_calculate_rolling_win_rate(self, rolling_metrics, sample_returns):
        """Test rolling win rate calculation."""
        window = 21  # ~1 month

        rolling_win_rate = rolling_metrics.calculate_rolling_win_rate(
            sample_returns, window=window
        )

        # Check that result is a Series
        assert isinstance(rolling_win_rate, pd.Series)

        # Check that index matches input
        assert rolling_win_rate.index.equals(sample_returns.index)

        # Check that values are reasonable
        assert rolling_win_rate.iloc[: window - 1].isna().all()
        assert not rolling_win_rate.iloc[window:].isna().all()

        # Win rate should be between 0 and 1
        valid_win_rate = rolling_win_rate.dropna()
        assert (valid_win_rate >= 0).all()
        assert (valid_win_rate <= 1).all()

    def test_calculate_rolling_metrics_comprehensive(
        self, rolling_metrics, sample_returns
    ):
        """Test comprehensive rolling metrics calculation."""
        window = 63

        metrics_df = rolling_metrics.calculate_rolling_metrics(
            sample_returns, window=window
        )

        # Check that result is a DataFrame
        assert isinstance(metrics_df, pd.DataFrame)

        # Check that all expected metrics are present
        expected_metrics = ["sharpe", "sortino", "drawdown", "calmar", "win_rate"]
        for metric in expected_metrics:
            assert metric in metrics_df.columns

        # Check that index matches input
        assert metrics_df.index.equals(sample_returns.index)

        # Check that values are reasonable
        for metric in expected_metrics:
            assert not metrics_df[metric].dropna().empty

    def test_insufficient_data_handling(self, rolling_metrics):
        """Test handling of insufficient data."""
        # Very short series
        short_dates = pd.date_range("2020-01-01", "2020-01-10")
        short_returns = pd.Series(
            [0.01, -0.005, 0.02, -0.01, 0.005, -0.015, 0.01, -0.005, 0.02, -0.01],
            index=short_dates,
        )

        window = 63  # Much larger than data length

        metrics_df = rolling_metrics.calculate_rolling_metrics(
            short_returns, window=window
        )

        # Should handle gracefully - mostly NaN values
        assert isinstance(metrics_df, pd.DataFrame)
        assert metrics_df.isna().all().all()

    def test_nan_values_in_input(self, rolling_metrics, sample_returns):
        """Test handling of NaN values in input."""
        # Add some NaN values
        returns_with_nan = sample_returns.copy()
        returns_with_nan.iloc[100:105] = np.nan
        returns_with_nan.iloc[200:205] = np.nan

        window = 63

        metrics_df = rolling_metrics.calculate_rolling_metrics(
            returns_with_nan, window=window
        )

        # Should handle NaN values gracefully
        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) == len(returns_with_nan)

        # Some values should be non-NaN
        assert not metrics_df.isna().all().all()

    def test_different_window_sizes(self, rolling_metrics, sample_returns):
        """Test with different window sizes."""
        windows = [21, 63, 126, 252]  # 1 month, 3 months, 6 months, 1 year

        for window in windows:
            metrics_df = rolling_metrics.calculate_rolling_metrics(
                sample_returns, window=window
            )

            # Should work for all window sizes
            assert isinstance(metrics_df, pd.DataFrame)
            assert len(metrics_df) == len(sample_returns)

            # Earlier values should be NaN
            assert metrics_df.iloc[: window - 1].isna().all().all()

            # Later values should not all be NaN
            assert not metrics_df.iloc[window:].isna().all().all()


class TestDistributionAnalysis:
    """Test DistributionAnalysis class."""

    @pytest.fixture
    def distribution_analyzer(self):
        """Create a DistributionAnalysis instance for testing."""
        return DistributionAnalysis()

    @pytest.fixture
    def sample_trade_returns(self):
        """Create sample trade returns for testing."""
        np.random.seed(123)

        # Generate trade returns with specific characteristics
        n_trades = 1000

        # Base returns
        returns = np.random.normal(0.001, 0.02, n_trades)

        # Add some fat tails (extreme returns)
        extreme_returns = np.random.normal(
            0, 0.1, int(n_trades * 0.05)
        )  # 5% extreme returns
        returns[: len(extreme_returns)] = extreme_returns

        # Add slight negative skew
        negative_skew = np.random.normal(
            -0.005, 0.01, int(n_trades * 0.1)
        )  # 10% negatively skewed
        returns[: len(negative_skew)] = negative_skew

        return pd.Series(returns)

    @pytest.fixture
    def sample_equity_returns(self):
        """Create sample equity returns for testing."""
        np.random.seed(456)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        # Generate daily returns
        returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)

        return returns

    def test_calculate_skewness(self, distribution_analyzer, sample_trade_returns):
        """Test skewness calculation."""
        skewness = distribution_analyzer.calculate_skewness(sample_trade_returns)

        # Check that result is a float
        assert isinstance(skewness, float)

        # Check that it's a reasonable value
        assert -5 <= skewness <= 5  # Reasonable bounds for skewness

    def test_calculate_kurtosis(self, distribution_analyzer, sample_trade_returns):
        """Test kurtosis calculation."""
        kurtosis = distribution_analyzer.calculate_kurtosis(sample_trade_returns)

        # Check that result is a float
        assert isinstance(kurtosis, float)

        # Check that it's a reasonable value
        # Normal distribution has kurtosis of 3, fat tails have higher values
        assert kurtosis >= -5  # Lower bound
        assert kurtosis <= 50  # Upper bound for fat tails

    def test_calculate_var(self, distribution_analyzer, sample_trade_returns):
        """Test Value at Risk (VaR) calculation."""
        # Test different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]

        for confidence in confidence_levels:
            var = distribution_analyzer.calculate_var(
                sample_trade_returns, confidence=confidence
            )

            # Check that result is a float
            assert isinstance(var, float)

            # VaR should be negative (loss)
            assert var <= 0

            # Higher confidence should give more extreme VaR
            if confidence > 0.90:
                var_95 = distribution_analyzer.calculate_var(
                    sample_trade_returns, confidence=0.95
                )
                assert var <= var_95  # More negative for higher confidence

    def test_calculate_cvar(self, distribution_analyzer, sample_trade_returns):
        """Test Conditional Value at Risk (CVaR) calculation."""
        # Test different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]

        for confidence in confidence_levels:
            cvar = distribution_analyzer.calculate_cvar(
                sample_trade_returns, confidence=confidence
            )

            # Check that result is a float
            assert isinstance(cvar, float)

            # CVaR should be negative (loss)
            assert cvar <= 0

            # CVaR should be more extreme than VaR
            var = distribution_analyzer.calculate_var(
                sample_trade_returns, confidence=confidence
            )
            assert cvar <= var  # More negative

    def test_calculate_all_distribution_metrics(
        self, distribution_analyzer, sample_trade_returns
    ):
        """Test calculation of all distribution metrics."""
        metrics = distribution_analyzer.calculate_all_distribution_metrics(
            sample_trade_returns
        )

        # Check that result is a dictionary
        assert isinstance(metrics, dict)

        # Check that all expected metrics are present
        expected_metrics = [
            "skewness",
            "kurtosis",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
        ]
        for metric in expected_metrics:
            assert metric in metrics

        # Check that values are reasonable
        assert isinstance(metrics["skewness"], float)
        assert isinstance(metrics["kurtosis"], float)
        assert isinstance(metrics["var_95"], float)
        assert isinstance(metrics["var_99"], float)
        assert isinstance(metrics["cvar_95"], float)
        assert isinstance(metrics["cvar_99"], float)

        # VaR and CVaR should be negative
        assert metrics["var_95"] <= 0
        assert metrics["var_99"] <= 0
        assert metrics["cvar_95"] <= 0
        assert metrics["cvar_99"] <= 0

        # CVaR should be more extreme than VaR
        assert metrics["cvar_95"] <= metrics["var_95"]
        assert metrics["cvar_99"] <= metrics["var_99"]

        # 99% metrics should be more extreme than 95%
        assert metrics["var_99"] <= metrics["var_95"]
        assert metrics["cvar_99"] <= metrics["cvar_95"]

    def test_rolling_distribution_analysis(
        self, distribution_analyzer, sample_equity_returns
    ):
        """Test rolling distribution analysis."""
        window = 126  # ~6 months

        rolling_metrics = distribution_analyzer.calculate_rolling_distribution_metrics(
            sample_equity_returns, window=window
        )

        # Check that result is a DataFrame
        assert isinstance(rolling_metrics, pd.DataFrame)

        # Check that all expected metrics are present
        expected_metrics = [
            "skewness",
            "kurtosis",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
        ]
        for metric in expected_metrics:
            assert metric in rolling_metrics.columns

        # Check that index matches input
        assert rolling_metrics.index.equals(sample_equity_returns.index)

        # Check that values are reasonable
        assert rolling_metrics.iloc[: window - 1].isna().all().all()  # Early values NaN
        assert (
            not rolling_metrics.iloc[window:].isna().all().all()
        )  # Later values not all NaN

        # VaR and CVaR should be negative where not NaN
        for metric in ["var_95", "var_99", "cvar_95", "cvar_99"]:
            valid_values = rolling_metrics[metric].dropna()
            assert (valid_values <= 0).all()

    def test_insufficient_trades_handling(self, distribution_analyzer):
        """Test handling of insufficient trade data."""
        # Very few trades
        few_returns = pd.Series([0.01, -0.005, 0.02, -0.01])

        metrics = distribution_analyzer.calculate_all_distribution_metrics(few_returns)

        # Should handle gracefully
        assert isinstance(metrics, dict)

        # Values might be extreme or NaN, but should be present
        expected_metrics = [
            "skewness",
            "kurtosis",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
        ]
        for metric in expected_metrics:
            assert metric in metrics

    def test_nan_values_handling(self, distribution_analyzer, sample_trade_returns):
        """Test handling of NaN values in input."""
        # Add some NaN values
        returns_with_nan = sample_trade_returns.copy()
        returns_with_nan.iloc[:10] = np.nan

        metrics = distribution_analyzer.calculate_all_distribution_metrics(
            returns_with_nan
        )

        # Should handle NaN values gracefully
        assert isinstance(metrics, dict)

        # Should still calculate metrics using valid data
        expected_metrics = [
            "skewness",
            "kurtosis",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
        ]
        for metric in expected_metrics:
            assert metric in metrics

    def test_extreme_values_handling(self, distribution_analyzer):
        """Test handling of extreme values."""
        # Create returns with extreme values
        normal_returns = np.random.normal(0.001, 0.02, 100)
        extreme_returns = np.array([10.0, -15.0, 20.0, -25.0])  # Extreme values
        all_returns = np.concatenate([normal_returns, extreme_returns])

        returns_series = pd.Series(all_returns)

        metrics = distribution_analyzer.calculate_all_distribution_metrics(
            returns_series
        )

        # Should handle extreme values gracefully
        assert isinstance(metrics, dict)

        # Skewness and kurtosis should reflect extreme values
        assert abs(metrics["skewness"]) > 0.5  # Should be skewed
        assert metrics["kurtosis"] > 5  # Should have fat tails

        # VaR and CVaR should capture extreme values
        assert metrics["var_99"] < -0.1  # Should be quite negative
        assert metrics["cvar_99"] < metrics["var_99"]  # CVaR more extreme

    def test_custom_confidence_levels(
        self, distribution_analyzer, sample_trade_returns
    ):
        """Test calculation with custom confidence levels."""
        # Test non-standard confidence levels
        var_80 = distribution_analyzer.calculate_var(
            sample_trade_returns, confidence=0.80
        )
        var_999 = distribution_analyzer.calculate_var(
            sample_trade_returns, confidence=0.999
        )

        cvar_80 = distribution_analyzer.calculate_cvar(
            sample_trade_returns, confidence=0.80
        )
        cvar_999 = distribution_analyzer.calculate_cvar(
            sample_trade_returns, confidence=0.999
        )

        # Check that results are reasonable
        assert isinstance(var_80, float)
        assert isinstance(var_999, float)
        assert isinstance(cvar_80, float)
        assert isinstance(cvar_999, float)

        # More extreme confidence should give more extreme values
        assert var_999 <= var_80  # More negative
        assert cvar_999 <= cvar_80  # More negative

        # CVaR should be more extreme than VaR
        assert cvar_80 <= var_80
        assert cvar_999 <= var_999


class TestPerformanceMetricsIntegration:
    """Integration tests for performance metrics components."""

    def test_rolling_and_distribution_integration(self):
        """Test integration between rolling metrics and distribution analysis."""
        np.random.seed(789)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        # Create returns with varying characteristics
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)

        # Add some regime changes
        returns.iloc[200:300] *= 2  # Higher volatility period
        returns.iloc[500:600] = np.random.normal(
            -0.002, 0.03, 100
        )  # Negative drift period

        # Calculate rolling metrics
        rolling_metrics = RollingMetrics()
        rolling_df = rolling_metrics.calculate_rolling_metrics(returns, window=126)

        # Calculate distribution metrics
        dist_analysis = DistributionAnalysis()
        dist_df = dist_analysis.calculate_rolling_distribution_metrics(
            returns, window=126
        )

        # Check that both produce valid results
        assert isinstance(rolling_df, pd.DataFrame)
        assert isinstance(dist_df, pd.DataFrame)

        # Check that indices match
        assert rolling_df.index.equals(dist_df.index)

        # Check relationship between metrics
        # Periods with high drawdown should have more negative skewness
        high_dd_periods = rolling_df["drawdown"] < -0.1
        if high_dd_periods.any():
            high_dd_skewness = dist_df.loc[high_dd_periods, "skewness"].mean()
            low_dd_skewness = dist_df.loc[~high_dd_periods, "skewness"].mean()

            # High drawdown periods should tend to have more negative skewness
            assert high_dd_skewness <= low_dd_skewness

    def test_comprehensive_performance_analysis(self):
        """Test comprehensive performance analysis combining all metrics."""
        np.random.seed(101)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        # Create realistic returns with multiple regimes
        base_returns = np.random.normal(0.001, 0.015, len(dates))

        # Add different market regimes
        # Bull market
        base_returns[100:300] = np.random.normal(0.003, 0.01, 200)

        # Bear market
        base_returns[400:600] = np.random.normal(-0.002, 0.025, 200)

        # High volatility
        base_returns[700:800] = np.random.normal(0.0, 0.04, 100)

        returns = pd.Series(base_returns, index=dates)

        # Calculate all metrics
        rolling_metrics = RollingMetrics()
        rolling_df = rolling_metrics.calculate_rolling_metrics(returns, window=63)

        dist_analysis = DistributionAnalysis()
        dist_df = dist_analysis.calculate_rolling_distribution_metrics(
            returns, window=126
        )

        # Overall distribution metrics
        overall_metrics = dist_analysis.calculate_all_distribution_metrics(returns)

        # Check that all components work together
        assert isinstance(rolling_df, pd.DataFrame)
        assert isinstance(dist_df, pd.DataFrame)
        assert isinstance(overall_metrics, dict)

        # Check that metrics capture different regimes
        # Bull market should have positive Sharpe
        bull_period = returns.index[100:300]
        bull_sharpe = rolling_df.loc[bull_period, "sharpe"].mean()
        assert bull_sharpe > 0

        # Bear market should have negative Sharpe
        bear_period = returns.index[400:600]
        bear_sharpe = rolling_df.loc[bear_period, "sharpe"].mean()
        assert bear_sharpe < 0

        # High volatility period should have higher kurtosis
        high_vol_period = returns.index[700:800]
        high_vol_kurtosis = dist_df.loc[high_vol_period, "kurtosis"].mean()
        normal_vol_kurtosis = dist_df.loc[returns.index[300:400], "kurtosis"].mean()
        assert high_vol_kurtosis > normal_vol_kurtosis

    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        rolling_metrics = RollingMetrics()
        dist_analysis = DistributionAnalysis()

        # Empty series
        empty_returns = pd.Series([], dtype=float)

        # Should handle gracefully
        rolling_df = rolling_metrics.calculate_rolling_metrics(empty_returns, window=63)
        dist_df = dist_analysis.calculate_rolling_distribution_metrics(
            empty_returns, window=63
        )
        overall_metrics = dist_analysis.calculate_all_distribution_metrics(
            empty_returns
        )

        assert isinstance(rolling_df, pd.DataFrame)
        assert isinstance(dist_df, pd.DataFrame)
        assert isinstance(overall_metrics, dict)

        # Should be empty but valid
        assert len(rolling_df) == 0
        assert len(dist_df) == 0
        # Overall metrics might have NaN values but should be present

    def test_performance_report_generation(self):
        """Test generation of comprehensive performance report."""
        np.random.seed(202)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        # Create realistic returns
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)

        # Add some drawdowns
        returns.iloc[200:250] = np.random.normal(-0.005, 0.015, 50)
        returns.iloc[500:550] = np.random.normal(-0.008, 0.02, 50)

        # Calculate metrics
        rolling_metrics = RollingMetrics()
        dist_analysis = DistributionAnalysis()

        # Different window sizes
        short_term_metrics = rolling_metrics.calculate_rolling_metrics(
            returns, window=21
        )
        medium_term_metrics = rolling_metrics.calculate_rolling_metrics(
            returns, window=63
        )
        long_term_metrics = rolling_metrics.calculate_rolling_metrics(
            returns, window=252
        )

        dist_metrics = dist_analysis.calculate_rolling_distribution_metrics(
            returns, window=126
        )
        overall_dist_metrics = dist_analysis.calculate_all_distribution_metrics(returns)

        # Create performance report
        performance_report = {
            "short_term_metrics": short_term_metrics,
            "medium_term_metrics": medium_term_metrics,
            "long_term_metrics": long_term_metrics,
            "distribution_metrics": dist_metrics,
            "overall_distribution": overall_dist_metrics,
        }

        # Check that report is comprehensive
        assert isinstance(performance_report, dict)
        assert len(performance_report) == 5

        # Check that all components are valid
        for key, value in performance_report.items():
            if "distribution" in key and key != "overall_distribution":
                assert isinstance(value, pd.DataFrame)
            elif key == "overall_distribution":
                assert isinstance(value, dict)
            else:
                assert isinstance(value, pd.DataFrame)

        # Check that we can extract meaningful insights
        # For example, compare short-term vs long-term Sharpe ratios
        short_term_sharpe = short_term_metrics["sharpe"].mean()
        long_term_sharpe = long_term_metrics["sharpe"].mean()

        # Both should be reasonable numbers
        assert isinstance(short_term_sharpe, float)
        assert isinstance(long_term_sharpe, float)
        assert not np.isnan(short_term_sharpe)
        assert not np.isnan(long_term_sharpe)
