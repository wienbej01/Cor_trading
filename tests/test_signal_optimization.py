"""
Unit tests for Signal Optimization Module.

Tests for:
- Z-score calculations (standard and robust)
- Kalman filter implementation
- OLS spread calculation
- Parameter sweep functionality
- Volatility-adjusted thresholds
- Look-ahead bias prevention
- Edge case handling and error conditions
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings

from src.features.signal_optimization import SignalOptimizer
from src.features.indicators import zscore, zscore_robust, atr_proxy
from src.features.spread import compute_spread
from test_utils import generate_synthetic_market_data, CustomAssertions


class TestSignalOptimizer:
    """Test class for SignalOptimizer functionality."""

    @pytest.fixture
    def optimizer(self, sample_config):
        """Create a SignalOptimizer instance for testing."""
        return SignalOptimizer(sample_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2021-12-31", seed=42
        )

    @pytest.fixture
    def noisy_data(self):
        """Create noisy market data with NaN values for testing."""
        fx_data, commodity_data = generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2021-12-31", seed=123
        )

        # Add some NaN values
        fx_data.iloc[50:55] = np.nan
        commodity_data.iloc[100:105] = np.nan

        # Add some extreme values
        fx_data.iloc[200] = fx_data.iloc[200] * 10
        commodity_data.iloc[300] = commodity_data.iloc[300] * 0.1

        return fx_data, commodity_data

    class TestGenerateThresholdCombinations:
        """Test threshold combination generation."""

        def test_default_combinations(self, optimizer):
            """Test generation of default threshold combinations."""
            combinations = optimizer.generate_threshold_combinations()

            # Check that combinations are generated
            assert len(combinations) > 0

            # Check that all combinations maintain entry > exit > stop constraint
            for combo in combinations:
                assert combo["entry_z"] > combo["exit_z"]
                assert combo["exit_z"] < combo["stop_z"]

            # Check that all required keys are present
            for combo in combinations:
                assert "entry_z" in combo
                assert "exit_z" in combo
                assert "stop_z" in combo

        def test_custom_ranges(self, optimizer):
            """Test generation with custom threshold ranges."""
            entry_range = [1.0, 1.5, 2.0]
            exit_range = [0.3, 0.6]
            stop_range = [2.5, 3.0]

            combinations = optimizer.generate_threshold_combinations(
                entry_range=entry_range, exit_range=exit_range, stop_range=stop_range
            )

            # Check that combinations use custom ranges
            entry_values = {combo["entry_z"] for combo in combinations}
            exit_values = {combo["exit_z"] for combo in combinations}
            stop_values = {combo["stop_z"] for combo in combinations}

            assert entry_values == set(entry_range)
            assert exit_values == set(exit_range)
            assert stop_values == set(stop_range)

        def test_empty_combinations(self, optimizer):
            """Test handling of ranges that produce no valid combinations."""
            # These ranges should produce no valid combinations
            entry_range = [0.5]
            exit_range = [1.0]  # exit > entry, violates constraint
            stop_range = [0.8]  # stop < exit, violates constraint

            combinations = optimizer.generate_threshold_combinations(
                entry_range=entry_range, exit_range=exit_range, stop_range=stop_range
            )

            # Should return empty list
            assert len(combinations) == 0

    class TestCalculateVolatilityAdjustedThresholds:
        """Test volatility-adjusted threshold calculation."""

        def test_basic_adjustment(self, optimizer, sample_data):
            """Test basic volatility adjustment calculation."""
            fx_series, commodity_series = sample_data

            # Create simple z-scores
            z_scores = zscore(fx_series, window=20)

            base_entry = 1.5
            base_exit = 0.5
            base_stop = 3.0

            entry_adj, exit_adj, stop_adj = (
                optimizer.calculate_volatility_adjusted_thresholds(
                    z_scores, base_entry, base_exit, base_stop
                )
            )

            # Check that thresholds are returned as series
            assert isinstance(entry_adj, pd.Series)
            assert isinstance(exit_adj, pd.Series)
            assert isinstance(stop_adj, pd.Series)

            # Check that indices match
            assert entry_adj.index.equals(z_scores.index)
            assert exit_adj.index.equals(z_scores.index)
            assert stop_adj.index.equals(z_scores.index)

            # Check that minimum thresholds are maintained
            assert (entry_adj >= 0.5).all()
            assert (exit_adj >= 0.2).all()
            assert (stop_adj >= 2.0).all()

            # Check that entry > exit constraint is maintained
            assert (entry_adj > exit_adj).all()

        def test_scaling_factor(self, optimizer, sample_data):
            """Test volatility scaling factor."""
            fx_series, commodity_series = sample_data

            z_scores = zscore(fx_series, window=20)

            base_entry = 1.5
            base_exit = 0.5
            base_stop = 3.0

            # Test with different scaling factors
            entry_adj_1, exit_adj_1, stop_adj_1 = (
                optimizer.calculate_volatility_adjusted_thresholds(
                    z_scores, base_entry, base_exit, base_stop, vol_scaling_factor=0.5
                )
            )

            entry_adj_2, exit_adj_2, stop_adj_2 = (
                optimizer.calculate_volatility_adjusted_thresholds(
                    z_scores, base_entry, base_exit, base_stop, vol_scaling_factor=2.0
                )
            )

            # Higher scaling factor should produce more variable thresholds
            assert entry_adj_2.std() > entry_adj_1.std()
            assert exit_adj_2.std() > exit_adj_1.std()
            assert stop_adj_2.std() > stop_adj_1.std()

        def test_window_size(self, optimizer, sample_data):
            """Test different window sizes for volatility calculation."""
            fx_series, commodity_series = sample_data

            z_scores = zscore(fx_series, window=20)

            base_entry = 1.5
            base_exit = 0.5
            base_stop = 3.0

            # Test with different window sizes
            entry_adj_1, _, _ = optimizer.calculate_volatility_adjusted_thresholds(
                z_scores, base_entry, base_exit, base_stop, vol_window=10
            )

            entry_adj_2, _, _ = optimizer.calculate_volatility_adjusted_thresholds(
                z_scores, base_entry, base_exit, base_stop, vol_window=50
            )

            # Different window sizes should produce different results
            assert not entry_adj_1.equals(entry_adj_2)

    class TestEnhancedKalmanSpread:
        """Test enhanced Kalman filter spread calculation."""

        def test_basic_kalman(self, optimizer, sample_data):
            """Test basic Kalman filter spread calculation."""
            fx_series, commodity_series = sample_data

            spread, alpha, beta, residual_z = optimizer.enhanced_kalman_spread(
                fx_series, commodity_series, beta_window=60
            )

            # Check that all series are returned
            assert isinstance(spread, pd.Series)
            assert isinstance(alpha, pd.Series)
            assert isinstance(beta, pd.Series)
            assert isinstance(residual_z, pd.Series)

            # Check that indices match
            assert spread.index.equals(fx_series.index)
            assert alpha.index.equals(fx_series.index)
            assert beta.index.equals(fx_series.index)
            assert residual_z.index.equals(fx_series.index)

            # Check that spread has reasonable values
            assert not spread.isna().all()
            assert not alpha.isna().all()
            assert not beta.isna().all()

            # Check that beta is within reasonable bounds
            assert (beta.abs() <= 10).all()

        def test_adaptive_lambda(self, optimizer, sample_data):
            """Test adaptive lambda adjustment."""
            fx_series, commodity_series = sample_data

            # Create data with varying volatility
            noisy_fx = fx_series.copy()
            noisy_fx.iloc[200:300] *= 1 + np.random.normal(0, 0.05, 100)

            spread, alpha, beta, residual_z = optimizer.enhanced_kalman_spread(
                noisy_fx, commodity_series, beta_window=60
            )

            # Should still produce valid results
            assert not spread.isna().all()
            assert not alpha.isna().all()
            assert not beta.isna().all()

        def test_edge_cases(self, optimizer):
            """Test edge cases in Kalman filter calculation."""
            # Create very short series
            short_dates = pd.date_range("2020-01-01", "2020-01-10")
            short_fx = pd.Series(
                [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                index=short_dates,
            )
            short_commodity = pd.Series(
                [60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5, 64.0, 64.5],
                index=short_dates,
            )

            # Should handle short series gracefully
            spread, alpha, beta, residual_z = optimizer.enhanced_kalman_spread(
                short_fx, short_commodity, beta_window=5
            )

            assert isinstance(spread, pd.Series)
            assert isinstance(alpha, pd.Series)
            assert isinstance(beta, pd.Series)
            assert isinstance(residual_z, pd.Series)

    class TestEnhancedOLSSpread:
        """Test enhanced OLS spread calculation."""

        def test_basic_ols(self, optimizer, sample_data):
            """Test basic OLS spread calculation."""
            fx_series, commodity_series = sample_data

            spread, alpha, beta, optimal_window = optimizer.enhanced_ols_spread(
                fx_series, commodity_series, beta_window=60
            )

            # Check that all series are returned
            assert isinstance(spread, pd.Series)
            assert isinstance(alpha, pd.Series)
            assert isinstance(beta, pd.Series)
            assert isinstance(optimal_window, pd.Series)

            # Check that indices match
            assert spread.index.equals(fx_series.index)
            assert alpha.index.equals(fx_series.index)
            assert beta.index.equals(fx_series.index)
            assert optimal_window.index.equals(fx_series.index)

            # Check that spread has reasonable values
            assert not spread.isna().all()
            assert not alpha.isna().all()
            assert not beta.isna().all()

            # Check that optimal window is within bounds
            assert (optimal_window >= 20).all()
            assert (optimal_window <= 120).all()

        def test_window_optimization(self, optimizer, sample_data):
            """Test window optimization functionality."""
            fx_series, commodity_series = sample_data

            spread, alpha, beta, optimal_window = optimizer.enhanced_ols_spread(
                fx_series,
                commodity_series,
                beta_window=60,
                min_window=30,
                max_window=90,
                window_step=15,
            )

            # Check that optimal window is one of the tested values
            unique_windows = optimal_window.unique()
            assert len(unique_windows) == 1
            assert unique_windows[0] in [30, 45, 60, 75, 90]

        def test_fallback_to_base(self, optimizer):
            """Test fallback to base window when optimization fails."""
            # Create very short series
            short_dates = pd.date_range("2020-01-01", "2020-01-10")
            short_fx = pd.Series(
                [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                index=short_dates,
            )
            short_commodity = pd.Series(
                [60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5, 64.0, 64.5],
                index=short_dates,
            )

            spread, alpha, beta, optimal_window = optimizer.enhanced_ols_spread(
                short_fx,
                short_commodity,
                beta_window=5,
                min_window=10,
                max_window=20,
                window_step=5,
            )

            # Should fall back to base window
            assert (optimal_window == 5).all()

    class TestCalculateHurstExponent:
        """Test Hurst exponent calculation."""

        def test_basic_hurst(self, optimizer):
            """Test basic Hurst exponent calculation."""
            # Create random walk (should have Hurst ~0.5)
            np.random.seed(42)
            random_walk = np.random.normal(0, 1, 100).cumsum()
            series = pd.Series(random_walk)

            hurst = optimizer._calculate_hurst_exponent(series)

            # Should be close to 0.5 for random walk
            assert 0.3 <= hurst <= 0.7

        def test_trending_series(self, optimizer):
            """Test Hurst exponent for trending series."""
            # Create trending series (should have Hurst > 0.5)
            np.random.seed(42)
            trend = np.arange(100) * 0.1
            noise = np.random.normal(0, 0.5, 100)
            series = pd.Series(trend + noise)

            hurst = optimizer._calculate_hurst_exponent(series)

            # Should be greater than 0.5 for trending series
            assert hurst > 0.5

        def test_mean_reverting_series(self, optimizer):
            """Test Hurst exponent for mean-reverting series."""
            # Create mean-reverting series (should have Hurst < 0.5)
            np.random.seed(42)
            mean_rev = np.zeros(100)
            for i in range(1, 100):
                mean_rev[i] = 0.8 * mean_rev[i - 1] + np.random.normal(0, 1)
            series = pd.Series(mean_rev)

            hurst = optimizer._calculate_hurst_exponent(series)

            # Should be less than 0.5 for mean-reverting series
            assert hurst < 0.5

        def test_short_series(self, optimizer):
            """Test Hurst exponent with very short series."""
            # Very short series
            series = pd.Series([1, 2, 3, 4, 5])

            hurst = optimizer._calculate_hurst_exponent(series)

            # Should return neutral value for short series
            assert hurst == 0.5

    class TestCalculateSignalQualityMetrics:
        """Test signal quality metrics calculation."""

        def test_basic_metrics(self, optimizer, sample_data):
            """Test basic signal quality metrics."""
            fx_series, commodity_series = sample_data

            # Create simple signals
            signals = pd.Series(0, index=fx_series.index)
            signals.iloc[100:200] = 1
            signals.iloc[300:400] = -1

            # Create simple returns
            returns = fx_series.pct_change().fillna(0)

            metrics = optimizer.calculate_signal_quality_metrics(
                signals, returns, window=20
            )

            # Check that all expected metrics are present
            expected_metrics = [
                "signal_to_noise",
                "predictive_power",
                "signal_stability",
                "win_rate",
                "signal_consistency",
            ]
            for metric in expected_metrics:
                assert metric in metrics.columns

            # Check that metrics are series with correct index
            for metric in expected_metrics:
                assert isinstance(metrics[metric], pd.Series)
                assert metrics[metric].index.equals(signals.index)

        def test_edge_cases(self, optimizer):
            """Test edge cases in signal quality calculation."""
            # Empty signals
            empty_signals = pd.Series([], dtype=float)
            empty_returns = pd.Series([], dtype=float)

            metrics = optimizer.calculate_signal_quality_metrics(
                empty_signals, empty_returns, window=20
            )

            # Should return empty DataFrame
            assert isinstance(metrics, pd.DataFrame)
            assert len(metrics) == 0

            # All zero signals
            dates = pd.date_range("2020-01-01", "2020-01-10")
            zero_signals = pd.Series(0, index=dates)
            zero_returns = pd.Series(0, index=dates)

            metrics = optimizer.calculate_signal_quality_metrics(
                zero_signals, zero_returns, window=5
            )

            # Should handle zero signals gracefully
            assert isinstance(metrics, pd.DataFrame)
            assert len(metrics) == len(dates)

    class TestGenerateEnhancedSignals:
        """Test enhanced signal generation."""

        def test_basic_signals(self, optimizer, sample_data):
            """Test basic signal generation."""
            fx_series, commodity_series = sample_data

            thresholds = {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0}

            signals_df = optimizer.generate_enhanced_signals(
                fx_series,
                commodity_series,
                thresholds,
                use_vol_adjustment=False,
                use_enhanced_kalman=False,
                use_enhanced_ols=False,
            )

            # Check that DataFrame is returned with expected columns
            expected_columns = [
                "fx_price",
                "comd_price",
                "spread",
                "alpha",
                "beta",
                "spread_z",
                "entry_threshold",
                "exit_threshold",
                "stop_threshold",
                "signal",
                "enter_long",
                "enter_short",
                "exit_long",
                "exit_short",
                "stop_long",
                "stop_short",
            ]

            for col in expected_columns:
                assert col in signals_df.columns

            # Check that signals are valid (-1, 0, 1)
            assert signals_df["signal"].isin([-1, 0, 1]).all()

            # Check that indices match
            assert signals_df.index.equals(fx_series.index)

        def test_volatility_adjustment(self, optimizer, sample_data):
            """Test volatility-adjusted signals."""
            fx_series, commodity_series = sample_data

            thresholds = {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0}

            # Generate signals with and without volatility adjustment
            signals_no_vol = optimizer.generate_enhanced_signals(
                fx_series, commodity_series, thresholds, use_vol_adjustment=False
            )

            signals_with_vol = optimizer.generate_enhanced_signals(
                fx_series, commodity_series, thresholds, use_vol_adjustment=True
            )

            # Thresholds should be different
            assert not signals_no_vol["entry_threshold"].equals(
                signals_with_vol["entry_threshold"]
            )
            assert not signals_no_vol["exit_threshold"].equals(
                signals_with_vol["exit_threshold"]
            )
            assert not signals_no_vol["stop_threshold"].equals(
                signals_with_vol["stop_threshold"]
            )

        def test_enhanced_kalman(self, optimizer, sample_data):
            """Test enhanced Kalman filter signals."""
            fx_series, commodity_series = sample_data

            thresholds = {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0}

            signals_df = optimizer.generate_enhanced_signals(
                fx_series, commodity_series, thresholds, use_enhanced_kalman=True
            )

            # Should include residual_z column
            assert "residual_z" in signals_df.columns
            assert isinstance(signals_df["residual_z"], pd.Series)

        def test_enhanced_ols(self, optimizer, sample_data):
            """Test enhanced OLS signals."""
            fx_series, commodity_series = sample_data

            thresholds = {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0}

            signals_df = optimizer.generate_enhanced_signals(
                fx_series, commodity_series, thresholds, use_enhanced_ols=True
            )

            # Should include optimal_window column
            assert "optimal_window" in signals_df.columns
            assert isinstance(signals_df["optimal_window"], pd.Series)

        def test_signal_flags(self, optimizer, sample_data):
            """Test signal flag columns."""
            fx_series, commodity_series = sample_data

            thresholds = {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0}

            signals_df = optimizer.generate_enhanced_signals(
                fx_series, commodity_series, thresholds
            )

            # Check that signal flags are boolean
            flag_columns = [
                "enter_long",
                "enter_short",
                "exit_long",
                "exit_short",
                "stop_long",
                "stop_short",
            ]
            for col in flag_columns:
                assert signals_df[col].dtype == bool

    class TestRunParameterSweep:
        """Test parameter sweep functionality."""

        def test_basic_sweep(self, optimizer, sample_data):
            """Test basic parameter sweep."""
            fx_series, commodity_series = sample_data

            # Use a small set of combinations for faster testing
            threshold_combinations = [
                {"entry_z": 1.0, "exit_z": 0.3, "stop_z": 2.5},
                {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
            ]

            results = optimizer.run_parameter_sweep(
                fx_series, commodity_series, threshold_combinations
            )

            # Check that results DataFrame is returned
            assert isinstance(results, pd.DataFrame)

            # Check that all combinations are tested
            assert len(results) == len(threshold_combinations)

            # Check that threshold parameters are included
            for param in ["entry_z", "exit_z", "stop_z"]:
                assert param in results.columns

            # Check that performance metrics are included
            expected_metrics = [
                "total_trades",
                "win_rate",
                "total_return",
                "annual_return",
                "annual_vol",
                "sharpe_ratio",
                "max_drawdown",
            ]
            for metric in expected_metrics:
                assert metric in results.columns

        def test_default_combinations(self, optimizer, sample_data):
            """Test parameter sweep with default combinations."""
            fx_series, commodity_series = sample_data

            results = optimizer.run_parameter_sweep(fx_series, commodity_series)

            # Should use default combinations
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0

            # Should include rank column
            assert "rank" in results.columns

        def test_failed_combinations(self, optimizer, sample_data):
            """Test handling of failed threshold combinations."""
            fx_series, commodity_series = sample_data

            # Create a combination that will fail
            bad_combinations = [
                {"entry_z": 1.0, "exit_z": 0.3, "stop_z": 2.5},
                {
                    "entry_z": -1.0,
                    "exit_z": 0.5,
                    "stop_z": 3.0,
                },  # Negative entry should cause issues
            ]

            # Should handle failed combinations gracefully
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = optimizer.run_parameter_sweep(
                    fx_series, commodity_series, bad_combinations
                )

            # Should still return results for valid combinations
            assert isinstance(results, pd.DataFrame)
            assert len(results) >= 1  # At least one combination should work

    class TestCalculateSignalPerformanceMetrics:
        """Test signal performance metrics calculation."""

        def test_basic_metrics(self, optimizer, sample_data):
            """Test basic performance metrics calculation."""
            fx_series, commodity_series = sample_data

            # Create simple signals DataFrame
            signals_df = pd.DataFrame(
                {
                    "fx_price": fx_series,
                    "comd_price": commodity_series,
                    "spread": fx_series - commodity_series,
                    "signal": pd.Series(0, index=fx_series.index),
                }
            )

            # Add some signals
            signals_df["signal"].iloc[100:200] = 1
            signals_df["signal"].iloc[300:400] = -1

            metrics = optimizer._calculate_signal_performance_metrics(signals_df)

            # Check that all expected metrics are present
            expected_metrics = [
                "total_trades",
                "win_trades",
                "win_rate",
                "total_return",
                "annual_return",
                "annual_vol",
                "sharpe_ratio",
                "max_drawdown",
            ]

            for metric in expected_metrics:
                assert metric in metrics

            # Check that metrics are reasonable
            assert metrics["total_trades"] >= 0
            assert 0 <= metrics["win_rate"] <= 1
            assert metrics["annual_vol"] >= 0
            assert metrics["max_drawdown"] <= 0

        def test_no_signals(self, optimizer, sample_data):
            """Test performance metrics with no signals."""
            fx_series, commodity_series = sample_data

            # Create signals DataFrame with no signals
            signals_df = pd.DataFrame(
                {
                    "fx_price": fx_series,
                    "comd_price": commodity_series,
                    "spread": fx_series - commodity_series,
                    "signal": pd.Series(0, index=fx_series.index),
                }
            )

            metrics = optimizer._calculate_signal_performance_metrics(signals_df)

            # Should handle no signals gracefully
            assert metrics["total_trades"] == 0
            assert metrics["win_rate"] == 0
            assert metrics["total_return"] == 0

        def test_edge_cases(self, optimizer):
            """Test edge cases in performance metrics calculation."""
            # Empty DataFrame
            empty_df = pd.DataFrame()

            metrics = optimizer._calculate_signal_performance_metrics(empty_df)

            # Should return empty metrics
            assert isinstance(metrics, dict)

            # DataFrame with NaN values
            dates = pd.date_range("2020-01-01", "2020-01-10")
            nan_df = pd.DataFrame(
                {
                    "fx_price": pd.Series([np.nan] * 10, index=dates),
                    "comd_price": pd.Series([np.nan] * 10, index=dates),
                    "spread": pd.Series([np.nan] * 10, index=dates),
                    "signal": pd.Series([0] * 10, index=dates),
                }
            )

            metrics = optimizer._calculate_signal_performance_metrics(nan_df)

            # Should handle NaN values gracefully
            assert isinstance(metrics, dict)


class TestSignalOptimizationIntegration:
    """Integration tests for signal optimization."""

    def test_end_to_end_workflow(self, sample_config, sample_data):
        """Test end-to-end signal optimization workflow."""
        fx_series, commodity_series = sample_data

        optimizer = SignalOptimizer(sample_config)

        # Generate threshold combinations
        combinations = optimizer.generate_threshold_combinations()

        # Run parameter sweep
        results = optimizer.run_parameter_sweep(
            fx_series, commodity_series, combinations[:3]
        )  # Test with 3 combinations

        # Check that results are valid
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

        # Find best combination
        best_result = results.loc[results["sharpe_ratio"].idxmax()]

        # Generate signals with best parameters
        best_thresholds = {
            "entry_z": best_result["entry_z"],
            "exit_z": best_result["exit_z"],
            "stop_z": best_result["stop_z"],
        }

        signals_df = optimizer.generate_enhanced_signals(
            fx_series, commodity_series, best_thresholds
        )

        # Check that signals are valid
        assert isinstance(signals_df, pd.DataFrame)
        assert signals_df["signal"].isin([-1, 0, 1]).all()

        # Calculate signal quality metrics
        if len(signals_df) > 20:
            spread_returns = signals_df["spread"].diff().shift(-1)
            quality_metrics = optimizer.calculate_signal_quality_metrics(
                signals_df["signal"], spread_returns, window=20
            )

            # Check that quality metrics are calculated
            assert isinstance(quality_metrics, pd.DataFrame)
            assert len(quality_metrics) == len(signals_df)

    def test_lookahead_bias_prevention(self, sample_config, sample_data):
        """Test that look-ahead bias is prevented."""
        fx_series, commodity_series = sample_data

        optimizer = SignalOptimizer(sample_config)

        thresholds = {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0}

        # Generate signals
        signals_df = optimizer.generate_enhanced_signals(
            fx_series, commodity_series, thresholds
        )

        # Check that signals don't use future information
        # This is a simple check - more sophisticated tests would be needed
        # to fully validate look-ahead bias prevention

        # Check that early signals don't depend on late data
        early_signals = signals_df.iloc[:100]
        late_signals = signals_df.iloc[-100:]

        # Early signals should be based on early data only
        assert not early_signals["signal"].isna().all()

        # The distribution of signals should be somewhat consistent
        # (this is a heuristic test)
        early_signal_mean = early_signals["signal"].abs().mean()
        late_signal_mean = late_signals["signal"].abs().mean()

        # They shouldn't be drastically different
        assert abs(early_signal_mean - late_signal_mean) < 0.5
