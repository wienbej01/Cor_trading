"""
Unit tests for Model Diversification Module.

Tests for:
- OLS model wrapper
- Kalman filter model wrapper
- Rolling correlation model wrapper
- ML-based residual prediction models (LSTM, Gradient Boosted Trees)
- Ensemble weighting mechanism
- Parallel backtesting framework
- Feature importance calculations
- Model diagnostics
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from ml.ensemble import (
    EnsembleModel,
    BaseModel,
    OLSModel,
    KalmanModel,
    RollingCorrelationModel,
)
from ml.diagnostics import MLDiagnostics
from backtest.parallel import ParallelBacktester
from test_utils import generate_synthetic_market_data, CustomAssertions


class TestModelWrapper:
    """Test base ModelWrapper class."""

    def test_abstract_methods(self):
        """Test that ModelWrapper has abstract methods."""
        # This test ensures that ModelWrapper cannot be instantiated directly
        with pytest.raises(TypeError):
            ModelWrapper()


class TestOLSModelWrapper:
    """Test OLSModelWrapper class."""

    @pytest.fixture
    def ols_model(self):
        """Create an OLSModelWrapper instance for testing."""
        return OLSModelWrapper()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000

        # Create correlated series
        x = np.random.normal(0, 1, n_samples)
        y = 0.5 * x + np.random.normal(0, 0.5, n_samples)

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)

        return x_series, y_series

    def test_initialization(self, ols_model):
        """Test OLSModelWrapper initialization."""
        assert ols_model is not None
        assert ols_model.model is None
        assert ols_model.is_fitted is False

    def test_fit(self, ols_model, sample_data):
        """Test model fitting."""
        x_series, y_series = sample_data

        ols_model.fit(x_series, y_series)

        # Check that model is fitted
        assert ols_model.is_fitted is True
        assert ols_model.model is not None

    def test_predict(self, ols_model, sample_data):
        """Test model prediction."""
        x_series, y_series = sample_data

        # Fit the model first
        ols_model.fit(x_series, y_series)

        # Make predictions
        predictions = ols_model.predict(x_series)

        # Check that predictions are returned
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(x_series)
        assert predictions.index.equals(x_series.index)

        # Check that predictions are reasonable
        assert not predictions.isna().all()
        assert predictions.std() > 0  # Should have some variation

    def test_predict_before_fit(self, ols_model, sample_data):
        """Test prediction before model is fitted."""
        x_series, y_series = sample_data

        # Should raise an error if model is not fitted
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            ols_model.predict(x_series)

    def test_get_coefficients(self, ols_model, sample_data):
        """Test getting model coefficients."""
        x_series, y_series = sample_data

        # Fit the model first
        ols_model.fit(x_series, y_series)

        # Get coefficients
        coefficients = ols_model.get_coefficients()

        # Check that coefficients are returned
        assert isinstance(coefficients, dict)
        assert "intercept" in coefficients
        assert "slope" in coefficients

        # Check that coefficients are reasonable
        assert isinstance(coefficients["intercept"], float)
        assert isinstance(coefficients["slope"], float)

    def test_get_coefficients_before_fit(self, ols_model):
        """Test getting coefficients before model is fitted."""
        # Should raise an error if model is not fitted
        with pytest.raises(
            ValueError, match="Model must be fitted before getting coefficients"
        ):
            ols_model.get_coefficients()

    def test_get_r_squared(self, ols_model, sample_data):
        """Test getting R-squared value."""
        x_series, y_series = sample_data

        # Fit the model first
        ols_model.fit(x_series, y_series)

        # Get R-squared
        r_squared = ols_model.get_r_squared()

        # Check that R-squared is returned
        assert isinstance(r_squared, float)
        assert 0 <= r_squared <= 1  # R-squared should be between 0 and 1

    def test_get_r_squared_before_fit(self, ols_model):
        """Test getting R-squared before model is fitted."""
        # Should raise an error if model is not fitted
        with pytest.raises(
            ValueError, match="Model must be fitted before getting R-squared"
        ):
            ols_model.get_r_squared()

    def test_with_nan_values(self, ols_model):
        """Test handling of NaN values."""
        # Create data with NaN values
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        x_series = pd.Series(np.random.normal(0, 1, 100), index=dates)
        y_series = pd.Series(np.random.normal(0, 1, 100), index=dates)

        # Add NaN values
        x_series.iloc[10:15] = np.nan
        y_series.iloc[20:25] = np.nan

        # Should handle NaN values gracefully
        ols_model.fit(x_series, y_series)
        assert ols_model.is_fitted is True

        predictions = ols_model.predict(x_series)
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(x_series)


class TestKalmanModel:
    """Test KalmanModel class."""

    @pytest.fixture
    def kalman_model(self):
        """Create a KalmanModel instance for testing."""
        return KalmanModel(delta=1e-5, lam=0.995)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000

        # Create time-varying relationship
        t = np.arange(n_samples)
        beta = 0.5 + 0.3 * np.sin(2 * np.pi * t / 200)  # Time-varying beta

        x = np.random.normal(0, 1, n_samples)
        y = beta * x + np.random.normal(0, 0.5, n_samples)

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)

        return x_series, y_series

    def test_initialization(self, kalman_model):
        """Test KalmanModelWrapper initialization."""
        assert kalman_model is not None
        assert kalman_model.delta == 1e-5
        assert kalman_model.lam == 0.995
        assert kalman_model.is_fitted is False

    def test_fit(self, kalman_model, sample_data):
        """Test model fitting."""
        x_series, y_series = sample_data

        kalman_model.fit(x_series, y_series)

        # Check that model is fitted
        assert kalman_model.is_fitted is True

    def test_predict(self, kalman_model, sample_data):
        """Test model prediction."""
        x_series, y_series = sample_data

        # Fit the model first
        kalman_model.fit(x_series, y_series)

        # Make predictions
        predictions = kalman_model.predict(x_series)

        # Check that predictions are returned
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(x_series)
        assert predictions.index.equals(x_series.index)

        # Check that predictions are reasonable
        assert not predictions.isna().all()
        assert predictions.std() > 0  # Should have some variation

    def test_get_time_varying_coefficients(self, kalman_model, sample_data):
        """Test getting time-varying coefficients."""
        x_series, y_series = sample_data

        # Fit the model first
        kalman_model.fit(x_series, y_series)

        # Get coefficients
        coefficients = kalman_model.get_time_varying_coefficients()

        # Check that coefficients are returned
        assert isinstance(coefficients, pd.DataFrame)
        assert "alpha" in coefficients.columns
        assert "beta" in coefficients.columns

        # Check that coefficients have correct index
        assert coefficients.index.equals(x_series.index)

        # Check that coefficients vary over time (should for Kalman filter)
        assert coefficients["beta"].std() > 0.01  # Should have some variation

    def test_different_parameters(self, sample_data):
        """Test with different Kalman parameters."""
        x_series, y_series = sample_data

        # Test with different delta values
        kalman_fast = KalmanModelWrapper(delta=1e-3, lam=0.99)  # Faster adaptation
        kalman_slow = KalmanModelWrapper(delta=1e-7, lam=0.999)  # Slower adaptation

        # Fit both models
        kalman_fast.fit(x_series, y_series)
        kalman_slow.fit(x_series, y_series)

        # Get coefficients
        fast_coef = kalman_fast.get_time_varying_coefficients()
        slow_coef = kalman_slow.get_time_varying_coefficients()

        # Fast adaptation should have more variable coefficients
        assert fast_coef["beta"].std() > slow_coef["beta"].std()


class TestRollingCorrelationModelWrapper:
    """Test RollingCorrelationModelWrapper class."""

    @pytest.fixture
    def rolling_model(self):
        """Create a RollingCorrelationModelWrapper instance for testing."""
        return RollingCorrelationModelWrapper(window=60)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000

        # Create series with changing correlation
        t = np.arange(n_samples)
        corr = 0.8 + 0.2 * np.sin(2 * np.pi * t / 300)  # Time-varying correlation

        x = np.random.normal(0, 1, n_samples)
        y = corr * x + np.sqrt(1 - corr**2) * np.random.normal(0, 1, n_samples)

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)

        return x_series, y_series

    def test_initialization(self, rolling_model):
        """Test RollingCorrelationModelWrapper initialization."""
        assert rolling_model is not None
        assert rolling_model.window == 60
        assert rolling_model.is_fitted is False

    def test_fit(self, rolling_model, sample_data):
        """Test model fitting."""
        x_series, y_series = sample_data

        rolling_model.fit(x_series, y_series)

        # Check that model is fitted
        assert rolling_model.is_fitted is True

    def test_predict(self, rolling_model, sample_data):
        """Test model prediction."""
        x_series, y_series = sample_data

        # Fit the model first
        rolling_model.fit(x_series, y_series)

        # Make predictions
        predictions = rolling_model.predict(x_series)

        # Check that predictions are returned
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(x_series)
        assert predictions.index.equals(x_series.index)

        # Check that predictions are reasonable
        # Early values should be NaN due to rolling window
        assert predictions.iloc[: rolling_model.window - 1].isna().all()
        assert not predictions.iloc[rolling_model.window :].isna().all()

    def test_get_rolling_correlation(self, rolling_model, sample_data):
        """Test getting rolling correlation."""
        x_series, y_series = sample_data

        # Fit the model first
        rolling_model.fit(x_series, y_series)

        # Get rolling correlation
        correlation = rolling_model.get_rolling_correlation()

        # Check that correlation is returned
        assert isinstance(correlation, pd.Series)
        assert len(correlation) == len(x_series)
        assert correlation.index.equals(x_series.index)

        # Check that correlation values are reasonable
        valid_corr = correlation.dropna()
        assert (-1 <= valid_corr).all()
        assert (valid_corr <= 1).all()

        # Should have some variation
        assert valid_corr.std() > 0.01

    def test_different_window_sizes(self, sample_data):
        """Test with different window sizes."""
        x_series, y_series = sample_data

        # Test with different window sizes
        model_short = RollingCorrelationModelWrapper(window=20)
        model_long = RollingCorrelationModelWrapper(window=120)

        # Fit both models
        model_short.fit(x_series, y_series)
        model_long.fit(x_series, y_series)

        # Get correlations
        corr_short = model_short.get_rolling_correlation()
        corr_long = model_long.get_rolling_correlation()

        # Shorter window should have more variable correlation
        valid_short = corr_short.dropna()
        valid_long = corr_long.dropna()

        assert valid_short.std() > valid_long.std()


class TestEnsembleModel:
    """Test EnsembleModel class."""

    @pytest.fixture
    def ensemble(self):
        """Create a EnsembleModel instance for testing."""
        models = {
            "ols": OLSModel(),
            "kalman": KalmanModel(),
            "rolling": RollingCorrelationModel(window=60),
        }
        return EnsembleModel(models, weights=None)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    def test_initialization(self, ensemble):
        """Test ModelEnsemble initialization."""
        assert ensemble is not None
        assert len(ensemble.models) == 3
        assert "ols" in ensemble.models
        assert "kalman" in ensemble.models
        assert "rolling" in ensemble.models

        # Default weights should be equal
        expected_weights = {"ols": 1 / 3, "kalman": 1 / 3, "rolling": 1 / 3}
        for model, weight in ensemble.weights.items():
            assert abs(weight - expected_weights[model]) < 1e-10

    def test_custom_weights(self):
        """Test ModelEnsemble with custom weights."""
        models = {"ols": OLSModelWrapper(), "kalman": KalmanModelWrapper()}
        custom_weights = {"ols": 0.7, "kalman": 0.3}

        ensemble = ModelEnsemble(models, weights=custom_weights)

        # Check that custom weights are used
        assert ensemble.weights == custom_weights

    def test_invalid_weights(self):
        """Test ModelEnsemble with invalid weights."""
        models = {"ols": OLSModelWrapper(), "kalman": KalmanModelWrapper()}

        # Test weights that don't sum to 1
        invalid_weights = {"ols": 0.7, "kalman": 0.5}  # Sum = 1.2

        # Should normalize weights
        ensemble = ModelEnsemble(models, weights=invalid_weights)
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-10

    def test_fit(self, ensemble, sample_data):
        """Test ensemble fitting."""
        x_series, y_series = sample_data

        ensemble.fit(x_series, y_series)

        # Check that all models are fitted
        for model in ensemble.models.values():
            assert model.is_fitted is True

    def test_predict(self, ensemble, sample_data):
        """Test ensemble prediction."""
        x_series, y_series = sample_data

        # Fit the ensemble first
        ensemble.fit(x_series, y_series)

        # Make predictions
        predictions = ensemble.predict(x_series)

        # Check that predictions are returned
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(x_series)
        assert predictions.index.equals(x_series.index)

        # Check that predictions are reasonable
        assert not predictions.isna().all()
        assert predictions.std() > 0  # Should have some variation

    def test_get_individual_predictions(self, ensemble, sample_data):
        """Test getting individual model predictions."""
        x_series, y_series = sample_data

        # Fit the ensemble first
        ensemble.fit(x_series, y_series)

        # Get individual predictions
        individual_preds = ensemble.get_individual_predictions(x_series)

        # Check that predictions are returned for each model
        assert isinstance(individual_preds, dict)
        assert len(individual_preds) == len(ensemble.models)

        for model_name, preds in individual_preds.items():
            assert model_name in ensemble.models
            assert isinstance(preds, pd.Series)
            assert len(preds) == len(x_series)
            assert preds.index.equals(x_series.index)

    def test_update_weights(self, ensemble, sample_data):
        """Test updating ensemble weights."""
        x_series, y_series = sample_data

        # Fit the ensemble first
        ensemble.fit(x_series, y_series)

        # Get initial weights
        initial_weights = ensemble.weights.copy()

        # Update weights based on recent performance
        new_weights = ensemble.update_weights(x_series, y_series, window=100)

        # Check that new weights are returned
        assert isinstance(new_weights, dict)
        assert len(new_weights) == len(ensemble.models)

        # Check that weights sum to 1
        assert abs(sum(new_weights.values()) - 1.0) < 1e-10

        # Check that ensemble weights are updated
        assert ensemble.weights == new_weights

        # Weights might be different from initial
        # (though they might be the same if all models perform similarly)

    def test_performance_based_weighting(self, sample_data):
        """Test performance-based weighting mechanism."""
        x_series, y_series = sample_data

        # Create models with different expected performance
        models = {
            "good_model": OLSModelWrapper(),
            "bad_model": OLSModelWrapper(),  # This will perform the same, but we'll test the mechanism
        }

        ensemble = ModelEnsemble(models)
        ensemble.fit(x_series, y_series)

        # Update weights
        new_weights = ensemble.update_weights(x_series, y_series, window=100)

        # Check that weights are updated
        assert isinstance(new_weights, dict)
        assert len(new_weights) == 2
        assert abs(sum(new_weights.values()) - 1.0) < 1e-10


class TestMLDiagnostics:
    """Test MLDiagnostics class."""

    @pytest.fixture
    def diagnostics(self):
        """Create a MLDiagnostics instance for testing."""
        return MLDiagnostics()

    @pytest.fixture
    def sample_model(self):
        """Create a fitted model for testing."""
        np.random.seed(42)
        n_samples = 1000

        x = np.random.normal(0, 1, n_samples)
        y = 0.5 * x + np.random.normal(0, 0.5, n_samples)

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)

        model = OLSModelWrapper()
        model.fit(x_series, y_series)

        return model, x_series, y_series

    def test_initialization(self, diagnostics):
        """Test ModelDiagnostics initialization."""
        assert diagnostics is not None

    def test_calculate_residuals(self, diagnostics, sample_model):
        """Test residual calculation."""
        model, x_series, y_series = sample_model

        residuals = diagnostics.calculate_residuals(model, x_series, y_series)

        # Check that residuals are returned
        assert isinstance(residuals, pd.Series)
        assert len(residuals) == len(x_series)
        assert residuals.index.equals(x_series.index)

        # Check that residuals have reasonable properties
        assert not residuals.isna().all()
        assert abs(residuals.mean()) < 0.1  # Should be close to zero for OLS

    def test_calculate_prediction_errors(self, diagnostics, sample_model):
        """Test prediction error calculation."""
        model, x_series, y_series = sample_model

        errors = diagnostics.calculate_prediction_errors(model, x_series, y_series)

        # Check that errors are returned
        assert isinstance(errors, pd.Series)
        assert len(errors) == len(x_series)
        assert errors.index.equals(x_series.index)

        # Check that errors have reasonable properties
        assert not errors.isna().all()
        assert errors.std() > 0  # Should have some variation

    def test_analyze_residuals(self, diagnostics, sample_model):
        """Test residual analysis."""
        model, x_series, y_series = sample_model

        analysis = diagnostics.analyze_residuals(model, x_series, y_series)

        # Check that analysis is returned
        assert isinstance(analysis, dict)

        # Check that all expected metrics are present
        expected_metrics = [
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "jarque_bera",
            "ljung_box",
        ]
        for metric in expected_metrics:
            assert metric in analysis

        # Check that values are reasonable
        assert isinstance(analysis["mean"], float)
        assert isinstance(analysis["std"], float)
        assert isinstance(analysis["skewness"], float)
        assert isinstance(analysis["kurtosis"], float)
        assert isinstance(
            analysis["jarque_bera"], (float, tuple)
        )  # Can be tuple from scipy
        assert isinstance(
            analysis["ljung_box"], (float, tuple)
        )  # Can be tuple from scipy

    def test_check_model_stability(self, diagnostics, sample_model):
        """Test model stability checking."""
        model, x_series, y_series = sample_model

        stability = diagnostics.check_model_stability(
            model, x_series, y_series, window=100
        )

        # Check that stability metrics are returned
        assert isinstance(stability, dict)

        # Check that all expected metrics are present
        expected_metrics = [
            "coefficient_stability",
            "r_squared_stability",
            "prediction_stability",
        ]
        for metric in expected_metrics:
            assert metric in stability

        # Check that values are reasonable
        for metric, value in stability.items():
            assert isinstance(value, float)
            assert 0 <= value <= 1  # Stability metrics should be between 0 and 1

    def test_compare_models(self, diagnostics, sample_data):
        """Test model comparison."""
        x_series, y_series = sample_data

        # Create two models
        model1 = OLSModelWrapper()
        model2 = KalmanModelWrapper()

        # Fit both models
        model1.fit(x_series, y_series)
        model2.fit(x_series, y_series)

        # Compare models
        comparison = diagnostics.compare_models(
            {"OLS": model1, "Kalman": model2}, x_series, y_series
        )

        # Check that comparison is returned
        assert isinstance(comparison, dict)
        assert len(comparison) == 2
        assert "OLS" in comparison
        assert "Kalman" in comparison

        # Check that comparison metrics are reasonable
        for model_name, metrics in comparison.items():
            assert isinstance(metrics, dict)
            assert "r_squared" in metrics
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "aic" in metrics
            assert "bic" in metrics

            # Check that values are reasonable
            assert isinstance(metrics["r_squared"], float)
            assert 0 <= metrics["r_squared"] <= 1
            assert metrics["rmse"] >= 0
            assert metrics["mae"] >= 0


class TestParallelBacktester:
    """Test ParallelBacktester class."""

    @pytest.fixture
    def parallel_backtest(self):
        """Create a ParallelBacktester instance for testing."""
        return ParallelBacktester(n_workers=2)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    @pytest.fixture
    def sample_configs(self):
        """Create sample configurations for testing."""
        return [
            {"entry_z": 1.0, "exit_z": 0.3, "stop_z": 2.5},
            {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
            {"entry_z": 2.0, "exit_z": 0.7, "stop_z": 3.5},
        ]

    def test_initialization(self, parallel_backtest):
        """Test ParallelBacktest initialization."""
        assert parallel_backtest is not None
        assert parallel_backtest.n_workers == 2

    def test_run_parallel_backtests(
        self, parallel_backtest, sample_data, sample_configs
    ):
        """Test running parallel backtests."""
        x_series, y_series = sample_data

        # Mock backtest function
        def mock_backtest(config, x_data, y_data):
            # Simple mock that returns some metrics
            return {
                "config": config,
                "sharpe_ratio": np.random.uniform(0.5, 2.0),
                "total_return": np.random.uniform(0.1, 0.3),
                "max_drawdown": np.random.uniform(-0.2, -0.05),
            }

        results = parallel_backtest.run_parallel_backtests(
            mock_backtest, sample_configs, x_series, y_series
        )

        # Check that results are returned
        assert isinstance(results, list)
        assert len(results) == len(sample_configs)

        # Check that each result contains expected data
        for result in results:
            assert isinstance(result, dict)
            assert "config" in result
            assert "sharpe_ratio" in result
            assert "total_return" in result
            assert "max_drawdown" in result

    def test_parameter_sweep(self, parallel_backtest, sample_data):
        """Test parameter sweep functionality."""
        x_series, y_series = sample_data

        # Create parameter grid
        param_grid = {
            "entry_z": [1.0, 1.5, 2.0],
            "exit_z": [0.3, 0.5],
            "stop_z": [2.5, 3.0],
        }

        # Mock backtest function
        def mock_backtest(config, x_data, y_data):
            return {
                "config": config,
                "sharpe_ratio": np.random.uniform(0.5, 2.0),
                "total_return": np.random.uniform(0.1, 0.3),
            }

        results = parallel_backtest.parameter_sweep(
            mock_backtest, param_grid, x_series, y_series
        )

        # Check that results are returned
        assert isinstance(results, list)

        # Should have results for all parameter combinations
        expected_combinations = 3 * 2 * 2  # entry_z * exit_z * stop_z
        assert len(results) == expected_combinations

        # Check that each result has a unique config
        configs = [result["config"] for result in results]
        assert len(configs) == len(set(str(config) for config in configs))  # All unique

    def test_error_handling(self, parallel_backtest, sample_data):
        """Test error handling in parallel backtests."""
        x_series, y_series = sample_data

        # Create configs where one will fail
        good_configs = [
            {"entry_z": 1.0, "exit_z": 0.3, "stop_z": 2.5},
            {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        ]

        # Mock backtest function that fails for certain configs
        def mock_backtest(config, x_data, y_data):
            if config["entry_z"] == 1.5:
                raise ValueError("Test error")
            return {"config": config, "sharpe_ratio": 1.0, "total_return": 0.2}

        # Should handle errors gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = parallel_backtest.run_parallel_backtests(
                mock_backtest, good_configs, x_series, y_series
            )

        # Should still return results for successful configs
        assert isinstance(results, list)
        assert len(results) == 1  # Only one config succeeded
        assert results[0]["config"]["entry_z"] == 1.0  # The successful one


class TestModelDiversificationIntegration:
    """Integration tests for model diversification components."""

    def test_end_to_end_workflow(self, sample_data):
        """Test end-to-end model diversification workflow."""
        x_series, y_series = sample_data

        # Create models
        models = {
            "ols": OLSModelWrapper(),
            "kalman": KalmanModelWrapper(),
            "rolling": RollingCorrelationModelWrapper(window=60),
        }

        # Create ensemble
        ensemble = ModelEnsemble(models)

        # Fit ensemble
        ensemble.fit(x_series, y_series)

        # Make predictions
        predictions = ensemble.predict(x_series)

        # Get individual predictions
        individual_preds = ensemble.get_individual_predictions(x_series)

        # Update weights based on performance
        new_weights = ensemble.update_weights(x_series, y_series, window=100)

        # Make predictions with updated weights
        updated_predictions = ensemble.predict(x_series)

        # Check that all steps work
        assert isinstance(predictions, pd.Series)
        assert isinstance(individual_preds, dict)
        assert isinstance(new_weights, dict)
        assert isinstance(updated_predictions, pd.Series)

        # Check that weights are updated
        assert ensemble.weights == new_weights

        # Check that predictions might be different after weight update
        # (though they might be the same if all models perform similarly)
        assert len(predictions) == len(updated_predictions)

    def test_model_selection_with_diagnostics(self, sample_data):
        """Test model selection using diagnostics."""
        x_series, y_series = sample_data

        # Create models
        models = {
            "ols": OLSModelWrapper(),
            "kalman": KalmanModelWrapper(),
            "rolling": RollingCorrelationModelWrapper(window=60),
        }

        # Fit all models
        for model in models.values():
            model.fit(x_series, y_series)

        # Create diagnostics
        diagnostics = ModelDiagnostics()

        # Compare models
        comparison = diagnostics.compare_models(models, x_series, y_series)

        # Select best model based on R-squared
        best_model_name = max(comparison.items(), key=lambda x: x[1]["r_squared"])[0]
        best_model = models[best_model_name]

        # Check that best model is selected
        assert best_model_name in models
        assert best_model.is_fitted is True

        # Use best model for prediction
        best_predictions = best_model.predict(x_series)

        # Check that predictions are valid
        assert isinstance(best_predictions, pd.Series)
        assert len(best_predictions) == len(x_series)

    def test_parallel_model_evaluation(self, sample_data):
        """Test parallel evaluation of multiple models."""
        x_series, y_series = sample_data

        # Create model configurations
        model_configs = [
            {"name": "ols", "model": OLSModelWrapper()},
            {"name": "kalman_delta_1e-5", "model": KalmanModelWrapper(delta=1e-5)},
            {"name": "kalman_delta_1e-6", "model": KalmanModelWrapper(delta=1e-6)},
            {"name": "rolling_30", "model": RollingCorrelationModelWrapper(window=30)},
            {"name": "rolling_60", "model": RollingCorrelationModelWrapper(window=60)},
        ]

        # Create parallel backtest
        parallel_backtest = ParallelBacktest(n_workers=2)

        # Mock evaluation function
        def evaluate_model(config, x_data, y_data):
            model = config["model"]
            model.fit(x_data, y_data)
            predictions = model.predict(x_data)

            # Calculate simple metrics
            errors = y_data - predictions
            rmse = np.sqrt(np.mean(errors**2))
            r_squared = 1 - np.sum(errors**2) / np.sum((y_data - y_data.mean()) ** 2)

            return {
                "config_name": config["name"],
                "rmse": rmse,
                "r_squared": r_squared,
                "model": model,
            }

        # Evaluate models in parallel
        results = parallel_backtest.run_parallel_backtests(
            evaluate_model, model_configs, x_series, y_series
        )

        # Check that all models are evaluated
        assert len(results) == len(model_configs)

        # Find best model
        best_result = max(results, key=lambda x: x["r_squared"])

        # Check that best model is selected
        assert "config_name" in best_result
        assert "rmse" in best_result
        assert "r_squared" in best_result
        assert "model" in best_result

        # Use best model for prediction
        best_predictions = best_result["model"].predict(x_series)

        # Check that predictions are valid
        assert isinstance(best_predictions, pd.Series)
        assert len(best_predictions) == len(x_series)
