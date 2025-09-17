"""
Unit tests for the ensemble model framework.
"""

import unittest
import numpy as np
import pandas as pd
from src.ml.ensemble import (
    OLSModel,
    KalmanModel,
    RollingCorrelationModel,
    EnsembleModel,
    ModelConfig,
    create_default_model_config,
)


class TestOLSModel(unittest.TestCase):
    """Test cases for the OLS model."""

    def setUp(self):
        """Set up test data."""
        # Create simple linear relationship data
        np.random.seed(42)
        self.X = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        # y = 2*x1 + 3*x2 + noise
        self.y = (
            2 * self.X["feature1"] + 3 * self.X["feature2"] + np.random.randn(100) * 0.1
        )

    def test_ols_model_initialization(self):
        """Test OLS model initialization."""
        model = OLSModel(window=50)
        self.assertEqual(model.window, 50)
        self.assertEqual(model.name, "ols")
        self.assertFalse(model.is_trained)

    def test_ols_model_fit_predict(self):
        """Test OLS model fitting and prediction."""
        model = OLSModel(window=50)

        # Fit the model
        model.fit(self.X, self.y)
        self.assertTrue(model.is_trained)

        # Make predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Check that predictions are reasonable (not all zeros)
        self.assertFalse(np.allclose(predictions, 0))

    def test_ols_model_feature_importance(self):
        """Test OLS model feature importance."""
        model = OLSModel(window=50)
        model.fit(self.X, self.y)

        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 2)  # Two features


class TestKalmanModel(unittest.TestCase):
    """Test cases for the Kalman model."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create simple linear relationship data
        self.X = pd.DataFrame({"feature": np.random.randn(100)})
        # y = 2*x + noise
        self.y = 2 * self.X["feature"] + np.random.randn(100) * 0.1

    def test_kalman_model_initialization(self):
        """Test Kalman model initialization."""
        model = KalmanModel(lam=0.99, delta=50.0)
        self.assertEqual(model.lam, 0.99)
        self.assertEqual(model.delta, 50.0)
        self.assertEqual(model.name, "kalman")
        self.assertFalse(model.is_trained)

    def test_kalman_model_fit_predict(self):
        """Test Kalman model fitting and prediction."""
        model = KalmanModel()

        # Fit the model
        model.fit(self.X, self.y)
        self.assertTrue(model.is_trained)

        # Make predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

    def test_kalman_model_feature_importance(self):
        """Test Kalman model feature importance."""
        model = KalmanModel()
        model.fit(self.X, self.y)

        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 2)  # Intercept and slope


class TestRollingCorrelationModel(unittest.TestCase):
    """Test cases for the rolling correlation model."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({"feature": np.random.randn(100)})
        self.y = (
            self.X["feature"] * 0.8 + np.random.randn(100) * 0.2
        )  # Correlated with noise

    def test_correlation_model_initialization(self):
        """Test correlation model initialization."""
        model = RollingCorrelationModel(window=10)
        self.assertEqual(model.window, 10)
        self.assertEqual(model.name, "corr")
        self.assertFalse(model.is_trained)

    def test_correlation_model_fit_predict(self):
        """Test correlation model fitting and prediction."""
        model = RollingCorrelationModel(window=20)

        # Fit the model
        model.fit(self.X, self.y)
        self.assertTrue(model.is_trained)

        # Make predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

    def test_correlation_model_feature_importance(self):
        """Test correlation model feature importance."""
        model = RollingCorrelationModel(window=20)
        model.fit(self.X, self.y)

        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 1)  # Single correlation value


class TestEnsembleModel(unittest.TestCase):
    """Test cases for the ensemble model."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        self.y = (
            2 * self.X["feature1"] + 3 * self.X["feature2"] + np.random.randn(100) * 0.1
        )

        # Create config
        self.config = ModelConfig(
            ols_window=50,
            kalman_lambda=0.99,
            corr_window=20,
            model_weights={"ols": 0.4, "kalman": 0.3, "corr": 0.3},
        )

    def test_ensemble_model_initialization(self):
        """Test ensemble model initialization."""
        ensemble = EnsembleModel(self.config)
        self.assertIsNotNone(ensemble.models)
        self.assertIn("ols", ensemble.models)
        self.assertIn("kalman", ensemble.models)
        self.assertIn("corr", ensemble.models)

    def test_ensemble_model_fit(self):
        """Test ensemble model fitting."""
        ensemble = EnsembleModel(self.config)
        scores = ensemble.fit(self.X, self.y)

        # Check that all models were attempted
        self.assertIn("ols", scores)
        self.assertIn("kalman", scores)
        self.assertIn("corr", scores)

    def test_ensemble_model_predict(self):
        """Test ensemble model predictions."""
        ensemble = EnsembleModel(self.config)
        ensemble.fit(self.X, self.y)

        # Get individual predictions
        predictions = ensemble.predict(self.X)
        self.assertIn("ols", predictions)
        self.assertIn("kalman", predictions)
        self.assertIn("corr", predictions)

        # Check prediction lengths
        for pred in predictions.values():
            self.assertEqual(len(pred), len(self.X))

    def test_ensemble_model_predict_ensemble(self):
        """Test ensemble model ensemble prediction."""
        ensemble = EnsembleModel(self.config)
        ensemble.fit(self.X, self.y)

        # Get ensemble prediction
        ensemble_pred = ensemble.predict_ensemble(self.X)
        self.assertEqual(len(ensemble_pred), len(self.X))

        # Should not be all zeros
        self.assertFalse(np.allclose(ensemble_pred, 0))

    def test_ensemble_model_feature_importance(self):
        """Test ensemble model feature importance."""
        ensemble = EnsembleModel(self.config)
        ensemble.fit(self.X, self.y)

        importances = ensemble.get_feature_importance()
        self.assertIn("ols", importances)
        self.assertIn("kalman", importances)
        self.assertIn("corr", importances)


class TestModelConfig(unittest.TestCase):
    """Test cases for the model configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = create_default_model_config()
        self.assertIsInstance(config, ModelConfig)

        # Check default values
        self.assertEqual(config.ols_window, 90)
        self.assertEqual(config.kalman_lambda, 0.995)
        self.assertEqual(config.corr_window, 20)

        # Check default weights
        self.assertIn("ols", config.model_weights)
        self.assertIn("kalman", config.model_weights)
        self.assertIn("corr", config.model_weights)
        self.assertIn("gb", config.model_weights)
        self.assertIn("lstm", config.model_weights)

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            ols_window=60,
            kalman_lambda=0.99,
            corr_window=15,
            model_weights={"ols": 0.5, "kalman": 0.5},
        )

        self.assertEqual(config.ols_window, 60)
        self.assertEqual(config.kalman_lambda, 0.99)
        self.assertEqual(config.corr_window, 15)
        self.assertEqual(config.model_weights["ols"], 0.5)
        self.assertEqual(config.model_weights["kalman"], 0.5)


if __name__ == "__main__":
    unittest.main()
