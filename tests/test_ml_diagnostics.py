"""
Unit tests for the ML diagnostics module.
"""

import unittest
import numpy as np
import pandas as pd
from ml.diagnostics import MLDiagnostics, calculate_permutation_importance


class TestMLDiagnostics(unittest.TestCase):
    """Test cases for the ML diagnostics."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        self.y = (
            2 * self.X["feature1"] + 3 * self.X["feature2"] + np.random.randn(100) * 0.1
        )

    def test_diagnostics_initialization(self):
        """Test diagnostics initialization."""
        diagnostics = MLDiagnostics()
        self.assertIsInstance(diagnostics, MLDiagnostics)
        self.assertEqual(len(diagnostics.feature_importances), 0)
        self.assertEqual(len(diagnostics.model_performance), 0)
        self.assertEqual(len(diagnostics.shap_values), 0)

    def test_calculate_feature_importance_with_coef(self):
        """Test feature importance calculation with coefficients."""

        # Create a mock linear model with coef_
        class MockLinearModel:
            def __init__(self):
                self.coef_ = np.array([2.0, 3.0, 1.5])

        model = MockLinearModel()
        diagnostics = MLDiagnostics()

        importance = diagnostics.calculate_feature_importance(
            model, "linear_model", self.X
        )

        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 3)
        self.assertEqual(importance.name, "linear_model")
        # Should be sorted by absolute coefficient values
        self.assertEqual(importance.iloc[0], 3.0)  # feature2 has highest coef

    def test_calculate_feature_importance_with_feature_importances(self):
        """Test feature importance calculation with feature_importances_."""

        # Create a mock tree model with feature_importances_
        class MockTreeModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.5, 0.3, 0.2])

        model = MockTreeModel()
        diagnostics = MLDiagnostics()

        importance = diagnostics.calculate_feature_importance(
            model, "tree_model", self.X
        )

        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 3)
        self.assertEqual(importance.name, "tree_model")
        # Should match the feature importances
        np.testing.assert_array_equal(importance.values, [0.5, 0.3, 0.2])

    def test_calculate_model_performance(self):
        """Test model performance calculation."""

        # Create a mock model with predict method
        class MockModel:
            def predict(self, X):
                # Simple prediction: sum of features
                return X.sum(axis=1)

        model = MockModel()
        diagnostics = MLDiagnostics()

        # Calculate performance
        metrics = diagnostics.calculate_model_performance(
            model, "mock_model", self.X, self.y
        )

        self.assertIsInstance(metrics, dict)
        # Should have common metrics
        self.assertIn("mse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

        # Values should be numeric
        for value in metrics.values():
            self.assertIsInstance(value, (int, float))

    def test_get_diagnostics_report(self):
        """Test getting diagnostics report."""
        diagnostics = MLDiagnostics()

        # Add some mock data
        diagnostics.feature_importances["model1"] = pd.Series([0.5, 0.3, 0.2])
        diagnostics.model_performance["model1"] = {"mse": 0.1, "mae": 0.05}

        report = diagnostics.get_diagnostics_report()

        self.assertIsInstance(report, dict)
        self.assertIn("feature_importances", report)
        self.assertIn("model_performance", report)
        self.assertIn("shap_available", report)
        self.assertIn("metrics_available", report)

        # Check content
        self.assertIn("model1", report["feature_importances"])
        self.assertIn("model1", report["model_performance"])


class TestPermutationImportance(unittest.TestCase):
    """Test cases for permutation importance calculation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        # Target with strong relationship to feature1
        self.y = 2 * self.X["feature1"] + np.random.randn(100) * 0.1

    def test_permutation_importance_calculation(self):
        """Test permutation importance calculation."""

        # Create a mock model with predict method
        class MockModel:
            def predict(self, X):
                # Simple prediction based on feature1
                return X["feature1"] * 2

        model = MockModel()

        # Calculate permutation importance
        importance = calculate_permutation_importance(
            model, self.X, self.y, n_repeats=3, random_state=42
        )

        self.assertIsInstance(importance, pd.Series)
        self.assertEqual(len(importance), 3)
        self.assertIn("feature1", importance.index)
        self.assertIn("feature2", importance.index)
        self.assertIn("feature3", importance.index)

        # feature1 should have highest importance since it's the true predictor
        self.assertGreater(importance["feature1"], importance["feature2"])
        self.assertGreater(importance["feature1"], importance["feature3"])


if __name__ == "__main__":
    unittest.main()
