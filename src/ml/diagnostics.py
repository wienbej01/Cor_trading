"""
ML diagnostics module for FX-Commodity correlation arbitrage strategy.
Implements feature importance calculations, SHAP values, and model performance diagnostics.
"""

from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Try to import SHAP, with fallback
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not available, SHAP values will not be calculated")

# Try to import sklearn metrics
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    logger.warning("sklearn metrics not available")


class MLDiagnostics:
    """ML diagnostics for feature importance and model performance."""

    def __init__(self):
        """Initialize ML diagnostics."""
        self.feature_importances = {}
        self.model_performance = {}
        self.shap_values = {}

    def calculate_feature_importance(
        self, model: Any, model_name: str, X: pd.DataFrame = None
    ) -> Optional[pd.Series]:
        """
        Calculate feature importance for a model.

        Args:
            model: Trained model object.
            model_name: Name of the model.
            X: Feature data (optional, used for permutation importance).

        Returns:
            Series with feature importances, or None if not available.
        """
        importance = None

        # Try to get feature importance directly from model
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            if X is not None:
                importance = pd.Series(model.feature_importances_, index=X.columns)
            else:
                importance = pd.Series(model.feature_importances_)
        elif hasattr(model, "coef_"):
            # Linear models
            coef = model.coef_
            if isinstance(coef, np.ndarray):
                if X is not None and len(coef) == X.shape[1]:
                    importance = pd.Series(np.abs(coef), index=X.columns)
                else:
                    importance = pd.Series(np.abs(coef))

        # Store importance
        if importance is not None:
            self.feature_importances[model_name] = importance

        return importance

    def calculate_shap_values(
        self, model: Any, model_name: str, X: pd.DataFrame, sample_size: int = 1000
    ) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for a model.

        Args:
            model: Trained model object.
            model_name: Name of the model.
            X: Feature data.
            sample_size: Number of samples to use for SHAP calculation.

        Returns:
            Array of SHAP values, or None if not available.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP values calculation")
            return None

        try:
            # Sample data if too large
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X

            # Create SHAP explainer based on model type
            if hasattr(model, "predict"):
                # For tree-based models
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)

                # Store SHAP values
                self.shap_values[model_name] = shap_values

                return shap_values.values
            else:
                logger.warning(
                    f"Model {model_name} does not have predict method, skipping SHAP"
                )
                return None
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values for {model_name}: {e}")
            return None

    def calculate_model_performance(
        self, model: Any, model_name: str, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate model performance metrics.

        Args:
            model: Trained model object.
            model_name: Name of the model.
            X_test: Test feature data.
            y_test: Test target data.

        Returns:
            Dictionary with performance metrics.
        """
        if not SKLEARN_METRICS_AVAILABLE:
            logger.warning(
                "sklearn metrics not available, skipping performance calculation"
            )
            return {}

        try:
            # Make predictions
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                logger.warning(f"Model {model_name} does not have predict method")
                return {}

            # Calculate metrics
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }

            # Store metrics
            self.model_performance[model_name] = metrics

            return metrics
        except Exception as e:
            logger.warning(
                f"Failed to calculate performance metrics for {model_name}: {e}"
            )
            return {}

    def plot_feature_importance(
        self,
        model_name: str = None,
        top_n: int = 20,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot feature importance for a model or all models.

        Args:
            model_name: Name of the model to plot. If None, plots all models.
            top_n: Number of top features to show.
            save_path: Path to save the plot.
            figsize: Figure size.
        """
        if model_name:
            if model_name not in self.feature_importances:
                logger.warning(f"No feature importance data for model {model_name}")
                return

            importance_data = {model_name: self.feature_importances[model_name]}
        else:
            importance_data = self.feature_importances

        if not importance_data:
            logger.warning("No feature importance data to plot")
            return

        # Create subplots
        n_models = len(importance_data)
        fig, axes = plt.subplots(n_models, 1, figsize=figsize)
        if n_models == 1:
            axes = [axes]

        for i, (name, importance) in enumerate(importance_data.items()):
            ax = axes[i] if n_models > 1 else axes[0]

            # Get top features
            if len(importance) > top_n:
                top_importance = (
                    importance.abs().sort_values(ascending=False).head(top_n)
                )
            else:
                top_importance = importance.abs().sort_values(ascending=False)

            # Plot
            top_importance.plot(kind="barh", ax=ax)
            ax.set_title(f"Feature Importance - {name}")
            ax.set_xlabel("Importance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_model_performance(
        self, save_path: str = None, figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot model performance comparison.

        Args:
            save_path: Path to save the plot.
            figsize: Figure size.
        """
        if not self.model_performance:
            logger.warning("No model performance data to plot")
            return

        # Convert to DataFrame
        perf_df = pd.DataFrame(self.model_performance).T

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        perf_df.plot(kind="bar", ax=ax)
        ax.set_title("Model Performance Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Metric Value")
        ax.legend(title="Metrics")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model performance plot saved to {save_path}")

        plt.show()

    def plot_shap_summary(
        self, model_name: str, save_path: str = None, figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot SHAP summary plot for a model.

        Args:
            model_name: Name of the model to plot.
            save_path: Path to save the plot.
            figsize: Figure size.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP summary plot")
            return

        if model_name not in self.shap_values:
            logger.warning(f"No SHAP values for model {model_name}")
            return

        # Plot SHAP summary
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values[model_name], show=False)
        plt.title(f"SHAP Summary Plot - {model_name}")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved to {save_path}")

        plt.show()

    def get_diagnostics_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive diagnostics report.

        Returns:
            Dictionary with diagnostics report.
        """
        report = {
            "feature_importances": {},
            "model_performance": {},
            "shap_available": SHAP_AVAILABLE,
            "metrics_available": SKLEARN_METRICS_AVAILABLE,
        }

        # Add feature importances
        for model_name, importance in self.feature_importances.items():
            report["feature_importances"][model_name] = importance.to_dict()

        # Add model performance
        report["model_performance"] = self.model_performance

        return report


def create_ml_diagnostics() -> MLDiagnostics:
    """
    Create an ML diagnostics instance.

    Returns:
        MLDiagnostics instance.
    """
    return MLDiagnostics()


def calculate_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    Calculate permutation feature importance.

    Args:
        model: Trained model object.
        X: Feature data.
        y: Target data.
        n_repeats: Number of times to permute each feature.
        random_state: Random seed.

    Returns:
        Series with permutation importances.
    """
    if not SKLEARN_METRICS_AVAILABLE:
        logger.warning("sklearn metrics not available, skipping permutation importance")
        return pd.Series()

    try:
        from sklearn.utils import shuffle

        # Calculate baseline score
        if hasattr(model, "predict"):
            y_pred = model.predict(X)
            baseline_score = r2_score(y, y_pred)
        else:
            logger.warning("Model does not have predict method")
            return pd.Series()

        # Calculate importance for each feature
        importances = {}
        rng = np.random.RandomState(random_state)

        for col in X.columns:
            scores = []
            for _ in range(n_repeats):
                # Create a copy of X with shuffled column
                X_permuted = X.copy()
                X_permuted[col] = shuffle(X_permuted[col].values, random_state=rng)

                # Calculate score with permuted column
                y_pred_permuted = model.predict(X_permuted)
                permuted_score = r2_score(y, y_pred_permuted)

                # Importance is decrease in score
                scores.append(baseline_score - permuted_score)

            importances[col] = np.mean(scores)

        return pd.Series(importances).sort_values(ascending=False)
    except Exception as e:
        logger.warning(f"Failed to calculate permutation importance: {e}")
        return pd.Series()
