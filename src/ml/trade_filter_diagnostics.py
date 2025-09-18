"""
ML Trade Filter Diagnostics module for FX-Commodity correlation arbitrage strategy.
Implements comprehensive diagnostics for ML-based signal filtering including
feature importance, model performance, and trading metrics.
"""

from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    from sklearn.metrics import (
        mean_squared_error, 
        mean_absolute_error, 
        r2_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )

    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    logger.warning("sklearn metrics not available")


class TradeFilterDiagnostics:
    """Comprehensive diagnostics for ML trade filter."""

    def __init__(self):
        """Initialize trade filter diagnostics."""
        self.feature_importances = {}
        self.model_performance = {}
        self.trading_metrics = {}
        self.shap_values = {}
        self.confusion_matrices = {}
        
    def calculate_trading_metrics(
        self, 
        signals_df: pd.DataFrame, 
        predictions: np.ndarray,
        labels: pd.Series,
        probability_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate trading-specific metrics for ML filter performance.
        
        Args:
            signals_df: DataFrame with trading signals and market data.
            predictions: Array of model predictions.
            labels: Series with true labels.
            probability_threshold: Threshold for classification.
            
        Returns:
            Dictionary with trading metrics.
        """
        if not SKLEARN_METRICS_AVAILABLE:
            logger.warning("sklearn metrics not available, skipping trading metrics calculation")
            return {}
            
        try:
            # Align indices
            common_index = signals_df.index.intersection(labels.index)
            if len(common_index) == 0:
                logger.warning("No common indices between signals and labels")
                return {}
                
            signals_aligned = signals_df.loc[common_index]
            labels_aligned = labels.loc[common_index]
            predictions_aligned = predictions[:len(common_index)]
            
            # Calculate basic classification metrics
            metrics = {
                "accuracy": accuracy_score(labels_aligned, predictions_aligned),
                "precision": precision_score(labels_aligned, predictions_aligned, zero_division=0),
                "recall": recall_score(labels_aligned, predictions_aligned, zero_division=0),
                "f1_score": f1_score(labels_aligned, predictions_aligned, zero_division=0),
            }
            
            # Calculate trading-specific metrics
            if len(predictions_aligned) > 0:
                # Signal acceptance rate
                signal_acceptance_rate = np.mean(predictions_aligned)
                metrics["signal_acceptance_rate"] = signal_acceptance_rate
                
                # Win rate of accepted signals
                if "signal" in signals_aligned.columns:
                    accepted_signals = signals_aligned[signals_aligned.index.isin(
                        labels_aligned[predictions_aligned == 1].index
                    )]
                    if len(accepted_signals) > 0:
                        win_rate = (accepted_signals["signal"] != 0).mean()
                        metrics["win_rate_accepted_signals"] = win_rate
                        
                # Profitability of accepted signals
                if "spread" in signals_aligned.columns:
                    # Calculate forward returns for accepted signals
                    forward_returns = signals_aligned["spread"].diff().shift(-1)
                    accepted_returns = forward_returns[signals_aligned.index.isin(
                        labels_aligned[predictions_aligned == 1].index
                    )]
                    if len(accepted_returns.dropna()) > 0:
                        avg_return = accepted_returns.mean()
                        metrics["avg_return_accepted_signals"] = avg_return
                        
                        # Sharpe ratio of accepted signals
                        if accepted_returns.std() > 0:
                            sharpe_ratio = avg_return / accepted_returns.std()
                            metrics["sharpe_ratio_accepted_signals"] = sharpe_ratio
            
            # Store metrics
            self.trading_metrics["ml_filter"] = metrics
            
            return metrics
        except Exception as e:
            logger.warning(f"Failed to calculate trading metrics: {e}")
            return {}
            
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
                
            # For classification models, also get probabilities if available
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }
            
            # Add regression metrics if applicable
            if y_test.dtype in [np.float64, np.int64]:
                metrics.update({
                    "mse": mean_squared_error(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                })
                
            # Add ROC AUC if probabilities are available
            if y_proba is not None and len(np.unique(y_test)) > 1:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                except Exception:
                    pass  # Skip if ROC AUC calculation fails
                    
            # Store metrics
            self.model_performance[model_name] = metrics
            
            # Store confusion matrix
            if len(np.unique(y_test)) <= 10:  # Only for classification with few classes
                self.confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
            
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
        
        # Select key metrics for plotting
        key_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        available_metrics = [m for m in key_metrics if m in perf_df.columns]
        
        if not available_metrics:
            logger.warning("No key metrics available for plotting")
            return
            
        plot_df = perf_df[available_metrics]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        plot_df.plot(kind="bar", ax=ax)
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
        
    def plot_confusion_matrices(
        self, save_path: str = None, figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot confusion matrices for all models.
        
        Args:
            save_path: Path to save the plot.
            figsize: Figure size.
        """
        if not self.confusion_matrices:
            logger.warning("No confusion matrices to plot")
            return
            
        n_models = len(self.confusion_matrices)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_models == 0:
            return
            
        for i, (model_name, cm) in enumerate(self.confusion_matrices.items()):
            ax = axes[i] if n_models > 1 else axes[0]
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
            ax.set_title(f"Confusion Matrix - {model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrices plot saved to {save_path}")
            
        plt.show()
        
    def plot_trading_metrics(
        self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot trading metrics.
        
        Args:
            save_path: Path to save the plot.
            figsize: Figure size.
        """
        if not self.trading_metrics:
            logger.warning("No trading metrics to plot")
            return
            
        # Convert to DataFrame
        metrics_df = pd.DataFrame(self.trading_metrics).T
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        metrics_df.plot(kind="bar", ax=ax)
        ax.set_title("Trading Metrics")
        ax.set_xlabel("Model/Filter")
        ax.set_ylabel("Metric Value")
        ax.legend(title="Metrics")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Trading metrics plot saved to {save_path}")
            
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
            "trading_metrics": {},
            "shap_available": SHAP_AVAILABLE,
            "metrics_available": SKLEARN_METRICS_AVAILABLE,
        }
        
        # Add feature importances
        for model_name, importance in self.feature_importances.items():
            report["feature_importances"][model_name] = importance.to_dict()
            
        # Add model performance
        report["model_performance"] = self.model_performance
        
        # Add trading metrics
        report["trading_metrics"] = self.trading_metrics
        
        return report
        

def create_trade_filter_diagnostics() -> TradeFilterDiagnostics:
    """
    Create a trade filter diagnostics instance.
    
    Returns:
        TradeFilterDiagnostics instance.
    """
    return TradeFilterDiagnostics()


def analyze_filter_performance(
    signals_df: pd.DataFrame,
    predictions: np.ndarray,
    labels: pd.Series,
    model: Any = None,
    features: pd.DataFrame = None,
    model_name: str = "ml_filter"
) -> Dict[str, Any]:
    """
    Comprehensive analysis of ML filter performance.
    
    Args:
        signals_df: DataFrame with trading signals and market data.
        predictions: Array of model predictions.
        labels: Series with true labels.
        model: Trained model object (optional).
        features: Feature DataFrame (optional).
        model_name: Name for the model in diagnostics.
        
    Returns:
        Dictionary with comprehensive analysis results.
    """
    diagnostics = create_trade_filter_diagnostics()
    
    # Calculate trading metrics
    trading_metrics = diagnostics.calculate_trading_metrics(
        signals_df, predictions, labels
    )
    
    # Calculate model performance if model is provided
    model_metrics = {}
    if model is not None and features is not None:
        # Split features and labels for train/test
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]
        
        model_metrics = diagnostics.calculate_model_performance(
            model, model_name, X_test, y_test
        )
        
        # Calculate feature importance
        diagnostics.calculate_feature_importance(model, model_name, X_train)
        
        # Calculate SHAP values if available
        if SHAP_AVAILABLE:
            diagnostics.calculate_shap_values(model, model_name, X_test)
    
    # Generate report
    report = diagnostics.get_diagnostics_report()
    
    # Add summary statistics
    report["summary"] = {
        "total_signals": len(predictions),
        "accepted_signals": int(np.sum(predictions)),
        "acceptance_rate": np.mean(predictions),
        "trading_metrics": trading_metrics,
        "model_metrics": model_metrics
    }
    
    return report