"""
ML Measurement Plan module for FX-Commodity correlation arbitrage strategy.
Defines KPIs, failure modes, and thresholds for evaluating ML models.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class MLMeasurementPlan:
    """Class to define and evaluate ML model performance metrics."""
    
    def __init__(self):
        """Initialize measurement plan with default KPIs and thresholds."""
        self.kpis = self._define_kpis()
        self.failure_modes = self._define_failure_modes()
        self.thresholds = self._define_thresholds()
        
    def _define_kpis(self) -> Dict:
        """
        Define key performance indicators for ML models.
        
        Returns:
            Dictionary with KPI definitions.
        """
        return {
            "accuracy": {
                "description": "Overall classification accuracy",
                "type": "classification",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            "precision": {
                "description": "Precision (positive predictive value)",
                "type": "classification",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            "recall": {
                "description": "Recall (sensitivity)",
                "type": "classification",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            "f1_score": {
                "description": "F1 score (harmonic mean of precision and recall)",
                "type": "classification",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            "roc_auc": {
                "description": "Area under ROC curve",
                "type": "classification",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            "information_coefficient": {
                "description": "Correlation between predictions and actual returns",
                "type": "regression",
                "range": [-1.0, 1.0],
                "higher_is_better": True
            },
            "sharpe_ratio": {
                "description": "Risk-adjusted return of filtered strategy",
                "type": "portfolio",
                "range": [-np.inf, np.inf],
                "higher_is_better": True
            },
            "max_drawdown": {
                "description": "Maximum drawdown of filtered strategy",
                "type": "portfolio",
                "range": [-np.inf, 0.0],
                "higher_is_better": False
            },
            "win_rate": {
                "description": "Percentage of profitable trades",
                "type": "trading",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            "profit_factor": {
                "description": "Ratio of gross profits to gross losses",
                "type": "trading",
                "range": [0.0, np.inf],
                "higher_is_better": True
            }
        }
        
    def _define_failure_modes(self) -> Dict:
        """
        Define failure modes for ML models.
        
        Returns:
            Dictionary with failure mode definitions.
        """
        return {
            "overfitting": {
                "description": "Model performs well on training data but poorly on validation data",
                "detection_method": "Compare training vs validation metrics",
                "severity": "high"
            },
            "data_leakage": {
                "description": "Model uses future or target information",
                "detection_method": "Analyze feature importance and temporal consistency",
                "severity": "critical"
            },
            "concept_drift": {
                "description": "Model performance degrades over time due to changing market conditions",
                "detection_method": "Monitor performance over time windows",
                "severity": "medium"
            },
            "bias_amplification": {
                "description": "Model amplifies existing biases in training data",
                "detection_method": "Analyze prediction distribution across subgroups",
                "severity": "medium"
            },
            "calibration_drift": {
                "description": "Model probabilities become uncalibrated over time",
                "detection_method": "Monitor calibration plots over time",
                "severity": "medium"
            },
            "feature_instability": {
                "description": "Model relies on unstable or spurious features",
                "detection_method": "Analyze feature importance stability over time",
                "severity": "high"
            }
        }
        
    def _define_thresholds(self) -> Dict:
        """
        Define performance thresholds for ML models.
        
        Returns:
            Dictionary with threshold definitions.
        """
        return {
            "min_accuracy": 0.52,
            "min_precision": 0.50,
            "min_recall": 0.30,
            "min_f1_score": 0.40,
            "min_roc_auc": 0.55,
            "min_information_coefficient": 0.05,
            "min_sharpe_ratio": 0.5,
            "max_drawdown_threshold": -0.20,
            "min_win_rate": 0.50,
            "min_profit_factor": 1.2
        }
        
    def evaluate_model_performance(
        self, 
        predictions: np.ndarray, 
        probabilities: np.ndarray, 
        actuals: np.ndarray,
        returns: np.ndarray = None,
        timestamps: np.ndarray = None
    ) -> Dict:
        """
        Evaluate model performance across all KPIs.
        
        Args:
            predictions: Model predictions.
            probabilities: Prediction probabilities.
            actuals: Actual target values.
            returns: Trading returns (optional).
            timestamps: Timestamps for time-based analysis (optional).
            
        Returns:
            Dictionary with performance metrics.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            roc_auc_score, confusion_matrix
        )
        
        metrics = {}
        
        # Classification metrics
        metrics["accuracy"] = accuracy_score(actuals, predictions)
        metrics["precision"] = precision_score(actuals, predictions, zero_division=0)
        metrics["recall"] = recall_score(actuals, predictions, zero_division=0)
        metrics["f1_score"] = f1_score(actuals, predictions, zero_division=0)
        
        # ROC AUC (if probabilities are available)
        if probabilities.shape[1] > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(actuals, probabilities[:, 1])
            except Exception:
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0
            
        # Information coefficient (correlation between predictions and returns)
        if returns is not None and len(returns) == len(predictions):
            metrics["information_coefficient"] = np.corrcoef(predictions, returns)[0, 1]
        else:
            metrics["information_coefficient"] = 0.0
            
        # Trading metrics (if returns are provided)
        if returns is not None:
            # Sharpe ratio (annualized)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                metrics["sharpe_ratio"] = sharpe
            else:
                metrics["sharpe_ratio"] = 0.0
                
            # Max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            metrics["max_drawdown"] = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            # Win rate
            win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
            metrics["win_rate"] = win_rate
            
            # Profit factor
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = np.abs(np.sum(returns[returns < 0]))
            if gross_losses > 0:
                metrics["profit_factor"] = gross_profits / gross_losses
            else:
                metrics["profit_factor"] = np.inf if gross_profits > 0 else 0.0
                
        # Log performance
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return metrics
        
    def check_failure_modes(
        self, 
        training_metrics: Dict, 
        validation_metrics: Dict,
        feature_importance_history: List[Dict] = None
    ) -> Dict:
        """
        Check for potential failure modes.
        
        Args:
            training_metrics: Metrics from training data.
            validation_metrics: Metrics from validation data.
            feature_importance_history: History of feature importance (optional).
            
        Returns:
            Dictionary with failure mode detection results.
        """
        failures = {}
        
        # Check for overfitting
        if "accuracy" in training_metrics and "accuracy" in validation_metrics:
            train_acc = training_metrics["accuracy"]
            val_acc = validation_metrics["accuracy"]
            if train_acc - val_acc > 0.1:  # Arbitrary threshold
                failures["overfitting"] = {
                    "detected": True,
                    "severity": "high",
                    "details": f"Accuracy gap: {train_acc - val_acc:.4f}"
                }
            else:
                failures["overfitting"] = {"detected": False}
                
        # Check for feature instability (if history is provided)
        if feature_importance_history and len(feature_importance_history) > 1:
            # Compare feature importance stability
            latest_importance = feature_importance_history[-1]
            previous_importance = feature_importance_history[-2]
            
            # Calculate stability metric (correlation between importance rankings)
            common_features = set(latest_importance.keys()) & set(previous_importance.keys())
            if len(common_features) > 5:  # Need enough features for meaningful comparison
                latest_values = [latest_importance[f] for f in common_features]
                previous_values = [previous_importance[f] for f in common_features]
                
                if np.std(latest_values) > 0 and np.std(previous_values) > 0:
                    stability = np.corrcoef(latest_values, previous_values)[0, 1]
                    if stability < 0.5:  # Arbitrary threshold
                        failures["feature_instability"] = {
                            "detected": True,
                            "severity": "high",
                            "details": f"Feature importance stability: {stability:.4f}"
                        }
                    else:
                        failures["feature_instability"] = {"detected": False}
                else:
                    failures["feature_instability"] = {"detected": False}
            else:
                failures["feature_instability"] = {"detected": False}
        else:
            failures["feature_instability"] = {"detected": False}
            
        # Log failure mode checks
        for failure_mode, result in failures.items():
            if result["detected"]:
                logger.warning(f"Failure mode detected: {failure_mode} - {result['details']}")
                
        return failures
        
    def validate_performance(
        self, 
        metrics: Dict,
        check_failures: bool = True,
        training_metrics: Dict = None,
        validation_metrics: Dict = None
    ) -> Tuple[bool, Dict]:
        """
        Validate model performance against defined thresholds.
        
        Args:
            metrics: Performance metrics.
            check_failures: Whether to check for failure modes.
            training_metrics: Metrics from training data (for failure mode checking).
            validation_metrics: Metrics from validation data (for failure mode checking).
            
        Returns:
            Tuple of (is_valid, validation_results).
        """
        validation_results = {}
        is_valid = True
        
        # Check each metric against thresholds
        for metric_name, threshold_key in [
            ("accuracy", "min_accuracy"),
            ("precision", "min_precision"),
            ("recall", "min_recall"),
            ("f1_score", "min_f1_score"),
            ("roc_auc", "min_roc_auc"),
            ("information_coefficient", "min_information_coefficient"),
            ("sharpe_ratio", "min_sharpe_ratio"),
            ("max_drawdown", "max_drawdown_threshold"),
            ("win_rate", "min_win_rate"),
            ("profit_factor", "min_profit_factor")
        ]:
            if metric_name in metrics and threshold_key in self.thresholds:
                metric_value = metrics[metric_name]
                threshold_value = self.thresholds[threshold_key]
                
                # For metrics where higher is better
                if self.kpis[metric_name]["higher_is_better"]:
                    is_met = metric_value >= threshold_value
                else:
                    # For metrics where lower is better (like max_drawdown)
                    is_met = metric_value >= threshold_value  # max_drawdown is negative
                    
                validation_results[metric_name] = {
                    "value": metric_value,
                    "threshold": threshold_value,
                    "met": is_met
                }
                
                if not is_met:
                    is_valid = False
                    logger.warning(
                        f"Threshold not met: {metric_name} = {metric_value:.4f} < {threshold_value}"
                    )
                    
        # Check failure modes if requested
        if check_failures and training_metrics and validation_metrics:
            failures = self.check_failure_modes(training_metrics, validation_metrics)
            validation_results["failure_modes"] = failures
            
            # Check if any critical failures were detected
            for failure_name, failure_result in failures.items():
                if failure_result["detected"]:
                    failure_severity = self.failure_modes[failure_name]["severity"]
                    if failure_severity == "critical":
                        is_valid = False
                        logger.error(f"Critical failure detected: {failure_name}")
                        
        return is_valid, validation_results
        
    def generate_performance_report(
        self, 
        metrics: Dict, 
        validation_results: Dict,
        model_name: str = "ML Model"
    ) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            metrics: Performance metrics.
            validation_results: Validation results.
            model_name: Name of the model.
            
        Returns:
            Formatted performance report string.
        """
        report = f"\n=== {model_name} Performance Report ===\n"
        
        # Overall status
        is_valid = all(result.get("met", True) for result in validation_results.values() 
                      if isinstance(result, dict) and "met" in result)
        report += f"Overall Status: {'PASS' if is_valid else 'FAIL'}\n\n"
        
        # Metrics section
        report += "Performance Metrics:\n"
        for metric_name, metric_value in metrics.items():
            report += f"  {metric_name}: {metric_value:.4f}"
            if metric_name in validation_results:
                threshold = validation_results[metric_name]["threshold"]
                met = validation_results[metric_name]["met"]
                status = "✓" if met else "✗"
                report += f" [{status} threshold: {threshold}]"
            report += "\n"
            
        # Failure modes section
        if "failure_modes" in validation_results:
            report += "\nFailure Mode Analysis:\n"
            for failure_name, failure_result in validation_results["failure_modes"].items():
                if failure_result["detected"]:
                    severity = self.failure_modes[failure_name]["severity"]
                    details = failure_result.get("details", "")
                    report += f"  {failure_name} ({severity}): {details}\n"
                else:
                    report += f"  {failure_name}: Not detected\n"
                    
        report += "=" * (len(model_name) + 30) + "\n"
        
        return report


def create_default_measurement_plan() -> MLMeasurementPlan:
    """
    Create a default measurement plan.
    
    Returns:
        MLMeasurementPlan instance with default settings.
    """
    return MLMeasurementPlan()


def evaluate_ml_model(
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    actuals: np.ndarray,
    returns: np.ndarray = None,
    timestamps: np.ndarray = None,
    model_name: str = "ML Model"
) -> Tuple[Dict, Dict, str]:
    """
    Comprehensive evaluation of ML model performance.
    
    Args:
        predictions: Model predictions.
        probabilities: Prediction probabilities.
        actuals: Actual target values.
        returns: Trading returns (optional).
        timestamps: Timestamps for time-based analysis (optional).
        model_name: Name of the model.
        
    Returns:
        Tuple of (metrics, validation_results, performance_report).
    """
    # Create measurement plan
    plan = create_default_measurement_plan()
    
    # Evaluate performance
    metrics = plan.evaluate_model_performance(
        predictions, probabilities, actuals, returns, timestamps
    )
    
    # Validate performance
    is_valid, validation_results = plan.validate_performance(metrics)
    
    # Generate report
    report = plan.generate_performance_report(metrics, validation_results, model_name)
    
    return metrics, validation_results, report


def example_measurement_plan() -> None:
    """
    Example of how to use the measurement plan.
    """
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample predictions and actuals
    predictions = np.random.randint(0, 2, n_samples)
    probabilities = np.random.rand(n_samples, 2)
    actuals = np.random.randint(0, 2, n_samples)
    returns = np.random.randn(n_samples) * 0.01  # 1% daily volatility
    
    # Evaluate model
    metrics, validation_results, report = evaluate_ml_model(
        predictions, probabilities, actuals, returns, model_name="Example ML Filter"
    )
    
    # Print report
    print(report)
    
    # Check specific metrics
    print("Key Metrics:")
    for metric in ["accuracy", "f1_score", "roc_auc", "sharpe_ratio"]:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")