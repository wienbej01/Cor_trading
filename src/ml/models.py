"""
ML Models Module for Trade Filter.

Implements baseline classifiers and training utilities for supervised learning.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """Baseline ML classifier for trade filter using HistGradientBoostingClassifier."""

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.model_params = {
            'max_iter': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_leaf': 20,
            'l2_regularization': 0.1,
            'random_state': random_state,
            **kwargs
        }

        # Initialize base model
        self.base_model = HistGradientBoostingClassifier(**self.model_params)

        # Calibrated model for probability estimates
        self.model = CalibratedClassifierCV(
            self.base_model,
            method='isotonic',
            cv=3,
            n_jobs=-1
        )

        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineClassifier':
        """Fit the model."""
        logger.info(f"Fitting model with {len(X)} samples, {y.sum()} positive labels")

        try:
            self.model.fit(X.values, y.values)
            self.is_fitted = True
            logger.info("Model fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X.values)

    def predict_threshold(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict using custom threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def time_series_cross_validation(
    model: BaselineClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 0
) -> Dict[str, Any]:
    """
    Perform time-series cross-validation.

    Args:
        model: Classifier to evaluate
        X: Features
        y: Labels
        n_splits: Number of CV splits
        gap: Gap between train and test sets

    Returns:
        Dictionary with CV results
    """
    logger.info(f"Running time-series CV with {n_splits} splits")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    cv_results = {
        'fold_results': [],
        'mean_metrics': {},
        'std_metrics': {}
    }

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.debug(f"Training fold {fold + 1}/{n_splits}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model_copy = BaselineClassifier(random_state=42)
        model_copy.fit(X_train, y_train)

        # Predict
        y_pred = model_copy.predict(X_test)
        y_pred_proba = model_copy.predict_proba(X_test)[:, 1]

        # Calculate metrics
        fold_result = {
            'fold': fold + 1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_rate_train': y_train.mean(),
            'positive_rate_test': y_test.mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'accuracy': (y_pred == y_test).mean(),
            'precision': (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0,
            'recall': (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0,
        }

        cv_results['fold_results'].append(fold_result)
        fold_metrics.append(fold_result)

        logger.debug(f"Fold {fold + 1} metrics: ROC-AUC={fold_result['roc_auc']:.3f}, PR-AUC={fold_result['pr_auc']:.3f}")

    # Calculate mean and std metrics
    metrics_df = pd.DataFrame(fold_metrics)
    for col in ['roc_auc', 'pr_auc', 'brier_score', 'accuracy', 'precision', 'recall']:
        cv_results['mean_metrics'][col] = metrics_df[col].mean()
        cv_results['std_metrics'][col] = metrics_df[col].std()

    logger.info(f"CV completed. Mean ROC-AUC: {cv_results['mean_metrics']['roc_auc']:.3f} ± {cv_results['std_metrics']['roc_auc']:.3f}")

    return cv_results


def plot_calibration_curve(y_true: pd.Series, y_pred_proba: np.ndarray, save_path: Optional[str] = None):
    """Plot calibration curve."""
    plt.figure(figsize=(8, 6))

    # Calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")

    plt.close()


def plot_feature_importance(model: BaselineClassifier, feature_names: List[str], save_path: Optional[str] = None):
    """Plot feature importance."""
    if not hasattr(model.base_model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return

    plt.figure(figsize=(10, 6))

    importances = model.base_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.close()


def save_model_artifacts(
    model: BaselineClassifier,
    cv_results: Dict,
    feature_names: List[str],
    output_dir: str
):
    """Save model and evaluation artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save CV results
    metrics_path = os.path.join(output_dir, 'cv_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    logger.info(f"CV metrics saved to {metrics_path}")

    # Generate plots
    if len(cv_results['fold_results']) > 0:
        # Calibration plot (using last fold predictions if available)
        cal_plot_path = os.path.join(output_dir, 'calibration_curve.png')
        # Note: Would need actual predictions for real calibration plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Calibration plot placeholder\n(requires prediction data)',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(cal_plot_path)
        plt.close()

    # Feature importance plot
    fi_plot_path = os.path.join(output_dir, 'feature_importance.png')
    plot_feature_importance(model, feature_names, fi_plot_path)

    logger.info(f"Artifacts saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 18

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))

    # Train model
    model = BaselineClassifier(random_state=42)
    model.fit(X, y)

    # CV evaluation
    cv_results = time_series_cross_validation(model, X, y, n_splits=3)

    print("CV Results:")
    print(f"Mean ROC-AUC: {cv_results['mean_metrics']['roc_auc']:.3f}")
    print(f"Mean PR-AUC: {cv_results['mean_metrics']['pr_auc']:.3f}")