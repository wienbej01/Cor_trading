"""
Machine learning filter module for FX-Commodity correlation arbitrage strategy.
Implements ML-based signal filtering with comprehensive feature engineering.
"""

from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Import feature engineering functions
from ml.feature_engineering import (
    create_comprehensive_features,
    create_training_labels
)

# Try to import sklearn for actual ML implementation
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using stub implementation")


class MLSignalFilter:
    """
    ML-based signal filter for trading signals.
    Filters signals based on predicted profitability.
    """

    def __init__(self, model_path: Optional[str] = None, **model_params):
        """
        Initialize the ML signal filter.

        Args:
            model_path: Path to saved ML model.
            **model_params: Parameters for the ML model.
        """
        self.model_path = model_path
        self.model_params = model_params
        self.is_trained = False
        self.model = None
        self.feature_names = None
        
        # Initialize model if sklearn is available
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=model_params.get('random_state', 42),
                n_jobs=-1
            )
        
        logger.info("Initialized ML signal filter")

    def train(
        self, features: pd.DataFrame, labels: pd.Series, validation_split: float = 0.2
    ) -> Dict:
        """
        Train the ML model.

        Args:
            features: Feature DataFrame for training.
            labels: Target labels for training.
            validation_split: Fraction of data to use for validation.

        Returns:
            Dictionary with training metrics.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using stub training")
            return self._stub_train(features, labels, validation_split)
            
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")

        if validation_split <= 0 or validation_split >= 1:
            raise ValueError("Validation split must be between 0 and 1")

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42, stratify=labels
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Validate model
        y_pred = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            "train_accuracy": self.model.score(X_train, y_train),
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_precision": precision_score(y_val, y_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
        }

        logger.info(f"Model trained - Validation F1: {metrics['val_f1']:.4f}")
        return metrics
        
    def _stub_train(
        self, features: pd.DataFrame, labels: pd.Series, validation_split: float = 0.2
    ) -> Dict:
        """Stub training implementation."""
        self.is_trained = True
        return {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0,
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            features: Feature DataFrame for prediction.

        Returns:
            Array of predictions.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using stub prediction")
            return self._stub_predict(features)
            
        if not self.is_trained:
            logger.warning("Model not trained, returning default predictions")
            return np.ones(len(features))

        # Ensure feature columns match training data
        if self.feature_names is not None:
            features = features.reindex(columns=self.feature_names, fill_value=0)

        return self.model.predict(features)
        
    def _stub_predict(self, features: pd.DataFrame) -> np.ndarray:
        """Stub prediction implementation."""
        return np.ones(len(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            features: Feature DataFrame for prediction.

        Returns:
            Array of prediction probabilities.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using stub probabilities")
            return self._stub_predict_proba(features)
            
        if not self.is_trained:
            logger.warning("Model not trained, returning default probabilities")
            return np.full((len(features), 2), [0.5, 0.5])

        # Ensure feature columns match training data
        if self.feature_names is not None:
            features = features.reindex(columns=self.feature_names, fill_value=0)

        return self.model.predict_proba(features)
        
    def _stub_predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Stub probability prediction implementation."""
        return np.full((len(features), 2), [0.5, 0.5])

    def save_model(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, cannot save model")
            return
            
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to load the model from.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, cannot load model")
            self.is_trained = True
            return
            
        import joblib
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


def create_features(
    signals_df: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20, 60]
) -> pd.DataFrame:
    """
    Create comprehensive features for ML model from signals data.

    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for feature calculation.

    Returns:
        DataFrame with engineered features.
    """
    features = create_comprehensive_features(signals_df, lookback_periods)
    logger.info(f"Created {features.shape[1]} features for {len(features)} samples")
    return features


def apply_ml_filter(
    signals_df: pd.DataFrame,
    ml_filter: MLSignalFilter,
    probability_threshold: float = 0.5,
    lookback_periods: List[int] = [5, 10, 20, 60]
) -> pd.DataFrame:
    """
    Apply ML filter to trading signals.

    Args:
        signals_df: DataFrame with signals.
        ml_filter: Trained ML filter instance.
        probability_threshold: Threshold for accepting signals.
        lookback_periods: Lookback periods for feature creation.

    Returns:
        DataFrame with ML-filtered signals.
    """
    result = signals_df.copy()

    # Create features
    features = create_features(signals_df, lookback_periods)

    # Get predictions
    if ml_filter.is_trained:
        predictions = ml_filter.predict(features)
        probabilities = ml_filter.predict_proba(features)

        # Apply filter
        ml_accepted = predictions == 1
        ml_confident = probabilities[:, 1] >= probability_threshold

        # Create filter series with same index as signals_df
        ml_filter_series = pd.Series(False, index=signals_df.index)
        ml_filter_series.loc[features.index[ml_accepted & ml_confident]] = True

        # Apply filter to signals
        result["ml_filter"] = ml_filter_series
        result["signal"] = result["signal"].where(ml_filter_series, 0)

        logger.info(
            f"ML filter: {ml_filter_series.sum()}/{len(ml_filter_series)} signals accepted"
        )
    else:
        logger.warning("ML filter not trained, accepting all signals")
        result["ml_filter"] = True

    return result


def create_training_labels(
    signals_df: pd.DataFrame,
    forward_return_window: int = 5,
    return_threshold: float = 0.0,
    label_type: str = "binary"
) -> pd.Series:
    """
    Create training labels for ML model.

    Args:
        signals_df: DataFrame with signals and returns.
        forward_return_window: Window for calculating forward returns.
        return_threshold: Threshold for positive/negative labels.
        label_type: Type of labels to create ("binary", "multi", "regression").

    Returns:
        Series with training labels.
    """
    labels = create_training_labels(
        signals_df, forward_return_window, return_threshold, label_type
    )
    return labels
