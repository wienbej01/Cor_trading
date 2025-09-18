"""
ML Trade Filter Ensemble module for FX-Commodity correlation arbitrage strategy.
Implements ensemble models specifically designed for trade filtering classification tasks.
"""

from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger

# Try to import ML libraries, with fallbacks
try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier,
        VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        roc_auc_score,
        confusion_matrix
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, classification models will not be available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, neural network models will not be available")


@dataclass
class TradeFilterConfig:
    """Configuration for trade filter ensemble models."""
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    
    # Gradient Boosting parameters
    gb_n_estimators: int = 100
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.1
    
    # Logistic Regression parameters
    lr_C: float = 1.0
    lr_penalty: str = 'l2'
    
    # SVM parameters
    svm_C: float = 1.0
    svm_kernel: str = 'rbf'
    
    # Neural Network parameters
    nn_hidden_layers: Tuple[int] = (100, 50)
    nn_max_iter: int = 500
    nn_learning_rate: float = 0.001
    
    # Ensemble parameters
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "rf": 0.25,
        "gb": 0.25,
        "lr": 0.25,
        "svm": 0.15,
        "nn": 0.10,
    })
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    validation_metric: str = "f1"  # Metric to use for model selection


class BaseTradeFilter(ABC):
    """Abstract base class for trade filter models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.model = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions using the fitted model."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using the fitted model."""
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores if available."""
        pass


class RandomForestTradeFilter(BaseTradeFilter):
    """Random Forest classifier for trade filtering."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 5, random_state: int = 42):
        super().__init__("rf")
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for RandomForestTradeFilter")
            
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Random Forest model."""
        # Handle missing values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Fit the model
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            logger.warning("Random Forest model not trained, returning zeros")
            return np.zeros(len(X))
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict(X_clean)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            logger.warning("Random Forest model not trained, returning 0.5 probabilities")
            return np.full((len(X), 2), 0.5)
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict_proba(X_clean)
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance."""
        if not self.is_trained:
            return None
            
        importance = pd.Series(
            self.model.feature_importances_, 
            index=self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
        )
        return importance


class GradientBoostingTradeFilter(BaseTradeFilter):
    """Gradient Boosting classifier for trade filtering."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 3, 
                 learning_rate: float = 0.1, random_state: int = 42):
        super().__init__("gb")
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for GradientBoostingTradeFilter")
            
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Gradient Boosting model."""
        # Handle missing values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Fit the model
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            logger.warning("Gradient Boosting model not trained, returning zeros")
            return np.zeros(len(X))
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict(X_clean)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            logger.warning("Gradient Boosting model not trained, returning 0.5 probabilities")
            return np.full((len(X), 2), 0.5)
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict_proba(X_clean)
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance."""
        if not self.is_trained:
            return None
            
        importance = pd.Series(
            self.model.feature_importances_, 
            index=self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
        )
        return importance


class LogisticRegressionTradeFilter(BaseTradeFilter):
    """Logistic Regression classifier for trade filtering."""
    
    def __init__(self, C: float = 1.0, penalty: str = 'l2', random_state: int = 42):
        super().__init__("lr")
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for LogisticRegressionTradeFilter")
            
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            random_state=random_state,
            max_iter=1000
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Logistic Regression model."""
        # Handle missing values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Fit the model
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            logger.warning("Logistic Regression model not trained, returning zeros")
            return np.zeros(len(X))
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict(X_clean)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            logger.warning("Logistic Regression model not trained, returning 0.5 probabilities")
            return np.full((len(X), 2), 0.5)
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict_proba(X_clean)
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance (coefficients)."""
        if not self.is_trained:
            return None
            
        importance = pd.Series(
            np.abs(self.model.coef_[0]), 
            index=self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
        )
        return importance


class SVMTradeFilter(BaseTradeFilter):
    """Support Vector Machine classifier for trade filtering."""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', random_state: int = 42):
        super().__init__("svm")
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for SVMTradeFilter")
            
        self.model = SVC(
            C=C,
            kernel=kernel,
            random_state=random_state,
            probability=True  # Enable probability estimates
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit SVM model."""
        # Handle missing values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Fit the model
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            logger.warning("SVM model not trained, returning zeros")
            return np.zeros(len(X))
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict(X_clean)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            logger.warning("SVM model not trained, returning 0.5 probabilities")
            return np.full((len(X), 2), 0.5)
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        return self.model.predict_proba(X_clean)
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """SVM doesn't provide direct feature importance."""
        return None


class NeuralNetworkTradeFilter(BaseTradeFilter):
    """Neural Network classifier for trade filtering."""
    
    def __init__(self, hidden_layers: Tuple[int] = (100, 50), 
                 max_iter: int = 500, learning_rate: float = 0.001):
        super().__init__("nn")
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for NeuralNetworkTradeFilter")
            
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _build_model(self, input_size: int, output_size: int) -> None:
        """Build the neural network model."""
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Neural Network model."""
        if not TORCH_AVAILABLE:
            logger.warning("torch not available, cannot fit Neural Network model")
            return
            
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_clean.values).to(self.device)
        y_tensor = torch.LongTensor(y_clean.values).to(self.device)
        
        # Build model
        self._build_model(X_tensor.shape[1], len(np.unique(y_clean)))
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained or self.model is None:
            logger.warning("Neural Network model not trained, returning zeros")
            return np.zeros(len(X))
            
        if not TORCH_AVAILABLE:
            logger.warning("torch not available, cannot make Neural Network predictions")
            return np.zeros(len(X))
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        # Ensure columns match training data
        if self.feature_names is not None:
            X_clean = X_clean.reindex(columns=self.feature_names, fill_value=0)
            
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_clean.values).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained or self.model is None:
            logger.warning("Neural Network model not trained, returning 0.5 probabilities")
            return np.full((len(X), 2), 0.5)
            
        if not TORCH_AVAILABLE:
            logger.warning("torch not available, cannot make Neural Network predictions")
            return np.full((len(X), 2), 0.5)
            
        # Handle missing values
        X_clean = X.fillna(0)
        
        # Ensure columns match training data
        if self.feature_names is not None:
            X_clean = X_clean.reindex(columns=self.feature_names, fill_value=0)
            
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_clean.values).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
        return outputs.cpu().numpy()
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Neural Network doesn't provide direct feature importance."""
        return None


class TradeFilterEnsemble:
    """Ensemble model that combines predictions from multiple trade filter models."""
    
    def __init__(self, config: TradeFilterConfig = None):
        self.config = config or TradeFilterConfig()
        self.models: Dict[str, BaseTradeFilter] = {}
        self._initialize_models()
        self.best_model = None
        self.validation_score = None
        
    def _initialize_models(self) -> None:
        """Initialize all models in the ensemble."""
        # Initialize Random Forest model if sklearn is available
        if SKLEARN_AVAILABLE:
            self.models["rf"] = RandomForestTradeFilter(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.random_state
            )
            
            # Initialize Gradient Boosting model
            self.models["gb"] = GradientBoostingTradeFilter(
                n_estimators=self.config.gb_n_estimators,
                max_depth=self.config.gb_max_depth,
                learning_rate=self.config.gb_learning_rate,
                random_state=self.config.random_state
            )
            
            # Initialize Logistic Regression model
            self.models["lr"] = LogisticRegressionTradeFilter(
                C=self.config.lr_C,
                penalty=self.config.lr_penalty,
                random_state=self.config.random_state
            )
            
            # Initialize SVM model
            self.models["svm"] = SVMTradeFilter(
                C=self.config.svm_C,
                kernel=self.config.svm_kernel,
                random_state=self.config.random_state
            )
            
        # Initialize Neural Network model if torch is available
        if TORCH_AVAILABLE:
            self.models["nn"] = NeuralNetworkTradeFilter(
                hidden_layers=self.config.nn_hidden_layers,
                max_iter=self.config.nn_max_iter,
                learning_rate=self.config.nn_learning_rate
            )
            
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Fit all models in the ensemble and return training scores.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with model training scores
        """
        scores = {}
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        best_score = -np.inf
        best_model_name = None
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                
                # Validate on validation set
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                # Calculate ROC AUC if probabilities are available
                roc_auc = 0.0
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    try:
                        roc_auc = roc_auc_score(y_val, y_proba[:, 1])
                    except Exception:
                        roc_auc = 0.0
                
                # Store scores
                scores[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                }
                
                # Select best model based on validation metric
                metric_score = scores[name].get(self.config.validation_metric, 0.0)
                if metric_score > best_score:
                    best_score = metric_score
                    best_model_name = name
                    
                logger.info(f"Successfully trained {name} model - F1: {f1:.4f}")
            except Exception as e:
                logger.warning(f"Failed to train {name} model: {e}")
                scores[name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.0
                }
                
        # Store best model
        if best_model_name:
            self.best_model = best_model_name
            self.validation_score = best_score
            
        return scores
        
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using all models in the ensemble.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Failed to predict with {name} model: {e}")
                predictions[name] = np.zeros(len(X))
                
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make probability predictions using all models in the ensemble.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with probability predictions from each model
        """
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                probabilities[name] = proba
            except Exception as e:
                logger.warning(f"Failed to predict probabilities with {name} model: {e}")
                # Return uniform probabilities
                probabilities[name] = np.full((len(X), 2), 0.5)
                
        return probabilities
        
    def predict_ensemble(self, X: pd.DataFrame, probability_threshold: float = 0.5) -> np.ndarray:
        """
        Make ensemble prediction using weighted average of all models.
        
        Args:
            X: Feature DataFrame
            probability_threshold: Threshold for classification
            
        Returns:
            Array of binary predictions
        """
        probabilities = self.predict_proba(X)
        
        # Initialize ensemble probabilities
        ensemble_proba = np.zeros(len(X))
        total_weight = 0.0
        
        # Weighted average of all model probabilities
        for name, proba in probabilities.items():
            weight = self.config.model_weights.get(name, 0.0)
            if weight > 0 and proba.shape[1] > 1:
                ensemble_proba += weight * proba[:, 1]  # Use probability of positive class
                total_weight += weight
                
        # Normalize by total weight
        if total_weight > 0:
            ensemble_proba /= total_weight
            
        # Convert to binary predictions
        ensemble_pred = (ensemble_proba >= probability_threshold).astype(int)
        
        return ensemble_pred
        
    def get_feature_importance(self) -> Dict[str, Optional[pd.Series]]:
        """Get feature importance from all models that support it."""
        importances = {}
        
        for name, model in self.models.items():
            try:
                importance = model.get_feature_importance()
                importances[name] = importance
            except Exception as e:
                logger.warning(f"Failed to get feature importance from {name} model: {e}")
                importances[name] = None
                
        return importances
        
    def get_model_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Calculate performance metrics for all models on test data.
        
        Args:
            X_test: Test feature DataFrame
            y_test: Test target Series
            
        Returns:
            Dictionary with performance metrics for each model
        """
        performance = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate ROC AUC if probabilities are available
                roc_auc = 0.0
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    try:
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    except Exception:
                        roc_auc = 0.0
                        
                performance[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                }
            except Exception as e:
                logger.warning(f"Failed to calculate performance for {name} model: {e}")
                performance[name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.0
                }
                
        return performance


def create_trade_filter_ensemble(config: TradeFilterConfig = None) -> TradeFilterEnsemble:
    """
    Create a trade filter ensemble with default or provided configuration.
    
    Args:
        config: Trade filter configuration. If None, uses default.
        
    Returns:
        Trade filter ensemble instance.
    """
    return TradeFilterEnsemble(config)


def create_default_trade_filter_config() -> TradeFilterConfig:
    """
    Create a default trade filter configuration.
    
    Returns:
        Default trade filter configuration.
    """
    return TradeFilterConfig()


def train_trade_filter_ensemble(
    features: pd.DataFrame, 
    labels: pd.Series,
    config: TradeFilterConfig = None
) -> TradeFilterEnsemble:
    """
    Train a trade filter ensemble model.
    
    Args:
        features: Feature DataFrame
        labels: Target Series
        config: Trade filter configuration
        
    Returns:
        Trained trade filter ensemble
    """
    # Create ensemble
    ensemble = create_trade_filter_ensemble(config)
    
    # Train ensemble
    training_scores = ensemble.fit(features, labels)
    
    # Log training results
    logger.info("Trade filter ensemble training completed")
    for model_name, scores in training_scores.items():
        logger.info(f"{model_name}: F1={scores['f1']:.4f}, ROC-AUC={scores['roc_auc']:.4f}")
        
    return ensemble