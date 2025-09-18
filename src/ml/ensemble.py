"""
Multi-model ensemble framework for FX-Commodity correlation arbitrage strategy.
Implements OLS, Kalman, rolling correlation, and ML-based residual prediction models.
"""

from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

# Try to import ML libraries, with fallbacks
try:
    from sklearn.ensemble import GradientBoostingRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available, GradientBoostingRegressor will not be available"
    )

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, LSTM model will not be available")


@dataclass
class ModelConfig:
    """Configuration for ensemble models."""

    # OLS model parameters
    ols_window: int = 90

    # Kalman filter parameters
    kalman_lambda: float = 0.995
    kalman_delta: float = 100.0

    # Rolling correlation parameters
    corr_window: int = 20

    # ML model parameters
    gb_n_estimators: int = 100
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.1

    # LSTM parameters
    lstm_hidden_size: int = 50
    lstm_num_layers: int = 2
    lstm_epochs: int = 50
    lstm_lr: float = 0.001

    # Ensemble parameters
    model_weights: Dict[str, float] = None  # Model weights for ensemble

    def __post_init__(self):
        if self.model_weights is None:
            # Default equal weights
            self.model_weights = {
                "ols": 0.2,
                "kalman": 0.2,
                "corr": 0.2,
                "gb": 0.2,
                "lstm": 0.2,
            }


class BaseModel(ABC):
    """Abstract base class for all models in the ensemble."""

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores if available."""
        pass


class OLSModel(BaseModel):
    """Ordinary Least Squares model wrapper."""

    def __init__(self, window: int = 90):
        super().__init__("ols")
        self.window = window
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit OLS model using rolling window."""
        if len(X) < self.window:
            raise ValueError(
                f"Not enough data for OLS model. Need at least {self.window} points."
            )

        # Use only the last window of data
        X_window = X.iloc[-self.window :]
        y_window = y.iloc[-self.window :]

        # Add intercept term
        X_with_intercept = np.c_[np.ones(X_window.shape[0]), X_window.values]

        # Calculate coefficients using normal equation
        try:
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y_window.values
            coef = np.linalg.solve(XtX, Xty)

            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
            self.is_trained = True
        except np.linalg.LinAlgError:
            logger.warning("OLS model fitting failed due to singular matrix")
            self.intercept_ = 0.0
            self.coef_ = np.zeros(X.shape[1])
            self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted OLS model."""
        if not self.is_trained:
            logger.warning("OLS model not trained, returning zeros")
            return np.zeros(len(X))

        if self.coef_ is None:
            return np.zeros(len(X))

        # Ensure coef_ has the right shape
        if len(self.coef_) != X.shape[1]:
            logger.warning(
                f"Dimension mismatch: coef_ has {len(self.coef_)} features, X has {X.shape[1]}"
            )
            return np.zeros(len(X))

        predictions = X.values @ self.coef_ + self.intercept_
        return predictions

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance based on coefficient magnitudes."""
        if not self.is_trained or self.coef_ is None:
            return None

        # Use absolute coefficient values as importance
        feature_names = [f"feature_{i}" for i in range(len(self.coef_))]
        importance = pd.Series(np.abs(self.coef_), index=feature_names)
        return importance


class KalmanModel(BaseModel):
    """Kalman filter model wrapper."""

    def __init__(self, lam: float = 0.995, delta: float = 100.0):
        super().__init__("kalman")
        self.lam = lam  # Forgetting factor
        self.delta = delta  # Initial state uncertainty
        self.theta = None  # Model parameters [intercept, slope]
        self.P = None  # Covariance matrix

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Kalman filter model."""
        if X.shape[1] != 1:
            raise ValueError("Kalman model expects single feature input")

        # Initialize parameters
        self.theta = np.zeros(2)  # [intercept, slope]
        self.P = self.delta * np.eye(2)

        # Process each data point
        for i in range(len(X)):
            x_val = X.iloc[i, 0]
            y_val = y.iloc[i]

            # Prediction step
            Xt = np.array([1.0, x_val]).reshape(1, 2)

            # Update step
            PI = self.P @ Xt.T
            K = PI / (self.lam + Xt @ PI)
            err = y_val - Xt @ self.theta
            self.theta = self.theta + (K.flatten() * err).reshape(-1)
            self.P = (self.P - K @ Xt @ self.P) / self.lam

        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted Kalman model."""
        if not self.is_trained or self.theta is None:
            logger.warning("Kalman model not trained, returning zeros")
            return np.zeros(len(X))

        if X.shape[1] != 1:
            logger.warning("Kalman model expects single feature input")
            return np.zeros(len(X))

        # Predict using current model parameters
        predictions = self.theta[0] + self.theta[1] * X.iloc[:, 0].values
        return predictions

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance based on coefficient magnitudes."""
        if not self.is_trained or self.theta is None:
            return None

        # Use absolute coefficient values as importance
        feature_names = ["intercept", "slope"]
        importance = pd.Series(np.abs(self.theta), index=feature_names)
        return importance


class RollingCorrelationModel(BaseModel):
    """Rolling correlation model wrapper."""

    def __init__(self, window: int = 20):
        super().__init__("corr")
        self.window = window
        self.correlation = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit rolling correlation model."""
        if X.shape[1] != 1:
            raise ValueError("Correlation model expects single feature input")

        # Calculate rolling correlation
        df = pd.concat([X.iloc[:, 0], y], axis=1).dropna()
        if len(df) >= self.window:
            self.correlation = (
                df.iloc[:, 0].rolling(self.window).corr(df.iloc[:, 1]).iloc[-1]
            )
        else:
            self.correlation = 0.0

        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted correlation model."""
        if not self.is_trained or self.correlation is None:
            logger.warning("Correlation model not trained, returning zeros")
            return np.zeros(len(X))

        if X.shape[1] != 1:
            logger.warning("Correlation model expects single feature input")
            return np.zeros(len(X))

        # Predict based on correlation
        predictions = self.correlation * X.iloc[:, 0].values
        return predictions

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance (correlation magnitude)."""
        if not self.is_trained or self.correlation is None:
            return None

        # Importance is the absolute correlation
        importance = pd.Series([abs(self.correlation)], index=["correlation"])
        return importance


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Trees model wrapper."""

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1
    ):
        super().__init__("gb")
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for GradientBoostingModel")

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Gradient Boosting model."""
        self.feature_names = X.columns.tolist()

        # Handle missing values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)

        # Fit the model
        self.model.fit(X_clean, y_clean)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted Gradient Boosting model."""
        if not self.is_trained:
            logger.warning("Gradient Boosting model not trained, returning zeros")
            return np.zeros(len(X))

        # Handle missing values
        X_clean = X.fillna(0)

        # Ensure columns match training data
        if self.feature_names is not None:
            # Reorder columns to match training data
            X_clean = X_clean.reindex(columns=self.feature_names, fill_value=0)

        predictions = self.model.predict(X_clean)
        return predictions

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from the Gradient Boosting model."""
        if not self.is_trained:
            return None

        if self.feature_names is None:
            return None

        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        )
        return importance


class LSTMModel(BaseModel):
    """LSTM model for residual prediction."""

    def __init__(
        self,
        hidden_size: int = 50,
        num_layers: int = 2,
        epochs: int = 50,
        lr: float = 0.001,
    ):
        super().__init__("lstm")
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LSTMModel")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.feature_count = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_size: int) -> None:
        """Build the LSTM model."""

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(LSTMRegressor, self).__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])  # Use last time step
                return out.squeeze()

        self.model = LSTMRegressor(input_size, self.hidden_size, self.num_layers).to(
            self.device
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LSTM model."""
        if not TORCH_AVAILABLE:
            logger.warning("torch not available, cannot fit LSTM model")
            return

        # Prepare data for LSTM (sequence format)
        sequence_length = min(10, len(X))  # Use last 10 points as sequence
        self.feature_count = X.shape[1]

        # Build model
        self._build_model(X.shape[1])

        # Prepare sequences
        X_seq, y_seq = self._create_sequences(X.values, y.values, sequence_length)

        if len(X_seq) == 0:
            logger.warning("Not enough data to create sequences for LSTM")
            return

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_trained = True

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []

        for i in range(seq_length, len(X)):
            X_seq.append(X[i - seq_length : i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted LSTM model."""
        if not self.is_trained or self.model is None:
            logger.warning("LSTM model not trained, returning zeros")
            return np.zeros(len(X))

        if not TORCH_AVAILABLE:
            logger.warning("torch not available, cannot make LSTM predictions")
            return np.zeros(len(X))

        # Prepare sequences
        sequence_length = min(10, len(X))
        X_seq, _ = self._create_sequences(X.values, np.zeros(len(X)), sequence_length)

        if len(X_seq) == 0:
            logger.warning("Not enough data to create sequences for LSTM prediction")
            return np.zeros(len(X))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        # Pad predictions to match input length
        padded_predictions = np.zeros(len(X))
        padded_predictions[sequence_length:] = predictions

        return padded_predictions

    def get_feature_importance(self) -> Optional[pd.Series]:
        """LSTM doesn't provide direct feature importance."""
        return None


class EnsembleModel:
    """Ensemble model that combines predictions from multiple models."""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models: Dict[str, BaseModel] = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize all models in the ensemble."""
        # Initialize OLS model
        self.models["ols"] = OLSModel(window=self.config.ols_window)

        # Initialize Kalman model
        self.models["kalman"] = KalmanModel(
            lam=self.config.kalman_lambda, delta=self.config.kalman_delta
        )

        # Initialize Rolling Correlation model
        self.models["corr"] = RollingCorrelationModel(window=self.config.corr_window)

        # Initialize Gradient Boosting model if sklearn is available
        if SKLEARN_AVAILABLE:
            self.models["gb"] = GradientBoostingModel(
                n_estimators=self.config.gb_n_estimators,
                max_depth=self.config.gb_max_depth,
                learning_rate=self.config.gb_learning_rate,
            )

        # Initialize LSTM model if torch is available
        if TORCH_AVAILABLE:
            self.models["lstm"] = LSTMModel(
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                epochs=self.config.lstm_epochs,
                lr=self.config.lstm_lr,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Fit all models in the ensemble and return training scores."""
        scores = {}

        for name, model in self.models.items():
            try:
                # Handle special case for Kalman and Correlation models (single feature)
                if name in ["kalman", "corr"] and X.shape[1] > 1:
                    # Use first feature for these models
                    X_model = X.iloc[:, [0]]
                else:
                    X_model = X

                model.fit(X_model, y)
                scores[name] = 1.0  # Simple success indicator
                logger.info(f"Successfully trained {name} model")
            except Exception as e:
                logger.warning(f"Failed to train {name} model: {e}")
                scores[name] = 0.0

        return scores

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using all models in the ensemble."""
        predictions = {}

        for name, model in self.models.items():
            try:
                # Handle special case for Kalman and Correlation models (single feature)
                if name in ["kalman", "corr"] and X.shape[1] > 1:
                    # Use first feature for these models
                    X_model = X.iloc[:, [0]]
                else:
                    X_model = X

                pred = model.predict(X_model)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Failed to predict with {name} model: {e}")
                predictions[name] = np.zeros(len(X))

        return predictions

    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction using weighted average of all models."""
        predictions = self.predict(X)

        # Initialize ensemble prediction
        ensemble_pred = np.zeros(len(X))
        total_weight = 0.0

        # Weighted average of all model predictions
        for name, pred in predictions.items():
            weight = self.config.model_weights.get(name, 0.0)
            if weight > 0:
                ensemble_pred += weight * pred
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            ensemble_pred /= total_weight

        return ensemble_pred

    def get_feature_importance(self) -> Dict[str, Optional[pd.Series]]:
        """Get feature importance from all models that support it."""
        importances = {}

        for name, model in self.models.items():
            try:
                importance = model.get_feature_importance()
                importances[name] = importance
            except Exception as e:
                logger.warning(
                    f"Failed to get feature importance from {name} model: {e}"
                )
                importances[name] = None

        return importances


def create_ensemble_model(config: ModelConfig = None) -> EnsembleModel:
    """
    Create an ensemble model with default or provided configuration.

    Args:
        config: Model configuration. If None, uses default.

    Returns:
        Ensemble model instance.
    """
    return EnsembleModel(config)


def create_default_model_config() -> ModelConfig:
    """
    Create a default model configuration.

    Returns:
        Default model configuration.
    """
    return ModelConfig()


"""
Multi-timeframe ensemble framework for combining H1 and D1 signals.
Provides sophisticated signal combination with risk management and position sizing.
"""

from typing import Dict, List, Optional, Tuple, Union
import asyncio
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import existing modules
from strategy.h1_mean_reversion import generate_h1_signals_from_data
from strategy.d1_mean_reversion import generate_d1_signals
from data.broker_api import get_h1_data, get_multi_symbol_h1_data
from data.yahoo_loader import download_daily, align_series


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe ensemble."""

    # Signal combination weights
    h1_weight: float = 0.6  # Weight for H1 signals (faster, more responsive)
    d1_weight: float = 0.4  # Weight for D1 signals (slower, more reliable)

    # Risk management
    max_h1_exposure: float = 0.3  # Maximum exposure from H1 signals
    max_d1_exposure: float = 0.7  # Maximum exposure from D1 signals
    max_total_exposure: float = 1.0  # Maximum total exposure

    # Signal filtering
    min_h1_confidence: float = 0.7  # Minimum confidence for H1 signals
    min_d1_confidence: float = 0.8  # Minimum confidence for D1 signals

    # Time alignment
    h1_alignment_window: int = 24  # Hours to look back for H1 alignment
    signal_decay_hours: int = 4  # Hours over which H1 signals decay

    # Position sizing
    base_position_size: float = 0.1  # Base position size per signal
    volatility_scaling: bool = True  # Scale positions by volatility


class MultiTimeframeEnsemble:
    """
    Multi-timeframe ensemble for combining H1 and D1 trading signals.
    Provides sophisticated signal combination with risk management.
    """

    def __init__(self, config: MultiTimeframeConfig = None):
        """
        Initialize multi-timeframe ensemble.

        Args:
            config: Multi-timeframe configuration
        """
        self.config = config or MultiTimeframeConfig()
        self.h1_signals: Optional[pd.DataFrame] = None
        self.d1_signals: Optional[pd.DataFrame] = None
        self.combined_signals: Optional[pd.DataFrame] = None

        logger.info("Initialized MultiTimeframeEnsemble")

    async def generate_multi_timeframe_signals(
        self,
        fx_symbol: str,
        comd_symbol: str,
        start_date: str,
        end_date: str,
        h1_config: Dict,
        d1_config: Dict,
    ) -> pd.DataFrame:
        """
        Generate combined H1 and D1 signals for a trading pair.

        Args:
            fx_symbol: FX symbol (e.g., "USDCAD=X")
            comd_symbol: Commodity symbol (e.g., "CL=F")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            h1_config: Configuration for H1 signal generation
            d1_config: Configuration for D1 signal generation

        Returns:
            DataFrame with combined multi-timeframe signals
        """
        logger.info(f"Generating multi-timeframe signals for {fx_symbol}-{comd_symbol}")

        # Generate H1 signals
        try:
            self.h1_signals = await self._generate_h1_signals(
                fx_symbol, comd_symbol, start_date, end_date, h1_config
            )
            logger.info(f"Generated {len(self.h1_signals)} H1 signals")
        except Exception as e:
            logger.error(f"Failed to generate H1 signals: {e}")
            self.h1_signals = pd.DataFrame()

        # Generate D1 signals
        try:
            self.d1_signals = await self._generate_d1_signals(
                fx_symbol, comd_symbol, start_date, end_date, d1_config
            )
            logger.info(f"Generated {len(self.d1_signals)} D1 signals")
        except Exception as e:
            logger.error(f"Failed to generate D1 signals: {e}")
            self.d1_signals = pd.DataFrame()

        # Combine signals
        if not self.h1_signals.empty and not self.d1_signals.empty:
            self.combined_signals = self._combine_signals()
        elif not self.h1_signals.empty:
            self.combined_signals = self._use_h1_only()
        elif not self.d1_signals.empty:
            self.combined_signals = self._use_d1_only()
        else:
            logger.warning("No signals generated from either timeframe")
            self.combined_signals = pd.DataFrame()

        return self.combined_signals

    async def _generate_h1_signals(
        self,
        fx_symbol: str,
        comd_symbol: str,
        start_date: str,
        end_date: str,
        config: Dict,
    ) -> pd.DataFrame:
        """
        Generate H1 signals using the H1 strategy module.
        """
        try:
            # Fetch H1 data
            h1_data = await get_multi_symbol_h1_data(
                [fx_symbol, comd_symbol], start_date, end_date, config
            )

            if fx_symbol not in h1_data or comd_symbol not in h1_data:
                raise ValueError(
                    f"H1 data not available for {fx_symbol} or {comd_symbol}"
                )

            fx_series = h1_data[fx_symbol].set_index("ts")["close"]
            comd_series = h1_data[comd_symbol].set_index("ts")["close"]

            # Generate H1 signals
            h1_signals = generate_h1_signals_from_data(fx_series, comd_series, config)

            # Add timeframe identifier
            h1_signals["timeframe"] = "H1"

            return h1_signals

        except Exception as e:
            logger.exception(f"H1 signal generation failed: {e}")
            raise

    async def _generate_d1_signals(
        self,
        fx_symbol: str,
        comd_symbol: str,
        start_date: str,
        end_date: str,
        config: Dict,
    ) -> pd.DataFrame:
        """
        Generate D1 signals using the D1 strategy module.
        """
        try:
            # Fetch daily data
            fx_daily = download_daily(fx_symbol, start_date, end_date)
            comd_daily = download_daily(comd_symbol, start_date, end_date)

            # Ensure distinct names to avoid column overwrite during alignment
            fx_daily.name = fx_symbol
            comd_daily.name = comd_symbol

            # Align series
            aligned_data = align_series(fx_daily, comd_daily)

            if aligned_data.empty:
                raise ValueError("Could not align daily data series")

            # Extract series by explicit column names to avoid positional errors
            if (
                fx_symbol not in aligned_data.columns
                or comd_symbol not in aligned_data.columns
            ):
                raise ValueError(
                    f"Aligned data missing expected columns: {list(aligned_data.columns)}"
                )

            fx_series = aligned_data[fx_symbol]
            comd_series = aligned_data[comd_symbol]

            # Generate D1 signals
            d1_signals = generate_d1_signals(fx_series, comd_series, config)

            # Add timeframe identifier
            d1_signals["timeframe"] = "D1"

            return d1_signals

        except Exception as e:
            # Log full traceback to console and logger to aid forensic debug
            import traceback

            tb = traceback.format_exc()
            logger.exception(f"D1 signal generation failed: {e}\n{tb}")
            print("D1 signal generation traceback:\n", tb)
            raise

    def _combine_signals(self) -> pd.DataFrame:
        """
        Combine H1 and D1 signals using weighted ensemble approach.
        """
        logger.info("Combining H1 and D1 signals")

        # Align signals by timestamp (resample D1 to H1 frequency)
        combined = self._align_signals_by_time()

        if combined.empty:
            return pd.DataFrame()

        # Calculate confidence scores
        combined["h1_confidence"] = self._calculate_h1_confidence(combined)
        combined["d1_confidence"] = self._calculate_d1_confidence(combined)

        # Apply confidence filters
        h1_valid = combined["h1_confidence"] >= self.config.min_h1_confidence
        d1_valid = combined["d1_confidence"] >= self.config.min_d1_confidence

        # Calculate weighted combined signal
        combined["combined_signal"] = 0.0

        # Both signals valid - use weighted average
        both_valid = h1_valid & d1_valid
        combined.loc[both_valid, "combined_signal"] = (
            self.config.h1_weight * combined.loc[both_valid, "h1_signal"]
            + self.config.d1_weight * combined.loc[both_valid, "d1_signal"]
        )

        # Only H1 valid
        h1_only = h1_valid & ~d1_valid
        combined.loc[h1_only, "combined_signal"] = combined.loc[
            h1_only, "h1_signal"
        ] * min(self.config.h1_weight * 1.5, 1.0)

        # Only D1 valid
        d1_only = ~h1_valid & d1_valid
        combined.loc[d1_only, "combined_signal"] = combined.loc[
            d1_only, "d1_signal"
        ] * min(self.config.d1_weight * 1.5, 1.0)

        # Apply position sizing and risk management
        combined = self._apply_position_sizing(combined)

        # Add signal metadata
        combined["signal_source"] = "multi_timeframe"
        combined["h1_weight"] = self.config.h1_weight
        combined["d1_weight"] = self.config.d1_weight

        logger.info(
            f"Combined signals: {len(combined[combined['combined_signal'] != 0])} active positions"
        )

        return combined

    def _align_signals_by_time(self) -> pd.DataFrame:
        """
        Align H1 and D1 signals by timestamp for combination.
        """
        if self.h1_signals.empty or self.d1_signals.empty:
            return pd.DataFrame()

        # Create common timestamp index (use H1 frequency)
        h1_index = self.h1_signals.index

        # Resample D1 signals to H1 frequency (forward fill)
        d1_resampled = self.d1_signals.reindex(h1_index, method="ffill")

        # Combine signals
        combined = pd.DataFrame(index=h1_index)
        combined["h1_signal"] = self.h1_signals["signal"]
        combined["d1_signal"] = d1_resampled["signal"]
        combined["h1_fx_price"] = self.h1_signals["fx_price"]
        combined["h1_comd_price"] = self.h1_signals["comd_price"]

        # Add D1 prices (resampled)
        if "fx_price" in d1_resampled.columns:
            combined["d1_fx_price"] = d1_resampled["fx_price"]
            combined["d1_comd_price"] = d1_resampled["comd_price"]

        return combined

    def _calculate_h1_confidence(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for H1 signals based on various factors.
        """
        confidence = pd.Series(0.5, index=signals.index)  # Base confidence

        # Higher confidence for stronger z-scores
        if "spread_z" in self.h1_signals.columns:
            z_score_strength = self.h1_signals["spread_z"].abs() / 2.0  # Normalize
            confidence += z_score_strength * 0.3

        # Higher confidence for signals that persist
        signal_persistence = (
            signals["h1_signal"]
            .rolling(4)
            .apply(lambda x: x.value_counts().max() / len(x))
        )
        confidence += signal_persistence * 0.2

        # Cap confidence at 1.0
        confidence = confidence.clip(0, 1)

        return confidence

    def _calculate_d1_confidence(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for D1 signals based on various factors.
        """
        confidence = pd.Series(
            0.7, index=signals.index
        )  # Higher base confidence for D1

        # Higher confidence for stronger z-scores
        if "spread_z" in self.d1_signals.columns:
            z_score_strength = self.d1_signals["spread_z"].abs() / 3.0  # Normalize
            confidence += z_score_strength * 0.2

        # Higher confidence for signals with good regime
        if "good_regime" in self.d1_signals.columns:
            regime_bonus = self.d1_signals["good_regime"].astype(int) * 0.1
            confidence += regime_bonus

        # Cap confidence at 1.0
        confidence = confidence.clip(0, 1)

        return confidence

    def _apply_position_sizing(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position sizing and risk management to combined signals.
        """
        # Calculate base position sizes
        signals["base_position"] = self.config.base_position_size

        # Apply volatility scaling if enabled
        if self.config.volatility_scaling:
            signals["volatility_factor"] = self._calculate_volatility_factor(signals)
            signals["base_position"] *= signals["volatility_factor"]

        # Apply signal strength scaling
        signal_strength = signals["combined_signal"].abs()
        signals["position_size"] = signals["base_position"] * signal_strength

        # Apply exposure limits
        signals["h1_exposure"] = signals["position_size"] * (
            signals["h1_signal"].abs() > 0
        ).astype(int)
        signals["d1_exposure"] = signals["position_size"] * (
            signals["d1_signal"].abs() > 0
        ).astype(int)

        # Cap exposures
        signals["h1_exposure"] = signals["h1_exposure"].clip(
            0, self.config.max_h1_exposure
        )
        signals["d1_exposure"] = signals["d1_exposure"].clip(
            0, self.config.max_d1_exposure
        )

        # Calculate total exposure
        signals["total_exposure"] = signals["h1_exposure"] + signals["d1_exposure"]
        signals["total_exposure"] = signals["total_exposure"].clip(
            0, self.config.max_total_exposure
        )

        # Final position size
        signals["final_position"] = (
            signals["total_exposure"] * signals["combined_signal"].sign()
        )

        return signals

    def _calculate_volatility_factor(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility-based position sizing factor.
        """
        # Use H1 price volatility as primary factor
        if "h1_fx_price" in signals.columns:
            h1_returns = signals["h1_fx_price"].pct_change()
            h1_vol = h1_returns.rolling(24).std()  # 24-hour volatility
            vol_factor = 1.0 / (1.0 + h1_vol)  # Inverse relationship
            vol_factor = vol_factor.fillna(0.5)  # Default factor
            return vol_factor
        else:
            return pd.Series(0.5, index=signals.index)

    def _use_h1_only(self) -> pd.DataFrame:
        """
        Fallback: Use only H1 signals when D1 signals are unavailable.
        """
        logger.warning("Using H1 signals only (D1 signals unavailable)")
        combined = self.h1_signals.copy()
        combined["combined_signal"] = combined["signal"] * self.config.h1_weight
        combined["signal_source"] = "h1_only"
        return combined

    def _use_d1_only(self) -> pd.DataFrame:
        """
        Fallback: Use only D1 signals when H1 signals are unavailable.
        """
        logger.warning("Using D1 signals only (H1 signals unavailable)")
        combined = self.d1_signals.copy()
        combined["combined_signal"] = combined["signal"] * self.config.d1_weight
        combined["signal_source"] = "d1_only"
        return combined

    def get_signal_statistics(self) -> Dict:
        """
        Get statistics about the generated signals.

        Returns:
            Dictionary with signal statistics
        """
        if self.combined_signals is None or self.combined_signals.empty:
            return {}

        stats = {
            "total_signals": len(self.combined_signals),
            "active_signals": len(
                self.combined_signals[self.combined_signals["combined_signal"] != 0]
            ),
            "h1_signals": len(self.h1_signals) if self.h1_signals is not None else 0,
            "d1_signals": len(self.d1_signals) if self.d1_signals is not None else 0,
            "avg_position_size": self.combined_signals["final_position"].abs().mean(),
            "max_exposure": self.combined_signals["total_exposure"].max(),
            "signal_distribution": self.combined_signals["combined_signal"]
            .value_counts()
            .to_dict(),
        }

        return stats


def create_multi_timeframe_ensemble(
    config: MultiTimeframeConfig = None,
) -> MultiTimeframeEnsemble:
    """
    Create a multi-timeframe ensemble with default or provided configuration.

    Args:
        config: Multi-timeframe configuration. If None, uses default.

    Returns:
        Multi-timeframe ensemble instance.
    """
    return MultiTimeframeEnsemble(config)


def create_default_multi_timeframe_config() -> MultiTimeframeConfig:
    """
    Create a default multi-timeframe configuration.

    Returns:
        Default multi-timeframe configuration.
    """
    return MultiTimeframeConfig()
