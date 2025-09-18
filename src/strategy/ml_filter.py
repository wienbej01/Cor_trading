import joblib
import pandas as pd
from pathlib import Path
from loguru import logger

# Assuming schema is in the ml directory, and project root is in path
from src.ml.schema import DEFAULT_FEATURES

class TradeFilter:
    """
    A filter that uses a trained ML model to accept or reject trades.
    """
    def __init__(self, model_path: str, threshold: float, enabled: bool = True):
        """
        Initializes the TradeFilter.

        Args:
            model_path (str): Path to the trained model (.joblib file).
            threshold (float): The probability threshold to accept a trade.
            enabled (bool): Whether the filter is enabled.
        """
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.enabled = enabled
        self.model = None

        if self.enabled:
            self._load_model()

    def _load_model(self):
        """Loads the model from the specified path."""
        if not self.model_path or not self.model_path.exists():
            logger.error(f"Model not found at {self.model_path}. The ML filter will be disabled.")
            self.enabled = False
            return
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded ML model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}. Disabling ML filter.")
            self.enabled = False

    def accept(self, context: pd.Series) -> bool:
        """
        Determines whether to accept a trade based on the model's prediction.

        Args:
            context (pd.Series): A series containing the features for the current timestep.

        Returns:
            bool: True if the trade is accepted, False otherwise.
        """
        if not self.enabled or self.model is None:
            return True  # If disabled or no model, accept all trades

        try:
            # Ensure context has all required features, in the correct order
            features_to_use = [f for f in self.model.feature_names_in_ if f in context.index]
            context_df = pd.DataFrame([context[features_to_use]])
            
            # Predict probability of a "good" trade (class 1)
            probability = self.model.predict_proba(context_df)[:, 1][0]

            logger.debug(f"ML Filter: Predicted probability = {probability:.4f}, Threshold = {self.threshold}")

            if probability >= self.threshold:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error during ML filter prediction: {e}. Defaulting to accept trade.")
            return True
