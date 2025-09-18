from dataclasses import dataclass, field
from typing import List

import pandas as pd

@dataclass
class LabelSchema:
    """
    Defines the schema for the ML model's target variable (label).

    The label is designed to capture whether a trading opportunity (e.g., a z-score crossing)
    results in a successful mean-reversion event within a specified future horizon.

    Attributes:
        reverted (bool): True if the spread reverted to the mean.
        reversion_time (int): Number of bars until the reversion occurred.
        target_profit (float): The price level for a successful exit.
        stop_loss (float): The price level for an unsuccessful exit.
        max_adverse_excursion (float): The worst price movement against the position before reversion.
    """
    reverted: bool
    reversion_time: int
    target_profit: float
    stop_loss: float
    max_adverse_excursion: float

@dataclass
class FeatureSchema:
    """
    Defines the schema for the ML model's input features.

    Features are selected based on economic rationale and parsimony to avoid overfitting.
    They aim to capture the state of the market and the spread at the time of the trading opportunity.
    """
    # Example features - to be refined
    z_score: float
    spread_std_20d: float
    roc_10d: float
    volatility_regime: int # e.g., 0 for low, 1 for medium, 2 for high
    trend_regime: int # e.g., -1 for down, 0 for sideways, 1 for up

# This could be a more formal list of features that the dataset builder will use.
# For now, this is a placeholder.
DEFAULT_FEATURES: List[str] = [
    'z_score',
    'spread_rolling_std_20',
    'spread_roc_10',
    'fx_vol_regime',
    'commodity_vol_regime',
    'fx_trend_regime',
    'commodity_trend_regime',
]

@dataclass
class MLData:
    """
    A container for the features (X) and labels (y) of the ML dataset.
    """
    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self):
        """Validate the integrity of the data."""
        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("Features X must be a pandas DataFrame.")
        if not isinstance(self.y, pd.Series):
            raise TypeError("Labels y must be a pandas Series.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Features and labels must have the same number of samples.")
