from dataclasses import dataclass, field
from typing import List, Dict, Optional

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


# ---------------------------------------------------------------------------
# Minimal, typed ML schema used by dataset builder
# ---------------------------------------------------------------------------

@dataclass
class LabelSpec:
    """
    Label specification controlling target construction.

    Timing and leakage controls follow docs/dev/ml_labeling.md:
    - horizon_bars: future window over which outcomes are evaluated (no lookahead).
    - embargo_bars: bar-based gap applied between positive outcomes to reduce leakage.
    - purge_window: additional purge window to avoid overlapping label horizons.
    - rr_threshold (risk-reward): minimum reward-to-risk ratio for a "positive" label.
    - max_adverse_excursion: optional guardrail for extreme adverse moves.

    The builder uses horizon_bars, embargo_bars, and purge_window to schedule samples
    and rr_threshold to parameterize outcome thresholds.
    """
    horizon_bars: int = 20
    rr_threshold: float = 1.5
    embargo_bars: int = 5
    purge_window: int = 20
    max_adverse_excursion: float = 0.0


@dataclass
class FeatureSpec:
    """
    Feature specification to encourage parsimony (see docs/dev/ml_labeling.md).

    Keep features economically motivated and avoid redundant or leaking inputs.
    - names: explicit list of permitted feature column names.
    - lookbacks: optional per-feature lookback window metadata (documentation aid).
    """
    names: List[str] = field(default_factory=list)
    lookbacks: Optional[Dict[str, int]] = None


@dataclass
class MLSchema:
    """
    Dataset schema collecting label and feature specifications.

    Also provides tiny helpers used by the dataset builder:
    - create_label: converts a future price path into a binary label by checking
      whether take-profit is reached before stop-loss within horizon.
    - validate_features: placeholder guard to assert feature frame integrity.

    Notes on leakage:
    - Labels are computed strictly from future data after the signal bar.
    - Embargo/purge windows help reduce overlap-induced leakage.
    - Feature parsimony and explicit naming reduce the chance of information bleed.
    """
    label_spec: LabelSpec = field(default_factory=LabelSpec)
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    seed: int = 42
    notes: str = ""

    def create_label(self, future_spread: pd.Series, entry_price: float, stop_loss: float, take_profit: float) -> int:
        """
        Return 1 if take-profit is reached before stop-loss within the provided
        future_spread window; otherwise 0. This function is direction-agnostic:
        it infers direction from the relative positions of entry/targets.

        Args:
            future_spread: Future path after the entry bar, length >= horizon_bars.
            entry_price: Entry price of the spread at signal time.
            stop_loss: Absolute price level for stop loss (constructed upstream).
            take_profit: Absolute price level for take profit (constructed upstream).

        Returns:
            int: 1 if profit is hit first, else 0.
        """
        if future_spread is None or len(future_spread) == 0:
            return 0

        # Determine direction by relative target placement
        profit_is_up = take_profit >= entry_price
        stop_is_up = stop_loss >= entry_price

        for v in future_spread:
            hit_profit = (v >= take_profit) if profit_is_up else (v <= take_profit)
            if hit_profit:
                return 1
            hit_stop = (v >= stop_loss) if stop_is_up else (v <= stop_loss)
            if hit_stop:
                return 0

        # Neither hit in horizon -> treat as negative outcome
        return 0

    def validate_features(self, features: pd.DataFrame) -> None:
        """
        Lightweight validation for features frame used by builder.

        Intent:
        - Ensure DataFrame type and non-empty columns.
        - Encourage numeric-only columns and absence of infs/NaNs from obvious issues.
        No mutation is performed to avoid surprising callers.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.shape[1] == 0:
            raise ValueError("features must contain at least one column")
        # Soft checks: prefer numeric dtypes; do not raise to preserve backward compatibility.
        # Any necessary cleaning should occur upstream in feature engineering.

# Public exports for external imports (dataset builder, etc.)
__all__ = [
    "MLSchema",
    "LabelSpec",
    "FeatureSpec",
    # Keep existing symbols available for backward compatibility
    "LabelSchema",
    "FeatureSchema",
    "MLData",
    "DEFAULT_FEATURES",
]
