"""
ML Schema Module for Supervised Trade Filter.

Defines label specifications, feature sets, and leakage prevention for spread reversion prediction.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class LabelSpec:
    """Specification for ML label construction."""
    horizon_bars: int = 20  # H: bars to check for reversion
    rr_threshold: float = 1.5  # RR≥X: minimum risk-reward ratio
    max_adverse_excursion: float = 2.0  # Max adverse excursion ≤ Y (in std dev units)
    reversion_threshold: float = 0.5  # Revert to within Y std dev of mean
    embargo_bars: int = 5  # Embargo period after label creation
    purge_window: int = 252  # Purge overlapping samples within this window


@dataclass
class FeatureSpec:
    """Specification for feature engineering."""
    spread_features: List[str] = None
    volatility_features: List[str] = None
    correlation_features: List[str] = None
    regime_features: List[str] = None
    temporal_features: List[str] = None

    def __post_init__(self):
        if self.spread_features is None:
            self.spread_features = [
                'spread_z_20',  # Z-score over 20 bars
                'spread_z_60',  # Z-score over 60 bars
                'spread_momentum_5',  # 5-bar momentum
                'spread_momentum_20',  # 20-bar momentum
            ]

        if self.volatility_features is None:
            self.volatility_features = [
                'spread_vol_20',  # Realized vol over 20 bars
                'spread_vol_60',  # Realized vol over 60 bars
                'fx_vol_20',  # FX volatility
                'comd_vol_20',  # Commodity volatility
            ]

        if self.correlation_features is None:
            self.correlation_features = [
                'rolling_corr_20',  # Rolling correlation 20 bars
                'rolling_corr_60',  # Rolling correlation 60 bars
                'corr_z_score',  # Correlation deviation from mean
            ]

        if self.regime_features is None:
            self.regime_features = [
                'trend_regime',  # -1,0,1 for down/range/up
                'vol_regime',  # 0,1,2 for low/normal/high vol
                'combined_regime',  # Combined trend+vol
            ]

        if self.temporal_features is None:
            self.temporal_features = [
                'day_of_week',  # 0-6
                'month_of_year',  # 1-12
                'quarter',  # 1-4
            ]

    @property
    def all_features(self) -> List[str]:
        """All features combined."""
        return (
            self.spread_features +
            self.volatility_features +
            self.correlation_features +
            self.regime_features +
            self.temporal_features
        )


class MLSchema:
    """Complete ML schema for trade filter."""

    def __init__(self, label_spec: Optional[LabelSpec] = None, feature_spec: Optional[FeatureSpec] = None):
        self.label_spec = label_spec or LabelSpec()
        self.feature_spec = feature_spec or FeatureSpec()

    def create_label(self, spread_series: pd.Series, entry_price: float, stop_loss: float, take_profit: float) -> int:
        """
        Create binary label for a trade opportunity.

        Args:
            spread_series: Future spread values after entry
            entry_price: Entry spread price
            stop_loss: Stop loss level
            take_profit: Take profit level

        Returns:
            1 if successful reversion within constraints, 0 otherwise
        """
        if len(spread_series) < self.label_spec.horizon_bars:
            return 0

        # Check reversion within horizon
        future_spread = spread_series.iloc[:self.label_spec.horizon_bars]
        spread_mean = spread_series.expanding().mean().iloc[-1]  # Use expanding mean up to entry
        spread_std = spread_series.expanding().std().iloc[-1]

        # Reversion condition: spread returns to within threshold of mean
        reversion_condition = (
            (future_spread >= spread_mean - self.label_spec.reversion_threshold * spread_std) &
            (future_spread <= spread_mean + self.label_spec.reversion_threshold * spread_std)
        ).any()

        if not reversion_condition:
            return 0

        # Risk-reward check
        rr_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        if rr_ratio < self.label_spec.rr_threshold:
            return 0

        # Max adverse excursion check
        adverse_excursion = abs(future_spread - entry_price).max()
        max_allowed = self.label_spec.max_adverse_excursion * spread_std
        if adverse_excursion > max_allowed:
            return 0

        return 1

    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """Validate feature set completeness and no NaN."""
        required_features = set(self.feature_spec.all_features)
        available_features = set(features_df.columns)

        missing = required_features - available_features
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Check for NaN values
        nan_counts = features_df.isnull().sum()
        if nan_counts.any():
            raise ValueError(f"Features contain NaN values: {nan_counts[nan_counts > 0]}")

        return True

    def get_feature_importance_template(self) -> Dict[str, str]:
        """Template for documenting feature importance and rationale."""
        template = {}
        for feature in self.feature_spec.all_features:
            template[feature] = "Economic rationale and expected importance"
        return template


# Default schema instance
default_schema = MLSchema()

# Example usage
if __name__ == "__main__":
    schema = MLSchema()

    print("Label Specification:")
    print(f"  Horizon: {schema.label_spec.horizon_bars} bars")
    print(f"  RR Threshold: {schema.label_spec.rr_threshold}")
    print(f"  Max Adverse Excursion: {schema.label_spec.max_adverse_excursion} std dev")
    print(f"  Reversion Threshold: {schema.label_spec.reversion_threshold} std dev")
    print(f"  Embargo: {schema.label_spec.embargo_bars} bars")
    print(f"  Purge Window: {schema.label_spec.purge_window} bars")

    print(f"\nTotal Features: {len(schema.feature_spec.all_features)}")
    print("Feature Categories:")
    print(f"  Spread: {len(schema.feature_spec.spread_features)}")
    print(f"  Volatility: {len(schema.feature_spec.volatility_features)}")
    print(f"  Correlation: {len(schema.feature_spec.correlation_features)}")
    print(f"  Regime: {len(schema.feature_spec.regime_features)}")
    print(f"  Temporal: {len(schema.feature_spec.temporal_features)}")