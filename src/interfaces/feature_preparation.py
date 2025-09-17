"""
Feature preparation interface for ML model integration.
Provides abstraction layer between strategy and ML components.
"""

from typing import Dict
from abc import ABC, abstractmethod
import pandas as pd


class FeaturePreparator(ABC):
    """Abstract interface for preparing features for ML models."""

    @abstractmethod
    def prepare_features(
        self, fx_series: pd.Series, comd_series: pd.Series, config: Dict
    ) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.

        Args:
            fx_series: FX time series.
            comd_series: Commodity time series.
            config: Configuration dictionary.

        Returns:
            DataFrame with prepared features.
        """
        pass


class DefaultFeaturePreparator(FeaturePreparator):
    """Default implementation of feature preparation."""

    def prepare_features(
        self, fx_series: pd.Series, comd_series: pd.Series, config: Dict
    ) -> pd.DataFrame:
        """
        Prepare features for model training/prediction using default strategy.

        Args:
            fx_series: FX time series.
            comd_series: Commodity time series.
            config: Configuration dictionary.

        Returns:
            DataFrame with prepared features.
        """
        # Import here to avoid circular dependencies
        from src.strategy.mean_reversion import _prepare_features_for_model

        return _prepare_features_for_model(fx_series, comd_series, config)


def create_feature_preparator(preparator_type: str = "default") -> FeaturePreparator:
    """
    Factory function to create feature preparators.

    Args:
        preparator_type: Type of preparator to create.

    Returns:
        FeaturePreparator instance.

    Raises:
        ValueError: If preparator_type is not supported.
    """
    if preparator_type == "default":
        return DefaultFeaturePreparator()
    else:
        raise ValueError(f"Unsupported preparator type: {preparator_type}")


# Backward compatibility function - delegates to the strategy module
def prepare_features_for_model(
    fx_series: pd.Series, comd_series: pd.Series, config: Dict
) -> pd.DataFrame:
    """
    Backward compatibility wrapper for feature preparation.

    This function maintains backward compatibility while providing
    a clean public interface for feature preparation.

    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.
        config: Configuration dictionary.

    Returns:
        DataFrame with prepared features.
    """
    preparator = create_feature_preparator("default")
    return preparator.prepare_features(fx_series, comd_series, config)
