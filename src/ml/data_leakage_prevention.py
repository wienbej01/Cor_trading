"""
Data Leakage Prevention module for ML models in FX-Commodity correlation arbitrage strategy.
Implements tools to detect and prevent lookahead bias and data leakage in feature engineering.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class DataLeakagePrevention:
    """Class to prevent data leakage and lookahead bias in ML features."""
    
    def __init__(self):
        """Initialize data leakage prevention tools."""
        self.leakage_rules = self._define_leakage_rules()
        self.validation_results = {}
        
    def _define_leakage_rules(self) -> Dict:
        """
        Define rules for detecting potential data leakage.
        
        Returns:
            Dictionary with leakage detection rules.
        """
        return {
            "lookahead_bias": {
                "description": "Features that use future information",
                "check_function": self._check_lookahead_bias
            },
            "target_leakage": {
                "description": "Features that directly encode target information",
                "check_function": self._check_target_leakage
            },
            "future_data_leakage": {
                "description": "Features that incorporate future data points",
                "check_function": self._check_future_data_leakage
            },
            "information_leakage": {
                "description": "Features that leak information from other samples",
                "check_function": self._check_information_leakage
            }
        }
        
    def validate_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series = None,
        signals_df: pd.DataFrame = None,
        timestamp_column: str = None
    ) -> Dict:
        """
        Validate features for potential data leakage.
        
        Args:
            features: DataFrame with features to validate.
            target: Target series (optional).
            signals_df: Original signals DataFrame (optional).
            timestamp_column: Name of timestamp column (optional).
            
        Returns:
            Dictionary with validation results.
        """
        validation_results = {}
        
        # Check each leakage rule
        for rule_name, rule_info in self.leakage_rules.items():
            try:
                is_valid, details = rule_info["check_function"](
                    features, target, signals_df, timestamp_column
                )
                validation_results[rule_name] = {
                    "valid": is_valid,
                    "details": details
                }
            except Exception as e:
                validation_results[rule_name] = {
                    "valid": False,
                    "details": f"Validation failed: {str(e)}"
                }
                
        self.validation_results = validation_results
        return validation_results
        
    def _check_lookahead_bias(
        self, 
        features: pd.DataFrame, 
        target: pd.Series = None,
        signals_df: pd.DataFrame = None,
        timestamp_column: str = None
    ) -> Tuple[bool, str]:
        """
        Check for lookahead bias in features.
        
        Args:
            features: DataFrame with features to check.
            target: Target series (optional).
            signals_df: Original signals DataFrame (optional).
            timestamp_column: Name of timestamp column (optional).
            
        Returns:
            Tuple of (is_valid, details).
        """
        issues = []
        
        # Check for future data usage in rolling calculations
        for col in features.columns:
            if col.startswith(("future_", "forward_", "lead_")):
                issues.append(f"Feature {col} may contain future data")
                
        # Check for features that correlate too strongly with target at lag 0
        if target is not None and len(features) == len(target):
            # Align indices
            combined = pd.concat([features, target], axis=1).dropna()
            if len(combined) > 0:
                target_col = combined.columns[-1]
                correlations = combined.corr()[target_col].abs()
                
                # Check for very high correlations (potential leakage)
                high_corr_features = correlations[correlations > 0.95].index.tolist()
                high_corr_features = [f for f in high_corr_features if f != target_col]
                
                if high_corr_features:
                    issues.append(
                        f"High correlation with target detected: {high_corr_features}"
                    )
                    
        if issues:
            return False, "; ".join(issues)
        return True, "No lookahead bias detected"
        
    def _check_target_leakage(
        self, 
        features: pd.DataFrame, 
        target: pd.Series = None,
        signals_df: pd.DataFrame = None,
        timestamp_column: str = None
    ) -> Tuple[bool, str]:
        """
        Check for target leakage in features.
        
        Args:
            features: DataFrame with features to check.
            target: Target series (optional).
            signals_df: Original signals DataFrame (optional).
            timestamp_column: Name of timestamp column (optional).
            
        Returns:
            Tuple of (is_valid, details).
        """
        issues = []
        
        # Check for features that directly encode target information
        forbidden_patterns = [
            "target", "label", "return", "pnl", "profit", 
            "loss", "gain", "outcome", "result"
        ]
        
        for col in features.columns:
            col_lower = col.lower()
            for pattern in forbidden_patterns:
                if pattern in col_lower and not col_lower.startswith("time_"):
                    issues.append(f"Feature {col} may encode target information")
                    
        # Check for exact match with target values
        if target is not None and len(features) == len(target):
            # Align indices
            common_index = features.index.intersection(target.index)
            features_aligned = features.loc[common_index]
            target_aligned = target.loc[common_index]
            
            if len(common_index) > 0:
                for col in features_aligned.columns:
                    # Check if feature values are identical to target
                    if np.array_equal(
                        features_aligned[col].dropna().values, 
                        target_aligned.dropna().values
                    ):
                        issues.append(f"Feature {col} is identical to target")
                        
        if issues:
            return False, "; ".join(issues)
        return True, "No target leakage detected"
        
    def _check_future_data_leakage(
        self, 
        features: pd.DataFrame, 
        target: pd.Series = None,
        signals_df: pd.DataFrame = None,
        timestamp_column: str = None
    ) -> Tuple[bool, str]:
        """
        Check for future data leakage in features.
        
        Args:
            features: DataFrame with features to check.
            target: Target series (optional).
            signals_df: Original signals DataFrame (optional).
            timestamp_column: Name of timestamp column (optional).
            
        Returns:
            Tuple of (is_valid, details).
        """
        issues = []
        
        # Check for features that use future data points
        # This is a heuristic check - in practice, this would require domain knowledge
        for col in features.columns:
            # Check for features that might be calculated using future data
            if any(keyword in col.lower() for keyword in [
                "future", "forward", "lead", "next", "ahead"
            ]):
                # Check if the feature has values that seem to predict future events
                if len(features[col].dropna()) > 10:
                    # Simple test: check if feature values are consistently higher/lower
                    # than what would be expected from past data
                    feature_values = features[col].dropna().values
                    if len(feature_values) > 5:
                        # Check if there are sudden jumps that might indicate future knowledge
                        diffs = np.diff(feature_values)
                        if np.std(diffs) > 0:
                            percentile_95 = np.percentile(np.abs(diffs), 95)
                            max_diff = np.max(np.abs(diffs))
                            if max_diff > percentile_95 * 5:  # Arbitrary threshold
                                issues.append(
                                    f"Feature {col} shows potential future data usage"
                                )
                                
        if issues:
            return False, "; ".join(issues)
        return True, "No future data leakage detected"
        
    def _check_information_leakage(
        self, 
        features: pd.DataFrame, 
        target: pd.Series = None,
        signals_df: pd.DataFrame = None,
        timestamp_column: str = None
    ) -> Tuple[bool, str]:
        """
        Check for information leakage between samples.
        
        Args:
            features: DataFrame with features to check.
            target: Target series (optional).
            signals_df: Original signals DataFrame (optional).
            timestamp_column: Name of timestamp column (optional).
            
        Returns:
            Tuple of (is_valid, details).
        """
        issues = []
        
        # Check for features that might leak information across samples
        # This is particularly important for time series data
        
        # Check for features that are constant or nearly constant
        for col in features.columns:
            unique_values = features[col].nunique()
            total_values = len(features[col].dropna())
            
            if total_values > 0:
                # If less than 2% of values are unique, it might be problematic
                if unique_values / total_values < 0.02:
                    issues.append(
                        f"Feature {col} has very low diversity ({unique_values}/{total_values} unique values)"
                    )
                    
        # Check for features that are perfectly correlated with each other
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            # Set diagonal to 0 to ignore self-correlations
            np.fill_diagonal(corr_matrix.values, 0)
            
            # Find pairs with correlation > 0.99
            high_corr_pairs = np.where(corr_matrix > 0.99)
            
            if len(high_corr_pairs[0]) > 0:
                correlated_features = set()
                for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                    if i < j:  # Avoid duplicates
                        correlated_features.add(features.columns[i])
                        correlated_features.add(features.columns[j])
                        
                if correlated_features:
                    issues.append(
                        f"Highly correlated features detected: {list(correlated_features)}"
                    )
                    
        if issues:
            return False, "; ".join(issues)
        return True, "No information leakage detected"
        
    def create_safe_features(
        self, 
        signals_df: pd.DataFrame,
        max_lookback: int = 60,
        embargo_period: int = 1
    ) -> pd.DataFrame:
        """
        Create features that are safe from data leakage.
        
        Args:
            signals_df: DataFrame with signals and market data.
            max_lookback: Maximum lookback period for features.
            embargo_period: Number of periods to embargo after each sample.
            
        Returns:
            DataFrame with leakage-safe features.
        """
        safe_features = pd.DataFrame(index=signals_df.index)
        
        # Only use past information
        # Shift all features by embargo_period to prevent leakage
        if "spread_z" in signals_df.columns:
            safe_features["spread_z"] = signals_df["spread_z"].shift(embargo_period)
            
            # Add lagged features with proper embargo
            for period in [5, 10, 20]:
                if period + embargo_period <= max_lookback:
                    safe_features[f"spread_z_lag_{period}"] = (
                        signals_df["spread_z"].shift(period + embargo_period)
                    )
                    
        # Add volatility features with embargo
        if "fx_price" in signals_df.columns:
            fx_returns = signals_df["fx_price"].pct_change()
            safe_features["fx_returns"] = fx_returns.shift(embargo_period)
            
            for period in [5, 10, 20]:
                if period + embargo_period <= max_lookback:
                    safe_features[f"fx_volatility_{period}"] = (
                        fx_returns.rolling(period).std().shift(embargo_period)
                    )
                    
        # Add time features (these are always safe)
        safe_features["day_of_week"] = signals_df.index.dayofweek
        safe_features["month"] = signals_df.index.month
        
        # Remove rows with NaN values (due to lookback periods)
        safe_features = safe_features.dropna()
        
        logger.info(f"Created {len(safe_features.columns)} leakage-safe features")
        return safe_features
        
    def purged_split(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        test_size: float = 0.2,
        embargo_period: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split with embargo to prevent data leakage.
        
        Args:
            features: DataFrame with features.
            target: Target series.
            test_size: Fraction of data to use for testing.
            embargo_period: Number of periods to embargo.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Align indices
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Calculate split point
        n_samples = len(common_index)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        # Apply embargo
        embargo_start = min(n_train + embargo_period, n_samples)
        embargo_end = max(n_train - embargo_period, 0)
        
        # Create splits
        train_indices = common_index[:embargo_end]
        test_indices = common_index[embargo_start:]
        
        X_train = features_aligned.loc[train_indices]
        X_test = features_aligned.loc[test_indices]
        y_train = target_aligned.loc[train_indices]
        y_test = target_aligned.loc[test_indices]
        
        logger.info(
            f"Purged split: {len(X_train)} train samples, {len(X_test)} test samples"
        )
        
        return X_train, X_test, y_train, y_test


def validate_feature_set(
    features: pd.DataFrame,
    target: pd.Series = None,
    signals_df: pd.DataFrame = None,
    timestamp_column: str = None
) -> Dict:
    """
    Validate a feature set for data leakage.
    
    Args:
        features: DataFrame with features to validate.
        target: Target series (optional).
        signals_df: Original signals DataFrame (optional).
        timestamp_column: Name of timestamp column (optional).
        
    Returns:
        Dictionary with validation results.
    """
    validator = DataLeakagePrevention()
    return validator.validate_features(features, target, signals_df, timestamp_column)


def create_leakage_safe_pipeline(
    signals_df: pd.DataFrame,
    max_lookback: int = 60,
    embargo_period: int = 1
) -> pd.DataFrame:
    """
    Create a feature pipeline that is safe from data leakage.
    
    Args:
        signals_df: DataFrame with signals and market data.
        max_lookback: Maximum lookback period for features.
        embargo_period: Number of periods to embargo after each sample.
        
    Returns:
        DataFrame with leakage-safe features.
    """
    validator = DataLeakagePrevention()
    safe_features = validator.create_safe_features(signals_df, max_lookback, embargo_period)
    
    # Validate the features
    validation_results = validator.validate_features(safe_features)
    
    # Log validation results
    for rule, result in validation_results.items():
        if result["valid"]:
            logger.info(f"✓ {rule}: {result['details']}")
        else:
            logger.warning(f"✗ {rule}: {result['details']}")
            
    return safe_features


def example_leakage_prevention(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Example of how to use the data leakage prevention tools.
    
    Args:
        signals_df: DataFrame with signals and market data.
        
    Returns:
        DataFrame with leakage-safe features.
    """
    # Create safe features
    safe_features = create_leakage_safe_pipeline(
        signals_df, 
        max_lookback=60, 
        embargo_period=1
    )
    
    # Validate features
    validation_results = validate_feature_set(safe_features)
    
    # Print validation summary
    print("Data Leakage Validation Results:")
    for rule, result in validation_results.items():
        status = "✓ PASS" if result["valid"] else "✗ FAIL"
        print(f"  {status}: {rule} - {result['details']}")
        
    return safe_features