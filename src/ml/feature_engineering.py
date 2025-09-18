"""
ML Feature Engineering module for FX-Commodity correlation arbitrage strategy.
Implements comprehensive feature creation and label generation for ML models.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
import warnings

# Import data leakage prevention tools
from src.ml.data_leakage_prevention import DataLeakagePrevention

# Suppress warnings
warnings.filterwarnings('ignore')


def create_technical_features(
    signals_df: pd.DataFrame,
    lookback_periods: List[int] = [5, 10, 20, 60],
    embargo_period: int = 1
) -> pd.DataFrame:
    """
    Create technical analysis features from signals data.
    
    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for feature calculation.
        embargo_period: Number of periods to embargo to prevent lookahead bias.
        
    Returns:
        DataFrame with technical features.
    """
    features = pd.DataFrame(index=signals_df.index)
    
    # Z-score features with embargo
    if "spread_z" in signals_df.columns:
        features["spread_z"] = signals_df["spread_z"].shift(embargo_period)
        
        # Lagged z-scores with embargo
        for period in lookback_periods:
            features[f"spread_z_lag_{period}"] = signals_df["spread_z"].shift(period + embargo_period)
            
        # Z-score changes with embargo
        for period in lookback_periods:
            features[f"spread_z_change_{period}"] = (
                signals_df["spread_z"].shift(embargo_period) -
                signals_df["spread_z"].shift(period + embargo_period)
            )
            
        # Z-score momentum with embargo
        for period in lookback_periods:
            features[f"spread_z_momentum_{period}"] = (
                signals_df["spread_z"].shift(embargo_period) -
                signals_df["spread_z"].shift(period + embargo_period)
            ) / period
            
    # Spread features with embargo
    if "spread" in signals_df.columns:
        features["spread"] = signals_df["spread"].shift(embargo_period)
        
        # Rolling statistics with embargo
        for period in lookback_periods:
            features[f"spread_mean_{period}"] = (
                signals_df["spread"].shift(embargo_period).rolling(period).mean()
            )
            features[f"spread_std_{period}"] = (
                signals_df["spread"].shift(embargo_period).rolling(period).std()
            )
            features[f"spread_min_{period}"] = (
                signals_df["spread"].shift(embargo_period).rolling(period).min()
            )
            features[f"spread_max_{period}"] = (
                signals_df["spread"].shift(embargo_period).rolling(period).max()
            )
            
        # Spread ratios with embargo
        for i, period1 in enumerate(lookback_periods[:-1]):
            for period2 in lookback_periods[i+1:]:
                features[f"spread_mean_ratio_{period1}_{period2}"] = (
                    features[f"spread_mean_{period1}"] / features[f"spread_mean_{period2}"]
                )
                
    # Price features with embargo
    for price_col in ["fx_price", "comd_price"]:
        if price_col in signals_df.columns:
            base_name = price_col.replace("_price", "")
            
            # Returns with embargo
            features[f"{base_name}_returns"] = signals_df[price_col].pct_change().shift(embargo_period)
            
            # Volatility with embargo
            for period in lookback_periods:
                features[f"{base_name}_volatility_{period}"] = (
                    features[f"{base_name}_returns"].rolling(period).std()
                )
                
            # Price momentum with embargo
            for period in lookback_periods:
                features[f"{base_name}_momentum_{period}"] = (
                    signals_df[price_col].shift(embargo_period) /
                    signals_df[price_col].shift(period + embargo_period) - 1
                )
                
            # Price ratios to moving averages with embargo
            for period in lookback_periods:
                ma = signals_df[price_col].shift(embargo_period).rolling(period).mean()
                features[f"{base_name}_price_to_ma_{period}"] = (
                    signals_df[price_col].shift(embargo_period) / ma
                )
                
    return features


def create_regime_features(
    signals_df: pd.DataFrame,
    lookback_periods: List[int] = [20, 60, 120]
) -> pd.DataFrame:
    """
    Create regime detection features.
    
    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for regime calculation.
        
    Returns:
        DataFrame with regime features.
    """
    features = pd.DataFrame(index=signals_df.index)
    
    # Volatility regime features
    if "fx_price" in signals_df.columns:
        fx_returns = signals_df["fx_price"].pct_change()
        
        for period in lookback_periods:
            # Rolling volatility
            vol = fx_returns.rolling(period).std()
            features[f"fx_volatility_{period}"] = vol
            
            # Volatility percentile rank
            features[f"fx_volatility_percentile_{period}"] = (
                vol.rolling(period * 2).rank(pct=True)
            )
            
            # Volatility changes
            features[f"fx_volatility_change_{period}"] = vol.diff(period)
            
    # Trend regime features
    if "fx_price" in signals_df.columns:
        for period in lookback_periods:
            # Price trend
            trend = signals_df["fx_price"].diff(period)
            features[f"fx_trend_{period}"] = trend
            
            # Trend strength (trend / volatility)
            vol = fx_returns.rolling(period).std()
            features[f"fx_trend_strength_{period}"] = trend / (vol + 1e-8)
            
    # Spread regime features
    if "spread" in signals_df.columns:
        for period in lookback_periods:
            spread_returns = signals_df["spread"].diff()
            
            # Spread volatility
            spread_vol = spread_returns.rolling(period).std()
            features[f"spread_volatility_{period}"] = spread_vol
            
            # Spread trend
            spread_trend = signals_df["spread"].diff(period)
            features[f"spread_trend_{period}"] = spread_trend
            
            # Spread stationarity (ADF-like feature)
            features[f"spread_stationarity_{period}"] = (
                spread_trend.abs() / (spread_vol + 1e-8)
            )
            
    return features


def create_signal_features(
    signals_df: pd.DataFrame,
    lookback_periods: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Create features based on trading signals.
    
    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for signal calculation.
        
    Returns:
        DataFrame with signal features.
    """
    features = pd.DataFrame(index=signals_df.index)
    
    # Signal strength features
    if "signal" in signals_df.columns:
        features["signal"] = signals_df["signal"]
        
        # Signal persistence
        for period in lookback_periods:
            features[f"signal_persistence_{period}"] = (
                signals_df["signal"].rolling(period).apply(
                    lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0
                )
            )
            
        # Signal changes
        features["signal_changes"] = signals_df["signal"].diff().abs()
        
        # Consecutive same signals
        signal_diff = signals_df["signal"].diff()
        consecutive = pd.Series(0, index=signals_df.index)
        for i in range(1, len(signal_diff)):
            if signal_diff.iloc[i] == 0 and signals_df["signal"].iloc[i] != 0:
                consecutive.iloc[i] = consecutive.iloc[i-1] + 1
            elif signals_df["signal"].iloc[i] != 0:
                consecutive.iloc[i] = 1
        features["consecutive_signals"] = consecutive
        
    # Entry/exit features
    for flag_col in ["enter_long", "enter_short", "exit_long", "exit_short"]:
        if flag_col in signals_df.columns:
            features[flag_col] = signals_df[flag_col].astype(int)
            
            # Recent activity
            for period in lookback_periods:
                features[f"{flag_col}_count_{period}"] = (
                    signals_df[flag_col].rolling(period).sum()
                )
                
    return features


def create_time_features(
    signals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create time-based features.
    
    Args:
        signals_df: DataFrame with signals and market data.
        
    Returns:
        DataFrame with time features.
    """
    features = pd.DataFrame(index=signals_df.index)
    
    # Basic time features
    features["day_of_week"] = signals_df.index.dayofweek
    features["month"] = signals_df.index.month
    features["quarter"] = signals_df.index.quarter
    features["year"] = signals_df.index.year
    
    # Cyclical time features (sin/cos encoding)
    features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    
    # Business cycle features
    features["is_month_start"] = (signals_df.index.day <= 3).astype(int)
    features["is_month_end"] = (signals_df.index.day >= 28).astype(int)
    features["is_quarter_start"] = signals_df.index.is_quarter_start.astype(int)
    features["is_quarter_end"] = signals_df.index.is_quarter_end.astype(int)
    
    return features


def create_advanced_features(
    signals_df: pd.DataFrame,
    lookback_periods: List[int] = [10, 20, 60]
) -> pd.DataFrame:
    """
    Create advanced statistical and econometric features.
    
    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for feature calculation.
        
    Returns:
        DataFrame with advanced features.
    """
    features = pd.DataFrame(index=signals_df.index)
    
    # Statistical features
    if "spread_z" in signals_df.columns:
        for period in lookback_periods:
            # Skewness and kurtosis
            features[f"spread_z_skew_{period}"] = (
                signals_df["spread_z"].rolling(period).apply(stats.skew, nan_policy='omit')
            )
            features[f"spread_z_kurt_{period}"] = (
                signals_df["spread_z"].rolling(period).apply(stats.kurtosis, nan_policy='omit')
            )
            
            # Autocorrelation
            features[f"spread_z_autocorr_{period}"] = (
                signals_df["spread_z"].rolling(period).apply(
                    lambda x: x.autocorr() if len(x.dropna()) > 2 else 0
                )
            )
            
    # Econometric features
    if "fx_price" in signals_df.columns and "comd_price" in signals_df.columns:
        fx_returns = signals_df["fx_price"].pct_change()
        comd_returns = signals_df["comd_price"].pct_change()
        
        for period in lookback_periods:
            # Correlation between FX and commodity returns
            rolling_corr = fx_returns.rolling(period).corr(comd_returns)
            features[f"fx_comd_corr_{period}"] = rolling_corr
            
            # Beta (FX sensitivity to commodity)
            valid_data = pd.concat([fx_returns, comd_returns], axis=1).dropna()
            if len(valid_data) >= period:
                features[f"fx_beta_{period}"] = (
                    valid_data.iloc[:, 0].rolling(period).cov(valid_data.iloc[:, 1]) / 
                    valid_data.iloc[:, 1].rolling(period).var()
                )
                
            # Cointegration residual features
            if "spread" in signals_df.columns:
                # Hurst exponent (simplified)
                def hurst_exponent(series):
                    if len(series.dropna()) < 10:
                        return 0.5
                    try:
                        lags = range(2, min(20, len(series) // 4))
                        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
                        poly = np.polyfit(np.log(lags), np.log(tau), 1)
                        return poly[0] * 2.0
                    except:
                        return 0.5
                        
                features[f"hurst_exponent_{period}"] = (
                    signals_df["spread"].rolling(period).apply(hurst_exponent, raw=False)
                )
                
    return features


def create_comprehensive_features(
    signals_df: pd.DataFrame,
    lookback_periods: List[int] = [5, 10, 20, 60]
) -> pd.DataFrame:
    """
    Create comprehensive feature set by combining all feature types.
    
    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for feature calculation.
        
    Returns:
        DataFrame with all engineered features.
    """
    # Create different types of features
    technical_features = create_technical_features(signals_df, lookback_periods)
    regime_features = create_regime_features(signals_df, lookback_periods)
    signal_features = create_signal_features(signals_df, lookback_periods)
    time_features = create_time_features(signals_df)
    advanced_features = create_advanced_features(signals_df, lookback_periods)
    
    # Combine all features
    all_features = pd.concat([
        technical_features,
        regime_features,
        signal_features,
        time_features,
        advanced_features
    ], axis=1)
    
    # Remove duplicate columns
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    # Handle infinite values
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0 (or use forward fill for time series)
    all_features = all_features.fillna(0)
    
    logger.info(f"Created {all_features.shape[1]} comprehensive features for {len(all_features)} samples")
    
    return all_features


def create_training_labels(
    signals_df: pd.DataFrame,
    forward_return_window: int = 5,
    return_threshold: float = 0.0,
    label_type: str = "binary"
) -> pd.Series:
    """
    Create training labels for ML model with multiple options.
    
    Args:
        signals_df: DataFrame with signals and returns.
        forward_return_window: Window for calculating forward returns.
        return_threshold: Threshold for positive/negative labels.
        label_type: Type of labels to create ("binary", "multi", "regression").
        
    Returns:
        Series with training labels.
    """
    # Calculate forward returns
    if "spread" in signals_df.columns:
        forward_returns = (
            signals_df["spread"].diff(forward_return_window).shift(-forward_return_window)
        )
    else:
        # Fallback to simple returns
        if "fx_price" in signals_df.columns:
            forward_returns = (
                signals_df["fx_price"].pct_change(forward_return_window).shift(-forward_return_window)
            )
        else:
            # Fallback to simple random returns
            forward_returns = pd.Series(
                np.random.randn(len(signals_df)), index=signals_df.index
            )
            
    # Create labels based on type
    if label_type == "binary":
        # Binary classification (profitable vs unprofitable)
        labels = (forward_returns > return_threshold).astype(int)
    elif label_type == "multi":
        # Multi-class classification (strong buy, buy, hold, sell, strong sell)
        labels = pd.Series(0, index=forward_returns.index)  # Default to hold
        labels[forward_returns > (return_threshold + 0.02)] = 2  # Strong buy
        labels[(forward_returns > return_threshold) & (forward_returns <= (return_threshold + 0.02))] = 1  # Buy
        labels[forward_returns < (return_threshold - 0.02)] = -2  # Strong sell
        labels[(forward_returns < return_threshold) & (forward_returns >= (return_threshold - 0.02))] = -1  # Sell
    else:  # regression
        # Regression target (actual returns)
        labels = forward_returns
        
    # Drop NaN values
    labels = labels.dropna()
    
    # Log label distribution
    if label_type == "binary":
        logger.info(f"Created {len(labels)} binary labels ({labels.mean():.2%} positive)")
    elif label_type == "multi":
        logger.info(f"Created {len(labels)} multi-class labels")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    else:
        logger.info(f"Created {len(labels)} regression labels")
        
    return labels


def create_features_and_labels(
    signals_df: pd.DataFrame,
    lookback_periods: List[int] = [5, 10, 20, 60],
    forward_return_window: int = 5,
    return_threshold: float = 0.0,
    label_type: str = "binary"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create both features and labels for ML model training.
    
    Args:
        signals_df: DataFrame with signals and market data.
        lookback_periods: List of lookback periods for feature calculation.
        forward_return_window: Window for calculating forward returns.
        return_threshold: Threshold for positive/negative labels.
        label_type: Type of labels to create.
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    # Create features
    features = create_comprehensive_features(signals_df, lookback_periods)
    
    # Create labels
    labels = create_training_labels(
        signals_df, forward_return_window, return_threshold, label_type
    )
    
    # Align features and labels by index
    common_index = features.index.intersection(labels.index)
    features_aligned = features.loc[common_index]
    labels_aligned = labels.loc[common_index]
    
    # Remove rows with all zero features (likely due to lookback periods)
    non_zero_rows = (features_aligned != 0).any(axis=1)
    features_final = features_aligned[non_zero_rows]
    labels_final = labels_aligned[non_zero_rows]
    
    logger.info(f"Final dataset: {len(features_final)} samples, {features_final.shape[1]} features")
    
    return features_final, labels_final


def get_feature_importance_ranks(
    features: pd.DataFrame,
    labels: pd.Series,
    method: str = "correlation"
) -> pd.Series:
    """
    Calculate feature importance ranks.
    
    Args:
        features: Feature DataFrame.
        labels: Target Series.
        method: Method for calculating importance ("correlation", "mutual_info").
        
    Returns:
        Series with feature importance ranks.
    """
    if method == "correlation":
        # Pearson correlation with target
        correlations = features.corrwith(labels).abs()
        return correlations.sort_values(ascending=False)
    elif method == "mutual_info":
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            if len(np.unique(labels)) <= 10:  # Classification
                importances = mutual_info_classif(features, labels)
            else:  # Regression
                importances = mutual_info_regression(features, labels)
                
            importance_series = pd.Series(importances, index=features.columns)
            return importance_series.sort_values(ascending=False)
        except ImportError:
            logger.warning("sklearn not available for mutual information calculation")
            return pd.Series()
    else:
        return pd.Series()


# Example usage function
def example_feature_pipeline(signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Example pipeline for creating features and labels.
    
    Args:
        signals_df: DataFrame with signals and market data.
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    # Create comprehensive features and labels
    features, labels = create_features_and_labels(
        signals_df,
        lookback_periods=[5, 10, 20, 60],
        forward_return_window=5,
        return_threshold=0.0,
        label_type="binary"
    )
    
    # Get feature importance ranks
    importance_ranks = get_feature_importance_ranks(features, labels, method="correlation")
    
    # Log top features
    logger.info("Top 10 most important features:")
    for i, (feature, importance) in enumerate(importance_ranks.head(10).items()):
        logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
    return features, labels