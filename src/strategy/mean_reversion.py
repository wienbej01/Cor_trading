"""
Mean reversion strategy module for FX-Commodity correlation arbitrage.
Implements signal generation and position sizing for spread trading.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.features.indicators import atr_proxy, zscore, zscore_robust, atr
from src.features.spread import compute_spread
from src.features.regime import combined_regime_filter, correlation_gate
from statsmodels.tsa.stattools import adfuller


def adf_pvalue(series: pd.Series) -> float:
    """
    Calculate the p-value of the Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test for stationarity.
        
    Returns:
        p-value from the ADF test.
    """
    try:
        result = adfuller(series.dropna())
        return result[1]  # p-value is the second element in the result tuple
    except Exception as e:
        logger.warning(f"ADF test failed: {e}")
        return 1.0  # Return high p-value (non-stationary) if test fails


def generate_signals(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    regime_filter: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate trading signals for mean reversion strategy.
    
    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.
        config: Configuration dictionary with strategy parameters.
        regime_filter: Optional boolean series for regime filtering.
        
    Returns:
        DataFrame with signals and related metrics.
        
    Raises:
        ValueError: If required config parameters are missing or series are invalid.
    """
    logger.info("Generating mean reversion signals")
    
    # Extract config parameters
    try:
        beta_window = config["lookbacks"]["beta_window"]
        z_window = config["lookbacks"]["z_window"]
        entry_z = config["thresholds"]["entry_z"]
        exit_z = config["thresholds"]["exit_z"]
        stop_z = config["thresholds"]["stop_z"]
        atr_window = config["sizing"]["atr_window"]
        corr_window = config["lookbacks"]["corr_window"]
        min_abs_corr = config["regime"]["min_abs_corr"]
        use_kalman = config.get("use_kalman", True)
        inverse_fx_for_quote_ccy_strength = config["inverse_fx_for_quote_ccy_strength"]
    except KeyError as e:
        raise ValueError(f"Missing required config parameter: {e}")
    
    # Validate series
    if len(fx_series) != len(comd_series):
        raise ValueError("FX and commodity series must have the same length")
    
    if len(fx_series) < max(beta_window, z_window) + 10:
        raise ValueError("Insufficient data for signal generation")
    
    # Create result DataFrame
    result = pd.DataFrame(index=fx_series.index)
    result["fx_price"] = fx_series
    result["comd_price"] = comd_series
    
    # Compute spread
    spread, alpha, beta = compute_spread(
        fx_series, comd_series, beta_window, use_kalman
    )
    result["spread"] = spread
    result["alpha"] = alpha
    result["beta"] = beta
    
    # Compute robust z-score
    z = zscore_robust(spread, z_window).rename("z")
    result["spread_z"] = z
    
    # Implement softer gating logic
    regime_ok = correlation_gate(fx_series, comd_series, corr_window, min_abs_corr)
    p_adf = adf_pvalue(spread)
    
    # STRONGER but realistic gating: allow trades when either corr OR cointegration passes
    adf_ok = (p_adf <= 0.10)
    good_regime = (regime_ok | adf_ok)
    result["good_regime"] = good_regime
    result["adf_p"] = p_adf
    
    # Generate raw signals
    result["raw_signal"] = 0  # 0: no position, 1: long spread, -1: short spread
    
    # entries/exits (we'll tune thresholds below in YAML)
    enter_long = (z <= -entry_z) & good_regime
    enter_short = (z >= entry_z) & good_regime
    exit_rule = (z.abs() <= exit_z)
    
    # Add entry/exit flags for diagnostics
    result["enter_long"] = enter_long
    result["enter_short"] = enter_short
    result["exit"] = exit_rule
    
    result.loc[enter_long, "raw_signal"] = 1
    result.loc[enter_short, "raw_signal"] = -1
    
    # Exit signals - ensure they have the same index as result
    long_exit = pd.Series(z >= -exit_z, index=result.index)
    short_exit = pd.Series(z <= exit_z, index=result.index)
    
    # Stop loss signals - ensure they have the same index as result
    long_stop = pd.Series(z >= -stop_z, index=result.index)
    short_stop = pd.Series(z <= stop_z, index=result.index)
    
    # Apply signal logic with proper state management
    position = 0  # Current position
    signals = []  # List to store signals
    
    for idx, row in result.iterrows():
        current_signal = row["raw_signal"]
        
        # If we have no position
        if position == 0:
            # Enter new position if signal is non-zero
            if current_signal != 0:
                position = current_signal
                signals.append(current_signal)
            else:
                signals.append(0)
        # If we have a long position
        elif position == 1:
            # Exit if we hit exit threshold or stop loss
            if long_exit.loc[idx] or long_stop.loc[idx]:
                position = 0
                signals.append(0)
            else:
                signals.append(1)
        # If we have a short position
        elif position == -1:
            # Exit if we hit exit threshold or stop loss
            if short_exit.loc[idx] or short_stop.loc[idx]:
                position = 0
                signals.append(0)
            else:
                signals.append(-1)
        else:
            signals.append(0)
    
    result["signal"] = signals
    
    # Apply regime filter if provided
    if regime_filter is not None:
        result["signal"] = result["signal"].where(regime_filter, 0)
        logger.info(f"Applied regime filter: {(result['signal'] != 0).sum()} active signals remaining")
    
    # Calculate position sizes
    position_sizes = calculate_position_sizes(
        result, atr_window, config["sizing"]["target_vol_per_leg"], inverse_fx_for_quote_ccy_strength
    )
    result = pd.concat([result, position_sizes], axis=1)
    
    # Add trade entry/exit flags
    result["entry"] = result["signal"].diff().abs() == 1
    result["exit"] = (result["signal"].diff() != 0) & (result["signal"] == 0)
    
    logger.info(f"Generated {result['entry'].sum()} entry signals and {result['exit'].sum()} exit signals")
    
    return result


def calculate_position_sizes(
    signals_df: pd.DataFrame,
    atr_window: int,
    target_vol_per_leg: float,
    inverse_fx_for_quote_ccy_strength: bool
) -> pd.DataFrame:
    """
    Calculate position sizes based on inverse volatility sizing.
    
    Args:
        signals_df: DataFrame with signals and prices.
        atr_window: Window for ATR calculation.
        target_vol_per_leg: Target volatility per leg of the trade.
        inverse_fx_for_quote_ccy_strength: Whether to inverse FX for quote currency strength.
        
    Returns:
        DataFrame with position sizes.
    """
    logger.debug("Calculating position sizes")
    
    # Calculate ATR
    fx_atr = atr(signals_df["fx_price"], atr_window)
    comd_atr = atr(signals_df["comd_price"], atr_window)
    
    # Calculate position sizes (inverse volatility)
    fx_size = target_vol_per_leg / fx_atr
    comd_size = target_vol_per_leg / comd_atr
    
    # Adjust for FX quote currency strength if needed
    if inverse_fx_for_quote_ccy_strength:
        fx_size = fx_size / signals_df["fx_price"]
    
    # Create result DataFrame
    result = pd.DataFrame(index=signals_df.index)
    result["fx_size"] = fx_size
    result["comd_size"] = comd_size
    
    # Apply position direction to sizes
    result["fx_position"] = result["fx_size"] * signals_df["signal"]
    result["comd_position"] = -result["comd_size"] * signals_df["signal"]  # Opposite side of spread
    
    return result


def apply_time_stop(
    signals_df: pd.DataFrame,
    max_days: int
) -> pd.DataFrame:
    """
    Apply time-based stop to positions.
    
    Args:
        signals_df: DataFrame with signals.
        max_days: Maximum number of days to hold a position.
        
    Returns:
        DataFrame with time-stop applied.
    """
    logger.debug(f"Applying time stop with max_days={max_days}")
    
    result = signals_df.copy()
    result["time_stop_exit"] = False
    
    # Track position entry dates
    position_entry_date = None
    current_position = 0
    
    for idx, row in result.iterrows():
        # Check for position entry
        if row["entry"] and row["signal"] != 0:
            position_entry_date = idx
            current_position = row["signal"]
        
        # Check for position exit
        elif row["exit"] or row["signal"] == 0:
            position_entry_date = None
            current_position = 0
        
        # Apply time stop if position has been held too long
        elif position_entry_date is not None and current_position != 0:
            days_held = (idx - position_entry_date).days
            
            if days_held >= max_days:
                result.loc[idx, "signal"] = 0
                result.loc[idx, "time_stop_exit"] = True
                position_entry_date = None
                current_position = 0
    
    # Recalculate positions after time stop
    position_sizes = calculate_position_sizes(
        result, 
        20,  # Default ATR window
        result["fx_size"].iloc[0] * result["fx_atr"].iloc[0] if "fx_atr" in result.columns else 0.01,
        True
    )
    
    result["fx_position"] = position_sizes["fx_position"]
    result["comd_position"] = position_sizes["comd_position"]
    
    return result


def generate_signals_with_regime_filter(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict
) -> pd.DataFrame:
    """
    Generate signals with regime filtering applied.
    
    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.
        config: Configuration dictionary.
        
    Returns:
        DataFrame with signals and regime filtering applied.
    """
    logger.info("Generating signals with regime filtering")
    
    # Calculate regime filter
    regime_filter = combined_regime_filter(fx_series, comd_series, config)
    
    # Generate signals with regime filter
    signals_df = generate_signals(fx_series, comd_series, config, regime_filter)
    
    # Apply time stop if specified
    if "time_stop" in config:
        max_days = config["time_stop"]["max_days"]
        signals_df = apply_time_stop(signals_df, max_days)
    
    return signals_df