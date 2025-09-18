"""
Mean reversion strategy module for FX-Commodity correlation arbitrage.
Implements signal generation and position sizing for spread trading.
"""

from typing import Dict, Optional

import pandas as pd
import numpy as np
from loguru import logger

from features.indicators import atr_proxy, zscore_robust
from features.spread import compute_spread
from features.regime import combined_regime_filter, correlation_gate, volatility_regime, trend_regime
from .filters import AllowTradeContext
from statsmodels.tsa.stattools import adfuller
from features.cointegration import ou_half_life, hurst_exponent

# Import ensemble model (commented to avoid circular import with h1_mean_reversion)
# from ml.ensemble import create_ensemble_model, ModelConfig

# Import validation interfaces for consistent error handling
from interfaces.validation import (
    validate_series_alignment,
    validate_trading_config,
    safe_parameter_extraction,
    ValidationError,
)


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
    regime_filter: Optional[pd.Series] = None,
    model_name: str = "default",
) -> pd.DataFrame:
    """
    Generate trading signals for mean reversion strategy.

    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.
        config: Configuration dictionary with strategy parameters.
        regime_filter: Optional boolean series for regime filtering.
        model_name: Name of the model to use for signal generation.

    Returns:
        DataFrame with signals and related metrics.

    Raises:
        ValueError: If required config parameters are missing or series are invalid.
        ValidationError: If input validation fails.
    """
    logger.info("Generating mean reversion signals")

    # Input validation using new validation interface
    try:
        validate_series_alignment(fx_series, comd_series)
        validate_trading_config(config)
    except ValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise ValueError(f"Input validation failed: {e}") from e

    # Extract config parameters using safe extraction
    try:
        required_params = [
            "lookbacks.beta_window",
            "lookbacks.z_window",
            "lookbacks.corr_window",
            "thresholds.entry_z",
            "thresholds.exit_z",
            "thresholds.stop_z",
            "sizing.atr_window",
            "regime.min_abs_corr",
        ]
        default_values = {"use_kalman": True, "inverse_fx_for_quote_ccy_strength": True}

        params = safe_parameter_extraction(config, required_params, default_values)

        beta_window = int(params["lookbacks.beta_window"])
        z_window = int(params["lookbacks.z_window"])
        corr_window = int(params["lookbacks.corr_window"])
        entry_z = float(params["thresholds.entry_z"])
        exit_z = float(params["thresholds.exit_z"])
        stop_z = float(params["thresholds.stop_z"])
        atr_window = int(params["sizing.atr_window"])
        min_abs_corr = float(params["regime.min_abs_corr"])
        use_kalman = params.get("use_kalman", True)
        inverse_fx_for_quote_ccy_strength = params.get(
            "inverse_fx_for_quote_ccy_strength", True
        )

    except (ValidationError, ValueError) as e:
        logger.error(f"Parameter extraction failed: {e}")
        raise ValueError(f"Configuration error: {e}") from e

    # Create result DataFrame
    result = pd.DataFrame(index=fx_series.index)
    result["fx_price"] = fx_series
    result["comd_price"] = comd_series

    # Compute spread using existing method (ensemble disabled due to circular import)
    # if model_name != "default":
    #     result = _generate_signals_with_model(
    #         result, fx_series, comd_series, config, model_name
    #     )
    # else:
    spread, alpha, beta = compute_spread(
        fx_series, comd_series, beta_window, use_kalman
    )
    result["spread"] = spread
    result["alpha"] = alpha
    result["beta"] = beta

    # Compute robust z-score
    z = zscore_robust(spread, z_window).rename("z")
    result["spread_z"] = z

    # Calculate ATR for risk management
    result["fx_atr"] = atr_proxy(fx_series, atr_window)
    result["comd_atr"] = atr_proxy(comd_series, atr_window)

    # Regime gating (corr + cointegration diagnostics)
    regime_ok = correlation_gate(fx_series, comd_series, corr_window, min_abs_corr)

    # Compute detailed regimes for AllowTradeContext
    vol_window = config["regime"].get("volatility_window", 20)
    trend_window = config["regime"].get("trend_window", 20)
    use_roc_hp = config["regime"].get("use_roc_hp", True)
    use_vol_quantiles = config["regime"].get("use_vol_quantiles", True)
    
    fx_vol_regime = volatility_regime(
        fx_series, vol_window, use_quantiles=use_vol_quantiles
    )
    comd_vol_regime = volatility_regime(
        comd_series, vol_window, use_quantiles=use_vol_quantiles
    )
    fx_trend_regime = trend_regime(
        fx_series, trend_window, use_roc_hp=use_roc_hp
    )
    comd_trend_regime = trend_regime(
        comd_series, trend_window, use_roc_hp=use_roc_hp
    )
    
    # Combined regime series (e.g., mean of trend regimes for simplicity; enhance as needed)
    combined_regime = (fx_trend_regime + comd_trend_regime) / 2

    # Structural diagnostics evaluated once over full sample (used as soft prior, not a hard block)
    p_adf = adf_pvalue(spread)
    adf_threshold = float(config.get("thresholds", {}).get("adf_p", 0.05))
    adf_ok = p_adf <= adf_threshold
    try:
        ou_hl_val = float(ou_half_life(spread.dropna(), cap=252.0))
    except Exception:
        ou_hl_val = float("inf")
    try:
        hurst_val = float(hurst_exponent(spread.dropna()))
    except Exception:
        hurst_val = 0.5

    # Soften structural checks: allow longer half-life and slightly anti-persistent series
    hl_ok = (ou_hl_val >= 2.0) and (ou_hl_val <= 252.0)  # 2–252 trading days
    hurst_ok = (hurst_val < 0.6)
    coint_ok = adf_ok and hl_ok and hurst_ok

    # External regime filter from caller if provided
    if regime_filter is not None:
        ext_gate = regime_filter.reindex(result.index).fillna(True)
    else:
        ext_gate = pd.Series(True, index=result.index)

    # Initialize AllowTradeContext for pre-entry gating
    filter_ctx = AllowTradeContext(
        config=config,
        current_regime=combined_regime,
        fx_vol_regime=fx_vol_regime,
        comd_vol_regime=comd_vol_regime,
        liquidity_proxy=1.0,  # Placeholder; enhance with real liquidity metric
    )

    # Basic correlation gate still applies
    good_regime = regime_ok & ext_gate

    # Persist diagnostics
    result["good_regime"] = good_regime
    result["adf_p"] = p_adf
    result["ou_half_life"] = ou_hl_val
    result["hurst"] = hurst_val

    # Dynamic z-thresholds (robust to volatility drifts)
    result["raw_signal"] = 0  # 0: flat, 1: long spread, -1: short spread

    spread_vol_20 = spread.rolling(20).std()
    vol_med_252 = spread_vol_20.rolling(252, min_periods=20).median()
    vol_scale = (spread_vol_20 / vol_med_252).clip(0.7, 1.5).fillna(1.0)

    structural_scale = 0.9 if coint_ok else 1.1
    entry_z_dyn = (entry_z * structural_scale * vol_scale).clip(lower=0.8)
    exit_z_dyn = (exit_z * (1.0 / structural_scale) * vol_scale).clip(upper=1.25)

    # Entries/exits with AllowTradeContext gate
    enter_long = (z <= -entry_z_dyn) & good_regime
    enter_short = (z >= entry_z_dyn) & good_regime
    # Apply trade context filter to entries (time-sensitive, so per-timestamp)
    for idx in result.index:
        if enter_long.loc[idx] or enter_short.loc[idx]:
            if not filter_ctx.accept(idx):
                enter_long.loc[idx] = False
                enter_short.loc[idx] = False
                logger.debug(f"Entry blocked by context filter at {idx}")
    
    exit_rule = z.abs() <= exit_z_dyn

    # Persist thresholds for diagnostics
    result["entry_z_dyn"] = entry_z_dyn
    result["exit_z_dyn"] = exit_z_dyn

    # Add profit targets and stop losses
    profit_target = 2.0  # 2x ATR for profit targets
    stop_loss = 1.5  # 1.5x ATR for stop losses

    # Add entry/exit flags for diagnostics
    result["enter_long"] = enter_long
    result["enter_short"] = enter_short
    result["exit"] = exit_rule

    # Enhanced signal generation with profit targets and stop losses
    result["raw_signal"] = 0

    # Dynamic position sizing based on volatility
    result.loc[enter_long, "raw_signal"] = 1
    result.loc[enter_short, "raw_signal"] = -1

    # Exit signals - ensure they have the same index as result
    long_exit = pd.Series(z >= -exit_z_dyn, index=result.index)
    short_exit = pd.Series(z <= exit_z_dyn, index=result.index)

    # Stop losses: cut when adverse excursion grows
    long_stop = pd.Series(z <= -stop_z, index=result.index)   # more negative
    short_stop = pd.Series(z >= stop_z, index=result.index)   # more positive

    # Profit targets (optional, default align with exit to avoid conflict)
    long_profit = pd.Series(False, index=result.index)
    short_profit = pd.Series(False, index=result.index)

    # Add enhanced signal tracking
    result["long_exit"] = long_exit
    result["short_exit"] = short_exit
    result["long_stop"] = long_stop
    result["short_stop"] = short_stop
    result["long_profit"] = long_profit
    result["short_profit"] = short_profit

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
        logger.info(
            f"Applied regime filter: {(result['signal'] != 0).sum()} active signals remaining"
        )

    # Calculate position sizes
    position_sizes = calculate_position_sizes(
        result,
        atr_window,
        config["sizing"]["target_vol_per_leg"],
        inverse_fx_for_quote_ccy_strength,
    )
    result = pd.concat([result, position_sizes], axis=1)

    # Add trade entry/exit flags
    result["entry"] = result["signal"].diff().abs() == 1
    result["exit"] = (result["signal"].diff() != 0) & (result["signal"] == 0)

    logger.info(
        f"Generated {result['entry'].sum()} entry signals and {result['exit'].sum()} exit signals"
    )

    return result


def calculate_position_sizes(
    signals_df: pd.DataFrame,
    atr_window: int,
    target_vol_per_leg: float,
    inverse_fx_for_quote_ccy_strength: bool,
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

    # Calculate ATR using atr_proxy since we only have close prices
    fx_atr = atr_proxy(signals_df["fx_price"], atr_window)
    comd_atr = atr_proxy(signals_df["comd_price"], atr_window)

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
    result["comd_position"] = (
        -result["comd_size"] * signals_df["signal"]
    )  # Opposite side of spread

    return result


def apply_time_stop(signals_df: pd.DataFrame, max_days: int) -> pd.DataFrame:
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

    # Derive a dynamic time-stop from OU half-life when available:
    # effective_max_days = min(max_days, 3 * OU_half_life) bounded to [2, max_days]
    effective_max_days = int(max_days)
    if "ou_half_life" in result.columns:
        try:
            hl = float(result["ou_half_life"].iloc[0])
            if hl > 0 and np.isfinite(hl):
                effective_max_days = max(2, min(int(round(hl * 3.0)), int(max_days)))
        except Exception:
            pass
    logger.info(f"Time-stop days used: {effective_max_days}")

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

            if days_held >= effective_max_days:
                result.loc[idx, "signal"] = 0
                result.loc[idx, "time_stop_exit"] = True
                position_entry_date = None
                current_position = 0

    # Recompute entry/exit flags after applying time stop to signals
    result["entry"] = result["signal"].diff().abs() == 1
    result["exit"] = (result["signal"].diff() != 0) & (result["signal"] == 0)

    # Maintain existing sizes; avoid recomputing size with invalid params
    if "fx_size" in result.columns and "comd_size" in result.columns:
        result["fx_position"] = result["fx_size"] * result["signal"]
        result["comd_position"] = -result["comd_size"] * result["signal"]
    else:
        logger.warning(
            "fx_size/comd_size not found; recomputing sizes with defaults "
            "(atr_window=20, target_vol_per_leg=0.01)"
        )
        position_sizes = calculate_position_sizes(
            result,
            20,           # Default ATR window
            0.01,         # Default target vol per leg (1%)
            True,         # Default inverse flag
        )
        result["fx_position"] = position_sizes["fx_position"]
        result["comd_position"] = position_sizes["comd_position"]

    return result


def generate_signals_with_regime_filter(
    fx_series: pd.Series, comd_series: pd.Series, config: Dict
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

    # Instantiate the ML Trade Filter
    ml_filter_config = config.get('ml_filter', {})
    trade_filter = TradeFilter(
        model_path=ml_filter_config.get('model_path', ''),
        threshold=ml_filter_config.get('threshold', 0.5),
        enabled=ml_filter_config.get('enabled', False)
    )

    # Generate signals with regime filter
    signals_df = generate_signals(fx_series, comd_series, config, regime_filter)

    # Apply time stop if specified
    if "time_stop" in config:
        max_days = config["time_stop"]["max_days"]
        signals_df = apply_time_stop(signals_df, max_days)

    return signals_df


# def _generate_signals_with_model(
#     result: pd.DataFrame,
#     fx_series: pd.Series,
#     comd_series: pd.Series,
#     config: Dict,
#     model_name: str,
# ) -> pd.DataFrame:
#     """
#     Generate signals using a specific model from the ensemble (disabled due to circular import).
#
#     Args:
#         result: Result DataFrame to populate.
#         fx_series: FX time series.
#         comd_series: Commodity time series.
#         config: Configuration dictionary.
#         model_name: Name of the model to use.
#
#     Returns:
#         Updated result DataFrame.
#     """
#     try:
#         # Create ensemble model
#         model_config = ModelConfig()
#         ensemble = create_ensemble_model(model_config)
#
#         # Prepare features for the model
#         features = _prepare_features_for_model(fx_series, comd_series, config)
#
#         # For now, we'll use a simple approach to demonstrate model integration
#         # In practice, this would involve training the model and using it for predictions
#
#         # Compute spread using existing method as fallback
#         beta_window = config["lookbacks"]["beta_window"]
#         use_kalman = config.get("use_kalman", True)
#         spread, alpha, beta = compute_spread(
#             fx_series, comd_series, beta_window, use_kalman
#         )
#         result["spread"] = spread
#         result["alpha"] = alpha
#         result["beta"] = beta
#
#         # Compute robust z-score
#         z_window = config["lookbacks"]["z_window"]
#         z = zscore_robust(spread, z_window).rename("z")
#         result["spread_z"] = z
#
#         logger.info(f"Generated signals using {model_name} model")
#     except Exception as e:
#         logger.warning(
#             f"Failed to generate signals with {model_name} model, using default: {e}"
#         )
#         # Fallback to default method
#         beta_window = config["lookbacks"]["beta_window"]
#         use_kalman = config.get("use_kalman", True)
#         spread, alpha, beta = compute_spread(
#             fx_series, comd_series, beta_window, use_kalman
#         )
#         result["spread"] = spread
#         result["alpha"] = alpha
#         result["beta"] = beta
#
#         # Compute robust z-score
#         z_window = config["lookbacks"]["z_window"]
#         z = zscore_robust(spread, z_window).rename("z")
#         result["spread_z"] = z
#
#     return result


# def _prepare_features_for_model(
#     fx_series: pd.Series, comd_series: pd.Series, config: Dict
# ) -> pd.DataFrame:
#     """
#     Prepare features for model training/prediction (disabled due to circular import).
#
#     Args:
#         fx_series: FX time series.
#         comd_series: Commodity time series.
#         config: Configuration dictionary.
#
#     Returns:
#         DataFrame with prepared features.
#     """
#     # Create features DataFrame
#     features = pd.DataFrame(index=fx_series.index)
#     features["fx_price"] = fx_series
#     features["comd_price"] = comd_series
#
#     # Add returns
#     features["fx_returns"] = fx_series.pct_change()
#     features["comd_returns"] = comd_series.pct_change()
#
#     # Add rolling statistics
#     lookback_windows = [5, 10, 20, 60]
#     for window in lookback_windows:
#         features[f"fx_vol_{window}"] = features["fx_returns"].rolling(window).std()
#         features[f"comd_vol_{window}"] = features["comd_returns"].rolling(window).std()
#         features[f"fx_ma_{window}"] = fx_series.rolling(window).mean()
#         features[f"comd_ma_{window}"] = comd_series.rolling(window).mean()
#
#     # Add spread-related features
#     beta_window = config["lookbacks"]["beta_window"]
#     use_kalman = config.get("use_kalman", True)
#     spread, alpha, beta = compute_spread(
#         fx_series, comd_series, beta_window, use_kalman
#     )
#     features["spread"] = spread
#     features["alpha"] = alpha
#     features["beta"] = beta
#
#     # Add z-score features
#     z_window = config["lookbacks"]["z_window"]
#     z = zscore_robust(spread, z_window)
#     features["spread_z"] = z
#
#     # Drop rows with NaN values
#     features = features.dropna()
#
#     return features


# def generate_signals_with_ensemble(
#     fx_series: pd.Series,
#     comd_series: pd.Series,
#     config: Dict,
#     regime_filter: Optional[pd.Series] = None,
# ) -> pd.DataFrame:
#     """
#     Generate signals using ensemble model predictions.
#
#     Args:
#         fx_series: FX time series.
#         comd_series: Commodity time series.
#         config: Configuration dictionary.
#         regime_filter: Optional boolean series for regime filtering.
#
#     Returns:
#         DataFrame with ensemble signals and related metrics.
#     """
#     logger.info("Generating signals with ensemble model")
#
#     # Create ensemble model
#     model_config = ModelConfig()
#     ensemble = create_ensemble_model(model_config)
#
#     # Prepare features
#     features = _prepare_features_for_model(fx_series, comd_series, config)
#
#     # Generate ensemble prediction
#     ensemble_pred = ensemble.predict_ensemble(features)
#
#     # Create result DataFrame with ensemble predictions
#     result = pd.DataFrame(index=fx_series.index)
#     result["fx_price"] = fx_series
#     result["comd_price"] = comd_series
#
#     # For now, we'll use the existing signal generation logic
#     # but in practice, we would use the ensemble predictions
#
#     return generate_signals(fx_series, comd_series, config, regime_filter)


def calculate_d1_position_sizes(
    signals_df: pd.DataFrame,
    atr_window: int,
    target_vol_per_leg: float,
    inverse_fx_for_quote_ccy_strength: bool,
) -> pd.DataFrame:
    """
    Calculate D1 position sizes based on inverse volatility sizing.

    Args:
        signals_df: DataFrame with D1 signals and prices
        atr_window: Window for ATR calculation (longer for D1)
        target_vol_per_leg: Target volatility per leg (larger for D1)
        inverse_fx_for_quote_ccy_strength: Whether to inverse FX for quote currency strength

    Returns:
        DataFrame with D1 position sizes
    """
    logger.debug("Calculating D1 position sizes")

    # Calculate ATR using atr_proxy (D1-optimized window)
    fx_atr = atr_proxy(signals_df["fx_price"], atr_window)
    comd_atr = atr_proxy(signals_df["comd_price"], atr_window)

    # Calculate position sizes (inverse volatility, larger for D1)
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
    result["comd_position"] = (
        -result["comd_size"] * signals_df["signal"]
    )  # Opposite side of spread

    return result


def _generate_d1_signals_with_model(
    result: pd.DataFrame,
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate D1 signals using a specific model from the ensemble.

    Args:
        result: Result DataFrame to populate
        fx_series: D1 FX time series
        comd_series: D1 Commodity time series
        config: Configuration dictionary
        model_name: Name of the model to use

    Returns:
        Updated result DataFrame
    """
    try:
        # Create ensemble model
        model_config = ModelConfig()
        ensemble = create_ensemble_model(model_config)

        # Prepare D1 features for the model
        features = _prepare_d1_features_for_model(fx_series, comd_series, config)

        # For now, we'll use a simple approach to demonstrate model integration
        # In practice, this would involve training the model and using it for predictions

        # Compute spread using existing method as fallback
        beta_window = config["d1"]["lookbacks"]["beta_window"]
        use_kalman = config["d1"].get("use_kalman", True)
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman
        )
        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        # Compute robust z-score
        z_window = config["d1"]["lookbacks"]["z_window"]
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

        logger.info(f"Generated D1 signals using {model_name} model")
    except Exception as e:
        logger.warning(
            f"Failed to generate D1 signals with {model_name} model, using default: {e}"
        )
        # Fallback to default method
        beta_window = config["d1"]["lookbacks"]["beta_window"]
        use_kalman = config["d1"].get("use_kalman", True)
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman
        )
        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        # Compute robust z-score
        z_window = config["d1"]["lookbacks"]["z_window"]
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

    return result


def _prepare_d1_features_for_model(
    fx_series: pd.Series, comd_series: pd.Series, config: Dict
) -> pd.DataFrame:
    """
    Prepare D1 features for model training/prediction.

    Args:
        fx_series: D1 FX time series
        comd_series: D1 Commodity time series
        config: Configuration dictionary

    Returns:
        DataFrame with prepared D1 features
    """
    # Create features DataFrame
    features = pd.DataFrame(index=fx_series.index)
    features["fx_price"] = fx_series
    features["comd_price"] = comd_series

    # Add returns (D1)
    features["fx_returns"] = fx_series.pct_change()
    features["comd_returns"] = comd_series.pct_change()

    # Add rolling statistics (D1-optimized windows)
    d1_lookback_windows = [20, 60, 120, 252]  # 1M, 3M, 6M, 1Y for D1
    for window in d1_lookback_windows:
        features[f"fx_vol_{window}"] = features["fx_returns"].rolling(window).std()
        features[f"comd_vol_{window}"] = features["comd_returns"].rolling(window).std()
        features[f"fx_ma_{window}"] = fx_series.rolling(window).mean()
        features[f"comd_ma_{window}"] = comd_series.rolling(window).mean()

    # Add spread-related features
    beta_window = config["d1"]["lookbacks"]["beta_window"]
    use_kalman = config["d1"].get("use_kalman", True)
    spread, alpha, beta = compute_spread(
        fx_series, comd_series, beta_window, use_kalman
    )
    features["spread"] = spread
    features["alpha"] = alpha
    features["beta"] = beta

    # Add z-score features
    z_window = config["d1"]["lookbacks"]["z_window"]
    z = zscore_robust(spread, z_window)
    features["spread_z"] = z

    # Drop rows with NaN values
    features = features.dropna()

    return features
 Drop rows with NaN values
    features = features.dropna()

    return features
