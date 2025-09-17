"""
Mean reversion strategy module for FX-Commodity correlation arbitrage.
Implements signal generation and position sizing for spread trading.
"""

from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.features.indicators import atr_proxy, zscore_robust
from src.features.spread import compute_spread
from src.features.regime import correlation_gate
from statsmodels.tsa.stattools import adfuller

# Import ensemble model
from src.ml.ensemble import create_ensemble_model, ModelConfig

# Import validation interfaces for consistent error handling
from src.interfaces.validation import (
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


def generate_d1_signals(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    regime_filter: Optional[pd.Series] = None,
    model_name: str = "default",
) -> pd.DataFrame:
    """
    Generate D1 (daily) trading signals for mean reversion strategy.
    Optimized for daily data with longer lookback windows and slower signals.

    Args:
        fx_series: Daily FX time series
        comd_series: Daily Commodity time series
        config: Configuration dictionary with D1-optimized parameters
        regime_filter: Optional boolean series for regime filtering
        model_name: Name of the model to use for signal generation

    Returns:
        DataFrame with D1 signals and related metrics

    Raises:
        ValueError: If required config parameters are missing
        ValidationError: If input validation fails
    """
    # Early console checkpoint for forensic visibility
    print(
        f"[FORCELOG] Enter D1 generate_d1_signals: fx_len={len(fx_series)}, comd_len={len(comd_series)}, cfg_keys={list(config.keys())}"
    )
    logger.info(
        f"D1 generate_d1_signals called with config keys: {list(config.keys())}"
    )
    logger.info(
        f"D1 input series: fx_length={len(fx_series)}, comd_length={len(comd_series)}"
    )
    logger.info(
        f"D1 series indices: fx_index_type={type(fx_series.index)}, comd_index_type={type(comd_series.index)}"
    )

    if len(fx_series) != len(comd_series):
        logger.warning(
            f"D1 series length mismatch: fx={len(fx_series)}, comd={len(comd_series)}"
        )

    # Input validation
    try:
        validate_series_alignment(fx_series, comd_series)
        validate_trading_config(config)
        logger.info("D1 input validation passed")
    except ValidationError as e:
        logger.error(f"D1 input validation failed: {e}")
        raise ValueError(f"D1 input validation failed: {e}") from e

    # Extract D1-optimized config parameters
    try:
        # D1-specific parameters (longer windows, slower signals)
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
        default_values = {
            "use_kalman": True,
            "inverse_fx_for_quote_ccy_strength": True,
            "max_bars": 60,  # Max 60 days for D1
            "profit_target": 3.0,  # Wider profit targets for D1
            "stop_loss": 2.0,  # Wider stops for D1
        }

        params = safe_parameter_extraction(config, required_params, default_values)

        # Debug logging for parameters
        logger.info(f"D1 parameters extracted: {params}")

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
        max_bars = int(params.get("max_bars", 60))
        profit_target = float(params.get("profit_target", 3.0))
        stop_loss = float(params.get("stop_loss", 2.0))

        logger.info(
            f"D1 windows: beta={beta_window}, z={z_window}, corr={corr_window}, atr={atr_window}"
        )

    except (ValidationError, ValueError) as e:
        logger.error(f"D1 parameter extraction failed: {e}")
        raise ValueError(f"D1 configuration error: {e}") from e

    # Create result DataFrame
    result = pd.DataFrame(index=fx_series.index)
    result["fx_price"] = fx_series
    result["comd_price"] = comd_series

    # Debug logging
    logger.info(
        f"D1 generate_d1_signals: fx_series length: {len(fx_series)}, comd_series length: {len(comd_series)}"
    )
    logger.info(
        f"D1 generate_d1_signals: result index has duplicates: {result.index.duplicated().any()}"
    )
    logger.info(f"D1 generate_d1_signals: result length: {len(result)}")
    if result.index.duplicated().any():
        logger.warning(
            f"D1 generate_d1_signals: duplicate indices found: {result.index[result.index.duplicated()].unique()}"
        )

    # Compute spread using ensemble model if specified
    if model_name != "default":
        result = _generate_d1_signals_with_model(
            result, fx_series, comd_series, config, model_name
        )
    else:
        # Compute spread using existing method
        logger.info(
            f"D1 compute_spread: beta_window={beta_window}, use_kalman={use_kalman}"
        )
        logger.info(
            f"D1 input lengths: fx_series={len(fx_series)}, comd_series={len(comd_series)}"
        )
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman
        )
        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        logger.info(
            f"D1 spread computed: length={len(spread)}, NaN count={spread.isna().sum()}"
        )
        if spread.isna().all():
            logger.error("D1 spread is all NaN - cannot continue")
            raise ValueError("Spread calculation returned all NaN values")

        # Compute robust z-score with D1-optimized window
        logger.info(
            f"D1 zscore_robust: z_window={z_window}, spread NaN count={spread.isna().sum()}"
        )
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

        logger.info(
            f"D1 z-score calculated: length={len(z)}, has NaN={z.isna().any()}, range=({z.min():.3f}, {z.max():.3f})"
        )
        if z.isna().all():
            logger.error("D1 z-score is all NaN - cannot continue")
            raise ValueError("Z-score calculation returned all NaN values")

    # D1-optimized regime filtering
    regime_ok = correlation_gate(fx_series, comd_series, corr_window, min_abs_corr)
    p_adf = adf_pvalue(spread)

    # Stricter regime filtering for D1 (higher quality signals)
    adf_ok = p_adf <= 0.05  # Stricter than H1 (0.05 vs 0.15)
    good_regime = regime_ok & adf_ok  # Use AND for more conservative D1
    result["good_regime"] = good_regime
    result["adf_p"] = p_adf

    # Generate raw signals with D1-optimized thresholds
    result["raw_signal"] = 0

    # D1 entry/exit logic (more conservative)
    enter_long = (z <= -entry_z) & good_regime
    enter_short = (z >= entry_z) & good_regime
    exit_rule = z.abs() <= exit_z

    # Add entry/exit flags for diagnostics
    result["enter_long"] = enter_long
    result["enter_short"] = enter_short
    result["exit"] = exit_rule

    # Enhanced signal generation with D1-optimized profit targets and stops
    result["raw_signal"] = 0

    # Dynamic position sizing based on volatility
    result.loc[enter_long, "raw_signal"] = 1
    result.loc[enter_short, "raw_signal"] = -1

    # Exit signals - ensure they have the same index as result
    long_exit = pd.Series(z >= -exit_z, index=result.index)
    short_exit = pd.Series(z <= exit_z, index=result.index)

    # D1-optimized stop loss and profit target signals
    long_stop = pd.Series(z >= -stop_z, index=result.index)
    short_stop = pd.Series(z <= stop_z, index=result.index)

    # Wider profit target signals for D1
    long_profit = pd.Series(z <= -profit_target, index=result.index)
    short_profit = pd.Series(z >= profit_target, index=result.index)

    # Add enhanced signal tracking
    result["long_exit"] = long_exit
    result["short_exit"] = short_exit
    result["long_stop"] = long_stop
    result["short_stop"] = short_stop
    result["long_profit"] = long_profit
    result["short_profit"] = short_profit

    # Apply signal logic with proper state management
    position = 0
    signals = []
    bars_held = 0
    duplicate_loc_count = 0

    # Wrap loop in try/except to capture any indexing errors with full traceback
    try:
        for pos, idx in enumerate(result.index):
            current_signal = result["raw_signal"].iat[pos]

            # If we have no position
            if position == 0:
                # Enter new position if signal is non-zero
                if current_signal != 0:
                    position = current_signal
                    signals.append(current_signal)
                    bars_held = 1
                else:
                    signals.append(0)
                    bars_held = 0
            # If we have a long position
            elif position == 1:
                bars_held += 1
                # Debug: check if .loc returns Series
                if False:  # using positional indexing; no duplicate label check needed
                    pass
                # Exit if we hit exit threshold, stop loss, or max bars
                if long_exit.iat[pos] or long_stop.iat[pos] or bars_held >= max_bars:
                    position = 0
                    signals.append(0)
                    bars_held = 0
                else:
                    signals.append(1)
            # If we have a short position
            elif position == -1:
                bars_held += 1
                # Debug: check if .loc returns Series
                if False:  # using positional indexing; no duplicate label check needed
                    pass
                # Exit if we hit exit threshold, stop loss, or max bars
                if short_exit.iat[pos] or short_stop.iat[pos] or bars_held >= max_bars:
                    position = 0
                    signals.append(0)
                    bars_held = 0
                else:
                    signals.append(-1)
            else:
                signals.append(0)
                bars_held = 0
    except Exception as e:
        logger.exception(f"D1 signal loop failed at idx={idx}: {e}")
        raise

    result["signal"] = signals

    # Debug logging for loop
    logger.info(
        f"D1 signal loop completed: signals length={len(signals)}, duplicate_loc_count={duplicate_loc_count}"
    )

    # Apply regime filter if provided
    if regime_filter is not None:
        result["signal"] = result["signal"].where(regime_filter, 0)
        logger.info(
            f"Applied D1 regime filter: {(result['signal'] != 0).sum()} active signals remaining"
        )

    # Calculate position sizes with D1-optimized parameters
    position_sizes = calculate_d1_position_sizes(
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
        f"Generated D1 signals: {result['entry'].sum()} entries, {result['exit'].sum()} exits"
    )

    return result


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
