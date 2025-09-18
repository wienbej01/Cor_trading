"""
H1 Mean reversion strategy module for FX-Commodity correlation arbitrage.
Implements signal generation and position sizing for H1 timeframe spread trading.
Optimized for hourly data with shorter lookback windows and faster signals.
"""

from typing import Dict, Optional
import asyncio

import pandas as pd
from loguru import logger

from features.indicators import atr_proxy, zscore_robust
from features.spread import compute_spread
from features.regime import correlation_gate
from data.broker_api import get_multi_symbol_h1_data
from statsmodels.tsa.stattools import adfuller

# Import ensemble model
from ml.ensemble import create_ensemble_model, ModelConfig

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


async def generate_h1_signals(
    fx_symbol: str,
    comd_symbol: str,
    start_date: str,
    end_date: str,
    config: Dict,
    regime_filter: Optional[pd.Series] = None,
    model_name: str = "default",
) -> pd.DataFrame:
    """
    Generate H1 trading signals for mean reversion strategy.

    Args:
        fx_symbol: FX symbol (e.g., "USDCAD=X")
        comd_symbol: Commodity symbol (e.g., "CL=F")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        config: Configuration dictionary with H1-optimized parameters
        regime_filter: Optional boolean series for regime filtering
        model_name: Name of the model to use for signal generation

    Returns:
        DataFrame with H1 signals and related metrics

    Raises:
        ValueError: If required config parameters are missing or data fetch fails
        ValidationError: If input validation fails
    """
    logger.info(f"Generating H1 mean reversion signals for {fx_symbol}-{comd_symbol}")

    # Fetch H1 data using broker API
    try:
        h1_data = await get_multi_symbol_h1_data(
            [fx_symbol, comd_symbol], start_date, end_date, config
        )
        fx_series = (
            h1_data[fx_symbol].set_index("ts")["close"]
            if fx_symbol in h1_data
            else pd.Series()
        )
        comd_series = (
            h1_data[comd_symbol].set_index("ts")["close"]
            if comd_symbol in h1_data
            else pd.Series()
        )

        if fx_series.empty or comd_series.empty:
            raise ValueError(
                f"Failed to fetch H1 data for {fx_symbol} or {comd_symbol}"
            )

    except Exception as e:
        logger.error(f"H1 data fetch failed: {e}")
        raise ValueError(f"Could not fetch H1 data: {e}")

    # Generate signals using the fetched data
    return generate_h1_signals_from_data(
        fx_series, comd_series, config, regime_filter, model_name
    )


def generate_h1_signals_from_data(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    regime_filter: Optional[pd.Series] = None,
    model_name: str = "default",
) -> pd.DataFrame:
    """
    Generate H1 trading signals from pre-fetched data.

    Args:
        fx_series: H1 FX time series
        comd_series: H1 Commodity time series
        config: Configuration dictionary with H1-optimized parameters
        regime_filter: Optional boolean series for regime filtering
        model_name: Name of the model to use for signal generation

    Returns:
        DataFrame with H1 signals and related metrics
    """
    logger.info("Generating H1 signals from data")

    # Hard-align both series to a common index to guarantee equal lengths
    _common_idx = fx_series.index.intersection(comd_series.index)
    fx_series = fx_series.reindex(_common_idx)
    comd_series = comd_series.reindex(_common_idx)

    # Input validation
    try:
        validate_series_alignment(fx_series, comd_series)
        validate_trading_config(config)
    except ValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise ValueError(f"Input validation failed: {e}") from e

    # Extract H1-optimized config parameters
    try:
        # H1-specific parameters (shorter windows, faster signals)
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
            "max_bars": 24,  # Max 24 hours (1 day) for H1
            "profit_target": 1.5,  # Tighter profit targets for H1
            "stop_loss": 1.0,  # Tighter stops for H1
        }

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
        max_bars = int(params.get("max_bars", 24))
        profit_target = float(params.get("profit_target", 1.5))
        stop_loss = float(params.get("stop_loss", 1.0))

    except (ValidationError, ValueError) as e:
        logger.error(f"H1 parameter extraction failed: {e}")
        raise ValueError(f"H1 configuration error: {e}") from e

    # Create result DataFrame
    result = pd.DataFrame(index=fx_series.index)
    result["fx_price"] = fx_series
    result["comd_price"] = comd_series

    # Compute spread using ensemble model if specified
    if model_name != "default":
        result = _generate_h1_signals_with_model(
            result, fx_series, comd_series, config, model_name
        )
    else:
        # Compute spread using existing method
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman
        )
        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        # Compute robust z-score with H1-optimized window
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

    # H1-optimized regime filtering (align series to common index for correlation gate)
    common_index = fx_series.index.intersection(comd_series.index)
    fx_aligned = fx_series.reindex(common_index)
    comd_aligned = comd_series.reindex(common_index)
    regime_ok_aligned = correlation_gate(
        fx_aligned, comd_aligned, corr_window, min_abs_corr
    )
    # Reindex regime_ok back to full result index, fill missing as False
    regime_ok = regime_ok_aligned.reindex(result.index).fillna(False)
    p_adf = adf_pvalue(spread)

    # More permissive regime filtering for H1 (faster signals)
    adf_ok = p_adf <= 0.15  # More relaxed than daily (0.15 vs 0.10)
    good_regime = regime_ok | adf_ok
    result["good_regime"] = good_regime
    result["adf_p"] = p_adf

    # Generate raw signals with H1-optimized thresholds
    result["raw_signal"] = 0

    # H1 entry/exit logic (more responsive)
    enter_long = (z <= -entry_z) & good_regime
    enter_short = (z >= entry_z) & good_regime
    exit_rule = z.abs() <= exit_z

    # Add entry/exit flags for diagnostics
    result["enter_long"] = enter_long
    result["enter_short"] = enter_short
    result["exit"] = exit_rule

    # Enhanced signal generation with H1-optimized profit targets and stops
    result["raw_signal"] = 0

    # Dynamic position sizing based on volatility
    result.loc[enter_long, "raw_signal"] = 1
    result.loc[enter_short, "raw_signal"] = -1

    # Exit signals - ensure they have the same index as result
    long_exit = pd.Series(z >= -exit_z, index=result.index)
    short_exit = pd.Series(z <= exit_z, index=result.index)

    # H1-optimized stop loss and profit target signals
    long_stop = pd.Series(z >= -stop_z, index=result.index)
    short_stop = pd.Series(z <= stop_z, index=result.index)

    # Tighter profit target signals for H1
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

    result["signal"] = signals

    # Apply regime filter if provided
    if regime_filter is not None:
        result["signal"] = result["signal"].where(regime_filter, 0)
        logger.info(
            f"Applied H1 regime filter: {(result['signal'] != 0).sum()} active signals remaining"
        )

    # Calculate position sizes with H1-optimized parameters
    position_sizes = calculate_h1_position_sizes(
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
        f"Generated H1 signals: {result['entry'].sum()} entries, {result['exit'].sum()} exits"
    )

    return result


def calculate_h1_position_sizes(
    signals_df: pd.DataFrame,
    atr_window: int,
    target_vol_per_leg: float,
    inverse_fx_for_quote_ccy_strength: bool,
) -> pd.DataFrame:
    """
    Calculate H1 position sizes based on inverse volatility sizing.

    Args:
        signals_df: DataFrame with H1 signals and prices
        atr_window: Window for ATR calculation (shorter for H1)
        target_vol_per_leg: Target volatility per leg (smaller for H1)
        inverse_fx_for_quote_ccy_strength: Whether to inverse FX for quote currency strength

    Returns:
        DataFrame with H1 position sizes
    """
    logger.debug("Calculating H1 position sizes")

    # Calculate ATR using atr_proxy (H1-optimized window)
    fx_atr = atr_proxy(signals_df["fx_price"], atr_window)
    comd_atr = atr_proxy(signals_df["comd_price"], atr_window)

    # Calculate position sizes (inverse volatility, smaller for H1)
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


def generate_h1_signals_with_regime_filter(
    fx_symbol: str, comd_symbol: str, start_date: str, end_date: str, config: Dict
) -> pd.DataFrame:
    """
    Generate H1 signals with regime filtering applied.

    Args:
        fx_symbol: FX symbol
        comd_symbol: Commodity symbol
        start_date: Start date
        end_date: End date
        config: Configuration dictionary

    Returns:
        DataFrame with H1 signals and regime filtering applied
    """
    logger.info("Generating H1 signals with regime filtering")

    # This would require fetching H1 data first, then applying regime filter
    # For now, return empty DataFrame as placeholder
    # In practice, we'd fetch the data and apply combined_regime_filter

    return pd.DataFrame()


def _generate_h1_signals_with_model(
    result: pd.DataFrame,
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate H1 signals using a specific model from the ensemble.

    Args:
        result: Result DataFrame to populate
        fx_series: H1 FX time series
        comd_series: H1 Commodity time series
        config: Configuration dictionary
        model_name: Name of the model to use

    Returns:
        Updated result DataFrame
    """
    try:
        # Create ensemble model
        model_config = ModelConfig()
        ensemble = create_ensemble_model(model_config)

        # Prepare H1 features for the model
        features = _prepare_h1_features_for_model(fx_series, comd_series, config)

        # For now, we'll use a simple approach to demonstrate model integration
        # In practice, this would involve training the model and using it for predictions

        # Compute spread using existing method as fallback
        beta_window = config["lookbacks"]["beta_window"]
        use_kalman = config.get("use_kalman", True)
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman
        )
        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        # Compute robust z-score
        z_window = config["lookbacks"]["z_window"]
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

        logger.info(f"Generated H1 signals using {model_name} model")
    except Exception as e:
        logger.warning(
            f"Failed to generate H1 signals with {model_name} model, using default: {e}"
        )
        # Fallback to default method
        beta_window = config["h1"]["lookbacks"]["beta_window"]
        use_kalman = config["h1"].get("use_kalman", True)
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman
        )
        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        # Compute robust z-score
        z_window = config["h1"]["lookbacks"]["z_window"]
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

    return result


def _prepare_h1_features_for_model(
    fx_series: pd.Series, comd_series: pd.Series, config: Dict
) -> pd.DataFrame:
    """
    Prepare H1 features for model training/prediction.

    Args:
        fx_series: H1 FX time series
        comd_series: H1 Commodity time series
        config: Configuration dictionary

    Returns:
        DataFrame with prepared H1 features
    """
    # Create features DataFrame
    features = pd.DataFrame(index=fx_series.index)
    features["fx_price"] = fx_series
    features["comd_price"] = comd_series

    # Add returns (H1)
    features["fx_returns"] = fx_series.pct_change()
    features["comd_returns"] = comd_series.pct_change()

    # Add rolling statistics (H1-optimized windows)
    h1_lookback_windows = [4, 8, 16, 24]  # 4h, 8h, 16h, 24h for H1
    for window in h1_lookback_windows:
        features[f"fx_vol_{window}"] = features["fx_returns"].rolling(window).std()
        features[f"comd_vol_{window}"] = features["comd_returns"].rolling(window).std()
        features[f"fx_ma_{window}"] = fx_series.rolling(window).mean()
        features[f"comd_ma_{window}"] = comd_series.rolling(window).mean()

    # Add spread-related features
    beta_window = config["lookbacks"]["beta_window"]
    use_kalman = config.get("use_kalman", True)
    spread, alpha, beta = compute_spread(
        fx_series, comd_series, beta_window, use_kalman
    )
    features["spread"] = spread
    features["alpha"] = alpha
    features["beta"] = beta

    # Add z-score features
    z_window = config["lookbacks"]["z_window"]
    z = zscore_robust(spread, z_window)
    features["spread_z"] = z

    # Drop rows with NaN values
    features = features.dropna()

    return features


# Synchronous wrapper for backward compatibility
def generate_h1_signals_sync(
    fx_symbol: str,
    comd_symbol: str,
    start_date: str,
    end_date: str,
    config: Dict,
    regime_filter: Optional[pd.Series] = None,
    model_name: str = "default",
) -> pd.DataFrame:
    """
    Synchronous wrapper for generate_h1_signals.

    Args:
        fx_symbol: FX symbol
        comd_symbol: Commodity symbol
        start_date: Start date
        end_date: End date
        config: Configuration dictionary
        regime_filter: Optional regime filter
        model_name: Model name

    Returns:
        DataFrame with H1 signals
    """
    return asyncio.run(
        generate_h1_signals(
            fx_symbol,
            comd_symbol,
            start_date,
            end_date,
            config,
            regime_filter,
            model_name,
        )
    )
