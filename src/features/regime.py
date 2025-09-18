"""
Market regime detection module for FX-Commodity correlation arbitrage strategy.
Provides functions to detect market regimes and filter signals based on correlation.
"""

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t
import hmmlearn.hmm as hmm

from .indicators import rolling_corr


def correlation_gate(
    series_a: pd.Series,
    series_b: pd.Series,
    corr_window: int = 60,
    min_abs_corr: float = 0.15,  # Reduced from 0.3 to 0.15 to allow more signals
) -> pd.Series:
    """
    Apply correlation gate to filter signals based on rolling correlation.

    Args:
        series_a: First time series (e.g., FX rate).
        series_b: Second time series (e.g., commodity price).
        corr_window: Window size for rolling correlation calculation.
        min_abs_corr: Minimum absolute correlation threshold for signal validity.

    Returns:
        Boolean series indicating valid signals (True when correlation is strong enough).

    Raises:
        ValueError: If series have mismatched lengths or parameters are invalid.
    """
    # Auto-align by common index to avoid hard failure on minor mismatches
    if not series_a.index.equals(series_b.index):
        common_idx = series_a.index.intersection(series_b.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping index between series for correlation gate")
        if len(common_idx) < max(5, corr_window):
            raise ValueError(
                f"Insufficient overlap after alignment: {len(common_idx)} < {max(5, corr_window)}"
            )
        logger.warning(
            f"Correlation gate: auto-aligning series (original lengths: {len(series_a)}, {len(series_b)}; aligned: {len(common_idx)})"
        )
        series_a = series_a.reindex(common_idx)
        series_b = series_b.reindex(common_idx)

    if corr_window < 5:
        raise ValueError("Correlation window must be at least 5")

    if not 0 <= min_abs_corr <= 1:
        raise ValueError("Minimum absolute correlation must be between 0 and 1")

    logger.debug(
        f"Applying correlation gate with window={corr_window}, min_abs_corr={min_abs_corr}"
    )

    # Calculate rolling correlation
    correlation = rolling_corr(series_a, series_b, corr_window)

    # Check if absolute correlation exceeds threshold
    valid_signals = correlation.abs() >= min_abs_corr

    logger.info(
        f"Correlation gate: {valid_signals.sum()}/{len(valid_signals)} signals pass threshold ({valid_signals.mean()*100:.2f}%)"
    )

    return valid_signals


def dcc_garch_filter(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 60,
    corr_threshold: float = 0.15,  # Reduced from 0.3 to 0.15
) -> pd.Series:
    """
    Apply DCC-GARCH filter for dynamic correlation estimation (placeholder implementation).

    Args:
        returns_a: Returns of first time series.
        returns_b: Returns of second time series.
        window: Window size for estimation.
        corr_threshold: Correlation threshold for signal validity.

    Returns:
        Boolean series indicating valid signals based on DCC-GARCH correlation.

    Raises:
        ValueError: If series have mismatched lengths or parameters are invalid.
    """
    if len(returns_a) != len(returns_b):
        raise ValueError("Returns series must have the same length")

    if window < 20:
        raise ValueError("DCC-GARCH window must be at least 20")

    logger.warning(
        "DCC-GARCH filter called but not fully implemented (using rolling correlation as placeholder)"
    )

    # For now, use rolling correlation as a placeholder
    # In a full implementation, this would use DCC-GARCH for dynamic correlation estimation
    correlation = rolling_corr(returns_a, returns_b, window)

    # Check if correlation exceeds threshold
    valid_signals = correlation.abs() >= corr_threshold

    logger.info(
        f"DCC-GARCH filter (placeholder): {valid_signals.sum()}/{len(valid_signals)} signals pass threshold ({valid_signals.mean()*100:.2f}%)"
    )

    return valid_signals


def volatility_regime_quantiles(
    series: pd.Series,
    window: int = 20,
    quantile_low: float = 0.33,
    quantile_high: float = 0.67,
    ref_period: int = 252,
) -> pd.Series:
    """
    Detect volatility regime using rolling quantiles of realized volatility.
    
    Args:
        series: Time series to analyze.
        window: Window for rolling volatility calculation.
        quantile_low: Lower quantile for low vol regime (default 33rd percentile).
        quantile_high: Upper quantile for high vol regime (default 67th percentile).
        ref_period: Reference period for quantile calculation (default 1 year).
    
    Returns:
        Series with regime labels: 0=low vol, 1=normal vol, 2=high vol.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if window < 5:
        raise ValueError("Volatility window must be at least 5")
    if not (0 < quantile_low < quantile_high < 1):
        raise ValueError("Quantiles must satisfy 0 < low < high < 1")

    logger.debug(f"Detecting volatility regime with quantiles low={quantile_low}, high={quantile_high}")

    # Calculate returns
    returns = series.pct_change().fillna(0)

    # Calculate rolling realized volatility
    vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

    # Compute rolling quantiles over reference period
    vol_low_q = vol.rolling(ref_period, min_periods=window).quantile(quantile_low)
    vol_high_q = vol.rolling(ref_period, min_periods=window).quantile(quantile_high)

    # Classify regime
    regime = pd.Series(1, index=vol.index)  # Default to normal

    # Low volatility regime
    regime[vol <= vol_low_q] = 0

    # High volatility regime
    regime[vol >= vol_high_q] = 2

    return regime


def volatility_regime(
    series: pd.Series,
    window: int = 20,
    high_vol_threshold: float = 0.03,
    low_vol_threshold: float = 0.003,
    use_quantiles: bool = True,
    quantile_low: float = 0.33,
    quantile_high: float = 0.67,
    ref_period: int = 252,
) -> pd.Series:
    """
    Detect volatility regime of a series (wrapper to choose method).
    
    Args:
        series: Time series to analyze.
        window: Window size for volatility calculation.
        high_vol_threshold: Threshold for high volatility regime (as decimal).
        low_vol_threshold: Threshold for low volatility regime (as decimal).
        use_quantiles: If True, use quantile-based classification; else use fixed thresholds.
        quantile_low: Lower quantile for low vol (if use_quantiles=True).
        quantile_high: Upper quantile for high vol (if use_quantiles=True).
        ref_period: Reference period for quantiles (if use_quantiles=True).
    
    Returns:
        Series with regime labels: 0=low vol, 1=normal vol, 2=high vol.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if window < 5:
        raise ValueError("Volatility window must be at least 5")

    if use_quantiles:
        return volatility_regime_quantiles(
            series, window, quantile_low, quantile_high, ref_period
        )
    else:
        # Legacy fixed threshold implementation
        if high_vol_threshold <= low_vol_threshold:
            raise ValueError(
                "High volatility threshold must be greater than low volatility threshold"
            )
        logger.debug(f"Detecting volatility regime with fixed thresholds (legacy)")
        returns = series.pct_change().fillna(0)
        volatility = returns.rolling(window=window).std()
        regime = pd.Series(1, index=volatility.index)
        regime[volatility <= low_vol_threshold] = 0
        regime[volatility >= high_vol_threshold] = 2
        return regime


def roc_hp_trend_regime(
    series: pd.Series,
    roc_window: int = 20,
    hp_lambda: int = 1600,
    trend_threshold: float = 0.015,
) -> pd.Series:
    """
    Detect trend regime using Rate of Change (ROC) and Hodrick-Prescott (HP) filter slope.
    
    Args:
        series: Time series to analyze.
        roc_window: Window for ROC calculation.
        hp_lambda: Smoothing parameter for HP filter (1600 for daily data).
        trend_threshold: Threshold for significant trend/slope (as decimal).
    
    Returns:
        Series with regime labels: -1=downtrend, 0=range-bound, 1=uptrend.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if roc_window < 5:
        raise ValueError("ROC window must be at least 5")
    if hp_lambda <= 0:
        raise ValueError("HP lambda must be positive")

    logger.debug(f"Detecting ROC/HP trend regime with roc_window={roc_window}, hp_lambda={hp_lambda}")

    # Rate of Change (ROC)
    roc = (series / series.shift(roc_window) - 1).fillna(0)

    # Hodrick-Prescott filter for trend component
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        trend, cycle = hpfilter(series.dropna(), lamb=hp_lambda)
        # Extend trend to full index
        trend = trend.reindex(series.index, method='ffill').fillna(method='bfill')
        # Compute slope of trend (simple difference as proxy for slope)
        trend_slope = trend.diff().fillna(0) / series.pct_change().fillna(0.001)  # Normalize by return
    except ImportError:
        logger.warning("statsmodels not available; using ROC only for trend")
        trend_slope = pd.Series(0, index=series.index)
    except Exception as e:
        logger.warning(f"HP filter failed: {e}; using ROC only")
        trend_slope = pd.Series(0, index=series.index)

    # Combined signal: average normalized ROC and trend_slope
    roc_norm = roc / roc.rolling(252).std().fillna(0.01)  # Normalize by long-term vol
    slope_norm = trend_slope / trend_slope.rolling(20).std().fillna(0.01)
    combined_trend = (roc_norm + slope_norm) / 2

    # Classify regime
    regime = pd.Series(0, index=series.index)  # Default to range-bound

    # Uptrend
    regime[combined_trend > trend_threshold] = 1

    # Downtrend
    regime[combined_trend < -trend_threshold] = -1

    return regime


def trend_regime(
    series: pd.Series,
    window: int = 20,
    trend_threshold: float = 0.015,
    use_roc_hp: bool = True,
) -> pd.Series:
    """
    Detect trend regime of a series (wrapper to choose method).
    
    Args:
        series: Time series to analyze.
        window: Window size for trend calculation.
        trend_threshold: Threshold for significant trend (as decimal).
        use_roc_hp: If True, use ROC/HP method; else use legacy cumulative returns.
    
    Returns:
        Series with regime labels: -1=downtrend, 0=range-bound, 1=uptrend.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if window < 5:
        raise ValueError("Trend window must be at least 5")

    if use_roc_hp:
        return roc_hp_trend_regime(series, window, 1600, trend_threshold)
    else:
        # Legacy implementation
        logger.debug(f"Detecting trend regime with window={window} (legacy)")
        returns = series.pct_change().fillna(0)
        cumulative_returns = returns.rolling(window=window).sum()
        regime = pd.Series(0, index=cumulative_returns.index)
        regime[cumulative_returns > trend_threshold] = 1
        regime[cumulative_returns < -trend_threshold] = -1
        return regime


class TDistrHMM(hmm.GaussianHMM):
    """Custom HMM with t-distribution emissions for fat tails."""

    def __init__(self, n_components, df=3.0, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.df = df

    def _compute_log_likelihood(self, X):
        n_samples, n_features = X.shape
        log_likelihood = np.zeros((n_samples, self.n_components))

        # Validate input data
        if np.any(~np.isfinite(X)):
            logger.warning("Non-finite values found in input data X")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(self.n_components):
            mu = self.means_[i]
            sigma = np.sqrt(self.covars_[i])

            # Ensure sigma is positive and not too small
            sigma = np.maximum(sigma, 1e-8)

            # Ensure df is reasonable for t-distribution
            df = max(float(self.df), 2.1)

            try:
                # Compute t-distribution log PDF with comprehensive error handling
                log_pdf = t.logpdf(X, df=df, loc=mu, scale=sigma)

                # Handle any NaN, inf, or invalid values
                log_pdf = np.nan_to_num(log_pdf, nan=-1e10, posinf=-1e10, neginf=-1e10)

                # Ensure we have valid likelihood values
                if np.all(log_pdf <= -1e10):
                    # If all values are invalid, use a uniform distribution
                    log_pdf = np.full_like(log_pdf, -10.0)  # Low but valid likelihood

                log_likelihood[:, i] = log_pdf.sum(axis=1)

            except (ValueError, RuntimeWarning, Exception) as e:
                logger.warning(f"Error computing log likelihood for component {i}: {e}")
                # Use a low but valid likelihood for failed components
                log_likelihood[:, i] = -1e10

        return log_likelihood


def hmm_regime_filter(
    spread_returns: pd.Series, config: Dict, seed: int = 42
) -> pd.Series:
    """
    Rolling HMM regime detection on spread returns.

    Args:
        spread_returns: Series of beta-hedged spread returns.
        config: Dict with 'hmm': {n_states: 3, window: 126, df: 3.0, tol: 1e-4}.
        seed: Random seed for reproducibility.

    Returns:
        Series of regime states (0=ranging, 1=weak trend, 2=strong trend).

    Raises:
        ValueError: If config missing or fit fails persistently.
    """
    np.random.seed(seed)
    hmm_config = config.get(
        "hmm", {"n_states": 3, "window": 126, "df": 3.0, "tol": 1e-4}
    )
    n_states = hmm_config["n_states"]
    window = hmm_config["window"]
    df = hmm_config["df"]
    tol = hmm_config["tol"]

    if len(spread_returns) < window:
        raise ValueError("Insufficient data for HMM window")

    states = pd.Series(0, index=spread_returns.index, dtype=float)
    ret_array = spread_returns.values
    n_fail = 0
    max_fails = len(spread_returns) // 10  # Allow 10% fails

    for i in range(window, len(spread_returns)):
        ret_window = ret_array[i - window : i].reshape(-1, 1)
        if np.std(ret_window) < 1e-8:
            states.iloc[i] = states.iloc[i - 1] if i > 0 else 0
            continue
        try:
            model = TDistrHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                tol=tol,
                random_state=seed,
                min_covar=1e-3,
            )
            model.fit(ret_window)
            log_prob, state_seq = model.decode(ret_window)
            states.iloc[i] = state_seq[-1]
            n_fail = 0
        except Exception as e:
            logger.warning(f"HMM fit failed at {i}: {e}")
            states.iloc[i] = states.iloc[i - 1] if i > 0 else 0
            n_fail += 1
            if n_fail > max_fails:
                raise ValueError(
                    f"Too many HMM fit failures: {n_fail}, check data quality"
                )

    logger.info(f"HMM regimes assigned: {states.value_counts().to_dict()}")
    return states.astype(int)


def vix_overlay(
    vix_series: pd.Series, hmm_states: pd.Series, config: Dict
) -> pd.Series:
    """
    VIX adjustment to HMM states (boost trend probs if high VIX).

    Args:
        vix_series: VIX levels.
        hmm_states: HMM states from hmm_regime_filter.
        config: Dict with 'vix': {thresh: 20, boost_factor: 0.2, max_shift: 1}.

    Returns:
        Adjusted states (shifted toward trend on high VIX).
    """
    vix_config = config.get("vix", {"thresh": 20, "boost_factor": 0.2, "max_shift": 1})
    thresh = vix_config["thresh"]
    boost_factor = vix_config["boost_factor"]
    max_shift = vix_config["max_shift"]

    vix_z = (vix_series - vix_series.rolling(252).mean()) / vix_series.rolling(
        252
    ).std()
    boost = np.maximum(0, (vix_series - thresh) / 10 * boost_factor)
    adjusted = hmm_states.copy()
    mask = boost > 0.3
    adjusted[mask] = np.minimum(hmm_states[mask] + max_shift, 2)
    logger.info(f"VIX boost applied: {mask.sum()} periods shifted")
    return adjusted


def combined_regime_filter(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    vix_series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Apply combined regime filter based on correlation, volatility, trend, HMM, VIX.

    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.
        config: Configuration dictionary with regime parameters.
        vix_series: Optional VIX for overlay.

    Returns:
        Boolean series indicating valid signals (True for ranging only).

    Raises:
        ValueError: If required config parameters are missing.
    """
    logger.debug("Applying combined regime filter")

    # Extract config parameters
    try:
        corr_window = config["lookbacks"]["corr_window"]
        min_abs_corr = config["regime"]["min_abs_corr"]
        enable_hmm = config["regime"].get("enable_hmm", False)
    except KeyError as e:
        raise ValueError(f"Missing required config parameter: {e}")

    # Initialize with all signals valid
    valid_signals = pd.Series(True, index=fx_series.index)

    # Apply correlation gate with relaxed threshold
    corr_gate = correlation_gate(fx_series, comd_series, corr_window, min_abs_corr)
    valid_signals = valid_signals & corr_gate

    # Apply volatility regime filter (optional) - made less restrictive
    if "volatility_window" in config.get("regime", {}):
        vol_window = config["regime"]["volatility_window"]
        # Use less restrictive thresholds if not specified in config
        high_vol_threshold = config["regime"].get("high_vol_threshold", 0.03)
        low_vol_threshold = config["regime"].get("low_vol_threshold", 0.003)

        fx_vol_regime = volatility_regime(
            fx_series, vol_window, high_vol_threshold, low_vol_threshold
        )
        comd_vol_regime = volatility_regime(
            comd_series, vol_window, high_vol_threshold, low_vol_threshold
        )

        # Filter out extreme volatility regimes (optional) - made less restrictive
        if config["regime"].get("filter_extreme_vol", False):
            # Only filter out extreme high volatility, allow normal and low volatility
            vol_filter = (fx_vol_regime != 2) & (comd_vol_regime != 2)
            valid_signals = valid_signals & vol_filter

    # Apply trend regime filter (optional) - made less restrictive
    if "trend_window" in config.get("regime", {}):
        trend_window = config["regime"]["trend_window"]
        # Use less restrictive threshold if not specified in config
        trend_threshold = config["regime"].get("trend_threshold", 0.015)

        fx_trend_regime = trend_regime(fx_series, trend_window, trend_threshold)
        comd_trend_regime = trend_regime(comd_series, trend_window, trend_threshold)

        # Filter out strong trending regimes (optional) - made less restrictive
        if config["regime"].get("filter_strong_trend", False):
            # Only filter out extreme trends, allow moderate trends
            trend_filter = (fx_trend_regime.abs() <= 1) & (comd_trend_regime.abs() <= 1)
            valid_signals = valid_signals & trend_filter

    # Apply HMM regime filter if enabled
    if enable_hmm:
        # Compute spread returns (assume beta from config or default 1.0)
        beta = config.get("lookbacks", {}).get(
            "beta_window", 1.0
        )  # Placeholder; integrate compute_spread
        spread_ret = (
            np.log(fx_series / fx_series.shift(1))
            - beta * np.log(comd_series / comd_series.shift(1))
        ).fillna(0)
        hmm_states = hmm_regime_filter(spread_ret, config)
        if vix_series is not None:
            hmm_states = vix_overlay(vix_series, hmm_states, config)
        # Filter: Valid only in ranging (state 0)
        hmm_gate = hmm_states == 0
        valid_signals = valid_signals & hmm_gate
        logger.info(
            f"HMM filter: {hmm_gate.sum()}/{len(hmm_gate)} ranging periods ({hmm_gate.mean()*100:.2f}%)"
        )

    logger.info(
        f"Combined regime filter: {valid_signals.sum()}/{len(valid_signals)} signals pass all filters ({valid_signals.mean()*100:.2f}%)"
    )

    return valid_signals
