import numpy as np
import pandas as pd
from loguru import logger


def _ensure_clean(y: pd.Series, x: pd.Series):
    df = pd.concat([y, x], axis=1).dropna()
    yy = df.iloc[:, 0].astype(float)
    xx = df.iloc[:, 1].astype(float)
    return yy, xx


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int):
    yy, xx = _ensure_clean(y, x)
    alpha = pd.Series(index=yy.index, dtype=float)
    beta = pd.Series(index=yy.index, dtype=float)
    for i in range(window, len(yy) + 1):
        yw = yy.iloc[i - window : i].values
        xw = xx.iloc[i - window : i].values
        X = np.c_[np.ones_like(xw), xw]
        b = np.linalg.lstsq(X, yw, rcond=None)[0]
        alpha.iloc[i - 1] = b[0]
        beta.iloc[i - 1] = b[1]
    return alpha.reindex(y.index).ffill(), beta.reindex(y.index).ffill()


def rls_beta(y: pd.Series, x: pd.Series, lam: float = 0.99, delta: float = 1000.0):
    """
    Recursive least squares with forgetting factor lam.
    Returns alpha_t, beta_t as Series aligned to y.index.
    """
    yy, xx = _ensure_clean(y, x)
    idx = yy.index
    theta = np.zeros(2)  # [alpha, beta]
    P = delta * np.eye(2)
    a = []
    b = []
    for t in range(len(yy)):
        Xt = np.array([1.0, xx.iloc[t]], dtype=float).reshape(1, 2)  # shape (1,2)
        yt = np.array([yy.iloc[t]], dtype=float)  # shape (1,)
        # RLS update
        PI = P @ Xt.T  # (2,1)
        K = PI @ np.linalg.inv(lam + Xt @ PI)  # (2,1)
        err = yt - Xt @ theta  # (1,)
        theta = theta + (K.flatten() * err.item())
        P = (P - K @ Xt @ P) / lam
        a.append(theta[0])
        b.append(theta[1])
    alpha = pd.Series(a, index=idx)
    beta = pd.Series(b, index=idx)
    return alpha.reindex(y.index).ffill(), beta.reindex(y.index).ffill()


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta_window: int,
    use_kalman: bool = True,
    _recursion_depth: int = 0,
):
    """
    Estimate alpha,beta (RLS when use_kalman=True else OLS) in LOG domain for stability,
    then compute a LEVEL-domain spread: S = y - exp(alpha_log) * x**beta.
    """
    # Prevent infinite recursion
    if _recursion_depth > 2:
        logger.error("Maximum recursion depth exceeded in compute_spread")
        return (
            pd.Series(index=y.index),
            pd.Series(index=y.index),
            pd.Series(index=y.index),
        )
    # Create copies to avoid modifying original series
    y = y.copy()
    x = x.copy()

    # Handle NaN and infinite values
    y = y.replace([np.inf, -np.inf], np.nan)
    x = x.replace([np.inf, -np.inf], np.nan)

    # Drop NaN values
    mask = ~(np.isnan(y) | np.isnan(x))
    y_clean = y[mask]
    x_clean = x[mask]

    # Check if we have enough data after cleaning
    if len(y_clean) < max(beta_window, 10):
        logger.warning(
            f"Insufficient clean data: {len(y_clean)} < {max(beta_window, 10)}, using OLS fallback"
        )
        return compute_spread(
            y, x, beta_window, use_kalman=False, _recursion_depth=_recursion_depth + 1
        )

    # Convert to log domain
    y_log = np.log(y_clean).replace([np.inf, -np.inf], np.nan)
    x_log = np.log(x_clean).replace([np.inf, -np.inf], np.nan)

    # Drop any NaN values that resulted from log transformation
    mask_log = ~(np.isnan(y_log) | np.isnan(x_log))
    y_log = y_log[mask_log]
    x_log = x_log[mask_log]

    # Check if we have enough data after log transformation
    if len(y_log) < max(beta_window, 10):
        logger.warning(
            f"Insufficient data after log transform: {len(y_log)} < {max(beta_window, 10)}, using OLS fallback"
        )
        return compute_spread(
            y, x, beta_window, use_kalman=False, _recursion_depth=_recursion_depth + 1
        )

    # Add small epsilon to prevent division by zero
    epsilon = 1e-8

    # Check for near-zero values
    if np.any(np.abs(x_log) < epsilon):
        logger.warning("Near-zero values in x_log, adding epsilon")
        x_log = x_log.copy()
        x_log[np.abs(x_log) < epsilon] = epsilon

    # Initialize alpha and beta variables
    alpha = None
    beta = None

    # Use Kalman filter (RLS) if requested and we have enough data
    if use_kalman:
        try:
            # Check for numerical stability
            y_std = np.std(y_log)
            x_std = np.std(x_log)
            if y_std < epsilon or x_std < epsilon:
                logger.warning(
                    f"Low volatility detected: y_std={y_std}, x_std={x_std}, using OLS fallback"
                )
                return compute_spread(
                    y,
                    x,
                    beta_window,
                    use_kalman=False,
                    _recursion_depth=_recursion_depth + 1,
                )

            # Use RLS (Kalman filter) on logs
            alpha_log, beta = rls_beta(y_log, x_log, lam=0.995, delta=100.0)

            # Check for valid results
            if alpha_log.isna().all() or beta.isna().all():
                logger.warning("RLS produced all NaN values, using OLS fallback")
                return compute_spread(
                    y,
                    x,
                    beta_window,
                    use_kalman=False,
                    _recursion_depth=_recursion_depth + 1,
                )

            # Convert alpha from log domain
            alpha = alpha_log.apply(np.exp)  # multiplicative intercept

        except Exception as e:
            logger.warning(f"Kalman filter error: {e}, using OLS fallback")
            return compute_spread(
                y,
                x,
                beta_window,
                use_kalman=False,
                _recursion_depth=_recursion_depth + 1,
            )
    else:
        # Use rolling OLS
        alpha_log, beta = rolling_ols_beta(y_log, x_log, beta_window)
        alpha = alpha_log.apply(np.exp)  # multiplicative intercept

    # Ensure alpha and beta are valid
    if alpha is None or beta is None:
        logger.error("Failed to compute alpha and beta")
        return (
            pd.Series(index=y.index),
            pd.Series(index=y.index),
            pd.Series(index=y.index),
        )

    # Clip extreme beta values for stability
    beta = beta.clip(-10, 10)

    # Ensure alpha is positive
    alpha = np.maximum(alpha, epsilon)

    # Compute spread in levels
    # We need to align the original series with the computed alpha and beta
    y_aligned = y_clean.reindex(alpha.index)
    x_aligned = x_clean.reindex(alpha.index)

    # Calculate spread: S = y - alpha * x^beta
    S = y_aligned - (alpha * (x_aligned**beta))

    # Reindex to original index and forward fill
    S = S.reindex(y.index).ffill()
    alpha = alpha.reindex(y.index).ffill()
    beta = beta.reindex(y.index).ffill()

    return S, alpha, beta
