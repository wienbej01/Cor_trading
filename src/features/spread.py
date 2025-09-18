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
):
    """
    Estimate alpha,beta (RLS when use_kalman=True else OLS) in LOG domain for stability,
    then compute a LEVEL-domain spread: S = y - alpha * x**beta.

    This implementation avoids recursive fallbacks and instead selects an effective
    beta window based on available clean data. If there is insufficient data
    (less than 10 usable points) the function returns empty Series aligned to the
    original index.
    """
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

    available = len(y_clean)
    epsilon = 1e-8

    # Require at least 10 clean points to proceed
    if available < 10:
        logger.warning(f"Insufficient clean data for compute_spread: {available} points (need >=10)")
        empty = pd.Series(index=y.index, dtype=float)
        return empty, empty, empty

    # Determine effective beta window (bounded by available data)
    effective_beta = max(10, min(int(beta_window), available))

    # Convert to log domain
    y_log = np.log(y_clean).replace([np.inf, -np.inf], np.nan)
    x_log = np.log(x_clean).replace([np.inf, -np.inf], np.nan)

    # Drop any NaN values that resulted from log transformation
    mask_log = ~(np.isnan(y_log) | np.isnan(x_log))
    y_log = y_log[mask_log]
    x_log = x_log[mask_log]

    if len(y_log) < 10:
        logger.warning(f"Insufficient log-transformed data: {len(y_log)} points (need >=10)")
        empty = pd.Series(index=y.index, dtype=float)
        return empty, empty, empty

    # Check for near-zero values
    if np.any(np.abs(x_log) < epsilon):
        logger.warning("Near-zero values in x_log, adding epsilon")
        x_log = x_log.copy()
        x_log[np.abs(x_log) < epsilon] = epsilon

    alpha = None
    beta = None

    # Try RLS/Kalman first if requested
    if use_kalman:
        try:
            y_std = float(np.std(y_log))
            x_std = float(np.std(x_log))
            if y_std < epsilon or x_std < epsilon:
                logger.warning("Low volatility detected, falling back to OLS")
                use_kalman = False
            else:
                alpha_log, beta = rls_beta(y_log, x_log, lam=0.995, delta=100.0)
                if alpha_log.isna().all() or beta.isna().all():
                    logger.warning("RLS produced invalid results, falling back to OLS")
                    use_kalman = False
                else:
                    alpha = alpha_log.apply(np.exp)
        except Exception as exc:
            logger.warning(f"Kalman/RLS error: {exc}; falling back to OLS")
            use_kalman = False

    # OLS fallback (rolling OLS) using effective_beta
    if not use_kalman:
        alpha_log, beta = rolling_ols_beta(y_log, x_log, effective_beta)
        alpha = alpha_log.apply(np.exp)

    # Validate results
    if alpha is None or beta is None or alpha.empty or beta.empty:
        logger.error("Failed to compute alpha/beta (empty results)")
        empty = pd.Series(index=y.index, dtype=float)
        return empty, empty, empty

    # Stabilize values
    beta = beta.clip(-10, 10)
    alpha = np.maximum(alpha, epsilon)

    # Align original series with computed parameters
    y_aligned = y_clean.reindex(alpha.index)
    x_aligned = x_clean.reindex(alpha.index)

    S = y_aligned - (alpha * (x_aligned ** beta))

    S = S.reindex(y.index).ffill()
    alpha = alpha.reindex(y.index).ffill()
    beta = beta.reindex(y.index).ffill()

    return S, alpha, beta
