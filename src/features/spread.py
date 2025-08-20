import numpy as np
import pandas as pd

def _ensure_clean(y: pd.Series, x: pd.Series):
    df = pd.concat([y, x], axis=1).dropna()
    yy = df.iloc[:,0].astype(float)
    xx = df.iloc[:,1].astype(float)
    return yy, xx

def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int):
    yy, xx = _ensure_clean(y, x)
    alpha = pd.Series(index=yy.index, dtype=float)
    beta  = pd.Series(index=yy.index, dtype=float)
    for i in range(window, len(yy)+1):
        yw = yy.iloc[i-window:i].values
        xw = xx.iloc[i-window:i].values
        X = np.c_[np.ones_like(xw), xw]
        b = np.linalg.lstsq(X, yw, rcond=None)[0]
        alpha.iloc[i-1] = b[0]
        beta.iloc[i-1]  = b[1]
    return alpha.reindex(y.index).ffill(), beta.reindex(y.index).ffill()

def rls_beta(y: pd.Series, x: pd.Series, lam: float = 0.99, delta: float = 1000.0):
    """
    Recursive least squares with forgetting factor lam.
    Returns alpha_t, beta_t as Series aligned to y.index.
    """
    yy, xx = _ensure_clean(y, x)
    idx = yy.index
    theta = np.zeros(2)          # [alpha, beta]
    P = delta * np.eye(2)
    a = []; b = []
    for t in range(len(yy)):
        Xt = np.array([1.0, xx.iloc[t]], dtype=float).reshape(1,2)  # shape (1,2)
        yt = np.array([yy.iloc[t]], dtype=float)                    # shape (1,)
        # RLS update
        PI = P @ Xt.T                           # (2,1)
        K = PI @ np.linalg.inv(lam + Xt @ PI)   # (2,1)
        err = yt - Xt @ theta                   # (1,)
        theta = theta + (K.flatten() * err.item())
        P = (P - K @ Xt @ P) / lam
        a.append(theta[0]); b.append(theta[1])
    alpha = pd.Series(a, index=idx)
    beta  = pd.Series(b, index=idx)
    return alpha.reindex(y.index).ffill(), beta.reindex(y.index).ffill()

def compute_spread(y: pd.Series, x: pd.Series, beta_window: int, use_kalman: bool = True):
    """
    Estimate alpha,beta (RLS when use_kalman=True else OLS) in LOG domain for stability,
    then compute a LEVEL-domain spread: S = y - exp(alpha_log) * x**beta.
    """
    y_log = np.log(y).replace([np.inf, -np.inf], np.nan)
    x_log = np.log(x).replace([np.inf, -np.inf], np.nan)
    y_log, x_log = _ensure_clean(y_log, x_log)

    if use_kalman:
        # use RLS on logs
        alpha_log, beta = rls_beta(y_log, x_log, lam=0.995, delta=100.0)
    else:
        alpha_log, beta = rolling_ols_beta(y_log, x_log, beta_window)

    alpha = alpha_log.apply(np.exp)  # multiplicative intercept
    # Spread in levels to match trading legs
    S = y - (alpha * (x ** beta))
    S = S.reindex(y.index).dropna()
    alpha = alpha.reindex(y.index).ffill()
    beta = beta.reindex(y.index).ffill()
    return S, alpha, beta