import yfinance as yf
import pandas as pd
import numpy as np
import hmmlearn.hmm as hmm
import matplotlib.pyplot as plt
from loguru import logger
import statsmodels.api as sm  # For OLS beta fallback

# Fetch data individually (2015-2024 daily)
start = "2015-01-01"
end = "2024-09-12"
fx_data = yf.download("USDCAD=X", start=start, end=end)["Close"]
wti_data = yf.download("CL=F", start=start, end=end)["Close"]
vix_data = yf.download("^VIX", start=start, end=end)["Close"]
data = pd.concat([fx_data, wti_data, vix_data], axis=1).dropna()
data.columns = ["FX", "WTI", "VIX"]
logger.info(
    f"Data loaded: {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}, shape {data.shape}"
)

# Compute beta-hedged spread returns (rolling OLS beta for simplicity, no leakage)
data["fx_ret"] = np.log(data["FX"] / data["FX"].shift(1))
data["wti_ret"] = np.log(data["WTI"] / data["WTI"].shift(1))
betas = [1.0] * len(data)  # Default
for i in range(252, len(data)):
    slice_idx = slice(i - 252, i)
    y = data["fx_ret"].iloc[slice_idx].dropna()
    x = data["wti_ret"].iloc[slice_idx].dropna()
    if len(y) > 10 and len(x) == len(y):  # Align lengths
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        betas[i] = model.params[1] if len(model.params) > 1 else 1.0
data["beta"] = pd.Series(betas, index=data.index)
data["spread_ret"] = (data["fx_ret"] - data["beta"] * data["wti_ret"]).fillna(0)

# Rolling HMM fit (GaussianHMM; for t-dist, custom log-likelihood with t(df=3))
window = 126
n_states = 3
hmm_states = np.zeros(len(data))
spread_rets = data["spread_ret"].values

for i in range(window, len(data)):
    ret_window = spread_rets[i - window : i].reshape(-1, 1)
    if np.std(ret_window) < 1e-8:  # Avoid singular
        hmm_states[i] = hmm_states[i - 1] if i > 0 else 0
        continue
    try:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=50,
            random_state=42,
            min_covar=1e-3,
        )
        model.fit(ret_window)
        # Viterbi path for sequence
        log_prob, state_seq = model.decode(ret_window)
        hmm_states[i] = state_seq[-1]  # Last state as regime at t
    except Exception as e:
        logger.warning(f"HMM fit failed at {i}: {e}")
        hmm_states[i] = hmm_states[i - 1] if i > 0 else 0
    # Note: For t-dist, override _compute_log_likelihood with t.logpdf(x, df=3, loc=mu, scale=sigma)

data["hmm_state"] = hmm_states
# Interpret: Fit tends to assign 0=low mean/var (ranging), 2=high abs mean (strong trend), 1=transitional

# VIX overlay (z-score >1 boosts trend states)
vix_thresh = 20
data["vix_z"] = (data["VIX"] - data["VIX"].rolling(252).mean()) / data["VIX"].rolling(
    252
).std()
data["vix_boost"] = np.maximum(0, (data["VIX"] - vix_thresh) / 10 * 0.2)
data["adjusted_state"] = data["hmm_state"].copy()
mask_high_vix = data["vix_boost"] > 0.3
data.loc[mask_high_vix, "adjusted_state"] = np.minimum(
    data.loc[mask_high_vix, "hmm_state"] + 1, 2
)

# Simple persistence KPI
persistence = data["hmm_state"].value_counts(normalize=True)
logger.info(f"Regime distribution: {persistence}")
trans_matrix = pd.crosstab(
    data["hmm_state"].shift(), data["hmm_state"], normalize="index"
)
logger.info(f"Transition matrix:\n{trans_matrix}")

# Visualize
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
ax[0].plot(data.index, data["spread_ret"], label="Spread Ret", alpha=0.7)
ax[0].set_title("Spread Returns")
ax[1].plot(data.index, data["hmm_state"], label="HMM State", marker="o", markersize=2)
ax[1].set_title("HMM Regimes (0=Ranging, 2=Strong Trend)")
ax[1].set_ylim(-0.5, 2.5)
ax[2].plot(
    data.index,
    data["adjusted_state"],
    label="Adjusted w/ VIX",
    marker="s",
    markersize=2,
)
ax[2].twinx().plot(data.index, data["VIX"], color="r", alpha=0.5, label="VIX")
ax[2].set_title("VIX Overlay (Boosts Trends)")
plt.tight_layout()
plt.savefig("hmm_prototype.png")
plt.show()
logger.info("Prototype complete: States assigned, VIX boosted trends, plot saved")
print(data[["spread_ret", "hmm_state", "vix_boost", "adjusted_state"]].tail(10))
