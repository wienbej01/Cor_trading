# ML Labeling Strategy

**Version:** 1.0
**Date:** 2025-09-18
**Author:** Gemini

## 1. Objective

The goal of the ML filter is to improve the precision of the mean-reversion strategy by learning to identify which trading opportunities are most likely to result in a profitable reversion. We formulate this as a binary classification problem: given a set of features at the time of a z-score entry signal, predict whether the trade will be "good" or "bad".

## 2. Label Definition

A trade opportunity is labeled as **`1` (good)** if it meets all of the following criteria:

1.  **Reversion:** The spread value crosses its moving average (the "mean") within a forward-looking horizon of `H` bars.
2.  **Risk-Reward Ratio:** The trade achieves a pre-defined risk-reward ratio (`RR`) of at least `X` before hitting its stop-loss. The target profit is defined by the initial z-score and spread volatility, while the stop-loss is based on a multiple of the Average True Range (ATR) or a fixed percentage.
3.  **Limited Drawdown:** The maximum adverse excursion (MAE) during the trade's lifetime (before exit) does not exceed `Y` times the initial stop-loss distance. This penalizes trades that experience extreme volatility, even if they eventually become profitable.

A trade is labeled as **`0` (bad)** if any of the above conditions are not met within the `H` bar horizon.

### Parameters

*   **`H` (Horizon):** The number of forward-looking bars to check for a successful reversion. This should be aligned with the strategy's expected holding period. (e.g., 20 bars for H1 timeframe).
*   **`X` (Risk-Reward Ratio):** The minimum acceptable profit-to-risk ratio. (e.g., 1.5).
*   **`Y` (Max Adverse Excursion Multiplier):** A factor to control for excessive interim drawdown. (e.g., 2.0).

## 3. Feature Specification

Features are chosen for their economic rationale and interpretability. We aim for parsimony to reduce the risk of overfitting. All features are calculated *at the time of the entry signal* and contain no forward-looking information.

### Core Features
*   `z_score`: The entry z-score itself.
*   `spread_rolling_std_20`: 20-period rolling standard deviation of the spread.
*   `spread_roc_10`: 10-period Rate of Change of the spread.

### Regime Features
*   `fx_vol_regime`: Volatility regime of the FX component (e.g., low, medium, high).
*   `commodity_vol_regime`: Volatility regime of the commodity component.
*   `fx_trend_regime`: Trend regime of the FX component (e.g., up, down, sideways).
*   `commodity_trend_regime`: Trend regime of the commodity component.

## 4. Preventing Data Leakage

Data leakage is a critical risk. We employ the following standard MLaaS (Machine Learning as a Service) techniques:

1.  **Purging:** When creating labels for a sample at time `t`, we must ensure that the data used to determine the outcome (i.e., data from `t+1` to `t+H`) is not used as features for *any other sample* between `t` and `t+H`. We will purge any training samples that fall within the labeling horizon of a previously labeled sample.
2.  **Embargoing:** To prevent serial correlation effects, we apply an "embargo" period after the end of the labeling horizon. A certain number of samples (`k` bars) following `t+H` are dropped from the training set. This helps ensure that the training samples are more independent.
3.  **Time-Series Cross-Validation:** Standard k-fold cross-validation is inappropriate for time-series data. We will use a walk-forward, expanding window cross-validation scheme. The training set will always precede the validation set in time.

## 5. Class Balance

Mean-reversion opportunities that meet our strict labeling criteria are often rare, leading to a significant class imbalance (many more "bad" trades than "good" ones).

The `src/ml/dataset.py` builder will have options to handle this, such as:
*   **Undersampling:** Randomly remove samples from the majority class.
*   **Oversampling (e.g., SMOTE):** Synthetically generate new samples for the minority class.

The choice of technique will be evaluated during model training. For the baseline, we will likely start with undersampling to create a balanced dataset.
