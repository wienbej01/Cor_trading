# ML Labeling Specification for Trade Filter

## Overview

This document specifies the supervised learning setup for predicting spread reversion quality in FX-commodity correlation arbitrage. The ML filter learns to gate trades that are likely to revert to the mean within acceptable risk parameters.

## Label Definition

### Primary Label: Reversion Success
**Definition**: Binary classification where label = 1 if the spread reverts to within 0.5 standard deviations of its historical mean within 20 trading bars (approximately 1 month), subject to risk constraints.

**Formal Specification**:
```
label = 1 if:
  - Spread reverts to [mean - 0.5σ, mean + 0.5σ] within 20 bars
  - Risk-reward ratio ≥ 1.5
  - Maximum adverse excursion ≤ 2.0 standard deviations
  - No violation of temporal integrity (no lookahead)
```

**Label Timing**:
- Features computed at bar close (t=0)
- Label determined by future spread behavior (t=1 to t=20)
- Signals evaluated at bar_close + 1 bar (no same-bar fills)

### Risk-Reward Calculation
```
RR = |take_profit - entry_price| / |stop_loss - entry_price|
```
Where take_profit and stop_loss are set based on spread volatility and strategy parameters.

### Adverse Excursion
Maximum deviation from entry price during the evaluation window, measured in standard deviations of the spread.

## Leakage Prevention

### Embargo Period
- **5 bars**: No samples created within 5 bars after a label is assigned
- Prevents overlap between training samples and ensures temporal separation

### Purge Window
- **252 bars (1 year)**: Remove overlapping samples within 1-year windows
- Ensures each market regime period contributes independently to training

### Feature Engineering Constraints
- All features computed using only historical data up to t=0
- No future information leakage
- Rolling statistics use expanding windows or fixed lookbacks
- Regime classifications based on pre-entry conditions only

## Feature Specification

### Parsimony Principle
Features selected for economic interpretability and minimal redundancy. Total: 18 features across 5 categories.

### Spread Features (4)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| spread_z_20 | Z-score over 20 bars | Current deviation from short-term mean |
| spread_z_60 | Z-score over 60 bars | Longer-term deviation context |
| spread_momentum_5 | 5-bar momentum | Recent directional bias |
| spread_momentum_20 | 20-bar momentum | Medium-term trend strength |

### Volatility Features (4)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| spread_vol_20 | Spread volatility (20 bars) | Current spread instability |
| spread_vol_60 | Spread volatility (60 bars) | Trend in volatility |
| fx_vol_20 | FX pair volatility | Base currency risk |
| comd_vol_20 | Commodity volatility | Quote currency risk |

### Correlation Features (3)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| rolling_corr_20 | Rolling correlation (20 bars) | Current pair relationship |
| rolling_corr_60 | Rolling correlation (60 bars) | Correlation trend |
| corr_z_score | Correlation deviation | Unusual correlation levels |

### Regime Features (3)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| trend_regime | -1/0/1 (down/range/up) | Market trend state |
| vol_regime | 0/1/2 (low/normal/high) | Volatility environment |
| combined_regime | Combined trend+vol | Overall market regime |

### Temporal Features (4)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| day_of_week | 0-6 (Mon-Sun) | Weekly seasonality |
| month_of_year | 1-12 | Annual seasonality |
| quarter | 1-4 | Quarterly patterns |
| holiday_proximity | Days to/from holiday | Event-driven volatility |

## Class Balance Strategy

### Target Distribution
- Positive class (successful reversions): 30-40%
- Negative class (failed reversions): 60-70%

### Balancing Techniques
1. **Undersampling negatives**: Random undersampling of failed trades
2. **Oversampling positives**: SMOTE or similar for rare successful cases
3. **Cost-sensitive learning**: Higher penalty for false negatives (missed opportunities)

### Rationale
- Mean reversion opportunities are rarer than failures
- False negatives (rejecting good trades) more costly than false positives
- Maintains economic realism while ensuring model learns from both outcomes

## Economic Rationale

### Why These Features?
- **Spread metrics**: Direct measure of deviation driving reversion
- **Volatility**: Risk context affects reversion probability and speed
- **Correlation**: Pair relationship strength influences mean reversion
- **Regime**: Market state determines strategy appropriateness
- **Temporal**: Seasonal patterns in FX/commodity markets

### Risk Management Integration
- Labels incorporate risk-reward and adverse excursion limits
- Features include volatility for position sizing context
- Regime awareness prevents inappropriate market conditions

## Validation Checks

### Feature Integrity
- No NaN values in feature sets
- Stationarity tests for time series features
- Correlation analysis to detect redundancy (< 0.95 threshold)

### Label Quality
- Temporal separation verification
- No overlap between train/validation splits
- Class balance within acceptable ranges

### Leakage Tests
- Feature importance analysis for suspicious patterns
- Cross-validation stability checks
- Out-of-sample performance monitoring

## Implementation Notes

### Data Pipeline
1. Generate candidate trade signals from strategy
2. Compute features at signal timestamp
3. Simulate trade outcome over 20-bar horizon
4. Assign label based on success criteria
5. Apply embargo and purge filters
6. Balance classes for training

### Monitoring
- Track feature distributions over time
- Monitor label stability across market regimes
- Validate no information leakage in production

## Version History

- v1.0: Initial specification (2025-09-18)
- Economic rationale: Mean reversion probability increases with deviation magnitude but decreases with volatility and weak correlation
- Risk focus: Adverse excursion control prevents catastrophic losses during evaluation period