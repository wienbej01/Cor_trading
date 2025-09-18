# Feature Specification - FX-Commodity Mean Reversion

This document defines the precise features used in the FX-commodity correlation mean-reversion system, their economic rationale, and ablation paths for research. Features are computed to avoid leakage and align with the embargoed label construction.

Code references
- Feature preparation: [interfaces.feature_preparation.prepare_features()](src/interfaces/feature_preparation.py:1)
- Spread features: [features.spread.compute_spread()](src/features/spread.py:54)
- Indicators: [features.indicators.zscore_robust()](src/features/indicators.py:233)
- Regime features: [features.regime.combined_regime_filter()](src/features/regime.py:355)
- Cointegration: [features.cointegration.is_cointegrated()](src/features/cointegration.py:164)

1) Feature set (parsimonious)
All features are computed at decision time t using only data up to and including t.

- Spread features:
  • spread_t = FX_t − alpha_t × COMD_t^{beta_t} via [features.spread.compute_spread()](src/features/spread.py:54)
  • z_t = (spread_t − median_{t−w:t}(spread)) / (1.4826×MAD_{t−w:t}(spread) + 1e−12) via [features.indicators.zscore_robust()](src/features/indicators.py:233)
  • spread_vol_t = stdev_20(spread)_t for volatility normalization

- Regime features:
  • corr_t = corr_{t−w:t}(FX, COMD) via [features.regime.correlation_gate()](src/features/regime.py:17)
  • vol_regime_t = classify_volatility(spread_vol_t) via [features.regime.volatility_regime()](src/features/regime.py:233)
  • trend_regime_t = classify_trend(spread_trend_20) via [features.regime.trend_regime()](src/features/regime.py:266)
  • hmm_state_t = TDistrHMM_state(spread_return_20) via [features.regime.TDistrHMM()](src/features/regime.py:145)

- Structural features:
  • adf_pvalue_t = ADF_test_pvalue(spread_{t−w:t}) via [features.cointegration.adf_pvalue()](src/features/cointegration.py:1)
  • ou_hl_t = OU_half_life(spread_{t−w:t}) via [features.cointegration.ou_half_life()](src/features/cointegration.py:52)
  • hurst_t = Hurst_exponent(spread_{t−w:t}) via [features.cointegration.hurst_exponent()](src/features/cointegration.py:124)

2) Economic rationale per feature group
- Spread features capture the instantaneous deviation from long-run equilibrium
- Regime features identify when the market environment supports mean-reversion trades
- Structural features quantify the statistical reliability of the cointegration relationship

3) Ablation paths for research
- Baseline: z_t + corr_t + vol_regime_t + trend_regime_t + hmm_state_t
- Ablation 1: Remove regime features (test pure z-score strategy)
- Ablation 2: Remove structural features (no cointegration diagnostics)
- Ablation 3: Replace robust z-score with standard z-score (mean/stdev)
- Ablation 4: Disable HMM (hmm_state_t = NaN or fixed value)
- Ablation 5: Use only structural features (test cointegration filter alone)

4) Feature preparation and leakage control
- All features computed in [interfaces.feature_preparation.prepare_features()](src/interfaces/feature_preparation.py:1)
- Strict embargoing via [interfaces.feature_preparation.embargoed_split()](src/interfaces/feature_preparation.py:201)
- No future data referenced in any feature calculation

5) Testability checklist
- [ ] All features reference only data up to decision time t
- [ ] Feature set aligns with rulebook regime gates and signal logic
- [ ] Ablation paths are cleanly separable in feature matrix
- [ ] Embargo periods match label windows (dynamic time-stop)

6) Implementation notes
- Features are stored in a DataFrame with a DateTime index
- Regime features can be one-hot encoded or used as categorical
- Structural features can be clipped or binned for ML stability
- Feature normalization (if needed) uses only in-sample data