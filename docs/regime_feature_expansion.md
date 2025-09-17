# Regime & Feature Expansion for FX-Commodity Correlation Arbitrage

## Overview

This document describes the implementation of regime classification and feature expansion modules for the FX-commodity correlation arbitrage system. These enhancements improve signal quality and filtering by incorporating market state awareness and additional technical factors.

## Features Implemented

### 1. Regime Classification

#### a) Volatility Regime Detection
- **Function**: `_volatility_regime`
- **Columns**: `feat_volatility_regime` (categorical), `feat_volatility_value` (numeric)
- **Description**: Classifies market volatility into low/normal/high regimes using rolling realized volatility percentiles. Optionally includes GARCH(1,1) volatility estimates.
- **Dependencies**: Commodity price series
- **Configuration**: 
  - `volatility_window`: Window for volatility calculation (default: 20)
  - `use_garch`: Enable GARCH volatility estimate (default: False)

#### b) Commodity Cycle Phase Detector
- **Function**: `_commodity_cycle_phase`
- **Columns**: `feat_commodity_cycle` (categorical), `feat_inventory_delta` (numeric)
- **Description**: Classifies commodity market cycle phases (expansion/peak/contraction/trough) using price trend and inventory signals.
- **Dependencies**: Commodity price series, EIA inventory data (optional)
- **Configuration**: 
  - `trend_window`: Window for price trend calculation (default: 20)

#### c) Macro Overlays
- **Function**: `_vix_and_yield_overlays`
- **Columns**: `feat_vix`, `feat_vix_regime`, `feat_yield_curve_slope`
- **Description**: Incorporates VIX-based regime classification and yield curve slope as macroeconomic factors.
- **Dependencies**: VIX data (via Yahoo Finance), yield curve data (configurable)
- **Configuration**: 
  - `vix_overlay`: Enable VIX overlay (default: True)
  - `yield_curve_symbol`: Symbol for yield curve data (optional)

### 2. New Features

#### a) Volume-Price Analysis (VPA)
- **Function**: `_compute_vpa`
- **Columns**: `feat_vpa_vwret`, `feat_vpa_volume_spike`, `feat_vpa_vol_adj_ret`
- **Description**: Computes volume-weighted price change metrics, volume spikes, and volume-adjusted returns. Uses volume proxy for FX pairs when actual volume is unavailable.
- **Dependencies**: Price series, volume data (optional)
- **Configuration**: 
  - `liquidity_proxy`: Proxy value for FX volume (default: 1.0)

#### b) ICT-Style Liquidity Sweep Detector
- **Function**: `_liquidity_sweep_detector`
- **Columns**: `feat_liquidity_sweep_bool`, `feat_liquidity_sweep_score`
- **Description**: Detects large single-bar wicks relative to ATR, followed by reversal with volume spike.
- **Dependencies**: OHLC data, volume data (optional)

#### c) Trend Filters
- **Function**: `_trend_filters_and_adx`
- **Columns**: `feat_trend_slope_*`, `feat_adx`, `feat_ma_crossover_*`
- **Description**: Multi-horizon slope measures, ADX approximation, and moving-average crossover signals.
- **Dependencies**: Price series, OHLC data (for ADX)

#### d) Correlation & Regime-Aware Features
- **Function**: Part of `compute_regime_and_features`
- **Columns**: `feat_corr_rolling`, `feat_beta_std`
- **Description**: Rolling correlation (clipped and normalized) and hedge ratio stability metric.
- **Dependencies**: FX and commodity price series

## API Usage

### Main Function
```python
compute_regime_and_features(df, config, lookahead_shift=1)
```

**Parameters**:
- `df`: Input DataFrame with price columns
- `config`: Configuration dictionary (from `configs/pairs.yaml`)
- `lookahead_shift`: Number of bars to shift features forward (default: 1)

**Returns**: DataFrame with original columns plus new `feat_*` columns, shifted by `lookahead_shift` to prevent look-ahead bias.

### Configuration Toggles
Add to `configs/pairs.yaml` under a new `regime_features` section:
```yaml
regime_features:
  volatility_regime: true
  commodity_cycle: true
  vix_overlay: true
  yield_curve_overlay: false
  vpa: true
  liquidity_sweep: true
  trend_filters: true
  correlation_features: true
```

## Diagnostic Functions

### Plotting Functions
Located in `src/features/diagnostics.py`:
- `plot_regime_timeline`: Visualize regime classification against price
- `plot_feature_correlation_heatmap`: Show correlation between features
- `plot_feature_with_price`: Overlay feature with price series
- `plot_signals_with_features`: Plot trading signals with selected features

### Test Script
`test_regime_features.py` demonstrates usage and generates diagnostic plots:
```bash
python test_regime_features.py
```

## Implementation Details

### Look-Ahead Prevention
All features are shifted forward by `lookahead_shift` bars before being returned, ensuring they are safe to use for decisions executed at `bar_close + lookahead_shift`.

### Data Handling
- Functions gracefully handle missing data with NaN values
- Categorical columns are explicitly typed as "object" for consistency
- Volume proxies are used for FX pairs when actual volume data is unavailable

### Performance Considerations
- All computations are vectorized using pandas/numpy for efficiency
- Optional GARCH calculations are isolated and won't affect performance when disabled
- Diagnostic plotting functions are separate from core computation logic

## Testing

Unit tests in `test_features_regime.py` cover:
- Volatility regime classification edge cases
- Inventory-based cycle detection with synthetic data
- VPA behavior with and without volume data
- Look-ahead artifact verification

Run tests with:
```bash
python -m pytest test_features_regime.py
```

## Limitations and Next Steps

### Known Limitations
1. GARCH functionality requires optional `arch` package
2. EIA inventory data integration is stubbed in current implementation
3. Some features require OHLC data for full functionality

### Recommended Next Steps
1. Implement full EIA API integration for inventory data
2. Add more sophisticated regime classification models
3. Extend feature set with additional technical indicators
4. Implement feature importance analysis for signal optimization