# Model Diversification for FX-Commodity Correlation Arbitrage

## Overview

This document describes the implementation of model diversification for the FX-Commodity correlation arbitrage system. The implementation includes:

1. A multi-model ensemble framework with various statistical and machine learning models
2. A parallel backtesting framework for model comparison
3. Feature importance calculations and model diagnostics
4. Configuration options for enabling/disabling individual models

## Multi-Model Ensemble Framework

The ensemble framework implements several models for spread prediction and trading signal generation:

### 1. OLS Model (Ordinary Least Squares)
- Implements rolling window OLS regression for hedge ratio calculation
- Uses a configurable window size (default: 90 periods)
- Provides feature importance based on coefficient magnitudes

### 2. Kalman Model (Kalman Filter)
- Uses Recursive Least Squares with forgetting factor for dynamic beta estimation
- Adapts to changing market conditions with configurable lambda parameter (default: 0.995)
- Provides feature importance for intercept and slope parameters

### 3. Rolling Correlation Model
- Calculates rolling correlation between FX and commodity prices
- Uses a configurable window size (default: 20 periods)
- Feature importance based on correlation magnitude

### 4. Gradient Boosting Model
- Machine learning model for residual prediction
- Configurable parameters:
  - Number of estimators (default: 100)
  - Maximum depth (default: 3)
  - Learning rate (default: 0.1)

### 5. LSTM Model
- Deep learning model for sequence prediction
- Configurable parameters:
  - Hidden size (default: 50)
  - Number of layers (default: 2)
  - Training epochs (default: 50)
  - Learning rate (default: 0.001)

### Ensemble Weighting
The ensemble combines predictions from all enabled models using weighted averaging. Default weights are equal (0.2 for each model), but can be configured in `pairs.yaml`.

## Parallel Backtesting Framework

The parallel backtesting framework enables efficient comparison of different models and parameter sets:

### Features
- Parallel execution of backtests using ProcessPoolExecutor
- Model comparison across different statistical and ML approaches
- Parameter sweep for threshold optimization
- Aggregated results with performance metrics

### Configuration
- Number of parallel workers (default: 4)
- Timeout for individual backtests (default: 3600 seconds)
- Parameter ranges for optimization

## Feature Importances & Model Diagnostics

### Feature Importance
- OLS and Kalman models: Coefficient magnitudes
- Gradient Boosting: Built-in feature importances
- LSTM: Not directly available

### SHAP Values
When the SHAP library is available, SHAP values can be calculated for model interpretation.

### Performance Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2)

## Configuration Parameters

Model diversification parameters are configured in `configs/pairs.yaml` under the `model_diversification` section:

```yaml
model_diversification:
  enable_ensemble: true           # Enable ensemble model
  enable_ols: true               # Enable OLS model
  enable_kalman_model: true      # Enable Kalman model
  enable_correlation_model: true # Enable correlation model
  enable_gb_model: true          # Enable Gradient Boosting model
  enable_lstm_model: true        # Enable LSTM model
  model_weights:
    ols: 0.2
    kalman: 0.2
    corr: 0.2
    gb: 0.2
    lstm: 0.2
```

## API Usage

### Ensemble Model
```python
from src.ml.ensemble import create_ensemble_model, ModelConfig

# Create ensemble model
config = ModelConfig()
ensemble = create_ensemble_model(config)

# Fit model
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
ensemble_prediction = ensemble.predict_ensemble(X_test)
```

### Parallel Backtesting
```python
from src.backtest.parallel import ParallelBacktester, BacktestConfig

# Create backtester
config = BacktestConfig()
backtester = ParallelBacktester(config)

# Run model comparison
results = backtester.run_model_comparison(data, base_config)

# Run parameter sweep
results = backtester.run_parameter_sweep(data, base_config, "ols")

# Get best results
best_results = backtester.get_best_results(metric="sharpe_ratio", top_n=5)
```

### ML Diagnostics
```python
from src.ml.diagnostics import MLDiagnostics

# Create diagnostics
diagnostics = MLDiagnostics()

# Calculate feature importance
importance = diagnostics.calculate_feature_importance(model, "model_name", X)

# Calculate model performance
metrics = diagnostics.calculate_model_performance(model, "model_name", X_test, y_test)

# Get diagnostics report
report = diagnostics.get_diagnostics_report()
```

## Model Comparison and Selection

The system provides tools for comparing different models:

1. **Performance Metrics**: Compare Sharpe ratio, maximum drawdown, and other metrics
2. **Feature Importance**: Understand which features drive each model
3. **Diagnostic Plots**: Visualize model performance and feature importances

## Examples

### Running Ensemble Backtest
```python
from src.backtest.engine import run_backtest_with_ensemble

# Run backtest with ensemble model
backtest_df, metrics = run_backtest_with_ensemble(
    fx_series, comd_series, config
)
```

### Comparing Models
```python
from src.backtest.engine import compare_models

# Compare different models
comparison_df = compare_models(
    fx_series, comd_series, config, 
    models=["default", "ols", "kalman", "corr", "ensemble"]
)
```

## Implementation Details

### File Structure
- `src/ml/ensemble.py`: Ensemble model implementation
- `src/backtest/parallel.py`: Parallel backtesting framework
- `src/ml/diagnostics.py`: Feature importance and diagnostics
- `configs/pairs.yaml`: Configuration parameters

### Dependencies
- scikit-learn (optional): For Gradient Boosting model
- torch (optional): For LSTM model
- shap (optional): For SHAP values

## Limitations and Future Work

### Current Limitations
1. LSTM model requires PyTorch
2. Gradient Boosting model requires scikit-learn
3. SHAP values require the shap library
4. Limited feature engineering in current implementation

### Future Enhancements
1. Additional ML models (Random Forest, XGBoost, etc.)
2. Advanced feature engineering
3. Online learning capabilities
4. More sophisticated ensemble methods (stacking, boosting, etc.)