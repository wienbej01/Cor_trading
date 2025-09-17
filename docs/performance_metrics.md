# Performance Metrics & Short-Window Validation

This document describes the enhanced performance metrics and short-window validation functionality implemented for the FX-Commodity Correlation Arbitrage system.

## Overview

The enhanced performance metrics system addresses several key issues:

1. **NaN Performance Stats Fix**: Implements minimum trade count requirements for meaningful annualized metrics
2. **Rolling Window Metrics**: Provides time-series analysis of performance through rolling calculations
3. **Trade Return Distribution Analysis**: Adds statistical measures to understand return characteristics
4. **Configuration-Driven Approach**: Allows customization through configuration files

## Key Features

### 1. NaN Performance Stats Fix

The system now implements a minimum trade count requirement (default: 10 trades) for calculating annualized metrics. When the trade count is below this threshold, the system returns appropriate default values instead of NaN.

**Configuration Parameter**: `min_trade_count` in `configs/pairs.yaml`

### 2. Rolling Window Metrics

Rolling window calculations provide insight into how performance metrics evolve over time:

- **Rolling Sharpe Ratio**: 30D, 60D, 90D windows
- **Rolling Sortino Ratio**: 30D, 60D, 90D windows
- **Rolling Maximum Drawdown**: 30D, 60D, 90D windows

**Configuration Parameters**: `rolling_metrics.windows` in `configs/pairs.yaml`

### 3. Trade Return Distribution Analysis

Enhanced statistical analysis of trade returns:

- **Skewness**: Measure of asymmetry in return distribution
- **Kurtosis**: Measure of tail heaviness in return distribution
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Expected Shortfall (CVaR)**: 95% and 99% confidence levels

## Implementation Details

### Modules

1. **`src/backtest/engine.py`**: Enhanced performance metrics calculation with minimum trade count handling
2. **`src/backtest/rolling_metrics.py`**: Rolling window calculations for Sharpe, Sortino, and drawdown metrics
3. **`src/backtest/distribution_analysis.py`**: Statistical analysis of trade return distributions

### Configuration

The system uses the following configuration parameters in `configs/pairs.yaml`:

```yaml
# Performance metrics
min_trade_count: 10  # Minimum number of trades required for meaningful annualized metrics

# Rolling metrics
rolling_metrics:
  windows:
    "30D": 30
    "60D": 60
    "90D": 90
```

## API Usage

### Performance Metrics

```python
from src.backtest.engine import calculate_performance_metrics

# Calculate performance metrics with default minimum trade count
metrics = calculate_performance_metrics(backtest_df)

# Calculate performance metrics with custom minimum trade count
metrics = calculate_performance_metrics(backtest_df, min_trade_count=15)
```

### Rolling Metrics

```python
from src.backtest.rolling_metrics import calculate_rolling_metrics

# Calculate rolling metrics with default windows
rolling_metrics = calculate_rolling_metrics(equity_series)

# Calculate rolling metrics with custom windows
custom_windows = {"20D": 20, "40D": 40, "80D": 80}
rolling_metrics = calculate_rolling_metrics(equity_series, custom_windows)
```

### Distribution Analysis

```python
from src.backtest.distribution_analysis import analyze_return_distribution

# Analyze trade return distribution
dist_analysis = analyze_return_distribution(trade_returns)
```

## Interpretation of Metrics

### Skewness
- **Positive**: More large positive returns
- **Negative**: More large negative returns
- **Near Zero**: Approximately symmetric distribution

### Kurtosis
- **> 3 (Leptokurtic)**: Fat tails, more extreme values
- **< 3 (Platykurtic)**: Thin tails, fewer extreme values
- **≈ 3 (Mesokurtic)**: Normal-like tail behavior

### Value at Risk (VaR)
- **95% VaR**: Loss threshold not exceeded 95% of the time
- **99% VaR**: Loss threshold not exceeded 99% of the time

### Expected Shortfall (CVaR)
- **95% CVaR**: Average loss when losses exceed 95% VaR
- **99% CVaR**: Average loss when losses exceed 99% VaR

## Testing

Unit tests are provided in `test_performance_metrics.py` covering:

- NaN handling with insufficient trade counts
- Rolling window metric calculations
- Distribution analysis functions
- Integration with existing performance framework

Run tests with:
```bash
python test_performance_metrics.py
```

## Backward Compatibility

The enhancements maintain full backward compatibility with existing code. All new features are opt-in through configuration parameters, and default behavior remains unchanged for existing implementations.