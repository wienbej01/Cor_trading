# Risk & Execution Controls

This document describes the risk management and execution controls implemented for the FX-Commodity correlation arbitrage system.

## Overview

The risk and execution controls are designed to make the trading system more robust and realistic by implementing various safeguards and cost models. These controls include:

1. Daily Maximum Drawdown Control
2. Per-Trade Risk Cap
3. Enhanced Slippage and Transaction Cost Models
4. Position Scaling Rules Based on Volatility or ATR

## 1. Daily Maximum Drawdown Control

### Purpose

The daily maximum drawdown control prevents new position entries when the daily drawdown exceeds a specified limit, helping to protect against severe intraday losses.

### Implementation

- **Configuration Parameter**: `daily_drawdown_limit` (default: 2%)
- **Location**: Added to the `risk` section in `configs/pairs.yaml`
- **Functionality**: 
  - Tracks daily equity high to calculate drawdown
  - Prevents new position entries when daily drawdown exceeds the limit
  - Logs drawdown events for diagnostic purposes

### Configuration

```yaml
risk:
  daily_drawdown_limit: 0.02  # Maximum 2% daily drawdown per pair
```

## 2. Per-Trade Risk Cap

### Purpose

The per-trade risk cap limits the maximum risk that can be taken on any single trade, calculated as position_size * stop_loss_distance.

### Implementation

- **Configuration Parameter**: `max_trade_risk` (default: 1% of portfolio)
- **Location**: Added to the `risk` section in `configs/pairs.yaml`
- **Functionality**:
  - Calculates risk as position_size * stop_loss_distance
  - Reduces position size when calculated risk exceeds the cap
  - Logs instances where position size is adjusted due to risk limits

### Configuration

```yaml
risk:
  max_trade_risk: 0.01  # Maximum 1% of portfolio per trade
```

## 3. Enhanced Slippage and Transaction Cost Models

### Purpose

The enhanced slippage and transaction cost models provide a more realistic representation of trading costs, including both fixed and percentage components, as well as volatility and volume-based slippage.

### Implementation

- **Configuration Parameters**:
  - `fx_fixed_cost` (default: 0.0001 for FX)
  - `comd_fixed_cost` (default: $1 for futures)
  - `fx_percentage_cost` (default: 0.001% for both)
  - `comd_percentage_cost` (default: 0.001% for both)
  - `atr_slippage_multiplier` (default: 0.5)
  - `volume_slippage_multiplier` (default: 0.3)
- **Location**: Added to the `ExecutionConfig` class in `src/exec/policy.py`
- **Functionality**:
  - Includes both fixed and percentage transaction costs
  - Implements volatility-based slippage using ATR as a proxy
  - Incorporates volume-based slippage when volume data is available
  - Combines all cost components for total execution cost calculation

### Configuration

The execution cost parameters are configured in the `ExecutionConfig` class:

```python
# Transaction costs
fx_fixed_cost: float = 0.0001  # Fixed cost per FX trade (0.0001 for FX)
comd_fixed_cost: float = 1.0   # Fixed cost per commodity trade ($1 for futures)
fx_percentage_cost: float = 0.00001  # Percentage cost per FX trade (0.001%)
comd_percentage_cost: float = 0.00001  # Percentage cost per commodity trade (0.001%)

# Slippage multipliers
atr_slippage_multiplier: float = 0.5  # Multiplier for ATR-based slippage
volume_slippage_multiplier: float = 0.3  # Multiplier for volume-based slippage
```

## 4. Position Scaling Rules Based on Volatility or ATR

### Purpose

The position scaling rules adjust position sizes inversely to volatility or ATR to maintain constant risk, with a maximum position size limit.

### Implementation

- **Configuration Parameters**:
  - `volatility_scaling_enabled` (default: True)
  - `target_volatility` (default: 1%)
  - `max_position_size` (default: 10% of portfolio)
- **Location**: Added to the `sizing` section in `configs/pairs.yaml`
- **Functionality**:
  - Scales position size inversely to volatility or ATR
  - Uses ATR when available, otherwise uses regular volatility
  - Ensures position size does not exceed `max_position_size`

### Configuration

```yaml
sizing:
  volatility_scaling_enabled: true  # Enable volatility scaling
  target_volatility: 0.01  # Target volatility (1%)
  max_position_size: 0.10  # Maximum position size (10% of portfolio)
```

## Integration with Existing Risk Framework

All new risk controls are integrated with the existing risk management framework:

- The `RiskManager` class has been enhanced to include the new controls
- The `ExecutionPolicy` class has been enhanced to include the new cost models
- The `StressTester` class has been updated with new scenarios to test the controls
- Configuration parameters are properly integrated and can be enabled/disabled via configuration

## Running Unit Tests

To run the unit tests for the risk execution controls:

```bash
python test_risk_execution.py
```

The tests cover:
- Daily drawdown limit functionality
- Per-trade risk cap calculations
- Transaction cost calculations
- Position scaling based on volatility
- Integration with existing risk controls

## Stress Testing

The stress testing module includes new scenarios specifically designed to test the new risk controls:

- `Daily Drawdown Stress`: Tests the daily drawdown limits
- `High Volatility Regime`: Tests position scaling under high volatility conditions

To run stress tests:

```python
from src.risk.stress_test import run_default_stress_tests
# Run stress tests on your equity series
results, pass_fail = run_default_stress_tests(equity_series)