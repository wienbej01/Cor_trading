# FX-Commodity Correlation Arbitrage System - API Reference

## Overview

This document provides comprehensive API reference for all public interfaces in the FX-Commodity Correlation Arbitrage trading system. All functions, classes, and methods documented here are considered stable public APIs with backward compatibility guarantees.

## Table of Contents

- [Core Module APIs](#core-module-apis)
- [Data Loading APIs](#data-loading-apis)
- [Feature Engineering APIs](#feature-engineering-apis)
- [Machine Learning APIs](#machine-learning-apis)
- [Strategy APIs](#strategy-apis)
- [Risk Management APIs](#risk-management-apis)
- [Execution APIs](#execution-apis)
- [Backtesting APIs](#backtesting-apis)
- [Utility APIs](#utility-apis)

## Core Module APIs

### ConfigManager Class

**Module**: `src.core.config`

```python
class ConfigManager:
    """Manages loading and access to configuration files."""
```

#### Constructor

```python
def __init__(self, config_path: Optional[Path] = None) -> None
```

**Parameters**:
- `config_path` (Optional[Path]): Path to configuration directory. Defaults to `../configs`.

#### Methods

##### load_pairs_config()

```python
def load_pairs_config() -> Dict[str, Any]
```

Load pairs configuration from YAML file.

**Returns**: Dictionary containing pairs configuration.

**Raises**: 
- `FileNotFoundError`: If pairs.yaml not found
- `yaml.YAMLError`: If YAML parsing fails

##### get_pair_config()

```python
def get_pair_config(pair_name: str) -> Dict[str, Any]
```

Get configuration for a specific trading pair.

**Parameters**:
- `pair_name` (str): Name of the pair (e.g., "usdcad_wti")

**Returns**: Dictionary containing pair configuration.

**Raises**: `KeyError`: If pair not found in configuration.

##### get_risk_config()

```python
def get_risk_config(pair_name: str) -> Dict[str, Any]
```

Get risk configuration for a specific pair.

**Parameters**:
- `pair_name` (str): Name of the pair

**Returns**: Dictionary containing risk configuration.

##### list_pairs()

```python
def list_pairs() -> List[str]
```

Get list of available pair names.

**Returns**: List of configured pair names.

## Data Loading APIs

### Yahoo Finance Data Loader

**Module**: `src.data.yahoo_loader`

#### Functions

##### download_daily()

```python
def download_daily(
    symbol: str, 
    start: str, 
    end: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> pd.Series
```

Download daily price data from Yahoo Finance with retry logic.

**Parameters**:
- `symbol` (str): Financial instrument symbol (e.g., "USDCAD=X", "CL=F")
- `start` (str): Start date in "YYYY-MM-DD" format
- `end` (str): End date in "YYYY-MM-DD" format
- `max_retries` (int): Maximum retry attempts (default: 3)
- `retry_delay` (float): Delay between retries in seconds (default: 1.0)

**Returns**: pandas Series with daily close prices indexed by date.

**Raises**:
- `ValueError`: Invalid symbol, date range, or no data found
- `ConnectionError`: Unable to connect after retries

##### align_series()

```python
def align_series(
    series_a: pd.Series, 
    series_b: pd.Series, 
    method: str = "inner"
) -> pd.DataFrame
```

Align two time series based on their dates.

**Parameters**:
- `series_a` (pd.Series): First time series
- `series_b` (pd.Series): Second time series  
- `method` (str): Alignment method ("inner", "outer", "left", "right")

**Returns**: DataFrame with aligned series.

**Raises**: `ValueError`: No overlapping dates or invalid series.

##### download_and_align_pair()

```python
def download_and_align_pair(
    fx_symbol: str,
    comd_symbol: str,
    start: str,
    end: str,
    fx_name: Optional[str] = None,
    comd_name: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> pd.DataFrame
```

Download and align FX and commodity data for a pair.

**Parameters**:
- `fx_symbol` (str): FX symbol (e.g., "USDCAD=X")
- `comd_symbol` (str): Commodity symbol (e.g., "CL=F")
- `start` (str): Start date in "YYYY-MM-DD" format
- `end` (str): End date in "YYYY-MM-DD" format
- `fx_name` (Optional[str]): Name for FX series (defaults to symbol)
- `comd_name` (Optional[str]): Name for commodity series (defaults to symbol)
- `max_retries` (int): Maximum retry attempts
- `retry_delay` (float): Delay between retries

**Returns**: DataFrame with aligned FX and commodity data.

**Raises**: 
- `ValueError`: Invalid inputs or alignment failure
- `ConnectionError`: Data source connection failure

### EIA Data Loader

**Module**: `src.data.eia_api`

#### EIADataFetcher Class

```python
class EIADataFetcher:
    """Fetcher for EIA (Energy Information Administration) data."""
```

##### Constructor

```python
def __init__(self, api_key: Optional[str] = None) -> None
```

**Parameters**:
- `api_key` (Optional[str]): EIA API key. If None, uses environment variable.

##### Methods

###### get_event_blackouts()

```python
def get_event_blackouts(self, start_date: str, end_date: str) -> pd.DataFrame
```

Get event blackout dates for trading restrictions.

**Parameters**:
- `start_date` (str): Start date in "YYYY-MM-DD" format
- `end_date` (str): End date in "YYYY-MM-DD" format

**Returns**: DataFrame with blackout dates and event details.

###### get_energy_data()

```python
def get_energy_data(self, commodity: str, start_date: str, end_date: str) -> pd.DataFrame
```

Get energy commodity price data from EIA.

**Parameters**:
- `commodity` (str): Commodity identifier
- `start_date` (str): Start date
- `end_date` (str): End date

**Returns**: DataFrame with energy price data.

## Feature Engineering APIs

### Technical Indicators

**Module**: `src.features.indicators`

#### Functions

##### zscore()

```python
def zscore(series: pd.Series, window: int = 20) -> pd.Series
```

Calculate z-score of a series over a rolling window.

**Parameters**:
- `series` (pd.Series): Input time series
- `window` (int): Rolling window size (default: 20)

**Returns**: Series with z-scores.

**Raises**: `ValueError`: Invalid window or series length.

##### zscore_robust()

```python
def zscore_robust(s: pd.Series, window: int) -> pd.Series
```

Calculate robust z-score using median and MAD.

**Parameters**:
- `s` (pd.Series): Input time series
- `window` (int): Rolling window size

**Returns**: Series with robust z-scores.

**Raises**: `ValueError`: Invalid window or series length.

##### atr_proxy()

```python
def atr_proxy(close: pd.Series, window: int = 14) -> pd.Series
```

Calculate ATR proxy using only close prices.

**Parameters**:
- `close` (pd.Series): Close price series
- `window` (int): Rolling window size (default: 14)

**Returns**: Series with ATR proxy values.

##### rolling_corr()

```python
def rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int = 20) -> pd.Series
```

Calculate rolling correlation between two series.

**Parameters**:
- `series_a` (pd.Series): First time series
- `series_b` (pd.Series): Second time series
- `window` (int): Rolling window size (default: 20)

**Returns**: Series with rolling correlation values.

**Raises**: `ValueError`: Invalid window or mismatched series lengths.

### Spread Calculation

**Module**: `src.features.spread`

#### Functions

##### compute_spread()

```python
def compute_spread(
    y: pd.Series, 
    x: pd.Series, 
    beta_window: int, 
    use_kalman: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]
```

Compute spread between two time series using dynamic hedge ratio.

**Parameters**:
- `y` (pd.Series): Dependent variable (e.g., FX rate)
- `x` (pd.Series): Independent variable (e.g., commodity price)
- `beta_window` (int): Window for hedge ratio calculation
- `use_kalman` (bool): Use Kalman filter if True, OLS if False (default: True)

**Returns**: Tuple of (spread, alpha, beta) series.

**Raises**: `ValueError`: Invalid inputs or calculation failure.

##### rls_beta()

```python
def rls_beta(
    y: pd.Series, 
    x: pd.Series, 
    lam: float = 0.99, 
    delta: float = 1000.0
) -> Tuple[pd.Series, pd.Series]
```

Recursive least squares with forgetting factor.

**Parameters**:
- `y` (pd.Series): Dependent variable
- `x` (pd.Series): Independent variable  
- `lam` (float): Forgetting factor (default: 0.99)
- `delta` (float): Initial state uncertainty (default: 1000.0)

**Returns**: Tuple of (alpha, beta) coefficient series.

### Regime Detection

**Module**: `src.features.regime`

#### Functions

##### correlation_gate()

```python
def correlation_gate(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
    min_abs_corr: float
) -> pd.Series
```

Create correlation-based regime filter.

**Parameters**:
- `series_a` (pd.Series): First time series
- `series_b` (pd.Series): Second time series
- `window` (int): Correlation calculation window
- `min_abs_corr` (float): Minimum absolute correlation threshold

**Returns**: Boolean series indicating valid correlation regime.

##### volatility_regime()

```python
def volatility_regime(
    series: pd.Series,
    window: int = 20,
    high_vol_threshold: float = 0.02,
    low_vol_threshold: float = 0.005
) -> pd.Series
```

Classify volatility regimes.

**Parameters**:
- `series` (pd.Series): Price series
- `window` (int): Volatility calculation window (default: 20)
- `high_vol_threshold` (float): High volatility threshold (default: 0.02)
- `low_vol_threshold` (float): Low volatility threshold (default: 0.005)

**Returns**: Series with volatility regime classifications.

##### combined_regime_filter()

```python
def combined_regime_filter(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict[str, Any]
) -> pd.Series
```

Create combined regime filter using multiple indicators.

**Parameters**:
- `fx_series` (pd.Series): FX price series
- `comd_series` (pd.Series): Commodity price series  
- `config` (Dict): Configuration dictionary with regime parameters

**Returns**: Boolean series indicating favorable trading regime.

### Cointegration Analysis

**Module**: `src.features.cointegration`

#### Functions

##### adf_pvalue()

```python
def adf_pvalue(series: pd.Series, max_lag: int = 1) -> float
```

Calculate p-value of Augmented Dickey-Fuller test for stationarity.

**Parameters**:
- `series` (pd.Series): Time series to test
- `max_lag` (int): Maximum lag for ADF test (default: 1)

**Returns**: ADF test p-value.

##### ou_half_life()

```python
def ou_half_life(spread: pd.Series, cap: float = 100.0) -> float
```

Calculate Ornstein-Uhlenbeck half-life for mean reversion speed.

**Parameters**:
- `spread` (pd.Series): Spread time series
- `cap` (float): Maximum half-life cap (default: 100.0)

**Returns**: Half-life in periods.

##### is_cointegrated()

```python
def is_cointegrated(
    y: pd.Series,
    x: pd.Series,
    significance_level: float = 0.05,
    max_lag: int = 1
) -> Tuple[bool, float, float]
```

Test if two series are cointegrated.

**Parameters**:
- `y` (pd.Series): Dependent variable
- `x` (pd.Series): Independent variable
- `significance_level` (float): Significance level (default: 0.05)
- `max_lag` (int): Maximum lag for tests (default: 1)

**Returns**: Tuple of (is_cointegrated, adf_pvalue, half_life).

## Machine Learning APIs

### Ensemble Models

**Module**: `src.ml.ensemble`

#### ModelConfig Class

```python
@dataclass
class ModelConfig:
    """Configuration for ensemble models."""
    ols_window: int = 90
    kalman_lambda: float = 0.995
    kalman_delta: float = 100.0
    corr_window: int = 20
    gb_n_estimators: int = 100
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.1
    lstm_hidden_size: int = 50
    lstm_num_layers: int = 2
    lstm_epochs: int = 50
    lstm_lr: float = 0.001
    model_weights: Dict[str, float] = None
```

#### EnsembleModel Class

```python
class EnsembleModel:
    """Ensemble model combining multiple prediction models."""
```

##### Constructor

```python
def __init__(self, config: ModelConfig = None) -> None
```

**Parameters**:
- `config` (ModelConfig): Model configuration. Uses default if None.

##### Methods

###### fit()

```python
def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
```

Fit all models in the ensemble.

**Parameters**:
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable

**Returns**: Dictionary of training scores by model.

###### predict()

```python
def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]
```

Make predictions using all models.

**Parameters**:
- `X` (pd.DataFrame): Feature matrix

**Returns**: Dictionary of predictions by model.

###### predict_ensemble()

```python
def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray
```

Make ensemble prediction using weighted average.

**Parameters**:
- `X` (pd.DataFrame): Feature matrix

**Returns**: Ensemble predictions array.

#### Functions

##### create_ensemble_model()

```python
def create_ensemble_model(config: ModelConfig = None) -> EnsembleModel
```

Create ensemble model with configuration.

**Parameters**:
- `config` (ModelConfig): Model configuration

**Returns**: Configured ensemble model instance.

## Strategy APIs

### Mean Reversion Strategy

**Module**: `src.strategy.mean_reversion`

#### Functions

##### generate_signals()

```python
def generate_signals(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict,
    regime_filter: Optional[pd.Series] = None,
    model_name: str = "default"
) -> pd.DataFrame
```

Generate trading signals for mean reversion strategy.

**Parameters**:
- `fx_series` (pd.Series): FX time series
- `comd_series` (pd.Series): Commodity time series
- `config` (Dict): Configuration dictionary
- `regime_filter` (Optional[pd.Series]): Boolean series for regime filtering
- `model_name` (str): Model name for signal generation (default: "default")

**Returns**: DataFrame with signals and related metrics.

**Raises**: `ValueError`: Missing config parameters or invalid series.

##### calculate_position_sizes()

```python
def calculate_position_sizes(
    signals_df: pd.DataFrame,
    atr_window: int,
    target_vol_per_leg: float,
    inverse_fx_for_quote_ccy_strength: bool
) -> pd.DataFrame
```

Calculate position sizes based on inverse volatility sizing.

**Parameters**:
- `signals_df` (pd.DataFrame): DataFrame with signals and prices
- `atr_window` (int): Window for ATR calculation
- `target_vol_per_leg` (float): Target volatility per leg
- `inverse_fx_for_quote_ccy_strength` (bool): Whether to inverse FX for quote currency strength

**Returns**: DataFrame with position sizes.

##### apply_time_stop()

```python
def apply_time_stop(signals_df: pd.DataFrame, max_days: int) -> pd.DataFrame
```

Apply time-based stop to positions.

**Parameters**:
- `signals_df` (pd.DataFrame): DataFrame with signals
- `max_days` (int): Maximum days to hold position

**Returns**: DataFrame with time-stop applied.

##### generate_signals_with_regime_filter()

```python
def generate_signals_with_regime_filter(
    fx_series: pd.Series,
    comd_series: pd.Series,
    config: Dict
) -> pd.DataFrame
```

Generate signals with regime filtering applied.

**Parameters**:
- `fx_series` (pd.Series): FX time series
- `comd_series` (pd.Series): Commodity time series
- `config` (Dict): Configuration dictionary

**Returns**: DataFrame with regime-filtered signals.

## Risk Management APIs

### Risk Manager

**Module**: `src.risk.manager`

#### RiskConfig Class

```python
@dataclass
class RiskConfig:
    """Configuration for risk management parameters."""
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.02
    weekly_loss_limit: float = 0.05
    daily_drawdown_limit: float = 0.02
    max_position_size_per_pair: float = 0.10
    max_total_exposure: float = 0.50
    volatility_scaling: bool = True
    max_trade_risk: float = 0.01
    enable_circuit_breaker: bool = True
    circuit_breaker_cooldown: int = 1
```

#### RiskManager Class

```python
class RiskManager:
    """Risk manager for trading strategies."""
```

##### Constructor

```python
def __init__(self, config: RiskConfig) -> None
```

**Parameters**:
- `config` (RiskConfig): Risk configuration parameters

##### Methods

###### update_account_state()

```python
def update_account_state(
    self,
    equity: float,
    current_date: datetime,
    pair_pnl: Dict[str, float] = None
) -> None
```

Update account state with current equity and PnL.

**Parameters**:
- `equity` (float): Current account equity
- `current_date` (datetime): Current date
- `pair_pnl` (Dict[str, float]): Dictionary of PnL by pair

###### check_drawdown_limit()

```python
def check_drawdown_limit(self, equity_series: pd.Series) -> bool
```

Check if drawdown limit has been breached.

**Parameters**:
- `equity_series` (pd.Series): Series of equity values

**Returns**: True if drawdown limit breached, False otherwise.

###### calculate_position_size()

```python
def calculate_position_size(
    self,
    pair_name: str,
    signal: int,
    fx_price: float,
    comd_price: float,
    fx_vol: float,
    comd_vol: float,
    stop_loss_distance: float,
    fx_atr: float = None,
    comd_atr: float = None,
    target_vol_per_leg: float = 0.01
) -> Tuple[float, float]
```

Calculate position sizes with risk management.

**Parameters**:
- `pair_name` (str): Name of trading pair
- `signal` (int): Trading signal (-1, 0, 1)
- `fx_price` (float): Current FX price
- `comd_price` (float): Current commodity price
- `fx_vol` (float): FX volatility
- `comd_vol` (float): Commodity volatility
- `stop_loss_distance` (float): Stop loss distance in price terms
- `fx_atr` (float): ATR for FX (optional)
- `comd_atr` (float): ATR for commodity (optional)
- `target_vol_per_leg` (float): Target volatility per leg

**Returns**: Tuple of (fx_position_size, comd_position_size).

###### can_trade_pair()

```python
def can_trade_pair(self, pair_name: str, current_date: datetime) -> bool
```

Check if trading is allowed for a specific pair.

**Parameters**:
- `pair_name` (str): Name of trading pair
- `current_date` (datetime): Current date

**Returns**: True if trading allowed, False otherwise.

#### Functions

##### create_risk_manager()

```python
def create_risk_manager(config: RiskConfig = None) -> RiskManager
```

Create risk manager with default or provided configuration.

**Parameters**:
- `config` (RiskConfig): Risk configuration. Uses default if None.

**Returns**: Risk manager instance.

## Execution APIs

### Execution Policy

**Module**: `src.exec.policy`

#### ExecutionConfig Class

```python
@dataclass
class ExecutionConfig:
    """Configuration for execution parameters."""
    fx_slippage_bps: float = 1.0
    comd_slippage_bps: float = 2.0
    default_order_type: str = "limit"
    market_impact_coefficient: float = 0.1
    max_position_impact: float = 0.05
    fx_venue_spread_bps: float = 0.5
    comd_venue_spread_bps: float = 1.5
    fx_fixed_cost: float = 0.0001
    comd_fixed_cost: float = 1.0
    fx_percentage_cost: float = 0.00001
    comd_percentage_cost: float = 0.00001
    atr_slippage_multiplier: float = 0.5
    volume_slippage_multiplier: float = 0.3
```

#### ExecutionPolicy Class

```python
class ExecutionPolicy:
    """Execution policy for trading strategies."""
```

##### Constructor

```python
def __init__(self, config: ExecutionConfig) -> None
```

**Parameters**:
- `config` (ExecutionConfig): Execution configuration

##### Methods

###### calculate_slippage()

```python
def calculate_slippage(
    self,
    fx_price: float,
    comd_price: float,
    fx_position: float,
    comd_position: float,
    fx_atr: float = None,
    comd_atr: float = None,
    fx_volume: float = None,
    comd_volume: float = None,
    order_type: str = None
) -> Tuple[float, float]
```

Calculate slippage for FX and commodity positions.

**Parameters**:
- `fx_price` (float): Current FX price
- `comd_price` (float): Current commodity price
- `fx_position` (float): FX position size
- `comd_position` (float): Commodity position size
- `fx_atr` (float): ATR for FX (optional)
- `comd_atr` (float): ATR for commodity (optional)
- `fx_volume` (float): Volume for FX (optional)
- `comd_volume` (float): Volume for commodity (optional)
- `order_type` (str): Order type (optional)

**Returns**: Tuple of (fx_slippage, comd_slippage) in absolute terms.

###### calculate_execution_costs()

```python
def calculate_execution_costs(
    self,
    fx_price: float,
    comd_price: float,
    fx_position: float,
    comd_position: float,
    order_type: str = None
) -> float
```

Calculate total execution costs.

**Parameters**:
- `fx_price` (float): Current FX price
- `comd_price` (float): Current commodity price
- `fx_position` (float): FX position size
- `comd_position` (float): Commodity position size
- `order_type` (str): Order type (optional)

**Returns**: Total execution costs.

#### Functions

##### create_execution_policy()

```python
def create_execution_policy(config: ExecutionConfig = None) -> ExecutionPolicy
```

Create execution policy with configuration.

**Parameters**:
- `config` (ExecutionConfig): Execution configuration

**Returns**: Execution policy instance.

## Backtesting APIs

### Backtest Engine

**Module**: `src.backtest.engine`

#### Functions

##### backtest_pair()

```python
def backtest_pair(
    df: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_bars: int,
    inverse_fx_for_quote_ccy_strength: bool
) -> pd.DataFrame
```

Backtest a single FX-Commodity pair with one-bar execution delay.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with signals and market data
- `entry_z` (float): Z-score threshold for entry
- `exit_z` (float): Z-score threshold for exit
- `stop_z` (float): Z-score threshold for stop loss
- `max_bars` (int): Maximum bars to hold position
- `inverse_fx_for_quote_ccy_strength` (bool): Whether to inverse FX for quote currency

**Returns**: DataFrame with backtest results and performance metrics.

**Raises**: `ValueError`: Missing required columns or invalid parameters.

##### calculate_performance_metrics()

```python
def calculate_performance_metrics(
    df: pd.DataFrame, 
    min_trade_count: int = 10
) -> Dict
```

Calculate comprehensive performance metrics.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with backtest results
- `min_trade_count` (int): Minimum trades for meaningful metrics (default: 10)

**Returns**: Dictionary with performance metrics.

##### run_backtest()

```python
def run_backtest(
    signals_df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, Dict]
```

Run complete backtest with configuration parameters.

**Parameters**:
- `signals_df` (pd.DataFrame): DataFrame with signals and market data
- `config` (Dict): Configuration dictionary

**Returns**: Tuple of (backtest_results_df, performance_metrics_dict).

##### create_backtest_report()

```python
def create_backtest_report(df: pd.DataFrame, metrics: Dict) -> str
```

Create human-readable backtest report.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with backtest results
- `metrics` (Dict): Dictionary with performance metrics

**Returns**: Formatted backtest report string.

### Parallel Backtesting

**Module**: `src.backtest.parallel`

#### BacktestConfig Class

```python
@dataclass
class BacktestConfig:
    """Configuration for parallel backtesting."""
    max_workers: int = 4
    timeout: int = 300
    chunk_size: int = 1
    models_to_test: List[str] = None
    param_ranges: Dict[str, Tuple[float, float, int]] = None
```

#### ParallelBacktester Class

```python
class ParallelBacktester:
    """Parallel backtesting framework for model comparison."""
```

##### Constructor

```python
def __init__(self, config: BacktestConfig = None) -> None
```

**Parameters**:
- `config` (BacktestConfig): Backtesting configuration

##### Methods

###### run_model_comparison()

```python
def run_model_comparison(
    self,
    fx_series: pd.Series,
    comd_series: pd.Series,
    base_config: Dict,
    models: List[str] = None
) -> pd.DataFrame
```

Run parallel comparison of multiple models.

**Parameters**:
- `fx_series` (pd.Series): FX time series
- `comd_series` (pd.Series): Commodity time series
- `base_config` (Dict): Base configuration for backtests
- `models` (List[str]): List of models to compare

**Returns**: DataFrame with comparison results.

###### run_parameter_sweep()

```python
def run_parameter_sweep(
    self,
    fx_series: pd.Series,
    comd_series: pd.Series,
    base_config: Dict
) -> pd.DataFrame
```

Run parameter sweep optimization.

**Parameters**:
- `fx_series` (pd.Series): FX time series
- `comd_series` (pd.Series): Commodity time series
- `base_config` (Dict): Base configuration

**Returns**: DataFrame with parameter sweep results.

### Rolling Metrics

**Module**: `src.backtest.rolling_metrics`

#### Functions

##### calculate_rolling_metrics()

```python
def calculate_rolling_metrics(
    equity: pd.Series,
    windows: Dict[str, int]
) -> Dict[str, pd.Series]
```

Calculate rolling performance metrics.

**Parameters**:
- `equity` (pd.Series): Equity curve series
- `windows` (Dict[str, int]): Dictionary of window names and sizes

**Returns**: Dictionary of rolling metric series.

##### calculate_rolling_sharpe()

```python
def calculate_rolling_sharpe(
    equity: pd.Series,
    window: int,
    ann_factor: int = 252
) -> pd.Series
```

Calculate rolling Sharpe ratio.

**Parameters**:
- `equity` (pd.Series): Equity curve series
- `window` (int): Rolling window size
- `ann_factor` (int): Annualization factor (default: 252)

**Returns**: Series with rolling Sharpe ratios.

## Utility APIs

### Logging Utilities

**Module**: `src.utils.logging`

#### Functions

##### setup_logging()

```python
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None
```

Set up logging configuration for the application.

**Parameters**:
- `log_level` (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
- `log_file` (Optional[Path]): Path to log file. If None, logs only to console
- `log_format` (Optional[str]): Custom log format
- `rotation` (str): Log file rotation setting (default: "10 MB")
- `retention` (str): Log file retention setting (default: "30 days")

##### get_logger()

```python
def get_logger(name: str)
```

Get a logger instance for a specific module.

**Parameters**:
- `name` (str): Name of the module (usually `__name__`)

**Returns**: Logger instance.

#### TradingLogger Class

```python
class TradingLogger:
    """Specialized logger for trading operations."""
```

##### Constructor

```python
def __init__(self, name: str = "Trading") -> None
```

**Parameters**:
- `name` (str): Name for the logger (default: "Trading")

##### Methods

###### log_signal()

```python
def log_signal(self, signal_type: str, details: dict) -> None
```

Log a trading signal.

**Parameters**:
- `signal_type` (str): Type of signal ("entry", "exit", "stop_loss")
- `details` (dict): Dictionary with signal details

###### log_trade()

```python
def log_trade(self, trade_id: str, action: str, details: dict) -> None
```

Log a trade action.

**Parameters**:
- `trade_id` (str): Unique trade identifier
- `action` (str): Trade action ("open", "close", "modify")
- `details` (dict): Dictionary with trade details

###### log_performance()

```python
def log_performance(self, metrics: dict) -> None
```

Log performance metrics.

**Parameters**:
- `metrics` (dict): Dictionary with performance metrics

## Error Handling

All APIs follow consistent error handling patterns:

### Common Exceptions

- `ValueError`: Invalid input parameters or data
- `TypeError`: Incorrect input types
- `FileNotFoundError`: Missing configuration or data files
- `ConnectionError`: Network or data source connectivity issues
- `RuntimeError`: Unexpected runtime errors

### Error Response Format

All functions log errors using the structured logging system and provide detailed error messages with context for debugging.

## Versioning and Compatibility

### API Versioning

The API follows semantic versioning principles:
- **Major Version**: Breaking changes to public APIs
- **Minor Version**: New features with backward compatibility
- **Patch Version**: Bug fixes with backward compatibility

### Backward Compatibility

All public APIs documented here maintain backward compatibility within major versions. Deprecated APIs will be marked and supported for at least one major version before removal.

### Usage Examples

For practical usage examples of these APIs, refer to:
- `src/run_backtest.py` - Complete system usage example
- Individual test files in `tests/` directory
- Example notebooks in `notebooks/` directory

## Support and Documentation

For additional support:
- Check the system architecture documentation for module relationships
- Review coding standards for implementation guidelines
- See configuration guide for parameter details
- Consult performance tuning guide for optimization tips