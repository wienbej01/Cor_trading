# FX-Commodity Correlation Arbitrage - Coding Standards and Conventions

## Overview

This document establishes coding standards and conventions for the FX-Commodity Correlation Arbitrage trading system. These standards ensure consistency, maintainability, and quality across the entire codebase.

## General Principles

### 1. Code Quality Principles
- **Readability**: Code should be self-documenting and easy to understand
- **Consistency**: Follow established patterns throughout the codebase
- **Modularity**: Functions should have single responsibilities
- **Testability**: Code should be designed for easy testing
- **Performance**: Optimize for clarity first, performance second

### 2. Python Version
- **Target Version**: Python 3.11+
- **Compatibility**: Maintain compatibility with Python 3.11 minimum
- **Features**: Leverage modern Python features appropriately

## Naming Conventions

### 1. Variables and Functions
```python
# ✅ Good - Snake case for variables and functions
def calculate_z_score(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std

# ❌ Bad - Mixed case styles
def calculateZScore(Series, Window):
    rollingMean = Series.rolling(Window).mean()
```

### 2. Constants
```python
# ✅ Good - All caps with underscores
DEFAULT_WINDOW_SIZE = 20
MAX_RETRY_ATTEMPTS = 3
API_BASE_URL = "https://api.example.com"

# ❌ Bad - Mixed case
defaultWindowSize = 20
maxRetryAttempts = 3
```

### 3. Classes
```python
# ✅ Good - PascalCase for classes
class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

class SignalOptimizer:
    pass

# ❌ Bad - Snake case for classes
class risk_manager:
    pass
```

### 4. Private Methods and Variables
```python
class DataProcessor:
    def __init__(self):
        self._internal_state = {}  # ✅ Single underscore for internal use
        self.__private_data = {}   # ✅ Double underscore for name mangling
    
    def _process_data(self, data):  # ✅ Internal method
        pass
    
    def __validate_input(self, data):  # ✅ Private method
        pass
```

### 5. Module and Package Names
```python
# ✅ Good - Short, lowercase names
import data.yahoo_loader
from features.indicators import zscore
from risk.manager import RiskManager

# ❌ Bad - Mixed case or overly long names
import Data.YahooLoader
from FeaturesAndIndicators.TechnicalIndicators import zScore
```

## Function Design Standards

### 1. Function Signatures
```python
# ✅ Standard pattern for data loading functions
def download_daily(
    symbol: str, 
    start: str, 
    end: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> pd.Series:
    """Download daily price data with retry logic."""
    pass

# ✅ Standard pattern for feature calculation functions  
def calculate_indicator(
    series: pd.Series, 
    window: int, 
    **kwargs
) -> pd.Series:
    """Calculate technical indicator with configurable parameters."""
    pass

# ✅ Standard pattern for signal generation functions
def generate_signals(
    fx_series: pd.Series,
    comd_series: pd.Series, 
    config: Dict,
    regime_filter: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Generate trading signals based on input data and configuration."""
    pass
```

### 2. Input Validation Pattern
```python
def calculate_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate z-score with standardized validation.
    
    Args:
        series: Input time series
        window: Rolling window size
        
    Returns:
        Series with z-scores
        
    Raises:
        ValueError: If window is invalid or series is empty
        TypeError: If inputs are not correct types
    """
    # ✅ Validate inputs at function start
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(series)}")
    
    if window < 2:
        raise ValueError(f"Window must be at least 2, got {window}")
    
    if window > len(series):
        raise ValueError(f"Window ({window}) cannot exceed series length ({len(series)})")
    
    if series.empty:
        raise ValueError("Cannot calculate z-score on empty series")
    
    # Function implementation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)
```

### 3. Error Handling Pattern
```python
from loguru import logger

def robust_calculation(data: pd.Series, window: int) -> pd.Series:
    """Standard error handling pattern."""
    logger.debug(f"Starting calculation with {len(data)} data points, window={window}")
    
    try:
        # Input validation
        if data.empty:
            raise ValueError("Empty data series provided")
        
        # Main calculation
        result = data.rolling(window).mean()
        
        # Validate output
        if result.isna().all():
            logger.warning("Calculation produced all NaN values")
        
        logger.debug(f"Calculation completed successfully, {result.dropna().size} valid results")
        return result
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        # Re-raise with context
        raise ValueError(f"Failed to calculate indicator: {e}") from e
```

## Type Hints Standards

### 1. Required Type Hints
```python
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# ✅ All public functions must have type hints
def process_data(
    data: pd.DataFrame,
    config: Dict[str, Any],
    symbols: List[str],
    start_date: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process data with complete type annotations."""
    pass

# ✅ Class attributes should be typed
@dataclass
class TradingConfig:
    entry_threshold: float
    exit_threshold: float
    max_position_size: float
    symbols: List[str] = field(default_factory=list)
```

### 2. Complex Type Definitions
```python
from typing import TypeAlias, Protocol

# ✅ Use TypeAlias for complex types
ConfigDict: TypeAlias = Dict[str, Union[str, float, int, Dict[str, Any]]]
SignalData: TypeAlias = Tuple[pd.DataFrame, Dict[str, float]]

# ✅ Use Protocol for structural typing
class DataLoader(Protocol):
    def load_data(self, symbol: str, start: str, end: str) -> pd.Series:
        """Data loader protocol."""
        ...
```

## Documentation Standards

### 1. Module Docstrings
```python
"""
Module docstring template.

Brief description of module purpose and functionality.
Includes information about main classes, functions, and usage patterns.

Example:
    Basic usage example::

        from module import main_function
        result = main_function(data, config)

Attributes:
    module_attribute: Description of module-level attributes.

Todo:
    * Known limitations or future improvements
"""
```

### 2. Function Docstrings
```python
def calculate_spread(
    y: pd.Series, 
    x: pd.Series, 
    beta_window: int, 
    use_kalman: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate spread between two time series using dynamic hedge ratio.
    
    This function estimates the relationship between two time series and computes
    the spread using either Kalman filter or rolling OLS regression.
    
    Args:
        y: Dependent variable time series (e.g., FX rate)
        x: Independent variable time series (e.g., commodity price)  
        beta_window: Rolling window size for hedge ratio calculation
        use_kalman: If True, use Kalman filter; otherwise use OLS
        
    Returns:
        Tuple containing:
            - spread: Calculated spread series
            - alpha: Intercept coefficient series
            - beta: Slope coefficient series
            
    Raises:
        ValueError: If series lengths don't match or window is invalid
        RuntimeError: If calculation fails due to numerical issues
        
    Example:
        >>> fx_data = pd.Series([1.0, 1.1, 1.2], name='USDCAD')
        >>> comd_data = pd.Series([50.0, 55.0, 60.0], name='WTI')  
        >>> spread, alpha, beta = calculate_spread(fx_data, comd_data, 20)
        >>> print(f"Spread range: {spread.min():.4f} to {spread.max():.4f}")
        
    Note:
        Function handles missing values by forward-filling. Extreme beta values
        are clipped to [-10, 10] range for numerical stability.
    """
    pass
```

### 3. Class Docstrings
```python
class RiskManager:
    """
    Risk management system for trading strategies.
    
    Implements position sizing, drawdown limits, and circuit breakers to control
    trading risk. Supports both pair-level and portfolio-level risk controls.
    
    Attributes:
        config: Risk configuration parameters
        account_equity: Current account equity value
        circuit_breaker_active: Whether circuit breaker is currently active
        
    Example:
        >>> config = RiskConfig(max_drawdown=0.15)
        >>> risk_mgr = RiskManager(config)
        >>> risk_mgr.update_account_state(100000.0, datetime.now())
        >>> can_trade = risk_mgr.can_trade_pair('USDCAD_WTI', datetime.now())
    """
    
    def __init__(self, config: RiskConfig):
        """Initialize risk manager with configuration."""
        pass
```

## Error Handling Standards

### 1. Exception Hierarchy
```python
# ✅ Custom exceptions for domain-specific errors
class TradingSystemError(Exception):
    """Base exception for trading system errors."""
    pass

class DataLoadError(TradingSystemError):
    """Raised when data loading fails."""
    pass

class SignalGenerationError(TradingSystemError):
    """Raised when signal generation fails.""" 
    pass

class RiskLimitError(TradingSystemError):
    """Raised when risk limits are breached."""
    pass
```

### 2. Error Handling Patterns
```python
def load_market_data(symbol: str, start: str, end: str) -> pd.Series:
    """Standard error handling with context and recovery."""
    logger.info(f"Loading data for {symbol} from {start} to {end}")
    
    try:
        # Main operation
        data = download_daily(symbol, start, end)
        
        # Validation
        if data.empty:
            raise DataLoadError(f"No data returned for {symbol}")
            
        return data
        
    except ConnectionError as e:
        logger.error(f"Network error loading {symbol}: {e}")
        raise DataLoadError(f"Failed to connect to data source: {e}") from e
        
    except ValueError as e:
        logger.error(f"Invalid parameters for {symbol}: {e}")
        raise DataLoadError(f"Invalid request parameters: {e}") from e
        
    except Exception as e:
        logger.error(f"Unexpected error loading {symbol}: {e}")
        raise DataLoadError(f"Unexpected error: {e}") from e
```

## Logging Standards

### 1. Logging Levels
```python
from loguru import logger

def process_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Standard logging level usage."""
    
    # DEBUG: Detailed internal state
    logger.debug(f"Processing {len(data)} data points")
    logger.debug(f"Data columns: {list(data.columns)}")
    
    # INFO: General flow and important events  
    logger.info("Starting signal processing")
    logger.info(f"Generated {signal_count} signals")
    
    # WARNING: Unusual but not error conditions
    logger.warning(f"Low data quality detected: {quality_score:.2f}")
    logger.warning("Using fallback calculation method")
    
    # ERROR: Error conditions that should be investigated
    logger.error(f"Signal generation failed: {error_msg}")
    logger.error("Risk limits breached, stopping processing")
    
    return processed_data
```

### 2. Structured Logging
```python
def log_trading_event(event_type: str, details: Dict[str, Any]) -> None:
    """Structured logging for trading events."""
    logger.info(
        f"Trading event: {event_type}",
        extra={
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **details
        }
    )

# Usage examples
log_trading_event("signal_generated", {
    "pair": "USDCAD_WTI",
    "signal_strength": 2.5,
    "position_size": 0.05
})

log_trading_event("risk_limit_breached", {
    "limit_type": "daily_drawdown",
    "current_value": -0.025,
    "limit_value": -0.02
})
```

## Configuration Standards

### 1. Configuration Structure
```python
# ✅ Use dataclasses for configuration
@dataclass
class BacktestConfig:
    """Backtesting configuration with validation."""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_bps: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission_rate < 0:
            raise ValueError("Commission rate cannot be negative")
```

### 2. Configuration Loading Pattern
```python
def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Standard configuration loading with validation."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required keys
        required_keys = ['fx_symbol', 'comd_symbol', 'lookbacks', 'thresholds']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Type validation
        if not isinstance(config['lookbacks']['beta_window'], int):
            raise TypeError("beta_window must be an integer")
            
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {e}")
        raise ValueError(f"Configuration file format error: {e}") from e
```

## Testing Standards

### 1. Test Structure
```python
import pytest
import pandas as pd
from unittest.mock import Mock, patch

class TestZScoreCalculation:
    """Test class for z-score calculations."""
    
    def setup_method(self):
        """Setup test data before each test."""
        self.sample_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.window = 5
    
    def test_zscore_normal_case(self):
        """Test z-score calculation with normal inputs."""
        result = zscore(self.sample_data, self.window)
        
        # Assertions
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert not result.isna().all()
    
    def test_zscore_invalid_window(self):
        """Test z-score calculation with invalid window."""
        with pytest.raises(ValueError, match="Window must be at least 2"):
            zscore(self.sample_data, window=1)
    
    def test_zscore_empty_series(self):
        """Test z-score calculation with empty series."""
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Cannot calculate.*empty"):
            zscore(empty_series, self.window)
    
    @patch('src.features.indicators.logger')
    def test_zscore_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        zscore(self.sample_data, self.window)
        mock_logger.debug.assert_called()
```

### 2. Fixture Patterns
```python
@pytest.fixture
def sample_price_data():
    """Fixture providing sample price data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    fx_prices = pd.Series(
        np.random.normal(1.0, 0.01, 100), 
        index=dates, 
        name='USDCAD'
    )
    comd_prices = pd.Series(
        np.random.normal(50.0, 2.0, 100), 
        index=dates, 
        name='WTI'
    )
    return fx_prices, comd_prices

@pytest.fixture
def trading_config():
    """Fixture providing standard trading configuration."""
    return {
        'lookbacks': {'beta_window': 60, 'z_window': 20},
        'thresholds': {'entry_z': 2.0, 'exit_z': 1.0},
        'risk': {'max_position_size': 0.1}
    }
```

## Performance Standards

### 1. Vectorization Requirements
```python
# ✅ Good - Vectorized operations
def calculate_returns(prices: pd.Series) -> pd.Series:
    """Use pandas vectorized operations."""
    return prices.pct_change()

def calculate_rolling_stats(data: pd.Series, window: int) -> pd.DataFrame:
    """Efficient rolling calculations."""
    return pd.DataFrame({
        'mean': data.rolling(window).mean(),
        'std': data.rolling(window).std(),
        'min': data.rolling(window).min(),
        'max': data.rolling(window).max()
    })

# ❌ Bad - Loops where vectorization possible
def calculate_returns_slow(prices: pd.Series) -> pd.Series:
    """Inefficient loop-based approach."""
    returns = []
    for i in range(1, len(prices)):
        returns.append((prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1])
    return pd.Series(returns, index=prices.index[1:])
```

### 2. Memory Efficiency
```python
def process_large_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Memory-efficient data processing."""
    # ✅ Process in chunks for large datasets
    chunk_size = 10000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    # ✅ Downcast numeric types where appropriate
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df
```

## Import Standards

### 1. Import Organization
```python
# ✅ Standard import order
# Standard library imports
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from loguru import logger
import yfinance as yf

# Local application imports
from src.core.config import ConfigManager
from src.features.indicators import zscore, atr_proxy
from src.utils.logging import setup_logging
```

### 2. Import Patterns
```python
# ✅ Specific imports for better readability
from src.features.indicators import (
    zscore,
    zscore_robust, 
    atr_proxy,
    rolling_corr
)

# ✅ Aliasing for commonly used modules
import pandas as pd
import numpy as np
from loguru import logger

# ❌ Avoid wildcard imports
from src.features.indicators import *  # Don't do this
```

## Code Organization Standards

### 1. Module Structure
```python
"""Module docstring."""

# Imports (organized as per import standards)

# Constants
DEFAULT_WINDOW = 20
MAX_ITERATIONS = 1000

# Type definitions
ConfigType = Dict[str, Union[str, float, int]]

# Utility functions (private functions starting with _)
def _validate_input(data: pd.Series) -> None:
    """Private utility function."""
    pass

# Public functions (main module functionality)
def main_function(data: pd.Series, config: ConfigType) -> pd.DataFrame:
    """Main public function."""
    pass

# Classes (if any)
class MainClass:
    """Main class in module."""
    pass

# Module-level execution (if needed)
if __name__ == "__main__":
    # Testing or example code
    pass
```

### 2. Function Ordering
```python
# 1. Input validation functions
def validate_inputs(data: pd.Series, window: int) -> None:
    pass

# 2. Core computation functions
def compute_indicator(data: pd.Series, window: int) -> pd.Series:
    pass

# 3. Helper/utility functions
def _format_output(result: pd.Series) -> pd.Series:
    pass

# 4. Main interface functions
def calculate_technical_indicator(
    data: pd.Series, 
    config: Dict
) -> pd.Series:
    """Main public interface function."""
    pass
```

## Quality Assurance Checklist

### Pre-Commit Checklist
- [ ] All functions have type hints
- [ ] All public functions have comprehensive docstrings
- [ ] Input validation is implemented
- [ ] Error handling follows standard patterns
- [ ] Logging is appropriate for function importance
- [ ] No hardcoded values (use constants or config)
- [ ] Performance considerations addressed
- [ ] Tests written for new functionality

### Code Review Checklist
- [ ] Naming conventions followed
- [ ] Function responsibilities are single and clear
- [ ] Error messages are informative
- [ ] Documentation is accurate and complete
- [ ] Performance implications considered
- [ ] Security implications considered (if applicable)
- [ ] Backward compatibility maintained

## Tools and Enforcement

### 1. Recommended Tools
- **Linting**: `flake8`, `pylint`
- **Formatting**: `black`
- **Type Checking**: `mypy`
- **Import Sorting**: `isort`
- **Documentation**: `pydocstyle`

### 2. Configuration Examples
```ini
# setup.cfg for flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503
per-file-ignores = __init__.py:F401

# pyproject.toml for black
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

# pyproject.toml for isort
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
```

This coding standards document provides comprehensive guidelines for maintaining consistency and quality across the FX-Commodity Correlation Arbitrage system. Regular review and updates of these standards ensure they remain relevant and useful as the codebase evolves.