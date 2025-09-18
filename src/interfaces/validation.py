"""
Input validation interface for consistent parameter validation across modules.
Provides standardized validation patterns and error handling.
"""

from typing import Dict, List, Any, Callable, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from loguru import logger


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, parameter: str = None, value: Any = None):
        self.message = message
        self.parameter = parameter
        self.value = value
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with context."""
        if self.parameter and self.value is not None:
            return f"Validation error for parameter '{self.parameter}' (value: {self.value}): {self.message}"
        elif self.parameter:
            return f"Validation error for parameter '{self.parameter}': {self.message}"
        else:
            return f"Validation error: {self.message}"


class ParameterValidator(ABC):
    """Abstract base class for parameter validators."""

    @abstractmethod
    def validate(self, value: Any, context: Dict = None) -> None:
        """
        Validate a parameter value.

        Args:
            value: Value to validate.
            context: Optional context for validation.

        Raises:
            ValidationError: If validation fails.
        """
        pass


class NumericRangeValidator(ParameterValidator):
    """Validates numeric values within a specified range."""

    def __init__(
        self,
        min_val: float = None,
        max_val: float = None,
        allow_none: bool = False,
        parameter_name: str = None,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.allow_none = allow_none
        self.parameter_name = parameter_name

    def validate(self, value: Any, context: Dict = None) -> None:
        """Validate numeric range."""
        if value is None and self.allow_none:
            return

        if value is None:
            raise ValidationError("Value cannot be None", self.parameter_name, value)

        if not isinstance(value, (int, float)):
            raise ValidationError("Value must be numeric", self.parameter_name, value)

        if np.isnan(value) or np.isinf(value):
            raise ValidationError(
                "Value cannot be NaN or infinite", self.parameter_name, value
            )

        if self.min_val is not None and value < self.min_val:
            raise ValidationError(
                f"Value must be >= {self.min_val}", self.parameter_name, value
            )

        if self.max_val is not None and value > self.max_val:
            raise ValidationError(
                f"Value must be <= {self.max_val}", self.parameter_name, value
            )


class SeriesValidator(ParameterValidator):
    """Validates pandas Series objects."""

    def __init__(
        self,
        min_length: int = 1,
        allow_na: bool = False,
        numeric_only: bool = True,
        parameter_name: str = None,
    ):
        self.min_length = min_length
        self.allow_na = allow_na
        self.numeric_only = numeric_only
        self.parameter_name = parameter_name

    def validate(self, value: Any, context: Dict = None) -> None:
        """Validate pandas Series."""
        if not isinstance(value, pd.Series):
            raise ValidationError(
                "Value must be a pandas Series", self.parameter_name, type(value)
            )

        if len(value) < self.min_length:
            raise ValidationError(
                f"Series must have at least {self.min_length} elements",
                self.parameter_name,
                len(value),
            )

        if self.numeric_only and not pd.api.types.is_numeric_dtype(value):
            raise ValidationError(
                "Series must contain numeric data", self.parameter_name, value.dtype
            )

        if not self.allow_na and value.isna().any():
            raise ValidationError(
                "Series cannot contain NaN values", self.parameter_name
            )


class ConfigValidator(ParameterValidator):
    """Validates configuration dictionaries."""

    def __init__(
        self, required_keys: List[str], schema: Dict[str, ParameterValidator] = None
    ):
        self.required_keys = required_keys
        self.schema = schema or {}

    def validate(self, value: Any, context: Dict = None) -> None:
        """Validate configuration dictionary."""
        if not isinstance(value, dict):
            raise ValidationError("Value must be a dictionary", "config", type(value))

        # Check required keys
        missing_keys = [key for key in self.required_keys if key not in value]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}", "config")

        # Validate individual parameters using schema
        for key, validator in self.schema.items():
            if key in value:
                try:
                    validator.validate(value[key], context)
                except ValidationError as e:
                    # Re-raise with parameter context
                    raise ValidationError(e.message, f"config.{key}", value[key])


class ValidationRegistry:
    """Registry for managing validation schemas across modules."""

    def __init__(self):
        self._schemas: Dict[str, Dict[str, ParameterValidator]] = {}
        self._register_default_schemas()

    def register_schema(
        self, module_name: str, schema: Dict[str, ParameterValidator]
    ) -> None:
        """Register a validation schema for a module."""
        self._schemas[module_name] = schema
        logger.debug(f"Registered validation schema for {module_name}")

    def get_schema(self, module_name: str) -> Optional[Dict[str, ParameterValidator]]:
        """Get validation schema for a module."""
        return self._schemas.get(module_name)

    def validate_parameters(self, module_name: str, parameters: Dict[str, Any]) -> None:
        """Validate parameters using registered schema."""
        schema = self.get_schema(module_name)
        if not schema:
            logger.warning(f"No validation schema found for module {module_name}")
            return

        for param_name, validator in schema.items():
            if param_name in parameters:
                try:
                    validator.validate(parameters[param_name])
                except ValidationError as e:
                    # Re-raise with module context
                    raise ValidationError(
                        e.message, f"{module_name}.{param_name}", parameters[param_name]
                    )

    def _register_default_schemas(self) -> None:
        """Register default validation schemas for core modules."""

        # Signal generation validation schema
        self.register_schema(
            "signal_generation",
            {
                "fx_series": SeriesValidator(
                    min_length=100, parameter_name="fx_series"
                ),
                "comd_series": SeriesValidator(
                    min_length=100, parameter_name="comd_series"
                ),
                "entry_z": NumericRangeValidator(
                    min_val=0.1, max_val=5.0, parameter_name="entry_z"
                ),
                "exit_z": NumericRangeValidator(
                    min_val=0.0, max_val=2.0, parameter_name="exit_z"
                ),
                "stop_z": NumericRangeValidator(
                    min_val=1.0, max_val=10.0, parameter_name="stop_z"
                ),
                "beta_window": NumericRangeValidator(
                    min_val=10, max_val=500, parameter_name="beta_window"
                ),
                "z_window": NumericRangeValidator(
                    min_val=5, max_val=200, parameter_name="z_window"
                ),
            },
        )

        # Backtest validation schema
        self.register_schema(
            "backtest",
            {
                "signals_df": SeriesValidator(
                    min_length=50, allow_na=True, parameter_name="signals_df"
                ),
                "entry_z": NumericRangeValidator(
                    min_val=0.1, max_val=5.0, parameter_name="entry_z"
                ),
                "exit_z": NumericRangeValidator(
                    min_val=0.0, max_val=2.0, parameter_name="exit_z"
                ),
                "stop_z": NumericRangeValidator(
                    min_val=1.0, max_val=10.0, parameter_name="stop_z"
                ),
                "max_bars": NumericRangeValidator(
                    min_val=1, max_val=1000, parameter_name="max_bars"
                ),
            },
        )

        # Risk management validation schema
        self.register_schema(
            "risk_management",
            {
                "max_position_pct": NumericRangeValidator(
                    min_val=0.001, max_val=1.0, parameter_name="max_position_pct"
                ),
                "max_daily_loss_pct": NumericRangeValidator(
                    min_val=0.001, max_val=0.5, parameter_name="max_daily_loss_pct"
                ),
                "max_drawdown_pct": NumericRangeValidator(
                    min_val=0.01, max_val=0.9, parameter_name="max_drawdown_pct"
                ),
                "position_size": NumericRangeValidator(
                    min_val=0.0, parameter_name="position_size"
                ),
            },
        )

        # Feature calculation validation schema
        self.register_schema(
            "feature_calculation",
            {
                "window": NumericRangeValidator(
                    min_val=2, max_val=1000, parameter_name="window"
                ),
                "atr_window": NumericRangeValidator(
                    min_val=5, max_val=100, parameter_name="atr_window"
                ),
                "corr_window": NumericRangeValidator(
                    min_val=5, max_val=500, parameter_name="corr_window"
                ),
                "min_abs_corr": NumericRangeValidator(
                    min_val=0.0, max_val=1.0, parameter_name="min_abs_corr"
                ),
            },
        )


class ValidationInterface:
    """Interface for comprehensive validation across modules."""

    def __init__(self):
        pass

    def validate_strategy_config(self, config: Dict) -> Dict:
        """Validate strategy configuration."""
        try:
            validate_trading_config(config)
            return {"is_valid": True, "issues": []}
        except ValidationError as e:
            return {"is_valid": False, "issues": [str(e)]}

    def validate_model_config(self, config: Dict) -> Dict:
        """Validate model configuration."""
        # Basic validation for model config
        if not isinstance(config, dict):
            return {"is_valid": False, "issues": ["Config must be a dictionary"]}

        required_keys = ["model_type"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            return {"is_valid": False, "issues": [f"Missing required keys: {missing}"]}

        return {"is_valid": True, "issues": []}

    def validate_execution_config(self, config: Dict) -> Dict:
        """Validate execution configuration."""
        # Basic validation for execution config
        if not isinstance(config, dict):
            return {"is_valid": False, "issues": ["Config must be a dictionary"]}

        return {"is_valid": True, "issues": []}

    def validate_input_data(self, fx_data: pd.Series, comd_data: pd.Series) -> Dict:
        """Validate input data."""
        try:
            validate_series_alignment(fx_data, comd_data)
            return {"is_valid": True, "issues": []}
        except ValidationError as e:
            return {"is_valid": False, "issues": [str(e)]}

    def validate_all_components(self, **kwargs) -> Dict:
        """Validate all components."""
        results = {
            "strategy_config": {"is_valid": True, "issues": []},
            "model_config": {"is_valid": True, "issues": []},
            "execution_config": {"is_valid": True, "issues": []},
            "input_data": {"is_valid": True, "issues": []},
            "overall_valid": True
        }

        if "strategy_config" in kwargs:
            results["strategy_config"] = self.validate_strategy_config(kwargs["strategy_config"])

        if "model_config" in kwargs:
            results["model_config"] = self.validate_model_config(kwargs["model_config"])

        if "execution_config" in kwargs:
            results["execution_config"] = self.validate_execution_config(kwargs["execution_config"])

        if "fx_data" in kwargs and "commodity_data" in kwargs:
            results["input_data"] = self.validate_input_data(kwargs["fx_data"], kwargs["commodity_data"])

        results["overall_valid"] = all(
            r["is_valid"] for r in results.values() if isinstance(r, dict) and "is_valid" in r
        )

        return results


# Global validation registry instance
validation_registry = ValidationRegistry()


def validate_trading_config(config: Dict) -> None:
    """
    Validate complete trading configuration.

    Args:
        config: Trading configuration dictionary.

    Raises:
        ValidationError: If validation fails.
    """
    # Define comprehensive config validation schema
    config_schema = {
        "lookbacks": ConfigValidator(
            required_keys=["beta_window", "z_window", "corr_window"],
            schema={
                "beta_window": NumericRangeValidator(min_val=10, max_val=500),
                "z_window": NumericRangeValidator(min_val=5, max_val=200),
                "corr_window": NumericRangeValidator(min_val=5, max_val=500),
            },
        ),
        "thresholds": ConfigValidator(
            required_keys=["entry_z", "exit_z", "stop_z"],
            schema={
                "entry_z": NumericRangeValidator(min_val=0.1, max_val=5.0),
                "exit_z": NumericRangeValidator(min_val=0.0, max_val=2.0),
                "stop_z": NumericRangeValidator(min_val=1.0, max_val=10.0),
            },
        ),
        "sizing": ConfigValidator(
            required_keys=["target_vol_per_leg", "atr_window"],
            schema={
                "target_vol_per_leg": NumericRangeValidator(min_val=0.001, max_val=0.1),
                "atr_window": NumericRangeValidator(min_val=5, max_val=100),
            },
        ),
        "regime": ConfigValidator(
            required_keys=["min_abs_corr"],
            schema={"min_abs_corr": NumericRangeValidator(min_val=0.0, max_val=1.0)},
        ),
    }

    # Validate main config structure
    main_validator = ConfigValidator(
        required_keys=["lookbacks", "thresholds", "sizing", "regime"],
        schema=config_schema,
    )

    main_validator.validate(config)
    logger.info("Trading configuration validation passed")


def validate_series_alignment(fx_series: pd.Series, comd_series: pd.Series) -> None:
    """
    Validate that two series are properly aligned for trading.

    Args:
        fx_series: FX time series.
        comd_series: Commodity time series.

    Raises:
        ValidationError: If series are not properly aligned.
    """
    # Validate individual series
    fx_validator = SeriesValidator(min_length=100, parameter_name="fx_series")
    comd_validator = SeriesValidator(min_length=100, parameter_name="comd_series")

    fx_validator.validate(fx_series)
    comd_validator.validate(comd_series)

    # Check alignment with tolerance for H1 data
    length_diff = abs(len(fx_series) - len(comd_series))
    max_length = max(len(fx_series), len(comd_series))

    # Allow small differences for H1 data (up to 1% difference)
    if length_diff > 0:
        length_tolerance = max(
            10, int(max_length * 0.01)
        )  # At least 10 points tolerance, or 1% of max length

        if length_diff > length_tolerance:
            raise ValidationError(
                f"Series length mismatch too large: FX={len(fx_series)}, Commodity={len(comd_series)} "
                f"(difference: {length_diff}, tolerance: {length_tolerance})",
                "series_alignment",
            )
        else:
            logger.warning(
                f"Series length mismatch within tolerance: FX={len(fx_series)}, "
                f"Commodity={len(comd_series)} (difference: {length_diff})"
            )

    # Check index alignment (if both have datetime index)
    if isinstance(fx_series.index, pd.DatetimeIndex) and isinstance(
        comd_series.index, pd.DatetimeIndex
    ):
        if not fx_series.index.equals(comd_series.index):
            # Check if there's significant overlap
            common_dates = fx_series.index.intersection(comd_series.index)
            overlap_ratio = len(common_dates) / min(len(fx_series), len(comd_series))

            if overlap_ratio < 0.8:  # Less than 80% overlap
                raise ValidationError(
                    f"Insufficient date overlap between series: {overlap_ratio:.1%}",
                    "series_alignment",
                )

            logger.warning(
                f"Series have {overlap_ratio:.1%} date overlap, consider aligning data"
            )

    logger.debug("Series alignment validation passed")


def safe_parameter_extraction(
    config: Dict, keys: List[str], default_values: Dict = None
) -> Dict:
    """
    Safely extract parameters from configuration with validation.

    Args:
        config: Configuration dictionary.
        keys: List of keys to extract (supports nested keys with dot notation).
        default_values: Default values for missing keys.

    Returns:
        Dictionary with extracted parameters.

    Raises:
        ValidationError: If required parameters are missing.
    """
    extracted = {}
    default_values = default_values or {}

    for key in keys:
        try:
            # Support nested key access with dot notation
            value = config
            for subkey in key.split("."):
                value = value[subkey]
            extracted[key] = value
        except (KeyError, TypeError):
            if key in default_values:
                extracted[key] = default_values[key]
                logger.debug(f"Using default value for {key}: {default_values[key]}")
            else:
                raise ValidationError(
                    f"Required configuration parameter '{key}' is missing", "config"
                )

    return extracted


def with_validation(validator_func: Callable = None, module_name: str = None):
    """
    Decorator to add automatic parameter validation to functions.

    Args:
        validator_func: Custom validation function.
        module_name: Module name for schema-based validation.

    Returns:
        Decorated function with validation.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Apply custom validation if provided
                if validator_func:
                    validator_func(*args, **kwargs)

                # Apply schema-based validation if module name provided
                if module_name:
                    validation_registry.validate_parameters(module_name, kwargs)

                # Execute original function
                return func(*args, **kwargs)

            except ValidationError:
                # Re-raise validation errors as-is
                raise
            except Exception as e:
                # Wrap unexpected errors with context
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise ValidationError(
                    f"Function execution failed: {str(e)}", func.__name__
                )

        return wrapper

    return decorator


# Convenience validation functions for common patterns
def validate_positive_numeric(value: Any, parameter_name: str = None) -> None:
    """Validate that a value is a positive number."""
    validator = NumericRangeValidator(min_val=0.0, parameter_name=parameter_name)
    validator.validate(value)


def validate_probability(value: Any, parameter_name: str = None) -> None:
    """Validate that a value is a valid probability (0-1)."""
    validator = NumericRangeValidator(
        min_val=0.0, max_val=1.0, parameter_name=parameter_name
    )
    validator.validate(value)


def validate_window_size(
    value: Any, min_window: int = 2, max_window: int = 1000, parameter_name: str = None
) -> None:
    """Validate that a value is a valid window size."""
    validator = NumericRangeValidator(
        min_val=min_window, max_val=max_window, parameter_name=parameter_name
    )
    validator.validate(value)
