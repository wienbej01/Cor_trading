"""
Unit tests for Architecture Improvements Module.

Tests for:
- Interface layer functionality
- Validation interfaces
- Parameter validation across modules
- Error handling consistency
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings
from typing import Dict, Any, Optional, Union, List

from src.interfaces.feature_preparation import FeaturePreparationInterface
from src.interfaces.validation import (
    ValidationInterface,
    ParameterValidator,
    DataValidator,
)
from tests.test_utils import generate_synthetic_market_data, CustomAssertions


class TestFeaturePreparationInterface:
    """Test FeaturePreparationInterface class."""

    @pytest.fixture
    def feature_preparation(self):
        """Create a FeaturePreparationInterface instance for testing."""
        return FeaturePreparationInterface()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    def test_initialization(self, feature_preparation):
        """Test FeaturePreparationInterface initialization."""
        assert feature_preparation is not None
        assert hasattr(feature_preparation, "prepare_features")
        assert hasattr(feature_preparation, "validate_features")
        assert hasattr(feature_preparation, "align_features")

    def test_prepare_features(self, feature_preparation, sample_data):
        """Test feature preparation."""
        fx_series, commodity_series = sample_data

        # Prepare features
        features = feature_preparation.prepare_features(fx_series, commodity_series)

        # Check that features are returned
        assert isinstance(features, pd.DataFrame)

        # Check that features have correct index
        assert features.index.equals(fx_series.index)

        # Check that expected features are present
        expected_features = [
            "fx_returns",
            "commodity_returns",
            "fx_volatility",
            "commodity_volatility",
        ]
        for feature in expected_features:
            assert feature in features.columns

    def test_validate_features(self, feature_preparation, sample_data):
        """Test feature validation."""
        fx_series, commodity_series = sample_data

        # Prepare features first
        features = feature_preparation.prepare_features(fx_series, commodity_series)

        # Validate features
        validation_result = feature_preparation.validate_features(features)

        # Check that validation result is returned
        assert isinstance(validation_result, dict)

        # Check that validation result contains expected keys
        expected_keys = ["is_valid", "issues", "warnings"]
        for key in expected_keys:
            assert key in validation_result

        # Check that validation result is reasonable
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["issues"], list)
        assert isinstance(validation_result["warnings"], list)

    def test_align_features(self, feature_preparation, sample_data):
        """Test feature alignment."""
        fx_series, commodity_series = sample_data

        # Create features with different indices
        fx_features = pd.DataFrame(
            {
                "fx_returns": fx_series.pct_change(),
                "fx_volatility": fx_series.rolling(window=20).std(),
            }
        )

        # Commodity features with different index (missing some dates)
        commodity_features = pd.DataFrame(
            {
                "commodity_returns": commodity_series.pct_change(),
                "commodity_volatility": commodity_series.rolling(window=20).std(),
            }
        )

        # Remove some rows from commodity features
        commodity_features = commodity_features.iloc[10:]

        # Align features
        aligned_features = feature_preparation.align_features(
            fx_features, commodity_features
        )

        # Check that aligned features are returned
        assert isinstance(aligned_features, pd.DataFrame)

        # Check that features are aligned (same index)
        assert len(aligned_features) == len(
            commodity_features
        )  # Should match shorter series

        # Check that all expected features are present
        expected_features = [
            "fx_returns",
            "fx_volatility",
            "commodity_returns",
            "commodity_volatility",
        ]
        for feature in expected_features:
            assert feature in aligned_features.columns

    def test_handle_missing_data(self, feature_preparation, sample_data):
        """Test handling of missing data."""
        fx_series, commodity_series = sample_data

        # Add NaN values to series
        fx_series_with_nan = fx_series.copy()
        fx_series_with_nan.iloc[100:110] = np.nan

        commodity_series_with_nan = commodity_series.copy()
        commodity_series_with_nan.iloc[200:210] = np.nan

        # Prepare features with missing data
        features = feature_preparation.prepare_features(
            fx_series_with_nan, commodity_series_with_nan
        )

        # Should handle missing data gracefully
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(fx_series)

        # Validate features with missing data
        validation_result = feature_preparation.validate_features(features)

        # Should detect missing data
        assert isinstance(validation_result, dict)
        assert "issues" in validation_result
        # Should have some issues related to missing data

    def test_feature_transformation(self, feature_preparation, sample_data):
        """Test feature transformation."""
        fx_series, commodity_series = sample_data

        # Prepare features
        features = feature_preparation.prepare_features(fx_series, commodity_series)

        # Apply transformations
        transformed_features = feature_preparation.transform_features(
            features, transformations=["normalize", "standardize"]
        )

        # Check that transformed features are returned
        assert isinstance(transformed_features, pd.DataFrame)

        # Check that transformed features have same index
        assert transformed_features.index.equals(features.index)

        # Check that transformations are applied
        # Normalized features should be between 0 and 1
        normalized_cols = [
            col for col in transformed_features.columns if "_normalized" in col
        ]
        for col in normalized_cols:
            valid_values = transformed_features[col].dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all()
                assert (valid_values <= 1).all()

        # Standardized features should have mean ~0 and std ~1
        standardized_cols = [
            col for col in transformed_features.columns if "_standardized" in col
        ]
        for col in standardized_cols:
            valid_values = transformed_features[col].dropna()
            if len(valid_values) > 0:
                assert abs(valid_values.mean()) < 0.1  # Close to 0
                assert abs(valid_values.std() - 1.0) < 0.1  # Close to 1


class TestParameterValidator:
    """Test ParameterValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a ParameterValidator instance for testing."""
        return ParameterValidator()

    def test_initialization(self, validator):
        """Test ParameterValidator initialization."""
        assert validator is not None
        assert hasattr(validator, "validate")
        assert hasattr(validator, "validate_range")
        assert hasattr(validator, "validate_type")
        assert hasattr(validator, "validate_choice")

    def test_validate_range(self, validator):
        """Test range validation."""
        # Test valid value
        result = validator.validate_range(1.5, 0.0, 2.0, "test_param")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test value below minimum
        result = validator.validate_range(-0.5, 0.0, 2.0, "test_param")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "below minimum" in result["issues"][0]

        # Test value above maximum
        result = validator.validate_range(2.5, 0.0, 2.0, "test_param")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "above maximum" in result["issues"][0]

        # Test inclusive bounds
        result = validator.validate_range(
            0.0, 0.0, 2.0, "test_param", inclusive_min=True
        )
        assert result["is_valid"] is True

        result = validator.validate_range(
            2.0, 0.0, 2.0, "test_param", inclusive_max=True
        )
        assert result["is_valid"] is True

    def test_validate_type(self, validator):
        """Test type validation."""
        # Test correct type
        result = validator.validate_type(1.5, float, "test_param")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test incorrect type
        result = validator.validate_type("1.5", float, "test_param")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "type" in result["issues"][0].lower()

        # Test None value with allow_none=True
        result = validator.validate_type(None, float, "test_param", allow_none=True)
        assert result["is_valid"] is True

        # Test None value with allow_none=False
        result = validator.validate_type(None, float, "test_param", allow_none=False)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_choice(self, validator):
        """Test choice validation."""
        choices = ["option1", "option2", "option3"]

        # Test valid choice
        result = validator.validate_choice("option2", choices, "test_param")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test invalid choice
        result = validator.validate_choice("option4", choices, "test_param")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "not in valid choices" in result["issues"][0]

        # Test case sensitivity
        result = validator.validate_choice(
            "Option2", choices, "test_param", case_sensitive=True
        )
        assert result["is_valid"] is False

        result = validator.validate_choice(
            "Option2", choices, "test_param", case_sensitive=False
        )
        assert result["is_valid"] is True

    def test_validate_dictionary(self, validator):
        """Test dictionary validation."""
        schema = {
            "param1": {"type": float, "min": 0.0, "max": 1.0},
            "param2": {"type": str, "choices": ["a", "b", "c"]},
            "param3": {"type": int, "min": 1, "max": 10, "required": False},
        }

        # Test valid parameters
        params = {"param1": 0.5, "param2": "b"}
        result = validator.validate_dictionary(params, schema, "test_params")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test invalid parameters
        params = {"param1": 1.5, "param2": "d"}  # Above maximum  # Invalid choice
        result = validator.validate_dictionary(params, schema, "test_params")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

        # Test missing required parameter
        params = {
            "param1": 0.5
            # Missing param2 (required)
        }
        result = validator.validate_dictionary(params, schema, "test_params")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "required" in result["issues"][0].lower()

        # Test extra parameters
        params = {"param1": 0.5, "param2": "b", "extra_param": "value"}  # Not in schema
        result = validator.validate_dictionary(params, schema, "test_params")
        assert result["is_valid"] is True  # Should still be valid
        assert len(result["warnings"]) > 0  # But should have warnings


class TestDataValidator:
    """TestDataValidator class."""

    @pytest.fixture
    def data_validator(self):
        """Create a DataValidator instance for testing."""
        return DataValidator()

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        data = pd.DataFrame(
            {
                "col1": np.random.randn(len(dates)),
                "col2": np.random.randn(len(dates)),
                "col3": np.random.choice(["A", "B", "C"], len(dates)),
            },
            index=dates,
        )
        return data

    def test_initialization(self, data_validator):
        """TestDataValidator initialization."""
        assert data_validator is not None
        assert hasattr(data_validator, "validate_dataframe")
        assert hasattr(data_validator, "validate_series")
        assert hasattr(data_validator, "check_missing_values")
        assert hasattr(data_validator, "check_data_types")

    def test_validate_dataframe(self, data_validator, sample_dataframe):
        """Test DataFrame validation."""
        # Test valid DataFrame
        result = data_validator.validate_dataframe(sample_dataframe)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test DataFrame with missing values
        df_with_nan = sample_dataframe.copy()
        df_with_nan.iloc[10:15, 0] = np.nan
        result = data_validator.validate_dataframe(df_with_nan)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "missing" in result["issues"][0].lower()

    def test_validate_series(self, data_validator):
        """Test Series validation."""
        # Test valid Series
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        series = pd.Series(np.random.randn(len(dates)), index=dates)

        result = data_validator.validate_series(series)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test Series with missing values
        series_with_nan = series.copy()
        series_with_nan.iloc[10:15] = np.nan
        result = data_validator.validate_series(series_with_nan)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_check_missing_values(self, data_validator, sample_dataframe):
        """Test missing value checking."""
        # Test DataFrame without missing values
        result = data_validator.check_missing_values(sample_dataframe)
        assert result["has_missing"] is False
        assert len(result["missing_counts"]) == 0

        # Test DataFrame with missing values
        df_with_nan = sample_dataframe.copy()
        df_with_nan.iloc[10:15, 0] = np.nan
        df_with_nan.iloc[20:25, 1] = np.nan

        result = data_validator.check_missing_values(df_with_nan)
        assert result["has_missing"] is True
        assert len(result["missing_counts"]) == 2
        assert result["missing_counts"]["col1"] == 5
        assert result["missing_counts"]["col2"] == 5

    def test_check_data_types(self, data_validator, sample_dataframe):
        """Test data type checking."""
        # Test with expected types
        expected_types = {"col1": "float64", "col2": "float64", "col3": "object"}

        result = data_validator.check_data_types(sample_dataframe, expected_types)
        assert result["types_match"] is True
        assert len(result["type_mismatches"]) == 0

        # Test with incorrect expected types
        expected_types = {
            "col1": "int64",  # Incorrect
            "col2": "float64",
            "col3": "object",
        }

        result = data_validator.check_data_types(sample_dataframe, expected_types)
        assert result["types_match"] is False
        assert len(result["type_mismatches"]) == 1
        assert "col1" in result["type_mismatches"]

    def test_check_date_index(self, data_validator):
        """Test date index validation."""
        # Test with valid DatetimeIndex
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        series = pd.Series(np.random.randn(len(dates)), index=dates)

        result = data_validator.check_date_index(series)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test with invalid index
        series_invalid = pd.Series(np.random.randn(10), index=range(10))

        result = data_validator.check_date_index(series_invalid)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "datetime" in result["issues"][0].lower()

    def test_check_data_frequency(self, data_validator):
        """Test data frequency validation."""
        # Test with daily frequency
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        series = pd.Series(np.random.randn(len(dates)), index=dates)

        result = data_validator.check_data_frequency(series, expected_freq="D")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test with incorrect frequency
        result = data_validator.check_data_frequency(series, expected_freq="M")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "frequency" in result["issues"][0].lower()


class TestValidationInterface:
    """Test ValidationInterface class."""

    @pytest.fixture
    def validation_interface(self):
        """Create a ValidationInterface instance for testing."""
        return ValidationInterface()

    def test_initialization(self, validation_interface):
        """Test ValidationInterface initialization."""
        assert validation_interface is not None
        assert hasattr(validation_interface, "parameter_validator")
        assert hasattr(validation_interface, "data_validator")
        assert isinstance(validation_interface.parameter_validator, ParameterValidator)
        assert isinstance(validation_interface.data_validator, DataValidator)

    def test_validate_strategy_config(self, validation_interface):
        """Test strategy configuration validation."""
        # Test valid configuration
        valid_config = {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
            "risk_management": {"max_position_size": 0.2, "max_drawdown": 0.1},
        }

        result = validation_interface.validate_strategy_config(valid_config)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test invalid configuration
        invalid_config = {
            "lookbacks": {"beta_window": -60, "z_window": 20},  # Invalid negative value
            "thresholds": {
                "entry_z": 1.5,
                "exit_z": 2.5,  # Invalid: exit > entry
                "stop_z": 3.0,
            },
        }

        result = validation_interface.validate_strategy_config(invalid_config)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_model_config(self, validation_interface):
        """Test model configuration validation."""
        # Test valid configuration
        valid_config = {
            "model_type": "kalman",
            "parameters": {"delta": 1e-5, "lam": 0.995},
            "features": ["returns", "volatility", "correlation"],
        }

        result = validation_interface.validate_model_config(valid_config)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test invalid configuration
        invalid_config = {
            "model_type": "invalid_model",  # Invalid model type
            "parameters": {
                "delta": -1e-5,  # Invalid negative value
                "lam": 1.5,  # Invalid: > 1
            },
        }

        result = validation_interface.validate_model_config(invalid_config)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_execution_config(self, validation_interface):
        """Test execution configuration validation."""
        # Test valid configuration
        valid_config = {
            "slippage_bps": {"fx": 1.0, "commodity": 2.0},
            "transaction_costs": {"fixed": 0.1, "percentage": 0.001},
            "order_type": "limit",
        }

        result = validation_interface.validate_execution_config(valid_config)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test invalid configuration
        invalid_config = {
            "slippage_bps": {"fx": -1.0, "commodity": 2.0},  # Invalid negative value
            "transaction_costs": {
                "fixed": -0.1,  # Invalid negative value
                "percentage": 1.5,  # Invalid: > 1
            },
            "order_type": "invalid_order_type",  # Invalid order type
        }

        result = validation_interface.validate_execution_config(invalid_config)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_input_data(self, validation_interface):
        """Test input data validation."""
        # Test valid data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        fx_data = pd.Series(np.random.randn(len(dates)), index=dates)
        commodity_data = pd.Series(np.random.randn(len(dates)), index=dates)

        result = validation_interface.validate_input_data(fx_data, commodity_data)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

        # Test invalid data (different lengths)
        dates_short = pd.date_range("2020-01-01", "2020-06-30", freq="D")
        commodity_data_short = pd.Series(
            np.random.randn(len(dates_short)), index=dates_short
        )

        result = validation_interface.validate_input_data(fx_data, commodity_data_short)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "length" in result["issues"][0].lower()

    def test_comprehensive_validation(self, validation_interface):
        """Test comprehensive validation of all components."""
        # Create valid configurations and data
        strategy_config = {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        }

        model_config = {
            "model_type": "kalman",
            "parameters": {"delta": 1e-5, "lam": 0.995},
        }

        execution_config = {
            "slippage_bps": {"fx": 1.0, "commodity": 2.0},
            "order_type": "limit",
        }

        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        fx_data = pd.Series(np.random.randn(len(dates)), index=dates)
        commodity_data = pd.Series(np.random.randn(len(dates)), index=dates)

        # Validate all components
        results = validation_interface.validate_all_components(
            strategy_config=strategy_config,
            model_config=model_config,
            execution_config=execution_config,
            fx_data=fx_data,
            commodity_data=commodity_data,
        )

        # Check that results are returned
        assert isinstance(results, dict)
        assert "strategy_config" in results
        assert "model_config" in results
        assert "execution_config" in results
        assert "input_data" in results
        assert "overall_valid" in results

        # Check that all components are valid
        assert results["strategy_config"]["is_valid"] is True
        assert results["model_config"]["is_valid"] is True
        assert results["execution_config"]["is_valid"] is True
        assert results["input_data"]["is_valid"] is True
        assert results["overall_valid"] is True

        # Test with invalid components
        invalid_strategy_config = {
            "lookbacks": {"beta_window": -60, "z_window": 20}  # Invalid
        }

        results = validation_interface.validate_all_components(
            strategy_config=invalid_strategy_config,
            model_config=model_config,
            execution_config=execution_config,
            fx_data=fx_data,
            commodity_data=commodity_data,
        )

        # Should detect invalid components
        assert results["strategy_config"]["is_valid"] is False
        assert results["overall_valid"] is False


class TestErrorHandlingConsistency:
    """Test error handling consistency across modules."""

    def test_consistent_error_messages(self):
        """Test that error messages are consistent across modules."""
        validator = ParameterValidator()
        data_validator = DataValidator()

        # Test range validation error
        range_result = validator.validate_range(5.0, 0.0, 1.0, "test_param")

        # Test type validation error
        type_result = validator.validate_type("string", float, "test_param")

        # Test choice validation error
        choice_result = validator.validate_choice("invalid", ["valid"], "test_param")

        # All should have consistent error message structure
        for result in [range_result, type_result, choice_result]:
            assert "is_valid" in result
            assert "issues" in result
            assert isinstance(result["is_valid"], bool)
            assert isinstance(result["issues"], list)

            if not result["is_valid"]:
                assert len(result["issues"]) > 0
                assert isinstance(result["issues"][0], str)

    def test_error_propagation(self):
        """Test that errors are properly propagated through the system."""
        validation_interface = ValidationInterface()

        # Test with invalid configuration that should cause multiple errors
        invalid_config = {
            "lookbacks": {
                "beta_window": -60,  # Invalid negative value
                "z_window": 0,  # Invalid zero value
            },
            "thresholds": {
                "entry_z": 1.5,
                "exit_z": 2.5,  # Invalid: exit > entry
                "stop_z": -1.0,  # Invalid negative value
            },
        }

        result = validation_interface.validate_strategy_config(invalid_config)

        # Should detect all issues
        assert result["is_valid"] is False
        assert len(result["issues"]) >= 3  # Should detect at least 3 issues

        # Issues should be specific and helpful
        for issue in result["issues"]:
            assert isinstance(issue, str)
            assert len(issue) > 0  # Should not be empty

    def test_graceful_degradation(self):
        """Test that the system degrades gracefully when errors occur."""
        validation_interface = ValidationInterface()

        # Test with partially valid configuration
        partially_valid_config = {
            "lookbacks": {"beta_window": 60, "z_window": 0},  # Valid  # Invalid
            "thresholds": {
                "entry_z": 1.5,  # Valid
                "exit_z": 0.5,  # Valid
                "stop_z": -1.0,  # Invalid
            },
        }

        result = validation_interface.validate_strategy_config(partially_valid_config)

        # Should detect invalid parts but not fail completely
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

        # Should provide specific information about what's wrong
        issue_text = " ".join(result["issues"]).lower()
        assert "z_window" in issue_text
        assert "stop_z" in issue_text

    def test_warning_system(self):
        """Test that warnings are properly generated for non-critical issues."""
        validator = ParameterValidator()

        # Test with parameters that are valid but have potential issues
        schema = {
            "param1": {"type": float, "min": 0.0, "max": 1.0},
            "param2": {"type": int, "min": 1, "max": 10, "required": False},
        }

        params = {
            "param1": 0.99,  # Valid but close to boundary
            "param2": 1,  # Valid but at minimum
            "extra_param": "value",  # Extra parameter not in schema
        }

        result = validator.validate_dictionary(params, schema, "test_params")

        # Should be valid but have warnings
        assert result["is_valid"] is True
        assert len(result["warnings"]) > 0

        # Warnings should be informative
        for warning in result["warnings"]:
            assert isinstance(warning, str)
            assert len(warning) > 0


class TestArchitectureIntegration:
    """Integration tests for architecture components."""

    def test_interface_integration(self):
        """Test integration between different interfaces."""
        feature_prep = FeaturePreparationInterface()
        validation_interface = ValidationInterface()

        # Create sample data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        fx_data = pd.Series(np.random.randn(len(dates)), index=dates)
        commodity_data = pd.Series(np.random.randn(len(dates)), index=dates)

        # Prepare features
        features = feature_prep.prepare_features(fx_data, commodity_data)

        # Validate features
        feature_validation = feature_prep.validate_features(features)

        # Validate input data
        data_validation = validation_interface.validate_input_data(
            fx_data, commodity_data
        )

        # Both should work together
        assert feature_validation["is_valid"] is True
        assert data_validation["is_valid"] is True

    def test_validation_pipeline(self):
        """Test complete validation pipeline."""
        validation_interface = ValidationInterface()

        # Create configurations
        strategy_config = {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        }

        model_config = {
            "model_type": "kalman",
            "parameters": {"delta": 1e-5, "lam": 0.995},
        }

        # Create data
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        fx_data = pd.Series(np.random.randn(len(dates)), index=dates)
        commodity_data = pd.Series(np.random.randn(len(dates)), index=dates)

        # Run complete validation pipeline
        pipeline_results = validation_interface.validate_all_components(
            strategy_config=strategy_config,
            model_config=model_config,
            execution_config=None,  # Optional
            fx_data=fx_data,
            commodity_data=commodity_data,
        )

        # Check pipeline results
        assert isinstance(pipeline_results, dict)
        assert "overall_valid" in pipeline_results

        # Should be valid with our test data
        assert pipeline_results["overall_valid"] is True

        # Should have results for all provided components
        assert "strategy_config" in pipeline_results
        assert "model_config" in pipeline_results
        assert "input_data" in pipeline_results

    def test_backward_compatibility(self):
        """Test that architecture changes maintain backward compatibility."""
        # This test ensures that new interfaces don't break existing functionality

        # Test with minimal configuration
        minimal_config = {"beta_window": 60, "entry_z": 1.5}

        validation_interface = ValidationInterface()

        # Should handle minimal configuration gracefully
        result = validation_interface.validate_strategy_config(minimal_config)

        # Should provide helpful feedback about missing required parameters
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "issues" in result

        # Should not crash with minimal input
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["issues"], list)
