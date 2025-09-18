"""
Integration tests for the FX-Commodity Correlation Arbitrage system.

Tests for:
- End-to-end signal generation to execution flow
- Multi-model ensemble integration
- Risk management integration
- Performance metrics integration
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings

from features.signal_optimization import SignalOptimizer
from risk.manager import RiskManager, RiskConfig
from exec.policy import ExecutionPolicy, ExecutionConfig
from backtest.engine import BacktestEngine
from backtest.rolling_metrics import RollingMetrics
from backtest.distribution_analysis import DistributionAnalysis
from ml.ensemble import ModelEnsemble, OLSModelWrapper, KalmanModelWrapper
from interfaces.validation import ValidationInterface
from tests.test_utils import (
    generate_synthetic_market_data,
    generate_synthetic_equity_curve,
    CustomAssertions,
)


class TestEndToEndSignalToExecutionFlow:
    """Test end-to-end signal generation to execution flow."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    @pytest.fixture
    def strategy_config(self):
        """Create strategy configuration for testing."""
        return {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
            "use_kalman": True,
        }

    @pytest.fixture
    def risk_config(self):
        """Create risk configuration for testing."""
        config = RiskConfig()
        config.max_position_size_per_pair = 0.2
        config.max_trade_risk = 0.02
        config.daily_drawdown_limit = 0.05
        config.volatility_scaling = True
        return config

    @pytest.fixture
    def execution_config(self):
        """Create execution configuration for testing."""
        return ExecutionConfig(
            fx_slippage_bps=1.0,
            comd_slippage_bps=2.0,
            default_order_type="limit",
            market_impact_coefficient=0.1,
            fx_fixed_cost=0.0001,
            comd_fixed_cost=1.0,
        )

    def test_complete_signal_generation_flow(self, sample_data, strategy_config):
        """Test complete signal generation flow."""
        fx_series, commodity_series = sample_data

        # Create signal optimizer
        signal_optimizer = SignalOptimizer(strategy_config)

        # Generate enhanced signals
        thresholds = {
            "entry_z": strategy_config["thresholds"]["entry_z"],
            "exit_z": strategy_config["thresholds"]["exit_z"],
            "stop_z": strategy_config["thresholds"]["stop_z"],
        }

        signals_df = signal_optimizer.generate_enhanced_signals(
            fx_series,
            commodity_series,
            thresholds,
            use_vol_adjustment=True,
            use_enhanced_kalman=True,
        )

        # Check that signals are generated
        assert isinstance(signals_df, pd.DataFrame)
        assert len(signals_df) == len(fx_series)

        # Check that all expected columns are present
        expected_columns = [
            "fx_price",
            "comd_price",
            "spread",
            "alpha",
            "beta",
            "spread_z",
            "entry_threshold",
            "exit_threshold",
            "stop_threshold",
            "signal",
            "enter_long",
            "enter_short",
            "exit_long",
            "exit_short",
            "stop_long",
            "stop_short",
        ]

        for col in expected_columns:
            assert col in signals_df.columns

        # Check that signals are valid (-1, 0, 1)
        valid_signals = signals_df["signal"].dropna()
        assert valid_signals.isin([-1, 0, 1]).all()

        # Check that signal logic is consistent
        # When signal is 1 (long), we should have entered long
        long_signals = signals_df[signals_df["signal"] == 1]
        if len(long_signals) > 0:
            # Check that entry conditions were met
            assert (long_signals["spread_z"] <= -long_signals["entry_threshold"]).any()

        # When signal is -1 (short), we should have entered short
        short_signals = signals_df[signals_df["signal"] == -1]
        if len(short_signals) > 0:
            # Check that entry conditions were met
            assert (short_signals["spread_z"] >= short_signals["entry_threshold"]).any()

    def test_signal_to_risk_integration(
        self, sample_data, strategy_config, risk_config
    ):
        """Test integration between signal generation and risk management."""
        fx_series, commodity_series = sample_data

        # Generate signals
        signal_optimizer = SignalOptimizer(strategy_config)
        thresholds = {
            "entry_z": strategy_config["thresholds"]["entry_z"],
            "exit_z": strategy_config["thresholds"]["exit_z"],
            "stop_z": strategy_config["thresholds"]["stop_z"],
        }

        signals_df = signal_optimizer.generate_enhanced_signals(
            fx_series, commodity_series, thresholds
        )

        # Create risk manager
        risk_manager = RiskManager(risk_config)
        risk_manager.account_equity = 100000  # Set initial equity

        # Test position sizing for each signal
        position_results = []

        for idx, row in signals_df.iterrows():
            if row["signal"] != 0:  # Only for active positions
                # Calculate volatility (simplified)
                fx_vol = (
                    fx_series.loc[:idx].rolling(window=20).std().iloc[-1]
                    if len(fx_series.loc[:idx]) >= 20
                    else 0.02
                )
                comd_vol = (
                    commodity_series.loc[:idx].rolling(window=20).std().iloc[-1]
                    if len(commodity_series.loc[:idx]) >= 20
                    else 0.02
                )

                # Calculate position size
                fx_pos, comd_pos = risk_manager.calculate_position_size(
                    pair_name="test_pair",
                    signal=row["signal"],
                    fx_price=row["fx_price"],
                    comd_price=row["comd_price"],
                    fx_vol=fx_vol,
                    comd_vol=comd_vol,
                    stop_loss_distance=0.02,  # 2% stop loss
                )

                position_results.append(
                    {
                        "date": idx,
                        "signal": row["signal"],
                        "fx_position": fx_pos,
                        "comd_position": comd_pos,
                    }
                )

        # Check that position results are generated
        assert len(position_results) > 0

        # Check that positions are consistent with signals
        for result in position_results:
            # Position direction should match signal direction
            assert np.sign(result["fx_position"]) == result["signal"]
            assert (
                np.sign(result["comd_position"]) == -result["signal"]
            )  # Opposite for spread

            # Positions should be non-zero when signal is non-zero
            assert result["fx_position"] != 0
            assert result["comd_position"] != 0

    def test_risk_to_execution_integration(
        self, sample_data, risk_config, execution_config
    ):
        """Test integration between risk management and execution."""
        fx_series, commodity_series = sample_data

        # Create risk manager and execution policy
        risk_manager = RiskManager(risk_config)
        risk_manager.account_equity = 100000

        execution_policy = ExecutionPolicy(execution_config)

        # Test a specific trade
        test_date = fx_series.index[100]  # Pick a specific date
        fx_price = fx_series.loc[test_date]
        comd_price = commodity_series.loc[test_date]

        # Calculate position size
        fx_vol = fx_series.loc[:test_date].rolling(window=20).std().iloc[-1]
        comd_vol = commodity_series.loc[:test_date].rolling(window=20).std().iloc[-1]

        signal = 1  # Long signal
        fx_pos, comd_pos = risk_manager.calculate_position_size(
            pair_name="test_pair",
            signal=signal,
            fx_price=fx_price,
            comd_price=comd_price,
            fx_vol=fx_vol,
            comd_vol=comd_vol,
            stop_loss_distance=0.02,
        )

        # Calculate execution costs
        execution_costs = execution_policy.calculate_execution_costs(
            fx_price, comd_price, fx_pos, comd_pos
        )

        # Apply slippage
        fx_exec_price, comd_exec_price = execution_policy.apply_slippage(
            fx_price, comd_price, fx_pos, comd_pos
        )

        # Check that execution is valid
        assert isinstance(execution_costs, float)
        assert execution_costs >= 0

        # Check that execution prices reflect slippage
        if fx_pos > 0:  # Long position
            assert fx_exec_price >= fx_price  # Buy at higher price
        else:
            assert fx_exec_price <= fx_price  # Sell at lower price

        if comd_pos < 0:  # Short position (opposite side of spread)
            assert comd_exec_price <= comd_price  # Sell at lower price
        else:
            assert comd_exec_price >= comd_price  # Buy at higher price

        # Check that total cost is reasonable
        expected_fx_cost = abs(fx_pos * fx_price) * (
            execution_config.fx_slippage_bps / 10000
            + execution_config.fx_percentage_cost
        )
        expected_comd_cost = abs(comd_pos * comd_price) * (
            execution_config.comd_slippage_bps / 10000
            + execution_config.comd_percentage_cost
        )
        expected_total_cost = (
            expected_fx_cost
            + expected_comd_cost
            + execution_config.fx_fixed_cost
            + execution_config.comd_fixed_cost
        )

        assert abs(execution_costs - expected_total_cost) < max(
            execution_costs * 0.1, 0.01
        )  # Within 10% or $0.01

    def test_complete_backtest_integration(
        self, sample_data, strategy_config, risk_config, execution_config
    ):
        """Test complete backtest integration."""
        fx_series, commodity_series = sample_data

        # Create backtest engine
        backtest_engine = BacktestEngine(
            strategy_config=strategy_config,
            risk_config=risk_config,
            execution_config=execution_config,
        )

        # Run backtest
        results = backtest_engine.run_backtest(fx_series, commodity_series)

        # Check that results are generated
        assert isinstance(results, dict)

        # Check that all expected result components are present
        expected_components = [
            "equity_curve",
            "returns",
            "positions",
            "signals",
            "trades",
            "performance_metrics",
            "drawdowns",
        ]

        for component in expected_components:
            assert component in results

        # Check equity curve
        assert isinstance(results["equity_curve"], pd.Series)
        assert len(results["equity_curve"]) == len(fx_series)
        assert results["equity_curve"].index.equals(fx_series.index)

        # Check returns
        assert isinstance(results["returns"], pd.Series)
        assert len(results["returns"]) == len(fx_series)
        assert results["returns"].index.equals(fx_series.index)

        # Check positions
        assert isinstance(results["positions"], pd.DataFrame)
        assert len(results["positions"]) == len(fx_series)
        assert results["positions"].index.equals(fx_series.index)

        # Check performance metrics
        assert isinstance(results["performance_metrics"], dict)
        expected_metrics = [
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        ]
        for metric in expected_metrics:
            assert metric in results["performance_metrics"]

        # Check that performance is reasonable
        assert isinstance(results["performance_metrics"]["total_return"], (int, float))
        assert isinstance(results["performance_metrics"]["sharpe_ratio"], (int, float))
        assert isinstance(results["performance_metrics"]["max_drawdown"], (int, float))
        assert (
            results["performance_metrics"]["max_drawdown"] <= 0
        )  # Drawdown should be negative or zero

    def test_performance_metrics_integration(
        self, sample_data, strategy_config, risk_config, execution_config
    ):
        """Test integration with performance metrics."""
        fx_series, commodity_series = sample_data

        # Run backtest
        backtest_engine = BacktestEngine(
            strategy_config=strategy_config,
            risk_config=risk_config,
            execution_config=execution_config,
        )

        results = backtest_engine.run_backtest(fx_series, commodity_series)

        # Calculate rolling metrics
        rolling_metrics = RollingMetrics()
        rolling_df = rolling_metrics.calculate_rolling_metrics(
            results["returns"], window=63
        )

        # Check rolling metrics
        assert isinstance(rolling_df, pd.DataFrame)
        assert len(rolling_df) == len(results["returns"])

        expected_rolling_metrics = [
            "sharpe",
            "sortino",
            "drawdown",
            "calmar",
            "win_rate",
        ]
        for metric in expected_rolling_metrics:
            assert metric in rolling_df.columns

        # Calculate distribution metrics
        dist_analysis = DistributionAnalysis()
        dist_metrics = dist_analysis.calculate_all_distribution_metrics(
            results["returns"].dropna()
        )

        # Check distribution metrics
        assert isinstance(dist_metrics, dict)
        expected_dist_metrics = [
            "skewness",
            "kurtosis",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
        ]
        for metric in expected_dist_metrics:
            assert metric in dist_metrics

        # Check that metrics are consistent
        # Max drawdown from rolling metrics should be close to max drawdown from backtest
        rolling_max_dd = rolling_df["drawdown"].min()
        backtest_max_dd = results["performance_metrics"]["max_drawdown"]

        assert abs(rolling_max_dd - backtest_max_dd) < abs(
            backtest_max_dd * 0.1
        )  # Within 10%

    def test_validation_integration(
        self, sample_data, strategy_config, risk_config, execution_config
    ):
        """Test integration with validation interfaces."""
        fx_series, commodity_series = sample_data

        # Create validation interface
        validation_interface = ValidationInterface()

        # Validate configurations
        strategy_validation = validation_interface.validate_strategy_config(
            strategy_config
        )

        # Validate input data
        data_validation = validation_interface.validate_input_data(
            fx_series, commodity_series
        )

        # Check that validation passes
        assert strategy_validation["is_valid"] is True
        assert data_validation["is_valid"] is True

        # Run backtest with validated configurations
        backtest_engine = BacktestEngine(
            strategy_config=strategy_config,
            risk_config=risk_config,
            execution_config=execution_config,
        )

        results = backtest_engine.run_backtest(fx_series, commodity_series)

        # Check that backtest runs successfully with validated configurations
        assert isinstance(results, dict)
        assert "equity_curve" in results
        assert len(results["equity_curve"]) > 0


class TestMultiModelEnsembleIntegration:
    """Test multi-model ensemble integration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    def test_ensemble_model_integration(self, sample_data):
        """Test integration of ensemble models with the trading system."""
        fx_series, commodity_series = sample_data

        # Create models
        models = {"ols": OLSModelWrapper(), "kalman": KalmanModelWrapper()}

        # Create ensemble
        ensemble = ModelEnsemble(models)

        # Fit ensemble
        ensemble.fit(fx_series, commodity_series)

        # Generate predictions
        predictions = ensemble.predict(fx_series)

        # Check that predictions are valid
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(fx_series)
        assert not predictions.isna().all()

        # Get individual model predictions
        individual_preds = ensemble.get_individual_predictions(fx_series)

        # Check individual predictions
        assert isinstance(individual_preds, dict)
        assert len(individual_preds) == len(models)

        for model_name, preds in individual_preds.items():
            assert isinstance(preds, pd.Series)
            assert len(preds) == len(fx_series)

        # Update weights based on performance
        new_weights = ensemble.update_weights(fx_series, commodity_series, window=100)

        # Check that weights are updated
        assert isinstance(new_weights, dict)
        assert len(new_weights) == len(models)
        assert abs(sum(new_weights.values()) - 1.0) < 1e-10

        # Generate predictions with updated weights
        updated_predictions = ensemble.predict(fx_series)

        # Check that updated predictions are valid
        assert isinstance(updated_predictions, pd.Series)
        assert len(updated_predictions) == len(fx_series)

    def test_ensemble_with_signal_generation(self, sample_data):
        """Test ensemble integration with signal generation."""
        fx_series, commodity_series = sample_data

        # Create ensemble
        models = {"ols": OLSModelWrapper(), "kalman": KalmanModelWrapper()}
        ensemble = ModelEnsemble(models)
        ensemble.fit(fx_series, commodity_series)

        # Generate spread predictions
        spread_predictions = ensemble.predict(fx_series)

        # Calculate z-scores from predictions
        z_window = 20
        z_scores = (
            spread_predictions - spread_predictions.rolling(window=z_window).mean()
        ) / spread_predictions.rolling(window=z_window).std()

        # Generate signals based on z-scores
        entry_threshold = 1.5
        exit_threshold = 0.5
        stop_threshold = 3.0

        signals = pd.Series(0, index=fx_series.index)

        # Entry signals
        signals[z_scores <= -entry_threshold] = 1  # Long
        signals[z_scores >= entry_threshold] = -1  # Short

        # Exit signals
        signals[(z_scores >= -exit_threshold) & (signals == 1)] = 0  # Exit long
        signals[(z_scores <= exit_threshold) & (signals == -1)] = 0  # Exit short

        # Stop signals
        signals[(z_scores >= -stop_threshold) & (signals == 1)] = 0  # Stop long
        signals[(z_scores <= stop_threshold) & (signals == -1)] = 0  # Stop short

        # Check that signals are valid
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(fx_series)
        assert signals.isin([-1, 0, 1]).all()

        # Check that we have some active signals
        assert (signals != 0).any()

        # Calculate simple returns from spread
        spread_returns = spread_predictions.diff()

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * spread_returns

        # Check that strategy returns are valid
        assert isinstance(strategy_returns, pd.Series)
        assert len(strategy_returns) == len(fx_series)
        assert not strategy_returns.isna().all()

        # Calculate performance metrics
        total_return = strategy_returns.sum()
        sharpe_ratio = (
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            if strategy_returns.std() > 0
            else 0
        )

        # Check that performance metrics are reasonable
        assert isinstance(total_return, (int, float))
        assert isinstance(sharpe_ratio, (int, float))

    def test_ensemble_with_risk_management(self, sample_data):
        """Test ensemble integration with risk management."""
        fx_series, commodity_series = sample_data

        # Create ensemble
        models = {"ols": OLSModelWrapper(), "kalman": KalmanModelWrapper()}
        ensemble = ModelEnsemble(models)
        ensemble.fit(fx_series, commodity_series)

        # Create risk manager
        risk_config = RiskConfig()
        risk_config.max_position_size_per_pair = 0.2
        risk_config.max_trade_risk = 0.02
        risk_manager = RiskManager(risk_config)
        risk_manager.account_equity = 100000

        # Generate predictions and signals
        spread_predictions = ensemble.predict(fx_series)
        z_window = 20
        z_scores = (
            spread_predictions - spread_predictions.rolling(window=z_window).mean()
        ) / spread_predictions.rolling(window=z_window).std()

        # Simple signal generation
        signals = pd.Series(0, index=fx_series.index)
        signals[z_scores <= -1.5] = 1
        signals[z_scores >= 1.5] = -1

        # Test position sizing with ensemble predictions
        position_results = []

        for idx, signal in signals.items():
            if signal != 0 and idx in fx_series.index:
                # Get current prices
                fx_price = fx_series.loc[idx]
                comd_price = commodity_series.loc[idx]

                # Calculate volatility
                fx_vol = (
                    fx_series.loc[:idx].rolling(window=20).std().iloc[-1]
                    if len(fx_series.loc[:idx]) >= 20
                    else 0.02
                )
                comd_vol = (
                    commodity_series.loc[:idx].rolling(window=20).std().iloc[-1]
                    if len(commodity_series.loc[:idx]) >= 20
                    else 0.02
                )

                # Calculate position size
                fx_pos, comd_pos = risk_manager.calculate_position_size(
                    pair_name="test_pair",
                    signal=signal,
                    fx_price=fx_price,
                    comd_price=comd_price,
                    fx_vol=fx_vol,
                    comd_vol=comd_vol,
                    stop_loss_distance=0.02,
                )

                position_results.append(
                    {
                        "date": idx,
                        "signal": signal,
                        "fx_position": fx_pos,
                        "comd_position": comd_pos,
                    }
                )

        # Check that position results are generated
        assert len(position_results) > 0

        # Check that positions are consistent with signals
        for result in position_results:
            assert np.sign(result["fx_position"]) == result["signal"]
            assert np.sign(result["comd_position"]) == -result["signal"]
            assert result["fx_position"] != 0
            assert result["comd_position"] != 0


class TestRiskManagementIntegration:
    """Test risk management integration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    def test_risk_limits_enforcement(self, sample_data):
        """Test that risk limits are properly enforced."""
        fx_series, commodity_series = sample_data

        # Create risk manager with strict limits
        risk_config = RiskConfig()
        risk_config.max_position_size_per_pair = 0.05  # Very small position limit
        risk_config.max_trade_risk = 0.005  # Very small risk limit
        risk_config.daily_drawdown_limit = 0.01  # Very small drawdown limit
        risk_manager = RiskManager(risk_config)
        risk_manager.account_equity = 100000

        # Test position sizing with strict limits
        test_date = fx_series.index[100]
        fx_price = fx_series.loc[test_date]
        comd_price = commodity_series.loc[test_date]

        # Calculate volatility
        fx_vol = fx_series.loc[:test_date].rolling(window=20).std().iloc[-1]
        comd_vol = commodity_series.loc[:test_date].rolling(window=20).std().iloc[-1]

        # Calculate position size
        fx_pos, comd_pos = risk_manager.calculate_position_size(
            pair_name="test_pair",
            signal=1,  # Long signal
            fx_price=fx_price,
            comd_price=comd_price,
            fx_vol=fx_vol,
            comd_vol=comd_vol,
            stop_loss_distance=0.02,
        )

        # Check that position size is limited
        max_fx_value = (
            risk_manager.account_equity * risk_config.max_position_size_per_pair
        )
        actual_fx_value = abs(fx_pos * fx_price)

        assert actual_fx_value <= max_fx_value

        # Check that risk is limited
        max_risk = risk_manager.account_equity * risk_config.max_trade_risk
        actual_risk = abs(fx_pos * fx_price * 0.02) + abs(comd_pos * comd_price * 0.02)

        assert actual_risk <= max_risk

    def test_drawdown_control(self, sample_data):
        """Test that drawdown control works properly."""
        fx_series, commodity_series = sample_data

        # Create risk manager
        risk_config = RiskConfig()
        risk_config.daily_drawdown_limit = 0.05  # 5% daily drawdown limit
        risk_manager = RiskManager(risk_config)
        risk_manager.account_equity = 100000
        risk_manager.daily_equity_high = 100000

        # Simulate equity curve with drawdown
        equity_curve = generate_synthetic_equity_curve(
            fx_series.index,
            base_equity=100000,
            annual_return=0.10,
            annual_volatility=0.20,
            max_drawdown=0.15,  # 15% max drawdown
            seed=42,
        )

        # Check drawdown at different points
        for date, equity in equity_curve.items():
            risk_manager.account_equity = equity

            # Check if daily drawdown limit is breached
            drawdown_breached = risk_manager.check_daily_drawdown_limit()

            # If equity is below daily high, check if drawdown exceeds limit
            if equity < risk_manager.daily_equity_high:
                drawdown = (
                    equity - risk_manager.daily_equity_high
                ) / risk_manager.daily_equity_high

                if drawdown < -risk_config.daily_drawdown_limit:
                    assert drawdown_breached is True
                    assert risk_manager.daily_drawdown_exceeded is True

    def test_circuit_breaker_integration(self, sample_data):
        """Test circuit breaker integration."""
        fx_series, commodity_series = sample_data

        # Create risk manager with circuit breaker enabled
        risk_config = RiskConfig()
        risk_config.enable_circuit_breaker = True
        risk_config.circuit_breaker_cooldown = 5  # 5 days cooldown
        risk_manager = RiskManager(risk_config)

        # Test circuit breaker activation
        current_date = fx_series.index[100]

        # Initially, circuit breaker should not be active
        assert risk_manager.check_circuit_breaker(current_date) is False

        # Trigger circuit breaker
        risk_manager.trigger_circuit_breaker(current_date)

        # Now circuit breaker should be active
        assert risk_manager.check_circuit_breaker(current_date) is True

        # Check that circuit breaker end date is set correctly
        expected_end_date = current_date.date() + timedelta(
            days=risk_config.circuit_breaker_cooldown
        )
        assert risk_manager.circuit_breaker_end_date == expected_end_date

        # Test that circuit breaker prevents trading
        can_trade = risk_manager.can_trade_pair("test_pair", current_date)
        assert can_trade is False

        # Test circuit breaker deactivation after cooldown
        future_date = current_date + timedelta(
            days=risk_config.circuit_breaker_cooldown + 1
        )
        assert risk_manager.check_circuit_breaker(future_date) is False

        # Now trading should be allowed again
        can_trade = risk_manager.can_trade_pair("test_pair", future_date)
        assert can_trade is True


class TestPerformanceMetricsIntegration:
    """Test performance metrics integration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2022-12-31", seed=42
        )

    def test_rolling_metrics_integration(self, sample_data):
        """Test integration of rolling metrics."""
        fx_series, commodity_series = sample_data

        # Create a simple strategy
        strategy_config = {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        }

        # Run backtest
        backtest_engine = BacktestEngine(strategy_config=strategy_config)
        results = backtest_engine.run_backtest(fx_series, commodity_series)

        # Calculate rolling metrics
        rolling_metrics = RollingMetrics()

        # Test different window sizes
        windows = [21, 63, 126, 252]

        for window in windows:
            rolling_df = rolling_metrics.calculate_rolling_metrics(
                results["returns"], window=window
            )

            # Check that rolling metrics are calculated correctly
            assert isinstance(rolling_df, pd.DataFrame)
            assert len(rolling_df) == len(results["returns"])

            # Check that all expected metrics are present
            expected_metrics = ["sharpe", "sortino", "drawdown", "calmar", "win_rate"]
            for metric in expected_metrics:
                assert metric in rolling_df.columns

            # Check that early values are NaN due to rolling window
            assert rolling_df.iloc[: window - 1].isna().all().all()

            # Check that later values are not all NaN
            assert not rolling_df.iloc[window:].isna().all().all()

    def test_distribution_metrics_integration(self, sample_data):
        """Test integration of distribution metrics."""
        fx_series, commodity_series = sample_data

        # Run backtest
        strategy_config = {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        }

        backtest_engine = BacktestEngine(strategy_config=strategy_config)
        results = backtest_engine.run_backtest(fx_series, commodity_series)

        # Calculate distribution metrics
        dist_analysis = DistributionAnalysis()

        # Overall distribution metrics
        overall_metrics = dist_analysis.calculate_all_distribution_metrics(
            results["returns"].dropna()
        )

        # Check that overall metrics are calculated correctly
        assert isinstance(overall_metrics, dict)

        expected_metrics = [
            "skewness",
            "kurtosis",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
        ]
        for metric in expected_metrics:
            assert metric in overall_metrics

        # Check that values are reasonable
        assert isinstance(overall_metrics["skewness"], float)
        assert isinstance(overall_metrics["kurtosis"], float)
        assert isinstance(overall_metrics["var_95"], float)
        assert isinstance(overall_metrics["var_99"], float)
        assert isinstance(overall_metrics["cvar_95"], float)
        assert isinstance(overall_metrics["cvar_99"], float)

        # VaR and CVaR should be negative
        assert overall_metrics["var_95"] <= 0
        assert overall_metrics["var_99"] <= 0
        assert overall_metrics["cvar_95"] <= 0
        assert overall_metrics["cvar_99"] <= 0

        # CVaR should be more extreme than VaR
        assert overall_metrics["cvar_95"] <= overall_metrics["var_95"]
        assert overall_metrics["cvar_99"] <= overall_metrics["var_99"]

        # 99% metrics should be more extreme than 95%
        assert overall_metrics["var_99"] <= overall_metrics["var_95"]
        assert overall_metrics["cvar_99"] <= overall_metrics["cvar_95"]

        # Rolling distribution metrics
        rolling_dist_df = dist_analysis.calculate_rolling_distribution_metrics(
            results["returns"], window=126
        )

        # Check that rolling distribution metrics are calculated correctly
        assert isinstance(rolling_dist_df, pd.DataFrame)
        assert len(rolling_dist_df) == len(results["returns"])

        for metric in expected_metrics:
            assert metric in rolling_dist_df.columns

        # Check that early values are NaN due to rolling window
        assert rolling_dist_df.iloc[:125].isna().all().all()

        # Check that later values are not all NaN
        assert not rolling_dist_df.iloc[126:].isna().all().all()

    def test_performance_report_generation(self, sample_data):
        """Test generation of comprehensive performance reports."""
        fx_series, commodity_series = sample_data

        # Run backtest
        strategy_config = {
            "lookbacks": {"beta_window": 60, "z_window": 20},
            "thresholds": {"entry_z": 1.5, "exit_z": 0.5, "stop_z": 3.0},
        }

        backtest_engine = BacktestEngine(strategy_config=strategy_config)
        results = backtest_engine.run_backtest(fx_series, commodity_series)

        # Calculate performance metrics
        rolling_metrics = RollingMetrics()
        dist_analysis = DistributionAnalysis()

        # Different window sizes for rolling metrics
        short_term_metrics = rolling_metrics.calculate_rolling_metrics(
            results["returns"], window=21
        )
        medium_term_metrics = rolling_metrics.calculate_rolling_metrics(
            results["returns"], window=63
        )
        long_term_metrics = rolling_metrics.calculate_rolling_metrics(
            results["returns"], window=252
        )

        # Distribution metrics
        dist_metrics = dist_analysis.calculate_rolling_distribution_metrics(
            results["returns"], window=126
        )
        overall_dist_metrics = dist_analysis.calculate_all_distribution_metrics(
            results["returns"].dropna()
        )

        # Create performance report
        performance_report = {
            "backtest_results": results,
            "short_term_metrics": short_term_metrics,
            "medium_term_metrics": medium_term_metrics,
            "long_term_metrics": long_term_metrics,
            "distribution_metrics": dist_metrics,
            "overall_distribution": overall_dist_metrics,
        }

        # Check that performance report is comprehensive
        assert isinstance(performance_report, dict)
        assert len(performance_report) == 6

        # Check that all components are valid
        for key, value in performance_report.items():
            if key == "overall_distribution":
                assert isinstance(value, dict)
            else:
                assert isinstance(value, (pd.DataFrame, dict))

        # Check that we can extract meaningful insights
        # For example, compare short-term vs long-term Sharpe ratios
        short_term_sharpe = short_term_metrics["sharpe"].mean()
        long_term_sharpe = long_term_metrics["sharpe"].mean()

        # Both should be reasonable numbers
        assert isinstance(short_term_sharpe, float)
        assert isinstance(long_term_sharpe, float)
        assert not np.isnan(short_term_sharpe)
        assert not np.isnan(long_term_sharpe)

        # Check that distribution metrics are consistent
        # Max drawdown from backtest should be close to min drawdown from rolling metrics
        backtest_max_dd = results["performance_metrics"]["max_drawdown"]
        rolling_max_dd = long_term_metrics["drawdown"].min()

        assert abs(rolling_max_dd - backtest_max_dd) < abs(
            backtest_max_dd * 0.1
        )  # Within 10%
