"""
Unit tests for Risk & Execution Controls Enhanced Module.

Tests for:
- Daily maximum drawdown control
- Per-trade risk cap calculations
- Enhanced slippage and transaction cost models
- Position scaling rules based on volatility/ATR
- Integration with existing risk framework
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

from src.risk.manager import (
    RiskManager,
    RiskConfig,
    create_default_risk_config,
    create_risk_manager,
)
from src.exec.policy import (
    ExecutionPolicy,
    ExecutionConfig,
    create_default_execution_config,
    create_execution_policy,
)
from tests.test_utils import generate_synthetic_market_data, CustomAssertions


class TestRiskConfig:
    """Test RiskConfig class."""

    def test_default_config(self):
        """Test default risk configuration."""
        config = RiskConfig()

        # Check default values
        assert config.max_position_size_per_pair == 0.20
        assert config.max_daily_loss_pct == 0.02
        assert config.max_weekly_loss_pct == 0.05
        assert config.max_drawdown_pct == 0.10
        assert config.max_trade_risk == 0.01
        assert config.max_total_exposure == 2.0
        assert config.daily_drawdown_limit == 0.05
        assert config.weekly_loss_limit == 0.03
        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_cooldown == 1
        assert config.volatility_scaling is True

    def test_custom_config(self):
        """Test custom risk configuration."""
        config = RiskConfig(
            max_position_size_per_pair=0.15,
            max_daily_loss_pct=0.03,
            max_weekly_loss_pct=0.08,
            max_drawdown_pct=0.15,
            max_trade_risk=0.02,
            max_total_exposure=3.0,
            daily_drawdown_limit=0.08,
            weekly_loss_limit=0.05,
            enable_circuit_breaker=False,
            circuit_breaker_cooldown=2,
            volatility_scaling=False,
        )

        # Check custom values
        assert config.max_position_size_per_pair == 0.15
        assert config.max_daily_loss_pct == 0.03
        assert config.max_weekly_loss_pct == 0.08
        assert config.max_drawdown_pct == 0.15
        assert config.max_trade_risk == 0.02
        assert config.max_total_exposure == 3.0
        assert config.daily_drawdown_limit == 0.08
        assert config.weekly_loss_limit == 0.05
        assert config.enable_circuit_breaker is False
        assert config.circuit_breaker_cooldown == 2
        assert config.volatility_scaling is False


class TestRiskManager:
    """Test RiskManager class."""

    @pytest.fixture
    def risk_manager(self):
        """Create a RiskManager instance for testing."""
        config = RiskConfig()
        return RiskManager(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        return generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2021-12-31", seed=42
        )

    def test_initialization(self, risk_manager):
        """Test RiskManager initialization."""
        assert risk_manager.config is not None
        assert risk_manager.account_equity == 100000.0
        assert risk_manager.daily_equity_high == 100000.0
        assert risk_manager.circuit_breaker_active is False
        assert risk_manager.circuit_breaker_end_date is None
        assert risk_manager.daily_drawdown_exceeded is False
        assert isinstance(risk_manager.pair_daily_pnl, dict)
        assert isinstance(risk_manager.pair_weekly_pnl, dict)

    def test_update_equity(self, risk_manager):
        """Test equity update functionality."""
        initial_equity = risk_manager.account_equity

        # Test equity increase
        risk_manager.update_equity(105000.0)
        assert risk_manager.account_equity == 105000.0
        assert risk_manager.daily_equity_high == 105000.0

        # Test equity decrease
        risk_manager.update_equity(103000.0)
        assert risk_manager.account_equity == 103000.0
        assert risk_manager.daily_equity_high == 105000.0  # Should remain at high

    def test_update_pair_pnl(self, risk_manager):
        """Test pair P&L update functionality."""
        pair_name = "USDCAD_WTI"

        # Test daily P&L update
        risk_manager.update_pair_pnl(pair_name, 1000.0, is_daily=True)
        assert risk_manager.pair_daily_pnl[pair_name] == 1000.0

        # Test weekly P&L update
        risk_manager.update_pair_pnl(pair_name, 5000.0, is_daily=False)
        assert risk_manager.pair_weekly_pnl[pair_name] == 5000.0

        # Test cumulative updates
        risk_manager.update_pair_pnl(pair_name, -500.0, is_daily=True)
        assert risk_manager.pair_daily_pnl[pair_name] == 500.0

        risk_manager.update_pair_pnl(pair_name, -2000.0, is_daily=False)
        assert risk_manager.pair_weekly_pnl[pair_name] == 3000.0

    def test_check_daily_loss_limit(self, risk_manager):
        """Test daily loss limit check."""
        # Test within limit
        risk_manager.account_equity = 99000.0  # 1% loss
        assert not risk_manager.check_daily_loss_limit()

        # Test at limit
        risk_manager.account_equity = 98000.0  # 2% loss
        assert not risk_manager.check_daily_loss_limit()

        # Test beyond limit
        risk_manager.account_equity = 97000.0  # 3% loss
        assert risk_manager.check_daily_loss_limit()
        assert risk_manager.daily_drawdown_exceeded is True

        # Test that it stays triggered
        risk_manager.account_equity = 99000.0
        assert risk_manager.check_daily_loss_limit()  # Should still be triggered

    def test_check_weekly_loss_limit(self, risk_manager):
        """Test weekly loss limit check."""
        # Test within limit
        risk_manager.account_equity = 97000.0  # 3% loss
        assert not risk_manager.check_weekly_loss_limit()

        # Test at limit
        risk_manager.account_equity = 95000.0  # 5% loss
        assert not risk_manager.check_weekly_loss_limit()

        # Test beyond limit
        risk_manager.account_equity = 94000.0  # 6% loss
        assert risk_manager.check_weekly_loss_limit()

    def test_check_pair_limits(self, risk_manager):
        """Test pair-specific limit checks."""
        pair_name = "USDCAD_WTI"

        # Test within limits
        risk_manager.pair_daily_pnl[pair_name] = -1000.0  # 1% loss
        risk_manager.pair_weekly_pnl[pair_name] = -2000.0  # 2% loss
        assert not risk_manager.check_pair_limits(pair_name)

        # Test daily limit breach
        risk_manager.pair_daily_pnl[pair_name] = -3000.0  # 3% loss
        assert risk_manager.check_pair_limits(pair_name)

        # Test weekly limit breach
        risk_manager.pair_daily_pnl[pair_name] = -1000.0  # Reset daily
        risk_manager.pair_weekly_pnl[pair_name] = -6000.0  # 6% loss
        assert risk_manager.check_pair_limits(pair_name)

    def test_check_circuit_breaker(self, risk_manager):
        """Test circuit breaker functionality."""
        current_date = datetime(2020, 1, 1)

        # Test inactive circuit breaker
        assert not risk_manager.check_circuit_breaker(current_date)

        # Test active circuit breaker
        risk_manager.circuit_breaker_active = True
        risk_manager.circuit_breaker_end_date = current_date.date() + timedelta(days=1)
        assert risk_manager.check_circuit_breaker(current_date)

        # Test circuit breaker expiration
        future_date = current_date + timedelta(days=2)
        assert not risk_manager.check_circuit_breaker(future_date)
        assert risk_manager.circuit_breaker_active is False
        assert risk_manager.circuit_breaker_end_date is None

    def test_trigger_circuit_breaker(self, risk_manager):
        """Test circuit breaker triggering."""
        current_date = datetime(2020, 1, 1)

        # Test triggering
        risk_manager.trigger_circuit_breaker(current_date)
        assert risk_manager.circuit_breaker_active is True
        assert risk_manager.circuit_breaker_end_date == current_date.date() + timedelta(
            days=1
        )

        # Test with disabled circuit breaker
        risk_manager.config.enable_circuit_breaker = False
        risk_manager.circuit_breaker_active = False
        risk_manager.trigger_circuit_breaker(current_date)
        assert not risk_manager.circuit_breaker_active

    def test_calculate_position_size_basic(self, risk_manager, sample_data):
        """Test basic position size calculation."""
        fx_series, commodity_series = sample_data

        pair_name = "USDCAD_WTI"
        signal = 1
        fx_price = 1.30
        comd_price = 60.0
        fx_vol = 0.007
        comd_vol = 0.02
        stop_loss_distance = 0.01

        fx_position, comd_position = risk_manager.calculate_position_size(
            pair_name,
            signal,
            fx_price,
            comd_price,
            fx_vol,
            comd_vol,
            stop_loss_distance,
        )

        # Check that positions are calculated
        assert isinstance(fx_position, float)
        assert isinstance(comd_position, float)

        # Check that positions have correct sign based on signal
        assert fx_position * signal > 0
        assert comd_position * signal < 0  # Opposite side for spread

        # Check zero signal case
        fx_position_zero, comd_position_zero = risk_manager.calculate_position_size(
            pair_name, 0, fx_price, comd_price, fx_vol, comd_vol, stop_loss_distance
        )
        assert fx_position_zero == 0.0
        assert comd_position_zero == 0.0

    def test_calculate_position_size_with_atr(self, risk_manager):
        """Test position size calculation with ATR."""
        pair_name = "USDCAD_WTI"
        signal = 1
        fx_price = 1.30
        comd_price = 60.0
        fx_vol = 0.007
        comd_vol = 0.02
        stop_loss_distance = 0.01
        fx_atr = 0.005
        comd_atr = 0.015

        # Test with ATR
        fx_position_atr, comd_position_atr = risk_manager.calculate_position_size(
            pair_name,
            signal,
            fx_price,
            comd_price,
            fx_vol,
            comd_vol,
            stop_loss_distance,
            fx_atr,
            comd_atr,
        )

        # Test without ATR
        fx_position_no_atr, comd_position_no_atr = risk_manager.calculate_position_size(
            pair_name,
            signal,
            fx_price,
            comd_price,
            fx_vol,
            comd_vol,
            stop_loss_distance,
        )

        # Positions should be different when using ATR
        assert abs(fx_position_atr - fx_position_no_atr) > 1e-10
        assert abs(comd_position_atr - comd_position_no_atr) > 1e-10

    def test_calculate_position_size_volatility_scaling(self, risk_manager):
        """Test position size calculation with volatility scaling."""
        pair_name = "USDCAD_WTI"
        signal = 1
        fx_price = 1.30
        comd_price = 60.0
        fx_vol = 0.05  # High volatility
        comd_vol = 0.1  # High volatility
        stop_loss_distance = 0.01

        # Test with volatility scaling enabled
        risk_manager.config.volatility_scaling = True
        fx_position_scaled, comd_position_scaled = risk_manager.calculate_position_size(
            pair_name,
            signal,
            fx_price,
            comd_price,
            fx_vol,
            comd_vol,
            stop_loss_distance,
        )

        # Test with volatility scaling disabled
        risk_manager.config.volatility_scaling = False
        fx_position_unscaled, comd_position_unscaled = (
            risk_manager.calculate_position_size(
                pair_name,
                signal,
                fx_price,
                comd_price,
                fx_vol,
                comd_vol,
                stop_loss_distance,
            )
        )

        # Scaled positions should be smaller due to high volatility
        assert abs(fx_position_scaled) < abs(fx_position_unscaled)
        assert abs(comd_position_scaled) < abs(comd_position_unscaled)

    def test_calculate_position_size_risk_cap(self, risk_manager):
        """Test position size calculation with risk cap."""
        pair_name = "USDCAD_WTI"
        signal = 1
        fx_price = 1.30
        comd_price = 60.0
        fx_vol = 0.007
        comd_vol = 0.02
        stop_loss_distance = 0.1  # Large stop loss

        # Test with risk cap enabled
        risk_manager.config.max_trade_risk = 0.01  # 1% max risk
        fx_position_capped, comd_position_capped = risk_manager.calculate_position_size(
            pair_name,
            signal,
            fx_price,
            comd_price,
            fx_vol,
            comd_vol,
            stop_loss_distance,
        )

        # Test with risk cap disabled
        risk_manager.config.max_trade_risk = 0.0
        fx_position_uncapped, comd_position_uncapped = (
            risk_manager.calculate_position_size(
                pair_name,
                signal,
                fx_price,
                comd_price,
                fx_vol,
                comd_vol,
                stop_loss_distance,
            )
        )

        # Capped positions should be smaller due to risk limit
        assert abs(fx_position_capped) < abs(fx_position_uncapped)
        assert abs(comd_position_capped) < abs(comd_position_uncapped)

    def test_check_total_exposure(self, risk_manager):
        """Test total exposure check."""
        # Test within limit
        current_positions = {"PAIR1": (10000.0, -5000.0), "PAIR2": (8000.0, -4000.0)}
        assert not risk_manager.check_total_exposure(current_positions)

        # Test beyond limit
        current_positions = {
            "PAIR1": (100000.0, -50000.0),
            "PAIR2": (80000.0, -40000.0),
        }
        assert risk_manager.check_total_exposure(current_positions)

    def test_check_daily_drawdown_limit(self, risk_manager):
        """Test daily drawdown limit check."""
        # Test within limit
        risk_manager.account_equity = 98000.0  # 2% drawdown
        risk_manager.daily_equity_high = 100000.0
        assert not risk_manager.check_daily_drawdown_limit()

        # Test at limit
        risk_manager.account_equity = 95000.0  # 5% drawdown
        assert not risk_manager.check_daily_drawdown_limit()

        # Test beyond limit
        risk_manager.account_equity = 94000.0  # 6% drawdown
        assert risk_manager.check_daily_drawdown_limit()
        assert risk_manager.daily_drawdown_exceeded is True

    def test_can_trade_pair(self, risk_manager):
        """Test overall trading permission check."""
        pair_name = "USDCAD_WTI"
        current_date = datetime(2020, 1, 1)

        # Test normal conditions
        assert risk_manager.can_trade_pair(pair_name, current_date)

        # Test with circuit breaker active
        risk_manager.circuit_breaker_active = True
        assert not risk_manager.can_trade_pair(pair_name, current_date)

        # Reset circuit breaker
        risk_manager.circuit_breaker_active = False

        # Test with daily loss limit breached
        risk_manager.account_equity = 97000.0  # 3% loss
        assert not risk_manager.can_trade_pair(pair_name, current_date)

        # Reset equity
        risk_manager.account_equity = 100000.0

        # Test with pair limits breached
        risk_manager.pair_daily_pnl[pair_name] = -3000.0  # 3% loss
        assert not risk_manager.can_trade_pair(pair_name, current_date)

        # Reset pair P&L
        risk_manager.pair_daily_pnl[pair_name] = 0.0

        # Test with daily drawdown limit breached
        risk_manager.account_equity = 94000.0  # 6% drawdown
        risk_manager.daily_equity_high = 100000.0
        assert not risk_manager.can_trade_pair(pair_name, current_date)


class TestExecutionConfig:
    """Test ExecutionConfig class."""

    def test_default_config(self):
        """Test default execution configuration."""
        config = ExecutionConfig()

        # Check default values
        assert config.fx_slippage_bps == 1.0
        assert config.comd_slippage_bps == 2.0
        assert config.default_order_type == "limit"
        assert config.market_impact_coefficient == 0.1
        assert config.max_position_impact == 0.05
        assert config.fx_venue_spread_bps == 0.5
        assert config.comd_venue_spread_bps == 1.5
        assert config.fx_fixed_cost == 0.0001
        assert config.comd_fixed_cost == 1.0
        assert config.fx_percentage_cost == 0.00001
        assert config.comd_percentage_cost == 0.00001
        assert config.atr_slippage_multiplier == 0.5
        assert config.volume_slippage_multiplier == 0.3

    def test_custom_config(self):
        """Test custom execution configuration."""
        config = ExecutionConfig(
            fx_slippage_bps=2.0,
            comd_slippage_bps=3.0,
            default_order_type="market",
            market_impact_coefficient=0.2,
            max_position_impact=0.1,
            fx_venue_spread_bps=1.0,
            comd_venue_spread_bps=2.0,
            fx_fixed_cost=0.0002,
            comd_fixed_cost=2.0,
            fx_percentage_cost=0.00002,
            comd_percentage_cost=0.00002,
            atr_slippage_multiplier=1.0,
            volume_slippage_multiplier=0.5,
        )

        # Check custom values
        assert config.fx_slippage_bps == 2.0
        assert config.comd_slippage_bps == 3.0
        assert config.default_order_type == "market"
        assert config.market_impact_coefficient == 0.2
        assert config.max_position_impact == 0.1
        assert config.fx_venue_spread_bps == 1.0
        assert config.comd_venue_spread_bps == 2.0
        assert config.fx_fixed_cost == 0.0002
        assert config.comd_fixed_cost == 2.0
        assert config.fx_percentage_cost == 0.00002
        assert config.comd_percentage_cost == 0.00002
        assert config.atr_slippage_multiplier == 1.0
        assert config.volume_slippage_multiplier == 0.5


class TestExecutionPolicy:
    """Test ExecutionPolicy class."""

    @pytest.fixture
    def execution_policy(self):
        """Create an ExecutionPolicy instance for testing."""
        config = ExecutionConfig()
        return ExecutionPolicy(config)

    def test_initialization(self, execution_policy):
        """Test ExecutionPolicy initialization."""
        assert execution_policy.config is not None

    def test_calculate_slippage_basic(self, execution_policy):
        """Test basic slippage calculation."""
        fx_price = 1.30
        comd_price = 60.0
        fx_position = 10000.0
        comd_position = -5000.0

        fx_slippage, comd_slippage = execution_policy.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position
        )

        # Check that slippage is calculated
        assert isinstance(fx_slippage, float)
        assert isinstance(comd_slippage, float)
        assert fx_slippage >= 0
        assert comd_slippage >= 0

    def test_calculate_slippage_order_types(self, execution_policy):
        """Test slippage calculation for different order types."""
        fx_price = 1.30
        comd_price = 60.0
        fx_position = 10000.0
        comd_position = -5000.0

        # Test different order types
        fx_slippage_market, comd_slippage_market = execution_policy.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position, order_type="market"
        )

        fx_slippage_limit, comd_slippage_limit = execution_policy.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position, order_type="limit"
        )

        fx_slippage_peg, comd_slippage_peg = execution_policy.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position, order_type="peg"
        )

        # Market orders should have higher slippage than limit orders
        assert fx_slippage_market > fx_slippage_limit
        assert comd_slippage_market > comd_slippage_limit

        # Peg orders should have slippage between market and limit
        assert fx_slippage_limit < fx_slippage_peg < fx_slippage_market
        assert comd_slippage_limit < comd_slippage_peg < comd_slippage_market

    def test_calculate_slippage_with_atr(self, execution_policy):
        """Test slippage calculation with ATR."""
        fx_price = 1.30
        comd_price = 60.0
        fx_position = 10000.0
        comd_position = -5000.0
        fx_atr = 0.005
        comd_atr = 0.015

        # Test with ATR
        fx_slippage_atr, comd_slippage_atr = execution_policy.calculate_slippage(
            fx_price,
            comd_price,
            fx_position,
            comd_position,
            fx_atr=fx_atr,
            comd_atr=comd_atr,
        )

        # Test without ATR
        fx_slippage_no_atr, comd_slippage_no_atr = execution_policy.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position
        )

        # ATR should increase slippage when it's significant
        assert fx_slippage_atr >= fx_slippage_no_atr
        assert comd_slippage_atr >= comd_slippage_no_atr

    def test_calculate_slippage_with_volume(self, execution_policy):
        """Test slippage calculation with volume."""
        fx_price = 1.30
        comd_price = 60.0
        fx_position = 10000.0
        comd_position = -5000.0
        fx_volume = 500000.0  # Low volume
        comd_volume = 200000.0  # Low volume

        # Test with volume
        fx_slippage_volume, comd_slippage_volume = execution_policy.calculate_slippage(
            fx_price,
            comd_price,
            fx_position,
            comd_position,
            fx_volume=fx_volume,
            comd_volume=comd_volume,
        )

        # Test without volume
        fx_slippage_no_volume, comd_slippage_no_volume = (
            execution_policy.calculate_slippage(
                fx_price, comd_price, fx_position, comd_position
            )
        )

        # Low volume should increase slippage
        assert fx_slippage_volume >= fx_slippage_no_volume
        assert comd_slippage_volume >= comd_slippage_no_volume

    def test_calculate_market_impact(self, execution_policy):
        """Test market impact calculation."""
        # Test small position
        small_impact = execution_policy._calculate_market_impact(1000.0, 1.30)
        assert small_impact >= 0

        # Test large position
        large_impact = execution_policy._calculate_market_impact(100000.0, 1.30)
        assert large_impact > small_impact

        # Test impact capping
        max_impact = 1.30 * execution_policy.config.max_position_impact
        assert large_impact <= max_impact

    def test_apply_slippage(self, execution_policy):
        """Test slippage application to prices."""
        fx_price = 1.30
        comd_price = 60.0
        fx_position = 10000.0  # Long position
        comd_position = -5000.0  # Short position

        fx_exec_price, comd_exec_price = execution_policy.apply_slippage(
            fx_price, comd_price, fx_position, comd_position
        )

        # Long positions should have higher execution prices
        assert fx_exec_price >= fx_price

        # Short positions should have lower execution prices
        assert comd_exec_price <= comd_price

        # Test zero position
        fx_exec_zero, comd_exec_zero = execution_policy.apply_slippage(
            fx_price, comd_price, 0.0, 0.0
        )

        # Zero positions should have no slippage
        assert fx_exec_zero == fx_price
        assert comd_exec_zero == comd_price

    def test_calculate_execution_costs(self, execution_policy):
        """Test execution cost calculation."""
        fx_price = 1.30
        comd_price = 60.0
        fx_position = 10000.0
        comd_position = -5000.0

        total_costs = execution_policy.calculate_execution_costs(
            fx_price, comd_price, fx_position, comd_position
        )

        # Check that costs are calculated
        assert isinstance(total_costs, float)
        assert total_costs >= 0

        # Test with different order types
        costs_market = execution_policy.calculate_execution_costs(
            fx_price, comd_price, fx_position, comd_position, order_type="market"
        )

        costs_limit = execution_policy.calculate_execution_costs(
            fx_price, comd_price, fx_position, comd_position, order_type="limit"
        )

        # Market orders should have higher costs
        assert costs_market > costs_limit


class TestRiskExecutionIntegration:
    """Integration tests for risk and execution components."""

    def test_risk_execution_workflow(self):
        """Test integrated risk and execution workflow."""
        # Create configurations
        risk_config = RiskConfig()
        exec_config = ExecutionConfig()

        # Create managers
        risk_manager = RiskManager(risk_config)
        exec_policy = ExecutionPolicy(exec_config)

        # Test trading parameters
        pair_name = "USDCAD_WTI"
        signal = 1
        fx_price = 1.30
        comd_price = 60.0
        fx_vol = 0.007
        comd_vol = 0.02
        stop_loss_distance = 0.01
        current_date = datetime(2020, 1, 1)

        # Check if trading is allowed
        can_trade = risk_manager.can_trade_pair(pair_name, current_date)
        assert can_trade

        # Calculate position size
        fx_position, comd_position = risk_manager.calculate_position_size(
            pair_name,
            signal,
            fx_price,
            comd_price,
            fx_vol,
            comd_vol,
            stop_loss_distance,
        )

        # Apply slippage
        fx_exec_price, comd_exec_price = exec_policy.apply_slippage(
            fx_price, comd_price, fx_position, comd_position
        )

        # Calculate execution costs
        total_costs = exec_policy.calculate_execution_costs(
            fx_price, comd_price, fx_position, comd_position
        )

        # Check that all components work together
        assert fx_position != 0
        assert comd_position != 0
        assert fx_exec_price != fx_price
        assert comd_exec_price != comd_price
        assert total_costs > 0

    def test_position_scaling_with_volatility(self):
        """Test position scaling based on volatility."""
        # Create risk manager with volatility scaling
        risk_config = RiskConfig(volatility_scaling=True)
        risk_manager = RiskManager(risk_config)

        # Test with low volatility
        fx_position_low, comd_position_low = risk_manager.calculate_position_size(
            "PAIR1", 1, 1.30, 60.0, 0.005, 0.01, 0.01
        )

        # Test with high volatility
        fx_position_high, comd_position_high = risk_manager.calculate_position_size(
            "PAIR1", 1, 1.30, 60.0, 0.05, 0.1, 0.01
        )

        # High volatility should result in smaller positions
        assert abs(fx_position_high) < abs(fx_position_low)
        assert abs(comd_position_high) < abs(comd_position_low)

    def test_risk_limits_enforcement(self):
        """Test that risk limits are properly enforced."""
        # Create risk manager with tight limits
        risk_config = RiskConfig(max_trade_risk=0.001)  # Very tight risk limit
        risk_manager = RiskManager(risk_manager)

        # Test with large stop loss
        fx_position, comd_position = risk_manager.calculate_position_size(
            "PAIR1", 1, 1.30, 60.0, 0.007, 0.02, 0.1  # Large stop loss
        )

        # Positions should be very small due to risk limit
        assert abs(fx_position) < 1000.0
        assert abs(comd_position) < 1000.0


class TestFactoryFunctions:
    """Test factory functions for risk and execution components."""

    def test_create_default_risk_config(self):
        """Test default risk config creation."""
        config = create_default_risk_config()
        assert isinstance(config, RiskConfig)
        assert config.max_position_size_per_pair == 0.20

    def test_create_risk_manager(self):
        """Test risk manager creation."""
        # Test with default config
        manager = create_risk_manager()
        assert isinstance(manager, RiskManager)
        assert manager.config.max_position_size_per_pair == 0.20

        # Test with custom config
        custom_config = RiskConfig(max_position_size_per_pair=0.15)
        manager = create_risk_manager(custom_config)
        assert manager.config.max_position_size_per_pair == 0.15

    def test_create_default_execution_config(self):
        """Test default execution config creation."""
        config = create_default_execution_config()
        assert isinstance(config, ExecutionConfig)
        assert config.fx_slippage_bps == 1.0

    def test_create_execution_policy(self):
        """Test execution policy creation."""
        # Test with default config
        policy = create_execution_policy()
        assert isinstance(policy, ExecutionPolicy)
        assert policy.config.fx_slippage_bps == 1.0

        # Test with custom config
        custom_config = ExecutionConfig(fx_slippage_bps=2.0)
        policy = create_execution_policy(custom_config)
        assert policy.config.fx_slippage_bps == 2.0
