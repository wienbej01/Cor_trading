"""
Unit tests for risk management and execution controls.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.risk.manager import RiskManager, RiskConfig
from src.exec.policy import ExecutionPolicy, ExecutionConfig


class TestRiskExecutionControls(unittest.TestCase):
    """Test cases for risk management and execution controls."""

    def setUp(self):
        """Set up test fixtures."""
        # Create risk config with test values
        self.risk_config = RiskConfig(
            max_drawdown=0.15,
            daily_loss_limit=0.02,
            weekly_loss_limit=0.05,
            daily_drawdown_limit=0.02,
            max_trade_risk=0.01,
            max_position_size_per_pair=0.10,
            max_total_exposure=0.50,
            volatility_scaling=True,
        )

        # Create execution config with test values
        self.exec_config = ExecutionConfig(
            fx_slippage_bps=1.0,
            comd_slippage_bps=2.0,
            fx_fixed_cost=0.0001,
            comd_fixed_cost=1.0,
            fx_percentage_cost=0.00001,
            comd_percentage_cost=0.00001,
            atr_slippage_multiplier=0.5,
            volume_slippage_multiplier=0.3,
        )

        # Create risk manager and execution policy
        self.risk_manager = RiskManager(self.risk_config)
        self.exec_policy = ExecutionPolicy(self.exec_config)

        # Test data
        self.fx_price = 1.25
        self.comd_price = 50.0
        self.fx_vol = 0.01
        self.comd_vol = 0.02
        self.fx_atr = 0.005
        self.comd_atr = 0.01
        self.stop_loss_distance = 0.005

    def test_daily_drawdown_limit(self):
        """Test daily drawdown limit functionality."""
        # Set initial equity
        initial_equity = 100000.0
        self.risk_manager.account_equity = initial_equity
        self.risk_manager.daily_equity_high = initial_equity

        # Update with a loss that exceeds daily drawdown limit
        new_equity = initial_equity * (1 - 0.03)  # 3% loss
        current_date = datetime.now()
        self.risk_manager.update_account_state(new_equity, current_date)

        # Check that daily drawdown limit is breached
        self.assertTrue(self.risk_manager.check_daily_drawdown_limit())

        # Check that trading is not allowed
        self.assertFalse(self.risk_manager.can_trade_pair("test_pair", current_date))

    def test_per_trade_risk_cap(self):
        """Test per-trade risk cap calculations."""
        # Calculate position size with risk cap
        fx_position, comd_position = self.risk_manager.calculate_position_size(
            "test_pair",
            1,
            self.fx_price,
            self.comd_price,
            self.fx_vol,
            self.comd_vol,
            self.stop_loss_distance,
            self.fx_atr,
            self.comd_atr,
        )

        # Calculate the risk for this position
        fx_risk = abs(fx_position * self.fx_price * self.stop_loss_distance)
        comd_risk = abs(comd_position * self.comd_price * self.stop_loss_distance)
        total_risk = fx_risk + comd_risk

        # Check that risk does not exceed the cap (1% of equity)
        max_allowed_risk = (
            self.risk_manager.account_equity * self.risk_config.max_trade_risk
        )
        self.assertLessEqual(total_risk, max_allowed_risk)

    def test_transaction_cost_calculation(self):
        """Test transaction cost calculations."""
        # Calculate position sizes
        fx_position, comd_position = self.risk_manager.calculate_position_size(
            "test_pair",
            1,
            self.fx_price,
            self.comd_price,
            self.fx_vol,
            self.comd_vol,
            self.stop_loss_distance,
            self.fx_atr,
            self.comd_atr,
        )

        # Calculate execution costs
        costs = self.exec_policy.calculate_execution_costs(
            self.fx_price, self.comd_price, fx_position, comd_position
        )

        # Check that costs are positive
        self.assertGreater(costs, 0)

        # Check that costs include all components
        # Fixed costs
        fixed_costs = self.exec_config.fx_fixed_cost + self.exec_config.comd_fixed_cost
        self.assertGreaterEqual(costs, fixed_costs)

    def test_position_scaling_with_volatility(self):
        """Test position scaling based on volatility."""
        # Calculate position size with normal volatility
        fx_position_normal, comd_position_normal = (
            self.risk_manager.calculate_position_size(
                "test_pair",
                1,
                self.fx_price,
                self.comd_price,
                self.fx_vol,
                self.comd_vol,
                self.stop_loss_distance,
                self.fx_atr,
                self.comd_atr,
            )
        )

        # Calculate position size with high volatility (2x)
        high_fx_vol = self.fx_vol * 2
        high_comd_vol = self.comd_vol * 2
        fx_position_high, comd_position_high = (
            self.risk_manager.calculate_position_size(
                "test_pair",
                1,
                self.fx_price,
                self.comd_price,
                high_fx_vol,
                high_comd_vol,
                self.stop_loss_distance,
                self.fx_atr * 2,
                self.comd_atr * 2,
            )
        )

        # Check that positions are smaller with higher volatility
        self.assertLess(abs(fx_position_high), abs(fx_position_normal))
        self.assertLess(abs(comd_position_high), abs(comd_position_normal))

    def test_slippage_with_atr(self):
        """Test slippage calculation with ATR."""
        # Calculate slippage without ATR
        slippage_no_atr = self.exec_policy.calculate_slippage(
            self.fx_price, self.comd_price, 1000, 100
        )

        # Calculate slippage with ATR
        slippage_with_atr = self.exec_policy.calculate_slippage(
            self.fx_price, self.comd_price, 1000, 100, self.fx_atr, self.comd_atr
        )

        # Check that slippage with ATR is at least as large as without ATR
        self.assertGreaterEqual(slippage_with_atr[0], slippage_no_atr[0])
        self.assertGreaterEqual(slippage_with_atr[1], slippage_no_atr[1])

    def test_integration_with_existing_risk_controls(self):
        """Test integration with existing risk controls."""
        # Test that all risk controls work together
        current_date = datetime.now()

        # Check that trading is allowed initially
        self.assertTrue(self.risk_manager.can_trade_pair("test_pair", current_date))

        # Update account state with normal conditions
        self.risk_manager.update_account_state(100000, current_date)

        # Check that trading is still allowed
        self.assertTrue(self.risk_manager.can_trade_pair("test_pair", current_date))

        # Calculate position size
        fx_position, comd_position = self.risk_manager.calculate_position_size(
            "test_pair",
            1,
            self.fx_price,
            self.comd_price,
            self.fx_vol,
            self.comd_vol,
            self.stop_loss_distance,
            self.fx_atr,
            self.comd_atr,
        )

        # Check that positions are calculated correctly
        self.assertIsInstance(fx_position, float)
        self.assertIsInstance(comd_position, float)


if __name__ == "__main__":
    unittest.main()
