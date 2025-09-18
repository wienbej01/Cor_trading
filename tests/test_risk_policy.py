import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.risk.policy import RiskPolicy

@pytest.fixture
def sample_equity_curve():
    """Creates a sample equity curve for testing."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    equity = pd.Series(range(100000, 100000 + 100 * 10, 10), index=dates)
    return equity

def test_cooldown_policy(sample_equity_curve):
    """Tests the trade cooldown policy."""
    policy_config = {'trade_cooldown': 2}
    policy = RiskPolicy(policy_config)
    
    timestamp1 = sample_equity_curve.index[10]
    policy.record_trade(timestamp1)
    
    # Should not be able to trade immediately after
    timestamp2 = sample_equity_curve.index[11]
    assert not policy.can_trade(timestamp2, 101000, sample_equity_curve.loc[:timestamp2])
    
    # Should be able to trade after cooldown
    timestamp3 = sample_equity_curve.index[13]
    assert policy.can_trade(timestamp3, 101000, sample_equity_curve.loc[:timestamp3])

def test_daily_loss_stop(sample_equity_curve):
    """Tests the daily loss stop policy."""
    policy_config = {'daily_loss_stop': {'enabled': True, 'threshold': 0.01}}
    policy = RiskPolicy(policy_config)
    
    timestamp = sample_equity_curve.index[20]
    
    # Simulate a loss
    current_equity = 98000 # More than 1% loss from 100000 + 20*10 = 100200
    
    assert not policy.can_trade(timestamp, current_equity, sample_equity_curve.loc[:timestamp])
    assert policy.is_halted_daily_loss

def test_max_drawdown_stop(sample_equity_curve):
    """Tests the max drawdown stop policy."""
    policy_config = {'max_drawdown_stop': {'enabled': True, 'threshold': 0.05, 'lookback': 50}}
    policy = RiskPolicy(policy_config)
    
    # Create a drawdown
    equity_with_dd = sample_equity_curve.copy()
    equity_with_dd.iloc[30:] = 95000
    
    timestamp = equity_with_dd.index[40]
    current_equity = 95000
    
    assert not policy.can_trade(timestamp, current_equity, equity_with_dd.loc[:timestamp])
    assert policy.is_halted_mdd

def test_can_trade_no_rules(sample_equity_curve):
    """Tests that trading is always allowed when no rules are enabled."""
    policy = RiskPolicy({})
    timestamp = sample_equity_curve.index[50]
    assert policy.can_trade(timestamp, 105000, sample_equity_curve.loc[:timestamp])
