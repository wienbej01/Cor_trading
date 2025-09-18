from datetime import timedelta
import pandas as pd
from loguru import logger

class RiskPolicy:
    """
    Enforces risk policies for a trading strategy.
    """
    def __init__(self, policy_config: dict, portfolio_equity: pd.Series):
        """
        Initializes the RiskPolicy.

        Args:
            policy_config (dict): The configuration for the risk policies.
            portfolio_equity (pd.Series): A series of the portfolio's equity over time.
        """
        self.config = policy_config
        self.equity = portfolio_equity
        self.last_trade_timestamp = None
        self.is_halted_daily_loss = False
        self.is_halted_mdd = False
        self.last_day_checked = None

    def can_trade(self, timestamp: pd.Timestamp, current_equity: float) -> bool:
        """
        Checks if a trade is allowed at the given timestamp based on all risk policies.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
            current_equity (float): The current portfolio equity.

        Returns:
            bool: True if a trade is allowed, False otherwise.
        """
        # Reset daily halt at the start of a new day
        if self.last_day_checked and timestamp.date() > self.last_day_checked:
            self.is_halted_daily_loss = False
        self.last_day_checked = timestamp.date()

        if self.is_halted_daily_loss:
            logger.warning(f"Trade blocked at {timestamp}: Halted due to daily loss stop.")
            return False
        
        if self.is_halted_mdd:
            logger.warning(f"Trade blocked at {timestamp}: Halted due to max drawdown stop.")
            return False

        if not self._check_cooldown(timestamp):
            return False
            
        if not self._check_daily_loss_stop(timestamp, current_equity):
            self.is_halted_daily_loss = True
            return False
            
        if not self._check_max_drawdown_stop(timestamp, current_equity):
            self.is_halted_mdd = True
            return False

        return True

    def record_trade(self, timestamp: pd.Timestamp):
        """Records the timestamp of a trade to enforce cooldowns."""
        self.last_trade_timestamp = timestamp

    def _check_cooldown(self, timestamp: pd.Timestamp) -> bool:
        """Checks if the trade cooldown period has passed."""
        cooldown_bars = self.config.get('trade_cooldown', 0)
        if cooldown_bars > 0 and self.last_trade_timestamp:
            # This is a simplification. In a real backtest, we'd need to know the bar frequency.
            # Assuming daily bars for now.
            if timestamp - self.last_trade_timestamp < timedelta(days=cooldown_bars):
                logger.debug(f"Trade blocked at {timestamp}: In cooldown period.")
                return False
        return True

    def _check_daily_loss_stop(self, timestamp: pd.Timestamp, current_equity: float) -> bool:
        """Checks if the daily loss stop has been breached."""
        daily_loss_config = self.config.get('daily_loss_stop', {})
        if not daily_loss_config.get('enabled', False):
            return True

        # Find the equity at the start of the day. This is a simplification.
        day_start_equity = self.equity.loc[self.equity.index.date == timestamp.date()][0]
        if pd.isna(day_start_equity):
             day_start_equity = self.equity.iloc[0] # Fallback for first day

        daily_drawdown = (current_equity / day_start_equity) - 1
        
        if daily_drawdown < -daily_loss_config['threshold']:
            logger.warning(f"Daily loss stop breached at {timestamp}. Drawdown: {daily_drawdown:.2%}")
            return False
        return True

    def _check_max_drawdown_stop(self, timestamp: pd.Timestamp, current_equity: float, equity_history: pd.Series) -> bool:
        """Checks if the max drawdown stop has been breached."""
        mdd_config = self.config.get('max_drawdown_stop', {})
        if not mdd_config.get('enabled', False) or equity_history.empty:
            return True
            
        lookback = mdd_config.get('lookback', 252)
        equity_window = equity_history.iloc[-lookback:]
        peak = equity_window.max()
        drawdown = (current_equity / peak) - 1
        
        if drawdown < -mdd_config['threshold']:
            logger.warning(f"Max drawdown stop breached at {timestamp}. Drawdown: {drawdown:.2%}")
            return False
        return True
