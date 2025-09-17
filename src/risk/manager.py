"""
Risk management module for FX-Commodity correlation arbitrage strategy.
Implements risk limits, position sizing, and circuit breakers.
"""

from typing import Dict, Tuple

from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger


@dataclass
class RiskConfig:
    """Configuration for risk management parameters."""

    # Drawdown limits
    max_drawdown: float = 0.15  # 15% maximum drawdown
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    weekly_loss_limit: float = 0.05  # 5% weekly loss limit
    daily_drawdown_limit: float = 0.02  # 2% daily drawdown limit (new)

    # Position sizing
    max_position_size_per_pair: float = 0.10  # 10% of equity per pair
    max_total_exposure: float = 0.50  # 50% total portfolio exposure
    volatility_scaling: bool = True  # Scale positions with volatility
    max_trade_risk: float = 0.01  # 1% of portfolio per trade (new)

    # Circuit breakers
    enable_circuit_breaker: bool = True
    circuit_breaker_cooldown: int = 1  # Days to wait after circuit breaker triggered


@dataclass
class PositionLimits:
    """Position limits for a specific pair."""

    max_position_size: float  # Maximum position size as % of equity
    max_daily_loss: float  # Maximum daily loss for this pair
    max_weekly_loss: float  # Maximum weekly loss for this pair


class RiskManager:
    """
    Risk manager for FX-Commodity correlation arbitrage strategy.
    Implements risk limits, position sizing, and circuit breakers.
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize the risk manager.

        Args:
            config: Risk configuration parameters.
        """
        self.config = config
        self.account_equity = 100000.0  # Starting equity
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.circuit_breaker_active = False
        self.circuit_breaker_end_date = None
        self.pair_daily_pnl: Dict[str, float] = {}
        self.pair_weekly_pnl: Dict[str, float] = {}
        self.daily_equity_high = (
            self.account_equity
        )  # Track daily equity high for drawdown
        self.daily_drawdown_exceeded = False  # Track if daily drawdown limit exceeded

        logger.info("Initialized risk manager")

    def update_account_state(
        self, equity: float, current_date: datetime, pair_pnl: Dict[str, float] = None
    ) -> None:
        """
        Update account state with current equity and PnL.

        Args:
            equity: Current account equity.
            current_date: Current date.
            pair_pnl: Dictionary of PnL by pair.
        """
        # Update equity
        self.account_equity = equity

        # Reset daily/weekly PnL if needed
        self._reset_pnl_if_needed(current_date)

        # Update daily equity high
        if equity > self.daily_equity_high:
            self.daily_equity_high = equity

        # Update PnL tracking
        if pair_pnl:
            for pair, pnl in pair_pnl.items():
                self.pair_daily_pnl[pair] = self.pair_daily_pnl.get(pair, 0.0) + pnl
                self.pair_weekly_pnl[pair] = self.pair_weekly_pnl.get(pair, 0.0) + pnl
                self.daily_pnl += pnl
                self.weekly_pnl += pnl

    def _reset_pnl_if_needed(self, current_date: datetime) -> None:
        """
        Reset daily/weekly PnL counters if needed.

        Args:
            current_date: Current date.
        """
        current_date = current_date.date()

        # Reset daily PnL if new day
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.pair_daily_pnl = {}
            self.last_reset_date = current_date
            self.daily_equity_high = self.account_equity  # Reset daily equity high
            self.daily_drawdown_exceeded = False  # Reset daily drawdown flag

            # Reset weekly PnL if new week (Monday)
            if current_date.weekday() == 0 and current_date > self.last_reset_date:
                self.weekly_pnl = 0.0
                self.pair_weekly_pnl = {}

    def check_drawdown_limit(self, equity_series: pd.Series) -> bool:
        """
        Check if drawdown limit has been breached.

        Args:
            equity_series: Series of equity values.

        Returns:
            True if drawdown limit is breached, False otherwise.
        """
        if len(equity_series) < 2:
            return False

        # Calculate current drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        current_drawdown = drawdown.iloc[-1]

        # Check against limit
        drawdown_breached = current_drawdown < -self.config.max_drawdown

        if drawdown_breached:
            logger.warning(
                f"Drawdown limit breached: {current_drawdown:.2%} < {-self.config.max_drawdown:.2%}"
            )

        return drawdown_breached

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been breached.

        Returns:
            True if daily loss limit is breached, False otherwise.
        """
        daily_loss = (
            self.daily_pnl / self.account_equity if self.account_equity > 0 else 0
        )
        daily_loss_breached = daily_loss < -self.config.daily_loss_limit

        if daily_loss_breached:
            logger.warning(
                f"Daily loss limit breached: {daily_loss:.2%} < {-self.config.daily_loss_limit:.2%}"
            )

        return daily_loss_breached

    def check_weekly_loss_limit(self) -> bool:
        """
        Check if weekly loss limit has been breached.

        Returns:
            True if weekly loss limit is breached, False otherwise.
        """
        weekly_loss = (
            self.weekly_pnl / self.account_equity if self.account_equity > 0 else 0
        )
        weekly_loss_breached = weekly_loss < -self.config.weekly_loss_limit

        if weekly_loss_breached:
            logger.warning(
                f"Weekly loss limit breached: {weekly_loss:.2%} < {-self.config.weekly_loss_limit:.2%}"
            )

        return weekly_loss_breached

    def check_pair_limits(self, pair_name: str) -> bool:
        """
        Check if pair-specific limits have been breached.

        Args:
            pair_name: Name of the pair to check.

        Returns:
            True if pair limits are breached, False otherwise.
        """
        # Check pair daily loss limit
        pair_daily_pnl = self.pair_daily_pnl.get(pair_name, 0.0)
        pair_daily_loss = (
            pair_daily_pnl / self.account_equity if self.account_equity > 0 else 0
        )
        daily_limit_breached = pair_daily_loss < -self.config.daily_loss_limit

        if daily_limit_breached:
            logger.warning(
                f"Pair {pair_name} daily loss limit breached: {pair_daily_loss:.2%}"
            )

        # Check pair weekly loss limit
        pair_weekly_pnl = self.pair_weekly_pnl.get(pair_name, 0.0)
        pair_weekly_loss = (
            pair_weekly_pnl / self.account_equity if self.account_equity > 0 else 0
        )
        weekly_limit_breached = pair_weekly_loss < -self.config.weekly_loss_limit

        if weekly_limit_breached:
            logger.warning(
                f"Pair {pair_name} weekly loss limit breached: {pair_weekly_loss:.2%}"
            )

        return daily_limit_breached or weekly_limit_breached

    def check_circuit_breaker(self, current_date: datetime) -> bool:
        """
        Check if circuit breaker is active.

        Args:
            current_date: Current date.

        Returns:
            True if circuit breaker is active, False otherwise.
        """
        if not self.config.enable_circuit_breaker:
            return False

        # Check if circuit breaker has expired
        if self.circuit_breaker_active and self.circuit_breaker_end_date:
            if current_date.date() >= self.circuit_breaker_end_date:
                self.circuit_breaker_active = False
                self.circuit_breaker_end_date = None
                logger.info("Circuit breaker deactivated")

        return self.circuit_breaker_active

    def trigger_circuit_breaker(self, current_date: datetime) -> None:
        """
        Trigger the circuit breaker.

        Args:
            current_date: Current date.
        """
        if not self.config.enable_circuit_breaker:
            return

        self.circuit_breaker_active = True
        self.circuit_breaker_end_date = current_date.date() + timedelta(
            days=self.config.circuit_breaker_cooldown
        )
        logger.warning(
            f"Circuit breaker triggered until {self.circuit_breaker_end_date}"
        )

    def calculate_position_size(
        self,
        pair_name: str,
        signal: int,
        fx_price: float,
        comd_price: float,
        fx_vol: float,
        comd_vol: float,
        stop_loss_distance: float,  # New parameter for stop loss distance
        fx_atr: float = None,  # ATR for FX (new)
        comd_atr: float = None,  # ATR for commodity (new)
        target_vol_per_leg: float = 0.01,
    ) -> Tuple[float, float]:
        """
        Calculate position sizes with risk management.

        Args:
            pair_name: Name of the pair.
            signal: Trading signal (-1, 0, 1).
            fx_price: Current FX price.
            comd_price: Current commodity price.
            fx_vol: FX volatility.
            comd_vol: Commodity volatility.
            stop_loss_distance: Stop loss distance in price terms.
            fx_atr: ATR for FX (optional).
            comd_atr: ATR for commodity (optional).
            target_vol_per_leg: Target volatility per leg.

        Returns:
            Tuple of (fx_position_size, comd_position_size).
        """
        if signal == 0:
            return 0.0, 0.0

        # Base position size based on volatility targeting
        # Use ATR if available, otherwise use regular volatility
        if fx_atr is not None and fx_atr > 0:
            fx_size = target_vol_per_leg / fx_atr
        else:
            fx_size = target_vol_per_leg / fx_vol if fx_vol > 0 else 0

        if comd_atr is not None and comd_atr > 0:
            comd_size = target_vol_per_leg / comd_atr
        else:
            comd_size = target_vol_per_leg / comd_vol if comd_vol > 0 else 0

        # Adjust for FX quote currency if needed
        # This would normally come from config, but we'll assume it's needed for FX pairs
        fx_size = fx_size / fx_price

        # Scale positions with volatility if enabled
        if self.config.volatility_scaling and (
            fx_vol > target_vol_per_leg or comd_vol > target_vol_per_leg
        ):
            vol_scaling_factor = min(1.0, target_vol_per_leg / max(fx_vol, comd_vol))
            fx_size *= vol_scaling_factor
            comd_size *= vol_scaling_factor
            logger.debug(f"Volatility scaling applied: {vol_scaling_factor:.3f}")

        # Apply position limits
        max_position_value = (
            self.account_equity * self.config.max_position_size_per_pair
        )
        current_position_value = abs(fx_size * fx_price) + abs(comd_size * comd_price)

        if current_position_value > max_position_value and max_position_value > 0:
            sizing_factor = max_position_value / current_position_value
            fx_size *= sizing_factor
            comd_size *= sizing_factor
            logger.debug(f"Position sizing adjusted: {sizing_factor:.3f}")

        # Apply per-trade risk cap
        if stop_loss_distance > 0 and self.config.max_trade_risk > 0:
            # Calculate current risk per leg
            fx_risk = abs(fx_size * fx_price * stop_loss_distance)
            comd_risk = abs(comd_size * comd_price * stop_loss_distance)
            total_risk = fx_risk + comd_risk

            # Maximum allowed risk
            max_allowed_risk = self.account_equity * self.config.max_trade_risk

            # If current risk exceeds maximum, scale down positions
            if total_risk > max_allowed_risk and total_risk > 0:
                risk_scaling_factor = max_allowed_risk / total_risk
                fx_size *= risk_scaling_factor
                comd_size *= risk_scaling_factor
                logger.debug(f"Risk cap applied: {risk_scaling_factor:.3f}")

        # Apply signal direction
        fx_position = fx_size * signal
        comd_position = -comd_size * signal  # Opposite side of spread

        return fx_position, comd_position

    def check_total_exposure(
        self, current_positions: Dict[str, Tuple[float, float]]
    ) -> bool:
        """
        Check if total exposure limit has been breached.

        Args:
            current_positions: Dictionary of current positions by pair.

        Returns:
            True if exposure limit is breached, False otherwise.
        """
        total_exposure = 0.0

        for fx_pos, comd_pos in current_positions.values():
            # This is a simplified calculation - in practice, you'd need current prices
            total_exposure += abs(fx_pos) + abs(comd_pos)

        exposure_ratio = (
            total_exposure / self.account_equity if self.account_equity > 0 else 0
        )
        exposure_breached = exposure_ratio > self.config.max_total_exposure

        if exposure_breached:
            logger.warning(
                f"Total exposure limit breached: {exposure_ratio:.2%} > {self.config.max_total_exposure:.2%}"
            )

        return exposure_breached

    def check_daily_drawdown_limit(self) -> bool:
        """
        Check if daily drawdown limit has been breached.

        Returns:
            True if daily drawdown limit is breached, False otherwise.
        """
        if self.daily_equity_high <= 0:
            return False

        # Calculate current daily drawdown
        daily_drawdown = (
            self.account_equity - self.daily_equity_high
        ) / self.daily_equity_high
        daily_drawdown_breached = daily_drawdown < -self.config.daily_drawdown_limit

        if daily_drawdown_breached and not self.daily_drawdown_exceeded:
            self.daily_drawdown_exceeded = True
            logger.warning(
                f"Daily drawdown limit breached: {daily_drawdown:.2%} < {-self.config.daily_drawdown_limit:.2%}"
            )

        return daily_drawdown_breached

    def can_trade_pair(self, pair_name: str, current_date: datetime) -> bool:
        """
        Check if trading is allowed for a specific pair.

        Args:
            pair_name: Name of the pair.
            current_date: Current date.

        Returns:
            True if trading is allowed, False otherwise.
        """
        # Check circuit breaker
        if self.check_circuit_breaker(current_date):
            return False

        # Check pair-specific limits
        if self.check_pair_limits(pair_name):
            return False

        # Check account-level limits
        if self.check_daily_loss_limit() or self.check_weekly_loss_limit():
            return False

        # Check daily drawdown limit
        if self.check_daily_drawdown_limit():
            return False

        return True


def create_default_risk_config() -> RiskConfig:
    """
    Create a default risk configuration.

    Returns:
        Default risk configuration.
    """
    return RiskConfig()


def create_risk_manager(config: RiskConfig = None) -> RiskManager:
    """
    Create a risk manager with default or provided configuration.

    Args:
        config: Risk configuration. If None, uses default.

    Returns:
        Risk manager instance.
    """
    if config is None:
        config = create_default_risk_config()

    return RiskManager(config)
