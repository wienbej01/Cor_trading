"""
Execution policy module for FX-Commodity correlation arbitrage strategy.
Implements slippage models and execution cost calculations.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class ExecutionConfig:
    """Configuration for execution parameters."""

    # Slippage assumptions
    fx_slippage_bps: float = 1.0  # FX slippage in basis points
    comd_slippage_bps: float = 2.0  # Commodity slippage in basis points

    # Order types
    default_order_type: str = "limit"  # "market", "limit", or "peg"

    # Market impact model (quadratic)
    market_impact_coefficient: float = 0.1  # Market impact coefficient
    quadratic_impact_coefficient: float = 0.001  # Quadratic impact coefficient
    max_position_impact: float = 0.05  # Maximum position impact (5%)

    # Venue assumptions
    fx_venue_spread_bps: float = 0.5  # FX venue spread in basis points
    comd_venue_spread_bps: float = 1.5  # Commodity venue spread in basis points

    # Transaction costs (tiered)
    fx_small_trade_threshold: float = (
        1000000  # Threshold for small vs large FX trades ($1M)
    )
    comd_small_trade_threshold: float = (
        100  # Threshold for small vs large commodity trades (100 contracts)
    )
    fx_small_trade_cost_bps: float = 0.5  # 0.5 bps for small FX trades
    fx_large_trade_cost_bps: float = 2.0  # 2.0 bps for large FX trades
    comd_small_trade_cost_bps: float = 0.5  # 0.5 bps for small commodity trades
    comd_large_trade_cost_bps: float = 2.0  # 2.0 bps for large commodity trades

    # Legacy costs (for backward compatibility)
    fx_fixed_cost: float = 0.0001  # Fixed cost per FX trade (0.0001 for FX)
    comd_fixed_cost: float = 1.0  # Fixed cost per commodity trade ($1 for futures)
    fx_percentage_cost: float = 0.00001  # Percentage cost per FX trade (0.001%)
    comd_percentage_cost: float = (
        0.00001  # Percentage cost per commodity trade (0.001%)
    )

    # Slippage multipliers (new)
    atr_slippage_multiplier: float = 0.5  # Multiplier for ATR-based slippage
    volume_slippage_multiplier: float = 0.3  # Multiplier for volume-based slippage


class ExecutionPolicy:
    """
    Execution policy for FX-Commodity correlation arbitrage strategy.
    Implements slippage models and execution cost calculations.
    """

    def __init__(self, config: ExecutionConfig):
        """
        Initialize the execution policy.

        Args:
            config: Execution configuration parameters.
        """
        self.config = config
        logger.info("Initialized execution policy")

    def calculate_slippage(
        self,
        fx_price: float,
        comd_price: float,
        fx_position: float,
        comd_position: float,
        fx_atr: float = None,  # ATR for FX (new)
        comd_atr: float = None,  # ATR for commodity (new)
        fx_volume: float = None,  # Volume for FX (new)
        comd_volume: float = None,  # Volume for commodity (new)
        order_type: str = None,
    ) -> Tuple[float, float]:
        """
        Calculate slippage for FX and commodity positions.

        Args:
            fx_price: Current FX price.
            comd_price: Current commodity price.
            fx_position: FX position size.
            comd_position: Commodity position size.
            fx_atr: ATR for FX (optional).
            comd_atr: ATR for commodity (optional).
            fx_volume: Volume for FX (optional).
            comd_volume: Volume for commodity (optional).
            order_type: Order type ("market", "limit", "peg"). If None, uses default.

        Returns:
            Tuple of (fx_slippage, comd_slippage) in absolute terms.
        """
        if order_type is None:
            order_type = self.config.default_order_type

        # Base slippage based on order type
        if order_type == "market":
            # Market orders have higher slippage
            fx_slippage_bps = self.config.fx_slippage_bps * 2
            comd_slippage_bps = self.config.comd_slippage_bps * 2
        elif order_type == "limit":
            # Limit orders have lower slippage but may not fill
            fx_slippage_bps = self.config.fx_slippage_bps * 0.5
            comd_slippage_bps = self.config.comd_slippage_bps * 0.5
        elif order_type == "peg":
            # Peg orders have medium slippage
            fx_slippage_bps = self.config.fx_slippage_bps
            comd_slippage_bps = self.config.comd_slippage_bps
        else:
            # Default to configured slippage
            fx_slippage_bps = self.config.fx_slippage_bps
            comd_slippage_bps = self.config.comd_slippage_bps

        # Calculate market impact based on position size
        fx_impact = self._calculate_market_impact(abs(fx_position), fx_price)
        comd_impact = self._calculate_market_impact(abs(comd_position), comd_price)

        # Base slippage from venue spread + execution slippage + market impact
        fx_base_slippage = (
            self.config.fx_venue_spread_bps + fx_slippage_bps
        ) / 10000 * fx_price + fx_impact
        comd_base_slippage = (
            self.config.comd_venue_spread_bps + comd_slippage_bps
        ) / 10000 * comd_price + comd_impact

        # Enhanced slippage based on ATR if available
        if fx_atr is not None and fx_atr > 0:
            fx_atr_slippage = fx_atr * self.config.atr_slippage_multiplier
            fx_base_slippage = max(fx_base_slippage, fx_atr_slippage)

        if comd_atr is not None and comd_atr > 0:
            comd_atr_slippage = comd_atr * self.config.atr_slippage_multiplier
            comd_base_slippage = max(comd_base_slippage, comd_atr_slippage)

        # Enhanced slippage based on volume if available
        if fx_volume is not None and fx_volume > 0:
            # Volume-based slippage (simplified model)
            # Higher volume = lower slippage, lower volume = higher slippage
            volume_factor = max(
                0.1, min(1.0, 1000000 / fx_volume)
            )  # Normalize volume impact
            fx_volume_slippage = (
                fx_base_slippage
                * volume_factor
                * self.config.volume_slippage_multiplier
            )
            fx_base_slippage = max(fx_base_slippage, fx_volume_slippage)

        if comd_volume is not None and comd_volume > 0:
            # Volume-based slippage (simplified model)
            volume_factor = max(
                0.1, min(1.0, 1000000 / comd_volume)
            )  # Normalize volume impact
            comd_volume_slippage = (
                comd_base_slippage
                * volume_factor
                * self.config.volume_slippage_multiplier
            )
            comd_base_slippage = max(comd_base_slippage, comd_volume_slippage)

        return fx_base_slippage, comd_base_slippage

    def _calculate_market_impact(self, position_size: float, price: float) -> float:
        """
        Calculate market impact for a given position size using quadratic model.

        Args:
            position_size: Size of the position.
            price: Current price.

        Returns:
            Market impact in absolute terms.
        """
        # Quadratic market impact model: impact = a * sqrt(size) + b * size^2
        linear_impact = (
            self.config.market_impact_coefficient * np.sqrt(position_size) * price
        )
        quadratic_impact = (
            getattr(self.config, 'quadratic_impact_coefficient', 0.001) * (position_size**2) * price
        )

        total_impact = linear_impact + quadratic_impact

        # Cap impact at maximum
        max_impact = price * self.config.max_position_impact
        return min(total_impact, max_impact)

    def apply_slippage(
        self,
        fx_price: float,
        comd_price: float,
        fx_position: float,
        comd_position: float,
        order_type: str = None,
    ) -> Tuple[float, float]:
        """
        Apply slippage to prices based on position direction.

        Args:
            fx_price: Current FX price.
            comd_price: Current commodity price.
            fx_position: FX position size (positive for long, negative for short).
            comd_position: Commodity position size (positive for long, negative for short).
            order_type: Order type. If None, uses default.

        Returns:
            Tuple of (fx_execution_price, comd_execution_price).
        """
        fx_slippage, comd_slippage = self.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position, order_type
        )

        # Apply slippage based on position direction
        # For long positions, we buy at a higher price (negative impact)
        # For short positions, we sell at a lower price (negative impact)
        fx_execution_price = (
            fx_price + np.sign(fx_position) * fx_slippage
            if fx_position != 0
            else fx_price
        )
        comd_execution_price = (
            comd_price + np.sign(comd_position) * comd_slippage
            if comd_position != 0
            else comd_price
        )

        return fx_execution_price, comd_execution_price

    def calculate_execution_costs(
        self,
        fx_price: float,
        comd_price: float,
        fx_position: float,
        comd_position: float,
        order_type: str = None,
    ) -> float:
        """
        Calculate total execution costs including tiered transaction costs and slippage.

        Args:
            fx_price: Current FX price.
            comd_price: Current commodity price.
            fx_position: FX position size.
            comd_position: Commodity position size.
            order_type: Order type. If None, uses default.

        Returns:
            Total execution costs.
        """
        fx_slippage, comd_slippage = self.calculate_slippage(
            fx_price, comd_price, fx_position, comd_position, order_type=order_type
        )

        # Calculate slippage costs based on position size
        fx_slippage_cost = fx_slippage * abs(fx_position)
        comd_slippage_cost = comd_slippage * abs(comd_position)

        # Calculate tiered transaction costs
        fx_trade_value = abs(fx_position * fx_price)
        comd_trade_value = abs(comd_position * comd_price)

        # FX tiered costs
        if fx_trade_value <= self.config.fx_small_trade_threshold:
            fx_transaction_cost = fx_trade_value * (
                self.config.fx_small_trade_cost_bps / 10000
            )
        else:
            fx_transaction_cost = fx_trade_value * (
                self.config.fx_large_trade_cost_bps / 10000
            )

        # Commodity tiered costs
        if abs(comd_position) <= self.config.comd_small_trade_threshold:
            comd_transaction_cost = comd_trade_value * (
                self.config.comd_small_trade_cost_bps / 10000
            )
        else:
            comd_transaction_cost = comd_trade_value * (
                self.config.comd_large_trade_cost_bps / 10000
            )

        # Legacy fixed costs (for backward compatibility)
        fx_fixed_cost = self.config.fx_fixed_cost
        comd_fixed_cost = self.config.comd_fixed_cost

        # Legacy percentage costs (for backward compatibility)
        fx_percentage_cost = fx_trade_value * self.config.fx_percentage_cost
        comd_percentage_cost = comd_trade_value * self.config.comd_percentage_cost

        # Total execution costs
        total_costs = (
            fx_slippage_cost
            + comd_slippage_cost
            + fx_transaction_cost
            + comd_transaction_cost
            + fx_fixed_cost
            + comd_fixed_cost
            + fx_percentage_cost
            + comd_percentage_cost
        )

        logger.debug(
            f"Execution costs breakdown: slippage=({fx_slippage_cost:.4f}, {comd_slippage_cost:.4f}), "
            f"transaction=({fx_transaction_cost:.4f}, {comd_transaction_cost:.4f}), "
            f"fixed=({fx_fixed_cost:.4f}, {comd_fixed_cost:.4f}), "
            f"percentage=({fx_percentage_cost:.4f}, {comd_percentage_cost:.4f})"
        )

        return total_costs


def create_default_execution_config() -> ExecutionConfig:
    """
    Create a default execution configuration.

    Returns:
        Default execution configuration.
    """
    return ExecutionConfig()


def create_execution_policy(config: ExecutionConfig = None) -> ExecutionPolicy:
    """
    Create an execution policy with default or provided configuration.

    Args:
        config: Execution configuration. If None, uses default.

    Returns:
        Execution policy instance.
    """
    if config is None:
        config = ExecutionConfig()

    return ExecutionPolicy(config)
