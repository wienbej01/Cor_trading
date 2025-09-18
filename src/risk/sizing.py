import numpy as np
import pandas as pd
from loguru import logger

def calculate_inverse_volatility_size(
    target_volatility: float,
    instrument_volatility: float,
    max_position_size: float,
    min_position_size: float,
    equity: float,
    price: float
) -> float:
    """
    Calculates position size based on inverse volatility.

    Args:
        target_volatility (float): The desired volatility contribution of the position.
        instrument_volatility (float): The volatility of the instrument (e.g., ATR).
        max_position_size (float): The maximum allowed position size as a fraction of equity.
        min_position_size (float): The minimum allowed position size as a fraction of equity.
        equity (float): The current portfolio equity.
        price (float): The current price of the instrument.

    Returns:
        float: The calculated position size in units of the instrument.
    """
    if instrument_volatility <= 1e-8:
        return 0.0

    # Calculate desired position size in terms of portfolio fraction
    size_fraction = target_volatility / instrument_volatility
    
    # Apply caps and floors
    size_fraction = np.clip(size_fraction, min_position_size, max_position_size)
    
    # Convert fraction to number of units
    position_size_in_currency = equity * size_fraction
    position_size_in_units = position_size_in_currency / price
    
    return position_size_in_units

def calculate_kelly_criterion_size(
    win_probability: float,
    win_loss_ratio: float,
    kelly_fraction: float,
    max_position_size: float,
    equity: float
) -> float:
    """
    Calculates position size based on the Kelly criterion.

    Args:
        win_probability (float): The probability of a winning trade.
        win_loss_ratio (float): The average gain of a winning trade / average loss of a losing trade.
        kelly_fraction (float): The fraction of the Kelly bet to take.
        max_position_size (float): The maximum allowed position size as a fraction of equity.
        equity (float): The current portfolio equity.

    Returns:
        float: The calculated position size as a fraction of equity.
    """
    if win_loss_ratio <= 0:
        return 0.0
        
    # Kelly formula: K% = W - [(1 - W) / R]
    kelly_percentage = win_probability - ((1 - win_probability) / win_loss_ratio)
    
    # Apply Kelly fraction and cap
    size_fraction = kelly_fraction * kelly_percentage
    size_fraction = np.clip(size_fraction, 0, max_position_size)
    
    return size_fraction

class PositionSizer:
    """
    Manages position sizing based on a configured policy.
    """
    def __init__(self, sizing_config: dict):
        self.config = sizing_config
        self.method = self.config.get('method', 'inverse_volatility')

    def calculate_size(self, context: dict) -> float:
        """
        Calculates the position size for a trade.

        Args:
            context (dict): A dictionary containing necessary information for sizing,
                            e.g., {'equity': 100000, 'price': 1.35, 'instrument_volatility': 0.01,
                                   'win_prob': 0.6, 'win_loss_ratio': 1.5}.

        Returns:
            float: The calculated position size in units.
        """
        if self.method == 'inverse_volatility':
            return calculate_inverse_volatility_size(
                target_volatility=self.config['target_volatility'],
                instrument_volatility=context['instrument_volatility'],
                max_position_size=self.config['max_position_size'],
                min_position_size=self.config['min_position_size'],
                equity=context['equity'],
                price=context['price']
            )
        elif self.method == 'kelly_criterion':
            # Note: Kelly requires win probability and win/loss ratio, which may not be
            # readily available. This is more for demonstration.
            size_fraction = calculate_kelly_criterion_size(
                win_probability=context['win_prob'],
                win_loss_ratio=context['win_loss_ratio'],
                kelly_fraction=self.config['kelly_fraction'],
                max_position_size=self.config['max_position_size'],
                equity=context['equity']
            )
            return (context['equity'] * size_fraction) / context['price']
        elif self.method == 'fixed_fractional':
            size_fraction = self.config.get('fixed_fraction', 0.01)
            return (context['equity'] * size_fraction) / context['price']
        else:
            logger.warning(f"Unknown sizing method: {self.method}. Defaulting to 0.")
            return 0.0
