import pandas as pd

def check_atr_stop_loss(
    position: int,
    entry_price: float,
    current_price: float,
    atr: float,
    stop_multiplier: float
) -> bool:
    """
    Checks if an ATR-based stop loss has been triggered.

    Args:
        position (int): The current position (1 for long, -1 for short).
        entry_price (float): The price at which the position was entered.
        current_price (float): The current price.
        atr (float): The current Average True Range.
        stop_multiplier (float): The multiplier for the ATR to determine the stop loss.

    Returns:
        bool: True if the stop loss is triggered, False otherwise.
    """
    if position == 0:
        return False

    stop_distance = atr * stop_multiplier
    
    if position == 1: # Long position
        stop_loss_price = entry_price - stop_distance
        if current_price <= stop_loss_price:
            return True
    elif position == -1: # Short position
        stop_loss_price = entry_price + stop_distance
        if current_price >= stop_loss_price:
            return True
            
    return False

def check_trailing_stop(
    position: int,
    entry_price: float,
    peak_price: float,
    current_price: float,
    trail_amount: float
) -> bool:
    """
    Checks if a trailing stop loss has been triggered.

    Args:
        position (int): The current position (1 for long, -1 for short).
        entry_price (float): The price at which the position was entered.
        peak_price (float): The highest price reached during the trade (for longs) or lowest (for shorts).
        current_price (float): The current price.
        trail_amount (float): The amount to trail the stop loss by.

    Returns:
        bool: True if the trailing stop is triggered, False otherwise.
    """
    if position == 0:
        return False
        
    if position == 1: # Long position
        stop_loss_price = peak_price - trail_amount
        if current_price <= stop_loss_price:
            return True
    elif position == -1: # Short position
        stop_loss_price = peak_price + trail_amount
        if current_price >= stop_loss_price:
            return True
            
    return False
