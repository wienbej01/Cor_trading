"""
Backtest engine module for FX-Commodity correlation arbitrage strategy.
Implements backtesting with one-bar execution delay and performance statistics.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Opt-in to future pandas behavior
pd.set_option('future.no_silent_downcasting', True)


def _safe_cagr(eq: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR) with safe handling of edge cases.
    
    Args:
        eq: Series of equity values.
        
    Returns:
        CAGR as a float, or 0.0 if calculation is not possible.
    """
    n = len(eq)
    if n < 2:
        return 0.0
    ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = n / 252.0
    if years <= 0 or eq.iloc[0] <= 0:
        return 0.0
    try:
        return (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1.0
    except Exception:
        return ret


def _safe_sharpe(p: pd.Series, ann_factor=252) -> float:
    """
    Calculate Sharpe ratio with safe handling of edge cases.
    
    Args:
        p: Series of returns.
        ann_factor: Annualization factor (default 252 for daily data).
        
    Returns:
        Sharpe ratio as a float, or 0.0 if calculation is not possible.
    """
    mu = p.mean()
    sd = p.std(ddof=0)
    if sd <= 1e-12:
        return 0.0
    return float((mu * ann_factor) / (sd * np.sqrt(ann_factor)))


def backtest_pair(
    df: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_bars: int,
    inverse_fx_for_quote_ccy_strength: bool
) -> pd.DataFrame:
    """
    Backtest a single FX-Commodity pair with one-bar execution delay.
    
    Args:
        df: DataFrame with signals and market data.
        entry_z: Z-score threshold for entry.
        exit_z: Z-score threshold for exit.
        stop_z: Z-score threshold for stop loss.
        max_bars: Maximum number of bars to hold a position.
        inverse_fx_for_quote_ccy_strength: Whether to inverse FX for quote currency strength.
        
    Returns:
        DataFrame with backtest results and performance metrics.
        
    Raises:
        ValueError: If required columns are missing or parameters are invalid.
    """
    logger.info("Running backtest with one-bar execution delay")
    
    # Validate input
    required_columns = ["fx_price", "comd_price", "signal", "fx_position", "comd_position", "spread_z"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if stop_z <= entry_z:
        raise ValueError("Stop Z must be greater than entry Z")
    
    if entry_z <= exit_z:
        raise ValueError("Entry Z must be greater than exit Z")
    
    # Create result DataFrame
    result = df.copy()
    
    # Generate trading signals based on z-score thresholds
    result["entry_signal"] = 0
    result["exit_signal"] = 0
    result["stop_signal"] = 0
    
    # Long entry when spread_z > entry_z
    result.loc[result["spread_z"] > entry_z, "entry_signal"] = 1
    # Short entry when spread_z < -entry_z
    result.loc[result["spread_z"] < -entry_z, "entry_signal"] = -1
    
    # Long exit when spread_z < exit_z
    result.loc[(result["spread_z"] < exit_z) & (result["spread_z"] > 0), "exit_signal"] = 1
    # Short exit when spread_z > -exit_z
    result.loc[(result["spread_z"] > -exit_z) & (result["spread_z"] < 0), "exit_signal"] = -1
    
    # Stop loss when spread_z > stop_z (for long) or spread_z < -stop_z (for short)
    result.loc[result["spread_z"] > stop_z, "stop_signal"] = 1
    result.loc[result["spread_z"] < -stop_z, "stop_signal"] = -1
    
    # Generate final trading signal
    result["raw_signal"] = 0
    # Entry signals
    result.loc[result["entry_signal"] == 1, "raw_signal"] = 1
    result.loc[result["entry_signal"] == -1, "raw_signal"] = -1
    # Exit signals override entry signals
    result.loc[result["exit_signal"] != 0, "raw_signal"] = 0
    # Stop signals override everything
    result.loc[result["stop_signal"] == 1, "raw_signal"] = 0
    result.loc[result["stop_signal"] == -1, "raw_signal"] = 0
    
    # Apply one-bar execution delay
    result["delayed_signal"] = result["raw_signal"].shift(1)
    result["delayed_fx_position"] = result["fx_position"].shift(1)
    result["delayed_comd_position"] = result["comd_position"].shift(1)
    
    # Calculate price changes
    result["fx_return"] = result["fx_price"].pct_change()
    result["comd_return"] = result["comd_price"].pct_change()
    
    # Calculate position PnL with one-bar delay
    result["fx_pnl"] = result["delayed_fx_position"] * result["fx_price"].diff()
    result["comd_pnl"] = result["delayed_comd_position"] * result["comd_price"].diff()
    result["total_pnl"] = result["fx_pnl"] + result["comd_pnl"]
    
    # Add transaction costs
    fx_bps = 1.0
    cm_bps = 2.0
    trade_flag = (result["delayed_signal"].diff().abs() == 1)
    cost_fx = trade_flag.shift(1) * (fx_bps/1e4) * result["delayed_fx_position"].abs()
    cost_cm = trade_flag.shift(1) * (cm_bps/1e4) * result["delayed_comd_position"].abs()
    result["total_pnl"] = result["total_pnl"] - cost_fx - cost_cm
    
    # Calculate cumulative PnL
    result["total_pnl"] = result["total_pnl"].fillna(0.0)
    result["cumulative_pnl"] = result["total_pnl"].cumsum()
    
    # Fill NaN values in PnL and calculate equity curve
    result["pnl"] = result["total_pnl"]
    result["equity"] = (1.0 + result["pnl"]).cumprod()
    
    # Calculate drawdown based on equity
    running_max_equity = result["equity"].cummax()
    drawdown_series = (result["equity"] - running_max_equity) / running_max_equity
    # result["drawdown"] = drawdown_series.fillna(0.0)
    result["drawdown"] = drawdown_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Calculate trade statistics
    result = _calculate_trade_stats(result)
    
    # Calculate robust performance metrics
    stats = {
        "CAGR": _safe_cagr(result["equity"]),
        "Sharpe": _safe_sharpe(result["pnl"]),
        "MaxDD": float(((result["equity"].cummax() - result["equity"]) / result["equity"].cummax()).max()),
        "Trades": int((result["delayed_signal"].diff().abs()==1).sum()/2),
        "HitRate": None
    }
    
    logger.info(f"Backtest completed: {len(result)} bars, "
                f"Final PnL: {result['cumulative_pnl'].iloc[-1]:.2f}, "
                f"Max drawdown: {result['drawdown'].min():.2%}")
    
    return result


def _calculate_trade_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trade-level statistics.
    
    Args:
        df: DataFrame with backtest results.
        
    Returns:
        DataFrame with trade statistics added.
    """
    logger.debug("Calculating trade statistics")
    
    # Identify trades
    df["trade_id"] = (df["delayed_signal"].diff() != 0).cumsum()
    
    # Calculate trade statistics
    trade_stats = []
    
    for trade_id, trade_df in df.groupby("trade_id"):
        if trade_df["delayed_signal"].iloc[0] == 0:
            continue  # Skip non-trade periods
        
        # Trade entry and exit
        entry_idx = trade_df.index[0]
        exit_idx = trade_df.index[-1]
        
        # Trade PnL
        trade_pnl = trade_df["total_pnl"].sum()
        
        # Trade duration
        trade_duration = len(trade_df)
        
        # Trade direction
        trade_direction = trade_df["delayed_signal"].iloc[0]
        
        # Entry and exit prices
        fx_entry = trade_df["fx_price"].iloc[0]
        fx_exit = trade_df["fx_price"].iloc[-1]
        comd_entry = trade_df["comd_price"].iloc[0]
        comd_exit = trade_df["comd_price"].iloc[-1]
        
        # Entry and exit z-scores
        entry_z = trade_df["spread_z"].iloc[0] if "spread_z" in trade_df.columns else 0
        exit_z = trade_df["spread_z"].iloc[-1] if "spread_z" in trade_df.columns else 0
        
        trade_stats.append({
            "trade_id": trade_id,
            "entry_date": entry_idx,
            "exit_date": exit_idx,
            "direction": trade_direction,
            "duration": trade_duration,
            "pnl": trade_pnl,
            "fx_entry": fx_entry,
            "fx_exit": fx_exit,
            "comd_entry": comd_entry,
            "comd_exit": comd_exit,
            "entry_z": entry_z,
            "exit_z": exit_z
        })
    
    # Create trade statistics DataFrame
    if trade_stats:
        trades_df = pd.DataFrame(trade_stats)
        trades_df["return"] = trades_df["pnl"] / (abs(trades_df["fx_entry"]) + abs(trades_df["comd_entry"]))
        
        # Store trade statistics in original DataFrame
        df["trade_pnl"] = 0.0
        df["trade_return"] = 0.0
        
        for _, trade in trades_df.iterrows():
            mask = (df.index >= trade["entry_date"]) & (df.index <= trade["exit_date"])
            df.loc[mask, "trade_pnl"] = trade["pnl"]
            df.loc[mask, "trade_return"] = trade["return"]
        
        logger.info(f"Calculated statistics for {len(trades_df)} trades")
    
    return df


def calculate_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        df: DataFrame with backtest results.
        
    Returns:
        Dictionary with performance metrics.
    """
    logger.debug("Calculating performance metrics")
    
    # Basic metrics
    total_pnl = df["total_pnl"].sum()
    total_return = df["equity"].iloc[-1] - 1 if "equity" in df.columns and not df.empty else 0
    
    # Risk metrics
    max_drawdown = df["drawdown"].min()
    volatility = df["total_pnl"].std() * np.sqrt(252)  # Annualized
    
    # Trade metrics
    trades = df[df["entry"] | df["exit"]]
    num_trades = len(trades[trades["entry"]])
    
    if num_trades > 0:
        # Extract trade PnLs
        trade_pnls = []
        for trade_id in df["trade_id"].unique():
            trade_df = df[df["trade_id"] == trade_id]
            if trade_df["delayed_signal"].iloc[0] != 0:
                trade_pnls.append(trade_df["trade_pnl"].iloc[0])
        
        win_trades = [pnl for pnl in trade_pnls if pnl > 0]
        loss_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(win_trades) / len(trade_pnls) if trade_pnls else 0
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        # Handle case where there are no losses to avoid infinity
        if loss_trades:
            profit_factor = abs(sum(win_trades) / sum(loss_trades))
        elif win_trades:
            # If there are wins but no losses, use a large but finite number
            profit_factor = 999.0
        else:
            # If there are no trades or no wins, profit factor is 0
            profit_factor = 0.0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Risk-adjusted metrics
    if volatility > 0:
        sharpe_ratio = total_return / volatility
    else:
        sharpe_ratio = 0
    
    # Calculate days in backtest
    days_in_backtest = (df.index[-1] - df.index[0]).days
    
    # Annualized return
    if days_in_backtest > 0:
        years = days_in_backtest / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0
    
    metrics = {
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "days_in_backtest": days_in_backtest
    }
    
    logger.info(f"Performance metrics: Sharpe={sharpe_ratio:.2f}, "
                f"Return={total_return:.2%}, MaxDD={max_drawdown:.2%}, "
                f"WinRate={win_rate:.2%}")
    
    return metrics


def create_backtest_report(df: pd.DataFrame, metrics: Dict) -> str:
    """
    Create a human-readable backtest report.
    
    Args:
        df: DataFrame with backtest results.
        metrics: Dictionary with performance metrics.
        
    Returns:
        String with formatted backtest report.
    """
    logger.debug("Creating backtest report")
    
    report = f"""
    FX-Commodity Correlation Arbitrage Backtest Report
    =================================================
    
    Period: {df.index[0].date()} to {df.index[-1].date()} ({metrics['days_in_backtest']} days)
    
    Performance Metrics:
    - Total PnL: {metrics['total_pnl']:,.2f}
    - Total Return: {metrics['total_return']:.2%}
    - Annual Return: {metrics['annual_return']:.2%}
    - Volatility (Annual): {metrics['volatility']:.2%}
    - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    - Maximum Drawdown: {metrics['max_drawdown']:.2%}
    
    Trading Statistics:
    - Number of Trades: {metrics['num_trades']}
    - Win Rate: {metrics['win_rate']:.2%}
    - Average Win: {metrics['avg_win']:,.2f}
    - Average Loss: {metrics['avg_loss']:,.2f}
    - Profit Factor: {metrics['profit_factor']:.2f}
    
    Note: All calculations include one-bar execution delay.
    """
    
    return report


def run_backtest(
    signals_df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete backtest with configuration parameters.
    
    Args:
        signals_df: DataFrame with signals and market data.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (backtest_results_df, performance_metrics_dict).
    """
    logger.info("Running complete backtest")
    
    # Extract parameters
    entry_z = config["thresholds"]["entry_z"]
    exit_z = config["thresholds"]["exit_z"]
    stop_z = config["thresholds"]["stop_z"]
    max_bars = config["time_stop"]["max_days"]
    inverse_fx_for_quote_ccy_strength = config["inverse_fx_for_quote_ccy_strength"]
    
    # Run backtest
    backtest_df = backtest_pair(
        signals_df,
        entry_z,
        exit_z,
        stop_z,
        max_bars,
        inverse_fx_for_quote_ccy_strength
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(backtest_df)
    
    return backtest_df, metrics