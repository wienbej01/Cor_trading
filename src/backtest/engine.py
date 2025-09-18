"""
Backtest engine module for FX-Commodity correlation arbitrage strategy.
Implements backtesting with one-bar execution delay and performance statistics.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.metrics import calculate_comprehensive_metrics, save_run_artifacts

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
        return (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0
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
    config: dict,
) -> pd.DataFrame:
    """
    Backtest a single FX-Commodity pair with a bar-by-bar event loop to support
    complex risk management and position sizing.

    Args:
        df: DataFrame with signals and market data.
        config: The configuration dictionary for the pair.

    Returns:
        DataFrame with backtest results.
    """
    logger.info("Running bar-by-bar backtest with integrated risk management.")

    # --- Initialization ---
    initial_equity = 100_000.0
    equity = initial_equity
    position = 0
    entry_price_spread = 0.0
    
    # Result lists
    equity_curve = []
    pnl_curve = []
    position_curve = []
    drawdown_curve = []
    running_max_equity = initial_equity

    # Risk and Sizing
    risk_config = config.get('risk', {})
    sizing_config = config.get('sizing', {})
    risk_policy = RiskPolicy(risk_config)
    position_sizer = PositionSizer(sizing_config)

    # --- Event Loop ---
    for i in range(1, len(df)):
        current_time = df.index[i]
        prev_time = df.index[i-1]
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # Update equity with PnL from the previous bar
        pnl = 0
        if position != 0:
            fx_pnl = position * (row['fx_price'] - prev_row['fx_price'])
            comd_pnl = -position * (row['comd_price'] - prev_row['comd_price'])
            pnl = fx_pnl + comd_pnl
        equity += pnl
        
        # Update drawdown
        running_max_equity = max(running_max_equity, equity)
        drawdown = (equity - running_max_equity) / running_max_equity if running_max_equity > 0 else 0

        # --- Exit Logic ---
        exit_signal = False
        if position == 1 and (row['spread_z'] >= -config['thresholds']['exit_z']):
            exit_signal = True
        elif position == -1 and (row['spread_z'] <= config['thresholds']['exit_z']):
            exit_signal = True
        
        # ATR Stop Loss
        if position != 0 and check_atr_stop_loss(position, entry_price_spread, row['spread'], row['fx_atr'], risk_config.get('atr_stop_loss_multiplier', 2.0)):
            exit_signal = True
            logger.debug(f"ATR Stop triggered at {current_time}")

        if exit_signal and position != 0:
            position = 0
            entry_price_spread = 0.0

        # --- Entry Logic ---
        entry_signal = row['signal']
        if entry_signal != 0 and position == 0:
            equity_history = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
            if risk_policy.can_trade(current_time, equity, equity_history):
                # Use position sizer here if needed, for now we use the pre-calculated size
                position = entry_signal
                entry_price_spread = row['spread']
                risk_policy.record_trade(current_time)

        # Append results for this bar
        equity_curve.append(equity)
        pnl_curve.append(pnl)
        position_curve.append(position)
        drawdown_curve.append(drawdown)

    # --- Post-processing ---
    result = df.copy()
    result['equity'] = pd.Series(equity_curve, index=df.index[1:])
    result['pnl'] = pd.Series(pnl_curve, index=df.index[1:])
    result['position'] = pd.Series(position_curve, index=df.index[1:])
    result['drawdown'] = pd.Series(drawdown_curve, index=df.index[1:])
    
    result.fillna(method='ffill', inplace=True)
    result.fillna(0, inplace=True)

    logger.info(
        f"Backtest completed: Final Equity: {result['equity'].iloc[-1]:.2f}, "
        f"Max drawdown: {result['drawdown'].min():.2%}"
    )

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

        trade_stats.append(
            {
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
                "exit_z": exit_z,
            }
        )

    # Create trade statistics DataFrame
    if trade_stats:
        trades_df = pd.DataFrame(trade_stats)
        trades_df["return"] = trades_df["pnl"] / (
            abs(trades_df["fx_entry"]) + abs(trades_df["comd_entry"])
        )

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
    equity = df["equity"] if ("equity" in df.columns and not df.empty) else pd.Series([1.0], index=[0])
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    
    # Risk metrics
    max_drawdown = df["drawdown"].min()
    volatility = df["pnl"].std() * np.sqrt(252)  # Annualized

    # Trade metrics
    num_trades = int((df["delayed_signal"].diff().abs() == 1).sum() / 2)

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
    sharpe_ratio = _safe_sharpe(df["pnl"])

    # Calculate days in backtest
    days_in_backtest = (df.index[-1] - df.index[0]).days

    # Annualized return
    annual_return = _safe_cagr(equity)

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
        "days_in_backtest": days_in_backtest,
    }

    logger.info(
        f"Performance metrics: Sharpe={sharpe_ratio:.2f}, "
        f"Return={total_return:.2%}, MaxDD={max_drawdown:.2%}, "
        f"WinRate={win_rate:.2%}"
    )

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


def run_backtest(signals_df: pd.DataFrame, config: Dict, run_id: str = None) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Run complete backtest with configuration parameters.

    Args:
        signals_df: DataFrame with signals and market data.
        config: Configuration dictionary.
        run_id: Optional run identifier.

    Returns:
        Tuple of (backtest_results_df, performance_metrics_dict, reports_path).
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
        signals_df, entry_z, exit_z, stop_z, max_bars, inverse_fx_for_quote_ccy_strength
    )

    # Calculate performance metrics
    metrics = calculate_performance_metrics(backtest_df)
    
    # Extract trades data
    trades_data = []
    if "trade_id" in backtest_df.columns:
        for trade_id in backtest_df["trade_id"].unique():
            trade_df = backtest_df[backtest_df["trade_id"] == trade_id]
            if len(trade_df) > 0 and trade_df["delayed_signal"].iloc[0] != 0:
                # Get first and last rows for trade entry/exit info
                first_row = trade_df.iloc[0]
                last_row = trade_df.iloc[-1]
                
                trade_info = {
                    "trade_id": trade_id,
                    "entry_date": first_row.name,
                    "exit_date": last_row.name,
                    "direction": first_row["position"],
                    "duration": len(trade_df),
                    "pnl": trade_df["total_pnl"].sum(),
                    "fx_entry": first_row["fx_price"],
                    "fx_exit": last_row["fx_price"],
                    "comd_entry": first_row["comd_price"],
                    "comd_exit": last_row["comd_price"]
                }
                if "spread_z" in trade_df.columns:
                    trade_info["entry_z"] = first_row["spread_z"]
                    trade_info["exit_z"] = last_row["spread_z"]
                
                trades_data.append(trade_info)
    
    trades_df = pd.DataFrame(trades_data)
    
    # Calculate comprehensive metrics
    comprehensive_metrics = calculate_comprehensive_metrics(backtest_df, trades_df, config)
    
    # Save run artifacts
    pair_name = config.get("pair", "unknown")
    reports_path = save_run_artifacts(pair_name, backtest_df, trades_df, config, comprehensive_metrics, run_id)

    return backtest_df, comprehensive_metrics, reports_path
_df, config)
    
    # Save run artifacts
    pair_name = config.get("pair", "unknown")
    reports_path = save_run_artifacts(pair_name, backtest_df, trades_df, config, comprehensive_metrics, run_id)

    return backtest_df, comprehensive_metrics, reports_path
air", "unknown")
    reports_path = save_run_artifacts(pair_name, backtest_df, trades_df, config, comprehensive_metrics, run_id)

    return backtest_df, comprehensive_metrics, reports_path
