#!/usr/bin/env python3
"""
Example script demonstrating multi-timeframe backtesting capabilities.
Run this script to test the enhanced trading system with H1 and D1 signals.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.engine import (
    run_multi_timeframe_backtest,
    run_h1_d1_comparison_backtest,
)
from loguru import logger


async def main():
    """Main function to run multi-timeframe backtest examples."""

    # Configure logging
    logger.add("multi_timeframe_backtest.log", rotation="10 MB", retention="30 days")
    logger.info("Starting multi-timeframe backtest example")

    # Example 1: Basic multi-timeframe backtest
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Multi-Timeframe Backtest")
    print("=" * 60)

    try:
        fx_symbol = "USDCAD=X"
        comd_symbol = "CL=F"  # WTI Crude Oil
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        config = {
            "h1": {
                "lookbacks": {
                    "beta_window": 24,  # 24 hours for H1 beta
                    "z_window": 12,  # 12 hours for H1 z-score
                    "corr_window": 24,  # 24 hours for H1 correlation
                },
                "thresholds": {
                    "entry_z": 1.5,  # H1 entry threshold
                    "exit_z": 0.5,  # H1 exit threshold
                    "stop_z": 2.0,  # H1 stop loss threshold
                },
                "sizing": {
                    "atr_window": 12,  # 12 hours for H1 ATR
                    "target_vol_per_leg": 0.01,  # 1% target vol per leg
                },
                "regime": {"min_abs_corr": 0.1},  # Minimum correlation for H1
            },
            "d1": {
                "lookbacks": {
                    "beta_window": 60,  # 60 days for D1 beta
                    "z_window": 30,  # 30 days for D1 z-score
                    "corr_window": 60,  # 60 days for D1 correlation
                },
                "thresholds": {
                    "entry_z": 2.0,  # D1 entry threshold (more conservative)
                    "exit_z": 0.8,  # D1 exit threshold
                    "stop_z": 3.0,  # D1 stop loss threshold
                },
                "sizing": {
                    "atr_window": 20,  # 20 days for D1 ATR
                    "target_vol_per_leg": 0.02,  # 2% target vol per leg
                },
                "regime": {"min_abs_corr": 0.2},  # Minimum correlation for D1
            },
            "multi_timeframe": {
                "h1_weight": 0.6,  # 60% weight to H1 signals
                "d1_weight": 0.4,  # 40% weight to D1 signals
                "max_h1_exposure": 0.3,  # Max 30% exposure from H1
                "max_d1_exposure": 0.7,  # Max 70% exposure from D1
                "max_total_exposure": 1.0,  # Max 100% total exposure
                "min_h1_confidence": 0.7,  # Min 70% confidence for H1
                "min_d1_confidence": 0.8,  # Min 80% confidence for D1
                "base_position_size": 0.1,  # 10% base position size
                "volatility_scaling": True,  # Enable volatility-based scaling
            },
            "exec": {
                "fx_small_trade_cost_bps": 0.5,  # 0.5 bps for small FX trades
                "fx_large_trade_cost_bps": 2.0,  # 2.0 bps for large FX trades
                "comd_small_trade_cost_bps": 0.5,  # 0.5 bps for small commodity trades
                "comd_large_trade_cost_bps": 2.0,  # 2.0 bps for large commodity trades
                "quadratic_impact_coefficient": 0.001,  # Quadratic impact coefficient
            },
        }

        print(f"Running multi-timeframe backtest for {fx_symbol}-{comd_symbol}")
        print(f"Period: {start_date} to {end_date}")

        # Run the backtest
        results, metrics = await run_multi_timeframe_backtest(
            fx_symbol=fx_symbol,
            comd_symbol=comd_symbol,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )

        if results.empty:
            print("❌ No results generated. Check data availability and configuration.")
            return

        # Display results
        print("\n✅ Multi-timeframe backtest completed!")
        print(f"📊 Total signals generated: {metrics.get('combined_signal_count', 0)}")
        print(f"💰 Final equity: ${results['equity'].iloc[-1]:,.2f}")
        print(f"📈 Total return: {metrics.get('total_return', 0):.2%}")
        print(f"⚡ Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"📉 Max drawdown: {metrics.get('max_drawdown', 0):.2%}")

        # Multi-timeframe specific metrics
        mtf_stats = metrics.get("mtf_stats", {})
        print("\n🔄 Multi-timeframe breakdown:")
        print(f"   H1 signals: {mtf_stats.get('h1_signals', 0)}")
        print(f"   D1 signals: {mtf_stats.get('d1_signals', 0)}")
        print(f"   Combined signals: {mtf_stats.get('active_signals', 0)}")
        print(f"   Avg position size: {mtf_stats.get('avg_position_size', 0):.4f}")

    except Exception as e:
        logger.error(f"Multi-timeframe backtest failed: {e}")
        print(f"❌ Error: {e}")
        return

    # Example 2: Strategy comparison
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Strategy Comparison (H1 vs D1 vs Multi-timeframe)")
    print("=" * 60)

    try:
        print("Running strategy comparison...")

        comparison_results = await run_h1_d1_comparison_backtest(
            fx_symbol=fx_symbol,
            comd_symbol=comd_symbol,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )

        print("\n📊 Strategy Comparison Results:")
        for strategy_name, (df, strategy_metrics) in comparison_results.items():
            if not df.empty:
                print(f"\n{strategy_name.upper()}:")
                print(f"   Return: {strategy_metrics.get('total_return', 0):.2f}")
                print(f"   Sharpe: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   MaxDD: {strategy_metrics.get('max_drawdown', 0):.2%}")
                print(f"   Trades: {strategy_metrics.get('num_trades', 0)}")
            else:
                print(f"\n{strategy_name.upper()}: No results")

    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        print(f"❌ Comparison error: {e}")

    print("\n" + "=" * 60)
    print("🎉 Multi-timeframe backtest examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
