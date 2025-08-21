#!/usr/bin/env python3
"""
CLI runner for FX-Commodity correlation arbitrage backtest.
Provides command-line interface for running backtests on configured pairs.
"""

import click
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import config
from src.data.yahoo_loader import download_and_align_pair
from src.strategy.mean_reversion import generate_signals_with_regime_filter
from src.backtest.engine import run_backtest, create_backtest_report
from src.utils.logging import setup_logging, trading_logger


@click.command()
@click.option(
    "--pair",
    type=str,
    required=True,
    help="Pair name to backtest (e.g., usdcad_wti, usdnok_brent)"
)
@click.option(
    "--start",
    type=str,
    required=True,
    help="Start date in YYYY-MM-DD format"
)
@click.option(
    "--end",
    type=str,
    required=True,
    help="End date in YYYY-MM-DD format"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
@click.option(
    "--output-dir",
    type=str,
    default="./backtest_results",
    help="Directory to save backtest results"
)
@click.option(
    "--save-data",
    is_flag=True,
    help="Save signal data to CSV"
)
@click.option(
    "--report-format",
    type=click.Choice(["txt", "md"]),
    default="txt",
    help="Format for backtest report"
)
@click.option(
    "--no-kalman",
    is_flag=True,
    help="Use OLS instead of Kalman filter for spread calculation"
)
def run_backtest_cli(
    pair,
    start,
    end,
    log_level,
    output_dir,
    save_data,
    report_format,
    no_kalman
):
    """
    Run backtest for FX-Commodity correlation arbitrage strategy.
    
    Example usage:
    python src/run_backtest.py --pair usdcad_wti --start 2015-01-01 --end 2025-08-15
    python src/run_backtest.py --pair usdnok_brent --start 2015-01-01 --end 2025-08-15
    """
    # Set up logging
    log_file = Path(output_dir) / f"{pair}_{start}_{end}.log"
    setup_logging(log_level=log_level, log_file=log_file)
    
    trading_logger.logger.info(f"Starting backtest for pair: {pair}")
    trading_logger.logger.info(f"Backtest period: {start} to {end}")
    
    try:
        # Validate dates
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
        
        # Load pair configuration
        pair_config = config.get_pair_config(pair)
        
        # Disable Kalman filter if requested
        if no_kalman:
            pair_config["use_kalman"] = False
            trading_logger.logger.info("Using OLS instead of Kalman filter")
        
        trading_logger.logger.info(f"Loaded configuration for {pair}")
        
        # Download and align data
        trading_logger.logger.info("Downloading market data")
        data = download_and_align_pair(
            fx_symbol=pair_config["fx_symbol"],
            comd_symbol=pair_config["comd_symbol"],
            start=start,
            end=end,
            fx_name="fx",
            comd_name="comd"
        )
        
        trading_logger.logger.info(f"Downloaded {len(data)} data points")
        
        # Generate signals
        trading_logger.logger.info("Generating trading signals")
        signals_df = generate_signals_with_regime_filter(
            fx_series=data["fx"],
            comd_series=data["comd"],
            config=pair_config
        )
        
        trading_logger.logger.info(f"Generated {signals_df['signal'].abs().sum()} total signals")
        
        # Add diagnostics
        sigs = signals_df
        print("\n=== Diagnostics ===")
        print("Observations:", len(sigs))
        print("Good-regime %:", float(((sigs['enter_long'])|(sigs['enter_short'])|(sigs['exit'])).mean()*100))
        print("Mean |z| @ entries:",
              float(sigs.loc[sigs['enter_long']|sigs['enter_short'],'spread_z'].abs().mean()))
        print("ADF p-value snapshot:", sigs['adf_p'] if isinstance(sigs['adf_p'], float) else sigs['adf_p'])
        print("Commodity big-gap days (|Δ|>5%):",
              int(sigs['comd_price'].pct_change().abs().gt(0.05).sum()))
        
        # Run backtest
        trading_logger.logger.info("Running backtest")
        backtest_df, metrics = run_backtest(signals_df, pair_config)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save signal data if requested
        if save_data:
            signals_file = output_path / f"{pair}_signals_{timestamp}.csv"
            signals_df.to_csv(signals_file)
            trading_logger.logger.info(f"Saved signals to {signals_file}")
            
            backtest_file = output_path / f"{pair}_backtest_{timestamp}.csv"
            backtest_df.to_csv(backtest_file)
            trading_logger.logger.info(f"Saved backtest results to {backtest_file}")
        
        # Create and save report
        report = create_backtest_report(backtest_df, metrics)
        
        if report_format == "txt":
            report_file = output_path / f"{pair}_report_{timestamp}.txt"
            with open(report_file, "w") as f:
                f.write(report)
        else:  # markdown
            report_file = output_path / f"{pair}_report_{timestamp}.md"
            with open(report_file, "w") as f:
                f.write("# FX-Commodity Correlation Arbitrage Backtest Report\n\n")
                f.write("```\n")
                f.write(report)
                f.write("\n```\n")
        
        trading_logger.logger.info(f"Saved report to {report_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print(f"BACKTEST SUMMARY: {pair.upper()}")
        print("="*60)
        print(f"Period: {start} to {end}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Number of Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("="*60)
        print(f"Full results saved to: {output_path}")
        print("="*60 + "\n")
        
        trading_logger.logger.info("Backtest completed successfully")
        
    except Exception as e:
        trading_logger.logger.error(f"Backtest failed: {str(e)}")
        raise e


@click.group()
def cli():
    """FX-Commodity Correlation Arbitrage Backtest Tool"""
    pass


@cli.command()
def list_pairs():
    """List available trading pairs"""
    pairs = config.list_pairs()
    
    print("\nAvailable Trading Pairs:")
    print("="*40)
    
    for pair in pairs:
        pair_config = config.get_pair_config(pair)
        print(f"\n{pair.upper()}:")
        print(f"  FX Symbol: {pair_config['fx_symbol']}")
        print(f"  Commodity Symbol: {pair_config['comd_symbol']}")
        print(f"  Entry Z: {pair_config['thresholds']['entry_z']}")
        print(f"  Exit Z: {pair_config['thresholds']['exit_z']}")
        print(f"  Stop Z: {pair_config['thresholds']['stop_z']}")
        print(f"  Max Days: {pair_config['time_stop']['max_days']}")
    
    print("\n" + "="*40 + "\n")


@cli.command()
@click.option(
    "--pair",
    type=str,
    required=True,
    help="Pair name to show config for"
)
def show_config(pair):
    """Show configuration for a specific pair"""
    try:
        pair_config = config.get_pair_config(pair)
        
        print(f"\nConfiguration for {pair.upper()}:")
        print("="*50)
        
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")
        
        print_dict(pair_config)
        print("\n" + "="*50 + "\n")
        
    except KeyError:
        print(f"Error: Pair '{pair}' not found. Use 'list-pairs' to see available pairs.")


# Add the main backtest command to the CLI group
cli.add_command(run_backtest_cli, name="run")


if __name__ == "__main__":
    cli()