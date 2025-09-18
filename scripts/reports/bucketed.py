#!/usr/bin/env python3
"""
Bucketed Performance Report Generator.

Aggregates backtest results by volatility/trend regimes and 2-year time windows.
Generates distributional metrics, plots, and markdown report.
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data.yahoo_loader import download_and_align_pair
from src.features.regime import volatility_regime, trend_regime


def load_summary_json(filepath: str) -> Dict:
    """Load summary.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_trades_parquet(filepath: str) -> pd.DataFrame:
    """Load trades.parquet file."""
    return pd.read_parquet(filepath)


def get_pair_data(pair: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load FX/commodity data for regime computation."""
    try:
        fx_series, comd_series = download_and_align_pair(pair, start_date, end_date)
        return pd.DataFrame({'fx_price': fx_series, 'comd_price': comd_series})
    except Exception as e:
        logger.warning(f"Failed to load data for {pair}: {e}")
        return pd.DataFrame()


def classify_regimes_at_dates(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Classify vol/trend regimes at given dates."""
    if data.empty:
        return pd.DataFrame(index=dates, columns=['vol_regime', 'trend_regime'])

    # Compute regimes
    fx_vol = volatility_regime(data['fx_price'], use_quantiles=True)
    comd_vol = volatility_regime(data['comd_price'], use_quantiles=True)
    fx_trend = trend_regime(data['fx_price'], use_roc_hp=True)
    comd_trend = trend_regime(data['comd_price'], use_roc_hp=True)

    # Average for pair-level regime
    vol_regime = ((fx_vol + comd_vol) / 2).round().astype(int)
    trend_regime = ((fx_trend + comd_trend) / 2).round().astype(int)

    # Map to labels
    vol_labels = {0: 'low_vol', 1: 'normal_vol', 2: 'high_vol'}
    trend_labels = {-1: 'downtrend', 0: 'ranging', 1: 'uptrend'}

    regimes = pd.DataFrame(index=data.index)
    regimes['vol_regime'] = vol_regime.map(vol_labels).fillna('unknown')
    regimes['trend_regime'] = trend_regime.map(trend_labels).fillna('unknown')

    # Reindex to dates
    return regimes.reindex(dates, method='nearest')


def create_time_buckets(dates: pd.DatetimeIndex) -> pd.Series:
    """Create 2-year time buckets."""
    years = dates.year
    bins = list(range(2014, 2027, 2))
    labels = [f"{y}-{y+1}" for y in bins[:-1]]
    return pd.cut(years, bins=bins, labels=labels, right=False)


def aggregate_by_buckets(trades: pd.DataFrame, regimes: pd.DataFrame) -> Dict:
    """Aggregate trade metrics by regime and time buckets."""
    if trades.empty:
        return {}

    # Merge regimes
    trades = trades.copy()
    trades['vol_regime'] = regimes['vol_regime']
    trades['trend_regime'] = regimes['trend_regime']
    trades['time_bucket'] = create_time_buckets(trades['entry_date'])

    # Compute metrics
    results = {}

    # By vol regime
    vol_agg = trades.groupby('vol_regime').agg({
        'pnl': ['count', 'sum', 'mean', 'std'],
        'duration': 'mean'
    }).round(3)
    vol_agg.columns = ['num_trades', 'total_pnl', 'avg_pnl', 'pnl_std', 'avg_duration']
    results['volatility'] = vol_agg.to_dict('index')

    # By trend regime
    trend_agg = trades.groupby('trend_regime').agg({
        'pnl': ['count', 'sum', 'mean', 'std'],
        'duration': 'mean'
    }).round(3)
    trend_agg.columns = ['num_trades', 'total_pnl', 'avg_pnl', 'pnl_std', 'avg_duration']
    results['trend'] = trend_agg.to_dict('index')

    # By time bucket
    time_agg = trades.groupby('time_bucket').agg({
        'pnl': ['count', 'sum', 'mean', 'std'],
        'duration': 'mean'
    }).round(3)
    time_agg.columns = ['num_trades', 'total_pnl', 'avg_pnl', 'pnl_std', 'avg_duration']
    results['time'] = time_agg.to_dict('index')

    # Combined vol + trend
    combined_agg = trades.groupby(['vol_regime', 'trend_regime']).agg({
        'pnl': ['count', 'sum', 'mean', 'std']
    }).round(3)
    combined_agg.columns = ['num_trades', 'total_pnl', 'avg_pnl', 'pnl_std']
    results['combined'] = combined_agg.to_dict('index')

    return results


def generate_plots(aggregates: Dict, output_dir: str) -> List[str]:
    """Generate bar plots for key metrics."""
    plot_files = []
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Vol regime plots
    if 'volatility' in aggregates:
        vol_data = aggregates['volatility']
        regimes = list(vol_data.keys())
        num_trades = [vol_data[r]['num_trades'] for r in regimes]
        avg_pnl = [vol_data[r]['avg_pnl'] for r in regimes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(regimes, num_trades, color='skyblue')
        ax1.set_title('Trades by Volatility Regime')
        ax1.set_ylabel('Number of Trades')

        ax2.bar(regimes, avg_pnl, color='lightgreen')
        ax2.set_title('Avg PnL by Volatility Regime')
        ax2.set_ylabel('Avg PnL')

        plt.tight_layout()
        plot_file = plots_dir / 'volatility_regime_performance.png'
        plt.savefig(plot_file)
        plt.close()
        plot_files.append(str(plot_file))

    # Trend regime plots
    if 'trend' in aggregates:
        trend_data = aggregates['trend']
        regimes = list(trend_data.keys())
        num_trades = [trend_data[r]['num_trades'] for r in regimes]
        avg_pnl = [trend_data[r]['avg_pnl'] for r in regimes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(regimes, num_trades, color='orange')
        ax1.set_title('Trades by Trend Regime')
        ax1.set_ylabel('Number of Trades')

        ax2.bar(regimes, avg_pnl, color='purple')
        ax2.set_title('Avg PnL by Trend Regime')
        ax2.set_ylabel('Avg PnL')

        plt.tight_layout()
        plot_file = plots_dir / 'trend_regime_performance.png'
        plt.savefig(plot_file)
        plt.close()
        plot_files.append(str(plot_file))

    # Time bucket plots
    if 'time' in aggregates:
        time_data = aggregates['time']
        buckets = list(time_data.keys())
        num_trades = [time_data[b]['num_trades'] for b in buckets]
        avg_pnl = [time_data[b]['avg_pnl'] for b in buckets]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(buckets, num_trades, color='red')
        ax1.set_title('Trades by Time Bucket')
        ax1.set_ylabel('Number of Trades')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(buckets, avg_pnl, color='blue')
        ax2.set_title('Avg PnL by Time Bucket')
        ax2.set_ylabel('Avg PnL')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_file = plots_dir / 'time_bucket_performance.png'
        plt.savefig(plot_file)
        plt.close()
        plot_files.append(str(plot_file))

    return plot_files


def generate_markdown_report(aggregates: Dict, plot_files: List[str], output_file: str):
    """Generate markdown report."""
    with open(output_file, 'w') as f:
        f.write("# Bucketed Performance Analysis\n\n")
        f.write("Analysis of backtest results aggregated by volatility/trend regimes and 2-year time windows.\n\n")

        # Volatility regimes
        if 'volatility' in aggregates:
            f.write("## Volatility Regimes\n\n")
            f.write("| Regime | Trades | Total PnL | Avg PnL | PnL Std | Avg Duration |\n")
            f.write("|--------|--------|-----------|---------|---------|--------------|\n")
            for regime, metrics in aggregates['volatility'].items():
                f.write(f"| {regime} | {metrics['num_trades']} | {metrics['total_pnl']:.2f} | {metrics['avg_pnl']:.2f} | {metrics['pnl_std']:.2f} | {metrics['avg_duration']:.1f} |\n")
            f.write("\n")

        # Trend regimes
        if 'trend' in aggregates:
            f.write("## Trend Regimes\n\n")
            f.write("| Regime | Trades | Total PnL | Avg PnL | PnL Std | Avg Duration |\n")
            f.write("|--------|--------|-----------|---------|---------|--------------|\n")
            for regime, metrics in aggregates['trend'].items():
                f.write(f"| {regime} | {metrics['num_trades']} | {metrics['total_pnl']:.2f} | {metrics['avg_pnl']:.2f} | {metrics['pnl_std']:.2f} | {metrics['avg_duration']:.1f} |\n")
            f.write("\n")

        # Time buckets
        if 'time' in aggregates:
            f.write("## Time Buckets (2-Year Periods)\n\n")
            f.write("| Bucket | Trades | Total PnL | Avg PnL | PnL Std | Avg Duration |\n")
            f.write("|--------|--------|-----------|---------|---------|--------------|\n")
            for bucket, metrics in aggregates['time'].items():
                f.write(f"| {bucket} | {metrics['num_trades']} | {metrics['total_pnl']:.2f} | {metrics['avg_pnl']:.2f} | {metrics['pnl_std']:.2f} | {metrics['avg_duration']:.1f} |\n")
            f.write("\n")

        # Combined
        if 'combined' in aggregates:
            f.write("## Combined Volatility + Trend Regimes\n\n")
            f.write("| Vol Regime | Trend Regime | Trades | Total PnL | Avg PnL | PnL Std |\n")
            f.write("|------------|--------------|--------|-----------|---------|---------|\n")
            for (vol, trend), metrics in aggregates['combined'].items():
                f.write(f"| {vol} | {trend} | {metrics['num_trades']} | {metrics['total_pnl']:.2f} | {metrics['avg_pnl']:.2f} | {metrics['pnl_std']:.2f} |\n")
            f.write("\n")

        # Plots
        if plot_files:
            f.write("## Performance Plots\n\n")
            for plot in plot_files:
                rel_path = os.path.relpath(plot, os.path.dirname(output_file))
                f.write(f"![{os.path.basename(plot)}]({rel_path})\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate bucketed performance reports")
    parser.add_argument('--input', required=True, help='Input directory containing reports')
    parser.add_argument('--out', required=True, help='Output markdown file')
    args = parser.parse_args()

    logger.info(f"Starting bucketed analysis from {args.input}")

    # Find all summary.json files
    summary_files = glob.glob(os.path.join(args.input, '**', 'summary.json'), recursive=True)
    logger.info(f"Found {len(summary_files)} summary files")

    all_aggregates = {}

    for summary_file in summary_files:
        try:
            summary = load_summary_json(summary_file)
            pair = summary.get('pair', 'unknown')
            run_id = summary.get('run_id', 'unknown')

            # Load trades
            trades_file = summary_file.replace('summary.json', 'trades.parquet')
            if not os.path.exists(trades_file):
                logger.warning(f"Trades file not found: {trades_file}")
                continue

            trades = load_trades_parquet(trades_file)
            if trades.empty:
                logger.warning(f"No trades in {trades_file}")
                continue

            # Load data for regime computation
            start_date = summary.get('start_date', '2015-01-01')
            end_date = summary.get('end_date', '2025-12-31')
            data = get_pair_data(pair, start_date, end_date)

            # Classify regimes at trade entry dates
            entry_dates = pd.to_datetime(trades['entry_date'])
            regimes = classify_regimes_at_dates(data, entry_dates)

            # Aggregate
            aggregates = aggregate_by_buckets(trades, regimes)
            for key, value in aggregates.items():
                if key not in all_aggregates:
                    all_aggregates[key] = {}
                for subkey, subvalue in value.items():
                    if subkey not in all_aggregates[key]:
                        all_aggregates[key][subkey] = []
                    all_aggregates[key][subkey].append(subvalue)

            logger.info(f"Processed {pair} run {run_id}")

        except Exception as e:
            logger.error(f"Failed to process {summary_file}: {e}")

    # Aggregate across runs (simple mean for now; could add distributions)
    final_aggregates = {}
    for key, subdict in all_aggregates.items():
        final_aggregates[key] = {}
        for subkey, values in subdict.items():
            if isinstance(values[0], dict):
                # Nested dict (e.g., metrics)
                final_aggregates[key][subkey] = {}
                for metric in values[0].keys():
                    metric_values = [v[metric] for v in values if metric in v]
                    if metric_values:
                        final_aggregates[key][subkey][metric] = np.mean(metric_values)
            else:
                final_aggregates[key][subkey] = np.mean(values)

    # Generate plots
    plot_files = generate_plots(final_aggregates, args.input)

    # Generate report
    generate_markdown_report(final_aggregates, plot_files, args.out)

    logger.info(f"Report generated: {args.out}")


if __name__ == '__main__':
    main()