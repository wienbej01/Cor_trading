import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_backtest_data():
    """Load all backtest data files"""
    base_path = Path("backtest_results")

    # Define file paths
    files = {
        "USDCAD-WTI-Kalman": {
            "report": base_path / "usdcad_wti_report_20250821_112149.txt",
            "backtest": base_path / "usdcad_wti_backtest_20250821_112149.csv",
            "signals": base_path / "usdcad_wti_signals_20250821_112149.csv",
        },
        "USDCAD-WTI-OLS": {
            "report": base_path / "usdcad_wti_report_20250821_112216.txt",
            "backtest": base_path / "usdcad_wti_backtest_20250821_112216.csv",
            "signals": base_path / "usdcad_wti_signals_20250821_112216.csv",
        },
        "USDNOK-Brent-Kalman": {
            "report": base_path / "usdnok_brent_report_20250821_112249.txt",
            "backtest": base_path / "usdnok_brent_backtest_20250821_112249.csv",
            "signals": base_path / "usdnok_brent_signals_20250821_112249.csv",
        },
        "USDNOK-Brent-OLS": {
            "report": base_path / "usdnok_brent_report_20250821_112327.txt",
            "backtest": base_path / "usdnok_brent_backtest_20250821_112327.csv",
            "signals": base_path / "usdnok_brent_signals_20250821_112327.csv",
        },
    }

    return files


def parse_report_file(file_path):
    """Parse the report text file and extract metrics"""
    with open(file_path, "r") as f:
        content = f.read()

    metrics = {}
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("- Total PnL:"):
            metrics["Total PnL"] = float(line.split(": ")[1])
        elif line.startswith("- Total Return:"):
            return_val = line.split(": ")[1]
            metrics["Total Return"] = (
                float(return_val[:-1]) if return_val != "nan%" else np.nan
            )
        elif line.startswith("- Annual Return:"):
            return_val = line.split(": ")[1]
            metrics["Annual Return"] = (
                float(return_val[:-1]) if return_val != "nan%" else np.nan
            )
        elif line.startswith("- Volatility (Annual):"):
            metrics["Volatility"] = float(line.split(": ")[1][:-1])
        elif line.startswith("- Sharpe Ratio:"):
            sharpe_val = line.split(": ")[1]
            metrics["Sharpe Ratio"] = (
                float(sharpe_val) if sharpe_val != "nan" else np.nan
            )
        elif line.startswith("- Maximum Drawdown:"):
            metrics["Max Drawdown"] = float(line.split(": ")[1][:-1])
        elif line.startswith("- Number of Trades:"):
            metrics["Number of Trades"] = int(line.split(": ")[1])
        elif line.startswith("- Win Rate:"):
            metrics["Win Rate"] = float(line.split(": ")[1][:-1])
        elif line.startswith("- Average Win:"):
            metrics["Average Win"] = float(line.split(": ")[1])
        elif line.startswith("- Average Loss:"):
            metrics["Average Loss"] = float(line.split(": ")[1])
        elif line.startswith("- Profit Factor:"):
            metrics["Profit Factor"] = float(line.split(": ")[1])

    return metrics


def analyze_csv_structure(file_path, sample_size=5):
    """Analyze the structure of CSV files without loading the entire file"""
    try:
        # Read just the header and a few sample rows
        df_sample = pd.read_csv(file_path, nrows=sample_size)
        print(f"\nFile: {file_path}")
        print(f"Columns: {list(df_sample.columns)}")
        print("Sample data:")
        print(df_sample.head())

        # Get total row count
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        print(f"Total rows: {total_rows}")

        return df_sample.columns.tolist(), total_rows
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], 0


def analyze_large_csv(file_path, chunk_size=10000):
    """Analyze large CSV files in chunks"""
    print(f"\nAnalyzing {file_path}...")

    # Initialize aggregators
    total_rows = 0
    pnl_sum = 0
    trade_count = 0
    win_count = 0
    losses = []
    wins = []

    try:
        # Read file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            total_rows += len(chunk)

            # Analyze PnL if column exists
            if "pnl" in chunk.columns:
                pnl_sum += chunk["pnl"].sum()

                # Count wins and losses
                wins_mask = chunk["pnl"] > 0
                losses_mask = chunk["pnl"] < 0

                win_count += wins_mask.sum()
                trade_count += wins_mask.sum() + losses_mask.sum()

                wins.extend(chunk[wins_mask]["pnl"].tolist())
                losses.extend(chunk[losses_mask]["pnl"].tolist())

        print(f"Total rows processed: {total_rows}")
        if trade_count > 0:
            print(f"Total trades: {trade_count}")
            print(f"Win rate: {win_count/trade_count*100:.2f}%")
            print(f"Total PnL: {pnl_sum:.4f}")
            if wins:
                print(f"Average win: {np.mean(wins):.4f}")
            if losses:
                print(f"Average loss: {np.mean(losses):.4f}")
                print(f"Profit factor: {sum(wins)/abs(sum(losses)):.4f}")

        return {
            "total_rows": total_rows,
            "total_pnl": pnl_sum,
            "trade_count": trade_count,
            "win_count": win_count,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0,
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def create_comparison_table(report_metrics):
    """Create a comprehensive comparison table"""
    # Convert to DataFrame
    df = pd.DataFrame(report_metrics).T

    # Reorder columns for better readability
    columns_order = [
        "Total PnL",
        "Total Return",
        "Annual Return",
        "Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Number of Trades",
        "Win Rate",
        "Average Win",
        "Average Loss",
        "Profit Factor",
    ]

    df = df[columns_order]

    # Format the table
    formatted_df = df.copy()
    for col in ["Total PnL", "Average Win", "Average Loss"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
    for col in [
        "Total Return",
        "Annual Return",
        "Volatility",
        "Win Rate",
        "Max Drawdown",
    ]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")
    formatted_df["Sharpe Ratio"] = formatted_df["Sharpe Ratio"].apply(
        lambda x: f"{x:.2f}" if not np.isnan(x) else "nan"
    )
    formatted_df["Number of Trades"] = formatted_df["Number of Trades"].apply(
        lambda x: f"{x:,}"
    )
    formatted_df["Profit Factor"] = formatted_df["Profit Factor"].apply(
        lambda x: f"{x:.2f}"
    )

    return formatted_df


def analyze_technical_issues(report_metrics):
    """Identify and analyze technical issues in the results"""
    print("\n=== TECHNICAL ISSUES ANALYSIS ===")

    issues = []

    for strategy, metrics in report_metrics.items():
        print(f"\nAnalyzing {strategy}:")

        # Check for NaN values
        nan_metrics = []
        for metric, value in metrics.items():
            if pd.isna(value):
                nan_metrics.append(metric)

        if nan_metrics:
            print(f"  ⚠️  NaN values found in: {', '.join(nan_metrics)}")
            issues.append(f"{strategy}: NaN values in {', '.join(nan_metrics)}")

        # Check for extreme drawdowns
        if metrics["Max Drawdown"] < -100:
            print(f"  ⚠️  Extreme drawdown: {metrics['Max Drawdown']:.2f}%")
            issues.append(
                f"{strategy}: Extreme drawdown of {metrics['Max Drawdown']:.2f}%"
            )

        # Check for zero or negative Sharpe ratio
        if not pd.isna(metrics["Sharpe Ratio"]) and metrics["Sharpe Ratio"] <= 0:
            print(f"  ⚠️  Non-positive Sharpe ratio: {metrics['Sharpe Ratio']:.2f}")
            issues.append(
                f"{strategy}: Non-positive Sharpe ratio of {metrics['Sharpe Ratio']:.2f}"
            )

        # Check for low win rate
        if metrics["Win Rate"] < 30:
            print(f"  ⚠️  Low win rate: {metrics['Win Rate']:.2f}%")
            issues.append(f"{strategy}: Low win rate of {metrics['Win Rate']:.2f}%")

    return issues


def main():
    """Main analysis function"""
    print("=== BACKTEST RESULTS ANALYSIS ===")

    # Load file paths
    files = load_backtest_data()

    # Parse report files
    report_metrics = {}
    for strategy, file_dict in files.items():
        print(f"\nParsing {strategy} report...")
        metrics = parse_report_file(file_dict["report"])
        report_metrics[strategy] = metrics
        print(f"  Total PnL: {metrics['Total PnL']:.4f}")
        print(f"  Win Rate: {metrics['Win Rate']:.2f}%")
        print(f"  Max Drawdown: {metrics['Max Drawdown']:.2f}%")

    # Create comparison table
    print("\n=== COMPREHENSIVE COMPARISON TABLE ===")
    comparison_table = create_comparison_table(report_metrics)
    print(comparison_table.to_string())

    # Analyze technical issues
    technical_issues = analyze_technical_issues(report_metrics)

    # Performance comparison analysis
    print("\n=== PERFORMANCE COMPARISON ANALYSIS ===")

    # USDCAD-WTI comparison
    print("\nUSDCAD-WTI Pair Comparison:")
    usdcad_kalman = report_metrics["USDCAD-WTI-Kalman"]
    usdcad_ols = report_metrics["USDCAD-WTI-OLS"]

    print(
        f"  Kalman Filter - PnL: {usdcad_kalman['Total PnL']:.4f}, Win Rate: {usdcad_kalman['Win Rate']:.2f}%, Max DD: {usdcad_kalman['Max Drawdown']:.2f}%"
    )
    print(
        f"  OLS - PnL: {usdcad_ols['Total PnL']:.4f}, Win Rate: {usdcad_ols['Win Rate']:.2f}%, Max DD: {usdcad_ols['Max Drawdown']:.2f}%"
    )

    if usdcad_ols["Total PnL"] > usdcad_kalman["Total PnL"]:
        print(
            f"  ✅ OLS outperformed Kalman by {usdcad_ols['Total PnL'] - usdcad_kalman['Total PnL']:.4f}"
        )
    else:
        print(
            f"  ✅ Kalman outperformed OLS by {usdcad_kalman['Total PnL'] - usdcad_ols['Total PnL']:.4f}"
        )

    # USDNOK-Brent comparison
    print("\nUSDNOK-Brent Pair Comparison:")
    usdnok_kalman = report_metrics["USDNOK-Brent-Kalman"]
    usdnok_ols = report_metrics["USDNOK-Brent-OLS"]

    print(
        f"  Kalman Filter - PnL: {usdnok_kalman['Total PnL']:.4f}, Win Rate: {usdnok_kalman['Win Rate']:.2f}%, Max DD: {usdnok_kalman['Max Drawdown']:.2f}%"
    )
    print(
        f"  OLS - PnL: {usdnok_ols['Total PnL']:.4f}, Win Rate: {usdnok_ols['Win Rate']:.2f}%, Max DD: {usdnok_ols['Max Drawdown']:.2f}%"
    )

    if usdnok_ols["Total PnL"] > usdnok_kalman["Total PnL"]:
        print(
            f"  ✅ OLS outperformed Kalman by {usdnok_ols['Total PnL'] - usdnok_kalman['Total PnL']:.4f}"
        )
    else:
        print(
            f"  ✅ Kalman outperformed OLS by {usdnok_kalman['Total PnL'] - usdnok_ols['Total PnL']:.4f}"
        )

    # Risk analysis
    print("\n=== RISK ANALYSIS ===")

    for strategy, metrics in report_metrics.items():
        print(f"\n{strategy}:")
        print(f"  Volatility: {metrics['Volatility']:.2f}%")
        print(f"  Max Drawdown: {metrics['Max Drawdown']:.2f}%")
        print(
            f"  Risk-Adjusted Return (PnL/Vol): {metrics['Total PnL']/metrics['Volatility']*100:.4f}"
        )

        # Calculate recovery ratio (simplified)
        if metrics["Max Drawdown"] < 0:
            recovery_ratio = abs(metrics["Total PnL"] / metrics["Max Drawdown"])
            print(f"  Recovery Ratio: {recovery_ratio:.4f}")

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")

    all_pnls = [metrics["Total PnL"] for metrics in report_metrics.values()]
    all_win_rates = [metrics["Win Rate"] for metrics in report_metrics.values()]
    all_drawdowns = [metrics["Max Drawdown"] for metrics in report_metrics.values()]

    print(f"Average PnL across all strategies: {np.mean(all_pnls):.4f}")
    print(f"Average Win Rate: {np.mean(all_win_rates):.2f}%")
    print(f"Average Max Drawdown: {np.mean(all_drawdowns):.2f}%")
    print(
        f"Best performing strategy: {max(report_metrics.keys(), key=lambda k: report_metrics[k]['Total PnL'])}"
    )
    print(
        f"Worst performing strategy: {min(report_metrics.keys(), key=lambda k: report_metrics[k]['Total PnL'])}"
    )

    # Save results to file
    with open("backtest_analysis_summary.txt", "w") as f:
        f.write("=== BACKTEST RESULTS ANALYSIS SUMMARY ===\n\n")
        f.write("COMPREHENSIVE COMPARISON TABLE:\n")
        f.write(comparison_table.to_string())
        f.write("\n\nTECHNICAL ISSUES:\n")
        for issue in technical_issues:
            f.write(f"- {issue}\n")

        f.write("\n\nKEY FINDINGS:\n")
        f.write(f"- Average PnL across all strategies: {np.mean(all_pnls):.4f}\n")
        f.write(f"- Average Win Rate: {np.mean(all_win_rates):.2f}%\n")
        f.write(f"- Average Max Drawdown: {np.mean(all_drawdowns):.2f}%\n")
        f.write(
            f"- Best performing strategy: {max(report_metrics.keys(), key=lambda k: report_metrics[k]['Total PnL'])}\n"
        )
        f.write(
            f"- Worst performing strategy: {min(report_metrics.keys(), key=lambda k: report_metrics[k]['Total PnL'])}\n"
        )

    print("\n✅ Analysis complete. Results saved to 'backtest_analysis_summary.txt'")


if __name__ == "__main__":
    main()
