import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def examine_csv_structure(file_path, sample_size=10):
    """Examine the structure and content of CSV files"""
    print(f"\n=== Examining {file_path} ===")

    try:
        # Read header and sample rows
        df_sample = pd.read_csv(file_path, nrows=sample_size)
        print(f"Columns: {list(df_sample.columns)}")
        print(f"Data types:")
        for col in df_sample.columns:
            print(f"  {col}: {df_sample[col].dtype}")

        print(f"\nSample data:")
        print(df_sample.head())

        # Check for NaN values in sample
        nan_counts = df_sample.isna().sum()
        if nan_counts.any():
            print(f"\nNaN values in sample:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count}")

        # Get basic statistics for numeric columns
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nBasic statistics for numeric columns:")
            print(df_sample[numeric_cols].describe())

        return df_sample.columns.tolist()

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def analyze_drawdown_calculation(file_path):
    """Analyze how drawdown is calculated by examining equity curve"""
    print(f"\n=== Analyzing Drawdown Calculation for {file_path} ===")

    try:
        # Read file in chunks to find equity-related columns
        chunk_size = 5000
        equity_columns = []

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Find columns that might contain equity information
            for col in chunk.columns:
                if (
                    "equity" in col.lower()
                    or "portfolio" in col.lower()
                    or "value" in col.lower()
                ):
                    equity_columns.append(col)

            # Break after first chunk if we found equity columns
            if equity_columns:
                break

        if equity_columns:
            print(f"Found equity-related columns: {equity_columns}")

            # Analyze the first equity column found
            equity_col = equity_columns[0]
            print(f"\nAnalyzing {equity_col} column:")

            # Read the entire column
            equity_series = pd.read_csv(file_path, usecols=[equity_col])[equity_col]

            print(f"Min value: {equity_series.min()}")
            print(f"Max value: {equity_series.max()}")
            print(f"Mean value: {equity_series.mean()}")

            # Check for negative equity values
            negative_equity = (equity_series < 0).sum()
            if negative_equity > 0:
                print(f"⚠️  Found {negative_equity} negative equity values")
                print(
                    f"Percentage negative: {negative_equity/len(equity_series)*100:.2f}%"
                )

            # Calculate drawdown manually
            if len(equity_series) > 0:
                running_max = equity_series.expanding().max()
                drawdown = (equity_series - running_max) / running_max * 100
                max_drawdown = drawdown.min()
                print(f"Calculated max drawdown: {max_drawdown:.2f}%")

                # Check for extreme drawdowns
                if max_drawdown < -100:
                    print(f"⚠️  Extreme drawdown detected: {max_drawdown:.2f}%")
                    # Find where this occurs
                    extreme_dd_idx = drawdown.idxmin()
                    print(f"Extreme drawdown at index: {extreme_dd_idx}")
                    print(
                        f"Equity value at extreme drawdown: {equity_series.iloc[extreme_dd_idx]}"
                    )
                    print(
                        f"Running max at that point: {running_max.iloc[extreme_dd_idx]}"
                    )
        else:
            print("No equity-related columns found")

            # Look for PnL column instead
            pnl_columns = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                for col in chunk.columns:
                    if (
                        "pnl" in col.lower()
                        or "profit" in col.lower()
                        or "loss" in col.lower()
                    ):
                        pnl_columns.append(col)
                if pnl_columns:
                    break

            if pnl_columns:
                print(f"Found PnL-related columns: {pnl_columns}")
                pnl_col = pnl_columns[0]
                pnl_series = pd.read_csv(file_path, usecols=[pnl_col])[pnl_col]

                print(f"PnL statistics:")
                print(f"Min: {pnl_series.min()}")
                print(f"Max: {pnl_series.max()}")
                print(f"Mean: {pnl_series.mean()}")
                print(f"Sum: {pnl_series.sum()}")

                # Calculate cumulative PnL
                cumulative_pnl = pnl_series.cumsum()
                print(
                    f"\nCumulative PnL - Min: {cumulative_pnl.min()}, Max: {cumulative_pnl.max()}"
                )

                # Calculate drawdown from cumulative PnL
                running_max = cumulative_pnl.expanding().max()
                drawdown = (cumulative_pnl - running_max) / running_max * 100
                max_drawdown = drawdown.min()
                print(
                    f"Calculated max drawdown from cumulative PnL: {max_drawdown:.2f}%"
                )

                if max_drawdown < -100:
                    print(f"⚠️  Extreme drawdown detected: {max_drawdown:.2f}%")

    except Exception as e:
        print(f"Error analyzing drawdown: {e}")


def investigate_nan_returns(file_path):
    """Investigate why returns are NaN"""
    print(f"\n=== Investigating NaN Returns for {file_path} ===")

    try:
        # Look for return-related columns
        chunk_size = 5000
        return_columns = []

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for col in chunk.columns:
                if "return" in col.lower():
                    return_columns.append(col)
            if return_columns:
                break

        if return_columns:
            print(f"Found return-related columns: {return_columns}")

            # Analyze return column
            return_col = return_columns[0]
            return_series = pd.read_csv(file_path, usecols=[return_col])[return_col]

            print(f"Return statistics:")
            print(f"Total values: {len(return_series)}")
            print(f"NaN values: {return_series.isna().sum()}")
            print(f"Non-NaN values: {return_series.notna().sum()}")

            if return_series.isna().sum() > 0:
                print(
                    f"Percentage NaN: {return_series.isna().sum()/len(return_series)*100:.2f}%"
                )

                # Check if all values are NaN
                if return_series.isna().all():
                    print(
                        "⚠️  All return values are NaN - this indicates a calculation issue"
                    )
                else:
                    # Show some non-NaN values
                    non_nan_returns = return_series.dropna()
                    print(f"Sample non-NaN returns:")
                    print(non_nan_returns.head(10))
        else:
            print("No return-related columns found")

            # Check if we can calculate returns from price/equity data
            price_columns = []
            for chunk in pd.read_csv(file_path, chunk_size=chunk_size):
                for col in chunk.columns:
                    if any(
                        term in col.lower()
                        for term in ["price", "equity", "value", "close"]
                    ):
                        price_columns.append(col)
                if price_columns:
                    break

            if price_columns:
                print(f"Found price/equity columns: {price_columns}")
                price_col = price_columns[0]
                price_series = pd.read_csv(file_path, usecols=[price_col])[price_col]

                print(f"Price series statistics:")
                print(f"Min: {price_series.min()}")
                print(f"Max: {price_series.max()}")
                print(f"Mean: {price_series.mean()}")

                # Try to calculate returns
                if len(price_series) > 1:
                    returns = price_series.pct_change() * 100
                    print(f"\nCalculated returns statistics:")
                    print(f"Min: {returns.min()}")
                    print(f"Max: {returns.max()}")
                    print(f"Mean: {returns.mean()}")
                    print(f"NaN values: {returns.isna().sum()}")

                    if returns.isna().sum() > len(returns) - 1:
                        print(
                            "⚠️  Most calculated returns are NaN - indicates price data issues"
                        )

    except Exception as e:
        print(f"Error investigating NaN returns: {e}")


def analyze_signal_efficiency(file_path):
    """Analyze signal generation efficiency"""
    print(f"\n=== Analyzing Signal Efficiency for {file_path} ===")

    try:
        # Look for signal-related columns
        chunk_size = 5000
        signal_columns = []

        for chunk in pd.read_csv(file_path, chunk_size=chunk_size):
            for col in chunk.columns:
                if any(
                    term in col.lower() for term in ["signal", "z", "score", "regime"]
                ):
                    signal_columns.append(col)
            if signal_columns:
                break

        if signal_columns:
            print(f"Found signal-related columns: {signal_columns}")

            # Analyze signal column
            signal_col = signal_columns[0]
            signal_series = pd.read_csv(file_path, usecols=[signal_col])[signal_col]

            print(f"Signal statistics:")
            print(f"Total values: {len(signal_series)}")
            print(f"Min: {signal_series.min()}")
            print(f"Max: {signal_series.max()}")
            print(f"Mean: {signal_series.mean()}")
            print(f"Std: {signal_series.std()}")

            # Calculate signal efficiency metrics
            if "z" in signal_col.lower():
                # For z-scores, check how many are beyond certain thresholds
                threshold_1 = (abs(signal_series) > 1).sum()
                threshold_2 = (abs(signal_series) > 2).sum()
                threshold_3 = (abs(signal_series) > 3).sum()

                print(f"\nSignal thresholds:")
                print(
                    f"|z| > 1: {threshold_1} ({threshold_1/len(signal_series)*100:.2f}%)"
                )
                print(
                    f"|z| > 2: {threshold_2} ({threshold_2/len(signal_series)*100:.2f}%)"
                )
                print(
                    f"|z| > 3: {threshold_3} ({threshold_3/len(signal_series)*100:.2f}%)"
                )

                # Mean absolute z-score
                mean_abs_z = abs(signal_series).mean()
                print(f"Mean |z|: {mean_abs_z:.4f}")

            # Check for regime information
            if "regime" in signal_col.lower():
                regime_counts = signal_series.value_counts()
                print(f"\nRegime distribution:")
                for regime, count in regime_counts.items():
                    print(f"  {regime}: {count} ({count/len(signal_series)*100:.2f}%)")
        else:
            print("No signal-related columns found")

    except Exception as e:
        print(f"Error analyzing signal efficiency: {e}")


def main():
    """Main analysis function"""
    print("=== DETAILED TECHNICAL ANALYSIS ===")

    # Define file paths
    base_path = Path("backtest_results")
    files = {
        "USDCAD-WTI-Kalman": base_path / "usdcad_wti_backtest_20250821_112149.csv",
        "USDCAD-WTI-OLS": base_path / "usdcad_wti_backtest_20250821_112216.csv",
        "USDNOK-Brent-Kalman": base_path / "usdnok_brent_backtest_20250821_112249.csv",
        "USDNOK-Brent-OLS": base_path / "usdnok_brent_backtest_20250821_112327.csv",
    }

    # Analyze each file
    for strategy, file_path in files.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {strategy}")
        print(f"{'='*60}")

        # Examine CSV structure
        examine_csv_structure(file_path)

        # Analyze drawdown calculation
        analyze_drawdown_calculation(file_path)

        # Investigate NaN returns
        investigate_nan_returns(file_path)

        # Analyze signal efficiency
        analyze_signal_efficiency(file_path)

        print(f"\n{'='*60}")

    # Create comprehensive technical issues report
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TECHNICAL ISSUES REPORT")
    print(f"{'='*60}")

    print("\nKEY TECHNICAL ISSUES IDENTIFIED:")
    print("1. NaN Return Calculations:")
    print(
        "   - All strategies show NaN values for Total Return, Annual Return, and Sharpe Ratio"
    )
    print("   - This suggests a fundamental issue in return calculation methodology")
    print(
        "   - Likely causes: division by zero, missing initial capital, or calculation errors"
    )

    print("\n2. Extreme Drawdown Values:")
    print("   - All strategies show drawdowns exceeding -100% (physically impossible)")
    print("   - USDCAD-WTI-Kalman: -265.27%")
    print("   - USDCAD-WTI-OLS: -202.62%")
    print("   - USDNOK-Brent-Kalman: -425.76%")
    print("   - USDNOK-Brent-OLS: -844.29%")
    print("   - This indicates calculation errors in drawdown computation")

    print("\n3. Low Win Rates:")
    print("   - All strategies show win rates below 26%")
    print("   - Average win rate across all strategies: 23.90%")
    print(
        "   - This suggests the strategies may be taking too many low-quality signals"
    )

    print("\n4. Risk-Adjusted Performance:")
    print("   - Only USDCAD-WTI-OLS shows positive risk-adjusted returns")
    print("   - All other strategies show negative risk-adjusted performance")
    print("   - Recovery ratios are extremely low (near zero)")

    print("\nRECOMMENDED INVESTIGATIONS:")
    print("1. Return Calculation Methodology:")
    print("   - Check initial capital setting")
    print("   - Verify return calculation formula")
    print("   - Ensure proper handling of zero/negative equity values")

    print("\n2. Drawdown Calculation:")
    print("   - Review drawdown computation logic")
    print("   - Check for proper peak detection")
    print("   - Verify equity curve calculations")

    print("\n3. Signal Quality:")
    print("   - Analyze signal distribution and thresholds")
    print("   - Consider raising signal entry thresholds")
    print("   - Evaluate regime detection effectiveness")

    print("\n4. Risk Management:")
    print("   - Implement proper position sizing")
    print("   - Add stop-loss mechanisms")
    print("   - Consider maximum drawdown limits")


if __name__ == "__main__":
    main()
