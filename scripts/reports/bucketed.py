
import argparse
import json
import glob
from pathlib import Path
import pandas as pd
from datetime import datetime

def generate_report(input_dir: str, output_file: str):
    """
    Generates a bucketed performance report.
    """
    report_paths = glob.glob(f"{input_dir}/**/summary.json", recursive=True)
    
    if not report_paths:
        print("No summary.json files found.")
        with open(output_file, "w") as f:
            f.write("# Bucketed Performance Report\n\n")
            f.write("No backtest reports found to generate the report.\n")
        return

    all_data = []
    for path in report_paths:
        p = Path(path)
        run_id = p.parent.name
        pair = p.parent.parent.name
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        data['pair'] = pair
        data['run_id'] = run_id
        
        # Extract year from a timestamp, if available
        try:
            # Try to parse different timestamp formats
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                dt = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                data['year'] = dt.year
            elif 'config' in data and 'start_date' in data['config'] and data['config']['start_date'] != 'unknown':
                dt = datetime.fromisoformat(data['config']['start_date'])
                data['year'] = dt.year
            else:
                 # Fallback to parsing from run_id if it contains a date
                dt = datetime.strptime(run_id.split('_')[0], '%Y%m%d')
                data['year'] = dt.year
        except (ValueError, TypeError, IndexError):
            data['year'] = None # Cannot determine year

        all_data.append(data)

    df = pd.json_normalize(all_data)
    
    # Ensure numeric types for aggregation
    for col in ['equity.total_return', 'trades.total_trades']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Create 2-year buckets
    df['bucket'] = (df['year'] // 2 * 2).astype(str) + '-' + ((df['year'] // 2 * 2) + 1).astype(str)

    # Per-pair breakdown
    pair_breakdown = df.groupby(['pair', 'bucket']).agg(
        return_pct=('equity.total_return', 'mean'),
        trades=('trades.total_trades', 'sum')
    ).reset_index()
    pair_breakdown.rename(columns={'return_pct': 'Return %', 'trades': 'Trades'}, inplace=True)


    with open(output_file, "w") as f:
        f.write("# Bucketed Performance by Pair and Regime\n\n")
        f.write("This report segments performance by 2-year time buckets.\n\n")
        
        f.write("## Per-Pair Breakdowns\n\n")

        for pair in sorted(pair_breakdown['pair'].unique()):
            f.write(f"### {pair}\n\n")
            pair_df = pair_breakdown[pair_breakdown['pair'] == pair][['bucket', 'Return %', 'Trades']]
            
            if not pair_df.empty:
                f.write(pair_df.to_markdown(index=False))
            else:
                f.write("No data available for this pair.")
            f.write("\n\n")

    print(f"Report generated at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a bucketed performance report.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing backtest reports.")
    parser.add_argument("--out", type=str, required=True, help="Output markdown file for the report.")
    args = parser.parse_args()

    generate_report(args.input, args.out)
