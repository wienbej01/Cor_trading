import click
import json
import pandas as pd
from pathlib import Path

def load_summary(file_path: Path) -> dict:
    """Loads a summary.json file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_comparison_table(summaries: dict) -> str:
    """Generates a markdown table comparing the metrics."""
    
    # Define the metrics to compare
    metrics_to_compare = [
        'total_return',
        'annual_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'num_trades',
        'avg_trade_duration'
    ]
    
    header = "| Metric | " + " | ".join(summaries.keys()) + " |"
    separator = "|---|" + "---|" * len(summaries)
    
    rows = []
    for metric in metrics_to_compare:
        row = f"| {metric} |"
        for summary in summaries.values():
            val = summary.get(metric, 'N/A')
            if isinstance(val, float):
                if 'rate' in metric or 'return' in metric or 'drawdown' in metric:
                    row += f" {val:.2%} |"
                else:
                    row += f" {val:.2f} |"
            else:
                row += f" {val} |"
        rows.append(row)
        
    return "\n".join([header, separator] + rows)


@click.command()
@click.option('--runs', multiple=True, required=True, help='Paths to summary.json files to compare. Use format <name>:<path>.')
@click.option('--out', required=True, type=click.Path(), help='Output markdown file path.')
def main(runs, out):
    """
    Compares multiple backtest runs and generates a markdown report.
    """
    click.echo(f"Comparing {len(runs)} backtest runs...")
    
    summaries = {}
    for run_arg in runs:
        try:
            name, path_str = run_arg.split(':', 1)
            path = Path(path_str)
            if not path.exists():
                click.echo(f"Warning: File not found at {path}", err=True)
                continue
            summaries[name] = load_summary(path)
        except ValueError:
            click.echo(f"Invalid format for --runs argument: {run_arg}. Use <name>:<path>.", err=True)
            continue

    if not summaries:
        click.echo("No valid summary files found to compare.", err=True)
        return

    # Generate report
    report = "# Backtest Comparison Report\n\n"
    report += "## Summary Metrics\n\n"
    report += generate_comparison_table(summaries)
    report += "\n\n"
    report += "## Full Summaries\n\n"
    
    for name, summary in summaries.items():
        report += f"### {name}\n\n"
        report += "```json\n"
        report += json.dumps(summary, indent=4)
        report += "\n```\n\n"

    # Write report to file
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
        
    click.echo(f"Comparison report saved to {output_path}")

if __name__ == '__main__':
    main()
