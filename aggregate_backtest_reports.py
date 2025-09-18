#!/usr/bin/env python3
"""
Aggregate latest backtest reports from backtest_results/ into a markdown summary.

Parses lines created by [backtest.engine.create_backtest_report()](src/backtest/engine.py:342)
and builds docs/performance_summary.md with a consolidated table per pair.

Usage:
  PYTHONPATH=. python aggregate_backtest_reports.py
"""

import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESULTS_DIR = Path("backtest_results")
OUTPUT_MD = Path("docs/performance_summary.md")

REPORT_LINE_PATTERNS = {
    "period": re.compile(r"Period:\s*([0-9\-]+)\s*to\s*([0-9\-]+)"),
    "total_pnl": re.compile(r"Total PnL:\s*([\-0-9\.,]+)"),
    "total_return": re.compile(r"Total Return:\s*([\-0-9\.]+)%"),
    "annual_return": re.compile(r"Annual Return:\s*([\-0-9\.]+)%"),
    "volatility": re.compile(r"Volatility \(Annual\):\s*([\-0-9\.]+)%"),
    "sharpe_ratio": re.compile(r"Sharpe Ratio:\s*([\-0-9\.]+)"),
    "max_drawdown": re.compile(r"Maximum Drawdown:\s*([\-0-9\.]+)%"),
    "num_trades": re.compile(r"Number of Trades:\s*([0-9]+)"),
    "win_rate": re.compile(r"Win Rate:\s*([\-0-9\.]+)%"),
    "avg_win": re.compile(r"Average Win:\s*([\-0-9\.,]+)"),
    "avg_loss": re.compile(r"Average Loss:\s*([\-0-9\.,]+)"),
    "profit_factor": re.compile(r"Profit Factor:\s*([\-0-9\.]+)"),
}

TS_RE = re.compile(r"_report_(\d{8})_(\d{6})\.txt$")

def parse_report(report_path: Path) -> dict:
    metrics = {}
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    for key, pat in REPORT_LINE_PATTERNS.items():
        m = pat.search(text)
        if m:
            if key in {"period"}:
                metrics["start"], metrics["end"] = m.group(1), m.group(2)
            else:
                val = m.group(1).replace(",", "")
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
        else:
            # leave missing; handle downstream
            pass
    return metrics

def pick_latest_reports() -> dict:
    """Return {pair: Path-to-latest-report}"""
    latest = {}
    for p in RESULTS_DIR.glob("*_report_*.txt"):
        name = p.name
        # Pair is prefix before _report_
        try:
            pair = name.split("_report_")[0]
        except Exception:
            continue
        m = TS_RE.search(name)
        if not m:
            continue
        ts = m.group(1) + m.group(2)  # YYYYMMDDHHMMSS
        if pair not in latest or ts > latest[pair][0]:
            latest[pair] = (ts, p)
    # strip timestamps
    return {pair: path for pair, (ts, path) in latest.items()}

def build_summary_table(latest_reports: dict) -> str:
    rows = []
    header = (
        "| Pair | Start | End | Total Return | Annual Return | Sharpe | MaxDD | "
        "Trades | Win Rate | Profit Factor | Report |\n"
        "|------|-------|-----|--------------|---------------|--------|-------|"
        "--------|----------|---------------|--------|\n"
    )
    for pair, path in sorted(latest_reports.items()):
        metrics = parse_report(path)
        start = metrics.get("start", "")
        end = metrics.get("end", "")
        tr = metrics.get("total_return", 0.0)
        ar = metrics.get("annual_return", 0.0)
        sh = metrics.get("sharpe_ratio", 0.0)
        mdd = metrics.get("max_drawdown", 0.0)
        ntr = int(metrics.get("num_trades", 0))
        wr = metrics.get("win_rate", 0.0)
        pf = metrics.get("profit_factor", 0.0)
        # Relative link to report file
        report_link = f"[link]({path.as_posix()})"
        row = (
            f"| {pair} | {start} | {end} | {tr:.2f}% | {ar:.2f}% | {sh:.2f} | "
            f"{mdd:.2f}% | {ntr} | {wr:.2f}% | {pf:.2f} | {report_link} |\n"
        )
        rows.append(row)
    return header + "".join(rows)

def main():
    latest = pick_latest_reports()
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Backtest Performance Summary\n",
        "",
        "This table aggregates the latest reports produced by "
        "[backtest.engine.create_backtest_report()](src/backtest/engine.py:342).",
        "",
        build_summary_table(latest),
        "",
        "Notes:",
        "- Values include one-bar execution delay and configured transaction costs.",
        "- Each row reflects the latest available report file per pair under backtest_results/.",
        "",
        f"Generated at: {datetime.utcnow().isoformat()}Z",
    ]
    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD.as_posix()} with {len(latest)} rows.")

if __name__ == "__main__":
    main()