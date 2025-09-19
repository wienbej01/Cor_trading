import argparse
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

# Matplotlib is assumed available per requirements-dev
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_year_from_record(data: Dict, run_id: str) -> Optional[int]:
    """
    Try to parse a year from summary.json content following robust fallbacks.
    """
    try:
        # Preferred: direct timestamp
        if "timestamp" in data and isinstance(data["timestamp"], str):
            # Accept ISO8601 with or without Z
            ts = data["timestamp"]
            if ts.endswith("Z"):
                ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            return dt.year

        # Fallback: config.start_date
        if "config" in data and isinstance(data["config"], dict):
            start_date = data["config"].get("start_date")
            if isinstance(start_date, str) and start_date != "unknown":
                dt = datetime.fromisoformat(start_date)
                return dt.year

        # Fallback: parse date from run_id prefix like YYYYMMDD_*
        # e.g., 20250115_102233
        prefix = run_id.split("_")[0]
        dt = datetime.strptime(prefix, "%Y%m%d")
        return dt.year
    except Exception:
        return None


def _set_plot_style():
    # Deterministic style
    plt.style.use("ggplot")


def _save_bar_plot(
    out_path: Path,
    x_labels: List[str],
    y_values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    rotation: int = 0,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x_labels, y_values, color="#4472c4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotation:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _detect_bucket_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect volatility and trend bucket columns in the flattened summary DataFrame.
    Returns a tuple: (vol_bucket_col, trend_bucket_col), either may be None.
    Detection rules: look for columns named exactly or containing:
      - volatility: 'vol_bucket', 'volatility_bucket'
      - trend: 'trend_bucket', 'trend'
    Prefer more specific names when multiple are found.
    """
    cols = list(df.columns)

    # Candidates ordered by specificity
    vol_candidates = [
        "vol_bucket",
        "volatility_bucket",
    ]
    trend_candidates = [
        "trend_bucket",
        "trend",
    ]

    def pick(colnames: List[str], candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in colnames:
                return c
        # fallback: contains-like (less strict)
        for c in candidates:
            matches = [col for col in colnames if c in col]
            if matches:
                # take shortest match for stability
                return sorted(matches, key=len)[0]
        return None

    vol_col = pick(cols, vol_candidates)
    trend_col = pick(cols, trend_candidates)
    return vol_col, trend_col


def _available_optional_metrics(df: pd.DataFrame) -> Dict[str, str]:
    """
    Determine which optional metrics are available in summary df for plotting/aggregation.
    Returns mapping of metric key to column name in df.
    - 'sharpe' - prefers equity.sharpe_ratio
    - 'profit_factor' - uses trades.profit_factor (if finite values exist)
    """
    out: Dict[str, str] = {}
    if "equity.sharpe_ratio" in df.columns:
        out["sharpe"] = "equity.sharpe_ratio"
    if "trades.profit_factor" in df.columns:
        out["profit_factor"] = "trades.profit_factor"
    return out


def generate_report(input_dir: str, output_file: str, plots_out: str = "reports/plots") -> None:
    """
    Generates a bucketed performance report with plots and optional vol/trend aggregations.

    - Scans input_dir recursively for **/summary.json
    - Aggregates by 2-year buckets per pair for Return % (mean) and Trades (sum)
    - Produces per-pair plots under plots_out/<pair>/
    - Updates markdown with:
        - Per-pair tables
        - Plots subsection (linked)
        - Volatility and Trend Buckets subsection (tables/plots if available; otherwise, a standardized note)
    """
    report_paths = glob.glob(f"{input_dir}/**/summary.json", recursive=True)

    # Prepare output directory for plots
    plots_root = Path(plots_out)
    _ensure_dir(plots_root)

    if not report_paths:
        print("No summary.json files found.")
        with open(output_file, "w") as f:
            f.write("# Bucketed Performance Report\n\n")
            f.write("No backtest reports found to generate the report.\n")
        return

    all_data: List[Dict] = []
    for path in report_paths:
        p = Path(path)
        run_id = p.parent.name
        pair = p.parent.parent.name

        with open(path, "r") as f:
            data = json.load(f)

        # Flatten at top-level for normalization while keeping pair/run_id/year
        record: Dict = {}
        record.update(data if isinstance(data, dict) else {})
        record["pair"] = pair
        record["run_id"] = run_id

        year = _parse_year_from_record(data, run_id=run_id)
        record["year"] = year

        all_data.append(record)

    # Flatten nested dict keys: e.g., equity.total_return
    df = pd.json_normalize(all_data)

    # Ensure numeric types for aggregation
    if "equity.total_return" in df.columns:
        df["equity.total_return"] = pd.to_numeric(df["equity.total_return"], errors="coerce")
    if "trades.total_trades" in df.columns:
        df["trades.total_trades"] = pd.to_numeric(df["trades.total_trades"], errors="coerce")
    if "equity.sharpe_ratio" in df.columns:
        df["equity.sharpe_ratio"] = pd.to_numeric(df["equity.sharpe_ratio"], errors="coerce")
    if "trades.profit_factor" in df.columns:
        df["trades.profit_factor"] = pd.to_numeric(df["trades.profit_factor"], errors="coerce")

    # Drop rows without a year
    if "year" in df.columns:
        df = df.dropna(subset=["year"])
        if not df.empty:
            df["year"] = df["year"].astype(int)

    if df.empty:
        # No valid data rows with year; still emit minimal report
        with open(output_file, "w") as f:
            f.write("# Bucketed Performance by Pair and Regime\n\n")
            f.write("This report segments performance by 2-year time buckets.\n\n")
            f.write("No valid records with year information were found.\n")
        print(f"Report generated at {output_file}")
        return

    # Create 2-year buckets
    df["bucket"] = (df["year"] // 2 * 2).astype(str) + "-" + ((df["year"] // 2 * 2) + 1).astype(str)

    # Per-pair breakdown (2-year)
    agg_spec: Dict[str, Tuple[str, str]] = {}
    if "equity.total_return" in df.columns:
        agg_spec["return_pct"] = ("equity.total_return", "mean")
    if "trades.total_trades" in df.columns:
        agg_spec["trades"] = ("trades.total_trades", "sum")

    pair_breakdown = pd.DataFrame()
    if agg_spec:
        pair_breakdown = (
            df.groupby(["pair", "bucket"]).agg(**agg_spec).reset_index()
        )
    if not pair_breakdown.empty:
        # Rename for presentation
        rename_map = {}
        if "return_pct" in pair_breakdown.columns:
            rename_map["return_pct"] = "Return %"
        if "trades" in pair_breakdown.columns:
            rename_map["trades"] = "Trades"
        pair_breakdown = pair_breakdown.rename(columns=rename_map)

    # Optional metrics available for plotting
    opt_metrics = _available_optional_metrics(df)

    # Prepare markdown
    output_dir = Path(output_file).parent
    _ensure_dir(output_dir)

    with open(output_file, "w") as f:
        f.write("# Bucketed Performance by Pair and Regime\n\n")
        f.write("This report segments performance by 2-year time buckets.\n\n")

        f.write("## Per-Pair Breakdowns\n\n")

        for pair in sorted(df["pair"].dropna().unique()):
            f.write(f"### {pair}\n\n")

            # Slice pair rows
            pair_df = pair_breakdown[pair_breakdown["pair"] == pair] if not pair_breakdown.empty else pd.DataFrame()

            # Present main table for this pair
            if not pair_df.empty:
                view_cols: List[str] = ["bucket"]
                if "Return %" in pair_df.columns:
                    view_cols.append("Return %")
                if "Trades" in pair_df.columns:
                    view_cols.append("Trades")
                f.write(pair_df[view_cols].sort_values("bucket").to_markdown(index=False))
            else:
                f.write("No data available for this pair.")
            f.write("\n\n")

            # PLOTS subsection
            f.write("#### Plots\n\n")
            pair_plot_dir = plots_root / pair
            _ensure_dir(pair_plot_dir)

            # 2-year Return % plot
            two_year_return_plot_rel: Optional[str] = None
            if not pair_df.empty and "Return %" in pair_df.columns:
                sorted_pair = pair_df.sort_values("bucket")
                by2y_path = pair_plot_dir / "by_2y_return.png"
                _save_bar_plot(
                    by2y_path,
                    x_labels=sorted_pair["bucket"].astype(str).tolist(),
                    y_values=sorted_pair["Return %"].astype(float).fillna(0.0).tolist(),
                    title=f"{pair} — Return % by 2-year window",
                    xlabel="2-year window",
                    ylabel="Return %",
                    rotation=45,
                )
                two_year_return_plot_rel = os.path.relpath(by2y_path, start=output_dir)
                f.write(f"![2-year Return %]({two_year_return_plot_rel})\n\n")
            else:
                f.write("_2-year Return % plot not available._\n\n")

            # Optional second plot: Sharpe or Profit Factor by 2-year window
            # Compute grouped means for the metric if available
            metric_label = None
            metric_col = None
            if "sharpe" in opt_metrics:
                metric_label = "Sharpe Ratio"
                metric_col = opt_metrics["sharpe"]
            elif "profit_factor" in opt_metrics:
                metric_label = "Profit Factor"
                metric_col = opt_metrics["profit_factor"]

            if metric_col is not None:
                # Build grouped metric per 2-year bucket for this pair
                sub = df[df["pair"] == pair]
                grp = (
                    sub.groupby("bucket")[metric_col].mean().reset_index()
                )
                if not grp.empty:
                    metric_plot_path = pair_plot_dir / f"by_2y_{'sharpe' if metric_label=='Sharpe Ratio' else 'profit_factor'}.png"
                    _save_bar_plot(
                        metric_plot_path,
                        x_labels=grp["bucket"].astype(str).tolist(),
                        y_values=pd.to_numeric(grp[metric_col], errors="coerce").fillna(0.0).tolist(),
                        title=f"{pair} — {metric_label} by 2-year window",
                        xlabel="2-year window",
                        ylabel=metric_label,
                        rotation=45,
                    )
                    metric_plot_rel = os.path.relpath(metric_plot_path, start=output_dir)
                    f.write(f"![2-year {metric_label}]({metric_plot_rel})\n\n")

            # VOLATILITY AND TREND BUCKETS subsection
            f.write("#### Volatility and Trend Buckets\n\n")

            # Detect bucket columns at summary level
            vol_col, trend_col = _detect_bucket_columns(df)

            # Helper to build and write a subsection for a bucket column
            def write_bucket_section(bucket_col: str, section_name: str, filename_stub: str) -> bool:
                """
                Returns True if written (available), False if no data to show.
                Aggregates Return %, Trades, and optional Sharpe/Profit Factor by bucket.
                """
                sub = df[df["pair"] == pair]
                # Need at least the bucket column present and some metrics
                if bucket_col not in sub.columns:
                    return False

                # At minimum, require Return % (equity.total_return) or Trades to present anything meaningful.
                got_return = "equity.total_return" in sub.columns and not sub["equity.total_return"].dropna().empty
                got_trades = "trades.total_trades" in sub.columns and not sub["trades.total_trades"].dropna().empty

                if not got_return and not got_trades:
                    return False

                agg_parts: Dict[str, Tuple[str, str]] = {}
                if got_return:
                    agg_parts["Return %"] = ("equity.total_return", "mean")
                if got_trades:
                    agg_parts["Trades"] = ("trades.total_trades", "sum")

                # Optional metrics
                if "equity.sharpe_ratio" in sub.columns and not sub["equity.sharpe_ratio"].dropna().empty:
                    agg_parts["Sharpe"] = ("equity.sharpe_ratio", "mean")
                if "trades.profit_factor" in sub.columns and not sub["trades.profit_factor"].dropna().empty:
                    agg_parts["Profit Factor"] = ("trades.profit_factor", "mean")

                grp = (
                    sub.groupby(bucket_col).agg(**agg_parts).reset_index().rename(columns={bucket_col: "Bucket"})
                )
                if grp.empty:
                    return False

                f.write(f"Bucket: {section_name}\n\n")
                # Determine presentation columns order
                present_cols: List[str] = ["Bucket"]
                for cname in ["Return %", "Sharpe", "Profit Factor", "Trades"]:
                    if cname in grp.columns:
                        present_cols.append(cname)

                f.write(grp[present_cols].sort_values("Bucket").to_markdown(index=False))
                f.write("\n\n")

                # Plot Return % by bucket if available
                if "Return %" in grp.columns:
                    plot_path = (plots_root / pair) / f"by_{filename_stub}_return.png"
                    _ensure_dir(plot_path.parent)
                    _save_bar_plot(
                        plot_path,
                        x_labels=grp["Bucket"].astype(str).tolist(),
                        y_values=pd.to_numeric(grp["Return %"], errors="coerce").fillna(0.0).tolist(),
                        title=f"{pair} — Return % by {section_name}",
                        xlabel=section_name,
                        ylabel="Return %",
                        rotation=45,
                    )
                    plot_rel = os.path.relpath(plot_path, start=output_dir)
                    f.write(f"![Return % by {section_name}]({plot_rel})\n\n")
                return True

            any_bucket_written = False
            if vol_col:
                any_bucket_written = write_bucket_section(vol_col, "Volatility Bucket", "vol_bucket") or any_bucket_written
            if trend_col:
                any_bucket_written = write_bucket_section(trend_col, "Trend Bucket", "trend_bucket") or any_bucket_written

            if not any_bucket_written:
                f.write("Not available: vol/trend bucket fields were not present in run artifacts for this pair.\n\n")

        # End file write

    print(f"Report generated at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a bucketed performance report with plots and optional volatility/trend bucket aggregations."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing backtest reports.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output markdown file for the report.",
    )
    parser.add_argument(
        "--plots-out",
        type=str,
        default="reports/plots",
        help="Directory to write plot PNGs (default: reports/plots). When volatility/trend bucket fields are unavailable in artifacts, those sections will include a standardized 'Not available' note.",
    )
    args = parser.parse_args()

    generate_report(args.input, args.out, plots_out=args.plots_out)
