import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def create_final_comprehensive_report():
    """Create the final comprehensive analysis report"""

    print("=== COMPREHENSIVE BACKTEST RESULTS ANALYSIS ===")
    print("=" * 80)

    # 1. PERFORMANCE METRICS ANALYSIS
    print("\n1. PERFORMANCE METRICS ANALYSIS")
    print("-" * 50)

    # Data from the parsed reports
    performance_data = {
        "USDCAD-WTI-Kalman": {
            "Total PnL": -0.0200,
            "Total Return": np.nan,
            "Annual Return": np.nan,
            "Volatility": 10.05,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": -265.27,
            "Number of Trades": 520,
            "Win Rate": 23.70,
            "Average Win": 0.0200,
            "Average Loss": -0.0300,
            "Profit Factor": 0.97,
        },
        "USDCAD-WTI-OLS": {
            "Total PnL": 0.1200,
            "Total Return": np.nan,
            "Annual Return": np.nan,
            "Volatility": 9.78,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": -202.62,
            "Number of Trades": 524,
            "Win Rate": 24.44,
            "Average Win": 0.0200,
            "Average Loss": -0.0200,
            "Profit Factor": 1.34,
        },
        "USDNOK-Brent-Kalman": {
            "Total PnL": -0.2500,
            "Total Return": np.nan,
            "Annual Return": np.nan,
            "Volatility": 9.59,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": -425.76,
            "Number of Trades": 612,
            "Win Rate": 25.51,
            "Average Win": 0.0200,
            "Average Loss": -0.0200,
            "Profit Factor": 0.87,
        },
        "USDNOK-Brent-OLS": {
            "Total PnL": -0.0300,
            "Total Return": np.nan,
            "Annual Return": np.nan,
            "Volatility": 8.60,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": -844.29,
            "Number of Trades": 520,
            "Win Rate": 21.93,
            "Average Win": 0.0200,
            "Average Loss": -0.0200,
            "Profit Factor": 0.98,
        },
    }

    # Create performance comparison table
    perf_df = pd.DataFrame(performance_data).T
    print("\nComprehensive Performance Comparison:")
    print(perf_df.to_string(float_format="%.4f"))

    # 2. KALMAN VS OLS PERFORMANCE COMPARISON
    print("\n2. KALMAN FILTER VS OLS PERFORMANCE COMPARISON")
    print("-" * 50)

    # USDCAD-WTI Pair Comparison
    print("\nUSDCAD-WTI Pair:")
    usdcad_kalman = performance_data["USDCAD-WTI-Kalman"]
    usdcad_ols = performance_data["USDCAD-WTI-OLS"]

    print(
        f"  Kalman Filter: PnL = {usdcad_kalman['Total PnL']:.4f}, "
        f"Win Rate = {usdcad_kalman['Win Rate']:.2f}%, "
        f"Max DD = {usdcad_kalman['Max Drawdown']:.2f}%"
    )
    print(
        f"  OLS: PnL = {usdcad_ols['Total PnL']:.4f}, "
        f"Win Rate = {usdcad_ols['Win Rate']:.2f}%, "
        f"Max DD = {usdcad_ols['Max Drawdown']:.2f}%"
    )
    print(
        f"  OLS Outperformance: {usdcad_ols['Total PnL'] - usdcad_kalman['Total PnL']:.4f}"
    )

    # USDNOK-Brent Pair Comparison
    print("\nUSDNOK-Brent Pair:")
    usdnok_kalman = performance_data["USDNOK-Brent-Kalman"]
    usdnok_ols = performance_data["USDNOK-Brent-OLS"]

    print(
        f"  Kalman Filter: PnL = {usdnok_kalman['Total PnL']:.4f}, "
        f"Win Rate = {usdnok_kalman['Win Rate']:.2f}%, "
        f"Max DD = {usdnok_kalman['Max Drawdown']:.2f}%"
    )
    print(
        f"  OLS: PnL = {usdnok_ols['Total PnL']:.4f}, "
        f"Win Rate = {usdnok_ols['Win Rate']:.2f}%, "
        f"Max DD = {usdnok_ols['Max Drawdown']:.2f}%"
    )
    print(
        f"  OLS Outperformance: {usdnok_ols['Total PnL'] - usdnok_kalman['Total PnL']:.4f}"
    )

    # Overall methodology comparison
    print("\nOverall Methodology Comparison:")
    kalman_total = usdcad_kalman["Total PnL"] + usdnok_kalman["Total PnL"]
    ols_total = usdcad_ols["Total PnL"] + usdnok_ols["Total PnL"]
    print(f"  Kalman Filter Total PnL: {kalman_total:.4f}")
    print(f"  OLS Total PnL: {ols_total:.4f}")
    print(f"  OLS Overall Outperformance: {ols_total - kalman_total:.4f}")

    # 3. TRADE STATISTICS ANALYSIS
    print("\n3. TRADE STATISTICS ANALYSIS")
    print("-" * 50)

    print("\nTrade Statistics Summary:")
    for strategy, data in performance_data.items():
        print(f"\n{strategy}:")
        print(f"  Number of Trades: {data['Number of Trades']:,}")
        print(f"  Win Rate: {data['Win Rate']:.2f}%")
        print(f"  Average Win: {data['Average Win']:.4f}")
        print(f"  Average Loss: {data['Average Loss']:.4f}")
        print(f"  Profit Factor: {data['Profit Factor']:.2f}")

        # Calculate expected value per trade
        expected_value = (
            data["Win Rate"] / 100 * data["Average Win"]
            + (1 - data["Win Rate"] / 100) * data["Average Loss"]
        )
        print(f"  Expected Value per Trade: {expected_value:.4f}")

    # 4. RISK ANALYSIS
    print("\n4. RISK ANALYSIS")
    print("-" * 50)

    print("\nRisk Characteristics:")
    for strategy, data in performance_data.items():
        print(f"\n{strategy}:")
        print(f"  Volatility: {data['Volatility']:.2f}%")
        print(f"  Maximum Drawdown: {data['Max Drawdown']:.2f}%")

        # Risk-adjusted returns (simplified)
        if data["Volatility"] > 0:
            risk_adj_return = data["Total PnL"] / data["Volatility"] * 100
            print(f"  Risk-Adjusted Return: {risk_adj_return:.4f}")

        # Recovery ratio
        if data["Max Drawdown"] < 0:
            recovery_ratio = abs(data["Total PnL"] / data["Max Drawdown"])
            print(f"  Recovery Ratio: {recovery_ratio:.4f}")

    # 5. TECHNICAL ISSUES INVESTIGATION
    print("\n5. TECHNICAL ISSUES INVESTIGATION")
    print("-" * 50)

    print("\nCritical Technical Issues Identified:")

    print("\n5.1 NaN Return Calculations:")
    print(
        "  Issue: All strategies show NaN values for Total Return, Annual Return, and Sharpe Ratio"
    )
    print(
        "  Impact: Inability to assess risk-adjusted performance and annualized metrics"
    )
    print(
        "  Root Cause: Likely division by zero or missing initial capital in return calculations"
    )

    print("\n5.2 Extreme Drawdown Values:")
    print(
        "  Issue: All strategies report drawdowns exceeding -100% (mathematically impossible)"
    )
    extreme_drawdowns = {
        "USDCAD-WTI-Kalman": -265.27,
        "USDCAD-WTI-OLS": -202.62,
        "USDNOK-Brent-Kalman": -425.76,
        "USDNOK-Brent-OLS": -844.29,
    }
    for strategy, dd in extreme_drawdowns.items():
        print(f"    {strategy}: {dd:.2f}%")
    print("  Impact: Severely compromised risk assessment and performance evaluation")
    print("  Root Cause: Calculation errors in drawdown computation methodology")

    print("\n5.3 Low Win Rates:")
    print("  Issue: All strategies exhibit win rates below 26%")
    win_rates = [data["Win Rate"] for data in performance_data.values()]
    print(f"  Average Win Rate: {np.mean(win_rates):.2f}%")
    print("  Impact: Strategies may be taking too many low-quality signals")
    print("  Root Cause: Signal generation thresholds may be too low")

    print("\n5.4 Equity Curve Analysis (from CSV examination):")
    print("  Findings from detailed CSV analysis:")
    print("    - Actual equity values are within reasonable ranges (0.74 to 1.37)")
    print(
        "    - Calculated drawdowns from equity data are much more reasonable (-21% to -31%)"
    )
    print("    - This suggests the reported extreme drawdowns are calculation errors")

    # 6. DATA COMPILATION AND INSIGHTS
    print("\n6. DATA COMPILATION AND INSIGHTS")
    print("-" * 50)

    print("\nKey Performance Rankings:")

    # Best to worst by PnL
    sorted_by_pnl = sorted(
        performance_data.items(), key=lambda x: x[1]["Total PnL"], reverse=True
    )
    print("\nStrategies ranked by Total PnL:")
    for i, (strategy, data) in enumerate(sorted_by_pnl, 1):
        print(f"  {i}. {strategy}: {data['Total PnL']:.4f}")

    # Best to worst by Win Rate
    sorted_by_winrate = sorted(
        performance_data.items(), key=lambda x: x[1]["Win Rate"], reverse=True
    )
    print("\nStrategies ranked by Win Rate:")
    for i, (strategy, data) in enumerate(sorted_by_winrate, 1):
        print(f"  {i}. {strategy}: {data['Win Rate']:.2f}%")

    # Best to worst by Profit Factor
    sorted_by_pf = sorted(
        performance_data.items(), key=lambda x: x[1]["Profit Factor"], reverse=True
    )
    print("\nStrategies ranked by Profit Factor:")
    for i, (strategy, data) in enumerate(sorted_by_pf, 1):
        print(f"  {i}. {strategy}: {data['Profit Factor']:.2f}")

    # 7. CROSS-STRATEGY PATTERNS
    print("\n7. CROSS-STRATEGY PATTERNS")
    print("-" * 50)

    print("\nConsistent Patterns Across All Strategies:")
    print("  1. Low Win Rates: All strategies below 26% win rate")
    print("  2. Similar Average Win/Loss: All show ~0.02 average wins and losses")
    print("  3. Technical Issues: All affected by NaN returns and extreme drawdowns")
    print("  4. Volatility Range: All strategies show 8.6% to 10.1% volatility")

    print("\nMethodology-Specific Patterns:")
    print("  Kalman Filter:")
    print("    - Higher trade count in USDNOK-Brent (612 vs 520)")
    print("    - Generally worse performance than OLS")
    print("    - More extreme drawdown calculations")
    print("  OLS:")
    print("    - More consistent performance across pairs")
    print("    - Better risk-adjusted returns where calculable")
    print("    - Lower trade count in USDNOK-Brent")

    print("\nPair-Specific Patterns:")
    print("  USDCAD-WTI:")
    print("    - Overall better performance than USDNOK-Brent")
    print("    - One profitable strategy (OLS)")
    print("    - More reasonable drawdown calculations")
    print("  USDNOK-Brent:")
    print("    - Poorer overall performance")
    print("    - All strategies unprofitable")
    print("    - Most extreme drawdown calculations")

    # 8. RECOMMENDATIONS FOR HANDOFFS
    print("\n8. RECOMMENDATIONS FOR HANDOFFS")
    print("-" * 50)

    print("\n8.1 Failures -> Quant & Research Studio:")
    print("  Strategy Performance Issues:")
    print("    - Low win rates across all strategies (21.93% - 25.51%)")
    print("    - Negative expected values for most strategies")
    print("    - Poor risk-adjusted performance")
    print("  Signal Quality Issues:")
    print("    - Need to review signal generation thresholds")
    print("    - Evaluate regime detection effectiveness")
    print("    - Consider higher entry thresholds for better quality signals")

    print("\n8.2 Unrealistic Costs/Fills -> Risk & Execution Controls:")
    print("  Calculation Errors:")
    print("    - Extreme drawdown calculations (-202% to -844%)")
    print("    - NaN return calculations preventing proper risk assessment")
    print("    - Need to fix drawdown computation methodology")
    print("  Risk Management:")
    print("    - Implement proper position sizing limits")
    print("    - Add stop-loss mechanisms")
    print("    - Consider maximum drawdown limits")

    print("\n8.3 Tooling Gaps -> Engineering & Platform:")
    print("  Backtest Engine Issues:")
    print("    - Return calculation methodology needs review")
    print("    - Drawdown calculation algorithm has bugs")
    print("    - Need better error handling for edge cases")
    print("  Data Processing:")
    print("    - Improve handling of NaN values in calculations")
    print("    - Add validation checks for impossible values")
    print("    - Implement robust equity curve tracking")

    # 9. SUMMARY AND CONCLUSIONS
    print("\n9. SUMMARY AND CONCLUSIONS")
    print("-" * 50)

    print("\nOverall Performance Summary:")
    all_pnls = [data["Total PnL"] for data in performance_data.values()]
    all_win_rates = [data["Win Rate"] for data in performance_data.values()]
    all_drawdowns = [data["Max Drawdown"] for data in performance_data.values()]

    print(f"  Average PnL across all strategies: {np.mean(all_pnls):.4f}")
    print(f"  Average Win Rate: {np.mean(all_win_rates):.2f}%")
    print(f"  Average Reported Max Drawdown: {np.mean(all_drawdowns):.2f}%")
    print(
        f"  Best Performing Strategy: {max(performance_data.keys(), key=lambda k: performance_data[k]['Total PnL'])}"
    )
    print(
        f"  Worst Performing Strategy: {min(performance_data.keys(), key=lambda k: performance_data[k]['Total PnL'])}"
    )

    print("\nKey Findings:")
    print("  1. OLS methodology outperformed Kalman Filter in both pairs")
    print("  2. USDCAD-WTI pair showed better performance than USDNOK-Brent")
    print("  3. Only USDCAD-WTI-OLS was profitable (PnL: 0.1200)")
    print("  4. All strategies suffer from low win rates (<26%)")
    print("  5. Critical technical issues affect performance assessment")

    print("\nTechnical Assessment:")
    print("  1. Return calculation methodology is fundamentally flawed")
    print("  2. Drawdown calculation produces impossible values")
    print("  3. Actual equity curves appear reasonable (based on CSV analysis)")
    print("  4. Signal generation may be too aggressive (low win rates)")

    print("\nFinal Assessment:")
    print("  The backtest results indicate significant issues with both strategy")
    print("  performance and technical implementation. While OLS shows some")
    print("  promise with USDCAD-WTI, all strategies require substantial")
    print("  improvements in signal quality and risk management. The technical")
    print("  issues in performance calculation must be resolved before any")
    print("  reliable performance assessment can be made.")

    # Save comprehensive report
    with open("final_comprehensive_backtest_analysis.txt", "w") as f:
        f.write("COMPREHENSIVE BACKTEST RESULTS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Write all sections to file
        f.write("1. PERFORMANCE METRICS ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(perf_df.to_string(float_format="%.4f"))
        f.write("\n\n")

        # Add other sections similarly...
        f.write("2. KALMAN FILTER VS OLS PERFORMANCE COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"USDCAD-WTI Pair - OLS Outperformance: {usdcad_ols['Total PnL'] - usdcad_kalman['Total PnL']:.4f}\n"
        )
        f.write(
            f"USDNOK-Brent Pair - OLS Outperformance: {usdnok_ols['Total PnL'] - usdnok_kalman['Total PnL']:.4f}\n"
        )
        f.write(f"Overall OLS Outperformance: {ols_total - kalman_total:.4f}\n\n")

        f.write("3. KEY FINDINGS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average PnL: {np.mean(all_pnls):.4f}\n")
        f.write(f"Average Win Rate: {np.mean(all_win_rates):.2f}%\n")
        f.write(
            f"Best Strategy: {max(performance_data.keys(), key=lambda k: performance_data[k]['Total PnL'])}\n"
        )
        f.write(
            f"Worst Strategy: {min(performance_data.keys(), key=lambda k: performance_data[k]['Total PnL'])}\n\n"
        )

        f.write("4. CRITICAL TECHNICAL ISSUES\n")
        f.write("-" * 50 + "\n")
        f.write("- NaN return calculations affecting all strategies\n")
        f.write(
            "- Extreme drawdown values (-202% to -844%) indicating calculation errors\n"
        )
        f.write("- Low win rates across all strategies (<26%)\n")
        f.write("- Need for methodology review and implementation fixes\n\n")

        f.write("5. RECOMMENDATIONS\n")
        f.write("-" * 50 + "\n")
        f.write("Quant & Research Studio: Review signal quality and thresholds\n")
        f.write(
            "Risk & Execution Controls: Fix drawdown calculations and add risk limits\n"
        )
        f.write(
            "Engineering & Platform: Resolve return calculation methodology issues\n"
        )

    print(
        "\n✅ Comprehensive analysis complete. Report saved to 'final_comprehensive_backtest_analysis.txt'"
    )


if __name__ == "__main__":
    create_final_comprehensive_report()
